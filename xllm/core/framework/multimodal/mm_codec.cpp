/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "mm_codec.h"

#include <algorithm>
#include <cmath>
#include <limits>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

namespace xllm {

namespace {

struct MemCtx {
  const uint8_t* mem_ptr;
  int64_t size;
  int64_t offset;
};

struct Reader {
  // AVIO read callback
  static int32_t read(void* opaque, uint8_t* buf, int32_t buf_size) {
    auto* mc = static_cast<MemCtx*>(opaque);
    if (mc->offset < 0) {
      return AVERROR(EINVAL);
    }
    int64_t remain = mc->size - mc->offset;
    int64_t n = std::min(remain, static_cast<int64_t>(buf_size));
    if (n <= 0) {
      return AVERROR_EOF;
    }
    std::memcpy(buf, mc->mem_ptr + mc->offset, static_cast<size_t>(n));
    mc->offset += n;
    return static_cast<int32_t>(n);
  }

  // AVIO seek callback
  static int64_t seek(void* opaque, int64_t offset, int32_t whence) {
    auto* mc = static_cast<MemCtx*>(opaque);

    if (whence == AVSEEK_SIZE) {
      return mc->size;
    }

    int64_t pos = 0;
    switch (whence) {
      case SEEK_SET:
        pos = offset;
        break;
      case SEEK_CUR:
        pos = mc->offset + offset;
        break;
      case SEEK_END:
        pos = mc->size + offset;
        break;
      default:
        return AVERROR(EINVAL);
    }

    if (pos < 0 || pos > mc->size) {
      return AVERROR(EINVAL);
    }

    mc->offset = pos;
    return pos;
  }
};

// Downstream multimodal preprocess expects a 3-channel RGB image. For BGRA
// input, blend alpha onto a white background instead of dropping alpha
// directly, so transparent regions keep visually stable colors.
void blend_bgra_to_rgb_with_white_background(const cv::Mat& image,
                                             cv::Mat& rgb_image) {
  rgb_image = cv::Mat(image.rows, image.cols, CV_8UC3);

  for (int32_t row = 0; row < image.rows; ++row) {
    const cv::Vec4b* src_row = image.ptr<cv::Vec4b>(row);
    cv::Vec3b* dst_row = rgb_image.ptr<cv::Vec3b>(row);

    for (int32_t col = 0; col < image.cols; ++col) {
      const float alpha = src_row[col][3] / 255.0f;
      const float blue = src_row[col][0];
      const float green = src_row[col][1];
      const float red = src_row[col][2];

      dst_row[col][0] = static_cast<uint8_t>(
          std::round(red * alpha + 255.0f * (1.0f - alpha)));
      dst_row[col][1] = static_cast<uint8_t>(
          std::round(green * alpha + 255.0f * (1.0f - alpha)));
      dst_row[col][2] = static_cast<uint8_t>(
          std::round(blue * alpha + 255.0f * (1.0f - alpha)));
    }
  }
}

// OpenCV decodes different image formats into different channel layouts:
// 1 channel -> GRAY, 3 channels -> BGR, 4 channels -> BGRA. Normalize them
// here so the decoder path always returns a 3-channel RGB image.
void convert_decoded_image_to_rgb(const cv::Mat& image, cv::Mat& rgb_image) {
  const int32_t channels = image.channels();

  if (channels == 4) {
    blend_bgra_to_rgb_with_white_background(image, rgb_image);
  } else if (channels == 3) {
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
  } else if (channels == 1) {
    cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2RGB);
  } else {
    LOG(FATAL) << "unsupported channel count: " << channels;
  }
}

}  // namespace

class MemoryMediaReader {
 public:
  MemoryMediaReader(const uint8_t* data, size_t size) {
    mc_.mem_ptr = data;
    if (size > static_cast<size_t>(INT64_MAX)) {
      LOG(FATAL) << "MemCtx size too large";
    }
    mc_.size = static_cast<int64_t>(size);
    mc_.offset = 0;
  }

  ~MemoryMediaReader() {
    if (frm_) {
      av_frame_free(&frm_);
    }
    if (pkt_) {
      av_packet_free(&pkt_);
    }
    if (codec_ctx_) {
      avcodec_free_context(&codec_ctx_);
    }
    if (fmt_ctx_) {
      avformat_close_input(&fmt_ctx_);
    }
    if (avio_ctx_) {
      av_freep(&avio_ctx_->buffer);
      avio_context_free(&avio_ctx_);
    } else if (avio_buf_) {
      av_freep(reinterpret_cast<void**>(&avio_buf_));
    }
  }

  bool init(AVMediaType type) {
    fmt_ctx_ = avformat_alloc_context();
    if (!fmt_ctx_) {
      return false;
    }
    constexpr int32_t avio_buf_sz = 1 << 16;
    avio_buf_ =
        static_cast<uint8_t*>(av_malloc(static_cast<size_t>(avio_buf_sz)));
    if (!avio_buf_) {
      return false;
    }

    avio_ctx_ = avio_alloc_context(
        avio_buf_, avio_buf_sz, 0, &mc_, &Reader::read, nullptr, &Reader::seek);
    if (!avio_ctx_) {
      return false;
    }
    avio_buf_ = nullptr;

    avio_ctx_->seekable = AVIO_SEEKABLE_NORMAL;
    fmt_ctx_->pb = avio_ctx_;
    fmt_ctx_->flags |= AVFMT_FLAG_CUSTOM_IO;

    if (avformat_open_input(&fmt_ctx_, nullptr, nullptr, nullptr) < 0) {
      return false;
    }

    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
      return false;
    }

    stream_index_ = av_find_best_stream(fmt_ctx_, type, -1, -1, nullptr, 0);
    if (stream_index_ < 0) {
      return false;
    }

    AVStream* st = fmt_ctx_->streams[stream_index_];
    const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!codec) {
      return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
      return false;
    }

    if (avcodec_parameters_to_context(codec_ctx_, st->codecpar) < 0 ||
        avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
      return false;
    }

    pkt_ = av_packet_alloc();
    frm_ = av_frame_alloc();
    if (!pkt_ || !frm_) {
      return false;
    }

    return true;
  }

  bool decode() {
    CHECK(fmt_ctx_ && codec_ctx_ && pkt_ && frm_) << "ffmpeg init failed";
    CHECK_GE(stream_index_, 0) << "stream index not found";

    // read packets, send to decoder, pull frames
    while (av_read_frame(fmt_ctx_, pkt_) >= 0) {
      if (pkt_->stream_index == stream_index_) {
        if (avcodec_send_packet(codec_ctx_, pkt_) == 0) {
          while (avcodec_receive_frame(codec_ctx_, frm_) == 0) {
            // handle_frame: video->push RGB frame, audio->append PCM
            if (!handle_frame(frm_)) {
              av_packet_unref(pkt_);
              return false;
            }
          }
        }
      }
      av_packet_unref(pkt_);
    }

    // flush decoder at end of stream
    avcodec_send_packet(codec_ctx_, nullptr);
    while (avcodec_receive_frame(codec_ctx_, frm_) == 0) {
      if (!handle_frame(frm_)) {
        return false;
      }
    }

    return true;
  }

  // video->RGB tensor, audio->PCM samples
  virtual bool handle_frame(AVFrame* f) = 0;

 protected:
  AVFormatContext* fmt_ctx_ = nullptr;
  uint8_t* avio_buf_ = nullptr;
  AVIOContext* avio_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVPacket* pkt_ = nullptr;
  AVFrame* frm_ = nullptr;
  MemCtx mc_{nullptr, 0, 0};
  int32_t stream_index_ = -1;
};

class MemoryVideoReader : public MemoryMediaReader {
 public:
  MemoryVideoReader(const uint8_t* data, size_t size)
      : MemoryMediaReader(data, size) {}

  ~MemoryVideoReader() {
    if (sws_ctx_) {
      sws_freeContext(sws_ctx_);
    }
    if (rgb_frame_) {
      av_frame_free(&rgb_frame_);
    }
  }

  bool init(VideoMetadata& metadata) {
    if (!MemoryMediaReader::init(AVMEDIA_TYPE_VIDEO)) {
      return false;
    }

    // init VideoMetadata
    AVStream* st = fmt_ctx_->streams[stream_index_];
    AVRational r =
        st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;
    metadata.fps = (r.num && r.den) ? av_q2d(r) : 0.0;
    metadata.total_num_frames = 0;
    metadata.duration = 0.0;
    return true;
  }

  bool read(torch::Tensor& tensor, VideoMetadata& metadata) {
    CHECK(frames_.empty()) << "frames is not cleared before read";

    if (!decode()) {
      return false;
    }
    if (frames_.empty()) {
      return false;
    }

    tensor = torch::stack(frames_);  // [T,C,H,W]
    metadata.total_num_frames = static_cast<int32_t>(frames_.size());
    metadata.duration =
        (metadata.fps > 0.0)
            ? static_cast<double>(metadata.total_num_frames) / metadata.fps
            : 0.0;
    return true;
  }

  bool handle_frame(AVFrame* f) override {
    // init colorspace converter once based on first frame
    if (!sws_ctx_) {
      sws_ctx_ = sws_getContext(f->width,
                                f->height,
                                static_cast<AVPixelFormat>(f->format),
                                f->width,
                                f->height,
                                AV_PIX_FMT_RGB24,
                                SWS_BILINEAR,
                                nullptr,
                                nullptr,
                                nullptr);
      if (!sws_ctx_) {
        return false;
      }
    }

    // use an FFmpeg-allocated frame so sws_scale writes into a buffer with the
    // correct padded linesize
    if (!rgb_frame_) {
      rgb_frame_ = av_frame_alloc();
      if (!rgb_frame_) {
        return false;
      }
    }

    // (re)allocate the RGB buffer when input changes
    if (rgb_frame_->width != f->width || rgb_frame_->height != f->height ||
        rgb_frame_->format != AV_PIX_FMT_RGB24 || !rgb_frame_->data[0]) {
      av_frame_unref(rgb_frame_);
      rgb_frame_->format = AV_PIX_FMT_RGB24;
      rgb_frame_->width = f->width;
      rgb_frame_->height = f->height;
      if (av_frame_get_buffer(rgb_frame_, 0) < 0) {
        return false;
      }
    }
    if (av_frame_make_writable(rgb_frame_) < 0) {
      return false;
    }

    // convert the current decoded frame into RGB24
    if (sws_scale(sws_ctx_,
                  f->data,
                  f->linesize,
                  0,
                  f->height,
                  rgb_frame_->data,
                  rgb_frame_->linesize) != f->height) {
      return false;
    }

    // build CHW uint8 tensor
    const int64_t H = f->height;
    const int64_t W = f->width;
    const int64_t src_ls = rgb_frame_->linesize[0];

    auto rgb = torch::from_blob(rgb_frame_->data[0],
                                {3, H, W},  // [C,H,W]
                                {1, src_ls, 3},
                                torch::TensorOptions().dtype(torch::kUInt8))
                   .contiguous();

    frames_.emplace_back(rgb.clone());
    return true;
  }

 private:
  SwsContext* sws_ctx_ = nullptr;
  AVFrame* rgb_frame_ = nullptr;
  std::vector<torch::Tensor> frames_;
};

class MemoryAudioReader : public MemoryMediaReader {
 public:
  MemoryAudioReader(const uint8_t* data,
                    size_t size,
                    int64_t target_sr = 16000,
                    int32_t target_channels = 1)
      : MemoryMediaReader(data, size) {
    target_sr_ = target_sr;
    target_ch_ = target_channels;
  }

  ~MemoryAudioReader() {
    if (swr_ctx_) {
      swr_free(&swr_ctx_);
    }
  }

  bool init(AudioMetadata& metadata) {
    if (target_sr_ <= 0 || (target_ch_ != 1 && target_ch_ != 2)) {
      return false;
    }
    if (!MemoryMediaReader::init(AVMEDIA_TYPE_AUDIO)) {
      return false;
    }

    AVStream* st = fmt_ctx_->streams[stream_index_];
    codec_ctx_->pkt_timebase = st->time_base;

    // setup resampler
    swr_ctx_ = swr_alloc();
    if (!swr_ctx_) {
      return false;
    }

    AVChannelLayout in_layout;
    if (av_channel_layout_copy(&in_layout, &codec_ctx_->ch_layout) < 0) {
      return false;
    }

    AVChannelLayout out_layout;
    av_channel_layout_default(&out_layout, target_ch_);

    if (swr_alloc_set_opts2(&swr_ctx_,
                            &out_layout,
                            AV_SAMPLE_FMT_FLT,
                            target_sr_,
                            &in_layout,
                            codec_ctx_->sample_fmt,
                            codec_ctx_->sample_rate,
                            0,
                            nullptr) < 0) {
      av_channel_layout_uninit(&out_layout);
      av_channel_layout_uninit(&in_layout);
      return false;
    }

    av_channel_layout_uninit(&out_layout);
    av_channel_layout_uninit(&in_layout);

    // if downmixing stereo -> mono, use customized remix matrix (L+R)/2
    int32_t in_ch = codec_ctx_->ch_layout.nb_channels;
    if (target_ch_ == 1 && in_ch == 2) {
      constexpr double matrix[2] = {0.5, 0.5};
      if (swr_set_matrix(swr_ctx_, matrix, in_ch) < 0) {
        return false;
      }
    }

    if (swr_init(swr_ctx_) < 0) {
      return false;
    }

    // init AudioMetadata
    metadata.sample_rate = target_sr_;
    metadata.num_channels = target_ch_;
    metadata.duration = 0.0;
    return true;
  }

  bool read(torch::Tensor& tensor, AudioMetadata& metadata) {
    CHECK(swr_ctx_) << "SwrContext is null";
    CHECK(pcm_.empty()) << "PCM buffer is not cleared before read";

    if (!decode()) {
      return false;
    }

    // flush resampler buffered samples after decode
    while (true) {
      if (resample_to_pcm(nullptr, 0) <= 0) {
        break;
      }
    }

    if (pcm_.empty()) {
      return false;
    }

    // build output tensor and compute metadata
    if (target_ch_ == 1) {
      tensor = torch::from_blob(pcm_.data(),
                                {static_cast<int32_t>(pcm_.size())},
                                torch::TensorOptions().dtype(torch::kFloat32))
                   .clone();
      metadata.duration = static_cast<double>(pcm_.size()) / target_sr_;
    } else {
      int32_t T =
          static_cast<int32_t>(pcm_.size() / static_cast<size_t>(target_ch_));
      tensor = torch::from_blob(pcm_.data(),
                                {T, target_ch_},
                                torch::TensorOptions().dtype(torch::kFloat32))
                   .permute({1, 0})
                   .clone();
      metadata.duration = static_cast<double>(T) / target_sr_;
    }
    metadata.sample_rate = target_sr_;
    metadata.num_channels = target_ch_;
    return true;
  }

  bool handle_frame(AVFrame* f) override {
    return resample_to_pcm((const uint8_t**)f->extended_data, f->nb_samples) >=
           0;
  }

  int32_t resample_to_pcm(const uint8_t** in_data, int32_t nb_samples) {
    int32_t out_nb = swr_get_out_samples(swr_ctx_, nb_samples);
    if (out_nb < 0) {
      return out_nb;
    }
    if (out_nb == 0) {
      return 0;
    }

    std::vector<float> out_buf(static_cast<size_t>(out_nb) *
                               static_cast<size_t>(target_ch_));
    uint8_t* out_data[1] = {reinterpret_cast<uint8_t*>(out_buf.data())};

    // convert input frame samples to target format
    int32_t converted =
        swr_convert(swr_ctx_, out_data, out_nb, in_data, nb_samples);
    if (converted < 0) {
      return converted;
    }
    if (converted == 0) {
      return 0;
    }

    // append converted samples to pcm buffer
    const int64_t n = static_cast<int64_t>(converted * target_ch_);
    pcm_.reserve(pcm_.size() + static_cast<size_t>(n));
    pcm_.insert(pcm_.end(), out_buf.data(), out_buf.data() + n);
    return converted;
  }

 private:
  SwrContext* swr_ctx_ = nullptr;
  int32_t target_sr_ = 16000;
  int32_t target_ch_ = 1;
  std::vector<float> pcm_;
};

bool OpenCVImageDecoder::decode(const std::string& raw_data, torch::Tensor& t) {
  cv::Mat buffer(1, raw_data.size(), CV_8UC1, (void*)raw_data.data());
  if (raw_data.empty()) {
    LOG(ERROR) << "opencv image decode got empty data";
    return false;
  }
  cv::Mat image = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
  if (image.empty()) {
    LOG(INFO) << "opencv image decode failed";
    return false;
  }

  cv::Mat rgb_image;
  convert_decoded_image_to_rgb(image, rgb_image);

  torch::Tensor tensor =
      torch::from_blob(rgb_image.data,
                       {rgb_image.rows, rgb_image.cols, 3},
                       torch::TensorOptions().dtype(torch::kUInt8));

  t = tensor.permute({2, 0, 1}).clone();  // [C, H, W]
  return true;
}

bool OpenCVImageEncoder::encode(const torch::Tensor& t, std::string& raw_data) {
  if (!valid(t)) {
    return false;
  }

  auto img = t.permute({1, 2, 0}).contiguous();
  cv::Mat mat(img.size(0), img.size(1), CV_32FC3, img.data_ptr<float>());

  cv::Mat mat_8u;
  mat.convertTo(mat_8u, CV_8UC3, 255.0);

  // rgb -> bgr
  cv::cvtColor(mat_8u, mat_8u, cv::COLOR_RGB2BGR);

  std::vector<uchar> data;
  if (!cv::imencode(".png", mat_8u, data)) {
    LOG(ERROR) << "image encode failed";
    return false;
  }

  raw_data.assign(data.begin(), data.end());
  return true;
}

bool OpenCVImageEncoder::valid(const torch::Tensor& t) {
  if (t.dim() != 3 || t.size(0) != 3) {
    LOG(ERROR) << "input tensor must be 3HW  tensor";
    return false;
  }

  if (t.scalar_type() != torch::kFloat32 || !t.device().is_cpu()) {
    LOG(ERROR) << "tensor must be cpu float32";
    return false;
  }

  return true;
}

bool FFmpegVideoDecoder::decode(const std::string& raw_data,
                                torch::Tensor& t,
                                VideoMetadata& metadata) {
  MemoryVideoReader reader(reinterpret_cast<const uint8_t*>(raw_data.data()),
                           raw_data.size());

  if (!reader.init(metadata) || !reader.read(t, metadata)) {
    LOG(INFO) << "video decode failed";
    return false;
  }
  return true;
}

bool FFmpegAudioDecoder::decode(const std::string& raw_data,
                                torch::Tensor& t,
                                AudioMetadata& metadata,
                                int64_t target_sr,
                                int32_t target_channels) {
  MemoryAudioReader reader(reinterpret_cast<const uint8_t*>(raw_data.data()),
                           raw_data.size(),
                           target_sr,
                           target_channels);

  if (!reader.init(metadata) || !reader.read(t, metadata)) {
    LOG(INFO) << "audio decode failed";
    return false;
  }
  return true;
}

// ---- MemoryMediaWriter (in-memory encoding base class) ----

namespace {

struct MemWriteCtx {
  std::vector<uint8_t>* buf;
  int64_t pos;
};

struct Writer {
  static int32_t write(void* opaque, uint8_t* buf, int32_t buf_size) {
    auto* mc = static_cast<MemWriteCtx*>(opaque);
    if (buf_size < 0 ||
        mc->pos > std::numeric_limits<int64_t>::max() - buf_size) {
      return AVERROR(EOVERFLOW);
    }
    int64_t end_pos = mc->pos + buf_size;
    if (end_pos > static_cast<int64_t>(mc->buf->size())) {
      mc->buf->resize(static_cast<size_t>(end_pos), 0);
    }
    std::memcpy(mc->buf->data() + mc->pos, buf, static_cast<size_t>(buf_size));
    mc->pos = end_pos;
    return buf_size;
  }

  static int64_t seek(void* opaque, int64_t offset, int32_t whence) {
    auto* mc = static_cast<MemWriteCtx*>(opaque);
    if (whence == AVSEEK_SIZE) {
      return static_cast<int64_t>(mc->buf->size());
    }
    whence &= ~AVSEEK_FORCE;
    int64_t pos = 0;
    switch (whence) {
      case SEEK_SET:
        pos = offset;
        break;
      case SEEK_CUR:
        if ((offset > 0 &&
             mc->pos > std::numeric_limits<int64_t>::max() - offset) ||
            (offset < 0 &&
             mc->pos < std::numeric_limits<int64_t>::min() - offset)) {
          return AVERROR(EOVERFLOW);
        }
        pos = mc->pos + offset;
        break;
      case SEEK_END:
        if (mc->buf->size() > static_cast<size_t>(INT64_MAX)) {
          return AVERROR(EOVERFLOW);
        }
        pos = static_cast<int64_t>(mc->buf->size());
        if ((offset > 0 &&
             pos > std::numeric_limits<int64_t>::max() - offset) ||
            (offset < 0 &&
             pos < std::numeric_limits<int64_t>::min() - offset)) {
          return AVERROR(EOVERFLOW);
        }
        pos += offset;
        break;
      default:
        return AVERROR(EINVAL);
    }
    if (pos < 0) {
      return AVERROR(EINVAL);
    }
    mc->pos = pos;
    return pos;
  }
};

}  // namespace

class MemoryMediaWriter {
 public:
  MemoryMediaWriter() = default;

  virtual ~MemoryMediaWriter() {
    if (pkt_) {
      av_packet_free(&pkt_);
    }
    if (codec_ctx_) {
      avcodec_free_context(&codec_ctx_);
    }
    if (fmt_ctx_) {
      if (!finished_) {
        av_write_trailer(fmt_ctx_);
      }
      avformat_free_context(fmt_ctx_);
    }
    if (avio_ctx_) {
      av_freep(&avio_ctx_->buffer);
      avio_context_free(&avio_ctx_);
    }
  }

 protected:
  bool init(const char* format,
            AVCodecID codec_id,
            int32_t width,
            int32_t height,
            double fps,
            AVPixelFormat pix_fmt,
            AVDictionary** opts = nullptr) {
    const AVCodec* codec = avcodec_find_encoder(codec_id);
    if (!codec) {
      LOG(ERROR) << "MemoryMediaWriter: encoder not found, codec_id="
                 << avcodec_get_name(codec_id);
      return false;
    }

    constexpr int32_t avio_buf_sz = 1 << 16;
    uint8_t* avio_buf =
        static_cast<uint8_t*>(av_malloc(static_cast<size_t>(avio_buf_sz)));
    if (!avio_buf) {
      return false;
    }

    avio_ctx_ = avio_alloc_context(avio_buf,
                                   avio_buf_sz,
                                   1,
                                   &write_ctx_,
                                   nullptr,
                                   &Writer::write,
                                   &Writer::seek);
    if (!avio_ctx_) {
      av_freep(reinterpret_cast<void**>(&avio_buf));
      return false;
    }
    avio_ctx_->seekable = AVIO_SEEKABLE_NORMAL;

    const AVOutputFormat* fmt = av_guess_format(format, nullptr, nullptr);
    if (!fmt) {
      LOG(ERROR) << "MemoryMediaWriter: no muxer for " << format;
      return false;
    }

    if (avformat_alloc_output_context2(&fmt_ctx_, fmt, nullptr, nullptr) < 0 ||
        !fmt_ctx_) {
      return false;
    }
    fmt_ctx_->pb = avio_ctx_;
    fmt_ctx_->flags |= AVFMT_FLAG_CUSTOM_IO;

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
      return false;
    }

    codec_ctx_->width = width;
    codec_ctx_->height = height;
    codec_ctx_->time_base = {1, static_cast<int32_t>(std::llround(fps))};
    codec_ctx_->framerate = {static_cast<int32_t>(std::llround(fps)), 1};
    codec_ctx_->pix_fmt = pix_fmt;

    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
      codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    if (avcodec_open2(codec_ctx_, codec, opts) < 0) {
      LOG(ERROR) << "MemoryMediaWriter: avcodec_open2 failed for "
                 << codec->name;
      return false;
    }

    stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    if (!stream_) {
      return false;
    }
    stream_->time_base = codec_ctx_->time_base;
    if (avcodec_parameters_from_context(stream_->codecpar, codec_ctx_) < 0) {
      return false;
    }

    if (avformat_write_header(fmt_ctx_, nullptr) < 0) {
      LOG(ERROR) << "MemoryMediaWriter: avformat_write_header failed";
      return false;
    }

    pkt_ = av_packet_alloc();
    if (!pkt_) {
      return false;
    }

    LOG(INFO) << "MemoryMediaWriter: initialized " << codec->name << " ["
              << format << "] " << width << "x" << height << " @ " << fps
              << " fps";
    return true;
  }

  bool send_frame(AVFrame* frame) {
    if (avcodec_send_frame(codec_ctx_, frame) < 0) {
      return false;
    }
    return drain_packets();
  }

  bool finish() {
    avcodec_send_frame(codec_ctx_, nullptr);
    if (!drain_packets()) {
      return false;
    }
    av_write_trailer(fmt_ctx_);
    finished_ = true;
    return true;
  }

  std::vector<uint8_t> take_output() { return std::move(out_buf_); }

  AVCodecContext* codec_ctx() { return codec_ctx_; }
  AVStream* stream() { return stream_; }

  bool drain_packets() {
    while (avcodec_receive_packet(codec_ctx_, pkt_) == 0) {
      av_packet_rescale_ts(pkt_, codec_ctx_->time_base, stream_->time_base);
      pkt_->stream_index = stream_->index;
      if (av_interleaved_write_frame(fmt_ctx_, pkt_) < 0) {
        av_packet_unref(pkt_);
        return false;
      }
      av_packet_unref(pkt_);
    }
    return true;
  }

  AVFormatContext* fmt_ctx_ = nullptr;
  AVIOContext* avio_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVPacket* pkt_ = nullptr;
  AVStream* stream_ = nullptr;
  MemWriteCtx write_ctx_{&out_buf_, 0};
  std::vector<uint8_t> out_buf_;
  bool finished_ = false;
};

class MemoryVideoWriter final : public MemoryMediaWriter {
 public:
  MemoryVideoWriter() = default;

  ~MemoryVideoWriter() {
    if (sws_ctx_) {
      sws_freeContext(sws_ctx_);
    }
    if (yuv_frame_) {
      av_frame_free(&yuv_frame_);
    }
  }

  bool write(const torch::Tensor& video,
             double fps,
             const std::string& format,
             std::string& raw_data) {
    if (video.dim() != 4 || video.size(1) != 3) {
      LOG(ERROR) << "MemoryVideoWriter: expects [T,C,H,W] with C=3, got "
                 << video.sizes();
      return false;
    }
    if (video.scalar_type() != torch::kFloat32 || !video.device().is_cpu()) {
      LOG(ERROR) << "MemoryVideoWriter: expects cpu float32 tensor";
      return false;
    }

    const int64_t T = video.size(0);
    const int64_t H = video.size(2);
    const int64_t W = video.size(3);
    if (T == 0 || H == 0 || W == 0) {
      LOG(ERROR) << "MemoryVideoWriter: empty dimensions T=" << T << " H=" << H
                 << " W=" << W;
      return false;
    }

    AVCodecID codec_id;
    AVPixelFormat pix_fmt;
    AVDictionary* opts = nullptr;

    if (format == "avi") {
      codec_id = AV_CODEC_ID_MJPEG;
      pix_fmt = AV_PIX_FMT_YUVJ420P;
    } else {
      const AVCodec* x264_codec = avcodec_find_encoder_by_name("libx264");

      if (x264_codec) {
        codec_id = x264_codec->id;
        pix_fmt = AV_PIX_FMT_YUV420P;
        av_dict_set(&opts, "crf", "18", 0);
        av_dict_set(&opts, "preset", "medium", 0);
        av_dict_set(&opts, "profile", "high", 0);
        av_dict_set(&opts, "level", "4.1", 0);
        LOG(INFO) << "Using libx264 H.264 encoder with CRF=18";
      } else {
        codec_id = AV_CODEC_ID_MPEG4;
        pix_fmt = AV_PIX_FMT_YUV420P;
        av_dict_set(&opts, "mbd", "2", 0);
        LOG(WARNING) << "libx264 not available, using MPEG4 fallback";
      }
    }

    if (!init(format.c_str(),
              codec_id,
              static_cast<int32_t>(W),
              static_cast<int32_t>(H),
              fps,
              pix_fmt,
              &opts)) {
      if (opts) av_dict_free(&opts);
      return false;
    }
    if (opts) av_dict_free(&opts);

    sws_ctx_ = sws_getContext(static_cast<int32_t>(W),
                              static_cast<int32_t>(H),
                              AV_PIX_FMT_RGB24,
                              static_cast<int32_t>(W),
                              static_cast<int32_t>(H),
                              pix_fmt,
                              SWS_BILINEAR,
                              nullptr,
                              nullptr,
                              nullptr);
    if (!sws_ctx_) {
      LOG(ERROR) << "MemoryVideoWriter: sws_getContext failed";
      return false;
    }

    yuv_frame_ = av_frame_alloc();
    if (!yuv_frame_) {
      return false;
    }
    yuv_frame_->format = pix_fmt;
    yuv_frame_->width = static_cast<int32_t>(W);
    yuv_frame_->height = static_cast<int32_t>(H);
    if (av_frame_get_buffer(yuv_frame_, 0) < 0) {
      return false;
    }

    auto video_acc = video.accessor<float, 4>();
    const int64_t stride = W * 3;
    std::vector<uint8_t> rgb_buf(static_cast<size_t>(H * stride));
    int64_t pts = 0;

    for (int64_t t = 0; t < T; ++t) {
      for (int64_t y = 0; y < H; ++y) {
        for (int64_t x = 0; x < W; ++x) {
          rgb_buf[static_cast<size_t>(y * stride + x * 3 + 0)] =
              static_cast<uint8_t>(
                  std::clamp(video_acc[t][0][y][x] * 255.0f, 0.0f, 255.0f));
          rgb_buf[static_cast<size_t>(y * stride + x * 3 + 1)] =
              static_cast<uint8_t>(
                  std::clamp(video_acc[t][1][y][x] * 255.0f, 0.0f, 255.0f));
          rgb_buf[static_cast<size_t>(y * stride + x * 3 + 2)] =
              static_cast<uint8_t>(
                  std::clamp(video_acc[t][2][y][x] * 255.0f, 0.0f, 255.0f));
        }
      }

      const uint8_t* src_data[1] = {rgb_buf.data()};
      int32_t src_linesize[1] = {static_cast<int32_t>(stride)};

      if (av_frame_make_writable(yuv_frame_) < 0) {
        return false;
      }
      sws_scale(sws_ctx_,
                src_data,
                src_linesize,
                0,
                static_cast<int32_t>(H),
                yuv_frame_->data,
                yuv_frame_->linesize);
      yuv_frame_->pts = pts++;

      if (!send_frame(yuv_frame_)) {
        return false;
      }
    }

    if (!finish()) {
      return false;
    }

    auto out = take_output();
    raw_data.assign(out.begin(), out.end());

    LOG(INFO) << "MemoryVideoWriter: encoded " << T << " frames (" << W << "x"
              << H << ") at " << fps << " fps [" << format << "], output "
              << out.size() << " bytes";
    return true;
  }

 private:
  SwsContext* sws_ctx_ = nullptr;
  AVFrame* yuv_frame_ = nullptr;
};

class MemoryVideoAudioWriter final {
 public:
  MemoryVideoAudioWriter() = default;

  ~MemoryVideoAudioWriter() {
    if (sws_ctx_) {
      sws_freeContext(sws_ctx_);
    }
    if (video_frame_) {
      av_frame_free(&video_frame_);
    }
    if (audio_frame_) {
      av_frame_free(&audio_frame_);
    }
    if (pkt_) {
      av_packet_free(&pkt_);
    }
    if (video_ctx_) {
      avcodec_free_context(&video_ctx_);
    }
    if (audio_ctx_) {
      avcodec_free_context(&audio_ctx_);
    }
    if (fmt_ctx_) {
      if (header_written_ && !trailer_attempted_) {
        trailer_attempted_ = true;
        av_write_trailer(fmt_ctx_);
      }
      avformat_free_context(fmt_ctx_);
    }
    if (avio_ctx_) {
      av_freep(&avio_ctx_->buffer);
      avio_context_free(&avio_ctx_);
    }
  }

  bool write(const torch::Tensor& video,
             const torch::Tensor& audio,
             double fps,
             int32_t audio_sample_rate,
             std::string& raw_data) {
    if (!validate_inputs(video, audio, fps, audio_sample_rate)) {
      return false;
    }

    const int32_t frames = static_cast<int32_t>(video.size(0));
    const int32_t height = static_cast<int32_t>(video.size(2));
    const int32_t width = static_cast<int32_t>(video.size(3));
    const int32_t channels = static_cast<int32_t>(audio.size(0));
    const int64_t samples = audio.size(1);

    if (!init_output() || !init_video(width, height, fps) ||
        !init_audio(channels, audio_sample_rate) ||
        avformat_write_header(fmt_ctx_, nullptr) < 0) {
      LOG(ERROR) << "MemoryVideoAudioWriter: failed to initialize MP4 streams";
      return false;
    }
    header_written_ = true;

    if (!init_frames(width, height, channels, audio_sample_rate)) {
      return false;
    }

    auto audio_acc = audio.accessor<float, 2>();
    int32_t video_idx = 0;
    int64_t audio_offset = 0;
    int64_t audio_pts = 0;

    while (video_idx < frames || audio_offset < samples) {
      const bool write_video =
          video_idx < frames && (audio_offset >= samples ||
                                 av_compare_ts(video_idx,
                                               video_ctx_->time_base,
                                               audio_offset,
                                               audio_ctx_->time_base) <= 0);
      if (write_video) {
        if (!write_video_frame(video, video_idx, height, width)) {
          return false;
        }
        ++video_idx;
      } else {
        if (!write_audio_frame(
                audio_acc, audio_offset, samples, channels, audio_pts)) {
          return false;
        }
      }
    }

    if (!flush_encoder(video_ctx_, video_stream_) ||
        !flush_encoder(audio_ctx_, audio_stream_)) {
      return false;
    }
    trailer_attempted_ = true;
    if (av_write_trailer(fmt_ctx_) < 0) {
      return false;
    }
    avio_flush(avio_ctx_);
    finished_ = true;
    raw_data.assign(out_buf_.begin(), out_buf_.end());
    LOG(INFO) << "MemoryVideoAudioWriter: encoded " << frames << " frames and "
              << samples << " samples into " << raw_data.size() << " MP4 bytes";
    return !raw_data.empty();
  }

 private:
  bool validate_inputs(const torch::Tensor& video,
                       const torch::Tensor& audio,
                       double fps,
                       int32_t audio_sample_rate) const {
    if (!video.defined() || video.dim() != 4 || video.size(1) != 3 ||
        !video.device().is_cpu() || video.scalar_type() != torch::kFloat32 ||
        !video.is_contiguous()) {
      LOG(ERROR) << "MemoryVideoAudioWriter: video must be contiguous CPU "
                    "float32 [T,3,H,W]";
      return false;
    }
    if (!audio.defined() || audio.dim() != 2 ||
        (audio.size(0) != 1 && audio.size(0) != 2) ||
        !audio.device().is_cpu() || audio.scalar_type() != torch::kFloat32 ||
        !audio.is_contiguous()) {
      LOG(ERROR) << "MemoryVideoAudioWriter: audio must be contiguous CPU "
                    "float32 [C,N] with one or two channels";
      return false;
    }
    if (video.size(0) <= 0 || video.size(2) <= 0 || video.size(3) <= 0 ||
        video.size(2) % 2 != 0 || video.size(3) % 2 != 0 ||
        audio.size(1) <= 0 || !std::isfinite(fps) || fps <= 0.0 ||
        fps > 1000.0 || audio_sample_rate < 8000 ||
        audio_sample_rate > 192000 ||
        video.size(0) > std::numeric_limits<int32_t>::max() ||
        video.size(2) > std::numeric_limits<int32_t>::max() ||
        video.size(3) > std::numeric_limits<int32_t>::max() / 3 ||
        static_cast<uint64_t>(video.size(2)) *
                static_cast<uint64_t>(video.size(3)) * 3 >
            std::numeric_limits<size_t>::max()) {
      LOG(ERROR) << "MemoryVideoAudioWriter: invalid media dimensions";
      return false;
    }
    return true;
  }

  bool init_output() {
    constexpr int32_t avio_buf_size = 1 << 16;
    uint8_t* avio_buf =
        static_cast<uint8_t*>(av_malloc(static_cast<size_t>(avio_buf_size)));
    if (!avio_buf) {
      return false;
    }
    avio_ctx_ = avio_alloc_context(avio_buf,
                                   avio_buf_size,
                                   1,
                                   &write_ctx_,
                                   nullptr,
                                   &Writer::write,
                                   &Writer::seek);
    if (!avio_ctx_) {
      av_free(avio_buf);
      return false;
    }
    avio_ctx_->seekable = AVIO_SEEKABLE_NORMAL;

    const AVOutputFormat* format = av_guess_format("mp4", nullptr, nullptr);
    if (!format ||
        avformat_alloc_output_context2(&fmt_ctx_, format, nullptr, nullptr) <
            0 ||
        !fmt_ctx_) {
      return false;
    }
    fmt_ctx_->pb = avio_ctx_;
    fmt_ctx_->flags |= AVFMT_FLAG_CUSTOM_IO;
    pkt_ = av_packet_alloc();
    return pkt_ != nullptr;
  }

  bool init_video(int32_t width, int32_t height, double fps) {
    const AVCodec* codec = avcodec_find_encoder_by_name("libx264");
    if (!codec || codec->id != AV_CODEC_ID_H264) {
      LOG(ERROR) << "MemoryVideoAudioWriter: libx264 encoder is required";
      return false;
    }
    video_ctx_ = avcodec_alloc_context3(codec);
    if (!video_ctx_) {
      return false;
    }
    const AVRational framerate = av_d2q(fps, 1000000);
    if (framerate.num <= 0 || framerate.den <= 0) {
      return false;
    }
    video_ctx_->width = width;
    video_ctx_->height = height;
    video_ctx_->time_base = av_inv_q(framerate);
    video_ctx_->framerate = framerate;
    video_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    video_ctx_->max_b_frames = 0;
    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
      video_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "crf", "18", 0);
    av_dict_set(&opts, "preset", "medium", 0);
    av_dict_set(&opts, "profile", "high", 0);
    av_dict_set(&opts, "tune", "zerolatency", 0);
    const int32_t open_result = avcodec_open2(video_ctx_, codec, &opts);
    av_dict_free(&opts);
    if (open_result < 0) {
      return false;
    }
    video_stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    if (!video_stream_) {
      return false;
    }
    video_stream_->time_base = video_ctx_->time_base;
    video_stream_->avg_frame_rate = video_ctx_->framerate;
    return avcodec_parameters_from_context(video_stream_->codecpar,
                                           video_ctx_) >= 0;
  }

  bool init_audio(int32_t channels, int32_t sample_rate) {
    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
    if (!codec) {
      LOG(ERROR) << "MemoryVideoAudioWriter: AAC encoder is required";
      return false;
    }
    if (codec->sample_fmts) {
      bool supports_fltp = false;
      for (const AVSampleFormat* fmt = codec->sample_fmts;
           *fmt != AV_SAMPLE_FMT_NONE;
           ++fmt) {
        supports_fltp |= *fmt == AV_SAMPLE_FMT_FLTP;
      }
      if (!supports_fltp) {
        LOG(ERROR) << "MemoryVideoAudioWriter: AAC encoder lacks FLTP support";
        return false;
      }
    }
    if (codec->supported_samplerates) {
      bool supports_rate = false;
      for (const int32_t* rate = codec->supported_samplerates; *rate != 0;
           ++rate) {
        supports_rate |= *rate == sample_rate;
      }
      if (!supports_rate) {
        LOG(ERROR) << "MemoryVideoAudioWriter: AAC encoder does not support "
                   << sample_rate << " Hz";
        return false;
      }
    }
    if ((codec->capabilities & AV_CODEC_CAP_SMALL_LAST_FRAME) == 0) {
      LOG(ERROR) << "MemoryVideoAudioWriter: AAC encoder must support a small "
                    "final frame";
      return false;
    }

    audio_ctx_ = avcodec_alloc_context3(codec);
    if (!audio_ctx_) {
      return false;
    }
    audio_ctx_->sample_fmt = AV_SAMPLE_FMT_FLTP;
    audio_ctx_->sample_rate = sample_rate;
    audio_ctx_->time_base = {1, sample_rate};
    audio_ctx_->bit_rate = channels == 2 ? 192000 : 128000;
    audio_ctx_->profile = FF_PROFILE_AAC_LOW;
    av_channel_layout_default(&audio_ctx_->ch_layout, channels);
    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
      audio_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    if (avcodec_open2(audio_ctx_, codec, nullptr) < 0) {
      return false;
    }
    audio_stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    if (!audio_stream_) {
      return false;
    }
    audio_stream_->time_base = audio_ctx_->time_base;
    return avcodec_parameters_from_context(audio_stream_->codecpar,
                                           audio_ctx_) >= 0;
  }

  bool init_frames(int32_t width,
                   int32_t height,
                   int32_t channels,
                   int32_t sample_rate) {
    sws_ctx_ = sws_getContext(width,
                              height,
                              AV_PIX_FMT_RGB24,
                              width,
                              height,
                              AV_PIX_FMT_YUV420P,
                              SWS_BILINEAR,
                              nullptr,
                              nullptr,
                              nullptr);
    video_frame_ = av_frame_alloc();
    audio_frame_ = av_frame_alloc();
    if (!sws_ctx_ || !video_frame_ || !audio_frame_) {
      return false;
    }
    video_frame_->format = AV_PIX_FMT_YUV420P;
    video_frame_->width = width;
    video_frame_->height = height;
    if (av_frame_get_buffer(video_frame_, 0) < 0) {
      return false;
    }

    audio_frame_capacity_ =
        audio_ctx_->frame_size > 0 ? audio_ctx_->frame_size : 1024;
    audio_frame_->format = AV_SAMPLE_FMT_FLTP;
    audio_frame_->sample_rate = sample_rate;
    audio_frame_->nb_samples = audio_frame_capacity_;
    if (av_channel_layout_copy(&audio_frame_->ch_layout,
                               &audio_ctx_->ch_layout) < 0 ||
        av_frame_get_buffer(audio_frame_, 0) < 0) {
      return false;
    }
    rgb_buf_.resize(static_cast<size_t>(height) * width * 3);
    return channels == audio_ctx_->ch_layout.nb_channels;
  }

  bool write_video_frame(const torch::Tensor& video,
                         int32_t frame_idx,
                         int32_t height,
                         int32_t width) {
    const int32_t stride = width * 3;
    torch::Tensor rgb = torch::nan_to_num(video[frame_idx], 0.0, 1.0, 0.0)
                            .clamp(0.0, 1.0)
                            .mul(255.0)
                            .to(torch::kUInt8)
                            .permute({1, 2, 0})
                            .contiguous();
    std::memcpy(rgb_buf_.data(),
                rgb.data_ptr<uint8_t>(),
                static_cast<size_t>(height) * stride);
    if (av_frame_make_writable(video_frame_) < 0) {
      return false;
    }
    const uint8_t* src_data[1] = {rgb_buf_.data()};
    int32_t src_linesize[1] = {stride};
    if (sws_scale(sws_ctx_,
                  src_data,
                  src_linesize,
                  0,
                  height,
                  video_frame_->data,
                  video_frame_->linesize) != height) {
      return false;
    }
    video_frame_->pts = frame_idx;
    return send_frame(video_ctx_, video_stream_, video_frame_);
  }

  bool write_audio_frame(torch::TensorAccessor<float, 2> audio,
                         int64_t& offset,
                         int64_t total_samples,
                         int32_t channels,
                         int64_t& pts) {
    const int32_t input_samples = static_cast<int32_t>(
        std::min<int64_t>(audio_frame_capacity_, total_samples - offset));
    const int32_t output_samples = input_samples;
    audio_frame_->nb_samples = audio_frame_capacity_;
    if (av_frame_make_writable(audio_frame_) < 0) {
      return false;
    }
    for (int32_t channel = 0; channel < channels; ++channel) {
      float* dst = reinterpret_cast<float*>(audio_frame_->data[channel]);
      std::fill(dst, dst + audio_frame_capacity_, 0.0f);
      for (int32_t sample = 0; sample < input_samples; ++sample) {
        const float value = audio[channel][offset + sample];
        dst[sample] =
            std::isfinite(value) ? std::clamp(value, -1.0f, 1.0f) : 0.0f;
      }
    }
    audio_frame_->nb_samples = output_samples;
    audio_frame_->pts = pts;
    if (!send_frame(audio_ctx_, audio_stream_, audio_frame_)) {
      return false;
    }
    offset += input_samples;
    pts += output_samples;
    return true;
  }

  bool send_frame(AVCodecContext* codec_ctx, AVStream* stream, AVFrame* frame) {
    if (avcodec_send_frame(codec_ctx, frame) < 0) {
      return false;
    }
    return drain_packets(codec_ctx, stream);
  }

  bool flush_encoder(AVCodecContext* codec_ctx, AVStream* stream) {
    const int32_t result = avcodec_send_frame(codec_ctx, nullptr);
    if (result < 0 && result != AVERROR_EOF) {
      return false;
    }
    return drain_packets(codec_ctx, stream);
  }

  bool drain_packets(AVCodecContext* codec_ctx, AVStream* stream) {
    while (true) {
      const int32_t result = avcodec_receive_packet(codec_ctx, pkt_);
      if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) {
        return true;
      }
      if (result < 0) {
        return false;
      }
      if (codec_ctx->codec_type == AVMEDIA_TYPE_VIDEO && pkt_->duration <= 0) {
        pkt_->duration = 1;
      }
      av_packet_rescale_ts(pkt_, codec_ctx->time_base, stream->time_base);
      pkt_->stream_index = stream->index;
      const int32_t write_result = av_interleaved_write_frame(fmt_ctx_, pkt_);
      av_packet_unref(pkt_);
      if (write_result < 0) {
        return false;
      }
    }
  }

  AVFormatContext* fmt_ctx_ = nullptr;
  AVIOContext* avio_ctx_ = nullptr;
  AVCodecContext* video_ctx_ = nullptr;
  AVCodecContext* audio_ctx_ = nullptr;
  AVStream* video_stream_ = nullptr;
  AVStream* audio_stream_ = nullptr;
  AVPacket* pkt_ = nullptr;
  AVFrame* video_frame_ = nullptr;
  AVFrame* audio_frame_ = nullptr;
  SwsContext* sws_ctx_ = nullptr;
  int32_t audio_frame_capacity_ = 0;
  std::vector<uint8_t> rgb_buf_;
  MemWriteCtx write_ctx_{&out_buf_, 0};
  std::vector<uint8_t> out_buf_;
  bool header_written_ = false;
  bool trailer_attempted_ = false;
  bool finished_ = false;
};

bool FFmpegVideoEncoder::encode(const torch::Tensor& video,
                                double fps,
                                const std::string& format,
                                std::string& raw_data) {
  MemoryVideoWriter writer;
  return writer.write(video, fps, format, raw_data);
}

bool FFmpegVideoAudioEncoder::encode(const torch::Tensor& video,
                                     const torch::Tensor& audio,
                                     double fps,
                                     int32_t audio_sample_rate,
                                     std::string& raw_data) {
  MemoryVideoAudioWriter writer;
  return writer.write(video, audio, fps, audio_sample_rate, raw_data);
}

}  // namespace xllm
