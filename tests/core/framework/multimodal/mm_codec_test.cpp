/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/framework/multimodal/mm_codec.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
}

namespace xllm {
namespace {

struct MemoryReadContext {
  const uint8_t* data = nullptr;
  int64_t size = 0;
  int64_t position = 0;
};

int read_packet(void* opaque, uint8_t* buffer, int buffer_size) {
  auto* context = static_cast<MemoryReadContext*>(opaque);
  const int64_t remaining = context->size - context->position;
  const int64_t bytes = std::min<int64_t>(remaining, buffer_size);
  if (bytes <= 0) {
    return AVERROR_EOF;
  }
  std::memcpy(buffer, context->data + context->position, bytes);
  context->position += bytes;
  return static_cast<int>(bytes);
}

int64_t seek_packet(void* opaque, int64_t offset, int whence) {
  auto* context = static_cast<MemoryReadContext*>(opaque);
  if (whence == AVSEEK_SIZE) {
    return context->size;
  }
  int64_t position = 0;
  if (whence == SEEK_SET) {
    position = offset;
  } else if (whence == SEEK_CUR) {
    position = context->position + offset;
  } else if (whence == SEEK_END) {
    position = context->size + offset;
  } else {
    return AVERROR(EINVAL);
  }
  if (position < 0 || position > context->size) {
    return AVERROR(EINVAL);
  }
  context->position = position;
  return position;
}

void expect_h264_aac_streams(const std::string& mp4) {
  MemoryReadContext read_context{reinterpret_cast<const uint8_t*>(mp4.data()),
                                 static_cast<int64_t>(mp4.size()),
                                 0};
  constexpr int kAvioBufferSize = 4096;
  uint8_t* avio_buffer = static_cast<uint8_t*>(av_malloc(kAvioBufferSize));
  ASSERT_NE(avio_buffer, nullptr);
  AVIOContext* avio_context = avio_alloc_context(avio_buffer,
                                                 kAvioBufferSize,
                                                 0,
                                                 &read_context,
                                                 &read_packet,
                                                 nullptr,
                                                 &seek_packet);
  ASSERT_NE(avio_context, nullptr);
  avio_context->seekable = AVIO_SEEKABLE_NORMAL;
  AVFormatContext* format_context = avformat_alloc_context();
  ASSERT_NE(format_context, nullptr);
  format_context->pb = avio_context;
  format_context->flags |= AVFMT_FLAG_CUSTOM_IO;
  ASSERT_GE(avformat_open_input(&format_context, nullptr, nullptr, nullptr), 0);
  ASSERT_GE(avformat_find_stream_info(format_context, nullptr), 0);
  ASSERT_EQ(format_context->nb_streams, 2u);

  int video_index = -1;
  int audio_index = -1;
  for (unsigned int index = 0; index < format_context->nb_streams; ++index) {
    const AVCodecParameters* params = format_context->streams[index]->codecpar;
    if (params->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_index = static_cast<int>(index);
      EXPECT_EQ(params->codec_id, AV_CODEC_ID_H264);
    } else if (params->codec_type == AVMEDIA_TYPE_AUDIO) {
      audio_index = static_cast<int>(index);
      EXPECT_EQ(params->codec_id, AV_CODEC_ID_AAC);
      EXPECT_EQ(params->ch_layout.nb_channels, 2);
      EXPECT_EQ(params->sample_rate, 32000);
      AVChannelLayout expected_layout;
      av_channel_layout_default(&expected_layout, 2);
      EXPECT_EQ(av_channel_layout_compare(&params->ch_layout, &expected_layout),
                0);
      av_channel_layout_uninit(&expected_layout);
    }
  }
  ASSERT_GE(video_index, 0);
  ASSERT_GE(audio_index, 0);
  const AVStream* video_stream = format_context->streams[video_index];
  const AVStream* audio_stream = format_context->streams[audio_index];
  EXPECT_EQ(video_stream->start_time, 0);
  EXPECT_EQ(audio_stream->start_time, 0);
  const double video_duration =
      video_stream->duration * av_q2d(video_stream->time_base);
  const double audio_duration =
      audio_stream->duration * av_q2d(audio_stream->time_base);
  EXPECT_NEAR(video_duration, 124.0 / 24.0, 1e-6);
  EXPECT_NEAR(audio_duration, 165600.0 / 32000.0, 1e-6);
  EXPECT_NEAR(audio_duration - video_duration, 1.0 / 120.0, 1e-6);

  std::vector<int64_t> last_dts(format_context->nb_streams, AV_NOPTS_VALUE);
  std::vector<int32_t> packet_counts(format_context->nb_streams, 0);
  AVPacket* packet = av_packet_alloc();
  ASSERT_NE(packet, nullptr);
  while (av_read_frame(format_context, packet) >= 0) {
    if (packet->dts != AV_NOPTS_VALUE) {
      const int stream_index = packet->stream_index;
      if (last_dts[stream_index] != AV_NOPTS_VALUE) {
        EXPECT_GT(packet->dts, last_dts[stream_index]);
      }
      last_dts[stream_index] = packet->dts;
    }
    ++packet_counts[packet->stream_index];
    av_packet_unref(packet);
  }
  EXPECT_EQ(packet_counts[video_index], 124);
  EXPECT_GT(packet_counts[audio_index], 0);

  av_packet_free(&packet);
  avformat_close_input(&format_context);
  av_freep(&avio_context->buffer);
  avio_context_free(&avio_context);
}

TEST(MMCodecTest, EncodesH264AacMp4InMemory) {
  constexpr int32_t kFrames = 124;
  constexpr int32_t kHeight = 16;
  constexpr int32_t kWidth = 16;
  constexpr double kFps = 24.0;
  constexpr int32_t kSampleRate = 32000;
  constexpr int32_t kSamples = 165600;

  torch::Tensor video = torch::zeros({kFrames, 3, kHeight, kWidth},
                                     torch::dtype(torch::kFloat32));
  for (int32_t frame = 0; frame < kFrames; ++frame) {
    video[frame][frame % 3].fill_(static_cast<float>(frame + 1) / kFrames);
  }

  torch::Tensor time =
      torch::arange(kSamples, torch::dtype(torch::kFloat32)) / kSampleRate;
  constexpr double kPi = 3.14159265358979323846;
  torch::Tensor audio =
      torch::stack({0.1f * torch::sin(2.0 * kPi * 440.0 * time),
                    0.1f * torch::sin(2.0 * kPi * 660.0 * time)})
          .contiguous();

  FFmpegVideoAudioEncoder encoder;
  std::string mp4;
  ASSERT_TRUE(encoder.encode(video, audio, kFps, kSampleRate, mp4));
  EXPECT_GT(mp4.size(), 1024u);
  EXPECT_NE(mp4.find("ftyp"), std::string::npos);
  expect_h264_aac_streams(mp4);

  FFmpegVideoDecoder video_decoder;
  torch::Tensor decoded_video;
  VideoMetadata video_metadata;
  ASSERT_TRUE(video_decoder.decode(mp4, decoded_video, video_metadata));
  EXPECT_EQ(decoded_video.size(0), kFrames);
  EXPECT_EQ(decoded_video.size(1), 3);
  EXPECT_EQ(decoded_video.size(2), kHeight);
  EXPECT_EQ(decoded_video.size(3), kWidth);
  EXPECT_NEAR(video_metadata.fps, kFps, 0.01);

  FFmpegAudioDecoder audio_decoder;
  torch::Tensor decoded_audio;
  AudioMetadata audio_metadata;
  ASSERT_TRUE(
      audio_decoder.decode(mp4, decoded_audio, audio_metadata, kSampleRate));
  EXPECT_TRUE(decoded_audio.defined());
  EXPECT_GE(decoded_audio.numel(), kSamples);
  EXPECT_LT(decoded_audio.numel(), kSamples + 1024);
  EXPECT_EQ(audio_metadata.sample_rate, kSampleRate);
}

TEST(MMCodecTest, RejectsInvalidPairedMediaShapes) {
  FFmpegVideoAudioEncoder encoder;
  std::string output;
  const torch::Tensor invalid_video = torch::zeros({2, 1, 16, 16});
  const torch::Tensor stereo_audio = torch::zeros({2, 3200});
  EXPECT_FALSE(encoder.encode(invalid_video, stereo_audio, 8.0, 32000, output));

  const torch::Tensor video = torch::zeros({2, 3, 16, 16});
  const torch::Tensor invalid_audio = torch::zeros({3, 3200});
  EXPECT_FALSE(encoder.encode(video, invalid_audio, 8.0, 32000, output));
}

}  // namespace
}  // namespace xllm
