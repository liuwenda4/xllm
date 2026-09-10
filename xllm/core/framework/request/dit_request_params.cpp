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

#include "dit_request_params.h"

#include <algorithm>

#include "butil/base64.h"
#include "core/common/instance_name.h"
#include "core/common/macros.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/multimodal/mm_codec.h"
#include "core/util/utils.h"
#include "core/util/uuid.h"
#include "request.h"

namespace xllm {
namespace {
thread_local ShortUUID short_uuid;

constexpr char kMiniMaxH3ConditionSchema[] =
    "xllm.minimax_h3.text_conditioning/v1";

bool has_supported_tensor_datatype(const proto::Tensor& tensor) {
  const std::string& datatype = tensor.datatype();
  return datatype == "BOOL" || datatype == "INT32" || datatype == "INT64" ||
         datatype == "UINT32" || datatype == "UINT64" || datatype == "FP32" ||
         datatype == "FP64" || datatype == "BYTES" || datatype == "FP16" ||
         datatype == "BF16";
}

bool json_integer_equals(const nlohmann::json& object,
                         const char* key,
                         int64_t expected) {
  const auto iter = object.find(key);
  return iter != object.end() && iter->is_number_integer() && *iter == expected;
}

bool json_string_equals(const nlohmann::json& object,
                        const char* key,
                        const std::string& expected) {
  const auto iter = object.find(key);
  return iter != object.end() && iter->is_string() && *iter == expected;
}

bool json_has_sha256(const nlohmann::json& object, const char* key) {
  const auto iter = object.find(key);
  if (iter == object.end() || !iter->is_string()) {
    return false;
  }
  const std::string value = *iter;
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char ch) {
           return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
         });
}

std::optional<std::string> verify_minimax_h3_condition_bundle(
    const DiTInputParams& input) {
  const bool has_any_bundle_field =
      input.text_token_tags_is_set || input.condition_schema.has_value() ||
      input.condition_source_backend.has_value() ||
      input.condition_manifest_json.has_value();
  if (!has_any_bundle_field) {
    return std::nullopt;
  }

  if (!input.text_token_tags.defined() || !input.condition_schema.has_value() ||
      !input.condition_source_backend.has_value() ||
      !input.condition_manifest_json.has_value()) {
    return "MiniMax-H3 condition bundle is incomplete";
  }

  if (*input.condition_schema != kMiniMaxH3ConditionSchema) {
    return "unsupported MiniMax-H3 condition_schema";
  }
  if (*input.condition_source_backend != "official_hf" &&
      *input.condition_source_backend != "xllm_native") {
    return "unsupported MiniMax-H3 condition_source_backend";
  }

  if (!input.prompt_embed.defined() ||
      input.prompt_embed.scalar_type() != torch::kBFloat16 ||
      !input.prompt_embed.is_contiguous() || input.prompt_embed.dim() != 2 ||
      input.prompt_embed.size(0) <= 0 || input.prompt_embed.size(1) != 5120) {
    return "MiniMax-H3 prompt_embed must be contiguous BF16 with shape "
           "[tokens,5120]";
  }

  if (input.text_token_tags.scalar_type() != torch::kInt64 ||
      !input.text_token_tags.is_contiguous() ||
      input.text_token_tags.dim() != 1 ||
      input.text_token_tags.size(0) != input.prompt_embed.size(0)) {
    return "MiniMax-H3 text_token_tags must be contiguous int64 with shape "
           "[tokens] matching prompt_embed";
  }
  if (!torch::logical_or(input.text_token_tags == 0, input.text_token_tags == 1)
           .all()
           .item<bool>()) {
    return "MiniMax-H3 text_token_tags values must be 0 or 1";
  }

  const nlohmann::json manifest =
      nlohmann::json::parse(*input.condition_manifest_json,
                            /*cb=*/nullptr,
                            /*allow_exceptions=*/false);
  if (manifest.is_discarded() || !manifest.is_object()) {
    return "MiniMax-H3 condition_manifest_json must be a JSON object";
  }
  if (!json_string_equals(manifest, "schema", *input.condition_schema) ||
      !json_string_equals(
          manifest, "source_backend", *input.condition_source_backend) ||
      !json_integer_equals(manifest, "decoder_layer_index", 49) ||
      !json_integer_equals(manifest, "hidden_state_slot", 50) ||
      !json_integer_equals(
          manifest, "token_count", input.prompt_embed.size(0)) ||
      !json_has_sha256(manifest, "hidden_digest") ||
      !json_has_sha256(manifest, "token_tags_digest")) {
    return "MiniMax-H3 condition manifest does not match the supplied bundle";
  }

  return std::nullopt;
}

std::string generate_request_id(const std::string& prefix) {
  return prefix + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

// Decode a base64-encoded WAV/audio blob into a float32 CPU tensor of shape
// (1, num_samples) at the given sample rate (mono).
// Returns true on success and writes to `out`; logs ERROR and returns false
// on any decoding failure.
bool decode_prompt_audio(const std::string& b64_audio,
                         int64_t target_sample_rate,
                         torch::Tensor& out) {
  std::string raw_bytes;
  if (!butil::Base64Decode(b64_audio, &raw_bytes)) {
    LOG(ERROR) << "Base64 prompt_audio decode failed";
    return false;
  }
  FFmpegAudioDecoder audio_decoder;
  AudioMetadata audio_meta;
  torch::Tensor audio_tensor;
  if (!audio_decoder.decode(
          raw_bytes, audio_tensor, audio_meta, target_sample_rate)) {
    LOG(ERROR) << "prompt_audio WAV decode failed";
    return false;
  }
  // Ensure shape (1, num_samples): FFmpegAudioDecoder may return
  // (num_samples,) or (num_channels, num_samples).
  if (audio_tensor.dim() == 1) {
    audio_tensor = audio_tensor.unsqueeze(0);
  } else if (audio_tensor.dim() == 2 && audio_tensor.size(0) > 1) {
    audio_tensor = audio_tensor.mean(0, /*keepdim=*/true);
  }
  out = audio_tensor.to(torch::kFloat32);
  return true;
}

}  // namespace

std::pair<int, int> split_resolution(const std::string& s) {
  size_t pos = s.find('*');
  int width = std::stoi(s.substr(0, pos));
  int height = std::stoi(s.substr(pos + 1));
  return {width, height};
}

// Decode a base64-encoded image string into a torch tensor via OpenCV.
bool decode_base64_image(const std::string& base64, torch::Tensor& out) {
  std::string raw_bytes;
  if (!butil::Base64Decode(base64, &raw_bytes)) {
    LOG(ERROR) << "Base64 decode failed";
    return false;
  }
  OpenCVImageDecoder decoder;
  if (!decoder.decode(raw_bytes, out)) {
    LOG(ERROR) << "Image decode failed";
    return false;
  }
  return true;
}

// Shared helper: populate DiTInputParams fields common to Image and Video
// input.
template <typename InputProto>
void fill_input_params(DiTInputParams& input_params,
                       const InputProto& input,
                       bool skip_unsupported_tensor_datatypes = false) {
  input_params.prompt = input.prompt();
  if (input.has_negative_prompt()) {
    input_params.negative_prompt = input.negative_prompt();
  }
  if (input.has_prompt_embed() &&
      (!skip_unsupported_tensor_datatypes ||
       has_supported_tensor_datatype(input.prompt_embed()))) {
    input_params.prompt_embed = util::proto_to_torch(input.prompt_embed());
  }
  if (input.has_negative_prompt_embed() &&
      (!skip_unsupported_tensor_datatypes ||
       has_supported_tensor_datatype(input.negative_prompt_embed()))) {
    input_params.negative_prompt_embed =
        util::proto_to_torch(input.negative_prompt_embed());
  }
  if (input.has_image()) {
    decode_base64_image(input.image(), input_params.image);
  }
}

// Shared helper: populate generation params common to Image and Video
// parameters.
template <typename ParamsProto>
void fill_generation_params(DiTGenerationParams& generation_params,
                            const ParamsProto& params) {
  if (params.has_size()) {
    auto [w, h] = split_resolution(params.size());
    generation_params.width = w;
    generation_params.height = h;
  }
  if (params.has_num_inference_steps()) {
    generation_params.num_inference_steps = params.num_inference_steps();
  }
  if (params.has_true_cfg_scale()) {
    generation_params.true_cfg_scale = params.true_cfg_scale();
  }
  if (params.has_guidance_scale()) {
    generation_params.guidance_scale = params.guidance_scale();
  }
  if (params.has_seed()) {
    generation_params.seed = params.seed();
    generation_params.seed_is_set = true;
  }
  if (params.has_max_sequence_length()) {
    generation_params.max_sequence_length = params.max_sequence_length();
  }
}

// ---------------------------------------------------------------------------
// Image generation constructor
// ---------------------------------------------------------------------------
DiTRequestParams::DiTRequestParams(const proto::ImageGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime) {
  request_kind = DiTRequestKind::kImage;
  request_id = request.has_request_id() ? request.request_id()
                                        : generate_request_id("imggen-");
  x_request_id = x_rid;
  x_request_time = x_rtime;
  model = request.model();

  if (request.has_input()) {
    const auto& input = request.input();
    fill_input_params(input_params, input);
    // Image-only input fields
    if (input.has_prompt_2()) {
      input_params.prompt_2 = input.prompt_2();
    }
    if (input.has_negative_prompt_2()) {
      input_params.negative_prompt_2 = input.negative_prompt_2();
    }
    if (input.has_pooled_prompt_embed()) {
      input_params.pooled_prompt_embed =
          util::proto_to_torch(input.pooled_prompt_embed());
    }
    if (input.has_negative_pooled_prompt_embed()) {
      input_params.negative_pooled_prompt_embed =
          util::proto_to_torch(input.negative_pooled_prompt_embed());
    }
    if (input.has_latent()) {
      input_params.latent = util::proto_to_torch(input.latent());
    }
    if (input.has_masked_image_latent()) {
      input_params.masked_image_latent =
          util::proto_to_torch(input.masked_image_latent());
    }
    if (input.has_mask_image()) {
      decode_base64_image(input.mask_image(), input_params.mask_image);
    }
    input_params.images.reserve(input.images().size());
    for (const auto& image : input.images()) {
      torch::Tensor tensor;
      if (!decode_base64_image(image, tensor)) {
        continue;
      }
      input_params.images.emplace_back(std::move(tensor));
    }
    if (input.has_control_image()) {
      decode_base64_image(input.control_image(), input_params.control_image);
    }
  }

  generation_params.num_images_per_prompt = 1;
  if (request.has_parameters()) {
    const auto& params = request.parameters();
    fill_generation_params(generation_params, params);
    if (params.has_num_images_per_prompt()) {
      generation_params.num_images_per_prompt =
          static_cast<uint32_t>(params.num_images_per_prompt());
    }
    // Image-only generation fields
    if (params.has_enable_cfg_renorm()) {
      generation_params.enable_cfg_renorm = params.enable_cfg_renorm();
    }
    if (params.has_cfg_renorm_min()) {
      generation_params.cfg_renorm_min = params.cfg_renorm_min();
    }
  }
}

// ---------------------------------------------------------------------------
// Text generation constructor (Cola-DLM)
// ---------------------------------------------------------------------------
DiTRequestParams::DiTRequestParams(const proto::TextGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime) {
  request_kind = DiTRequestKind::kText;
  // Cola-DLM official defaults (inference.py / run_cola.py), not image DiT.
  generation_params.guidance_scale = 7.0f;
  generation_params.max_new_tokens = 32;
  request_id = request.has_request_id() ? request.request_id()
                                        : generate_request_id("textgen-");
  x_request_id = x_rid;
  x_request_time = x_rtime;
  model = request.model();

  if (request.has_input()) {
    input_params.prompt = request.input().prompt();
  }

  if (request.has_parameters()) {
    const auto& params = request.parameters();
    if (params.has_seed()) {
      generation_params.seed = params.seed();
      generation_params.seed_is_set = true;
    }
    if (params.has_max_new_tokens()) {
      generation_params.max_new_tokens = params.max_new_tokens();
    }
    if (params.has_diffusion_steps()) {
      generation_params.diffusion_steps = params.diffusion_steps();
    }
    if (params.has_guidance_scale()) {
      generation_params.guidance_scale = params.guidance_scale();
    }
    if (params.has_temperature()) {
      generation_params.temperature = params.temperature();
    }
    if (params.has_top_k()) {
      generation_params.top_k = params.top_k();
    }
    if (params.has_top_p()) {
      generation_params.top_p = params.top_p();
    }
    if (params.has_repetition_penalty()) {
      generation_params.repetition_penalty = params.repetition_penalty();
    }
  }
}

// ---------------------------------------------------------------------------
// Audio generation constructor
// ---------------------------------------------------------------------------
DiTRequestParams::DiTRequestParams(const proto::AudioGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime) {
  request_kind = DiTRequestKind::kAudio;
  if (request.has_request_id()) {
    request_id = request.request_id();
  } else {
    request_id = generate_request_id("audiogen-");
  }
  x_request_id = x_rid;
  x_request_time = x_rtime;
  model = request.model();

  // generation params — must be parsed before input to make audio_sampling_rate
  // available for prompt audio decoding.
  const auto& params = request.parameters();
  if (params.has_seed()) {
    generation_params.seed = params.seed();
    generation_params.seed_is_set = true;
  }
  if (params.has_max_sequence_length()) {
    generation_params.max_sequence_length = params.max_sequence_length();
  }
  if (params.has_guidance_scale()) {
    generation_params.guidance_scale = params.guidance_scale();
  }
  if (params.has_audio_duration_frames()) {
    generation_params.audio_duration_frames = params.audio_duration_frames();
  }
  if (params.has_audio_steps()) {
    generation_params.audio_steps = params.audio_steps();
  }
  if (params.has_audio_guidance_method()) {
    generation_params.audio_guidance_method = params.audio_guidance_method();
  }
  if (params.has_sampling_rate()) {
    generation_params.audio_sampling_rate = params.sampling_rate();
  }

  // input params — decode prompt audio using the model's sampling rate.
  const auto& input = request.input();
  input_params.prompt = input.prompt();

  if (input.has_prompt_audio()) {
    decode_prompt_audio(input.prompt_audio(),
                        generation_params.audio_sampling_rate,
                        input_params.prompt_audio);
  }
  if (input.has_prompt_text()) {
    input_params.audio_prompt_text = input.prompt_text();
  }
}

// ---------------------------------------------------------------------------
// Video generation constructor
// ---------------------------------------------------------------------------
DiTRequestParams::DiTRequestParams(const proto::VideoGenerationRequest& request,
                                   const std::string& x_rid,
                                   const std::string& x_rtime) {
  request_kind = DiTRequestKind::kVideo;
  request_id = request.has_request_id() ? request.request_id()
                                        : generate_request_id("vidgen-");
  x_request_id = x_rid;
  x_request_time = x_rtime;
  model = request.model();

  generation_params.force_video_output = true;

  if (request.has_input()) {
    const auto& input = request.input();
    const bool has_condition_bundle = input.has_text_token_tags() ||
                                      input.has_condition_schema() ||
                                      input.has_condition_source_backend() ||
                                      input.has_condition_manifest_json();
    fill_input_params(input_params,
                      input,
                      /*skip_unsupported_tensor_datatypes=*/
                      has_condition_bundle);
    // Video-only input fields
    if (input.has_last_image()) {
      decode_base64_image(input.last_image(), input_params.last_image);
    }
    if (input.has_text_token_tags()) {
      input_params.text_token_tags_is_set = true;
      if (has_supported_tensor_datatype(input.text_token_tags())) {
        input_params.text_token_tags =
            util::proto_to_torch(input.text_token_tags());
      }
    }
    if (input.has_condition_schema()) {
      input_params.condition_schema = input.condition_schema();
    }
    if (input.has_condition_source_backend()) {
      input_params.condition_source_backend = input.condition_source_backend();
    }
    if (input.has_condition_manifest_json()) {
      input_params.condition_manifest_json = input.condition_manifest_json();
    }
  }

  if (request.has_parameters()) {
    const auto& params = request.parameters();
    fill_generation_params(generation_params, params);
    if (params.has_num_videos_per_prompt()) {
      generation_params.num_videos_per_prompt =
          static_cast<uint32_t>(params.num_videos_per_prompt());
    }
    if (params.has_num_frames()) {
      generation_params.num_frames = params.num_frames();
    }
    if (params.has_fps()) {
      generation_params.video_fps = params.fps();
    }
    if (params.has_guidance_scale_2()) {
      generation_params.guidance_scale_2 = params.guidance_scale_2();
    }
    if (params.has_seconds()) {
      generation_params.seconds = params.seconds();
    }
    if (params.has_boundary_ratio()) {
      generation_params.boundary_ratio = params.boundary_ratio();
    }
    if (params.has_flow_shift()) {
      generation_params.flow_shift = params.flow_shift();
    }
  }
}

bool DiTRequestParams::verify_params(
    std::function<bool(DiTRequestOutput)> callback) const {
  if (input_params.prompt.empty() && !input_params.prompt_embed.defined()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "prompt is empty");
    return false;
  }

  if (request_kind == DiTRequestKind::kText) {
    if (model.empty()) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "model is empty");
      return false;
    }
    if (generation_params.max_new_tokens <= 0) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "max_new_tokens must be positive.");
      return false;
    }
    if (generation_params.diffusion_steps <= 0) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                          "diffusion_steps must be positive.");
      return false;
    }
    return true;
  }

  if (request_kind == DiTRequestKind::kAudio) {
    return true;
  }

  if (request_kind == DiTRequestKind::kVideo) {
    if (const auto error = verify_minimax_h3_condition_bundle(input_params)) {
      CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, *error);
      return false;
    }
  }

  if (generation_params.width <= 0 || generation_params.height <= 0) {
    CALLBACK_WITH_ERROR(
        StatusCode::INVALID_ARGUMENT,
        "Invalid image dimensions: width and height must be positive.");
    return false;
  }

  if (generation_params.num_inference_steps <= 0) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "num_inference_steps must be positive.");
    return false;
  }

  if (generation_params.num_images_per_prompt == 0) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "num_images_per_prompt must be greater than 0.");
    return false;
  }

  if (generation_params.num_videos_per_prompt == 0) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "num_videos_per_prompt must be greater than 0.");
    return false;
  }

  // Check if the image area exceeds the maximum allowed area.
  if (::xllm::DiTConfig::get_instance().dit_generation_image_area_max() > 0) {
    int64_t area = static_cast<int64_t>(generation_params.width) *
                   static_cast<int64_t>(generation_params.height);
    if (area >
        ::xllm::DiTConfig::get_instance().dit_generation_image_area_max()) {
      CALLBACK_WITH_ERROR(
          StatusCode::INVALID_ARGUMENT,
          "Requested image area (" + std::to_string(area) +
              ") exceeds the maximum allowed area (" +
              std::to_string(::xllm::DiTConfig::get_instance()
                                 .dit_generation_image_area_max()) +
              ").");
      return false;
    }
  }

  return true;
}

}  // namespace xllm
