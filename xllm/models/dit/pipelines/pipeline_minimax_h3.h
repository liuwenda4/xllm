/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <glog/logging.h>
#include <openssl/sha.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/dit_model_context.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/multimodal/mm_codec.h"
#include "core/platform/device.h"
#include "core/runtime/dit_forward_params.h"
#include "core/util/json_reader.h"
#include "models/dit/autoencoders/autoencoder_kl_minimax_h3.h"
#include "models/dit/autoencoders/autoencoder_kl_minimax_h3_audio.h"
#include "models/dit/pipelines/minimax_h3_ref2va_eager.h"
#include "models/dit/transformers/minimax_h3_denoiser.h"
#include "models/dit/transformers/minimax_h3_tp_uaa_denoiser.h"
#include "models/dit/transformers/transformer_minimax_h3.h"
#include "models/dit/utils/lanczos_resample.h"
#include "models/dit/utils/minimax_h3_packing.h"
#include "models/model_registry.h"

namespace xllm {

struct MiniMaxH3ModelIndexSummary {
  std::string class_name;
  int64_t schema_version = 0;
  std::string partition;
  std::vector<std::string> tasks;
  double video_sigma_shift = 0.0;
  double audio_sigma_shift = 0.0;
  std::unordered_map<std::string, std::string> component_classes;
};

class MiniMaxH3ModelIndexContract {
 public:
  static MiniMaxH3ModelIndexSummary parse(const nlohmann::json& root) {
    if (!root.is_object()) {
      fail("root", "must be an object");
    }

    const std::unordered_set<std::string> allowed_root_keys = {
        "_class_name",
        "_diffusers_version",
        "_minimax_h3",
        "transformer",
        "video_vae",
        "audio_vae",
        "text_encoder",
        "tokenizer",
        "processor",
        "scheduler"};
    for (const auto& [key, value] : root.items()) {
      (void)value;
      if (!allowed_root_keys.contains(key)) {
        fail("root", "unknown key `" + key + "`");
      }
    }
    for (const auto& key : {"_class_name",
                            "_minimax_h3",
                            "transformer",
                            "video_vae",
                            "audio_vae",
                            "text_encoder",
                            "tokenizer",
                            "processor",
                            "scheduler"}) {
      if (!root.contains(key)) {
        fail("root", "missing required key `" + std::string(key) + "`");
      }
    }
    if (root.contains("_diffusers_version") &&
        !root.at("_diffusers_version").is_string()) {
      fail("_diffusers_version", "must be a string");
    }

    MiniMaxH3ModelIndexSummary summary;
    summary.class_name = require_string(root, "_class_name");
    if (summary.class_name != "MiniMaxH3Pipeline") {
      fail("_class_name", "must be `MiniMaxH3Pipeline`");
    }

    const auto& h3 = root.at("_minimax_h3");
    if (!h3.is_object()) {
      fail("_minimax_h3", "must be an object");
    }
    const std::unordered_set<std::string> allowed_h3_keys = {
        "schema_version",
        "partition",
        "tasks",
        "task_aliases",
        "sigma_shift_scales"};
    for (const auto& [key, value] : h3.items()) {
      (void)value;
      if (!allowed_h3_keys.contains(key)) {
        fail("_minimax_h3", "unknown key `" + key + "`");
      }
    }
    for (const auto& key :
         {"schema_version", "partition", "tasks", "sigma_shift_scales"}) {
      if (!h3.contains(key)) {
        fail("_minimax_h3", "missing required key `" + std::string(key) + "`");
      }
    }
    if (!h3.at("schema_version").is_number_integer()) {
      fail("_minimax_h3.schema_version", "must be integer 1");
    }
    summary.schema_version = h3.at("schema_version").get<int64_t>();
    if (summary.schema_version != 1) {
      fail("_minimax_h3.schema_version", "must be 1");
    }
    summary.partition = require_string(h3, "partition");
    if (summary.partition != "ref2va") {
      fail("_minimax_h3.partition",
           "must be `ref2va`; FL2VA and other partitions are not valid for "
           "this pipeline");
    }
    if (!h3.at("tasks").is_array()) {
      fail("_minimax_h3.tasks", "must be [`ref2va`]");
    }
    for (const auto& task : h3.at("tasks")) {
      if (!task.is_string()) {
        fail("_minimax_h3.tasks", "must contain only strings");
      }
    }
    summary.tasks = h3.at("tasks").get<std::vector<std::string>>();
    if (summary.tasks != std::vector<std::string>({"ref2va"})) {
      fail("_minimax_h3.tasks", "must be exactly [`ref2va`]");
    }
    if (h3.contains("task_aliases") && (!h3.at("task_aliases").is_object() ||
                                        !h3.at("task_aliases").empty())) {
      fail("_minimax_h3.task_aliases", "must be an empty object");
    }

    const auto& shifts = h3.at("sigma_shift_scales");
    if (!shifts.is_object() || shifts.size() != 2 ||
        !shifts.contains("video") || !shifts.contains("audio") ||
        !shifts.at("video").is_number() || !shifts.at("audio").is_number()) {
      fail("_minimax_h3.sigma_shift_scales",
           "must contain exactly numeric `video` and `audio` values");
    }
    summary.video_sigma_shift = shifts.at("video").get<double>();
    summary.audio_sigma_shift = shifts.at("audio").get<double>();
    if (!std::isfinite(summary.video_sigma_shift) ||
        summary.video_sigma_shift != 12.0) {
      fail("_minimax_h3.sigma_shift_scales.video", "must be 12");
    }
    if (!std::isfinite(summary.audio_sigma_shift) ||
        summary.audio_sigma_shift != 3.0) {
      fail("_minimax_h3.sigma_shift_scales.audio", "must be 3");
    }

    summary.component_classes.emplace(
        "transformer",
        require_component(
            root, "transformer", "diffusers", {"MiniMaxH3DiTModel"}));
    summary.component_classes.emplace(
        "video_vae",
        require_component(
            root, "video_vae", "diffusers", {"MiniMaxH3VideoVAE"}));
    summary.component_classes.emplace(
        "audio_vae",
        require_component(
            root, "audio_vae", "diffusers", {"MiniMaxH3AudioVAE"}));
    summary.component_classes.emplace(
        "text_encoder",
        require_component(root,
                          "text_encoder",
                          "transformers",
                          {"MiniMaxH3Qwen3VLHFEncoder"}));
    summary.component_classes.emplace(
        "tokenizer",
        require_component(
            root, "tokenizer", "transformers", {"Qwen2TokenizerFast"}));
    summary.component_classes.emplace(
        "processor",
        require_component(
            root, "processor", "transformers", {"Qwen3VLProcessor"}));
    if (!root.at("scheduler").is_null()) {
      fail("scheduler", "must be null and is not a loadable component");
    }

    return summary;
  }

  static MiniMaxH3ModelIndexSummary parse_file(const std::string& path) {
    JsonReader reader;
    if (!reader.parse(path)) {
      throw std::invalid_argument(
          "MiniMax-H3 model index validation failed: cannot parse `" + path +
          "`");
    }
    return parse(reader.data());
  }

 private:
  [[noreturn]] static void fail(const std::string& field,
                                const std::string& message) {
    throw std::invalid_argument("MiniMax-H3 model index `" + field + "` " +
                                message);
  }

  static std::string require_string(const nlohmann::json& object,
                                    const std::string& key) {
    if (!object.contains(key) || !object.at(key).is_string()) {
      fail(key, "must be a string");
    }
    return object.at(key).get<std::string>();
  }

  static std::string require_component(
      const nlohmann::json& root,
      const std::string& name,
      const std::string& expected_library,
      const std::unordered_set<std::string>& allowed_classes) {
    const auto& component = root.at(name);
    if (!component.is_array() || component.size() != 2 ||
        !component[0].is_string() || !component[1].is_string()) {
      fail(name, "must be a two-string component descriptor");
    }
    const std::string library = component[0].get<std::string>();
    const std::string class_name = component[1].get<std::string>();
    if (library != expected_library || !allowed_classes.contains(class_name)) {
      fail(name,
           "has unsupported component descriptor [" + library + ", " +
               class_name + "]");
    }
    return class_name;
  }
};

struct MiniMaxH3DryRunShapeInput {
  int64_t height = 768;
  int64_t width = 1344;
  int64_t requested_num_frames = 124;
  std::vector<int64_t> condition_shape;
  std::vector<int64_t> token_tags_shape;
};

struct MiniMaxH3DryRunShapeTrace {
  int64_t batch_size = 0;
  int64_t condition_token_count = 0;
  int64_t height = 0;
  int64_t width = 0;
  int64_t requested_num_frames = 0;
  int64_t aligned_num_frames = 0;
  double requested_duration_seconds = 0.0;
  double aligned_duration_seconds = 0.0;
  int64_t video_latent_channels = 0;
  int64_t video_latent_frames = 0;
  int64_t video_latent_height = 0;
  int64_t video_latent_width = 0;
  int64_t video_rows_per_sample = 0;
  int64_t audio_channels = 0;
  int64_t audio_latent_channels = 0;
  int64_t audio_latents_per_channel = 0;
  int64_t audio_rows_per_sample = 0;
  int64_t minimum_packed_rows_per_sample = 0;
};

inline MiniMaxH3DryRunShapeTrace minimax_h3_dry_run_shape_trace(
    const MiniMaxH3DryRunShapeInput& input) {
  constexpr int64_t kFps = 24;
  constexpr double kMinSeconds = 4.0;
  constexpr double kMaxSeconds = 15.0;
  constexpr int64_t kFramesPerChunk = 17;
  constexpr int64_t kLatentsPerChunk = 5;
  constexpr int64_t kVideoSpatialCompression = 16;
  constexpr int64_t kAudioLatentsPerSecond = 40;
  constexpr int64_t kAudioChannels = 2;

  if (input.height != 768 || input.width != 1344) {
    throw std::invalid_argument(
        "MiniMax-H3 H3-C2 dry run requires fixed 1344x768 output geometry");
  }
  if (input.condition_shape.size() != 3 || input.condition_shape[0] <= 0 ||
      input.condition_shape[1] <= 0 ||
      input.condition_shape[2] != MiniMaxH3TransformerConfig::kTextDim) {
    throw std::invalid_argument(
        "MiniMax-H3 condition must have shape [B,N,5120] with B,N > 0");
  }
  if (input.token_tags_shape.size() != 2 ||
      input.token_tags_shape[0] != input.condition_shape[0] ||
      input.token_tags_shape[1] != input.condition_shape[1]) {
    throw std::invalid_argument(
        "MiniMax-H3 token tags must have shape [B,N] matching the condition");
  }
  if (input.requested_num_frames <= 0) {
    throw std::invalid_argument(
        "MiniMax-H3 requested frame count must be positive");
  }

  const double requested_duration =
      static_cast<double>(input.requested_num_frames) / kFps;
  if (requested_duration < kMinSeconds || requested_duration > kMaxSeconds) {
    throw std::invalid_argument(
        "MiniMax-H3 requested duration must be between 4 and 15 seconds at "
        "24 fps");
  }

  int64_t aligned_frames = input.requested_num_frames;
  while (aligned_frames % kFramesPerChunk != kLatentsPerChunk) {
    ++aligned_frames;
  }
  const double aligned_duration = static_cast<double>(aligned_frames) / kFps;
  if (aligned_duration > kMaxSeconds) {
    throw std::invalid_argument(
        "MiniMax-H3 frame alignment to 17n+5 exceeds the 15 second limit");
  }

  MiniMaxH3DryRunShapeTrace trace;
  trace.batch_size = input.condition_shape[0];
  trace.condition_token_count = input.condition_shape[1];
  trace.height = input.height;
  trace.width = input.width;
  trace.requested_num_frames = input.requested_num_frames;
  trace.aligned_num_frames = aligned_frames;
  trace.requested_duration_seconds = requested_duration;
  trace.aligned_duration_seconds = aligned_duration;
  trace.video_latent_channels = MiniMaxH3TransformerConfig::kVideoLatentDim;
  trace.video_latent_frames =
      (aligned_frames - kLatentsPerChunk) / kFramesPerChunk * kLatentsPerChunk +
      2;
  trace.video_latent_height = input.height / kVideoSpatialCompression;
  trace.video_latent_width = input.width / kVideoSpatialCompression;
  trace.video_rows_per_sample = trace.video_latent_frames *
                                (trace.video_latent_height / 2) *
                                (trace.video_latent_width / 2);
  trace.audio_channels = kAudioChannels;
  trace.audio_latent_channels = MiniMaxH3TransformerConfig::kAudioLatentDim;
  trace.audio_latents_per_channel = static_cast<int64_t>(std::llround(
      aligned_duration * static_cast<double>(kAudioLatentsPerSecond)));
  trace.audio_rows_per_sample =
      trace.audio_channels * trace.audio_latents_per_channel;
  trace.minimum_packed_rows_per_sample = trace.condition_token_count +
                                         trace.video_rows_per_sample +
                                         trace.audio_rows_per_sample;
  return trace;
}

inline std::vector<torch::Tensor> minimax_h3_draw_sequential_cpu_noise(
    int64_t seed,
    const std::vector<std::vector<int64_t>>& shapes) {
  torch::Generator generator = torch::make_generator<torch::CPUGeneratorImpl>();
  generator.set_current_seed(seed);
  std::vector<torch::Tensor> values;
  values.reserve(shapes.size());
  for (const std::vector<int64_t>& shape : shapes) {
    if (shape.empty()) {
      throw std::invalid_argument("MiniMax-H3 noise shape must not be empty");
    }
    values.emplace_back(torch::randn(
        shape,
        generator,
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)));
  }
  return values;
}

inline std::string minimax_h3_tensor_sha256(const torch::Tensor& tensor) {
  torch::Tensor value = tensor.detach().to(torch::kCPU).contiguous();
  const size_t bytes = static_cast<size_t>(value.numel()) *
                       static_cast<size_t>(value.element_size());
  unsigned char digest[SHA256_DIGEST_LENGTH];
  if (SHA256(static_cast<const unsigned char*>(value.const_data_ptr()),
             bytes,
             digest) == nullptr) {
    throw std::runtime_error("MiniMax-H3 SHA256 calculation failed");
  }
  std::ostringstream encoded;
  encoded << std::hex << std::setfill('0');
  for (unsigned char byte : digest) {
    encoded << std::setw(2) << static_cast<int>(byte);
  }
  return encoded.str();
}

inline std::pair<int64_t, int64_t> minimax_h3_reference_resize_geometry(
    int64_t height,
    int64_t width) {
  constexpr int64_t kShortEdge = 2048;
  constexpr int64_t kMultiple = 32;
  if (height <= 0 || width <= 0 || width > 4 * height || height > 4 * width) {
    throw std::invalid_argument(
        "MiniMax-H3 reference image must have positive dimensions within 1:4");
  }
  const double scale =
      static_cast<double>(kShortEdge) / std::min(height, width);
  const auto round_ties_to_even = [](double value) {
    const double lower = std::floor(value);
    const double fraction = value - lower;
    if (fraction < 0.5) {
      return static_cast<int64_t>(lower);
    }
    if (fraction > 0.5) {
      return static_cast<int64_t>(lower + 1.0);
    }
    const int64_t lower_integer = static_cast<int64_t>(lower);
    return lower_integer % 2 == 0 ? lower_integer : lower_integer + 1;
  };
  const int64_t target_height = std::max<int64_t>(
      kMultiple, round_ties_to_even(height * scale / kMultiple) * kMultiple);
  const int64_t target_width = std::max<int64_t>(
      kMultiple, round_ties_to_even(width * scale / kMultiple) * kMultiple);
  return {target_height, target_width};
}

class MiniMaxH3PipelineImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3PipelineImpl(const DiTModelContext& context)
      : context_(context),
        options_(context.get_tensor_options()),
        transformer_config_(validate_context(context)) {}

  static void validate_public_request_contract(const DiTForwardInput& input) {
    validate_public_input(input);
    torch::Tensor image = select_reference_image(input);
    const auto geometry =
        minimax_h3_reference_resize_geometry(image.size(2), image.size(3));
    if (geometry.first != kReferenceHeight ||
        geometry.second != kReferenceWidth) {
      throw std::invalid_argument(
          "MiniMax-H3 public reference does not match production geometry");
    }
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    std::lock_guard<std::mutex> lock(forward_mutex_);
    using Clock = std::chrono::steady_clock;
    const auto request_started = Clock::now();
    auto stage_started = request_started;
    if (!loaded_) {
      throw std::logic_error(
          "MiniMax-H3 public runtime requires loaded component weights");
    }
    const ParallelArgs& parallel = context_.get_parallel_args();
    validate_parallel_topology(parallel);
    const torch::Device device = options_.device();
    const uint64_t request_sequence = ++request_sequence_;
    const auto log_stage_timing = [&](const char* stage) {
      const auto now = Clock::now();
      LOG(INFO) << "MINIMAX_H3_STAGE_TIMING rank=" << parallel.rank()
                << " request=" << request_sequence << " stage=" << stage
                << " elapsed_ms="
                << std::chrono::duration<double, std::milli>(now -
                                                             stage_started)
                       .count()
                << " cumulative_ms="
                << std::chrono::duration<double, std::milli>(now -
                                                             request_started)
                       .count();
      stage_started = now;
    };
    const auto log_request_total = [&] {
      const auto now = Clock::now();
      const double elapsed_ms =
          std::chrono::duration<double, std::milli>(now - request_started)
              .count();
      LOG(INFO) << "MINIMAX_H3_STAGE_TIMING rank=" << parallel.rank()
                << " request=" << request_sequence
                << " stage=request_total elapsed_ms=" << elapsed_ms
                << " cumulative_ms=" << elapsed_ms;
    };

    torch::Tensor reference_pixels;
    std::exception_ptr input_error;
    try {
      validate_public_input(input);
      reference_pixels = select_reference_image(input);
      const auto [reference_height, reference_width] =
          minimax_h3_reference_resize_geometry(reference_pixels.size(2),
                                               reference_pixels.size(3));
      if (reference_height != kReferenceHeight ||
          reference_width != kReferenceWidth) {
        throw std::invalid_argument(
            "MiniMax-H3 public runtime requires a reference that resizes to "
            "5536x2048");
      }
      reference_pixels = resize_reference_image(
          reference_pixels, reference_height, reference_width);
      reference_pixels =
          reference_pixels.unsqueeze(2).to(device, torch::kUInt8).contiguous();
    } catch (...) {
      input_error = std::current_exception();
    }
    agree_stage(parallel, input_error, "input preparation");
    log_stage_timing("input_preparation");

    torch::Tensor posterior_epsilon;
    torch::Tensor visual_latent;
    std::exception_ptr reference_error;
    try {
      torch::Generator posterior_generator =
          torch::make_generator<torch::CPUGeneratorImpl>();
      posterior_generator.set_current_seed(kPosteriorSeed);
      posterior_epsilon = torch::randn(
          {1, 24, 1, 128, 346},
          posterior_generator,
          torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
      if (!resident_video_vae_) {
        resident_video_vae_ = MiniMaxH3VideoVAE(options_);
        resident_video_vae_->load_model(*component_loader("video_vae"));
      }
      visual_latent = resident_video_vae_->encode_condition(
          reference_pixels, posterior_epsilon.to(device));
      synchronize_device();
      require_tensor(visual_latent,
                     {1, 24, 1, 128, 346},
                     torch::kFloat32,
                     device,
                     "reference visual latent");
    } catch (...) {
      reference_error = std::current_exception();
      resident_video_vae_ = nullptr;
      clear_device_cache();
    }
    agree_stage(parallel, reference_error, "reference VAE encode");
    log_stage_timing("reference_vae_load_encode");

    const H3TargetLatents target = {.audio_t = 207,
                                    .audio_channels = 2,
                                    .latent_t = 37,
                                    .latent_h = 48,
                                    .latent_w = 84};
    H3PackedLayout layout;
    torch::Tensor initial_video_rows;
    torch::Tensor initial_audio_rows;
    std::exception_ptr packing_error;
    try {
      std::vector<torch::Tensor> noise = minimax_h3_draw_sequential_cpu_noise(
          input.generation_params.seed,
          {{1, 24, 1, 128, 346}, {1, 24, 37, 48, 84}, {414, 32}});
      torch::Tensor visual_anchor =
          kVisualAnchorTimestep * visual_latent +
          (1.0F - kVisualAnchorTimestep) * noise[0].to(device);
      torch::Tensor initial_target_video = noise[1].to(device);
      initial_audio_rows = noise[2].to(device).contiguous();
      const std::vector<H3ReferenceBlock> references = {
          {.kind = H3ReferenceBlockKind::IMAGE,
           .latent_t = 1,
           .latent_h = 128,
           .latent_w = 346}};
      layout = minimax_h3_build_ref2va_packed_layout(
          input.prompt_embeds,
          input.text_token_tags,
          target,
          references,
          /*sequence_length=*/std::nullopt);
      const int64_t condition_tokens = input.prompt_embeds.size(1);
      if (layout.used_length - condition_tokens != kFixedNonTextRows) {
        throw std::logic_error(
            "MiniMax-H3 public runtime packed layout does not match the "
            "production video, audio and reference geometry");
      }
      if (layout.aligned_length < layout.used_length ||
          layout.aligned_length % kMiniMaxH3UaaSize != 0) {
        throw std::logic_error(
            "MiniMax-H3 public runtime packed layout is not divisible by the "
            "UAA degree");
      }
      LOG(INFO) << "MINIMAX_H3_GEOMETRY rank=" << parallel.rank()
                << " request=" << request_sequence
                << " condition_tokens=" << condition_tokens
                << " used_rows=" << layout.used_length
                << " aligned_rows=" << layout.aligned_length
                << " local_rows=" << layout.aligned_length / kMiniMaxH3UaaSize
                << " padding_rows="
                << layout.aligned_length - layout.used_length;
      initial_video_rows =
          torch::cat({minimax_h3_patchify_video_latent(visual_anchor),
                      minimax_h3_patchify_video_latent(initial_target_video)},
                     0)
              .contiguous();
      synchronize_device();
    } catch (...) {
      packing_error = std::current_exception();
    }
    agree_stage(parallel, packing_error, "latent packing");
    visual_latent = torch::Tensor();
    reference_pixels = torch::Tensor();
    posterior_epsilon = torch::Tensor();
    log_stage_timing("latent_packing");

    MiniMaxH3TPUAATrajectoryOutput trajectory;
    std::exception_ptr denoiser_load_error;
    try {
      if (!resident_denoiser_) {
        resident_denoiser_ =
            MiniMaxH3TPUAAResidentDenoiser(MiniMaxH3C4Config{},
                                           parallel.dit_tp_group_,
                                           parallel.dit_sp_group_,
                                           options_.dtype(torch::kBFloat16),
                                           parallel.process_group_,
                                           context_.get_dit_config(),
                                           parallel.dit_sp_q_group_,
                                           parallel.dit_sp_k_group_,
                                           parallel.dit_sp_v_group_);
        resident_denoiser_->load_source_weights(
            component_loader("transformer")->get_state_dicts());
      }
      synchronize_device();
    } catch (...) {
      denoiser_load_error = std::current_exception();
    }
    if (denoiser_load_error != nullptr) {
      resident_denoiser_ = nullptr;
      clear_device_cache();
    }
    try {
      agree_stage(parallel, denoiser_load_error, "resident denoiser load");
    } catch (...) {
      resident_denoiser_ = nullptr;
      clear_device_cache();
      throw;
    }
    log_stage_timing("denoiser_load");
    try {
      trajectory = resident_denoiser_->run_base_trajectory(
          layout, initial_video_rows, initial_audio_rows);
      synchronize_device();
    } catch (...) {
      resident_denoiser_ = nullptr;
      clear_device_cache();
      throw;
    }
    const bool cache_enabled =
        context_.get_dit_config().selected_policy == PolicyType::CacheDiT;
    const int64_t maximum_similarity_checks =
        cache_enabled ? 49 - context_.get_dit_config().cache_dit.warmup_steps
                      : 0;
    const MiniMaxH3CacheBlockPlan cache_block_plan =
        minimax_h3_cache_block_plan(
            cache_enabled ? context_.get_dit_config().cache_dit.front_blocks
                          : 1,
            cache_enabled ? context_.get_dit_config().cache_dit.back_blocks
                          : 0);
    const int64_t expected_blocks =
        49 * kMiniMaxH3ResidentBlockCount -
        trajectory.cache_hits * cache_block_plan.skipped_blocks;
    if (trajectory.transformer_forwards != 49 ||
        trajectory.dense_forwards + trajectory.cache_hits != 49 ||
        trajectory.block_forwards != expected_blocks ||
        trajectory.similarity_checks < trajectory.cache_hits ||
        trajectory.similarity_checks > maximum_similarity_checks ||
        trajectory.cache_hit_forwards.size() !=
            static_cast<size_t>(trajectory.cache_hits) ||
        (!cache_enabled &&
         (trajectory.block_forwards != 2450 || trajectory.cache_hits != 0 ||
          trajectory.dense_forwards != 49))) {
      throw std::logic_error(
          "MiniMax-H3 public runtime violated trajectory counters");
    }
    std::ostringstream cache_hit_list;
    for (size_t index = 0; index < trajectory.cache_hit_forwards.size();
         ++index) {
      if (index != 0) {
        cache_hit_list << ',';
      }
      cache_hit_list << trajectory.cache_hit_forwards[index];
    }
    LOG(INFO) << "MINIMAX_H3_CACHE_RESULT rank=" << parallel.rank()
              << " enabled=" << cache_enabled
              << " dense_forwards=" << trajectory.dense_forwards
              << " cache_hits=" << trajectory.cache_hits
              << " similarity_checks=" << trajectory.similarity_checks
              << " block_forwards=" << trajectory.block_forwards
              << " hit_executed_blocks="
              << (cache_enabled ? cache_block_plan.hit_executed_blocks : 0)
              << " hit_forwards=" << cache_hit_list.str();
    log_stage_timing("denoise");

    const torch::Tensor image_update = layout.update_mask.to(device);
    const torch::Tensor audio_update = layout.audio_update_mask.to(device);
    torch::Tensor final_video_latent = minimax_h3_unpatchify_video_tokens(
        trajectory.video_rows.index({image_update}).contiguous(),
        {.channels = MiniMaxH3TransformerConfig::kVideoLatentDim,
         .temporal = target.latent_t,
         .height = target.latent_h,
         .width = target.latent_w});
    torch::Tensor final_audio_latent = minimax_h3_unpack_audio_tokens(
        trajectory.audio_rows.index({audio_update}).contiguous(),
        target.audio_channels,
        target.audio_t);
    trajectory = {};
    initial_video_rows = torch::Tensor();
    initial_audio_rows = torch::Tensor();
    log_stage_timing("final_unpack");

    if (parallel.rank() != 0) {
      log_request_total();
      return {};
    }

    torch::Tensor video_cpu;
    try {
      if (!resident_video_vae_) {
        resident_video_vae_ = MiniMaxH3VideoVAE(options_);
        resident_video_vae_->load_model(*component_loader("video_vae"));
      }
      log_stage_timing("video_vae_load");
      torch::Tensor decoded_video =
          resident_video_vae_->decode_normalized(final_video_latent);
      synchronize_device();
      require_tensor(decoded_video,
                     {1, 3, kOutputFrames, kOutputHeight, kOutputWidth},
                     torch::kFloat32,
                     device,
                     "decoded video");
      video_cpu = decoded_video.squeeze(0)
                      .permute({1, 0, 2, 3})
                      .to(torch::kCPU)
                      .contiguous();
      decoded_video = torch::Tensor();
      final_video_latent = torch::Tensor();
    } catch (...) {
      resident_video_vae_ = nullptr;
      clear_device_cache();
      throw;
    }
    log_stage_timing("video_vae_decode_transfer");

    torch::Tensor audio_cpu;
    try {
      if (!resident_audio_vae_) {
        resident_audio_vae_ = MiniMaxH3AudioVAE(options_);
        resident_audio_vae_->load_model(*component_loader("audio_vae"));
      }
      log_stage_timing("audio_vae_load");
      torch::Tensor decoded_audio =
          resident_audio_vae_->decode_normalized(final_audio_latent);
      synchronize_device();
      require_tensor(decoded_audio,
                     {1, 2, kOutputAudioSamples},
                     torch::kFloat32,
                     device,
                     "decoded audio");
      audio_cpu = decoded_audio.squeeze(0).to(torch::kCPU).contiguous();
      decoded_audio = torch::Tensor();
      final_audio_latent = torch::Tensor();
    } catch (...) {
      resident_audio_vae_ = nullptr;
      clear_device_cache();
      throw;
    }
    log_stage_timing("audio_vae_decode_transfer");

    DiTEncodedMedia media;
    FFmpegVideoAudioEncoder encoder;
    if (!encoder.encode(
            video_cpu, audio_cpu, kOutputFps, kAudioSampleRate, media.data)) {
      throw std::runtime_error(
          "MiniMax-H3 failed to encode the paired H.264/AAC MP4 output");
    }
    media.mime_type = "video/mp4";
    media.container = "mp4";
    media.width = kOutputWidth;
    media.height = kOutputHeight;
    media.num_frames = kOutputFrames;
    media.fps = kOutputFps;
    media.audio_sample_rate = kAudioSampleRate;
    media.audio_channels = 2;
    DiTForwardOutput output;
    output.encoded_media.emplace_back(std::move(media));
    log_stage_timing("media_encode");
    log_request_total();
    return output;
  }

  static MiniMaxH3DryRunShapeTrace dry_run_shape_trace(
      const MiniMaxH3DryRunShapeInput& input) {
    return minimax_h3_dry_run_shape_trace(input);
  }

  static H3PackedLayout dry_run_packed_layout(
      const DiTForwardInput& input,
      const H3TargetLatents& target,
      const std::vector<H3ReferenceBlock>& reference_blocks,
      std::optional<int64_t> sequence_length = std::nullopt) {
    return minimax_h3_build_ref2va_packed_layout(input.prompt_embeds,
                                                 input.text_token_tags,
                                                 target,
                                                 reference_blocks,
                                                 sequence_length);
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    if (loader == nullptr) {
      throw std::invalid_argument(
          "MiniMax-H3 load_model requires a non-null loader");
    }
    if (loaded_) {
      throw std::logic_error("MiniMax-H3 pipeline is already loaded");
    }

    const auto model_index = MiniMaxH3ModelIndexContract::parse_file(
        loader->model_root_path() + "/model_index.json");

    static const std::vector<std::string> kRequiredComponents = {"transformer",
                                                                 "video_vae",
                                                                 "audio_vae",
                                                                 "text_encoder",
                                                                 "tokenizer",
                                                                 "processor"};
    for (const auto& component : kRequiredComponents) {
      if (!loader->has_component(component)) {
        throw std::invalid_argument(
            "MiniMax-H3 loader is missing required component `" + component +
            "`");
      }
    }
    if (loader->has_component("scheduler")) {
      throw std::invalid_argument(
          "MiniMax-H3 scheduler must be null, not a loadable component");
    }

    std::unordered_map<std::string, std::unique_ptr<DiTFolderLoader>>
        component_loaders;
    for (const auto& component : kRequiredComponents) {
      component_loaders.emplace(component,
                                loader->take_component_loader(component));
    }

    const auto source_layout = MiniMaxH3SourceLayoutValidator::validate(
        component_loaders.at("transformer")->get_state_dicts());

    model_index_summary_ = model_index;
    source_layout_summary_ = source_layout;
    component_loaders_ = std::move(component_loaders);
    loaded_ = true;
  }

  bool is_loaded() const { return loaded_; }

  const std::optional<MiniMaxH3ModelIndexSummary>& model_index_summary() const {
    return model_index_summary_;
  }

  const std::optional<MiniMaxH3SourceLayoutSummary>& source_layout_summary()
      const {
    return source_layout_summary_;
  }

  void load_c4_probe() {
    if (!loaded_) {
      throw std::logic_error(
          "MiniMax-H3 C4 probe requires retained transformer weights");
    }
    if (c4_harness_) {
      throw std::logic_error("MiniMax-H3 C4 probe is already loaded");
    }
    c4_harness_ = register_module("c4_harness", MiniMaxH3C4Harness(options_));
    c4_harness_->load_source_weights(
        component_loaders_.at("transformer")->get_state_dicts());
  }

  MiniMaxH3C4Trace probe_c4(const H3PackedLayout& layout,
                            const torch::Tensor& video_rows,
                            const torch::Tensor& audio_rows,
                            const torch::Tensor& timesteps,
                            const torch::Tensor& inverse_indices) const {
    if (!c4_harness_) {
      throw std::logic_error("MiniMax-H3 C4 probe has not been loaded");
    }
    return c4_harness_->forward(
        layout, video_rows, audio_rows, timesteps, inverse_indices);
  }

  MiniMaxH3C4Harness c4_harness() const { return c4_harness_; }

  void load_c5_denoiser() {
    if (!loaded_) {
      throw std::logic_error(
          "MiniMax-H3 C5 denoiser requires retained transformer weights");
    }
    if (c5_denoiser_) {
      throw std::logic_error("MiniMax-H3 C5 denoiser is already loaded");
    }
    MiniMaxH3StreamingDenoiser candidate(options_);
    candidate->load_fixed_weights(
        component_loaders_.at("transformer")->get_state_dicts());
    c5_denoiser_ = register_module("c5_denoiser", candidate);
  }

  MiniMaxH3TrajectoryOutput probe_c5_trajectory(
      const H3PackedLayout& layout,
      const torch::Tensor& initial_video_rows,
      const torch::Tensor& initial_audio_rows,
      const MiniMaxH3LayerObserver& layer_observer = nullptr,
      const MiniMaxH3StepObserver& step_observer = nullptr) {
    if (!c5_denoiser_) {
      throw std::logic_error("MiniMax-H3 C5 denoiser has not been loaded");
    }
    return c5_denoiser_->run_base_trajectory(
        component_loaders_.at("transformer")->get_state_dicts(),
        layout,
        initial_video_rows,
        initial_audio_rows,
        layer_observer,
        step_observer);
  }

  MiniMaxH3DenoiserOutput probe_c5_forward(
      const H3PackedLayout& layout,
      const torch::Tensor& video_rows,
      const torch::Tensor& audio_rows,
      const torch::Tensor& timesteps,
      const torch::Tensor& inverse_indices,
      int64_t step,
      const MiniMaxH3LayerObserver& layer_observer = nullptr) {
    if (!c5_denoiser_) {
      throw std::logic_error("MiniMax-H3 C5 denoiser has not been loaded");
    }
    return c5_denoiser_->forward(
        component_loaders_.at("transformer")->get_state_dicts(),
        layout,
        video_rows,
        audio_rows,
        timesteps,
        inverse_indices,
        step,
        layer_observer,
        /*retain_preparation=*/true);
  }

  MiniMaxH3ResidualBranchTrace probe_c5_streaming_block(
      int64_t layer,
      const torch::Tensor& hidden,
      const torch::Tensor& time_embedding,
      const torch::Tensor& combined_indices,
      const torch::Tensor& rope_frequencies,
      const torch::Tensor& cu_seqlens) {
    if (!c5_denoiser_) {
      throw std::logic_error("MiniMax-H3 C5 denoiser has not been loaded");
    }
    return c5_denoiser_->probe_streaming_block(
        component_loaders_.at("transformer")->get_state_dicts(),
        layer,
        hidden,
        time_embedding,
        combined_indices,
        rope_frequencies,
        cu_seqlens);
  }

  MiniMaxH3StreamingDenoiser c5_denoiser() const { return c5_denoiser_; }

  void load_c6a_video_vae() {
    if (!loaded_) {
      throw std::logic_error(
          "MiniMax-H3 C6a video VAE requires retained component weights");
    }
    if (c6a_video_vae_) {
      throw std::logic_error("MiniMax-H3 C6a video VAE is already loaded");
    }
    auto loader = component_loaders_.find("video_vae");
    if (loader == component_loaders_.end() || loader->second == nullptr) {
      throw std::logic_error(
          "MiniMax-H3 C6a video VAE component loader is unavailable");
    }
    MiniMaxH3VideoVAE candidate(options_);
    candidate->load_model(*loader->second);
    c6a_video_vae_ = register_module("c6a_video_vae", candidate);
    component_loaders_.erase(loader);
  }

  MiniMaxH3VideoVAE c6a_video_vae() const { return c6a_video_vae_; }

  void load_c6b_audio_vae() {
    if (!loaded_) {
      throw std::logic_error(
          "MiniMax-H3 C6b audio VAE requires retained component weights");
    }
    if (c6b_audio_vae_) {
      throw std::logic_error("MiniMax-H3 C6b audio VAE is already loaded");
    }
    auto loader = component_loaders_.find("audio_vae");
    if (loader == component_loaders_.end() || loader->second == nullptr) {
      throw std::logic_error(
          "MiniMax-H3 C6b audio VAE component loader is unavailable");
    }
    MiniMaxH3AudioVAE candidate(options_);
    candidate->load_model(*loader->second);
    c6b_audio_vae_ = register_module("c6b_audio_vae", candidate);
    component_loaders_.erase(loader);
  }

  MiniMaxH3AudioVAE c6b_audio_vae() const { return c6b_audio_vae_; }

  void load_c7_eager() {
    if (!loaded_) {
      throw std::logic_error(
          "MiniMax-H3 C7 eager path requires retained component weights");
    }
    if (c5_denoiser_ && c6a_video_vae_ && c6b_audio_vae_) {
      throw std::logic_error("MiniMax-H3 C7 eager path is already loaded");
    }
    if (!c5_denoiser_) {
      load_c5_denoiser();
    }
    if (!c6a_video_vae_) {
      load_c6a_video_vae();
    }
    if (!c6b_audio_vae_) {
      load_c6b_audio_vae();
    }
  }

  MiniMaxH3Ref2VAEagerOutput run_c7_eager(
      const MiniMaxH3Ref2VAEagerInput& input) {
    if (!c5_denoiser_ || !c6a_video_vae_ || !c6b_audio_vae_) {
      throw std::logic_error(
          "MiniMax-H3 C7 eager path has not been fully loaded");
    }
    return minimax_h3_run_ref2va_eager(
        input,
        component_loaders_.at("transformer")->get_state_dicts(),
        c5_denoiser_,
        c6a_video_vae_,
        c6b_audio_vae_);
  }

 private:
  static constexpr int64_t kReferenceHeight = 2048;
  static constexpr int64_t kReferenceWidth = 5536;
  static constexpr int64_t kFixedNonTextRows = 48782;
  static constexpr int32_t kOutputWidth = 1344;
  static constexpr int32_t kOutputHeight = 768;
  static constexpr int32_t kOutputFrames = 124;
  static constexpr double kOutputFps = 24.0;
  static constexpr int32_t kAudioSampleRate = 32000;
  static constexpr int64_t kOutputAudioSamples = 165600;
  static constexpr int64_t kPosteriorSeed = 42;
  static constexpr float kVisualAnchorTimestep = 0.999F;

  static bool has_sha256(const nlohmann::json& value, const std::string& key) {
    const auto iterator = value.find(key);
    if (iterator == value.end() || !iterator->is_string()) {
      return false;
    }
    const std::string digest = *iterator;
    return digest.size() == 64 &&
           std::all_of(digest.begin(), digest.end(), [](unsigned char byte) {
             return (byte >= '0' && byte <= '9') ||
                    (byte >= 'a' && byte <= 'f');
           });
  }

  static void validate_public_input(const DiTForwardInput& input) {
    if (input.batch_size != 1 || input.prompt_embeds.dim() != 3 ||
        input.prompt_embeds.size(0) != 1 || input.prompt_embeds.size(1) <= 0 ||
        input.prompt_embeds.size(2) != MiniMaxH3TransformerConfig::kTextDim ||
        input.prompt_embeds.scalar_type() != torch::kBFloat16 ||
        !input.prompt_embeds.is_contiguous() ||
        !torch::isfinite(input.prompt_embeds).all().item<bool>()) {
      throw std::invalid_argument(
          "MiniMax-H3 public runtime requires one contiguous BF16 "
          "condition [1,token_count,5120] with a positive token_count");
    }
    const int64_t condition_tokens = input.prompt_embeds.size(1);
    if (!input.text_token_tags.defined() || input.text_token_tags.dim() != 2 ||
        input.text_token_tags.sizes().vec() !=
            std::vector<int64_t>{1, condition_tokens} ||
        input.text_token_tags.scalar_type() != torch::kInt64 ||
        !input.text_token_tags.is_contiguous() ||
        !torch::logical_or(input.text_token_tags == 0,
                           input.text_token_tags == 1)
             .all()
             .item<bool>()) {
      throw std::invalid_argument(
          "MiniMax-H3 public runtime requires int64 binary token tags "
          "[1,token_count] matching the condition token_count");
    }
    if (input.condition_schemas.size() != 1 ||
        input.condition_source_backends.size() != 1 ||
        input.condition_manifest_jsons.size() != 1 ||
        input.condition_schemas[0] != kMiniMaxH3ConditionSchemaV1 ||
        (input.condition_source_backends[0] != "official_hf" &&
         input.condition_source_backends[0] != "xllm_native")) {
      throw std::invalid_argument(
          "MiniMax-H3 public runtime requires one complete supported condition "
          "bundle");
    }
    const nlohmann::json manifest =
        nlohmann::json::parse(input.condition_manifest_jsons[0],
                              /*cb=*/nullptr,
                              /*allow_exceptions=*/false);
    const auto string_equals = [&](const std::string& key,
                                   const std::string& expected) {
      const auto iterator = manifest.find(key);
      return iterator != manifest.end() && iterator->is_string() &&
             *iterator == expected;
    };
    const auto integer_equals = [&](const std::string& key, int64_t expected) {
      const auto iterator = manifest.find(key);
      return iterator != manifest.end() && iterator->is_number_integer() &&
             *iterator == expected;
    };
    if (manifest.is_discarded() || !manifest.is_object() ||
        !string_equals("schema", input.condition_schemas[0]) ||
        !string_equals("source_backend", input.condition_source_backends[0]) ||
        !integer_equals("decoder_layer_index", 49) ||
        !integer_equals("hidden_state_slot", 50) ||
        !integer_equals("token_count", input.prompt_embeds.size(1)) ||
        !has_sha256(manifest, "hidden_digest") ||
        !has_sha256(manifest, "token_tags_digest") ||
        !has_sha256(manifest, "condition_cache_key") ||
        !has_sha256(manifest, "reference_pixels_digest") ||
        manifest.at("hidden_digest").get<std::string>() !=
            minimax_h3_tensor_sha256(input.prompt_embeds.squeeze(0)) ||
        manifest.at("token_tags_digest").get<std::string>() !=
            minimax_h3_tensor_sha256(input.text_token_tags.squeeze(0))) {
      throw std::invalid_argument(
          "MiniMax-H3 public condition manifest does not match its tensors");
    }
    const DiTGenerationParams& params = input.generation_params;
    if (params.width != kOutputWidth || params.height != kOutputHeight ||
        params.num_frames != kOutputFrames || params.video_fps != kOutputFps ||
        params.num_inference_steps != MiniMaxH3Scheduler::kBasePointCount ||
        params.num_images_per_prompt != 1 ||
        params.num_videos_per_prompt != 1 || !params.force_video_output ||
        !params.seed_is_set) {
      throw std::invalid_argument(
          "MiniMax-H3 public runtime requires fixed 1344x768, 124-frame, "
          "24-fps, 50-point Base generation with one seeded output");
    }
    if (input.latents.defined() || input.last_images.defined() ||
        input.mask_images.defined() || input.control_image.defined() ||
        input.masked_image_latents.defined() ||
        input.negative_prompt_embeds.defined() ||
        input.negative_pooled_prompt_embeds.defined() ||
        !input.negative_prompts.empty() || !input.negative_prompts_2.empty()) {
      throw std::invalid_argument(
          "MiniMax-H3 public runtime does not accept target latents, last "
          "images, masks, controls, or negative conditioning");
    }
  }

  static torch::Tensor select_reference_image(const DiTForwardInput& input) {
    torch::Tensor image;
    if (input.images.defined()) {
      image = input.images;
    }
    if (!input.images_list.empty()) {
      if (input.images_list.size() != 1 || !input.images_list[0].defined()) {
        throw std::invalid_argument(
            "MiniMax-H3 public runtime requires exactly one image reference");
      }
      if (image.defined() &&
          (image.sizes() != input.images_list[0].sizes() ||
           image.scalar_type() != input.images_list[0].scalar_type() ||
           !torch::equal(image, input.images_list[0]))) {
        throw std::invalid_argument(
            "MiniMax-H3 images and images_list references differ");
      }
      image = input.images_list[0];
    }
    if (!image.defined() || image.dim() != 4 || image.size(0) != 1 ||
        image.size(1) != 3 || image.scalar_type() != torch::kUInt8) {
      throw std::invalid_argument(
          "MiniMax-H3 public runtime requires one uint8 RGB image [1,3,H,W]");
    }
    image = image.to(torch::kCPU).contiguous();
    const nlohmann::json manifest =
        nlohmann::json::parse(input.condition_manifest_jsons[0]);
    if (manifest.at("reference_pixels_digest").get<std::string>() !=
        minimax_h3_tensor_sha256(image)) {
      throw std::invalid_argument(
          "MiniMax-H3 condition manifest reference digest does not match the "
          "supplied image");
    }
    return image;
  }

  static torch::Tensor resize_reference_image(const torch::Tensor& image,
                                              int64_t height,
                                              int64_t width) {
    if (!image.device().is_cpu() || image.scalar_type() != torch::kUInt8 ||
        image.dim() != 4 || image.size(0) != 1 || image.size(1) != 3 ||
        height > std::numeric_limits<int32_t>::max() ||
        width > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument(
          "MiniMax-H3 reference resize received an invalid image");
    }
    torch::Tensor source = image.squeeze(0).permute({1, 2, 0}).contiguous();
    torch::Tensor resized = torch::empty({height, width, 3}, torch::kUInt8);
    lanczos::resize_8bpc(source.data_ptr<uint8_t>(),
                         static_cast<int32_t>(source.size(1)),
                         static_cast<int32_t>(source.size(0)),
                         /*channels=*/3,
                         static_cast<int32_t>(width),
                         static_cast<int32_t>(height),
                         resized.data_ptr<uint8_t>());
    return resized.permute({2, 0, 1}).unsqueeze(0).contiguous();
  }

  static void validate_parallel_topology(const ParallelArgs& parallel) {
    if (parallel.process_group_ == nullptr ||
        parallel.process_group_->world_size() != kMiniMaxH3TPUAAWorldSize ||
        parallel.world_size() != kMiniMaxH3TPUAAWorldSize ||
        parallel.dp_size() != 1 || parallel.tp_size() != kMiniMaxH3TPSize ||
        parallel.sp_size() != kMiniMaxH3UaaSize || parallel.cfg_size() != 1 ||
        parallel.vae_size() != 1 || parallel.text_encoder_tp_size() != 1) {
      throw std::invalid_argument(
          "MiniMax-H3 public runtime requires world16 TP2 x U8 groups");
    }
    if (parallel.dit_communication_domains_ == nullptr) {
      throw std::invalid_argument(
          "MiniMax-H3 public runtime requires communication-domain metadata");
    }

    const auto require_domain = [&](const std::string& name,
                                    int32_t expected_size,
                                    ProcessGroup* expected_group) {
      const CommunicationDomain& domain =
          parallel.dit_communication_domains_->require(name);
      if (domain.size() != expected_size || domain.process_group() == nullptr ||
          domain.process_group() != expected_group ||
          domain.local_rank() != expected_group->rank() ||
          domain.ranks().at(static_cast<size_t>(domain.local_rank())) !=
              parallel.rank()) {
        throw std::invalid_argument("MiniMax-H3 communication domain `" + name +
                                    "` is invalid");
      }
    };
    require_domain("tp", kMiniMaxH3TPSize, parallel.dit_tp_group_);
    require_domain("sp", kMiniMaxH3UaaSize, parallel.dit_sp_group_);
    require_domain("sp_q", kMiniMaxH3UaaSize, parallel.dit_sp_q_group_);
    require_domain("sp_k", kMiniMaxH3UaaSize, parallel.dit_sp_k_group_);
    require_domain("sp_v", kMiniMaxH3UaaSize, parallel.dit_sp_v_group_);
    require_domain("cfg", 1, parallel.dit_cfg_group_);
    require_domain("dp", 1, parallel.dit_dp_group_);
    require_domain("vae", 1, parallel.dit_vae_group_);
    require_domain("text_encoder_tp", 1, parallel.dit_text_encoder_tp_group_);
  }

  void agree_stage(const ParallelArgs& parallel,
                   const std::exception_ptr& local_error,
                   const std::string& stage) const {
    torch::Tensor failure = torch::tensor(
        {local_error == nullptr ? 0 : 1},
        torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));
    parallel.dit_tp_group_->allreduce(failure);
    parallel.dit_sp_group_->allreduce(failure);
    if (failure.item<int32_t>() == 0) {
      return;
    }
    if (local_error != nullptr) {
      std::rethrow_exception(local_error);
    }
    throw std::runtime_error("MiniMax-H3 " + stage + " failed on another rank");
  }

  static void require_tensor(const torch::Tensor& tensor,
                             const std::vector<int64_t>& shape,
                             torch::ScalarType dtype,
                             const torch::Device& device,
                             const std::string& name) {
    if (!tensor.defined() || tensor.sizes().vec() != shape ||
        tensor.scalar_type() != dtype || tensor.device() != device ||
        !tensor.is_contiguous() ||
        !torch::isfinite(tensor).all().item<bool>()) {
      throw std::runtime_error("MiniMax-H3 " + name +
                               " violated its production tensor contract");
    }
  }

  DiTFolderLoader* component_loader(const std::string& name) const {
    const auto iterator = component_loaders_.find(name);
    if (iterator == component_loaders_.end() || iterator->second == nullptr) {
      throw std::logic_error("MiniMax-H3 component loader `" + name +
                             "` is unavailable");
    }
    return iterator->second.get();
  }

  void synchronize_device() const {
    Device device(options_.device());
    if (device.synchronize_default_stream() != 0) {
      throw std::runtime_error("MiniMax-H3 NPU synchronization failed");
    }
  }

  void clear_device_cache() const {
    Device::empty_cache(options_.device().index());
  }

  static MiniMaxH3TransformerConfig validate_context(
      const DiTModelContext& context) {
    for (const auto& component :
         {"transformer", "video_vae", "audio_vae", "text_encoder"}) {
      if (!context.has_component(component)) {
        throw std::invalid_argument(
            "MiniMax-H3 context is missing ModelArgs for component `" +
            std::string(component) + "`");
      }
    }

    const auto config = MiniMaxH3TransformerConfig::from_model_args(
        context.get_model_args("transformer"));
    if (context.get_model_args("video_vae").h3_video_latent_dim() !=
        MiniMaxH3TransformerConfig::kVideoLatentDim) {
      throw std::invalid_argument(
          "MiniMax-H3 video VAE and transformer latent dimensions differ");
    }
    if (context.get_model_args("audio_vae").h3_audio_latent_dim() !=
        MiniMaxH3TransformerConfig::kAudioLatentDim) {
      throw std::invalid_argument(
          "MiniMax-H3 audio VAE and transformer latent dimensions differ");
    }
    if (context.get_model_args("text_encoder").h3_text_dim() !=
        MiniMaxH3TransformerConfig::kTextDim) {
      throw std::invalid_argument(
          "MiniMax-H3 text encoder and transformer condition dimensions "
          "differ");
    }
    return config;
  }

  DiTModelContext context_;
  torch::TensorOptions options_;
  MiniMaxH3TransformerConfig transformer_config_;
  std::unordered_map<std::string, std::unique_ptr<DiTFolderLoader>>
      component_loaders_;
  std::optional<MiniMaxH3ModelIndexSummary> model_index_summary_;
  std::optional<MiniMaxH3SourceLayoutSummary> source_layout_summary_;
  MiniMaxH3C4Harness c4_harness_{nullptr};
  MiniMaxH3StreamingDenoiser c5_denoiser_{nullptr};
  MiniMaxH3VideoVAE c6a_video_vae_{nullptr};
  MiniMaxH3AudioVAE c6b_audio_vae_{nullptr};
  MiniMaxH3TPUAAResidentDenoiser resident_denoiser_{nullptr};
  MiniMaxH3VideoVAE resident_video_vae_{nullptr};
  MiniMaxH3AudioVAE resident_audio_vae_{nullptr};
  std::mutex forward_mutex_;
  uint64_t request_sequence_ = 0;
  bool loaded_ = false;
};
TORCH_MODULE(MiniMaxH3Pipeline);

#if !defined(XLLM_MINIMAX_H3_DISABLE_REGISTRATION)
REGISTER_DIT_MODEL(MiniMaxH3Pipeline, MiniMaxH3Pipeline);
#endif

}  // namespace xllm
