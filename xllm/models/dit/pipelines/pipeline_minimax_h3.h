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

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/dit_model_context.h"
#include "core/framework/dit_model_loader.h"
#include "core/runtime/dit_forward_params.h"
#include "core/util/json_reader.h"
#include "models/dit/transformers/minimax_h3_denoiser.h"
#include "models/dit/transformers/transformer_minimax_h3.h"
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

class MiniMaxH3PipelineImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3PipelineImpl(const DiTModelContext& context)
      : context_(context),
        options_(context.get_tensor_options()),
        transformer_config_(validate_context(context)) {}

  DiTForwardOutput forward(const DiTForwardInput& input) {
    (void)input;
    throw_forward_unavailable();
  }

  [[noreturn]] static void throw_forward_unavailable() {
    throw std::logic_error(
        "MiniMax-H3 H3-C5 pipeline has a full streaming denoiser but no "
        "VAE-backed production generation path");
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
        layer_observer);
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

 private:
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
  bool loaded_ = false;
};
TORCH_MODULE(MiniMaxH3Pipeline);

#if !defined(XLLM_MINIMAX_H3_DISABLE_REGISTRATION)
REGISTER_DIT_MODEL(MiniMaxH3Pipeline, MiniMaxH3Pipeline);
#endif

}  // namespace xllm
