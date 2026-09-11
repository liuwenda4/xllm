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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "models/dit/schedulers/minimax_h3_scheduler.h"
#include "models/dit/transformers/transformer_minimax_h3.h"

namespace xllm {

struct MiniMaxH3RowTimestepPlan {
  torch::Tensor row_timesteps;
  torch::Tensor unique_timesteps;
  torch::Tensor inverse_indices;
};

struct MiniMaxH3DenoiserOutput {
  torch::Tensor condition_projection;
  torch::Tensor refined_condition;
  torch::Tensor video_embedding;
  torch::Tensor audio_embedding;
  torch::Tensor packed_hidden;
  torch::Tensor time_embedding;
  torch::Tensor rope_frequencies;
  torch::Tensor combined_indices;
  torch::Tensor final_activation;
  torch::Tensor video_velocity;
  torch::Tensor audio_velocity;
  int64_t executed_layers = 0;
};

struct MiniMaxH3TrajectoryStep {
  int64_t step = 0;
  MiniMaxH3RowTimestepPlan timestep_plan;
  MiniMaxH3DenoiserOutput denoiser;
  torch::Tensor video_x0;
  torch::Tensor audio_x0;
  torch::Tensor video_rows_after;
  torch::Tensor audio_rows_after;
};

struct MiniMaxH3TrajectoryOutput {
  torch::Tensor video_rows;
  torch::Tensor audio_rows;
  int64_t transformer_forwards = 0;
  int64_t block_forwards = 0;
};

using MiniMaxH3LayerObserver =
    std::function<void(int64_t step,
                       int64_t layer,
                       const MiniMaxH3ResidualBranchTrace& used_trace)>;
using MiniMaxH3StepObserver =
    std::function<void(const MiniMaxH3TrajectoryStep& trace)>;

inline MiniMaxH3RowTimestepPlan minimax_h3_build_row_timestep_plan(
    const H3PackedLayout& layout,
    float video_timestep,
    float audio_timestep) {
  if (!std::isfinite(video_timestep) || !std::isfinite(audio_timestep) ||
      video_timestep < 0.0F || video_timestep > 1.0F || audio_timestep < 0.0F ||
      audio_timestep > 1.0F) {
    throw std::invalid_argument(
        "MiniMax-H3 row timesteps must be finite and in [0,1]");
  }
  if (layout.aligned_length <= 0 || layout.used_length <= 0 ||
      layout.aligned_length < layout.used_length ||
      layout.img_pos.scalar_type() != torch::kInt64 ||
      layout.audio_pos.scalar_type() != torch::kInt64 ||
      layout.update_mask.scalar_type() != torch::kBool ||
      layout.audio_update_mask.scalar_type() != torch::kBool ||
      layout.img_pos.numel() != layout.update_mask.numel() ||
      layout.audio_pos.numel() != layout.audio_update_mask.numel()) {
    throw std::invalid_argument(
        "MiniMax-H3 row timestep layout metadata is invalid");
  }

  std::vector<float> rows(static_cast<size_t>(layout.aligned_length),
                          video_timestep);
  const torch::Tensor image_positions =
      layout.img_pos.to(torch::kCPU).contiguous();
  const torch::Tensor audio_positions =
      layout.audio_pos.to(torch::kCPU).contiguous();
  const torch::Tensor image_update =
      layout.update_mask.to(torch::kCPU).contiguous();
  const torch::Tensor audio_update =
      layout.audio_update_mask.to(torch::kCPU).contiguous();
  const int64_t* image_position_values =
      image_positions.const_data_ptr<int64_t>();
  const int64_t* audio_position_values =
      audio_positions.const_data_ptr<int64_t>();
  const bool* image_update_values = image_update.const_data_ptr<bool>();
  const bool* audio_update_values = audio_update.const_data_ptr<bool>();
  for (int64_t index = 0; index < image_positions.numel(); ++index) {
    const int64_t position = image_position_values[index];
    if (position < 0 || position >= layout.used_length) {
      throw std::invalid_argument(
          "MiniMax-H3 image position is outside used rows");
    }
    rows[static_cast<size_t>(position)] =
        image_update_values[index] ? video_timestep
                                   : std::max(video_timestep, 0.999F);
  }
  for (int64_t index = 0; index < audio_positions.numel(); ++index) {
    const int64_t position = audio_position_values[index];
    if (position < 0 || position >= layout.used_length) {
      throw std::invalid_argument(
          "MiniMax-H3 audio position is outside used rows");
    }
    rows[static_cast<size_t>(position)] = audio_update_values[index]
                                              ? audio_timestep
                                              : std::max(audio_timestep, 1.0F);
  }

  std::vector<float> unique = rows;
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  std::vector<int64_t> inverse(rows.size());
  for (size_t index = 0; index < rows.size(); ++index) {
    inverse[index] = static_cast<int64_t>(
        std::lower_bound(unique.begin(), unique.end(), rows[index]) -
        unique.begin());
  }
  return {
      .row_timesteps =
          torch::from_blob(
              rows.data(), {static_cast<int64_t>(rows.size())}, torch::kFloat32)
              .clone(),
      .unique_timesteps =
          torch::from_blob(unique.data(),
                           {static_cast<int64_t>(unique.size())},
                           torch::kFloat32)
              .clone(),
      .inverse_indices =
          torch::from_blob(inverse.data(),
                           {static_cast<int64_t>(inverse.size())},
                           torch::kInt64)
              .clone(),
  };
}

class MiniMaxH3StreamingDenoiserImpl final : public torch::nn::Module {
 public:
  static constexpr size_t kFixedTensorCount = 35;
  static constexpr size_t kBlockTensorCount = 10;

  explicit MiniMaxH3StreamingDenoiserImpl(const torch::TensorOptions& options)
      : options_(options) {
    const MiniMaxH3C4Config config;
    const torch::TensorOptions fp32 = options.dtype(torch::kFloat32);
    const torch::TensorOptions bf16 = options.dtype(torch::kBFloat16);
    video_patch_proj_ = register_module("video_patch_proj",
                                        MiniMaxH3Dense(config.video_patch_dim,
                                                       config.hidden_size,
                                                       /*with_bias=*/true,
                                                       fp32));
    audio_patch_proj_ = register_module("audio_patch_proj",
                                        MiniMaxH3Dense(config.audio_dim,
                                                       config.hidden_size,
                                                       /*with_bias=*/true,
                                                       fp32));
    condition_proj_ = register_module("condition_proj",
                                      MiniMaxH3Dense(config.text_dim,
                                                     config.hidden_size,
                                                     /*with_bias=*/true,
                                                     bf16));
    time_embedder_ = register_module("time_embedder",
                                     MiniMaxH3TimeEmbedder(config, options));
    rope_ = register_module("rope",
                            MiniMaxH3Rope(config.rope_inv_freq_len, options));
    token_refiner_ = register_module("token_refiner",
                                     MiniMaxH3TokenRefiner(config, options));
    streaming_block_ =
        register_module("streaming_block", MiniMaxH3DiTBlock(config, options));
    final_layer_ =
        register_module("final_layer", MiniMaxH3FinalLayer(config, options));
  }

  void load_fixed_weights(
      const std::vector<std::unique_ptr<StateDict>>& shards) {
    if (fixed_loaded_) {
      throw std::logic_error(
          "MiniMax-H3 streaming denoiser fixed weights are already loaded");
    }
    MiniMaxH3SourceLayoutValidator::validate(shards);
    const SourceIndex source = build_source_index(shards);
    auto parameters = named_parameters(/*recurse=*/true);
    auto buffers = named_buffers(/*recurse=*/true);
    torch::NoGradGuard no_grad;
    for (auto& parameter : parameters) {
      if (!parameter.key().starts_with("streaming_block.")) {
        copy_fixed_tensor(parameter.key(), parameter.value(), source);
      }
    }
    for (auto& buffer : buffers) {
      if (!buffer.key().starts_with("streaming_block.")) {
        copy_fixed_tensor(buffer.key(), buffer.value(), source);
      }
    }
    if (loaded_fixed_names_.size() != kFixedTensorCount) {
      throw std::logic_error(
          "MiniMax-H3 streaming denoiser fixed tensor count mismatch");
    }
    fixed_loaded_ = true;
  }

  MiniMaxH3DenoiserOutput forward(
      const std::vector<std::unique_ptr<StateDict>>& shards,
      const H3PackedLayout& layout,
      const torch::Tensor& video_rows,
      const torch::Tensor& audio_rows,
      const torch::Tensor& timesteps,
      const torch::Tensor& inverse_indices,
      int64_t step,
      const MiniMaxH3LayerObserver& observer = nullptr) {
    verify_fixed_weights();
    validate_inputs(layout, video_rows, audio_rows, timesteps, inverse_indices);
    const SourceIndex source = build_source_index(shards);
    const torch::Device device = condition_proj_->weight().device();
    const MiniMaxH3C4Config config;

    torch::Tensor condition = layout.condition_hidden.squeeze(0).to(device);
    torch::Tensor projected_condition = condition_proj_->forward(condition);
    torch::Tensor refiner_cu = torch::tensor(
        std::vector<int64_t>{0, condition.size(0)},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto [refined_condition, ignored_refiner_trace] =
        token_refiner_->forward(projected_condition, refiner_cu);
    (void)ignored_refiner_trace;
    torch::Tensor video_embedding =
        video_patch_proj_->forward(video_rows.to(device));
    torch::Tensor audio_embedding =
        audio_patch_proj_->forward(audio_rows.to(device));
    torch::Tensor hidden =
        torch::zeros({layout.aligned_length, config.hidden_size},
                     projected_condition.options());
    hidden.index_copy_(0, layout.text_pos.to(device), refined_condition);
    hidden.index_copy_(
        0, layout.img_pos.to(device), video_embedding.to(torch::kBFloat16));
    hidden.index_copy_(
        0, layout.audio_pos.to(device), audio_embedding.to(torch::kBFloat16));
    torch::Tensor time_embedding =
        time_embedder_->forward(timesteps.to(device));
    torch::Tensor rope_frequencies =
        rope_->forward(layout.position_ids.to(device));
    torch::Tensor device_inverse = inverse_indices.to(device);
    torch::Tensor combined_indices = minimax_h3_combined_adaln_indices(
        device_inverse, layout.token_tags.to(device));
    MiniMaxH3DenoiserOutput output{
        .condition_projection = projected_condition,
        .refined_condition = refined_condition,
        .video_embedding = video_embedding,
        .audio_embedding = audio_embedding,
        .packed_hidden = hidden.slice(0, 0, layout.used_length),
        .time_embedding = time_embedding,
        .rope_frequencies = rope_frequencies.slice(0, 0, layout.used_length),
        .combined_indices = combined_indices.slice(0, 0, layout.used_length),
    };
    hidden = hidden.slice(0, 0, layout.used_length);
    combined_indices = combined_indices.slice(0, 0, layout.used_length);
    rope_frequencies = rope_frequencies.slice(0, 0, layout.used_length);
    device_inverse = device_inverse.slice(0, 0, layout.used_length);
    const torch::Tensor used_cu_seqlens = torch::tensor(
        {0, static_cast<int32_t>(layout.used_length)}, torch::kInt32);
    for (int64_t layer = 0; layer < MiniMaxH3TransformerConfig::kNumLayers;
         ++layer) {
      load_block_weights(source, layer);
      MiniMaxH3ResidualBranchTrace block_trace =
          streaming_block_->forward(hidden,
                                    time_embedding,
                                    combined_indices,
                                    rope_frequencies,
                                    used_cu_seqlens);
      hidden = block_trace.output;
      if (!torch::isfinite(hidden).all().item<bool>()) {
        throw std::runtime_error("MiniMax-H3 block " + std::to_string(layer) +
                                 " produced NaN or Inf");
      }
      if (observer) {
        observer(
            step,
            layer,
            {.adaln_parameters = block_trace.adaln_parameters,
             .shift_msa = block_trace.shift_msa.slice(0, 0, layout.used_length),
             .scale_msa = block_trace.scale_msa.slice(0, 0, layout.used_length),
             .gate_msa = block_trace.gate_msa.slice(0, 0, layout.used_length),
             .shift_mlp = block_trace.shift_mlp.slice(0, 0, layout.used_length),
             .scale_mlp = block_trace.scale_mlp.slice(0, 0, layout.used_length),
             .gate_mlp = block_trace.gate_mlp.slice(0, 0, layout.used_length),
             .norm1_output =
                 block_trace.norm1_output.slice(0, 0, layout.used_length),
             .attention_input =
                 block_trace.attention_input.slice(0, 0, layout.used_length),
             .attention_output =
                 block_trace.attention_output.slice(0, 0, layout.used_length),
             .attention_delta =
                 block_trace.attention_delta.slice(0, 0, layout.used_length),
             .norm2_output =
                 block_trace.norm2_output.slice(0, 0, layout.used_length),
             .mlp_input = block_trace.mlp_input.slice(0, 0, layout.used_length),
             .mlp_output =
                 block_trace.mlp_output.slice(0, 0, layout.used_length),
             .mlp_delta = block_trace.mlp_delta.slice(0, 0, layout.used_length),
             .output = hidden.slice(0, 0, layout.used_length)});
      }
    }
    MiniMaxH3FinalOutput final =
        final_layer_->forward(hidden, time_embedding, device_inverse, layout);
    output.final_activation = final.activation.slice(0, 0, layout.used_length);
    output.video_velocity = final.raw_selected_video_logits;
    output.audio_velocity = final.raw_selected_audio_logits;
    output.executed_layers = MiniMaxH3TransformerConfig::kNumLayers;
    return output;
  }

  MiniMaxH3TrajectoryOutput run_base_trajectory(
      const std::vector<std::unique_ptr<StateDict>>& shards,
      const H3PackedLayout& layout,
      const torch::Tensor& initial_video_rows,
      const torch::Tensor& initial_audio_rows,
      const MiniMaxH3LayerObserver& layer_observer = nullptr,
      const MiniMaxH3StepObserver& step_observer = nullptr) {
    torch::NoGradGuard no_grad;
    const MiniMaxH3DualSigmaSchedule schedule =
        MiniMaxH3Scheduler::build_base();
    if (schedule.video.forward_count() != schedule.audio.forward_count() ||
        schedule.video.forward_count() != 49) {
      throw std::logic_error(
          "MiniMax-H3 Base dual schedule must have 49 forwards");
    }
    const torch::Device device = condition_proj_->weight().device();
    const torch::Tensor image_update = layout.update_mask.to(device);
    const torch::Tensor audio_update = layout.audio_update_mask.to(device);
    torch::Tensor video_rows = initial_video_rows.to(device).clone();
    torch::Tensor audio_rows = initial_audio_rows.to(device).clone();
    validate_latent_rows(layout, video_rows, audio_rows);
    const torch::Tensor video_anchor =
        video_rows.index({~image_update}).clone();
    const torch::Tensor audio_anchor =
        audio_rows.index({~audio_update}).clone();

    for (int64_t step = 0; step < schedule.video.forward_count(); ++step) {
      const float video_timestep = schedule.video.timesteps[step].item<float>();
      const float audio_timestep = schedule.audio.timesteps[step].item<float>();
      MiniMaxH3RowTimestepPlan plan = minimax_h3_build_row_timestep_plan(
          layout, video_timestep, audio_timestep);
      MiniMaxH3DenoiserOutput denoiser = forward(shards,
                                                 layout,
                                                 video_rows,
                                                 audio_rows,
                                                 plan.unique_timesteps,
                                                 plan.inverse_indices,
                                                 step,
                                                 layer_observer);
      torch::Tensor video_target = video_rows.index({image_update});
      torch::Tensor audio_target = audio_rows.index({audio_update});
      torch::Tensor video_velocity =
          denoiser.video_velocity.index({image_update});
      torch::Tensor audio_velocity =
          denoiser.audio_velocity.index({audio_update});
      torch::Tensor video_x0 = MiniMaxH3Scheduler::velocity_to_x0(
          video_target, video_velocity, schedule.video.timesteps[step]);
      torch::Tensor audio_x0 = MiniMaxH3Scheduler::velocity_to_x0(
          audio_target, audio_velocity, schedule.audio.timesteps[step]);
      torch::Tensor next_video = MiniMaxH3Scheduler::step_eta0(
          video_target,
          video_x0,
          schedule.video.sigmas[step].item<float>(),
          schedule.video.sigmas[step + 1].item<float>());
      torch::Tensor next_audio = MiniMaxH3Scheduler::step_eta0(
          audio_target,
          audio_x0,
          schedule.audio.sigmas[step].item<float>(),
          schedule.audio.sigmas[step + 1].item<float>());
      video_rows = video_rows.clone();
      audio_rows = audio_rows.clone();
      video_rows.index_put_({image_update}, next_video);
      audio_rows.index_put_({audio_update}, next_audio);
      video_rows.index_put_({~image_update}, video_anchor);
      audio_rows.index_put_({~audio_update}, audio_anchor);
      if (!torch::equal(video_rows.index({~image_update}), video_anchor) ||
          !torch::equal(audio_rows.index({~audio_update}), audio_anchor)) {
        throw std::runtime_error(
            "MiniMax-H3 condition anchor changed during denoising");
      }
      if (step_observer) {
        step_observer({.step = step,
                       .timestep_plan = std::move(plan),
                       .denoiser = std::move(denoiser),
                       .video_x0 = std::move(video_x0),
                       .audio_x0 = std::move(audio_x0),
                       .video_rows_after = video_rows,
                       .audio_rows_after = audio_rows});
      }
    }
    return {.video_rows = std::move(video_rows),
            .audio_rows = std::move(audio_rows),
            .transformer_forwards = 49,
            .block_forwards = 49 * MiniMaxH3TransformerConfig::kNumLayers};
  }

  MiniMaxH3ResidualBranchTrace probe_streaming_block(
      const std::vector<std::unique_ptr<StateDict>>& shards,
      int64_t layer,
      const torch::Tensor& hidden,
      const torch::Tensor& time_embedding,
      const torch::Tensor& combined_indices,
      const torch::Tensor& rope_frequencies,
      const torch::Tensor& cu_seqlens) {
    const SourceIndex source = build_source_index(shards);
    load_block_weights(source, layer);
    return streaming_block_->forward(
        hidden, time_embedding, combined_indices, rope_frequencies, cu_seqlens);
  }

  size_t loaded_fixed_tensor_count() const {
    return loaded_fixed_names_.size();
  }

  int64_t loaded_block_index() const { return loaded_block_index_; }

 private:
  using SourceIndex = std::unordered_map<std::string, const torch::Tensor*>;

  static SourceIndex build_source_index(
      const std::vector<std::unique_ptr<StateDict>>& shards) {
    SourceIndex source;
    source.reserve(MiniMaxH3SourceLayoutValidator::kExpectedTensorCount);
    for (const std::unique_ptr<StateDict>& shard : shards) {
      if (shard == nullptr) {
        throw std::invalid_argument(
            "MiniMax-H3 streaming source contains a null shard");
      }
      for (const auto& [name, tensor] : *shard) {
        if (!source.emplace(name, &tensor).second) {
          throw std::invalid_argument(
              "MiniMax-H3 streaming source contains duplicate tensor `" + name +
              "`");
        }
      }
    }
    return source;
  }

  void copy_fixed_tensor(const std::string& name,
                         torch::Tensor& target,
                         const SourceIndex& source) {
    const auto iterator = source.find(name);
    if (iterator == source.end()) {
      throw std::invalid_argument(
          "MiniMax-H3 streaming source is missing fixed tensor `" + name + "`");
    }
    torch::Tensor value = *iterator->second;
    if (name.ends_with(".attn.qkv_proj.weight")) {
      value = minimax_h3_reorder_grouped_qkv(
          value,
          MiniMaxH3TransformerConfig::kNumAttentionHeads,
          MiniMaxH3TransformerConfig::kAttentionHeadDim);
    } else if (name.ends_with(".mlp.fc1.weight")) {
      value = minimax_h3_reorder_gate_up_to_up_gate(value);
    }
    if (target.sizes() != value.sizes() ||
        target.scalar_type() != value.scalar_type()) {
      throw std::invalid_argument(
          "MiniMax-H3 streaming fixed tensor metadata mismatch for `" + name +
          "`");
    }
    target.copy_(value.to(target.device()));
    loaded_fixed_names_.emplace(name);
  }

  void load_block_weights(const SourceIndex& source, int64_t layer) {
    if (layer < 0 || layer >= MiniMaxH3TransformerConfig::kNumLayers) {
      throw std::invalid_argument(
          "MiniMax-H3 streaming block index is out of range");
    }
    auto targets = streaming_block_->named_parameters(/*recurse=*/true);
    if (targets.size() != kBlockTensorCount) {
      throw std::logic_error(
          "MiniMax-H3 streaming block tensor count mismatch");
    }
    const std::string prefix = "blocks." + std::to_string(layer) + ".";
    torch::NoGradGuard no_grad;
    for (auto& target_entry : targets) {
      const std::string source_name = prefix + target_entry.key();
      const auto iterator = source.find(source_name);
      if (iterator == source.end()) {
        throw std::invalid_argument(
            "MiniMax-H3 streaming source is missing block tensor `" +
            source_name + "`");
      }
      torch::Tensor value = *iterator->second;
      if (target_entry.key() == "attn.qkv_proj.weight") {
        value = minimax_h3_reorder_grouped_qkv(
            value,
            MiniMaxH3TransformerConfig::kNumAttentionHeads,
            MiniMaxH3TransformerConfig::kAttentionHeadDim);
      } else if (target_entry.key() == "mlp.fc1.weight") {
        value = minimax_h3_reorder_gate_up_to_up_gate(value);
      }
      torch::Tensor& target = target_entry.value();
      if (target.sizes() != value.sizes() ||
          target.scalar_type() != value.scalar_type()) {
        throw std::invalid_argument(
            "MiniMax-H3 streaming block tensor metadata mismatch for `" +
            source_name + "`");
      }
      target.copy_(value.to(target.device()));
    }
    loaded_block_index_ = layer;
  }

  void verify_fixed_weights() const {
    if (!fixed_loaded_ || loaded_fixed_names_.size() != kFixedTensorCount) {
      throw std::logic_error(
          "MiniMax-H3 streaming denoiser fixed weights are incomplete");
    }
  }

  static void validate_latent_rows(const H3PackedLayout& layout,
                                   const torch::Tensor& video_rows,
                                   const torch::Tensor& audio_rows) {
    if (!video_rows.defined() || video_rows.dim() != 2 ||
        video_rows.size(0) != layout.img_pos.numel() ||
        video_rows.size(1) != 96 ||
        video_rows.scalar_type() != torch::kFloat32 || !audio_rows.defined() ||
        audio_rows.dim() != 2 ||
        audio_rows.size(0) != layout.audio_pos.numel() ||
        audio_rows.size(1) != 32 ||
        audio_rows.scalar_type() != torch::kFloat32) {
      throw std::invalid_argument(
          "MiniMax-H3 denoise rows must be FP32 in packed position order");
    }
  }

  static void validate_inputs(const H3PackedLayout& layout,
                              const torch::Tensor& video_rows,
                              const torch::Tensor& audio_rows,
                              const torch::Tensor& timesteps,
                              const torch::Tensor& inverse_indices) {
    validate_latent_rows(layout, video_rows, audio_rows);
    if (layout.condition_hidden.sizes().vec() !=
            std::vector<int64_t>({1,
                                  layout.text_pos.numel(),
                                  MiniMaxH3TransformerConfig::kTextDim}) ||
        layout.condition_hidden.scalar_type() != torch::kBFloat16 ||
        layout.position_ids.sizes().vec() !=
            std::vector<int64_t>({layout.aligned_length, 3}) ||
        layout.position_ids.scalar_type() != torch::kFloat64 ||
        layout.token_tags.sizes().vec() !=
            std::vector<int64_t>({layout.aligned_length}) ||
        layout.token_tags.scalar_type() != torch::kInt64 ||
        layout.cu_seqlens.scalar_type() != torch::kInt32 ||
        !torch::equal(
            layout.cu_seqlens,
            torch::tensor({0,
                           static_cast<int32_t>(layout.used_length),
                           static_cast<int32_t>(layout.aligned_length)},
                          torch::kInt32))) {
      throw std::invalid_argument(
          "MiniMax-H3 denoiser packed layout metadata is invalid");
    }
    if (!timesteps.defined() || timesteps.dim() != 1 ||
        timesteps.numel() <= 0 || timesteps.scalar_type() != torch::kFloat32 ||
        !torch::isfinite(timesteps).all().item<bool>() ||
        timesteps.min().item<float>() < 0.0F ||
        timesteps.max().item<float>() > 1.0F || !inverse_indices.defined() ||
        inverse_indices.dim() != 1 ||
        inverse_indices.numel() != layout.aligned_length ||
        inverse_indices.scalar_type() != torch::kInt64 ||
        inverse_indices.min().item<int64_t>() < 0 ||
        inverse_indices.max().item<int64_t>() >= timesteps.numel()) {
      throw std::invalid_argument(
          "MiniMax-H3 denoiser timestep metadata is invalid");
    }

    std::vector<bool> occupied(static_cast<size_t>(layout.used_length), false);
    const auto claim = [&occupied, &layout](const torch::Tensor& positions,
                                            const char* name) {
      if (positions.scalar_type() != torch::kInt64 || positions.dim() != 1) {
        throw std::invalid_argument(std::string("MiniMax-H3 ") + name +
                                    " positions must be int64 [N]");
      }
      const torch::Tensor cpu = positions.to(torch::kCPU).contiguous();
      const int64_t* values = cpu.const_data_ptr<int64_t>();
      for (int64_t index = 0; index < cpu.numel(); ++index) {
        const int64_t position = values[index];
        if (position < 0 || position >= layout.used_length ||
            occupied[static_cast<size_t>(position)]) {
          throw std::invalid_argument(
              "MiniMax-H3 packed positions overlap or are out of range");
        }
        occupied[static_cast<size_t>(position)] = true;
      }
    };
    claim(layout.text_pos, "text");
    claim(layout.img_pos, "image");
    claim(layout.audio_pos, "audio");
    if (std::find(occupied.begin(), occupied.end(), false) != occupied.end() ||
        layout.update_mask.scalar_type() != torch::kBool ||
        layout.update_mask.numel() != layout.img_pos.numel() ||
        layout.audio_update_mask.scalar_type() != torch::kBool ||
        layout.audio_update_mask.numel() != layout.audio_pos.numel()) {
      throw std::invalid_argument(
          "MiniMax-H3 packed positions or update masks are incomplete");
    }
  }

  torch::TensorOptions options_;
  MiniMaxH3Dense video_patch_proj_{nullptr};
  MiniMaxH3Dense audio_patch_proj_{nullptr};
  MiniMaxH3Dense condition_proj_{nullptr};
  MiniMaxH3TimeEmbedder time_embedder_{nullptr};
  MiniMaxH3Rope rope_{nullptr};
  MiniMaxH3TokenRefiner token_refiner_{nullptr};
  MiniMaxH3DiTBlock streaming_block_{nullptr};
  MiniMaxH3FinalLayer final_layer_{nullptr};
  std::unordered_set<std::string> loaded_fixed_names_;
  int64_t loaded_block_index_ = -1;
  bool fixed_loaded_ = false;
};
TORCH_MODULE(MiniMaxH3StreamingDenoiser);

}  // namespace xllm
