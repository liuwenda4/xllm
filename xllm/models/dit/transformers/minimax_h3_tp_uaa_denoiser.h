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

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "models/dit/transformers/minimax_h3_denoiser.h"
#include "models/dit/transformers/minimax_h3_tp_uaa_resident.h"

namespace xllm {

struct MiniMaxH3TPUAAPreparedForward {
  torch::Tensor local_hidden;
  torch::Tensor time_embedding;
  torch::Tensor local_combined_indices;
  torch::Tensor local_rope_frequencies;
  torch::Tensor device_inverse_indices;
};

struct MiniMaxH3TPUAAOneForwardOutput {
  MiniMaxH3TPUAAPreparedForward prepared;
  torch::Tensor local_transformer_output;
  torch::Tensor full_transformer_output;
  MiniMaxH3FinalOutput final_output;
};

struct MiniMaxH3TPUAATrajectoryOutput {
  torch::Tensor video_rows;
  torch::Tensor audio_rows;
  int64_t transformer_forwards = 0;
  int64_t block_forwards = 0;
};

struct MiniMaxH3TPUAABoundary {
  int64_t forward_index = 0;
  torch::Tensor video_rows_before;
  torch::Tensor audio_rows_before;
  torch::Tensor raw_video_velocity;
  torch::Tensor raw_audio_velocity;
  torch::Tensor video_x0;
  torch::Tensor audio_x0;
  torch::Tensor video_rows_after;
  torch::Tensor audio_rows_after;
};

using MiniMaxH3TPUAATrajectoryObserver =
    std::function<void(const MiniMaxH3TPUAABoundary&)>;

class MiniMaxH3TPUAAFixedStageImpl final : public torch::nn::Module {
 public:
  static constexpr size_t kFixedTensorCount = 35;

  MiniMaxH3TPUAAFixedStageImpl(const MiniMaxH3C4Config& config,
                               ProcessGroup* u_group,
                               const torch::TensorOptions& options)
      : config_(config), u_group_(u_group), options_(options) {
    if (u_group_ == nullptr || u_group_->world_size() != kMiniMaxH3UaaSize) {
      throw std::invalid_argument(
          "MiniMax-H3 fixed stage requires an exact U8 group");
    }
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
    final_layer_ =
        register_module("final_layer", MiniMaxH3FinalLayer(config, options));
  }

  void load_source_weights(
      const std::vector<std::unique_ptr<StateDict>>& shards) {
    if (loaded_) {
      throw std::logic_error(
          "MiniMax-H3 fixed stage weights are already loaded");
    }
    MiniMaxH3SourceLayoutValidator::validate(shards);
    const SourceIndex source = build_source_index(shards);
    auto parameters = named_parameters(/*recurse=*/true);
    auto buffers = named_buffers(/*recurse=*/true);
    torch::NoGradGuard no_grad;
    for (auto& parameter : parameters) {
      copy_tensor(parameter.key(), parameter.value(), source);
    }
    for (auto& buffer : buffers) {
      copy_tensor(buffer.key(), buffer.value(), source);
    }
    if (loaded_names_.size() != kFixedTensorCount) {
      throw std::logic_error("MiniMax-H3 fixed stage tensor count mismatch");
    }
    minimax_h3_synchronize_weight_load(condition_proj_->weight().device());
    loaded_ = true;
  }

  MiniMaxH3TPUAAPreparedForward prepare(const H3PackedLayout& layout,
                                        const torch::Tensor& video_rows,
                                        const torch::Tensor& audio_rows,
                                        const torch::Tensor& timesteps,
                                        const torch::Tensor& inverse_indices) {
    verify_loaded();
    if (layout.aligned_length % kMiniMaxH3UaaSize != 0 ||
        video_rows.size(0) != layout.img_pos.numel() ||
        audio_rows.size(0) != layout.audio_pos.numel() ||
        inverse_indices.numel() != layout.aligned_length) {
      throw std::invalid_argument(
          "MiniMax-H3 fixed stage input geometry mismatch");
    }
    const torch::Device device = condition_proj_->weight().device();
    const torch::Tensor condition =
        layout.condition_hidden.squeeze(0).to(device);
    const torch::Tensor projected_condition =
        condition_proj_->forward(condition);
    const torch::Tensor refiner_cu = torch::tensor(
        std::vector<int64_t>{0, condition.size(0)},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    const torch::Tensor refined_condition =
        token_refiner_->forward_output_only(projected_condition, refiner_cu);
    const torch::Tensor video_embedding =
        video_patch_proj_->forward(video_rows.to(device));
    const torch::Tensor audio_embedding =
        audio_patch_proj_->forward(audio_rows.to(device));
    torch::Tensor hidden =
        torch::zeros({layout.aligned_length, config_.hidden_size},
                     options_.dtype(torch::kBFloat16));
    hidden.index_copy_(0, layout.text_pos.to(device), refined_condition);
    hidden.index_copy_(
        0, layout.img_pos.to(device), video_embedding.to(torch::kBFloat16));
    hidden.index_copy_(
        0, layout.audio_pos.to(device), audio_embedding.to(torch::kBFloat16));
    const torch::Tensor time_embedding =
        time_embedder_->forward(timesteps.to(device));
    const torch::Tensor rope_frequencies =
        rope_->forward(layout.position_ids.to(device));
    const torch::Tensor device_inverse = inverse_indices.to(device);
    const torch::Tensor combined_indices = minimax_h3_combined_adaln_indices(
        device_inverse, layout.token_tags.to(device));
    const int64_t local_rows = layout.aligned_length / kMiniMaxH3UaaSize;
    const int64_t start = u_group_->rank() * local_rows;
    return {.local_hidden = hidden.narrow(0, start, local_rows).contiguous(),
            .time_embedding = time_embedding,
            .local_combined_indices =
                combined_indices.narrow(0, start, local_rows).contiguous(),
            .local_rope_frequencies =
                rope_frequencies.narrow(0, start, local_rows).contiguous(),
            .device_inverse_indices = device_inverse};
  }

  MiniMaxH3FinalOutput finalize(const torch::Tensor& local_hidden,
                                const torch::Tensor& time_embedding,
                                const torch::Tensor& device_inverse_indices,
                                const H3PackedLayout& layout) {
    verify_loaded();
    const torch::Tensor full_hidden =
        parallel_state::gather(local_hidden, u_group_, /*dim=*/0);
    return final_layer_->forward(
        full_hidden, time_embedding, device_inverse_indices, layout);
  }

  ProcessGroup* u_group() const { return u_group_; }
  torch::Device device() const { return condition_proj_->weight().device(); }

 private:
  using SourceIndex = std::unordered_map<std::string, const torch::Tensor*>;

  static SourceIndex build_source_index(
      const std::vector<std::unique_ptr<StateDict>>& shards) {
    SourceIndex source;
    source.reserve(MiniMaxH3SourceLayoutValidator::kExpectedTensorCount);
    for (const std::unique_ptr<StateDict>& shard : shards) {
      if (shard == nullptr) {
        throw std::invalid_argument(
            "MiniMax-H3 fixed source contains a null shard");
      }
      for (const auto& [name, tensor] : *shard) {
        if (!source.emplace(name, &tensor).second) {
          throw std::invalid_argument(
              "MiniMax-H3 fixed source contains duplicate tensor `" + name +
              "`");
        }
      }
    }
    return source;
  }

  void copy_tensor(const std::string& name,
                   torch::Tensor& target,
                   const SourceIndex& source) {
    const auto iterator = source.find(name);
    if (iterator == source.end()) {
      throw std::invalid_argument(
          "MiniMax-H3 fixed source is missing tensor `" + name + "`");
    }
    torch::Tensor value = *iterator->second;
    if (name.ends_with(".attn.qkv_proj.weight")) {
      value = minimax_h3_reorder_grouped_qkv(
          value, config_.num_attention_heads, config_.attention_head_dim);
    } else if (name.ends_with(".mlp.fc1.weight")) {
      value = minimax_h3_reorder_gate_up_to_up_gate(value);
    }
    if (target.sizes() != value.sizes() ||
        target.scalar_type() != value.scalar_type()) {
      throw std::invalid_argument(
          "MiniMax-H3 fixed tensor metadata mismatch for `" + name + "`");
    }
    target.copy_(value.to(target.device()));
    loaded_names_.emplace(name);
  }

  void verify_loaded() const {
    if (!loaded_ || loaded_names_.size() != kFixedTensorCount) {
      throw std::logic_error("MiniMax-H3 fixed stage weights are incomplete");
    }
  }

  MiniMaxH3C4Config config_;
  ProcessGroup* u_group_;
  torch::TensorOptions options_;
  MiniMaxH3Dense video_patch_proj_{nullptr};
  MiniMaxH3Dense audio_patch_proj_{nullptr};
  MiniMaxH3Dense condition_proj_{nullptr};
  MiniMaxH3TimeEmbedder time_embedder_{nullptr};
  MiniMaxH3Rope rope_{nullptr};
  MiniMaxH3TokenRefiner token_refiner_{nullptr};
  MiniMaxH3FinalLayer final_layer_{nullptr};
  std::unordered_set<std::string> loaded_names_;
  bool loaded_ = false;
};
TORCH_MODULE(MiniMaxH3TPUAAFixedStage);

class MiniMaxH3TPUAAResidentDenoiserImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TPUAAResidentDenoiserImpl(const MiniMaxH3C4Config& config,
                                     ProcessGroup* tp_group,
                                     ProcessGroup* u_group,
                                     const torch::TensorOptions& options) {
    fixed_ = register_module(
        "fixed", MiniMaxH3TPUAAFixedStage(config, u_group, options));
    transformer_ = register_module(
        "transformer",
        MiniMaxH3TPUAAResidentTransformer(config, tp_group, u_group, options));
  }

  void load_source_weights(
      const std::vector<std::unique_ptr<StateDict>>& shards) {
    fixed_->load_source_weights(shards);
    transformer_->load_source_weights(shards);
  }

  MiniMaxH3TPUAAOneForwardOutput forward(const H3PackedLayout& layout,
                                         const torch::Tensor& video_rows,
                                         const torch::Tensor& audio_rows,
                                         const torch::Tensor& timesteps,
                                         const torch::Tensor& inverse_indices) {
    MiniMaxH3TPUAAOneForwardOutput output;
    output.prepared = fixed_->prepare(
        layout, video_rows, audio_rows, timesteps, inverse_indices);
    output.local_transformer_output =
        transformer_->forward(output.prepared.local_hidden,
                              output.prepared.time_embedding,
                              output.prepared.local_combined_indices,
                              output.prepared.local_rope_frequencies,
                              layout.cu_seqlens);
    output.full_transformer_output = parallel_state::gather(
        output.local_transformer_output, fixed_->u_group(), /*dim=*/0);
    output.final_output =
        fixed_->finalize(output.local_transformer_output,
                         output.prepared.time_embedding,
                         output.prepared.device_inverse_indices,
                         layout);
    return output;
  }

  MiniMaxH3FinalOutput forward_output_only(
      const H3PackedLayout& layout,
      const torch::Tensor& video_rows,
      const torch::Tensor& audio_rows,
      const torch::Tensor& timesteps,
      const torch::Tensor& inverse_indices) {
    const MiniMaxH3TPUAAPreparedForward prepared = fixed_->prepare(
        layout, video_rows, audio_rows, timesteps, inverse_indices);
    const torch::Tensor local_transformer_output =
        transformer_->forward(prepared.local_hidden,
                              prepared.time_embedding,
                              prepared.local_combined_indices,
                              prepared.local_rope_frequencies,
                              layout.cu_seqlens);
    return fixed_->finalize(local_transformer_output,
                            prepared.time_embedding,
                            prepared.device_inverse_indices,
                            layout);
  }

  MiniMaxH3TPUAATrajectoryOutput run_base_trajectory(
      const H3PackedLayout& layout,
      const torch::Tensor& initial_video_rows,
      const torch::Tensor& initial_audio_rows,
      const MiniMaxH3TPUAATrajectoryObserver& observer = nullptr) {
    torch::NoGradGuard no_grad;
    if (!initial_video_rows.defined() || initial_video_rows.dim() != 2 ||
        initial_video_rows.size(0) != layout.img_pos.numel() ||
        initial_video_rows.size(1) != 96 ||
        initial_video_rows.scalar_type() != torch::kFloat32 ||
        !initial_audio_rows.defined() || initial_audio_rows.dim() != 2 ||
        initial_audio_rows.size(0) != layout.audio_pos.numel() ||
        initial_audio_rows.size(1) != 32 ||
        initial_audio_rows.scalar_type() != torch::kFloat32) {
      throw std::invalid_argument(
          "MiniMax-H3 trajectory rows must be FP32 in packed position order");
    }
    const MiniMaxH3DualSigmaSchedule schedule =
        MiniMaxH3Scheduler::build_base();
    if (schedule.video.forward_count() != schedule.audio.forward_count() ||
        schedule.video.forward_count() != 49) {
      throw std::logic_error(
          "MiniMax-H3 Base dual schedule must have 49 forwards");
    }
    const torch::Device device = fixed_->device();
    const torch::Tensor image_update = layout.update_mask.to(device);
    const torch::Tensor audio_update = layout.audio_update_mask.to(device);
    torch::Tensor video_rows = initial_video_rows.to(device).clone();
    torch::Tensor audio_rows = initial_audio_rows.to(device).clone();
    const torch::Tensor video_anchor =
        video_rows.index({~image_update}).clone();
    const torch::Tensor audio_anchor =
        audio_rows.index({~audio_update}).clone();
    for (int64_t step = 0; step < schedule.video.forward_count(); ++step) {
      const torch::Tensor video_rows_before = video_rows;
      const torch::Tensor audio_rows_before = audio_rows;
      const MiniMaxH3RowTimestepPlan plan = minimax_h3_build_row_timestep_plan(
          layout,
          schedule.video.timesteps[step].item<float>(),
          schedule.audio.timesteps[step].item<float>());
      const MiniMaxH3FinalOutput denoiser =
          forward_output_only(layout,
                              video_rows,
                              audio_rows,
                              plan.unique_timesteps,
                              plan.inverse_indices);
      const torch::Tensor video_target = video_rows.index({image_update});
      const torch::Tensor audio_target = audio_rows.index({audio_update});
      const torch::Tensor video_x0 = MiniMaxH3Scheduler::velocity_to_x0(
          video_target,
          denoiser.raw_selected_video_logits.index({image_update}),
          schedule.video.timesteps[step]);
      const torch::Tensor audio_x0 = MiniMaxH3Scheduler::velocity_to_x0(
          audio_target,
          denoiser.raw_selected_audio_logits.index({audio_update}),
          schedule.audio.timesteps[step]);
      const torch::Tensor next_video = MiniMaxH3Scheduler::step_eta0(
          video_target,
          video_x0,
          schedule.video.sigmas[step].item<float>(),
          schedule.video.sigmas[step + 1].item<float>());
      const torch::Tensor next_audio = MiniMaxH3Scheduler::step_eta0(
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
      if (observer) {
        observer({.forward_index = step + 1,
                  .video_rows_before = video_rows_before,
                  .audio_rows_before = audio_rows_before,
                  .raw_video_velocity = denoiser.raw_selected_video_logits,
                  .raw_audio_velocity = denoiser.raw_selected_audio_logits,
                  .video_x0 = video_x0,
                  .audio_x0 = audio_x0,
                  .video_rows_after = video_rows,
                  .audio_rows_after = audio_rows});
      }
    }
    return {.video_rows = std::move(video_rows),
            .audio_rows = std::move(audio_rows),
            .transformer_forwards = 49,
            .block_forwards = 49 * kMiniMaxH3ResidentBlockCount};
  }

  MiniMaxH3TPUAAResidentTransformer transformer() const { return transformer_; }

 private:
  MiniMaxH3TPUAAFixedStage fixed_{nullptr};
  MiniMaxH3TPUAAResidentTransformer transformer_{nullptr};
};
TORCH_MODULE(MiniMaxH3TPUAAResidentDenoiser);

}  // namespace xllm
