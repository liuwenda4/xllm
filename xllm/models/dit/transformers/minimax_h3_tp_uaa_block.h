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
#include <stdexcept>
#include <utility>
#include <vector>

#include "models/dit/transformers/minimax_h3_blocks.h"
#include "models/dit/utils/minimax_h3_uaa.h"

namespace xllm {

inline constexpr int32_t kMiniMaxH3TPUAAWorldSize = 16;
inline constexpr int32_t kMiniMaxH3TPSize = 2;

struct MiniMaxH3TPUAAParallelCoordinates {
  int32_t global_rank;
  int32_t tp_rank;
  int32_t u_rank;
  std::vector<int32_t> tp_group_ranks;
  std::vector<int32_t> u_group_ranks;
};

inline MiniMaxH3TPUAAParallelCoordinates minimax_h3_tp_uaa_coordinates(
    int32_t global_rank) {
  if (global_rank < 0 || global_rank >= kMiniMaxH3TPUAAWorldSize) {
    throw std::invalid_argument(
        "MiniMax-H3 TP2 x U8 global rank must be in [0,16)");
  }
  const int32_t tp_rank = global_rank % kMiniMaxH3TPSize;
  const int32_t u_rank = global_rank / kMiniMaxH3TPSize;
  std::vector<int32_t> u_group_ranks;
  u_group_ranks.reserve(kMiniMaxH3UaaSize);
  for (int32_t rank = tp_rank; rank < kMiniMaxH3TPUAAWorldSize;
       rank += kMiniMaxH3TPSize) {
    u_group_ranks.push_back(rank);
  }
  return {.global_rank = global_rank,
          .tp_rank = tp_rank,
          .u_rank = u_rank,
          .tp_group_ranks = {2 * u_rank, 2 * u_rank + 1},
          .u_group_ranks = std::move(u_group_ranks)};
}

struct MiniMaxH3TPUAAAttentionDiagnostics {
  torch::Tensor query_u;
  torch::Tensor key_u;
  torch::Tensor value_u;
  torch::Tensor attention_u;
};

// The communicator owns both groups and must outlive this inference module.
class MiniMaxH3TPUAADiTBlockImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TPUAADiTBlockImpl(const MiniMaxH3C4Config& config,
                             ProcessGroup* tp_group,
                             ProcessGroup* u_group,
                             const torch::TensorOptions& options)
      : config_(config), tp_group_(tp_group), u_group_(u_group) {
    if (tp_group_ == nullptr || tp_group_->world_size() != kMiniMaxH3TPSize ||
        u_group_ == nullptr || u_group_->world_size() != kMiniMaxH3UaaSize ||
        tp_group_->device() != u_group_->device()) {
      throw std::invalid_argument(
          "MiniMax-H3 combined block requires colocated TP2 and U8 groups");
    }
    if (config_.num_attention_heads != 56 ||
        config_.attention_head_dim != kMiniMaxH3UaaHeadDim ||
        config_.num_attention_heads / kMiniMaxH3TPSize !=
            kMiniMaxH3UaaLogicalHeads) {
      throw std::invalid_argument(
          "MiniMax-H3 combined block requires locked 56-head geometry");
    }
    tp_block_ = register_module(
        "tp_block",
        MiniMaxH3TPDiTBlock(
            config_, tp_group_, options, /*dense_output_projection=*/true));
    fc2_weight_dense_ = register_buffer(
        "fc2_weight_dense",
        torch::empty({config_.hidden_size, config_.ffn_hidden_size},
                     options.dtype(torch::kBFloat16)));
  }

  void load_source_weights(
      const std::vector<std::unique_ptr<StateDict>>& shards,
      int64_t layer_index) {
    tp_block_->load_source_weights(shards, layer_index);
    const torch::Tensor fc2_weight = source_tensor(
        shards, "blocks." + std::to_string(layer_index) + ".mlp.fc2.weight");
    if (fc2_weight.sizes() != fc2_weight_dense_.sizes() ||
        fc2_weight.scalar_type() != torch::kBFloat16) {
      throw std::invalid_argument(
          "MiniMax-H3 combined block dense FC2 weight mismatch");
    }
    fc2_weight_dense_.copy_(fc2_weight);
    minimax_h3_synchronize_weight_load(fc2_weight_dense_.device());
    row_weights_loaded_ = true;
  }

  MiniMaxH3ResidualBranchTrace forward(
      const torch::Tensor& input,
      const torch::Tensor& time_embedding,
      const torch::Tensor& combined_indices,
      const torch::Tensor& rope_frequencies,
      const torch::Tensor& global_cu_seqlens,
      MiniMaxH3TPUAAAttentionDiagnostics* diagnostics = nullptr) {
    tp_block_->verify_loaded_weights();
    validate_inputs(input,
                    time_embedding,
                    combined_indices,
                    rope_frequencies,
                    global_cu_seqlens);

    torch::Tensor parameters =
        tp_block_->adaln_projection()->forward(time_embedding);
    torch::Tensor shift_msa =
        minimax_h3_select_adaln_parameter(parameters, 0, combined_indices);
    torch::Tensor scale_msa =
        minimax_h3_select_adaln_parameter(parameters, 1, combined_indices);
    torch::Tensor gate_msa =
        minimax_h3_select_adaln_parameter(parameters, 2, combined_indices);
    torch::Tensor shift_mlp =
        minimax_h3_select_adaln_parameter(parameters, 3, combined_indices);
    torch::Tensor scale_mlp =
        minimax_h3_select_adaln_parameter(parameters, 4, combined_indices);
    torch::Tensor gate_mlp =
        minimax_h3_select_adaln_parameter(parameters, 5, combined_indices);

    MiniMaxH3ResidualBranchTrace trace;
    trace.adaln_parameters = parameters;
    trace.shift_msa = shift_msa;
    trace.scale_msa = scale_msa;
    trace.gate_msa = gate_msa;
    trace.shift_mlp = shift_mlp;
    trace.scale_mlp = scale_mlp;
    trace.gate_mlp = gate_mlp;
    trace.norm1_output =
        minimax_h3_rms_norm(input, tp_block_->norm1_weight(), config_.norm_eps);
    trace.attention_input = trace.norm1_output * (scale_msa + 1.0) + shift_msa;
    trace.attention_output = forward_attention(trace.attention_input,
                                               rope_frequencies,
                                               global_cu_seqlens,
                                               diagnostics);
    trace.attention_delta = gate_msa * trace.attention_output;
    torch::Tensor after_attention = input + trace.attention_delta;
    trace.norm2_output = minimax_h3_rms_norm(
        after_attention, tp_block_->norm2_weight(), config_.norm_eps);
    trace.mlp_input = trace.norm2_output * (scale_mlp + 1.0) + shift_mlp;
    trace.mlp_output = forward_mlp(trace.mlp_input, global_cu_seqlens);
    trace.mlp_delta = gate_mlp * trace.mlp_output;
    trace.output = after_attention + trace.mlp_delta;
    return trace;
  }

  torch::Tensor forward_output_only(const torch::Tensor& input,
                                    const torch::Tensor& time_embedding,
                                    const torch::Tensor& combined_indices,
                                    const torch::Tensor& rope_frequencies,
                                    const torch::Tensor& global_cu_seqlens) {
    tp_block_->verify_loaded_weights();
    validate_inputs(input,
                    time_embedding,
                    combined_indices,
                    rope_frequencies,
                    global_cu_seqlens);
    return forward_output_only_impl(input,
                                    time_embedding,
                                    combined_indices,
                                    rope_frequencies,
                                    global_cu_seqlens);
  }

  void validate_forward_inputs(const torch::Tensor& input,
                               const torch::Tensor& time_embedding,
                               const torch::Tensor& combined_indices,
                               const torch::Tensor& rope_frequencies,
                               const torch::Tensor& global_cu_seqlens) const {
    validate_inputs(input,
                    time_embedding,
                    combined_indices,
                    rope_frequencies,
                    global_cu_seqlens);
  }

  torch::Tensor forward_output_only_assuming_validated(
      const torch::Tensor& input,
      const torch::Tensor& time_embedding,
      const torch::Tensor& combined_indices,
      const torch::Tensor& rope_frequencies,
      const torch::Tensor& global_cu_seqlens) {
    tp_block_->verify_loaded_weights();
    return forward_output_only_impl(input,
                                    time_embedding,
                                    combined_indices,
                                    rope_frequencies,
                                    global_cu_seqlens);
  }

  MiniMaxH3TPDiTBlock tp_block() const { return tp_block_; }

 private:
  torch::Tensor forward_output_only_impl(
      const torch::Tensor& input,
      const torch::Tensor& time_embedding,
      const torch::Tensor& combined_indices,
      const torch::Tensor& rope_frequencies,
      const torch::Tensor& global_cu_seqlens) {
    const torch::Tensor parameters =
        tp_block_->adaln_projection()->forward(time_embedding);
    const torch::Tensor shift_msa =
        minimax_h3_select_adaln_parameter(parameters, 0, combined_indices);
    const torch::Tensor scale_msa =
        minimax_h3_select_adaln_parameter(parameters, 1, combined_indices);
    const torch::Tensor gate_msa =
        minimax_h3_select_adaln_parameter(parameters, 2, combined_indices);
    const torch::Tensor shift_mlp =
        minimax_h3_select_adaln_parameter(parameters, 3, combined_indices);
    const torch::Tensor scale_mlp =
        minimax_h3_select_adaln_parameter(parameters, 4, combined_indices);
    const torch::Tensor gate_mlp =
        minimax_h3_select_adaln_parameter(parameters, 5, combined_indices);
    const torch::Tensor attention_input =
        minimax_h3_rms_norm(
            input, tp_block_->norm1_weight(), config_.norm_eps) *
            (scale_msa + 1.0) +
        shift_msa;
    const torch::Tensor after_attention =
        input + gate_msa * forward_attention(attention_input,
                                             rope_frequencies,
                                             global_cu_seqlens,
                                             /*diagnostics=*/nullptr);
    const torch::Tensor mlp_input =
        minimax_h3_rms_norm(
            after_attention, tp_block_->norm2_weight(), config_.norm_eps) *
            (scale_mlp + 1.0) +
        shift_mlp;
    return after_attention +
           gate_mlp * forward_mlp(mlp_input, global_cu_seqlens);
  }
  torch::Tensor forward_attention(
      const torch::Tensor& input,
      const torch::Tensor& rope_frequencies,
      const torch::Tensor& global_cu_seqlens,
      MiniMaxH3TPUAAAttentionDiagnostics* diagnostics) {
    MiniMaxH3TPAttention attention = tp_block_->attention();
    MiniMaxH3TPAttentionQKV local =
        attention->project_local_qkv(input, rope_frequencies);

    torch::Tensor query_u =
        minimax_h3_uaa_launch_forward(local.query.unsqueeze(0), u_group_)
            .finish();
    torch::Tensor key_u =
        minimax_h3_uaa_launch_forward(local.key.unsqueeze(0), u_group_)
            .finish();
    torch::Tensor value_u =
        minimax_h3_uaa_launch_forward(local.value.unsqueeze(0), u_group_)
            .finish();

    torch::Tensor attention_u = minimax_h3_segmented_sdpa(query_u.squeeze(0),
                                                          key_u.squeeze(0),
                                                          value_u.squeeze(0),
                                                          global_cu_seqlens);
    torch::Tensor local_heads =
        minimax_h3_uaa_launch_inverse(attention_u.unsqueeze(0), u_group_)
            .finish()
            .squeeze(0);
    const torch::Tensor full_heads =
        parallel_state::gather(local_heads, tp_group_, /*dim=*/1);
    torch::Tensor output = attention->project_dense_output(full_heads);

    if (diagnostics != nullptr) {
      diagnostics->query_u = std::move(query_u);
      diagnostics->key_u = std::move(key_u);
      diagnostics->value_u = std::move(value_u);
      diagnostics->attention_u = attention_u.unsqueeze(0);
    }
    return output;
  }

  torch::Tensor forward_mlp(const torch::Tensor& input,
                            const torch::Tensor& global_cu_seqlens) {
    if (!row_weights_loaded_) {
      throw std::logic_error(
          "MiniMax-H3 combined block row weights are not loaded");
    }
    MiniMaxH3TPMLP mlp = tp_block_->mlp();
    (void)global_cu_seqlens;
    const std::vector<torch::Tensor> up_gate =
        torch::nn::functional::linear(input, mlp->fc1_weight()).chunk(2, -1);
    const torch::Tensor activation = up_gate[0] * torch::silu(up_gate[1]);
    const torch::Tensor full_activation =
        parallel_state::gather(activation, tp_group_, /*dim=*/-1);
    return torch::nn::functional::linear(full_activation, fc2_weight_dense_);
  }

  static torch::Tensor source_tensor(
      const std::vector<std::unique_ptr<StateDict>>& shards,
      const std::string& name) {
    torch::Tensor found;
    for (const std::unique_ptr<StateDict>& shard : shards) {
      if (shard == nullptr) {
        throw std::invalid_argument(
            "MiniMax-H3 combined block source contains a null shard");
      }
      const torch::Tensor candidate = shard->get_tensor(name);
      if (!candidate.defined()) {
        continue;
      }
      if (found.defined()) {
        throw std::invalid_argument(
            "MiniMax-H3 combined block source tensor is duplicated: `" + name +
            "`");
      }
      found = candidate;
    }
    if (!found.defined()) {
      throw std::invalid_argument(
          "MiniMax-H3 combined block source tensor is missing: `" + name + "`");
    }
    return found;
  }

  void validate_inputs(const torch::Tensor& input,
                       const torch::Tensor& time_embedding,
                       const torch::Tensor& combined_indices,
                       const torch::Tensor& rope_frequencies,
                       const torch::Tensor& global_cu_seqlens) const {
    if (!input.defined() || input.dim() != 2 || input.size(0) <= 0 ||
        input.size(1) != config_.hidden_size ||
        input.scalar_type() != torch::kBFloat16 ||
        input.device() != tp_group_->device()) {
      throw std::invalid_argument(
          "MiniMax-H3 combined block input must be device-local BF16 [S,H]");
    }
    if (!time_embedding.defined() || time_embedding.dim() != 2 ||
        time_embedding.size(1) != config_.time_embed_dim ||
        time_embedding.scalar_type() != torch::kFloat32 ||
        time_embedding.device() != input.device()) {
      throw std::invalid_argument(
          "MiniMax-H3 combined block time embedding mismatch");
    }
    if (!combined_indices.defined() || combined_indices.dim() != 1 ||
        combined_indices.size(0) != input.size(0) ||
        combined_indices.scalar_type() != torch::kInt64 ||
        combined_indices.device() != input.device()) {
      throw std::invalid_argument(
          "MiniMax-H3 combined block AdaLN indices mismatch");
    }
    if (combined_indices.min().item<int64_t>() < 0 ||
        combined_indices.max().item<int64_t>() >= 3 * time_embedding.size(0)) {
      throw std::invalid_argument(
          "MiniMax-H3 combined block AdaLN indices are out of range");
    }
    if (!rope_frequencies.defined() || rope_frequencies.dim() != 2 ||
        rope_frequencies.size(0) != input.size(0) ||
        rope_frequencies.size(1) != 6 * config_.rope_inv_freq_len ||
        rope_frequencies.scalar_type() != torch::kFloat32 ||
        rope_frequencies.device() != input.device()) {
      throw std::invalid_argument(
          "MiniMax-H3 combined block local RoPE mismatch");
    }
    if (!global_cu_seqlens.defined() || global_cu_seqlens.dim() != 1 ||
        global_cu_seqlens.numel() < 2 ||
        (global_cu_seqlens.scalar_type() != torch::kInt32 &&
         global_cu_seqlens.scalar_type() != torch::kInt64)) {
      throw std::invalid_argument(
          "MiniMax-H3 combined block global sequence boundaries mismatch");
    }
    const torch::Tensor boundaries =
        global_cu_seqlens.to(torch::kCPU).to(torch::kInt64).contiguous();
    const int64_t* values = boundaries.data_ptr<int64_t>();
    if (values[0] != 0 ||
        values[boundaries.numel() - 1] != input.size(0) * kMiniMaxH3UaaSize) {
      throw std::invalid_argument(
          "MiniMax-H3 combined block global sequence boundaries mismatch");
    }
    for (int64_t index = 0; index + 1 < boundaries.numel(); ++index) {
      if (values[index] < 0 || values[index + 1] < values[index]) {
        throw std::invalid_argument(
            "MiniMax-H3 combined block sequence boundaries must be monotonic");
      }
    }
  }

  MiniMaxH3C4Config config_;
  ProcessGroup* tp_group_;
  ProcessGroup* u_group_;
  MiniMaxH3TPDiTBlock tp_block_{nullptr};
  torch::Tensor fc2_weight_dense_;
  bool row_weights_loaded_ = false;
};
TORCH_MODULE(MiniMaxH3TPUAADiTBlock);

}  // namespace xllm
