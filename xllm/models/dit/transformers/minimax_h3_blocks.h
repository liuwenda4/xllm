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

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/parallel_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/linear.h"
#include "models/dit/utils/minimax_h3_packing.h"
#include "platform/device.h"

namespace xllm {

struct MiniMaxH3C4Config {
  int64_t hidden_size = 5376;
  int64_t num_attention_heads = 56;
  int64_t attention_head_dim = 128;
  int64_t ffn_hidden_size = 14336;
  int64_t video_patch_dim = 96;
  int64_t audio_dim = 32;
  int64_t text_dim = 5120;
  int64_t timestep_input_dim = 256;
  int64_t time_embed_hidden_size = 5376;
  int64_t time_embed_dim = 2688;
  int64_t rope_inv_freq_len = 16;
  int64_t num_refiner_layers = 2;
  double norm_eps = 1e-5;
  double qk_norm_eps = 1e-5;
  double final_norm_eps = 1e-5;
};

struct MiniMaxH3C4SourceTensorSpec {
  std::string name;
  std::vector<int64_t> shape;
  torch::ScalarType dtype;
};

struct MiniMaxH3ResidualBranchTrace {
  torch::Tensor adaln_parameters;
  torch::Tensor shift_msa;
  torch::Tensor scale_msa;
  torch::Tensor gate_msa;
  torch::Tensor shift_mlp;
  torch::Tensor scale_mlp;
  torch::Tensor gate_mlp;
  torch::Tensor norm1_output;
  torch::Tensor attention_input;
  torch::Tensor attention_output;
  torch::Tensor attention_delta;
  torch::Tensor norm2_output;
  torch::Tensor mlp_input;
  torch::Tensor mlp_output;
  torch::Tensor mlp_delta;
  torch::Tensor output;
};

struct MiniMaxH3FinalOutput {
  torch::Tensor activation;
  torch::Tensor all_video_logits;
  torch::Tensor all_audio_logits;
  torch::Tensor raw_selected_video_logits;
  torch::Tensor raw_selected_audio_logits;
  torch::Tensor selected_video_logits;
  torch::Tensor selected_audio_logits;
};

struct MiniMaxH3C4Trace {
  torch::Tensor condition_projection;
  std::vector<MiniMaxH3ResidualBranchTrace> token_refiner_blocks;
  torch::Tensor refined_condition;
  torch::Tensor video_embedding;
  torch::Tensor audio_embedding;
  torch::Tensor packed_hidden;
  torch::Tensor time_embedding;
  torch::Tensor rope_frequencies;
  torch::Tensor combined_indices;
  MiniMaxH3ResidualBranchTrace block0;
  MiniMaxH3FinalOutput final_output;
};

inline torch::Tensor minimax_h3_reorder_grouped_qkv(const torch::Tensor& weight,
                                                    int64_t num_heads,
                                                    int64_t head_dim) {
  if (!weight.defined() || weight.dim() < 1 || num_heads <= 0 ||
      head_dim <= 0 || weight.size(0) != 3 * num_heads * head_dim) {
    throw std::invalid_argument(
        "MiniMax-H3 grouped QKV weight has incompatible shape");
  }
  std::vector<int64_t> grouped_shape = weight.sizes().vec();
  grouped_shape[0] = num_heads;
  grouped_shape.insert(grouped_shape.begin() + 1, 3 * head_dim);
  torch::Tensor grouped = weight.reshape(grouped_shape);

  std::vector<int64_t> output_shape = weight.sizes().vec();
  output_shape[0] = num_heads * head_dim;
  torch::Tensor query = grouped.narrow(1, 0, head_dim).reshape(output_shape);
  torch::Tensor key =
      grouped.narrow(1, head_dim, head_dim).reshape(output_shape);
  torch::Tensor value =
      grouped.narrow(1, 2 * head_dim, head_dim).reshape(output_shape);
  return torch::cat({query, key, value}, 0).contiguous();
}

inline torch::Tensor minimax_h3_reorder_gate_up_to_up_gate(
    const torch::Tensor& weight) {
  if (!weight.defined() || weight.dim() < 1 || weight.size(0) % 2 != 0) {
    throw std::invalid_argument(
        "MiniMax-H3 gate/up weight has incompatible shape");
  }
  const std::vector<torch::Tensor> gate_up = weight.chunk(2, 0);
  return torch::cat({gate_up[1], gate_up[0]}, 0).contiguous();
}

inline torch::Tensor minimax_h3_tp_shard_grouped_qkv(
    const torch::Tensor& weight,
    int64_t tp_rank,
    int64_t tp_size,
    int64_t num_heads,
    int64_t head_dim) {
  if (!weight.defined() || weight.dim() < 1 || tp_size <= 0 || tp_rank < 0 ||
      tp_rank >= tp_size || num_heads <= 0 || num_heads % tp_size != 0 ||
      head_dim <= 0 || weight.size(0) != 3 * num_heads * head_dim) {
    throw std::invalid_argument(
        "MiniMax-H3 TP grouped QKV shard has incompatible geometry");
  }
  const int64_t local_heads = num_heads / tp_size;
  std::vector<int64_t> grouped_shape = weight.sizes().vec();
  grouped_shape[0] = num_heads;
  grouped_shape.insert(grouped_shape.begin() + 1, 3 * head_dim);
  torch::Tensor local_grouped =
      weight.reshape(grouped_shape)
          .narrow(0, tp_rank * local_heads, local_heads)
          .contiguous();
  std::vector<int64_t> local_shape = weight.sizes().vec();
  local_shape[0] = 3 * local_heads * head_dim;
  return minimax_h3_reorder_grouped_qkv(
      local_grouped.reshape(local_shape), local_heads, head_dim);
}

inline torch::Tensor minimax_h3_tp_shard_gate_up(const torch::Tensor& weight,
                                                 int64_t tp_rank,
                                                 int64_t tp_size) {
  if (!weight.defined() || weight.dim() < 1 || weight.size(0) % 2 != 0 ||
      tp_size <= 0 || tp_rank < 0 || tp_rank >= tp_size ||
      (weight.size(0) / 2) % tp_size != 0) {
    throw std::invalid_argument(
        "MiniMax-H3 TP gate/up shard has incompatible geometry");
  }
  const int64_t full_ffn = weight.size(0) / 2;
  const int64_t local_ffn = full_ffn / tp_size;
  const torch::Tensor gate = weight.narrow(0, tp_rank * local_ffn, local_ffn);
  const torch::Tensor up =
      weight.narrow(0, full_ffn + tp_rank * local_ffn, local_ffn);
  return torch::cat({up, gate}, 0).contiguous();
}

inline void minimax_h3_synchronize_weight_load(const torch::Device& device) {
  if (device.is_cpu() || device.type() == c10::DeviceType::Meta) {
    return;
  }
  const int32_t status = Device(device).synchronize_default_stream();
  if (status != 0) {
    throw std::runtime_error(
        "MiniMax-H3 TP weight load stream synchronization failed");
  }
}

inline torch::Tensor minimax_h3_tp_fp32_all_reduce(const torch::Tensor& partial,
                                                   ProcessGroup* tp_group) {
  if (!partial.defined() ||
      (partial.scalar_type() != torch::kBFloat16 &&
       partial.scalar_type() != torch::kFloat32) ||
      tp_group == nullptr || tp_group->world_size() != 2) {
    throw std::invalid_argument(
        "MiniMax-H3 TP output reduction requires BF16/FP32 and TP2");
  }
  torch::Tensor reduced = partial.scalar_type() == torch::kFloat32
                              ? partial
                              : partial.to(torch::kFloat32);
  reduced = parallel_state::reduce(reduced, tp_group);
  return reduced.to(torch::kBFloat16);
}

inline torch::Tensor minimax_h3_rms_norm(const torch::Tensor& input,
                                         const torch::Tensor& weight,
                                         double eps) {
  if (!input.defined() || !weight.defined() || input.dim() < 1 ||
      weight.dim() != 1 || input.size(-1) != weight.size(0)) {
    throw std::invalid_argument("MiniMax-H3 RMSNorm shape mismatch");
  }
  const torch::ScalarType input_dtype = input.scalar_type();
  torch::Tensor normalized = input.to(torch::kFloat32);
  torch::Tensor variance = normalized.pow(2).mean(-1, /*keepdim=*/true);
  normalized = normalized * torch::rsqrt(variance + eps);
  normalized = normalized * weight.to(torch::kFloat32);
  return normalized.to(input_dtype);
}

inline torch::Tensor minimax_h3_rope_frequencies(
    const torch::Tensor& position_ids,
    const torch::Tensor& inv_freq) {
  if (!position_ids.defined() || position_ids.dim() != 2 ||
      position_ids.size(1) != 3 || !inv_freq.defined() || inv_freq.dim() != 1) {
    throw std::invalid_argument(
        "MiniMax-H3 RoPE expects position IDs [S,3] and inv_freq [F]");
  }
  torch::Tensor per_axis = position_ids.to(torch::kFloat32).unsqueeze(-1) *
                           inv_freq.to(torch::kFloat32).view({1, 1, -1});
  torch::Tensor half = per_axis.reshape({position_ids.size(0), -1});
  return torch::cat({half, half}, -1);
}

inline torch::Tensor minimax_h3_apply_rope(const torch::Tensor& input,
                                           const torch::Tensor& frequencies) {
  if (!input.defined() || input.dim() != 3 || !frequencies.defined() ||
      frequencies.dim() != 2 || input.size(0) != frequencies.size(0) ||
      frequencies.size(1) <= 0 || frequencies.size(1) % 2 != 0 ||
      frequencies.size(1) > input.size(2)) {
    throw std::invalid_argument("MiniMax-H3 RoPE tensor shape mismatch");
  }
  const int64_t rotary_dim = frequencies.size(1);
  const int64_t half = rotary_dim / 2;
  torch::Tensor rotary = input.narrow(-1, 0, rotary_dim);
  torch::Tensor rotated = torch::cat(
      {-rotary.narrow(-1, half, half), rotary.narrow(-1, 0, half)}, -1);
  torch::Tensor cosine =
      torch::cos(frequencies).to(input.scalar_type()).unsqueeze(1);
  torch::Tensor sine =
      torch::sin(frequencies).to(input.scalar_type()).unsqueeze(1);
  torch::Tensor output_rotary = rotary * cosine + rotated * sine;
  torch::Tensor output_pass =
      input.narrow(-1, rotary_dim, input.size(2) - rotary_dim);
  return torch::cat({output_rotary, output_pass}, -1).contiguous();
}

inline torch::Tensor minimax_h3_combined_adaln_indices(
    const torch::Tensor& inverse_timestep_indices,
    const torch::Tensor& token_tags) {
  if (!inverse_timestep_indices.defined() || !token_tags.defined() ||
      inverse_timestep_indices.dim() != 1 || token_tags.dim() != 1 ||
      inverse_timestep_indices.size(0) != token_tags.size(0) ||
      inverse_timestep_indices.scalar_type() != torch::kInt64 ||
      token_tags.scalar_type() != torch::kInt64) {
    throw std::invalid_argument(
        "MiniMax-H3 AdaLN indices require matching int64 vectors");
  }
  return inverse_timestep_indices * 3 + token_tags.clamp_min(0);
}

inline torch::Tensor minimax_h3_timestep_embedding(
    const torch::Tensor& timestep,
    int64_t embedding_size) {
  if (!timestep.defined() || timestep.dim() != 1 ||
      !timestep.is_floating_point() || embedding_size <= 0 ||
      embedding_size % 2 != 0) {
    throw std::invalid_argument(
        "MiniMax-H3 timestep embedding requires [M] and an even dimension");
  }
  const int64_t half = embedding_size / 2;
  torch::Tensor frequency =
      torch::exp(-std::log(10000.0) *
                 torch::arange(half,
                               torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(timestep.device())) /
                 static_cast<double>(half));
  torch::Tensor arguments =
      timestep.to(torch::kFloat32).view({-1, 1}) * frequency.view({1, -1});
  return torch::cat({torch::cos(arguments), torch::sin(arguments)}, -1);
}

inline torch::Tensor minimax_h3_select_adaln_parameter(
    const torch::Tensor& parameters,
    int64_t logical_parameter,
    const torch::Tensor& combined_indices) {
  if (!parameters.defined() || parameters.dim() != 4 ||
      parameters.size(1) != 3 || logical_parameter < 0 ||
      logical_parameter >= parameters.size(2) ||
      combined_indices.scalar_type() != torch::kInt64 ||
      combined_indices.dim() != 1) {
    throw std::invalid_argument(
        "MiniMax-H3 AdaLN parameter selection mismatch");
  }
  return parameters.flatten(0, 1)
      .select(1, logical_parameter)
      .index_select(0, combined_indices);
}

inline torch::Tensor minimax_h3_segmented_sdpa(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& cu_seqlens) {
  if (!query.defined() || query.dim() != 3 || key.sizes() != query.sizes() ||
      value.sizes() != query.sizes() || !cu_seqlens.defined() ||
      cu_seqlens.dim() != 1 || cu_seqlens.numel() < 2 ||
      (cu_seqlens.scalar_type() != torch::kInt32 &&
       cu_seqlens.scalar_type() != torch::kInt64)) {
    throw std::invalid_argument("MiniMax-H3 segmented SDPA shape mismatch");
  }
  torch::Tensor bounds =
      cu_seqlens.to(torch::kCPU).to(torch::kInt64).contiguous();
  const int64_t* values = bounds.data_ptr<int64_t>();
  if (values[0] != 0 || values[bounds.numel() - 1] != query.size(0)) {
    throw std::invalid_argument(
        "MiniMax-H3 cu_seqlens must cover every packed row");
  }

  torch::Tensor output = torch::empty_like(query);
  for (int64_t index = 0; index + 1 < bounds.numel(); ++index) {
    const int64_t start = values[index];
    const int64_t stop = values[index + 1];
    if (start < 0 || stop < start || stop > query.size(0)) {
      throw std::invalid_argument(
          "MiniMax-H3 cu_seqlens must be monotonic and in range");
    }
    if (start == stop) {
      continue;
    }
    const int64_t length = stop - start;
    torch::Tensor segment_query =
        query.narrow(0, start, length).transpose(0, 1).unsqueeze(0);
    torch::Tensor segment_key =
        key.narrow(0, start, length).transpose(0, 1).unsqueeze(0);
    torch::Tensor segment_value =
        value.narrow(0, start, length).transpose(0, 1).unsqueeze(0);
    torch::Tensor segment_output =
        torch::scaled_dot_product_attention(segment_query,
                                            segment_key,
                                            segment_value,
                                            torch::nullopt,
                                            /*dropout_p=*/0.0,
                                            /*is_causal=*/false);
    output.narrow(0, start, length)
        .copy_(segment_output.squeeze(0).transpose(0, 1));
  }
  return output;
}

class MiniMaxH3DenseImpl final : public torch::nn::Module {
 public:
  MiniMaxH3DenseImpl(int64_t input_size,
                     int64_t output_size,
                     bool with_bias,
                     const torch::TensorOptions& options) {
    weight_ = register_parameter(
        "weight", torch::empty({output_size, input_size}, options));
    if (with_bias) {
      bias_ = register_parameter("bias", torch::empty({output_size}, options));
    }
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    if (bias_.defined()) {
      return torch::nn::functional::linear(input, weight_, bias_);
    }
    return torch::nn::functional::linear(input, weight_);
  }

  const torch::Tensor& weight() const { return weight_; }

 private:
  torch::Tensor weight_;
  torch::Tensor bias_;
};
TORCH_MODULE(MiniMaxH3Dense);

class MiniMaxH3RMSNormImpl final : public torch::nn::Module {
 public:
  MiniMaxH3RMSNormImpl(int64_t size,
                       double eps,
                       const torch::TensorOptions& options)
      : eps_(eps) {
    weight_ = register_parameter("weight", torch::empty({size}, options));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    return minimax_h3_rms_norm(input, weight_, eps_);
  }

  void load_state_dict(const StateDict& state_dict) {
    const torch::Tensor value = state_dict.get_tensor("weight");
    if (!value.defined() || value.sizes() != weight_.sizes() ||
        value.scalar_type() != weight_.scalar_type()) {
      throw std::invalid_argument(
          "MiniMax-H3 RMSNorm checkpoint weight metadata mismatch");
    }
    torch::NoGradGuard no_grad;
    weight_.copy_(value);
    weight_loaded_ = true;
  }

  bool is_weight_loaded() const { return weight_loaded_; }
  const torch::Tensor& weight() const { return weight_; }

 private:
  torch::Tensor weight_;
  double eps_;
  bool weight_loaded_ = false;
};
TORCH_MODULE(MiniMaxH3RMSNorm);

class MiniMaxH3RopeImpl final : public torch::nn::Module {
 public:
  MiniMaxH3RopeImpl(int64_t inv_freq_len, const torch::TensorOptions& options) {
    inv_freq_ = register_buffer(
        "inv_freq",
        torch::empty({inv_freq_len}, options.dtype(torch::kFloat32)));
  }

  torch::Tensor forward(const torch::Tensor& position_ids) const {
    return minimax_h3_rope_frequencies(position_ids, inv_freq_);
  }

 private:
  torch::Tensor inv_freq_;
};
TORCH_MODULE(MiniMaxH3Rope);

class MiniMaxH3AttentionImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AttentionImpl(const MiniMaxH3C4Config& config,
                         const torch::TensorOptions& options)
      : num_heads_(config.num_attention_heads),
        head_dim_(config.attention_head_dim) {
    const int64_t inner_size = num_heads_ * head_dim_;
    const torch::TensorOptions bf16_options = options.dtype(torch::kBFloat16);
    q_norm_ = register_module(
        "q_norm",
        MiniMaxH3RMSNorm(head_dim_, config.qk_norm_eps, bf16_options));
    k_norm_ = register_module(
        "k_norm",
        MiniMaxH3RMSNorm(head_dim_, config.qk_norm_eps, bf16_options));
    qkv_proj_ = register_module("qkv_proj",
                                MiniMaxH3Dense(config.hidden_size,
                                               3 * inner_size,
                                               /*with_bias=*/false,
                                               bf16_options));
    out_proj_ = register_module("out_proj",
                                MiniMaxH3Dense(inner_size,
                                               config.hidden_size,
                                               /*with_bias=*/false,
                                               bf16_options));
  }

  torch::Tensor forward(const torch::Tensor& input,
                        const std::optional<torch::Tensor>& rope_frequencies,
                        const torch::Tensor& cu_seqlens) const {
    if (!input.defined() || input.dim() != 2 ||
        input.scalar_type() != torch::kBFloat16) {
      throw std::invalid_argument(
          "MiniMax-H3 attention input must be BF16 [T,H]");
    }
    const int64_t row_count = input.size(0);
    const int64_t inner_size = num_heads_ * head_dim_;
    const std::vector<torch::Tensor> qkv_weights =
        qkv_proj_->weight().split({inner_size, inner_size, inner_size}, 0);
    std::vector<torch::Tensor> qkv;
    qkv.reserve(3);
    for (const torch::Tensor& weight : qkv_weights) {
      qkv.emplace_back(torch::nn::functional::linear(input, weight));
    }
    torch::Tensor query =
        q_norm_->forward(qkv[0].view({row_count, num_heads_, head_dim_}));
    torch::Tensor key =
        k_norm_->forward(qkv[1].view({row_count, num_heads_, head_dim_}));
    torch::Tensor value = qkv[2].view({row_count, num_heads_, head_dim_});
    if (rope_frequencies.has_value()) {
      query = minimax_h3_apply_rope(query, *rope_frequencies);
      key = minimax_h3_apply_rope(key, *rope_frequencies);
    }
    torch::Tensor output =
        minimax_h3_segmented_sdpa(query, key, value, cu_seqlens);
    return out_proj_->forward(output.reshape({row_count, inner_size}));
  }

 private:
  int64_t num_heads_;
  int64_t head_dim_;
  MiniMaxH3RMSNorm q_norm_{nullptr};
  MiniMaxH3RMSNorm k_norm_{nullptr};
  MiniMaxH3Dense qkv_proj_{nullptr};
  MiniMaxH3Dense out_proj_{nullptr};
};
TORCH_MODULE(MiniMaxH3Attention);

class MiniMaxH3MLPImpl final : public torch::nn::Module {
 public:
  MiniMaxH3MLPImpl(const MiniMaxH3C4Config& config,
                   const torch::TensorOptions& options) {
    const torch::TensorOptions bf16_options = options.dtype(torch::kBFloat16);
    fc1_ = register_module("fc1",
                           MiniMaxH3Dense(config.hidden_size,
                                          2 * config.ffn_hidden_size,
                                          /*with_bias=*/false,
                                          bf16_options));
    fc2_ = register_module("fc2",
                           MiniMaxH3Dense(config.ffn_hidden_size,
                                          config.hidden_size,
                                          /*with_bias=*/false,
                                          bf16_options));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    std::vector<torch::Tensor> up_gate = fc1_->forward(input).chunk(2, -1);
    return fc2_->forward(up_gate[0] * torch::silu(up_gate[1]));
  }

 private:
  MiniMaxH3Dense fc1_{nullptr};
  MiniMaxH3Dense fc2_{nullptr};
};
TORCH_MODULE(MiniMaxH3MLP);

class MiniMaxH3TokenRefinerBlockImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TokenRefinerBlockImpl(const MiniMaxH3C4Config& config,
                                 const torch::TensorOptions& options) {
    const torch::TensorOptions bf16_options = options.dtype(torch::kBFloat16);
    norm1_ = register_module(
        "norm1",
        MiniMaxH3RMSNorm(config.hidden_size, config.norm_eps, bf16_options));
    norm2_ = register_module(
        "norm2",
        MiniMaxH3RMSNorm(config.hidden_size, config.norm_eps, bf16_options));
    attn_ = register_module("attn", MiniMaxH3Attention(config, options));
    mlp_ = register_module("mlp", MiniMaxH3MLP(config, options));
  }

  MiniMaxH3ResidualBranchTrace forward(const torch::Tensor& input,
                                       const torch::Tensor& cu_seqlens) const {
    MiniMaxH3ResidualBranchTrace trace;
    trace.attention_delta =
        attn_->forward(norm1_->forward(input), std::nullopt, cu_seqlens);
    torch::Tensor after_attention = input + trace.attention_delta;
    trace.mlp_delta = mlp_->forward(norm2_->forward(after_attention));
    trace.output = after_attention + trace.mlp_delta;
    return trace;
  }

  torch::Tensor forward_output_only(const torch::Tensor& input,
                                    const torch::Tensor& cu_seqlens) const {
    torch::Tensor after_attention;
    {
      torch::Tensor attention_delta =
          attn_->forward(norm1_->forward(input), std::nullopt, cu_seqlens);
      after_attention = input + attention_delta;
    }
    return after_attention + mlp_->forward(norm2_->forward(after_attention));
  }

 private:
  MiniMaxH3RMSNorm norm1_{nullptr};
  MiniMaxH3RMSNorm norm2_{nullptr};
  MiniMaxH3Attention attn_{nullptr};
  MiniMaxH3MLP mlp_{nullptr};
};
TORCH_MODULE(MiniMaxH3TokenRefinerBlock);

class MiniMaxH3TokenRefinerImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TokenRefinerImpl(const MiniMaxH3C4Config& config,
                            const torch::TensorOptions& options) {
    blocks_ = register_module("blocks", torch::nn::ModuleList());
    block_layers_.reserve(static_cast<size_t>(config.num_refiner_layers));
    for (int64_t index = 0; index < config.num_refiner_layers; ++index) {
      MiniMaxH3TokenRefinerBlock block(config, options);
      blocks_->push_back(block);
      block_layers_.emplace_back(std::move(block));
    }
    final_norm_ =
        register_module("final_norm",
                        MiniMaxH3RMSNorm(config.hidden_size,
                                         config.final_norm_eps,
                                         options.dtype(torch::kBFloat16)));
  }

  std::pair<torch::Tensor, std::vector<MiniMaxH3ResidualBranchTrace>> forward(
      const torch::Tensor& input,
      const torch::Tensor& cu_seqlens) const {
    torch::Tensor hidden = input;
    std::vector<MiniMaxH3ResidualBranchTrace> traces;
    traces.reserve(block_layers_.size());
    for (const MiniMaxH3TokenRefinerBlock& block : block_layers_) {
      MiniMaxH3ResidualBranchTrace trace = block->forward(hidden, cu_seqlens);
      hidden = trace.output;
      traces.emplace_back(std::move(trace));
    }
    return {final_norm_->forward(hidden), std::move(traces)};
  }

  torch::Tensor forward_output_only(const torch::Tensor& input,
                                    const torch::Tensor& cu_seqlens) const {
    torch::Tensor hidden = input;
    for (const MiniMaxH3TokenRefinerBlock& block : block_layers_) {
      hidden = block->forward_output_only(hidden, cu_seqlens);
    }
    return final_norm_->forward(hidden);
  }

 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<MiniMaxH3TokenRefinerBlock> block_layers_;
  MiniMaxH3RMSNorm final_norm_{nullptr};
};
TORCH_MODULE(MiniMaxH3TokenRefiner);

class MiniMaxH3AdaLNProjectionImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AdaLNProjectionImpl(int64_t time_embed_dim,
                               int64_t hidden_size,
                               int64_t parameter_count,
                               int64_t modality_count,
                               const torch::TensorOptions& options)
      : hidden_size_(hidden_size),
        parameter_count_(parameter_count),
        modality_count_(modality_count) {
    linear_ = register_module(
        "linear",
        MiniMaxH3Dense(time_embed_dim,
                       hidden_size * parameter_count * modality_count,
                       /*with_bias=*/true,
                       options.dtype(torch::kBFloat16)));
  }

  torch::Tensor forward(const torch::Tensor& time_embedding) const {
    if (!time_embedding.defined() || time_embedding.dim() != 2 ||
        time_embedding.scalar_type() != torch::kFloat32) {
      throw std::invalid_argument(
          "MiniMax-H3 AdaLN time embedding must be FP32 [M,D]");
    }
    torch::Tensor projected =
        linear_->forward(torch::silu(time_embedding).to(torch::kBFloat16));
    return projected.view({time_embedding.size(0),
                           modality_count_,
                           parameter_count_,
                           hidden_size_});
  }

 private:
  int64_t hidden_size_;
  int64_t parameter_count_;
  int64_t modality_count_;
  MiniMaxH3Dense linear_{nullptr};
};
TORCH_MODULE(MiniMaxH3AdaLNProjection);

class MiniMaxH3DiTBlockImpl final : public torch::nn::Module {
 public:
  MiniMaxH3DiTBlockImpl(const MiniMaxH3C4Config& config,
                        const torch::TensorOptions& options) {
    const torch::TensorOptions bf16_options = options.dtype(torch::kBFloat16);
    norm1_ = register_module(
        "norm1",
        MiniMaxH3RMSNorm(config.hidden_size, config.norm_eps, bf16_options));
    norm2_ = register_module(
        "norm2",
        MiniMaxH3RMSNorm(config.hidden_size, config.norm_eps, bf16_options));
    attn_ = register_module("attn", MiniMaxH3Attention(config, options));
    mlp_ = register_module("mlp", MiniMaxH3MLP(config, options));
    adaln_proj_ =
        register_module("adaln_proj",
                        MiniMaxH3AdaLNProjection(config.time_embed_dim,
                                                 config.hidden_size,
                                                 /*parameter_count=*/6,
                                                 /*modality_count=*/3,
                                                 options));
  }

  MiniMaxH3ResidualBranchTrace forward(const torch::Tensor& input,
                                       const torch::Tensor& time_embedding,
                                       const torch::Tensor& combined_indices,
                                       const torch::Tensor& rope_frequencies,
                                       const torch::Tensor& cu_seqlens) const {
    torch::Tensor parameters = adaln_proj_->forward(time_embedding);
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
    trace.norm1_output = norm1_->forward(input);
    trace.attention_input = trace.norm1_output * (scale_msa + 1.0) + shift_msa;
    trace.attention_output =
        attn_->forward(trace.attention_input, rope_frequencies, cu_seqlens);
    trace.attention_delta = gate_msa * trace.attention_output;
    torch::Tensor after_attention = input + trace.attention_delta;

    trace.norm2_output = norm2_->forward(after_attention);
    trace.mlp_input = trace.norm2_output * (scale_mlp + 1.0) + shift_mlp;
    trace.mlp_output = mlp_->forward(trace.mlp_input);
    trace.mlp_delta = gate_mlp * trace.mlp_output;
    trace.output = after_attention + trace.mlp_delta;
    return trace;
  }

  torch::Tensor forward_output_only(const torch::Tensor& input,
                                    const torch::Tensor& time_embedding,
                                    const torch::Tensor& combined_indices,
                                    const torch::Tensor& rope_frequencies,
                                    const torch::Tensor& cu_seqlens) const {
    torch::Tensor parameters = adaln_proj_->forward(time_embedding);
    torch::Tensor after_attention;
    {
      torch::Tensor shift_msa =
          minimax_h3_select_adaln_parameter(parameters, 0, combined_indices);
      torch::Tensor scale_msa =
          minimax_h3_select_adaln_parameter(parameters, 1, combined_indices);
      torch::Tensor gate_msa =
          minimax_h3_select_adaln_parameter(parameters, 2, combined_indices);
      torch::Tensor attention_input =
          norm1_->forward(input) * (scale_msa + 1.0) + shift_msa;
      torch::Tensor attention_output =
          attn_->forward(attention_input, rope_frequencies, cu_seqlens);
      after_attention = input + gate_msa * attention_output;
    }
    torch::Tensor shift_mlp =
        minimax_h3_select_adaln_parameter(parameters, 3, combined_indices);
    torch::Tensor scale_mlp =
        minimax_h3_select_adaln_parameter(parameters, 4, combined_indices);
    torch::Tensor gate_mlp =
        minimax_h3_select_adaln_parameter(parameters, 5, combined_indices);
    torch::Tensor mlp_input =
        norm2_->forward(after_attention) * (scale_mlp + 1.0) + shift_mlp;
    return after_attention + gate_mlp * mlp_->forward(mlp_input);
  }

 private:
  MiniMaxH3RMSNorm norm1_{nullptr};
  MiniMaxH3RMSNorm norm2_{nullptr};
  MiniMaxH3Attention attn_{nullptr};
  MiniMaxH3MLP mlp_{nullptr};
  MiniMaxH3AdaLNProjection adaln_proj_{nullptr};
};
TORCH_MODULE(MiniMaxH3DiTBlock);

struct MiniMaxH3TPAttentionQKV {
  torch::Tensor query;
  torch::Tensor key;
  torch::Tensor value;
};

class MiniMaxH3TPAttentionImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TPAttentionImpl(const MiniMaxH3C4Config& config,
                           ProcessGroup* tp_group,
                           const torch::TensorOptions& options,
                           bool dense_output_projection = false)
      : tp_group_(tp_group),
        num_heads_(config.num_attention_heads),
        head_dim_(config.attention_head_dim),
        dense_output_projection_(dense_output_projection) {
    validate_tp_group(tp_group_);
    if (num_heads_ % tp_group_->world_size() != 0) {
      throw std::invalid_argument(
          "MiniMax-H3 TP Attention heads must divide TP size");
    }
    local_heads_ = num_heads_ / tp_group_->world_size();
    const int64_t inner = num_heads_ * head_dim_;
    const torch::TensorOptions bf16 = options.dtype(torch::kBFloat16);
    if (dense_output_projection_) {
      out_projection_weight_ = register_buffer(
          "out_weight_dense", torch::empty({config.hidden_size, inner}, bf16));
    } else {
      out_projection_weight_ = register_buffer(
          "out_weight_fp32",
          torch::empty({config.hidden_size, inner / tp_group_->world_size()},
                       options.dtype(torch::kFloat32)));
    }
    q_norm_ = register_module(
        "q_norm", MiniMaxH3RMSNorm(head_dim_, config.qk_norm_eps, bf16));
    k_norm_ = register_module(
        "k_norm", MiniMaxH3RMSNorm(head_dim_, config.qk_norm_eps, bf16));
    qkv_proj_ =
        register_module("qkv_proj",
                        layer::ColumnParallelLinear(config.hidden_size,
                                                    3 * inner,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    QuantArgs{},
                                                    tp_group_,
                                                    bf16));
    if (!dense_output_projection_) {
      out_proj_ = register_module(
          "out_proj",
          layer::RowParallelLinear(inner,
                                   config.hidden_size,
                                   /*bias=*/false,
                                   /*input_is_parallelized=*/true,
                                   /*enable_result_reduction=*/false,
                                   QuantArgs{},
                                   tp_group_,
                                   bf16));
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    q_norm_->load_state_dict(state_dict.get_dict_with_prefix("q_norm."));
    k_norm_->load_state_dict(state_dict.get_dict_with_prefix("k_norm."));

    const torch::Tensor grouped = state_dict.get_tensor("qkv_proj.weight");
    if (!grouped.defined()) {
      throw std::invalid_argument(
          "MiniMax-H3 TP Attention is missing grouped QKV weight");
    }
    StateDict reordered_qkv(
        {{"weight",
          minimax_h3_reorder_grouped_qkv(grouped, num_heads_, head_dim_)}});
    const int64_t local_inner = local_heads_ * head_dim_;
    qkv_proj_->load_state_dict(reordered_qkv,
                               /*shard_tensor_count=*/3,
                               {local_inner, local_inner, local_inner});
    const torch::Tensor out_weight = state_dict.get_tensor("out_proj.weight");
    if (!out_weight.defined() ||
        out_weight.sizes() !=
            torch::IntArrayRef(
                {out_projection_weight_.size(0), num_heads_ * head_dim_}) ||
        out_weight.scalar_type() != torch::kBFloat16) {
      throw std::invalid_argument(
          "MiniMax-H3 TP Attention dense output weight mismatch");
    }
    if (dense_output_projection_) {
      out_projection_weight_.copy_(out_weight);
    } else {
      out_proj_->load_state_dict(state_dict.get_dict_with_prefix("out_proj."));
      out_projection_weight_.copy_(out_proj_->weight());
    }
    minimax_h3_synchronize_weight_load(qkv_proj_->weight().device());
  }

  torch::Tensor forward(const torch::Tensor& input,
                        const std::optional<torch::Tensor>& rope_frequencies,
                        const torch::Tensor& cu_seqlens) {
    torch::Tensor local_heads =
        forward_local_heads(input, rope_frequencies, cu_seqlens);
    if (dense_output_projection_) {
      const torch::Tensor full_heads =
          parallel_state::gather(local_heads, tp_group_, /*dim=*/1);
      return project_dense_output(full_heads);
    }
    torch::Tensor partial = project_local_output_fp32(local_heads);
    return minimax_h3_tp_fp32_all_reduce(partial, tp_group_);
  }

  torch::Tensor forward_local_heads(
      const torch::Tensor& input,
      const std::optional<torch::Tensor>& rope_frequencies,
      const torch::Tensor& cu_seqlens) {
    MiniMaxH3TPAttentionQKV qkv = project_local_qkv(input, rope_frequencies);
    return minimax_h3_segmented_sdpa(qkv.query, qkv.key, qkv.value, cu_seqlens);
  }

  MiniMaxH3TPAttentionQKV project_local_qkv(
      const torch::Tensor& input,
      const std::optional<torch::Tensor>& rope_frequencies) {
    verify_loaded_weights();
    if (!input.defined() || input.dim() != 2 ||
        input.scalar_type() != torch::kBFloat16) {
      throw std::invalid_argument(
          "MiniMax-H3 TP Attention input must be BF16 [T,H]");
    }
    const int64_t rows = input.size(0);
    const int64_t local_inner = local_heads_ * head_dim_;
    const std::vector<torch::Tensor> qkv = qkv_proj_->forward(input).split(
        {local_inner, local_inner, local_inner}, -1);
    torch::Tensor query =
        q_norm_->forward(qkv[0].view({rows, local_heads_, head_dim_}));
    torch::Tensor key =
        k_norm_->forward(qkv[1].view({rows, local_heads_, head_dim_}));
    torch::Tensor value =
        qkv[2].view({rows, local_heads_, head_dim_}).contiguous();
    if (rope_frequencies.has_value()) {
      query = minimax_h3_apply_rope(query, *rope_frequencies);
      key = minimax_h3_apply_rope(key, *rope_frequencies);
    }
    return {.query = std::move(query),
            .key = std::move(key),
            .value = std::move(value)};
  }

  torch::Tensor project_local_query(
      const torch::Tensor& input,
      const std::optional<torch::Tensor>& rope_frequencies) {
    validate_projection_input(input);
    const int64_t rows = input.size(0);
    const int64_t local_inner = local_heads_ * head_dim_;
    torch::Tensor query = q_norm_->forward(
        torch::nn::functional::linear(
            input, qkv_proj_->weight().narrow(0, 0, local_inner))
            .view({rows, local_heads_, head_dim_}));
    if (rope_frequencies.has_value()) {
      query = minimax_h3_apply_rope(query, *rope_frequencies);
    }
    return query;
  }

  torch::Tensor project_local_key(
      const torch::Tensor& input,
      const std::optional<torch::Tensor>& rope_frequencies) {
    validate_projection_input(input);
    const int64_t rows = input.size(0);
    const int64_t local_inner = local_heads_ * head_dim_;
    torch::Tensor key = k_norm_->forward(
        torch::nn::functional::linear(
            input, qkv_proj_->weight().narrow(0, local_inner, local_inner))
            .view({rows, local_heads_, head_dim_}));
    if (rope_frequencies.has_value()) {
      key = minimax_h3_apply_rope(key, *rope_frequencies);
    }
    return key;
  }

  torch::Tensor project_local_value(const torch::Tensor& input) {
    validate_projection_input(input);
    const int64_t rows = input.size(0);
    const int64_t local_inner = local_heads_ * head_dim_;
    return torch::nn::functional::linear(
               input,
               qkv_proj_->weight().narrow(0, 2 * local_inner, local_inner))
        .view({rows, local_heads_, head_dim_})
        .contiguous();
  }

  torch::Tensor project_local_output(const torch::Tensor& local_heads) {
    if (!out_proj_) {
      throw std::logic_error(
          "MiniMax-H3 TP Attention sharded output projection is not allocated "
          "because the dense output projection owns this block");
    }
    if (!local_heads.defined() || local_heads.dim() != 3 ||
        local_heads.size(1) != local_heads_ ||
        local_heads.size(2) != head_dim_) {
      throw std::invalid_argument(
          "MiniMax-H3 TP local Attention output shape mismatch");
    }
    return out_proj_->forward(
        local_heads.reshape({local_heads.size(0), local_heads_ * head_dim_}));
  }

  torch::Tensor project_local_output_fp32(const torch::Tensor& local_heads) {
    if (dense_output_projection_ || !out_projection_weight_.defined() ||
        out_projection_weight_.scalar_type() != torch::kFloat32 ||
        out_projection_weight_.device() != local_heads.device() ||
        out_projection_weight_.sizes() != out_proj_->weight().sizes()) {
      throw std::logic_error(
          "MiniMax-H3 TP FP32 Attention projection weight is not loaded");
    }
    if (!local_heads.defined() || local_heads.dim() != 3 ||
        local_heads.size(1) != local_heads_ ||
        local_heads.size(2) != head_dim_) {
      throw std::invalid_argument(
          "MiniMax-H3 TP local Attention output shape mismatch");
    }
    return torch::nn::functional::linear(
        local_heads.reshape({local_heads.size(0), local_heads_ * head_dim_})
            .to(torch::kFloat32),
        out_projection_weight_);
  }

  torch::Tensor project_dense_output(const torch::Tensor& full_heads) {
    if (!dense_output_projection_ || !full_heads.defined() ||
        full_heads.dim() != 3 || full_heads.size(1) != num_heads_ ||
        full_heads.size(2) != head_dim_ ||
        full_heads.scalar_type() != torch::kBFloat16 ||
        full_heads.device() != out_projection_weight_.device() ||
        out_projection_weight_.scalar_type() != torch::kBFloat16) {
      throw std::invalid_argument(
          "MiniMax-H3 dense Attention output shape mismatch");
    }
    return torch::nn::functional::linear(
        full_heads.reshape({full_heads.size(0), num_heads_ * head_dim_}),
        out_projection_weight_);
  }

  void verify_loaded_weights() const {
    if (!q_norm_->is_weight_loaded() || !k_norm_->is_weight_loaded() ||
        !qkv_proj_->is_weight_loaded() || !out_projection_weight_.defined() ||
        (out_proj_ && !out_proj_->is_weight_loaded())) {
      throw std::logic_error(
          "MiniMax-H3 TP Attention weights are not completely loaded");
    }
  }

  int64_t local_heads() const { return local_heads_; }
  bool has_sharded_out_proj() const { return static_cast<bool>(out_proj_); }
  const torch::Tensor& q_norm_weight() const { return q_norm_->weight(); }
  const torch::Tensor& k_norm_weight() const { return k_norm_->weight(); }
  torch::Tensor qkv_weight() const { return qkv_proj_->weight(); }
  torch::Tensor out_weight() const {
    if (!out_proj_) {
      throw std::logic_error(
          "MiniMax-H3 TP Attention sharded output projection is not allocated "
          "because the dense output projection owns this block");
    }
    return out_proj_->weight();
  }

 private:
  void validate_projection_input(const torch::Tensor& input) const {
    verify_loaded_weights();
    if (!input.defined() || input.dim() != 2 ||
        input.scalar_type() != torch::kBFloat16) {
      throw std::invalid_argument(
          "MiniMax-H3 TP Attention input must be BF16 [T,H]");
    }
  }

  static void validate_tp_group(ProcessGroup* tp_group) {
    if (tp_group == nullptr || tp_group->world_size() != 2 ||
        tp_group->rank() < 0 || tp_group->rank() >= 2) {
      throw std::invalid_argument(
          "MiniMax-H3 C8B Attention requires an exact TP2 process group");
    }
  }

  ProcessGroup* tp_group_;
  int64_t num_heads_;
  int64_t head_dim_;
  int64_t local_heads_ = 0;
  bool dense_output_projection_ = false;
  MiniMaxH3RMSNorm q_norm_{nullptr};
  MiniMaxH3RMSNorm k_norm_{nullptr};
  layer::ColumnParallelLinear qkv_proj_{nullptr};
  layer::RowParallelLinear out_proj_{nullptr};
  torch::Tensor out_projection_weight_;
};
TORCH_MODULE(MiniMaxH3TPAttention);

class MiniMaxH3TPMLPImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TPMLPImpl(const MiniMaxH3C4Config& config,
                     ProcessGroup* tp_group,
                     const torch::TensorOptions& options,
                     bool allocate_sharded_fc2 = true)
      : tp_group_(tp_group),
        ffn_hidden_size_(config.ffn_hidden_size),
        allocate_sharded_fc2_(allocate_sharded_fc2) {
    if (tp_group_ == nullptr || tp_group_->world_size() != 2 ||
        ffn_hidden_size_ % 2 != 0) {
      throw std::invalid_argument(
          "MiniMax-H3 C8B MLP requires TP2-divisible FFN geometry");
    }
    local_ffn_ = ffn_hidden_size_ / 2;
    const torch::TensorOptions bf16 = options.dtype(torch::kBFloat16);
    fc1_ = register_module("fc1",
                           layer::ColumnParallelLinear(config.hidden_size,
                                                       2 * ffn_hidden_size_,
                                                       /*bias=*/false,
                                                       /*gather_output=*/false,
                                                       QuantArgs{},
                                                       tp_group_,
                                                       bf16));
    if (allocate_sharded_fc2_) {
      fc2_ = register_module(
          "fc2",
          layer::RowParallelLinear(ffn_hidden_size_,
                                   config.hidden_size,
                                   /*bias=*/false,
                                   /*input_is_parallelized=*/true,
                                   /*enable_result_reduction=*/false,
                                   QuantArgs{},
                                   tp_group_,
                                   bf16));
    }
  }

  void load_state_dict(const StateDict& state_dict) {
    const torch::Tensor gate_up = state_dict.get_tensor("fc1.weight");
    if (!gate_up.defined()) {
      throw std::invalid_argument("MiniMax-H3 TP MLP is missing FC1 weight");
    }
    StateDict reordered_fc1(
        {{"weight", minimax_h3_reorder_gate_up_to_up_gate(gate_up)}});
    fc1_->load_state_dict(
        reordered_fc1, /*shard_tensor_count=*/2, {local_ffn_, local_ffn_});
    if (fc2_) {
      fc2_->load_state_dict(state_dict.get_dict_with_prefix("fc2."));
    }
    minimax_h3_synchronize_weight_load(fc1_->weight().device());
  }

  torch::Tensor forward(const torch::Tensor& input) {
    if (!fc2_) {
      throw std::logic_error(
          "MiniMax-H3 TP MLP sharded FC2 is not allocated because the dense "
          "FC2 weight owns this block");
    }
    verify_loaded_weights();
    const std::vector<torch::Tensor> up_gate =
        fc1_->forward(input).chunk(2, -1);
    torch::Tensor partial = fc2_->forward(up_gate[0] * torch::silu(up_gate[1]));
    return minimax_h3_tp_fp32_all_reduce(partial, tp_group_);
  }

  void verify_loaded_weights() const {
    if (!fc1_->is_weight_loaded() || (fc2_ && !fc2_->is_weight_loaded())) {
      throw std::logic_error(
          "MiniMax-H3 TP MLP weights are not completely loaded");
    }
  }

  bool has_sharded_fc2() const { return static_cast<bool>(fc2_); }
  torch::Tensor fc1_weight() const { return fc1_->weight(); }
  torch::Tensor fc2_weight() const {
    if (!fc2_) {
      throw std::logic_error(
          "MiniMax-H3 TP MLP sharded FC2 is not allocated because the dense "
          "FC2 weight owns this block");
    }
    return fc2_->weight();
  }

 private:
  ProcessGroup* tp_group_;
  int64_t ffn_hidden_size_;
  int64_t local_ffn_ = 0;
  bool allocate_sharded_fc2_ = true;
  layer::ColumnParallelLinear fc1_{nullptr};
  layer::RowParallelLinear fc2_{nullptr};
};
TORCH_MODULE(MiniMaxH3TPMLP);

class MiniMaxH3TPAdaLNProjectionImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TPAdaLNProjectionImpl(const MiniMaxH3C4Config& config,
                                 ProcessGroup* tp_group,
                                 const torch::TensorOptions& options)
      : hidden_size_(config.hidden_size),
        time_embed_dim_(config.time_embed_dim),
        tp_group_(tp_group) {
    if (tp_group == nullptr || tp_group->world_size() != 2 ||
        (18 * hidden_size_) % 2 != 0) {
      throw std::invalid_argument(
          "MiniMax-H3 C8B AdaLN requires TP2-divisible output geometry");
    }
    linear_ = register_module(
        "linear",
        layer::ColumnParallelLinear(config.time_embed_dim,
                                    18 * hidden_size_,
                                    /*bias=*/true,
                                    /*gather_output=*/false,
                                    QuantArgs{},
                                    tp_group,
                                    options.dtype(torch::kBFloat16)));
  }

  void load_state_dict(const StateDict& state_dict) {
    const torch::Tensor weight = state_dict.get_tensor("linear.weight");
    const torch::Tensor bias = state_dict.get_tensor("linear.bias");
    if (!weight.defined() ||
        weight.sizes() !=
            torch::IntArrayRef({18 * hidden_size_, time_embed_dim_}) ||
        weight.scalar_type() != torch::kBFloat16 || !bias.defined() ||
        bias.sizes() != torch::IntArrayRef({18 * hidden_size_}) ||
        bias.scalar_type() != torch::kBFloat16) {
      throw std::invalid_argument(
          "MiniMax-H3 TP AdaLN checkpoint weight/bias metadata mismatch");
    }
    linear_->load_state_dict(state_dict.get_dict_with_prefix("linear."));
    minimax_h3_synchronize_weight_load(linear_->weight().device());
    loaded_ = true;
  }

  torch::Tensor forward_local(const torch::Tensor& time_embedding) {
    if (!loaded_ || !linear_->is_weight_loaded()) {
      throw std::logic_error("MiniMax-H3 TP AdaLN weights are not loaded");
    }
    if (!time_embedding.defined() || time_embedding.dim() != 2 ||
        time_embedding.scalar_type() != torch::kFloat32) {
      throw std::invalid_argument(
          "MiniMax-H3 TP AdaLN time embedding must be FP32 [M,D]");
    }
    return linear_->forward(torch::silu(time_embedding).to(torch::kBFloat16));
  }

  torch::Tensor forward(const torch::Tensor& time_embedding) {
    torch::Tensor projected =
        parallel_state::gather(forward_local(time_embedding), tp_group_, -1);
    return projected.view({time_embedding.size(0), 3, 6, hidden_size_});
  }

  torch::Tensor weight() const { return linear_->weight(); }
  std::optional<torch::Tensor> bias() const { return linear_->bias(); }
  bool is_loaded() const { return loaded_ && linear_->is_weight_loaded(); }

 private:
  int64_t hidden_size_;
  int64_t time_embed_dim_;
  ProcessGroup* tp_group_;
  layer::ColumnParallelLinear linear_{nullptr};
  bool loaded_ = false;
};
TORCH_MODULE(MiniMaxH3TPAdaLNProjection);

// The communicator owns tp_group and must outlive this inference-only module.
class MiniMaxH3TPDiTBlockImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TPDiTBlockImpl(const MiniMaxH3C4Config& config,
                          ProcessGroup* tp_group,
                          const torch::TensorOptions& options,
                          bool dense_output_projection = false,
                          bool allocate_sharded_fc2 = true)
      : config_(config), tp_group_(tp_group) {
    if (tp_group_ == nullptr || tp_group_->world_size() != 2) {
      throw std::invalid_argument(
          "MiniMax-H3 C8B block requires an exact TP2 process group");
    }
    const torch::TensorOptions bf16 = options.dtype(torch::kBFloat16);
    norm1_ = register_module(
        "norm1", MiniMaxH3RMSNorm(config.hidden_size, config.norm_eps, bf16));
    norm2_ = register_module(
        "norm2", MiniMaxH3RMSNorm(config.hidden_size, config.norm_eps, bf16));
    attn_ = register_module(
        "attn",
        MiniMaxH3TPAttention(
            config, tp_group_, options, dense_output_projection));
    mlp_ = register_module(
        "mlp",
        MiniMaxH3TPMLP(config,
                       tp_group_,
                       options,
                       /*allocate_sharded_fc2=*/allocate_sharded_fc2));
    adaln_proj_ = register_module(
        "adaln_proj", MiniMaxH3TPAdaLNProjection(config, tp_group_, options));
  }

  void load_source_weights(
      const std::vector<std::unique_ptr<StateDict>>& shards,
      int64_t layer_index) {
    if (loaded_) {
      throw std::logic_error(
          "MiniMax-H3 TP block source weights are already loaded");
    }
    if (layer_index < 0 || layer_index >= 50) {
      throw std::invalid_argument(
          "MiniMax-H3 TP block layer index must be in [0,50)");
    }
    const std::string prefix = "blocks." + std::to_string(layer_index) + ".";
    std::unordered_map<std::string, torch::Tensor> block_tensors;
    for (const std::string& suffix : source_suffixes()) {
      const std::string name = prefix + suffix;
      torch::Tensor value;
      for (const std::unique_ptr<StateDict>& shard : shards) {
        if (shard == nullptr) {
          throw std::invalid_argument(
              "MiniMax-H3 TP block source contains a null shard");
        }
        const torch::Tensor candidate = shard->get_tensor(name);
        if (!candidate.defined()) {
          continue;
        }
        if (value.defined()) {
          throw std::invalid_argument(
              "MiniMax-H3 TP block source tensor is duplicated: `" + name +
              "`");
        }
        value = candidate;
      }
      if (!value.defined()) {
        throw std::invalid_argument(
            "MiniMax-H3 TP block source tensor is missing: `" + name + "`");
      }
      block_tensors.emplace(suffix, value);
    }
    StateDict block_state(std::move(block_tensors));
    load_state_dict(block_state);
    loaded_ = true;
    verify_loaded_weights();
  }

  void load_state_dict(const StateDict& state_dict) {
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
    attn_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
    adaln_proj_->load_state_dict(
        state_dict.get_dict_with_prefix("adaln_proj."));
    loaded_ = true;
  }

  MiniMaxH3ResidualBranchTrace forward(const torch::Tensor& input,
                                       const torch::Tensor& time_embedding,
                                       const torch::Tensor& combined_indices,
                                       const torch::Tensor& rope_frequencies,
                                       const torch::Tensor& cu_seqlens) {
    verify_loaded_weights();
    torch::Tensor parameters = adaln_proj_->forward(time_embedding);
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
    trace.norm1_output = norm1_->forward(input);
    trace.attention_input = trace.norm1_output * (scale_msa + 1.0) + shift_msa;
    trace.attention_output =
        attn_->forward(trace.attention_input, rope_frequencies, cu_seqlens);
    trace.attention_delta = gate_msa * trace.attention_output;
    torch::Tensor after_attention = input + trace.attention_delta;
    trace.norm2_output = norm2_->forward(after_attention);
    trace.mlp_input = trace.norm2_output * (scale_mlp + 1.0) + shift_mlp;
    trace.mlp_output = mlp_->forward(trace.mlp_input);
    trace.mlp_delta = gate_mlp * trace.mlp_output;
    trace.output = after_attention + trace.mlp_delta;
    return trace;
  }

  void verify_loaded_weights() const {
    if (!loaded_ || !norm1_->is_weight_loaded() ||
        !norm2_->is_weight_loaded() || !adaln_proj_->is_loaded()) {
      throw std::logic_error(
          "MiniMax-H3 TP block weights are not completely loaded");
    }
    attn_->verify_loaded_weights();
    mlp_->verify_loaded_weights();
  }

  MiniMaxH3TPAttention attention() const { return attn_; }
  MiniMaxH3TPMLP mlp() const { return mlp_; }
  MiniMaxH3TPAdaLNProjection adaln_projection() const { return adaln_proj_; }
  const torch::Tensor& norm1_weight() const { return norm1_->weight(); }
  const torch::Tensor& norm2_weight() const { return norm2_->weight(); }

 private:
  static const std::array<std::string, 10>& source_suffixes() {
    static const std::array<std::string, 10> kSuffixes = {
        "norm1.weight",
        "norm2.weight",
        "attn.q_norm.weight",
        "attn.k_norm.weight",
        "attn.qkv_proj.weight",
        "attn.out_proj.weight",
        "mlp.fc1.weight",
        "mlp.fc2.weight",
        "adaln_proj.linear.weight",
        "adaln_proj.linear.bias",
    };
    return kSuffixes;
  }

  MiniMaxH3C4Config config_;
  ProcessGroup* tp_group_;
  MiniMaxH3RMSNorm norm1_{nullptr};
  MiniMaxH3RMSNorm norm2_{nullptr};
  MiniMaxH3TPAttention attn_{nullptr};
  MiniMaxH3TPMLP mlp_{nullptr};
  MiniMaxH3TPAdaLNProjection adaln_proj_{nullptr};
  bool loaded_ = false;
};
TORCH_MODULE(MiniMaxH3TPDiTBlock);

class MiniMaxH3TimeEmbedderImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TimeEmbedderImpl(const MiniMaxH3C4Config& config,
                            const torch::TensorOptions& options)
      : frequency_embedding_size_(config.timestep_input_dim) {
    const torch::TensorOptions fp32_options = options.dtype(torch::kFloat32);
    proj_in_ = register_module("proj_in",
                               MiniMaxH3Dense(config.timestep_input_dim,
                                              config.time_embed_hidden_size,
                                              /*with_bias=*/true,
                                              fp32_options));
    proj_out_ = register_module("proj_out",
                                MiniMaxH3Dense(config.time_embed_hidden_size,
                                               config.time_embed_dim,
                                               /*with_bias=*/true,
                                               fp32_options));
  }

  torch::Tensor forward(const torch::Tensor& timestep) const {
    if (!timestep.defined() || timestep.dim() != 1 ||
        !timestep.is_floating_point()) {
      throw std::invalid_argument(
          "MiniMax-H3 timestep must be a floating-point vector");
    }
    if (!torch::logical_and(timestep >= 0, timestep <= 1).all().item<bool>()) {
      throw std::invalid_argument("MiniMax-H3 timestep must be in [0,1]");
    }
    torch::Tensor sinusoidal =
        minimax_h3_timestep_embedding(timestep, frequency_embedding_size_);
    return proj_out_->forward(torch::silu(proj_in_->forward(sinusoidal)));
  }

 private:
  int64_t frequency_embedding_size_;
  MiniMaxH3Dense proj_in_{nullptr};
  MiniMaxH3Dense proj_out_{nullptr};
};
TORCH_MODULE(MiniMaxH3TimeEmbedder);

class MiniMaxH3FinalLayerImpl final : public torch::nn::Module {
 public:
  MiniMaxH3FinalLayerImpl(const MiniMaxH3C4Config& config,
                          const torch::TensorOptions& options) {
    norm_ = register_module("norm",
                            MiniMaxH3RMSNorm(config.hidden_size,
                                             config.final_norm_eps,
                                             options.dtype(torch::kBFloat16)));
    adaln_proj_ =
        register_module("adaln_proj",
                        MiniMaxH3AdaLNProjection(config.time_embed_dim,
                                                 config.hidden_size,
                                                 /*parameter_count=*/2,
                                                 /*modality_count=*/1,
                                                 options));
    video_out_ =
        register_module("video_out",
                        MiniMaxH3Dense(config.hidden_size,
                                       config.video_patch_dim,
                                       /*with_bias=*/true,
                                       options.dtype(torch::kFloat32)));
    audio_out_ =
        register_module("audio_out",
                        MiniMaxH3Dense(config.hidden_size,
                                       config.audio_dim,
                                       /*with_bias=*/true,
                                       options.dtype(torch::kFloat32)));
  }

  MiniMaxH3FinalOutput forward(const torch::Tensor& input,
                               const torch::Tensor& time_embedding,
                               const torch::Tensor& inverse_indices,
                               const H3PackedLayout& layout) const {
    if (layout.used_length <= 0 || layout.used_length > input.size(0)) {
      throw std::invalid_argument(
          "MiniMax-H3 final layer used-row geometry mismatch");
    }
    torch::Tensor parameters = adaln_proj_->forward(time_embedding);
    torch::Tensor shift =
        parameters.select(1, 0).select(1, 0).index_select(0, inverse_indices);
    torch::Tensor scale =
        parameters.select(1, 0).select(1, 1).index_select(0, inverse_indices);
    MiniMaxH3FinalOutput output;
    output.activation = norm_->forward(input) * (scale + 1.0) + shift;
    const torch::Tensor fp32_activation =
        output.activation.narrow(0, 0, layout.used_length).to(torch::kFloat32);
    const torch::Tensor used_video_logits =
        video_out_->forward(fp32_activation);
    const torch::Tensor used_audio_logits =
        audio_out_->forward(fp32_activation);
    output.all_video_logits =
        torch::zeros({input.size(0), used_video_logits.size(1)},
                     used_video_logits.options());
    output.all_audio_logits =
        torch::zeros({input.size(0), used_audio_logits.size(1)},
                     used_audio_logits.options());
    output.all_video_logits.narrow(0, 0, layout.used_length)
        .copy_(used_video_logits);
    output.all_audio_logits.narrow(0, 0, layout.used_length)
        .copy_(used_audio_logits);

    torch::Tensor image_positions = layout.img_pos.to(input.device());
    torch::Tensor audio_positions = layout.audio_pos.to(input.device());
    output.raw_selected_video_logits =
        output.all_video_logits.index_select(0, image_positions);
    output.raw_selected_audio_logits =
        output.all_audio_logits.index_select(0, audio_positions);
    torch::Tensor video_mask =
        layout.update_mask.to(input.device()).to(torch::kFloat32).unsqueeze(-1);
    torch::Tensor audio_mask = layout.audio_update_mask.to(input.device())
                                   .to(torch::kFloat32)
                                   .unsqueeze(-1);
    output.selected_video_logits =
        output.raw_selected_video_logits * video_mask;
    output.selected_audio_logits =
        output.raw_selected_audio_logits * audio_mask;
    return output;
  }

 private:
  MiniMaxH3RMSNorm norm_{nullptr};
  MiniMaxH3AdaLNProjection adaln_proj_{nullptr};
  MiniMaxH3Dense video_out_{nullptr};
  MiniMaxH3Dense audio_out_{nullptr};
};
TORCH_MODULE(MiniMaxH3FinalLayer);

class MiniMaxH3C4SourceLayoutValidator final {
 public:
  static constexpr size_t kExpectedTensorCount = 45;

  static std::vector<MiniMaxH3C4SourceTensorSpec> expected_source_tensors() {
    const MiniMaxH3C4Config config;
    const int64_t inner =
        config.num_attention_heads * config.attention_head_dim;
    std::vector<MiniMaxH3C4SourceTensorSpec> specs;
    specs.reserve(kExpectedTensorCount);
    const auto add = [&specs](std::string name,
                              std::vector<int64_t> shape,
                              torch::ScalarType dtype) {
      specs.emplace_back(MiniMaxH3C4SourceTensorSpec{
          .name = std::move(name), .shape = std::move(shape), .dtype = dtype});
    };
    add("video_patch_proj.weight",
        {config.hidden_size, config.video_patch_dim},
        torch::kFloat32);
    add("video_patch_proj.bias", {config.hidden_size}, torch::kFloat32);
    add("audio_patch_proj.weight",
        {config.hidden_size, config.audio_dim},
        torch::kFloat32);
    add("audio_patch_proj.bias", {config.hidden_size}, torch::kFloat32);
    add("condition_proj.weight",
        {config.hidden_size, config.text_dim},
        torch::kBFloat16);
    add("condition_proj.bias", {config.hidden_size}, torch::kBFloat16);
    add("time_embedder.proj_in.weight",
        {config.time_embed_hidden_size, config.timestep_input_dim},
        torch::kFloat32);
    add("time_embedder.proj_in.bias",
        {config.time_embed_hidden_size},
        torch::kFloat32);
    add("time_embedder.proj_out.weight",
        {config.time_embed_dim, config.time_embed_hidden_size},
        torch::kFloat32);
    add("time_embedder.proj_out.bias",
        {config.time_embed_dim},
        torch::kFloat32);
    add("rope.inv_freq", {config.rope_inv_freq_len}, torch::kFloat32);

    const auto add_block = [&](const std::string& prefix) {
      add(prefix + ".norm1.weight", {config.hidden_size}, torch::kBFloat16);
      add(prefix + ".norm2.weight", {config.hidden_size}, torch::kBFloat16);
      add(prefix + ".attn.q_norm.weight",
          {config.attention_head_dim},
          torch::kBFloat16);
      add(prefix + ".attn.k_norm.weight",
          {config.attention_head_dim},
          torch::kBFloat16);
      add(prefix + ".attn.qkv_proj.weight",
          {3 * inner, config.hidden_size},
          torch::kBFloat16);
      add(prefix + ".attn.out_proj.weight",
          {config.hidden_size, inner},
          torch::kBFloat16);
      add(prefix + ".mlp.fc1.weight",
          {2 * config.ffn_hidden_size, config.hidden_size},
          torch::kBFloat16);
      add(prefix + ".mlp.fc2.weight",
          {config.hidden_size, config.ffn_hidden_size},
          torch::kBFloat16);
    };
    for (int64_t index = 0; index < config.num_refiner_layers; ++index) {
      add_block("token_refiner.blocks." + std::to_string(index));
    }
    add("token_refiner.final_norm.weight",
        {config.hidden_size},
        torch::kBFloat16);
    add_block("blocks.0");
    add("blocks.0.adaln_proj.linear.weight",
        {18 * config.hidden_size, config.time_embed_dim},
        torch::kBFloat16);
    add("blocks.0.adaln_proj.linear.bias",
        {18 * config.hidden_size},
        torch::kBFloat16);
    add("final_layer.norm.weight", {config.hidden_size}, torch::kBFloat16);
    add("final_layer.adaln_proj.linear.weight",
        {2 * config.hidden_size, config.time_embed_dim},
        torch::kBFloat16);
    add("final_layer.adaln_proj.linear.bias",
        {2 * config.hidden_size},
        torch::kBFloat16);
    add("final_layer.video_out.weight",
        {config.video_patch_dim, config.hidden_size},
        torch::kFloat32);
    add("final_layer.video_out.bias",
        {config.video_patch_dim},
        torch::kFloat32);
    add("final_layer.audio_out.weight",
        {config.audio_dim, config.hidden_size},
        torch::kFloat32);
    add("final_layer.audio_out.bias", {config.audio_dim}, torch::kFloat32);
    return specs;
  }

  static std::unordered_map<std::string, const torch::Tensor*> validate(
      const std::vector<std::unique_ptr<StateDict>>& shards) {
    std::unordered_map<std::string, const torch::Tensor*> selected;
    for (const std::unique_ptr<StateDict>& shard : shards) {
      if (shard == nullptr) {
        throw std::invalid_argument(
            "MiniMax-H3 C4 source layout contains a null shard");
      }
      for (const auto& [name, tensor] : *shard) {
        if (!is_selected_prefix(name)) {
          continue;
        }
        if (!selected.emplace(name, &tensor).second) {
          throw std::invalid_argument(
              "MiniMax-H3 C4 source tensor loaded more than once: `" + name +
              "`");
        }
      }
    }

    const std::vector<MiniMaxH3C4SourceTensorSpec> specs =
        expected_source_tensors();
    std::unordered_map<std::string, const MiniMaxH3C4SourceTensorSpec*>
        expected;
    expected.reserve(specs.size());
    for (const MiniMaxH3C4SourceTensorSpec& spec : specs) {
      expected.emplace(spec.name, &spec);
    }
    for (const auto& [name, tensor] : selected) {
      (void)tensor;
      if (!expected.contains(name)) {
        throw std::invalid_argument(
            "MiniMax-H3 C4 source layout has unknown selected tensor `" + name +
            "`");
      }
    }
    for (const MiniMaxH3C4SourceTensorSpec& spec : specs) {
      const auto iterator = selected.find(spec.name);
      if (iterator == selected.end()) {
        throw std::invalid_argument(
            "MiniMax-H3 C4 source layout is missing tensor `" + spec.name +
            "`");
      }
      const torch::Tensor& tensor = *iterator->second;
      if (!tensor.defined() || tensor.sizes().vec() != spec.shape ||
          tensor.scalar_type() != spec.dtype) {
        throw std::invalid_argument(
            "MiniMax-H3 C4 source tensor metadata mismatch for `" + spec.name +
            "`: expected " + format_shape(spec.shape) + " " +
            c10::toString(spec.dtype));
      }
    }
    if (selected.size() != kExpectedTensorCount) {
      throw std::logic_error(
          "MiniMax-H3 C4 source tensor count contract is invalid");
    }
    return selected;
  }

 private:
  static bool starts_with(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() &&
           value.compare(0, prefix.size(), prefix) == 0;
  }

  static bool is_selected_prefix(const std::string& name) {
    static const std::array<std::string, 8> kPrefixes = {
        "video_patch_proj.",
        "audio_patch_proj.",
        "condition_proj.",
        "time_embedder.",
        "rope.",
        "token_refiner.",
        "blocks.0.",
        "final_layer.",
    };
    for (const std::string& prefix : kPrefixes) {
      if (starts_with(name, prefix)) {
        return true;
      }
    }
    return false;
  }

  static std::string format_shape(const std::vector<int64_t>& shape) {
    std::ostringstream stream;
    stream << "[";
    for (size_t index = 0; index < shape.size(); ++index) {
      if (index > 0) {
        stream << ",";
      }
      stream << shape[index];
    }
    stream << "]";
    return stream.str();
  }
};

class MiniMaxH3C4HarnessImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3C4HarnessImpl(const torch::TensorOptions& options)
      : config_(), options_(options) {
    const torch::TensorOptions fp32_options = options.dtype(torch::kFloat32);
    const torch::TensorOptions bf16_options = options.dtype(torch::kBFloat16);
    video_patch_proj_ = register_module("video_patch_proj",
                                        MiniMaxH3Dense(config_.video_patch_dim,
                                                       config_.hidden_size,
                                                       /*with_bias=*/true,
                                                       fp32_options));
    audio_patch_proj_ = register_module("audio_patch_proj",
                                        MiniMaxH3Dense(config_.audio_dim,
                                                       config_.hidden_size,
                                                       /*with_bias=*/true,
                                                       fp32_options));
    condition_proj_ = register_module("condition_proj",
                                      MiniMaxH3Dense(config_.text_dim,
                                                     config_.hidden_size,
                                                     /*with_bias=*/true,
                                                     bf16_options));
    time_embedder_ = register_module("time_embedder",
                                     MiniMaxH3TimeEmbedder(config_, options));
    rope_ = register_module("rope",
                            MiniMaxH3Rope(config_.rope_inv_freq_len, options));
    token_refiner_ = register_module("token_refiner",
                                     MiniMaxH3TokenRefiner(config_, options));
    blocks_ = register_module("blocks", torch::nn::ModuleList());
    block0_ = MiniMaxH3DiTBlock(config_, options);
    blocks_->push_back(block0_);
    final_layer_ =
        register_module("final_layer", MiniMaxH3FinalLayer(config_, options));
  }

  void load_source_weights(
      const std::vector<std::unique_ptr<StateDict>>& shards) {
    if (loaded_) {
      throw std::logic_error(
          "MiniMax-H3 C4 source weights have already been loaded");
    }
    const auto source = MiniMaxH3C4SourceLayoutValidator::validate(shards);
    auto parameters = named_parameters(/*recurse=*/true);
    auto buffers = named_buffers(/*recurse=*/true);
    std::unordered_map<std::string, torch::Tensor*> targets;
    targets.reserve(parameters.size() + buffers.size());
    for (auto& parameter : parameters) {
      targets.emplace(parameter.key(), &parameter.value());
    }
    for (auto& buffer : buffers) {
      targets.emplace(buffer.key(), &buffer.value());
    }

    const std::vector<MiniMaxH3C4SourceTensorSpec> specs =
        MiniMaxH3C4SourceLayoutValidator::expected_source_tensors();
    if (targets.size() != specs.size()) {
      throw std::logic_error(
          "MiniMax-H3 C4 module/source tensor counts differ");
    }
    torch::NoGradGuard no_grad;
    for (const MiniMaxH3C4SourceTensorSpec& spec : specs) {
      const auto target_iterator = targets.find(spec.name);
      if (target_iterator == targets.end()) {
        throw std::logic_error(
            "MiniMax-H3 C4 module is missing registered tensor `" + spec.name +
            "`");
      }
      torch::Tensor& target = *target_iterator->second;
      if (target.sizes().vec() != spec.shape ||
          target.scalar_type() != spec.dtype) {
        throw std::logic_error(
            "MiniMax-H3 C4 registered tensor metadata mismatch for `" +
            spec.name + "`");
      }
      torch::Tensor value = *source.at(spec.name);
      if (spec.name.ends_with(".attn.qkv_proj.weight")) {
        value = minimax_h3_reorder_grouped_qkv(
            value, config_.num_attention_heads, config_.attention_head_dim);
      } else if (spec.name.ends_with(".mlp.fc1.weight")) {
        value = minimax_h3_reorder_gate_up_to_up_gate(value);
      }
      target.copy_(value.to(target.device()));
      loaded_source_names_.emplace(spec.name);
    }
    loaded_ = true;
    verify_loaded_weights();
  }

  void verify_loaded_weights() const {
    const std::vector<MiniMaxH3C4SourceTensorSpec> specs =
        MiniMaxH3C4SourceLayoutValidator::expected_source_tensors();
    if (!loaded_ || loaded_source_names_.size() != specs.size()) {
      throw std::logic_error(
          "MiniMax-H3 C4 harness does not have every required source tensor");
    }
    for (const MiniMaxH3C4SourceTensorSpec& spec : specs) {
      if (!loaded_source_names_.contains(spec.name)) {
        throw std::logic_error(
            "MiniMax-H3 C4 harness did not load source tensor `" + spec.name +
            "`");
      }
    }
    for (const auto& parameter : named_parameters(/*recurse=*/true)) {
      if (parameter.value().scalar_type() != expected_dtype(parameter.key())) {
        throw std::logic_error(
            "MiniMax-H3 C4 mixed dtype invariant failed for `" +
            parameter.key() + "`");
      }
    }
    for (const auto& buffer : named_buffers(/*recurse=*/true)) {
      if (buffer.value().scalar_type() != expected_dtype(buffer.key())) {
        throw std::logic_error(
            "MiniMax-H3 C4 mixed dtype invariant failed for buffer `" +
            buffer.key() + "`");
      }
    }
  }

  MiniMaxH3C4Trace forward(const H3PackedLayout& layout,
                           const torch::Tensor& video_rows,
                           const torch::Tensor& audio_rows,
                           const torch::Tensor& timesteps,
                           const torch::Tensor& inverse_indices) const {
    verify_loaded_weights();
    validate_inputs(layout, video_rows, audio_rows, timesteps, inverse_indices);
    const torch::Device device = condition_proj_->weight().device();

    torch::Tensor condition = layout.condition_hidden.squeeze(0).to(device);
    MiniMaxH3C4Trace trace;
    trace.condition_projection = condition_proj_->forward(condition);
    torch::Tensor refiner_cu = torch::tensor(
        std::vector<int64_t>{0, condition.size(0)},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto [refined_condition, refiner_traces] =
        token_refiner_->forward(trace.condition_projection, refiner_cu);
    trace.refined_condition = refined_condition;
    trace.token_refiner_blocks = std::move(refiner_traces);

    trace.video_embedding = video_patch_proj_->forward(video_rows.to(device));
    trace.audio_embedding = audio_patch_proj_->forward(audio_rows.to(device));
    trace.packed_hidden =
        torch::zeros({layout.aligned_length, config_.hidden_size},
                     options_.dtype(torch::kBFloat16));
    trace.packed_hidden.index_copy_(
        0, layout.text_pos.to(device), trace.refined_condition);
    trace.packed_hidden.index_copy_(0,
                                    layout.img_pos.to(device),
                                    trace.video_embedding.to(torch::kBFloat16));
    trace.packed_hidden.index_copy_(0,
                                    layout.audio_pos.to(device),
                                    trace.audio_embedding.to(torch::kBFloat16));

    trace.time_embedding = time_embedder_->forward(timesteps.to(device));
    trace.rope_frequencies = rope_->forward(layout.position_ids.to(device));
    torch::Tensor device_inverse = inverse_indices.to(device);
    trace.combined_indices = minimax_h3_combined_adaln_indices(
        device_inverse, layout.token_tags.to(device));
    trace.block0 = block0_->forward(trace.packed_hidden,
                                    trace.time_embedding,
                                    trace.combined_indices,
                                    trace.rope_frequencies,
                                    layout.cu_seqlens);
    trace.final_output = final_layer_->forward(
        trace.block0.output, trace.time_embedding, device_inverse, layout);
    return trace;
  }

  MiniMaxH3DiTBlock block0() const { return block0_; }

 private:
  static torch::ScalarType expected_dtype(const std::string& name) {
    static const std::unordered_set<std::string> kFp32Names = {
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "audio_patch_proj.bias",
        "time_embedder.proj_in.weight",
        "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight",
        "time_embedder.proj_out.bias",
        "rope.inv_freq",
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    };
    return kFp32Names.contains(name) ? torch::kFloat32 : torch::kBFloat16;
  }

  void validate_inputs(const H3PackedLayout& layout,
                       const torch::Tensor& video_rows,
                       const torch::Tensor& audio_rows,
                       const torch::Tensor& timesteps,
                       const torch::Tensor& inverse_indices) const {
    if (layout.used_length <= 0 || layout.aligned_length < layout.used_length ||
        !layout.condition_hidden.defined() ||
        layout.condition_hidden.dim() != 3 ||
        layout.condition_hidden.size(0) != 1 ||
        layout.condition_hidden.size(2) != config_.text_dim ||
        layout.condition_hidden.scalar_type() != torch::kBFloat16) {
      throw std::invalid_argument("MiniMax-H3 C4 packed layout is invalid");
    }
    if (!video_rows.defined() || video_rows.dim() != 2 ||
        video_rows.size(0) != layout.img_pos.numel() ||
        video_rows.size(1) != config_.video_patch_dim ||
        video_rows.scalar_type() != torch::kFloat32) {
      throw std::invalid_argument(
          "MiniMax-H3 C4 video rows must be FP32 in img_pos order");
    }
    if (!audio_rows.defined() || audio_rows.dim() != 2 ||
        audio_rows.size(0) != layout.audio_pos.numel() ||
        audio_rows.size(1) != config_.audio_dim ||
        audio_rows.scalar_type() != torch::kFloat32) {
      throw std::invalid_argument(
          "MiniMax-H3 C4 audio rows must be FP32 in audio_pos order");
    }
    if (!timesteps.defined() || timesteps.dim() != 1 ||
        timesteps.scalar_type() != torch::kFloat32 || timesteps.numel() <= 0) {
      throw std::invalid_argument(
          "MiniMax-H3 C4 timesteps must be nonempty FP32 [M]");
    }
    if (!inverse_indices.defined() || inverse_indices.dim() != 1 ||
        inverse_indices.size(0) != layout.aligned_length ||
        inverse_indices.scalar_type() != torch::kInt64 ||
        inverse_indices.min().item<int64_t>() < 0 ||
        inverse_indices.max().item<int64_t>() >= timesteps.numel()) {
      throw std::invalid_argument(
          "MiniMax-H3 C4 inverse timestep indices are invalid");
    }
    if (layout.position_ids.sizes().vec() !=
            std::vector<int64_t>({layout.aligned_length, 3}) ||
        layout.position_ids.scalar_type() != torch::kFloat64 ||
        layout.token_tags.sizes().vec() !=
            std::vector<int64_t>({layout.aligned_length}) ||
        layout.token_tags.scalar_type() != torch::kInt64 ||
        layout.cu_seqlens.scalar_type() != torch::kInt32) {
      throw std::invalid_argument(
          "MiniMax-H3 C4 layout position/tag/document metadata is invalid");
    }
  }

  MiniMaxH3C4Config config_;
  torch::TensorOptions options_;
  MiniMaxH3Dense video_patch_proj_{nullptr};
  MiniMaxH3Dense audio_patch_proj_{nullptr};
  MiniMaxH3Dense condition_proj_{nullptr};
  MiniMaxH3TimeEmbedder time_embedder_{nullptr};
  MiniMaxH3Rope rope_{nullptr};
  MiniMaxH3TokenRefiner token_refiner_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  MiniMaxH3DiTBlock block0_{nullptr};
  MiniMaxH3FinalLayer final_layer_{nullptr};
  std::unordered_set<std::string> loaded_source_names_;
  bool loaded_ = false;
};
TORCH_MODULE(MiniMaxH3C4Harness);

}  // namespace xllm
