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

#include <ATen/autocast_mode.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/util/json_reader.h"
#include "framework/model_context.h"

namespace xllm {

struct MiniMaxH3VideoVAEConfig {
  static constexpr int64_t kInputChannels = 3;
  static constexpr int64_t kOutputChannels = 3;
  static constexpr int64_t kLatentChannels = 24;
  static constexpr int64_t kMomentChannels = 48;
  static constexpr int64_t kNormGroups = 32;
  static constexpr double kEncoderNormEps = 1e-6;
  static constexpr int64_t kSpatialRatio = 16;
  static constexpr int64_t kTemporalRatio = 4;
  static constexpr int64_t kClipLength = 17;
  static constexpr int64_t kTokenDrop = 3;
  static constexpr int64_t kTokensPerChunk = 5;
  static constexpr int64_t kTokenOverlap = 2;
  static constexpr int64_t kFramePrePadding = 3;
  static constexpr int64_t kFrameOverlap = 5;
  static constexpr int64_t kTileSize = 256;
  static constexpr int64_t kTileMinOverlap = 64;

  static constexpr int64_t kDecoderLayers = 36;
  static constexpr int64_t kDecoderHeads = 32;
  static constexpr int64_t kDecoderHeadDim = 64;
  static constexpr int64_t kDecoderHiddenSize = 2048;
  static constexpr int64_t kDecoderFfnSize = 8192;
  static constexpr int64_t kDecoderRegisterTokens = 4;
  static constexpr int64_t kDecoderRotaryDim = 48;
  static constexpr double kDecoderRopeTheta = 100.0;
  static constexpr double kDecoderNormEps = 1e-5;

  static constexpr std::array<int64_t, 6> kBlockChannels =
      {128, 256, 256, 512, 512, 1024};
  static constexpr std::array<int64_t, 6> kSpatialDownsample =
      {2, 2, 2, 2, 1, 1};
  static constexpr std::array<int64_t, 6> kTemporalDownsample =
      {1, 2, 2, 1, 1, 1};
};

struct MiniMaxH3VideoVAESourceTensorSpec {
  std::string name;
  std::vector<int64_t> shape;
};

using MiniMaxH3VideoVAETraceHook = std::function<
    void(std::string_view name, int64_t index, const torch::Tensor& value)>;

inline void minimax_h3_vae_trace(const MiniMaxH3VideoVAETraceHook& hook,
                                 std::string_view name,
                                 int64_t index,
                                 const torch::Tensor& value) {
  if (hook) {
    hook(name, index, value);
  }
}

inline std::string minimax_h3_vae_shape_string(
    const std::vector<int64_t>& shape) {
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

inline void minimax_h3_vae_require_finite(const torch::Tensor& value,
                                          std::string_view name) {
  if (!value.defined() || !value.is_floating_point() ||
      !torch::isfinite(value).all().item<bool>()) {
    throw std::invalid_argument(
        "MiniMax-H3 video VAE tensor `" + std::string(name) +
        "` must be defined, floating-point, and finite");
  }
}

inline torch::Tensor minimax_h3_vae_reorder_per_head_qkv(
    const torch::Tensor& source) {
  constexpr int64_t kHeads = MiniMaxH3VideoVAEConfig::kDecoderHeads;
  constexpr int64_t kHeadDim = MiniMaxH3VideoVAEConfig::kDecoderHeadDim;
  constexpr int64_t kInner = kHeads * kHeadDim;
  if (!source.defined() || source.dim() < 1 || source.size(0) != 3 * kInner) {
    throw std::invalid_argument(
        "MiniMax-H3 video VAE source QKV tensor has an invalid shape");
  }

  std::vector<int64_t> grouped_shape = source.sizes().vec();
  grouped_shape[0] = kHeads;
  grouped_shape.insert(grouped_shape.begin() + 1, 3 * kHeadDim);
  torch::Tensor grouped = source.reshape(grouped_shape);
  std::vector<int64_t> projection_shape = source.sizes().vec();
  projection_shape[0] = kInner;
  torch::Tensor query =
      grouped.narrow(1, 0, kHeadDim).reshape(projection_shape);
  torch::Tensor key =
      grouped.narrow(1, kHeadDim, kHeadDim).reshape(projection_shape);
  torch::Tensor value =
      grouped.narrow(1, 2 * kHeadDim, kHeadDim).reshape(projection_shape);
  return torch::cat({query, key, value}, 0).contiguous();
}

inline torch::Tensor minimax_h3_vae_reorder_gate_up_to_up_gate(
    const torch::Tensor& source) {
  if (!source.defined() || source.dim() < 1 || source.size(0) % 2 != 0) {
    throw std::invalid_argument(
        "MiniMax-H3 video VAE source gated FC1 tensor has an invalid shape");
  }
  std::vector<torch::Tensor> gate_up = source.chunk(2, 0);
  return torch::cat({gate_up[1], gate_up[0]}, 0).contiguous();
}

class MiniMaxH3VAEDenseImpl final : public torch::nn::Module {
 public:
  MiniMaxH3VAEDenseImpl(int64_t input_size,
                        int64_t output_size,
                        const torch::TensorOptions& options) {
    weight_ = register_parameter(
        "weight", torch::empty({output_size, input_size}, options));
    bias_ = register_parameter("bias", torch::empty({output_size}, options));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    return torch::nn::functional::linear(input, weight_, bias_);
  }

  const torch::Tensor& weight() const { return weight_; }
  const torch::Tensor& bias() const { return bias_; }

 private:
  torch::Tensor weight_;
  torch::Tensor bias_;
};
TORCH_MODULE(MiniMaxH3VAEDense);

class MiniMaxH3VAECausalConv3dImpl final : public torch::nn::Module {
 public:
  MiniMaxH3VAECausalConv3dImpl(int64_t input_channels,
                               int64_t output_channels,
                               int64_t kernel_size,
                               std::array<int64_t, 3> stride,
                               int64_t spatial_padding,
                               int64_t temporal_padding,
                               const torch::TensorOptions& options)
      : stride_(stride),
        spatial_padding_(spatial_padding),
        temporal_padding_(temporal_padding) {
    weight_ = register_parameter("weight",
                                 torch::empty({output_channels,
                                               input_channels,
                                               kernel_size,
                                               kernel_size,
                                               kernel_size},
                                              options));
    bias_ =
        register_parameter("bias", torch::empty({output_channels}, options));
  }

  MiniMaxH3VAECausalConv3dImpl(int64_t input_channels,
                               int64_t output_channels,
                               int64_t kernel_size,
                               const torch::TensorOptions& options)
      : MiniMaxH3VAECausalConv3dImpl(input_channels,
                                     output_channels,
                                     kernel_size,
                                     {1, 1, 1},
                                     kernel_size == 3 ? 1 : 0,
                                     kernel_size == 3 ? 2 : 0,
                                     options) {}

  torch::Tensor forward(const torch::Tensor& input) const {
    torch::Tensor hidden = input;
    if (spatial_padding_ > 0) {
      hidden = torch::nn::functional::pad(
          hidden,
          torch::nn::functional::PadFuncOptions({spatial_padding_,
                                                 spatial_padding_,
                                                 spatial_padding_,
                                                 spatial_padding_,
                                                 0,
                                                 0})
              .mode(torch::kReflect));
    }
    if (temporal_padding_ > 0) {
      hidden =
          torch::nn::functional::pad(hidden,
                                     torch::nn::functional::PadFuncOptions(
                                         {0, 0, 0, 0, temporal_padding_, 0})
                                         .mode(torch::kConstant)
                                         .value(0.0));
    }
    return torch::nn::functional::conv3d(
        hidden,
        weight_,
        torch::nn::functional::Conv3dFuncOptions()
            .bias(bias_)
            .stride({stride_[0], stride_[1], stride_[2]})
            .padding(0));
  }

  const torch::Tensor& weight() const { return weight_; }

 private:
  torch::Tensor weight_;
  torch::Tensor bias_;
  std::array<int64_t, 3> stride_;
  int64_t spatial_padding_;
  int64_t temporal_padding_;
};
TORCH_MODULE(MiniMaxH3VAECausalConv3d);

class MiniMaxH3VAEGroupNormImpl final : public torch::nn::Module {
 public:
  MiniMaxH3VAEGroupNormImpl(int64_t channels,
                            const torch::TensorOptions& options) {
    weight_ = register_parameter("weight", torch::empty({channels}, options));
    bias_ = register_parameter("bias", torch::empty({channels}, options));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    if (input.dim() != 5) {
      throw std::invalid_argument(
          "MiniMax-H3 time-isolated GroupNorm expects [B,C,T,H,W]");
    }
    const int64_t batch = input.size(0);
    const int64_t channels = input.size(1);
    const int64_t frames = input.size(2);
    const int64_t height = input.size(3);
    const int64_t width = input.size(4);
    torch::Tensor merged =
        input.permute({0, 2, 1, 3, 4})
            .contiguous()
            .view({batch * frames, channels, 1, height, width});
    merged = torch::nn::functional::group_norm(
        merged,
        torch::nn::functional::GroupNormFuncOptions(
            MiniMaxH3VideoVAEConfig::kNormGroups)
            .weight(weight_)
            .bias(bias_)
            .eps(MiniMaxH3VideoVAEConfig::kEncoderNormEps));
    return merged.view({batch, frames, channels, height, width})
        .permute({0, 2, 1, 3, 4})
        .contiguous();
  }

 private:
  torch::Tensor weight_;
  torch::Tensor bias_;
};
TORCH_MODULE(MiniMaxH3VAEGroupNorm);

class MiniMaxH3VAEResnetBlock3dImpl final : public torch::nn::Module {
 public:
  MiniMaxH3VAEResnetBlock3dImpl(int64_t input_channels,
                                int64_t output_channels,
                                const torch::TensorOptions& options) {
    norm1_ = register_module("norm1",
                             MiniMaxH3VAEGroupNorm(input_channels, options));
    conv1_ = register_module(
        "conv1",
        MiniMaxH3VAECausalConv3d(input_channels, output_channels, 3, options));
    norm2_ = register_module("norm2",
                             MiniMaxH3VAEGroupNorm(output_channels, options));
    conv2_ = register_module(
        "conv2",
        MiniMaxH3VAECausalConv3d(output_channels, output_channels, 3, options));
    if (input_channels != output_channels) {
      shortcut_ =
          register_module("nin_shortcut",
                          MiniMaxH3VAECausalConv3d(
                              input_channels, output_channels, 1, options));
    }
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    torch::Tensor hidden = conv1_->forward(torch::silu(norm1_->forward(input)));
    hidden = conv2_->forward(torch::silu(norm2_->forward(hidden)));
    torch::Tensor residual = shortcut_ ? shortcut_->forward(input) : input;
    return residual + hidden;
  }

 private:
  MiniMaxH3VAEGroupNorm norm1_{nullptr};
  MiniMaxH3VAECausalConv3d conv1_{nullptr};
  MiniMaxH3VAEGroupNorm norm2_{nullptr};
  MiniMaxH3VAECausalConv3d conv2_{nullptr};
  MiniMaxH3VAECausalConv3d shortcut_{nullptr};
};
TORCH_MODULE(MiniMaxH3VAEResnetBlock3d);

class MiniMaxH3VAEDownsample3dImpl final : public torch::nn::Module {
 public:
  MiniMaxH3VAEDownsample3dImpl(int64_t channels,
                               int64_t temporal_stride,
                               int64_t spatial_stride,
                               const torch::TensorOptions& options)
      : spatial_stride_(spatial_stride) {
    conv_ = register_module(
        "conv",
        MiniMaxH3VAECausalConv3d(
            channels,
            channels,
            3,
            std::array<int64_t, 3>{
                temporal_stride, spatial_stride, spatial_stride},
            /*spatial_padding=*/0,
            /*temporal_padding=*/2,
            options));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    torch::Tensor hidden = input;
    if (spatial_stride_ == 2) {
      hidden = torch::nn::functional::pad(
          hidden,
          torch::nn::functional::PadFuncOptions({0, 1, 0, 1, 0, 0})
              .mode(torch::kReflect));
    }
    return conv_->forward(hidden);
  }

 private:
  int64_t spatial_stride_;
  MiniMaxH3VAECausalConv3d conv_{nullptr};
};
TORCH_MODULE(MiniMaxH3VAEDownsample3d);

class MiniMaxH3VAEEncoderLevelImpl final : public torch::nn::Module {
 public:
  MiniMaxH3VAEEncoderLevelImpl(int64_t input_channels,
                               int64_t output_channels,
                               int64_t temporal_stride,
                               int64_t spatial_stride,
                               const torch::TensorOptions& options) {
    blocks_ = register_module("block", torch::nn::ModuleList());
    for (int64_t index = 0; index < 2; ++index) {
      MiniMaxH3VAEResnetBlock3d block(
          index == 0 ? input_channels : output_channels,
          output_channels,
          options);
      blocks_->push_back(block);
      block_layers_.emplace_back(std::move(block));
    }
    if (temporal_stride * spatial_stride > 1) {
      downsample_ = register_module(
          "downsample",
          MiniMaxH3VAEDownsample3d(
              output_channels, temporal_stride, spatial_stride, options));
    }
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    torch::Tensor hidden = input;
    for (const MiniMaxH3VAEResnetBlock3d& block : block_layers_) {
      hidden = block->forward(hidden);
    }
    if (downsample_) {
      hidden = downsample_->forward(hidden);
    }
    return hidden;
  }

 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<MiniMaxH3VAEResnetBlock3d> block_layers_;
  MiniMaxH3VAEDownsample3d downsample_{nullptr};
};
TORCH_MODULE(MiniMaxH3VAEEncoderLevel);

class MiniMaxH3VAEEncoder3dImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3VAEEncoder3dImpl(const torch::TensorOptions& options) {
    conv_in_ = register_module(
        "conv_in",
        MiniMaxH3VAECausalConv3d(MiniMaxH3VideoVAEConfig::kInputChannels,
                                 MiniMaxH3VideoVAEConfig::kBlockChannels[0],
                                 3,
                                 options));
    down_ = register_module("down", torch::nn::ModuleList());
    for (size_t index = 0;
         index < MiniMaxH3VideoVAEConfig::kBlockChannels.size();
         ++index) {
      const int64_t input_channels =
          index == 0 ? MiniMaxH3VideoVAEConfig::kBlockChannels[0]
                     : MiniMaxH3VideoVAEConfig::kBlockChannels[index - 1];
      MiniMaxH3VAEEncoderLevel level(
          input_channels,
          MiniMaxH3VideoVAEConfig::kBlockChannels[index],
          MiniMaxH3VideoVAEConfig::kTemporalDownsample[index],
          MiniMaxH3VideoVAEConfig::kSpatialDownsample[index],
          options);
      down_->push_back(level);
      levels_.emplace_back(std::move(level));
    }
    norm_out_ = register_module(
        "norm_out",
        MiniMaxH3VAEGroupNorm(MiniMaxH3VideoVAEConfig::kBlockChannels.back(),
                              options));
    conv_out_ = register_module(
        "conv_out",
        MiniMaxH3VAECausalConv3d(MiniMaxH3VideoVAEConfig::kBlockChannels.back(),
                                 MiniMaxH3VideoVAEConfig::kMomentChannels,
                                 3,
                                 options));
  }

  torch::Tensor forward(const torch::Tensor& input,
                        const MiniMaxH3VideoVAETraceHook& hook) const {
    torch::Tensor hidden = conv_in_->forward(input);
    minimax_h3_vae_trace(hook, "encoder.conv_in", -1, hidden);
    for (size_t index = 0; index < levels_.size(); ++index) {
      hidden = levels_[index]->forward(hidden);
      minimax_h3_vae_trace(
          hook, "encoder.down", static_cast<int64_t>(index), hidden);
    }
    hidden = conv_out_->forward(torch::silu(norm_out_->forward(hidden)));
    minimax_h3_vae_trace(hook, "encoder.conv_out", -1, hidden);
    return hidden;
  }

 private:
  MiniMaxH3VAECausalConv3d conv_in_{nullptr};
  torch::nn::ModuleList down_{nullptr};
  std::vector<MiniMaxH3VAEEncoderLevel> levels_;
  MiniMaxH3VAEGroupNorm norm_out_{nullptr};
  MiniMaxH3VAECausalConv3d conv_out_{nullptr};
};
TORCH_MODULE(MiniMaxH3VAEEncoder3d);

inline torch::Tensor minimax_h3_vae_rms_norm(
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& weight,
    double eps) {
  const torch::ScalarType input_dtype = input.scalar_type();
  torch::Tensor value = input.to(torch::kFloat32);
  std::optional<torch::Tensor> fp32_weight;
  if (weight.has_value()) {
    fp32_weight = weight->to(torch::kFloat32);
  }
  return at::rms_norm(value, {value.size(-1)}, fp32_weight, eps)
      .to(input_dtype);
}

class MiniMaxH3VAERMSNormImpl final : public torch::nn::Module {
 public:
  MiniMaxH3VAERMSNormImpl(int64_t size,
                          double eps,
                          const torch::TensorOptions& options)
      : eps_(eps) {
    weight_ = register_parameter("weight", torch::empty({size}, options));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    return minimax_h3_vae_rms_norm(input, weight_, eps_);
  }

 private:
  torch::Tensor weight_;
  double eps_;
};
TORCH_MODULE(MiniMaxH3VAERMSNorm);

class MiniMaxH3VAELayerNormImpl final : public torch::nn::Module {
 public:
  MiniMaxH3VAELayerNormImpl(int64_t size,
                            double eps,
                            const torch::TensorOptions& options)
      : size_(size), eps_(eps) {
    weight_ = register_parameter("weight", torch::empty({size}, options));
    bias_ = register_parameter("bias", torch::empty({size}, options));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    return torch::nn::functional::layer_norm(
        input,
        torch::nn::functional::LayerNormFuncOptions({size_})
            .weight(weight_)
            .bias(bias_)
            .eps(eps_));
  }

 private:
  int64_t size_;
  double eps_;
  torch::Tensor weight_;
  torch::Tensor bias_;
};
TORCH_MODULE(MiniMaxH3VAELayerNorm);

class MiniMaxH3VAEAttentionImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3VAEAttentionImpl(const torch::TensorOptions& options) {
    to_qkv_ = register_module(
        "to_qkv",
        MiniMaxH3VAEDense(MiniMaxH3VideoVAEConfig::kDecoderHiddenSize,
                          3 * MiniMaxH3VideoVAEConfig::kDecoderHiddenSize,
                          options));
    to_out_ = register_module(
        "to_out",
        MiniMaxH3VAEDense(MiniMaxH3VideoVAEConfig::kDecoderHiddenSize,
                          MiniMaxH3VideoVAEConfig::kDecoderHiddenSize,
                          options));
  }

  torch::Tensor forward(
      const torch::Tensor& input,
      const std::pair<torch::Tensor, torch::Tensor>& rotary) const {
    constexpr int64_t kHeads = MiniMaxH3VideoVAEConfig::kDecoderHeads;
    constexpr int64_t kHeadDim = MiniMaxH3VideoVAEConfig::kDecoderHeadDim;
    constexpr int64_t kInner = kHeads * kHeadDim;
    const int64_t batch = input.size(0);
    const int64_t sequence = input.size(1);

    std::vector<torch::Tensor> weights = to_qkv_->weight().split(kInner, 0);
    std::vector<torch::Tensor> biases = to_qkv_->bias().split(kInner, 0);
    torch::Tensor query =
        torch::nn::functional::linear(input, weights[0], biases[0]);
    torch::Tensor key =
        torch::nn::functional::linear(input, weights[1], biases[1]);
    torch::Tensor value =
        torch::nn::functional::linear(input, weights[2], biases[2]);
    query = query.view({batch, sequence, kHeads, kHeadDim});
    key = key.view({batch, sequence, kHeads, kHeadDim});
    value = value.view({batch, sequence, kHeads, kHeadDim});

    query = minimax_h3_vae_rms_norm(
        query, std::nullopt, MiniMaxH3VideoVAEConfig::kDecoderNormEps);
    key = minimax_h3_vae_rms_norm(
        key, std::nullopt, MiniMaxH3VideoVAEConfig::kDecoderNormEps);
    query = apply_rotary(query, rotary);
    key = apply_rotary(key, rotary);

    torch::Tensor attended =
        torch::scaled_dot_product_attention(query.transpose(1, 2),
                                            key.transpose(1, 2),
                                            value.transpose(1, 2),
                                            torch::nullopt,
                                            /*dropout_p=*/0.0,
                                            /*is_causal=*/false);
    attended = attended.transpose(1, 2).reshape({batch, sequence, kInner});
    return to_out_->forward(attended);
  }

 private:
  static torch::Tensor apply_rotary(
      const torch::Tensor& input,
      const std::pair<torch::Tensor, torch::Tensor>& rotary) {
    const int64_t rotary_dim = rotary.first.size(-1);
    torch::Tensor rotated_input = input.narrow(-1, 0, rotary_dim);
    const int64_t half = rotary_dim / 2;
    torch::Tensor rotate_half =
        torch::cat({-rotated_input.narrow(-1, half, half),
                    rotated_input.narrow(-1, 0, half)},
                   -1);
    torch::Tensor cosine = rotary.first.to(input.scalar_type());
    torch::Tensor sine = rotary.second.to(input.scalar_type());
    torch::Tensor output_rotary = rotated_input * cosine + rotate_half * sine;
    torch::Tensor output_pass =
        input.narrow(-1, rotary_dim, input.size(-1) - rotary_dim);
    return torch::cat({output_rotary, output_pass}, -1);
  }

  MiniMaxH3VAEDense to_qkv_{nullptr};
  MiniMaxH3VAEDense to_out_{nullptr};
};
TORCH_MODULE(MiniMaxH3VAEAttention);

class MiniMaxH3VAEFeedForwardImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3VAEFeedForwardImpl(const torch::TensorOptions& options) {
    w1_ = register_module(
        "w1",
        MiniMaxH3VAEDense(MiniMaxH3VideoVAEConfig::kDecoderHiddenSize,
                          2 * MiniMaxH3VideoVAEConfig::kDecoderFfnSize,
                          options));
    w2_ = register_module(
        "w2",
        MiniMaxH3VAEDense(MiniMaxH3VideoVAEConfig::kDecoderFfnSize,
                          MiniMaxH3VideoVAEConfig::kDecoderHiddenSize,
                          options));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    std::vector<torch::Tensor> up_gate = w1_->forward(input).chunk(2, -1);
    return w2_->forward(up_gate[0] * torch::silu(up_gate[1]));
  }

 private:
  MiniMaxH3VAEDense w1_{nullptr};
  MiniMaxH3VAEDense w2_{nullptr};
};
TORCH_MODULE(MiniMaxH3VAEFeedForward);

class MiniMaxH3VAETransformerBlockImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3VAETransformerBlockImpl(
      const torch::TensorOptions& options) {
    norm1_ = register_module(
        "norm1",
        MiniMaxH3VAERMSNorm(MiniMaxH3VideoVAEConfig::kDecoderHiddenSize,
                            MiniMaxH3VideoVAEConfig::kDecoderNormEps,
                            options));
    attention_ = register_module("attn", MiniMaxH3VAEAttention(options));
    scale1_ = register_parameter(
        "scale1",
        torch::empty({MiniMaxH3VideoVAEConfig::kDecoderHiddenSize}, options));
    norm2_ = register_module(
        "norm2",
        MiniMaxH3VAERMSNorm(MiniMaxH3VideoVAEConfig::kDecoderHiddenSize,
                            MiniMaxH3VideoVAEConfig::kDecoderNormEps,
                            options));
    feed_forward_ = register_module("ff", MiniMaxH3VAEFeedForward(options));
    scale2_ = register_parameter(
        "scale2",
        torch::empty({MiniMaxH3VideoVAEConfig::kDecoderHiddenSize}, options));
  }

  torch::Tensor forward(
      const torch::Tensor& input,
      const std::pair<torch::Tensor, torch::Tensor>& rotary) const {
    torch::Tensor normalized =
        norm1_->forward(input.to(torch::kFloat32)).to(input.scalar_type());
    torch::Tensor hidden =
        input + attention_->forward(normalized, rotary) * scale1_;
    normalized =
        norm2_->forward(hidden.to(torch::kFloat32)).to(hidden.scalar_type());
    return hidden + feed_forward_->forward(normalized) * scale2_;
  }

 private:
  MiniMaxH3VAERMSNorm norm1_{nullptr};
  MiniMaxH3VAEAttention attention_{nullptr};
  torch::Tensor scale1_;
  MiniMaxH3VAERMSNorm norm2_{nullptr};
  MiniMaxH3VAEFeedForward feed_forward_{nullptr};
  torch::Tensor scale2_;
};
TORCH_MODULE(MiniMaxH3VAETransformerBlock);

class MiniMaxH3VAEViTDecoder3dImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3VAEViTDecoder3dImpl(const torch::TensorOptions& options) {
    x_embedder_ = register_module(
        "x_embedder",
        MiniMaxH3VAEDense(MiniMaxH3VideoVAEConfig::kLatentChannels,
                          MiniMaxH3VideoVAEConfig::kDecoderHiddenSize,
                          options));
    register_tokens_ = register_parameter(
        "register_tokens",
        torch::empty({1,
                      MiniMaxH3VideoVAEConfig::kDecoderRegisterTokens,
                      MiniMaxH3VideoVAEConfig::kDecoderHiddenSize},
                     options));
    transformer_blocks_ =
        register_module("transformer_blocks", torch::nn::ModuleList());
    block_layers_.reserve(MiniMaxH3VideoVAEConfig::kDecoderLayers);
    for (int64_t index = 0; index < MiniMaxH3VideoVAEConfig::kDecoderLayers;
         ++index) {
      MiniMaxH3VAETransformerBlock block(options);
      transformer_blocks_->push_back(block);
      block_layers_.emplace_back(std::move(block));
    }
    norm_out_ = register_module(
        "norm_out",
        MiniMaxH3VAELayerNorm(MiniMaxH3VideoVAEConfig::kDecoderHiddenSize,
                              MiniMaxH3VideoVAEConfig::kDecoderNormEps,
                              options));
    constexpr int64_t kPatchOutput = MiniMaxH3VideoVAEConfig::kOutputChannels *
                                     MiniMaxH3VideoVAEConfig::kTemporalRatio *
                                     MiniMaxH3VideoVAEConfig::kSpatialRatio *
                                     MiniMaxH3VideoVAEConfig::kSpatialRatio;
    proj_out_ = register_module(
        "proj_out",
        MiniMaxH3VAEDense(MiniMaxH3VideoVAEConfig::kDecoderHiddenSize,
                          kPatchOutput,
                          options));
    mask_token_ = register_buffer(
        "mask_token",
        torch::empty({1, 1, MiniMaxH3VideoVAEConfig::kDecoderHiddenSize},
                     options));
  }

  torch::Tensor forward(const torch::Tensor& input,
                        const MiniMaxH3VideoVAETraceHook& hook) const {
    if (input.dim() != 5 ||
        input.size(1) != MiniMaxH3VideoVAEConfig::kLatentChannels) {
      throw std::invalid_argument(
          "MiniMax-H3 ViT decoder expects [B,24,T,H,W]");
    }
    const int64_t batch = input.size(0);
    const int64_t frames = input.size(2);
    const int64_t height = input.size(3);
    const int64_t width = input.size(4);
    const int64_t patches = frames * height * width;

    torch::Tensor hidden =
        input.permute({0, 2, 3, 4, 1})
            .reshape(
                {batch, patches, MiniMaxH3VideoVAEConfig::kLatentChannels});
    hidden = x_embedder_->forward(hidden);
    torch::Tensor registers = register_tokens_.expand(
        {batch,
         MiniMaxH3VideoVAEConfig::kDecoderRegisterTokens,
         MiniMaxH3VideoVAEConfig::kDecoderHiddenSize});
    torch::Tensor zero_suffix = torch::zeros_like(hidden.slice(1, 0, 1));
    hidden = torch::cat({hidden, registers, zero_suffix}, 1);

    torch::Tensor position_ids =
        make_position_ids(batch, frames, height, width, hidden.device());
    auto rotary = make_rotary(position_ids);
    minimax_h3_vae_trace(hook, "decoder.rope.position_ids", -1, position_ids);
    minimax_h3_vae_trace(hook, "decoder.rope.cos", -1, rotary.first);
    minimax_h3_vae_trace(hook, "decoder.rope.sin", -1, rotary.second);
    for (size_t index = 0; index < block_layers_.size(); ++index) {
      hidden = block_layers_[index]->forward(hidden, rotary);
      minimax_h3_vae_trace(
          hook, "decoder.block", static_cast<int64_t>(index), hidden);
    }
    hidden = norm_out_->forward(hidden);
    minimax_h3_vae_trace(hook, "decoder.norm_out", -1, hidden);
    hidden = proj_out_->forward(hidden).slice(1, 0, patches);
    minimax_h3_vae_trace(hook, "decoder.proj_out", -1, hidden);

    hidden = hidden.view({batch,
                          frames,
                          height,
                          width,
                          MiniMaxH3VideoVAEConfig::kOutputChannels,
                          MiniMaxH3VideoVAEConfig::kTemporalRatio,
                          MiniMaxH3VideoVAEConfig::kSpatialRatio,
                          MiniMaxH3VideoVAEConfig::kSpatialRatio});
    hidden = hidden.permute({0, 4, 1, 5, 2, 6, 3, 7}).contiguous();
    return hidden.reshape({batch,
                           MiniMaxH3VideoVAEConfig::kOutputChannels,
                           frames * MiniMaxH3VideoVAEConfig::kTemporalRatio,
                           height * MiniMaxH3VideoVAEConfig::kSpatialRatio,
                           width * MiniMaxH3VideoVAEConfig::kSpatialRatio});
  }

 private:
  static torch::Tensor make_position_ids(int64_t batch,
                                         int64_t frames,
                                         int64_t height,
                                         int64_t width,
                                         const torch::Device& device) {
    const torch::TensorOptions options =
        torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto axis = [&options](int64_t size) {
      return 2.0 * (torch::arange(size, options) + 0.5) /
                 static_cast<double>(size) -
             1.0;
    };
    torch::Tensor t = axis(frames);
    torch::Tensor y = axis(height);
    torch::Tensor x = axis(width);
    torch::Tensor positions =
        torch::stack({t.view({frames, 1, 1}).expand({frames, height, width}),
                      y.view({1, height, 1}).expand({frames, height, width}),
                      x.view({1, 1, width}).expand({frames, height, width})},
                     -1)
            .reshape({1, frames * height * width, 3})
            .expand({batch, frames * height * width, 3});
    torch::Tensor suffix = torch::zeros(
        {batch, MiniMaxH3VideoVAEConfig::kDecoderRegisterTokens + 1, 3},
        options);
    return torch::cat({positions, suffix}, 1);
  }

  static std::pair<torch::Tensor, torch::Tensor> make_rotary(
      const torch::Tensor& position_ids) {
    constexpr int64_t kAxisFrequencyCount =
        MiniMaxH3VideoVAEConfig::kDecoderRotaryDim / 6;
    torch::Tensor exponent =
        torch::arange(kAxisFrequencyCount, position_ids.options()) /
        static_cast<double>(kAxisFrequencyCount);
    torch::Tensor inv_freq =
        torch::pow(MiniMaxH3VideoVAEConfig::kDecoderRopeTheta, -exponent);
    torch::Tensor angles = (2.0 * 3.14159265358979323846) *
                           position_ids.unsqueeze(-1) *
                           inv_freq.view({1, 1, 1, -1});
    angles = angles.flatten(2, 3);
    angles = torch::cat({angles, angles}, -1).unsqueeze(2);
    return {torch::cos(angles), torch::sin(angles)};
  }

  MiniMaxH3VAEDense x_embedder_{nullptr};
  torch::Tensor register_tokens_;
  torch::nn::ModuleList transformer_blocks_{nullptr};
  std::vector<MiniMaxH3VAETransformerBlock> block_layers_;
  MiniMaxH3VAELayerNorm norm_out_{nullptr};
  MiniMaxH3VAEDense proj_out_{nullptr};
  torch::Tensor mask_token_;
};
TORCH_MODULE(MiniMaxH3VAEViTDecoder3d);

class MiniMaxH3VAEDiagonalGaussianDistribution {
 public:
  explicit MiniMaxH3VAEDiagonalGaussianDistribution(
      const torch::Tensor& moments) {
    if (!moments.defined() || moments.dim() != 5 ||
        moments.scalar_type() != torch::kFloat32 ||
        moments.size(1) != MiniMaxH3VideoVAEConfig::kMomentChannels) {
      throw std::invalid_argument(
          "MiniMax-H3 posterior moments must be FP32 [B,48,T,H,W]");
    }
    minimax_h3_vae_require_finite(moments, "posterior.moments");
    parameters_ = moments;
    std::vector<torch::Tensor> split = moments.chunk(2, 1);
    mean_ = split[0];
    raw_logvar_ = split[1];
    logvar_ = raw_logvar_.clamp(-30.0, 20.0);
    std_ = torch::exp(0.5 * logvar_);
  }

  torch::Tensor sample(const torch::Tensor& epsilon,
                       bool fp16_round_trip = false) const {
    if (!epsilon.defined() || epsilon.sizes() != mean_.sizes() ||
        epsilon.scalar_type() != torch::kFloat32 ||
        epsilon.device() != mean_.device()) {
      throw std::invalid_argument(
          "MiniMax-H3 posterior epsilon must be FP32 on the posterior device "
          "and match its mean shape");
    }
    minimax_h3_vae_require_finite(epsilon, "posterior.epsilon");
    torch::Tensor value = mean_ + std_ * epsilon;
    minimax_h3_vae_require_finite(value, "posterior.sample");
    if (fp16_round_trip) {
      value = value.to(torch::kFloat16).to(torch::kFloat32);
      minimax_h3_vae_require_finite(value, "posterior.rounded_fp32");
    }
    return value.contiguous();
  }

  torch::Tensor mode() const { return mean_; }
  const torch::Tensor& parameters() const { return parameters_; }
  const torch::Tensor& mean() const { return mean_; }
  const torch::Tensor& raw_logvar() const { return raw_logvar_; }
  const torch::Tensor& logvar() const { return logvar_; }
  const torch::Tensor& std() const { return std_; }

 private:
  torch::Tensor parameters_;
  torch::Tensor mean_;
  torch::Tensor raw_logvar_;
  torch::Tensor logvar_;
  torch::Tensor std_;
};

class MiniMaxH3VideoVAESourceLayoutValidator final {
 public:
  static constexpr size_t kExpectedTensorCount = 560;

  static std::vector<MiniMaxH3VideoVAESourceTensorSpec>
  expected_source_tensors() {
    std::vector<MiniMaxH3VideoVAESourceTensorSpec> specs;
    specs.reserve(kExpectedTensorCount);
    auto add = [&specs](std::string name, std::vector<int64_t> shape) {
      specs.push_back({std::move(name), std::move(shape)});
    };
    auto add_pair = [&add](const std::string& name,
                           std::vector<int64_t> weight_shape,
                           int64_t output_size) {
      add(name + ".weight", std::move(weight_shape));
      add(name + ".bias", {output_size});
    };

    add_pair("encoder.conv_in", {128, 3, 3, 3, 3}, 128);
    for (size_t level = 0;
         level < MiniMaxH3VideoVAEConfig::kBlockChannels.size();
         ++level) {
      const int64_t level_input =
          level == 0 ? MiniMaxH3VideoVAEConfig::kBlockChannels[0]
                     : MiniMaxH3VideoVAEConfig::kBlockChannels[level - 1];
      const int64_t level_output =
          MiniMaxH3VideoVAEConfig::kBlockChannels[level];
      for (int64_t block = 0; block < 2; ++block) {
        const int64_t block_input = block == 0 ? level_input : level_output;
        const std::string prefix = "encoder.down." + std::to_string(level) +
                                   ".block." + std::to_string(block);
        add(prefix + ".norm1.weight", {block_input});
        add(prefix + ".norm1.bias", {block_input});
        add_pair(prefix + ".conv1",
                 {level_output, block_input, 3, 3, 3},
                 level_output);
        add(prefix + ".norm2.weight", {level_output});
        add(prefix + ".norm2.bias", {level_output});
        add_pair(prefix + ".conv2",
                 {level_output, level_output, 3, 3, 3},
                 level_output);
        if (block_input != level_output) {
          add_pair(prefix + ".nin_shortcut",
                   {level_output, block_input, 1, 1, 1},
                   level_output);
        }
      }
      if (MiniMaxH3VideoVAEConfig::kSpatialDownsample[level] *
              MiniMaxH3VideoVAEConfig::kTemporalDownsample[level] >
          1) {
        add_pair("encoder.down." + std::to_string(level) + ".downsample.conv",
                 {level_output, level_output, 3, 3, 3},
                 level_output);
      }
    }
    add("encoder.norm_out.weight", {1024});
    add("encoder.norm_out.bias", {1024});
    add_pair("encoder.conv_out", {48, 1024, 3, 3, 3}, 48);
    add_pair("quant_conv", {48, 48, 1, 1, 1}, 48);
    add_pair("post_quant_conv", {24, 24, 1, 1, 1}, 24);

    add_pair("decoder.x_embedder", {2048, 24}, 2048);
    add("decoder.register_tokens", {1, 4, 2048});
    for (int64_t block = 0; block < MiniMaxH3VideoVAEConfig::kDecoderLayers;
         ++block) {
      const std::string prefix =
          "decoder.transformer_blocks." + std::to_string(block);
      add(prefix + ".norm1.weight", {2048});
      add_pair(prefix + ".attn.to_qkv", {6144, 2048}, 6144);
      add_pair(prefix + ".attn.to_out", {2048, 2048}, 2048);
      add(prefix + ".scale1", {2048});
      add(prefix + ".norm2.weight", {2048});
      add_pair(prefix + ".ff.w1", {16384, 2048}, 16384);
      add_pair(prefix + ".ff.w2", {2048, 8192}, 2048);
      add(prefix + ".scale2", {2048});
    }
    add("decoder.norm_out.weight", {2048});
    add("decoder.norm_out.bias", {2048});
    add_pair("decoder.proj_out", {3072, 2048}, 3072);
    add("decoder.mask_token", {1, 1, 2048});

    if (specs.size() != kExpectedTensorCount) {
      throw std::logic_error(
          "MiniMax-H3 video VAE source inventory declaration is invalid");
    }
    return specs;
  }

  static std::unordered_map<std::string, const torch::Tensor*> validate(
      const std::vector<std::unique_ptr<StateDict>>& shards) {
    if (shards.empty()) {
      throw std::invalid_argument(
          "MiniMax-H3 video VAE source inventory has no shards");
    }
    const std::vector<MiniMaxH3VideoVAESourceTensorSpec> specs =
        expected_source_tensors();
    std::unordered_map<std::string, const MiniMaxH3VideoVAESourceTensorSpec*>
        expected;
    expected.reserve(specs.size());
    for (const MiniMaxH3VideoVAESourceTensorSpec& spec : specs) {
      expected.emplace(spec.name, &spec);
    }

    std::unordered_map<std::string, const torch::Tensor*> source;
    source.reserve(kExpectedTensorCount);
    for (const std::unique_ptr<StateDict>& shard : shards) {
      if (shard == nullptr) {
        throw std::invalid_argument(
            "MiniMax-H3 video VAE source inventory contains a null shard");
      }
      for (const auto& [name, tensor] : *shard) {
        if (!expected.contains(name)) {
          throw std::invalid_argument(
              "MiniMax-H3 video VAE source inventory has unknown tensor `" +
              name + "`");
        }
        if (!source.emplace(name, &tensor).second) {
          throw std::invalid_argument(
              "MiniMax-H3 video VAE source tensor occurs more than once: `" +
              name + "`");
        }
      }
    }
    if (source.size() != kExpectedTensorCount) {
      throw std::invalid_argument(
          "MiniMax-H3 video VAE source inventory must contain exactly 560 "
          "tensors");
    }
    for (const MiniMaxH3VideoVAESourceTensorSpec& spec : specs) {
      const auto iterator = source.find(spec.name);
      if (iterator == source.end()) {
        throw std::invalid_argument(
            "MiniMax-H3 video VAE source inventory is missing tensor `" +
            spec.name + "`");
      }
      const torch::Tensor& tensor = *iterator->second;
      if (!tensor.defined() || tensor.scalar_type() != torch::kFloat32 ||
          tensor.sizes().vec() != spec.shape) {
        throw std::invalid_argument(
            "MiniMax-H3 video VAE source tensor metadata mismatch for `" +
            spec.name + "`: expected FP32 " +
            minimax_h3_vae_shape_string(spec.shape));
      }
    }
    return source;
  }
};

class MiniMaxH3VideoVAEImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3VideoVAEImpl(const ModelContext& context)
      : MiniMaxH3VideoVAEImpl(context.get_tensor_options()) {}

  explicit MiniMaxH3VideoVAEImpl(const torch::TensorOptions& options)
      : options_(options.dtype(torch::kFloat32)) {
    encoder_ = register_module("encoder", MiniMaxH3VAEEncoder3d(options_));
    quant_conv_ = register_module(
        "quant_conv",
        MiniMaxH3VAECausalConv3d(MiniMaxH3VideoVAEConfig::kMomentChannels,
                                 MiniMaxH3VideoVAEConfig::kMomentChannels,
                                 1,
                                 options_));
    post_quant_conv_ = register_module(
        "post_quant_conv",
        MiniMaxH3VAECausalConv3d(MiniMaxH3VideoVAEConfig::kLatentChannels,
                                 MiniMaxH3VideoVAEConfig::kLatentChannels,
                                 1,
                                 options_));
    decoder_ = register_module("decoder", MiniMaxH3VAEViTDecoder3d(options_));
    for (auto& parameter : named_parameters(/*recurse=*/true)) {
      parameter.value().set_requires_grad(false);
    }
  }

  MiniMaxH3VAEDiagonalGaussianDistribution encode(
      const torch::Tensor& imagenet_normalized_video,
      const MiniMaxH3VideoVAETraceHook& hook = nullptr) const {
    verify_loaded_weights();
    validate_video(imagenet_normalized_video,
                   MiniMaxH3VideoVAEConfig::kInputChannels,
                   "encode input");
    if (imagenet_normalized_video.device() != quant_conv_->weight().device()) {
      throw std::invalid_argument(
          "MiniMax-H3 video VAE encode input is on the wrong device");
    }
    torch::NoGradGuard no_grad;
    torch::Tensor input = imagenet_normalized_video.to(torch::kFloat32);
    torch::Tensor moments = encode_temporal(input, hook);
    moments = moments.to(torch::kFloat32).contiguous();
    minimax_h3_vae_require_finite(moments, "encoder.final_moments");
    minimax_h3_vae_trace(hook, "encoder.final_moments", -1, moments);
    return MiniMaxH3VAEDiagonalGaussianDistribution(moments);
  }

  torch::Tensor decode(const torch::Tensor& denormalized_latents,
                       const MiniMaxH3VideoVAETraceHook& hook = nullptr) const {
    verify_loaded_weights();
    validate_video(denormalized_latents,
                   MiniMaxH3VideoVAEConfig::kLatentChannels,
                   "decode input",
                   /*require_spatial_alignment=*/false);
    if (denormalized_latents.device() != post_quant_conv_->weight().device()) {
      throw std::invalid_argument(
          "MiniMax-H3 video VAE decode input is on the wrong device");
    }
    torch::NoGradGuard no_grad;
    torch::Tensor latents = denormalized_latents.to(torch::kFloat32);
    const c10::DeviceType device_type = latents.device().type();
    const bool use_npu_autocast = device_type == c10::DeviceType::PrivateUse1;
    const bool previous_autocast =
        torch::autocast::is_autocast_enabled(device_type);
    const torch::ScalarType previous_dtype =
        torch::autocast::get_autocast_dtype(device_type);
    if (use_npu_autocast) {
      torch::autocast::set_autocast_dtype(device_type, torch::kFloat16);
      torch::autocast::set_autocast_enabled(device_type, true);
    }
    torch::Tensor decoded;
    try {
      decoded = decode_temporal(latents, hook);
    } catch (...) {
      if (use_npu_autocast) {
        torch::autocast::set_autocast_enabled(device_type, previous_autocast);
        torch::autocast::set_autocast_dtype(device_type, previous_dtype);
      }
      throw;
    }
    if (use_npu_autocast) {
      torch::autocast::set_autocast_enabled(device_type, previous_autocast);
      torch::autocast::set_autocast_dtype(device_type, previous_dtype);
    }
    minimax_h3_vae_require_finite(decoded, "decoder.raw");
    minimax_h3_vae_trace(hook, "decoder.raw", -1, decoded);
    return decoded.contiguous();
  }

  torch::Tensor encode_condition(
      const torch::Tensor& pixels,
      const torch::Tensor& epsilon,
      const MiniMaxH3VideoVAETraceHook& hook = nullptr) const {
    torch::Tensor prepared = imagenet_preprocess(pixels);
    MiniMaxH3VAEDiagonalGaussianDistribution posterior = encode(prepared, hook);
    torch::Tensor rounded =
        posterior.sample(epsilon.to(posterior.mean().device(), torch::kFloat32),
                         /*fp16_round_trip=*/true);
    minimax_h3_vae_trace(hook, "posterior.rounded_fp32", -1, rounded);
    torch::Tensor normalized = normalize_latents(rounded);
    minimax_h3_vae_trace(hook, "posterior.normalized", -1, normalized);
    return normalized;
  }

  torch::Tensor decode_normalized(
      const torch::Tensor& normalized_latents,
      const MiniMaxH3VideoVAETraceHook& hook = nullptr) const {
    torch::Tensor denormalized = denormalize_latents(normalized_latents);
    minimax_h3_vae_trace(
        hook, "decoder.denormalized_latents", -1, denormalized);
    return imagenet_postprocess(decode(denormalized, hook));
  }

  static torch::Tensor imagenet_preprocess(const torch::Tensor& pixels) {
    validate_channel_tensor(pixels, 3, "ImageNet preprocess input");
    torch::Tensor value = pixels.to(torch::kFloat32);
    if (pixels.scalar_type() == torch::kUInt8) {
      value = value / 255.0;
    }
    const torch::Tensor mean = channel_values(value, imagenet_mean());
    const torch::Tensor std = channel_values(value, imagenet_std());
    value = (value - mean) / std;
    minimax_h3_vae_require_finite(value, "ImageNet preprocess output");
    return value.contiguous();
  }

  static torch::Tensor imagenet_postprocess(
      const torch::Tensor& normalized_pixels) {
    validate_channel_tensor(normalized_pixels, 3, "ImageNet postprocess input");
    torch::Tensor value = normalized_pixels.to(torch::kFloat32);
    value = value * channel_values(value, imagenet_std()) +
            channel_values(value, imagenet_mean());
    value = value.clamp(0.0, 1.0);
    minimax_h3_vae_require_finite(value, "ImageNet postprocess output");
    return value.contiguous();
  }

  static torch::Tensor normalize_latents(const torch::Tensor& latents) {
    validate_channel_tensor(latents,
                            MiniMaxH3VideoVAEConfig::kLatentChannels,
                            "latent normalize input");
    torch::Tensor value = latents.to(torch::kFloat32);
    value = (value - channel_values(value, latent_mean())) /
            channel_values(value, latent_std());
    minimax_h3_vae_require_finite(value, "latent normalize output");
    return value.contiguous();
  }

  static torch::Tensor denormalize_latents(const torch::Tensor& latents) {
    validate_channel_tensor(latents,
                            MiniMaxH3VideoVAEConfig::kLatentChannels,
                            "latent denormalize input");
    torch::Tensor value = latents.to(torch::kFloat32);
    value = value * channel_values(value, latent_std()) +
            channel_values(value, latent_mean());
    minimax_h3_vae_require_finite(value, "latent denormalize output");
    return value.contiguous();
  }

  void load_source_weights(
      const std::vector<std::unique_ptr<StateDict>>& shards) {
    if (loaded_) {
      throw std::logic_error(
          "MiniMax-H3 video VAE source weights are already loaded");
    }
    const auto source =
        MiniMaxH3VideoVAESourceLayoutValidator::validate(shards);
    const auto specs =
        MiniMaxH3VideoVAESourceLayoutValidator::expected_source_tensors();
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
    if (targets.size() != specs.size()) {
      throw std::logic_error(
          "MiniMax-H3 video VAE module/source tensor counts differ");
    }

    torch::NoGradGuard no_grad;
    for (const MiniMaxH3VideoVAESourceTensorSpec& spec : specs) {
      const auto target_iterator = targets.find(spec.name);
      if (target_iterator == targets.end()) {
        throw std::logic_error(
            "MiniMax-H3 video VAE module is missing registered tensor `" +
            spec.name + "`");
      }
      torch::Tensor& target = *target_iterator->second;
      if (target.scalar_type() != torch::kFloat32 ||
          target.sizes().vec() != spec.shape) {
        throw std::logic_error(
            "MiniMax-H3 video VAE registered tensor metadata mismatch for `" +
            spec.name + "`");
      }
      torch::Tensor value = *source.at(spec.name);
      if (spec.name.ends_with(".attn.to_qkv.weight") ||
          spec.name.ends_with(".attn.to_qkv.bias")) {
        value = minimax_h3_vae_reorder_per_head_qkv(value);
      } else if (spec.name.ends_with(".ff.w1.weight") ||
                 spec.name.ends_with(".ff.w1.bias")) {
        value = minimax_h3_vae_reorder_gate_up_to_up_gate(value);
      }
      if (spec.name == "decoder.mask_token" &&
          value.count_nonzero().item<int64_t>() != 0) {
        throw std::invalid_argument(
            "MiniMax-H3 video VAE decoder.mask_token must be all zero");
      }
      target.copy_(value.to(target.device(), torch::kFloat32));
      loaded_source_names_.emplace(spec.name);
    }
    loaded_ = true;
    verify_loaded_weights();
  }

  void load_model(DiTFolderLoader& loader) {
    validate_config_files(loader.model_weights_path());
    auto& shards = loader.get_state_dicts();
    if (!shards.empty()) {
      load_source_weights(shards);
      return;
    }

    std::vector<std::unique_ptr<StateDict>> source_shard;
    source_shard.emplace_back(StateDictFromSafeTensor::load(
        loader.model_weights_path() + "/source/model.safetensors"));
    load_source_weights(source_shard);
  }

  void verify_loaded_weights() const {
    const auto specs =
        MiniMaxH3VideoVAESourceLayoutValidator::expected_source_tensors();
    if (!loaded_ || loaded_source_names_.size() != specs.size()) {
      throw std::logic_error(
          "MiniMax-H3 video VAE does not have all 560 source tensors loaded");
    }
    for (const MiniMaxH3VideoVAESourceTensorSpec& spec : specs) {
      if (!loaded_source_names_.contains(spec.name)) {
        throw std::logic_error(
            "MiniMax-H3 video VAE did not load source tensor `" + spec.name +
            "`");
      }
    }
    for (const auto& parameter : named_parameters(/*recurse=*/true)) {
      if (parameter.value().scalar_type() != torch::kFloat32) {
        throw std::logic_error("MiniMax-H3 video VAE parameter is not FP32: `" +
                               parameter.key() + "`");
      }
    }
    for (const auto& buffer : named_buffers(/*recurse=*/true)) {
      if (buffer.value().scalar_type() != torch::kFloat32) {
        throw std::logic_error("MiniMax-H3 video VAE buffer is not FP32: `" +
                               buffer.key() + "`");
      }
    }
  }

  bool is_loaded() const { return loaded_; }

 private:
  struct TilePlan {
    std::vector<int64_t> starts;
    std::vector<int64_t> lengths;
    std::vector<int64_t> overlaps;
  };

  static void validate_channel_tensor(const torch::Tensor& value,
                                      int64_t channels,
                                      std::string_view name) {
    if (!value.defined() || (value.dim() != 4 && value.dim() != 5) ||
        value.size(0) <= 0 || value.size(1) != channels ||
        value.size(-2) <= 0 || value.size(-1) <= 0 ||
        (value.dim() == 5 && value.size(2) <= 0) ||
        (!value.is_floating_point() && value.scalar_type() != torch::kUInt8)) {
      throw std::invalid_argument("MiniMax-H3 " + std::string(name) +
                                  " has an invalid shape or dtype");
    }
    minimax_h3_vae_require_finite(value.to(torch::kFloat32), name);
  }

  static nlohmann::json parse_config(const std::string& path) {
    JsonReader reader;
    if (!reader.parse(path)) {
      throw std::invalid_argument("MiniMax-H3 video VAE cannot parse config `" +
                                  path + "`");
    }
    return reader.data();
  }

  static void require_config_value(const nlohmann::json& config,
                                   const std::string& path,
                                   const std::string& key,
                                   const nlohmann::json& expected) {
    if (!config.contains(key) || config.at(key) != expected) {
      throw std::invalid_argument("MiniMax-H3 video VAE config `" + path +
                                  "` has invalid field `" + key + "`");
    }
  }

  static void validate_config_files(const std::string& component_path) {
    const std::string wrapper_path = component_path + "/config.json";
    const nlohmann::json wrapper = parse_config(wrapper_path);
    require_config_value(
        wrapper, wrapper_path, "_class_name", "MiniMaxH3VideoVAE");
    require_config_value(wrapper, wrapper_path, "mode", "standalone");
    require_config_value(wrapper, wrapper_path, "source_path", "source");
    require_config_value(
        wrapper, wrapper_path, "source_class_name", "AutoencoderKLLegacy");
    require_config_value(wrapper, wrapper_path, "vae_clip_length", 17);
    require_config_value(wrapper, wrapper_path, "vae_token_drop", 3);
    require_config_value(wrapper, wrapper_path, "vae_encoder_tiling", 1);
    require_config_value(wrapper, wrapper_path, "vae_decoder_tiling", 1);
    require_config_value(wrapper, wrapper_path, "vae_tile_size", 256);
    require_config_value(wrapper, wrapper_path, "vae_tile_overlap_min", 64);
    require_config_value(wrapper, wrapper_path, "latent_channels", 24);
    if (!wrapper.contains("latents_mean") || !wrapper.contains("latents_std") ||
        wrapper.at("latents_mean").get<std::vector<float>>() != latent_mean() ||
        wrapper.at("latents_std").get<std::vector<float>>() != latent_std()) {
      throw std::invalid_argument(
          "MiniMax-H3 video VAE latent statistics do not match the release");
    }

    const std::string source_path = component_path + "/source/config.json";
    const nlohmann::json source = parse_config(source_path);
    require_config_value(
        source, source_path, "_class_name", "AutoencoderKLLegacy");
    require_config_value(source, source_path, "causal_encoder", true);
    require_config_value(source, source_path, "causal_decoder", false);
    require_config_value(source, source_path, "use_3d_conv", true);
    require_config_value(source, source_path, "use_vit_decoder", true);
    require_config_value(source, source_path, "use_t_isolated_gn", true);
    require_config_value(source, source_path, "padding_mode", "reflect");
    require_config_value(source, source_path, "pixel_norm_type", "imagenet");
    require_config_value(source, source_path, "ch", 128);
    require_config_value(source, source_path, "ch_mult", {1, 2, 2, 4, 4, 8});
    require_config_value(source, source_path, "num_res_blocks", 2);
    require_config_value(source, source_path, "space_down", {2, 2, 2, 2, 1, 1});
    require_config_value(source, source_path, "time_down", {1, 2, 2, 1, 1, 1});
    require_config_value(source, source_path, "embed_dim", 24);
    require_config_value(source, source_path, "z_channels", 24);
    const auto& vit = source.at("vit_decoder_kwargs");
    require_config_value(vit, source_path, "heads", 32);
    require_config_value(vit, source_path, "dim_head", 64);
    require_config_value(vit, source_path, "num_layers", 36);
    require_config_value(vit, source_path, "norm_type", "rms_norm");
    require_config_value(vit, source_path, "qk_norm_type", "rms_norm");
    require_config_value(vit, source_path, "qk_norm_affine", false);
    require_config_value(vit, source_path, "ffn_activation_fn", "silu");
    require_config_value(vit, source_path, "ffn_use_gated", true);
    require_config_value(vit, source_path, "rope_theta", 100.0);
    require_config_value(vit, source_path, "rope_dim_ratio", 0.75);
  }

  static void validate_video(const torch::Tensor& value,
                             int64_t channels,
                             std::string_view name,
                             bool require_spatial_alignment = true) {
    if (!value.defined() || value.dim() != 5 || value.size(0) <= 0 ||
        value.size(1) != channels || value.size(2) <= 0 || value.size(3) <= 0 ||
        value.size(4) <= 0 || !value.is_floating_point()) {
      throw std::invalid_argument("MiniMax-H3 " + std::string(name) +
                                  " must be floating-point [B,C,T,H,W]");
    }
    if (require_spatial_alignment &&
        (value.size(3) % MiniMaxH3VideoVAEConfig::kSpatialRatio != 0 ||
         value.size(4) % MiniMaxH3VideoVAEConfig::kSpatialRatio != 0)) {
      throw std::invalid_argument(
          "MiniMax-H3 encode height and width must be divisible by 16");
    }
    minimax_h3_vae_require_finite(value, name);
  }

  static torch::Tensor channel_values(const torch::Tensor& reference,
                                      const std::vector<float>& values) {
    std::vector<int64_t> shape(static_cast<size_t>(reference.dim()), 1);
    shape[1] = static_cast<int64_t>(values.size());
    return torch::tensor(values,
                         torch::TensorOptions()
                             .dtype(torch::kFloat32)
                             .device(reference.device()))
        .view(shape);
  }

  static const std::vector<float>& imagenet_mean() {
    static const std::vector<float> values = {0.485F, 0.456F, 0.406F};
    return values;
  }

  static const std::vector<float>& imagenet_std() {
    static const std::vector<float> values = {0.229F, 0.224F, 0.225F};
    return values;
  }

  static const std::vector<float>& latent_mean() {
    static const std::vector<float> values = {
        0.858090341091156F,  -0.960659146308899F, 1.066164016723633F,
        -0.509032547473907F, -0.272758185863495F, -1.367541432380676F,
        -0.255325496196747F, -0.269075542688370F, -0.537684082984924F,
        -0.046409729868174F, 0.665737032890320F,  0.196901276707649F,
        -0.546060800552368F, -0.403534203767776F, -0.236830249428749F,
        0.259284526109695F,  -0.301339447498322F, 0.211341992020607F,
        -1.120684862136841F, 0.358193337917328F,  -0.042251437902451F,
        0.260482996702194F,  0.228640928864479F,  0.705603182315826F};
    return values;
  }

  static const std::vector<float>& latent_std() {
    static const std::vector<float> values = {
        1.222377419471741F, 1.276726365089416F, 1.683177471160889F,
        1.754945516586304F, 1.563621640205383F, 2.194143533706665F,
        0.965313792228699F, 1.056988596916199F, 0.841948926448822F,
        0.772995293140411F, 1.895593762397766F, 0.946841835975647F,
        0.799680948257446F, 0.449889004230499F, 0.719739973545074F,
        0.693629324436188F, 2.961095094680786F, 2.769419908523560F,
        3.049618482589722F, 2.108805418014526F, 3.276226282119751F,
        3.162735700607300F, 2.281681299209595F, 2.612784385681152F};
    return values;
  }

  static TilePlan split_tiles(int64_t length) {
    constexpr int64_t kTile = MiniMaxH3VideoVAEConfig::kTileSize;
    constexpr int64_t kMinOverlap = MiniMaxH3VideoVAEConfig::kTileMinOverlap;
    constexpr int64_t kRatio = MiniMaxH3VideoVAEConfig::kSpatialRatio;
    if (length <= 0 || length % kRatio != 0) {
      throw std::invalid_argument(
          "MiniMax-H3 tile axis must be positive and divisible by 16");
    }
    if (kTile >= length) {
      return {{0}, {length}, {}};
    }

    int64_t count = (length + kTile - 1) / kTile;
    while (kTile * count - kMinOverlap * (count - 1) < length) {
      ++count;
    }
    TilePlan plan;
    plan.overlaps.assign(static_cast<size_t>(count - 1), kMinOverlap);
    const int64_t remaining =
        kTile * count - kMinOverlap * (count - 1) - length;
    if (remaining % kRatio != 0) {
      throw std::logic_error("MiniMax-H3 tile slack is not latent-aligned");
    }
    for (int64_t index = 0; index < remaining / kRatio; ++index) {
      plan.overlaps[static_cast<size_t>(index % (count - 1))] += kRatio;
    }
    plan.starts.reserve(count);
    plan.lengths.assign(static_cast<size_t>(count), kTile);
    plan.starts.push_back(0);
    for (int64_t index = 0; index < count - 1; ++index) {
      plan.starts.push_back(plan.starts.back() + kTile -
                            plan.overlaps[static_cast<size_t>(index)]);
    }
    if (plan.starts.back() + plan.lengths.back() != length) {
      throw std::logic_error(
          "MiniMax-H3 tile plan does not cover its axis exactly");
    }
    return plan;
  }

  static torch::Tensor blend(const torch::Tensor& a,
                             const torch::Tensor& b,
                             int64_t extent,
                             int64_t dim) {
    const int64_t axis = dim < 0 ? b.dim() + dim : dim;
    extent = std::min({a.size(axis), b.size(axis), extent});
    if (extent <= 0) {
      return b;
    }
    torch::Tensor positions = torch::arange(
        extent,
        torch::TensorOptions().dtype(b.scalar_type()).device(b.device()));
    std::vector<int64_t> shape(static_cast<size_t>(b.dim()), 1);
    shape[static_cast<size_t>(axis)] = extent;
    torch::Tensor weight_a =
        (1.0 - positions / static_cast<double>(extent)).view(shape);
    torch::Tensor weight_b =
        (positions / static_cast<double>(extent)).view(shape);
    torch::Tensor blended =
        a.slice(axis, a.size(axis) - extent, a.size(axis)) * weight_a +
        b.slice(axis, 0, extent) * weight_b;
    if (extent == b.size(axis)) {
      return blended;
    }
    return torch::cat({blended, b.slice(axis, extent, b.size(axis))}, axis);
  }

  static torch::Tensor stitch_tiles(
      const std::vector<std::vector<torch::Tensor>>& tiles,
      const std::vector<int64_t>& height_overlaps,
      const std::vector<int64_t>& width_overlaps) {
    std::vector<torch::Tensor> result_rows;
    result_rows.reserve(tiles.size());
    for (size_t row_index = 0; row_index < tiles.size(); ++row_index) {
      const auto& row = tiles[row_index];
      std::vector<torch::Tensor> result_row;
      result_row.reserve(row.size());
      for (size_t column_index = 0; column_index < row.size(); ++column_index) {
        torch::Tensor tile = row[column_index];
        if (row_index > 0) {
          tile = blend(tiles[row_index - 1][column_index],
                       tile,
                       height_overlaps[row_index - 1],
                       -2);
        }
        if (column_index > 0) {
          tile = blend(row[column_index - 1],
                       tile,
                       width_overlaps[column_index - 1],
                       -1);
        }
        if (row_index + 1 < tiles.size()) {
          tile = tile.slice(-2, 0, tile.size(-2) - height_overlaps[row_index]);
        }
        if (column_index + 1 < row.size()) {
          tile =
              tile.slice(-1, 0, tile.size(-1) - width_overlaps[column_index]);
        }
        result_row.emplace_back(std::move(tile));
      }
      result_rows.emplace_back(torch::cat(result_row, -1));
    }
    return torch::cat(result_rows, -2);
  }

  torch::Tensor encode_clip(const torch::Tensor& input,
                            const MiniMaxH3VideoVAETraceHook& hook) const {
    const TilePlan height_plan = split_tiles(input.size(-2));
    const TilePlan width_plan = split_tiles(input.size(-1));
    std::vector<std::vector<torch::Tensor>> rows;
    rows.reserve(height_plan.starts.size());
    for (size_t row = 0; row < height_plan.starts.size(); ++row) {
      std::vector<torch::Tensor> columns;
      columns.reserve(width_plan.starts.size());
      for (size_t column = 0; column < width_plan.starts.size(); ++column) {
        torch::Tensor tile =
            input
                .slice(-2,
                       height_plan.starts[row],
                       height_plan.starts[row] + height_plan.lengths[row])
                .slice(-1,
                       width_plan.starts[column],
                       width_plan.starts[column] + width_plan.lengths[column]);
        torch::Tensor moments =
            quant_conv_->forward(encoder_->forward(tile.contiguous(), hook));
        minimax_h3_vae_trace(hook, "encoder.quant_conv", -1, moments);
        columns.emplace_back(std::move(moments));
      }
      rows.emplace_back(std::move(columns));
    }
    std::vector<int64_t> latent_height_overlaps;
    std::vector<int64_t> latent_width_overlaps;
    for (int64_t overlap : height_plan.overlaps) {
      latent_height_overlaps.push_back(overlap /
                                       MiniMaxH3VideoVAEConfig::kSpatialRatio);
    }
    for (int64_t overlap : width_plan.overlaps) {
      latent_width_overlaps.push_back(overlap /
                                      MiniMaxH3VideoVAEConfig::kSpatialRatio);
    }
    return stitch_tiles(rows, latent_height_overlaps, latent_width_overlaps);
  }

  torch::Tensor decode_clip(const torch::Tensor& input,
                            const MiniMaxH3VideoVAETraceHook& hook) const {
    const int64_t pixel_height =
        input.size(-2) * MiniMaxH3VideoVAEConfig::kSpatialRatio;
    const int64_t pixel_width =
        input.size(-1) * MiniMaxH3VideoVAEConfig::kSpatialRatio;
    const TilePlan height_plan = split_tiles(pixel_height);
    const TilePlan width_plan = split_tiles(pixel_width);
    std::vector<std::vector<torch::Tensor>> rows;
    rows.reserve(height_plan.starts.size());
    for (size_t row = 0; row < height_plan.starts.size(); ++row) {
      std::vector<torch::Tensor> columns;
      columns.reserve(width_plan.starts.size());
      for (size_t column = 0; column < width_plan.starts.size(); ++column) {
        const int64_t latent_y =
            height_plan.starts[row] / MiniMaxH3VideoVAEConfig::kSpatialRatio;
        const int64_t latent_x =
            width_plan.starts[column] / MiniMaxH3VideoVAEConfig::kSpatialRatio;
        const int64_t latent_height =
            height_plan.lengths[row] / MiniMaxH3VideoVAEConfig::kSpatialRatio;
        const int64_t latent_width =
            width_plan.lengths[column] / MiniMaxH3VideoVAEConfig::kSpatialRatio;
        torch::Tensor tile = input.slice(-2, latent_y, latent_y + latent_height)
                                 .slice(-1, latent_x, latent_x + latent_width)
                                 .contiguous();
        torch::Tensor projected = post_quant_conv_->forward(tile);
        minimax_h3_vae_trace(hook, "decoder.post_quant_conv", -1, projected);
        columns.emplace_back(decoder_->forward(projected, hook));
      }
      rows.emplace_back(std::move(columns));
    }
    return stitch_tiles(rows, height_plan.overlaps, width_plan.overlaps);
  }

  torch::Tensor encode_temporal(const torch::Tensor& input,
                                const MiniMaxH3VideoVAETraceHook& hook) const {
    if (input.size(2) == 1) {
      return encode_clip(input, hook);
    }
    torch::Tensor padded = input;
    const int64_t pad_frames =
        (MiniMaxH3VideoVAEConfig::kClipLength -
         input.size(2) % MiniMaxH3VideoVAEConfig::kClipLength) %
        MiniMaxH3VideoVAEConfig::kClipLength;
    if (pad_frames > 0) {
      padded = torch::cat({input,
                           input.slice(2, input.size(2) - 1, input.size(2))
                               .repeat({1, 1, pad_frames, 1, 1})},
                          2);
    }
    std::vector<torch::Tensor> moments;
    const int64_t chunks =
        padded.size(2) / MiniMaxH3VideoVAEConfig::kClipLength;
    moments.reserve(chunks);
    for (int64_t index = 0; index < chunks; ++index) {
      const int64_t start = index * MiniMaxH3VideoVAEConfig::kClipLength;
      moments.emplace_back(encode_clip(
          padded.slice(2, start, start + MiniMaxH3VideoVAEConfig::kClipLength),
          hook));
    }
    torch::Tensor output = torch::cat(moments, 2);
    if (MiniMaxH3VideoVAEConfig::kTokenDrop > 0) {
      output = output.slice(
          2, 0, output.size(2) - MiniMaxH3VideoVAEConfig::kTokenDrop);
    }
    return output;
  }

  static int64_t decode_pad_frames(int64_t padded_latent_frames,
                                   int64_t pad_tokens) {
    if (pad_tokens <= 0) {
      return 0;
    }
    const int64_t unpadded = padded_latent_frames - pad_tokens;
    const int64_t intra_tail = MiniMaxH3VideoVAEConfig::kClipLength %
                               MiniMaxH3VideoVAEConfig::kTemporalRatio;
    int64_t frames = 0;
    for (int64_t index = 0; index < pad_tokens; ++index) {
      frames +=
          (intra_tail > 0 &&
           (unpadded + index) % MiniMaxH3VideoVAEConfig::kTokensPerChunk == 0)
              ? intra_tail
              : MiniMaxH3VideoVAEConfig::kTemporalRatio;
    }
    return frames;
  }

  torch::Tensor decode_temporal(const torch::Tensor& input,
                                const MiniMaxH3VideoVAETraceHook& hook) const {
    const int64_t logical_tokens =
        input.size(2) + MiniMaxH3VideoVAEConfig::kTokenDrop;
    const int64_t pad_tokens =
        (MiniMaxH3VideoVAEConfig::kTokensPerChunk -
         logical_tokens % MiniMaxH3VideoVAEConfig::kTokensPerChunk) %
        MiniMaxH3VideoVAEConfig::kTokensPerChunk;
    const int64_t chunks = (logical_tokens + pad_tokens) /
                               MiniMaxH3VideoVAEConfig::kTokensPerChunk -
                           1;
    if (chunks <= 0) {
      throw std::invalid_argument(
          "MiniMax-H3 decode requires at least seven latent frames");
    }
    torch::Tensor padded = input;
    if (pad_tokens > 0) {
      padded = torch::cat({input,
                           input.slice(2, input.size(2) - 1, input.size(2))
                               .repeat({1, 1, pad_tokens, 1, 1})},
                          2);
    }
    const int64_t pad_frames = decode_pad_frames(padded.size(2), pad_tokens);
    const int64_t total_frames = chunks * MiniMaxH3VideoVAEConfig::kClipLength +
                                 MiniMaxH3VideoVAEConfig::kFrameOverlap;
    const int64_t output_frames = total_frames - pad_frames;
    if (output_frames <= 0) {
      throw std::logic_error(
          "MiniMax-H3 decode temporal plan has no output frames");
    }

    torch::Tensor output;
    torch::Tensor overlap;
    int64_t write_position = 0;
    int64_t logical_output_frames = 0;
    auto write = [&](const torch::Tensor& part) {
      const int64_t part_frames = part.size(2);
      logical_output_frames += part_frames;
      if (!output.defined()) {
        std::vector<int64_t> shape = part.sizes().vec();
        shape[2] = output_frames;
        output = torch::empty(shape, part.options());
      }
      const int64_t copied =
          std::min(part_frames, output_frames - write_position);
      if (copied > 0) {
        output.slice(2, write_position, write_position + copied)
            .copy_(part.slice(2, 0, copied));
        write_position += copied;
      }
    };

    constexpr int64_t kDecodedChunkFrames =
        MiniMaxH3VideoVAEConfig::kTokensPerChunk *
        MiniMaxH3VideoVAEConfig::kTemporalRatio;
    for (int64_t index = 0; index < chunks; ++index) {
      const int64_t start = index * MiniMaxH3VideoVAEConfig::kTokensPerChunk;
      torch::Tensor clip = decode_clip(
          padded.slice(2,
                       start,
                       start + MiniMaxH3VideoVAEConfig::kTokensPerChunk +
                           MiniMaxH3VideoVAEConfig::kTokenOverlap),
          hook);
      torch::Tensor primary =
          clip.slice(2, 0, kDecodedChunkFrames)
              .slice(2,
                     MiniMaxH3VideoVAEConfig::kFramePrePadding,
                     kDecodedChunkFrames)
              .contiguous();
      if (overlap.defined()) {
        primary =
            blend(overlap, primary, MiniMaxH3VideoVAEConfig::kFrameOverlap, 2);
      }
      write(primary);
      overlap = clip.slice(2, kDecodedChunkFrames, clip.size(2))
                    .slice(2,
                           MiniMaxH3VideoVAEConfig::kFramePrePadding,
                           clip.size(2) - kDecodedChunkFrames)
                    .contiguous();
    }
    write(overlap);
    if (!output.defined() || logical_output_frames != total_frames ||
        write_position != output_frames) {
      throw std::logic_error(
          "MiniMax-H3 decode temporal stitch plan is inconsistent");
    }
    return output;
  }

  torch::TensorOptions options_;
  MiniMaxH3VAEEncoder3d encoder_{nullptr};
  MiniMaxH3VAECausalConv3d quant_conv_{nullptr};
  MiniMaxH3VAECausalConv3d post_quant_conv_{nullptr};
  MiniMaxH3VAEViTDecoder3d decoder_{nullptr};
  std::unordered_set<std::string> loaded_source_names_;
  bool loaded_ = false;
};
TORCH_MODULE(MiniMaxH3VideoVAE);

using AutoencoderKLMiniMaxH3 = MiniMaxH3VideoVAE;

}  // namespace xllm
