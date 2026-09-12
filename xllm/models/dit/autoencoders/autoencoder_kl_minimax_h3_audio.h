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
#include <ATen/ops/_weight_norm.h>
#include <torch/torch.h>

#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
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

struct MiniMaxH3AudioVAEConfig {
  static constexpr int64_t kInputChannels = 1;
  static constexpr int64_t kStereoChannels = 2;
  static constexpr int64_t kEncoderDim = 64;
  static constexpr int64_t kLatentDim = 2048;
  static constexpr int64_t kLatentChannels = 32;
  static constexpr int64_t kAttentionHeads = 8;
  static constexpr int64_t kAttentionHeadDim = 256;
  static constexpr int64_t kDecoderDim = 1024;
  static constexpr int64_t kSampleRate = 32000;
  static constexpr int64_t kHopLength = 800;
  static constexpr int64_t kAliasFreeRatio = 2;
  static constexpr int64_t kAliasFreeKernel = 12;
  static constexpr double kLayerNormEps = 1e-5;
  static constexpr int64_t kMaximumTraceEvents = 24;

  static constexpr std::array<int64_t, 5> kEncoderRates = {2, 4, 4, 5, 5};
  static constexpr std::array<int64_t, 7> kDecoderRates = {5, 5, 2, 2, 2, 2, 2};
  static constexpr std::array<int64_t, 7> kDecoderKernels =
      {9, 9, 4, 4, 4, 4, 4};
  static constexpr std::array<int64_t, 3> kResBlockKernels = {3, 7, 11};
  static constexpr std::array<int64_t, 3> kResBlockDilations = {1, 3, 5};
  static constexpr std::array<int64_t, 3> kEncoderResidualDilations = {1, 3, 9};
};

struct MiniMaxH3AudioVAESourceTensorSpec {
  std::string name;
  std::vector<int64_t> shape;
};

using MiniMaxH3AudioVAETraceHook = std::function<
    void(std::string_view name, int64_t index, const torch::Tensor& value)>;

inline std::string minimax_h3_audio_shape_string(
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

inline void minimax_h3_audio_require_finite(const torch::Tensor& value,
                                            std::string_view name) {
  if (!value.defined() || !value.is_floating_point() ||
      !torch::isfinite(value).all().item<bool>()) {
    throw std::invalid_argument(
        "MiniMax-H3 audio VAE tensor `" + std::string(name) +
        "` must be defined, floating-point, and finite");
  }
}

inline torch::TensorOptions minimax_h3_audio_fp32_options(
    const torch::TensorOptions& options) {
  return options.dtype(torch::kFloat32);
}

class MiniMaxH3AudioVAETraceContext final {
 public:
  explicit MiniMaxH3AudioVAETraceContext(const MiniMaxH3AudioVAETraceHook& hook)
      : hook_(hook) {}

  void emit(std::string_view name, int64_t index, const torch::Tensor& value) {
    if (!hook_) {
      return;
    }
    if (events_ >= MiniMaxH3AudioVAEConfig::kMaximumTraceEvents) {
      throw std::logic_error(
          "MiniMax-H3 audio VAE exceeded its bounded trace event budget");
    }
    ++events_;
    hook_(name, index, value);
  }

  int64_t events() const { return events_; }

 private:
  const MiniMaxH3AudioVAETraceHook& hook_;
  int64_t events_ = 0;
};

class MiniMaxH3AudioAutocastGuard final {
 public:
  explicit MiniMaxH3AudioAutocastGuard(const torch::Device& device)
      : device_type_(device.type()),
        was_enabled_(torch::autocast::is_autocast_enabled(device_type_)) {
    if (was_enabled_) {
      torch::autocast::set_autocast_enabled(device_type_, false);
    }
  }

  ~MiniMaxH3AudioAutocastGuard() {
    if (was_enabled_) {
      torch::autocast::set_autocast_enabled(device_type_, true);
    }
  }

  MiniMaxH3AudioAutocastGuard(const MiniMaxH3AudioAutocastGuard&) = delete;
  MiniMaxH3AudioAutocastGuard& operator=(const MiniMaxH3AudioAutocastGuard&) =
      delete;

 private:
  c10::DeviceType device_type_;
  bool was_enabled_;
};

class MiniMaxH3AudioConv1dImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AudioConv1dImpl(int64_t input_channels,
                           int64_t output_channels,
                           int64_t kernel_size,
                           int64_t stride,
                           int64_t padding,
                           int64_t dilation,
                           bool bias,
                           const torch::TensorOptions& options)
      : stride_(stride), padding_(padding), dilation_(dilation) {
    const torch::TensorOptions fp32 = minimax_h3_audio_fp32_options(options);
    weight_ = register_parameter(
        "weight",
        torch::empty({output_channels, input_channels, kernel_size}, fp32));
    if (bias) {
      bias_ = register_parameter("bias", torch::empty({output_channels}, fp32));
    }
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    validate_input(input);
    return torch::nn::functional::conv1d(
        input,
        weight_,
        torch::nn::functional::Conv1dFuncOptions()
            .bias(bias_)
            .stride(stride_)
            .padding(padding_)
            .dilation(dilation_));
  }

  const torch::Tensor& weight() const { return weight_; }
  const torch::Tensor& bias() const { return bias_; }

 private:
  void validate_input(const torch::Tensor& input) const {
    if (!input.defined() || input.dim() != 3 ||
        input.scalar_type() != torch::kFloat32 ||
        input.device() != weight_.device()) {
      throw std::invalid_argument(
          "MiniMax-H3 Conv1d expects an FP32 [B,C,T] tensor on its weight "
          "device");
    }
  }

  torch::Tensor weight_;
  torch::Tensor bias_;
  int64_t stride_;
  int64_t padding_;
  int64_t dilation_;
};
TORCH_MODULE(MiniMaxH3AudioConv1d);

class MiniMaxH3AudioConvTranspose1dImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AudioConvTranspose1dImpl(int64_t input_channels,
                                    int64_t output_channels,
                                    int64_t kernel_size,
                                    int64_t stride,
                                    int64_t padding,
                                    bool bias,
                                    const torch::TensorOptions& options)
      : stride_(stride), padding_(padding) {
    const torch::TensorOptions fp32 = minimax_h3_audio_fp32_options(options);
    weight_ = register_parameter(
        "weight",
        torch::empty({input_channels, output_channels, kernel_size}, fp32));
    if (bias) {
      bias_ = register_parameter("bias", torch::empty({output_channels}, fp32));
    }
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    if (!input.defined() || input.dim() != 3 ||
        input.scalar_type() != torch::kFloat32 ||
        input.device() != weight_.device()) {
      throw std::invalid_argument(
          "MiniMax-H3 ConvTranspose1d expects an FP32 [B,C,T] tensor on its "
          "weight device");
    }
    return torch::nn::functional::conv_transpose1d(
        input,
        weight_,
        torch::nn::functional::ConvTranspose1dFuncOptions()
            .bias(bias_)
            .stride(stride_)
            .padding(padding_));
  }

  const torch::Tensor& weight() const { return weight_; }

 private:
  torch::Tensor weight_;
  torch::Tensor bias_;
  int64_t stride_;
  int64_t padding_;
};
TORCH_MODULE(MiniMaxH3AudioConvTranspose1d);

class MiniMaxH3AudioLinearImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AudioLinearImpl(int64_t input_size,
                           int64_t output_size,
                           bool bias,
                           const torch::TensorOptions& options) {
    const torch::TensorOptions fp32 = minimax_h3_audio_fp32_options(options);
    weight_ = register_parameter("weight",
                                 torch::empty({output_size, input_size}, fp32));
    if (bias) {
      bias_ = register_parameter("bias", torch::empty({output_size}, fp32));
    }
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    return torch::nn::functional::linear(input, weight_, bias_);
  }

  const torch::Tensor& weight() const { return weight_; }

 private:
  torch::Tensor weight_;
  torch::Tensor bias_;
};
TORCH_MODULE(MiniMaxH3AudioLinear);

class MiniMaxH3AudioLayerNormImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AudioLayerNormImpl(int64_t size, const torch::TensorOptions& options)
      : size_(size) {
    const torch::TensorOptions fp32 = minimax_h3_audio_fp32_options(options);
    weight_ = register_parameter("weight", torch::empty({size}, fp32));
    bias_ = register_parameter("bias", torch::empty({size}, fp32));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    return torch::nn::functional::layer_norm(
        input,
        torch::nn::functional::LayerNormFuncOptions({size_})
            .weight(weight_)
            .bias(bias_)
            .eps(MiniMaxH3AudioVAEConfig::kLayerNormEps));
  }

 private:
  int64_t size_;
  torch::Tensor weight_;
  torch::Tensor bias_;
};
TORCH_MODULE(MiniMaxH3AudioLayerNorm);

class MiniMaxH3AudioSnake1dImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AudioSnake1dImpl(int64_t channels,
                            const torch::TensorOptions& options) {
    alpha_ = register_parameter(
        "alpha",
        torch::empty({1, channels, 1}, minimax_h3_audio_fp32_options(options)));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    return input + torch::sin(alpha_ * input).square() / (alpha_ + 1e-9);
  }

 private:
  torch::Tensor alpha_;
};
TORCH_MODULE(MiniMaxH3AudioSnake1d);

class MiniMaxH3AudioSnakeBetaImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AudioSnakeBetaImpl(int64_t channels,
                              const torch::TensorOptions& options) {
    const torch::TensorOptions fp32 = minimax_h3_audio_fp32_options(options);
    alpha_ = register_parameter("alpha", torch::empty({channels}, fp32));
    beta_ = register_parameter("beta", torch::empty({channels}, fp32));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    const torch::Tensor alpha = torch::exp(alpha_).view({1, -1, 1});
    const torch::Tensor beta = torch::exp(beta_).view({1, -1, 1});
    return input + torch::sin(alpha * input).square() / (beta + 1e-9);
  }

 private:
  torch::Tensor alpha_;
  torch::Tensor beta_;
};
TORCH_MODULE(MiniMaxH3AudioSnakeBeta);

class MiniMaxH3AudioLowPassFilter1dImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AudioLowPassFilter1dImpl(int64_t stride,
                                    int64_t kernel_size,
                                    const torch::TensorOptions& options)
      : pad_left_(kernel_size / 2 - (kernel_size % 2 == 0 ? 1 : 0)),
        pad_right_(kernel_size / 2),
        stride_(stride) {
    filter_ =
        register_buffer("filter",
                        torch::empty({1, 1, kernel_size},
                                     minimax_h3_audio_fp32_options(options)));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    const int64_t channels = input.size(1);
    torch::Tensor hidden = torch::nn::functional::pad(
        input,
        torch::nn::functional::PadFuncOptions({pad_left_, pad_right_})
            .mode(torch::kReplicate));
    return torch::nn::functional::conv1d(
        hidden,
        filter_.expand({channels, -1, -1}),
        torch::nn::functional::Conv1dFuncOptions().stride(stride_).groups(
            channels));
  }

 private:
  torch::Tensor filter_;
  int64_t pad_left_;
  int64_t pad_right_;
  int64_t stride_;
};
TORCH_MODULE(MiniMaxH3AudioLowPassFilter1d);

class MiniMaxH3AudioUpSample1dImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AudioUpSample1dImpl(int64_t ratio,
                               int64_t kernel_size,
                               const torch::TensorOptions& options)
      : ratio_(ratio),
        pad_(kernel_size / ratio - 1),
        pad_left_(pad_ * ratio + (kernel_size - ratio) / 2),
        pad_right_(pad_ * ratio + (kernel_size - ratio + 1) / 2) {
    filter_ =
        register_buffer("filter",
                        torch::empty({1, 1, kernel_size},
                                     minimax_h3_audio_fp32_options(options)));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    const int64_t channels = input.size(1);
    torch::Tensor hidden = torch::nn::functional::pad(
        input,
        torch::nn::functional::PadFuncOptions({pad_, pad_})
            .mode(torch::kReplicate));
    hidden = static_cast<double>(ratio_) *
             torch::nn::functional::conv_transpose1d(
                 hidden,
                 filter_.expand({channels, -1, -1}),
                 torch::nn::functional::ConvTranspose1dFuncOptions()
                     .stride(ratio_)
                     .groups(channels));
    return hidden.slice(-1, pad_left_, hidden.size(-1) - pad_right_);
  }

 private:
  torch::Tensor filter_;
  int64_t ratio_;
  int64_t pad_;
  int64_t pad_left_;
  int64_t pad_right_;
};
TORCH_MODULE(MiniMaxH3AudioUpSample1d);

class MiniMaxH3AudioDownSample1dImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AudioDownSample1dImpl(int64_t ratio,
                                 int64_t kernel_size,
                                 const torch::TensorOptions& options) {
    lowpass_ = register_module(
        "lowpass", MiniMaxH3AudioLowPassFilter1d(ratio, kernel_size, options));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    return lowpass_->forward(input);
  }

 private:
  MiniMaxH3AudioLowPassFilter1d lowpass_{nullptr};
};
TORCH_MODULE(MiniMaxH3AudioDownSample1d);

class MiniMaxH3AudioActivation1dImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AudioActivation1dImpl(int64_t channels,
                                 const torch::TensorOptions& options) {
    act_ = register_module("act", MiniMaxH3AudioSnakeBeta(channels, options));
    upsample_ = register_module(
        "upsample",
        MiniMaxH3AudioUpSample1d(MiniMaxH3AudioVAEConfig::kAliasFreeRatio,
                                 MiniMaxH3AudioVAEConfig::kAliasFreeKernel,
                                 options));
    downsample_ = register_module(
        "downsample",
        MiniMaxH3AudioDownSample1d(MiniMaxH3AudioVAEConfig::kAliasFreeRatio,
                                   MiniMaxH3AudioVAEConfig::kAliasFreeKernel,
                                   options));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    return downsample_->forward(act_->forward(upsample_->forward(input)));
  }

 private:
  MiniMaxH3AudioSnakeBeta act_{nullptr};
  MiniMaxH3AudioUpSample1d upsample_{nullptr};
  MiniMaxH3AudioDownSample1d downsample_{nullptr};
};
TORCH_MODULE(MiniMaxH3AudioActivation1d);

class MiniMaxH3AudioResidualUnitImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AudioResidualUnitImpl(int64_t channels,
                                 int64_t dilation,
                                 const torch::TensorOptions& options) {
    block_ = register_module("block", torch::nn::ModuleList());
    snake1_ = MiniMaxH3AudioSnake1d(channels, options);
    conv1_ = MiniMaxH3AudioConv1d(
        channels, channels, 7, 1, 3 * dilation, dilation, true, options);
    snake2_ = MiniMaxH3AudioSnake1d(channels, options);
    conv2_ =
        MiniMaxH3AudioConv1d(channels, channels, 1, 1, 0, 1, true, options);
    block_->push_back(snake1_);
    block_->push_back(conv1_);
    block_->push_back(snake2_);
    block_->push_back(conv2_);
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    torch::Tensor residual = snake1_->forward(input);
    residual = conv1_->forward(residual);
    residual = snake2_->forward(residual);
    residual = conv2_->forward(residual);
    const int64_t pad = (input.size(-1) - residual.size(-1)) / 2;
    torch::Tensor shortcut = input;
    if (pad > 0) {
      shortcut = shortcut.slice(-1, pad, shortcut.size(-1) - pad);
    }
    return shortcut + residual;
  }

 private:
  torch::nn::ModuleList block_{nullptr};
  MiniMaxH3AudioSnake1d snake1_{nullptr};
  MiniMaxH3AudioConv1d conv1_{nullptr};
  MiniMaxH3AudioSnake1d snake2_{nullptr};
  MiniMaxH3AudioConv1d conv2_{nullptr};
};
TORCH_MODULE(MiniMaxH3AudioResidualUnit);

class MiniMaxH3AudioEncoderBlockImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AudioEncoderBlockImpl(int64_t input_channels,
                                 int64_t output_channels,
                                 int64_t stride,
                                 const torch::TensorOptions& options) {
    block_ = register_module("block", torch::nn::ModuleList());
    for (int64_t dilation :
         MiniMaxH3AudioVAEConfig::kEncoderResidualDilations) {
      MiniMaxH3AudioResidualUnit unit(input_channels, dilation, options);
      block_->push_back(unit);
      units_.emplace_back(std::move(unit));
    }
    snake_ = MiniMaxH3AudioSnake1d(input_channels, options);
    downsample_ = MiniMaxH3AudioConv1d(input_channels,
                                       output_channels,
                                       2 * stride,
                                       stride,
                                       (stride + 1) / 2,
                                       1,
                                       true,
                                       options);
    block_->push_back(snake_);
    block_->push_back(downsample_);
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    torch::Tensor hidden = input;
    for (const MiniMaxH3AudioResidualUnit& unit : units_) {
      hidden = unit->forward(hidden);
    }
    hidden = snake_->forward(hidden);
    return downsample_->forward(hidden);
  }

 private:
  torch::nn::ModuleList block_{nullptr};
  std::vector<MiniMaxH3AudioResidualUnit> units_;
  MiniMaxH3AudioSnake1d snake_{nullptr};
  MiniMaxH3AudioConv1d downsample_{nullptr};
};
TORCH_MODULE(MiniMaxH3AudioEncoderBlock);

class MiniMaxH3AudioEncoderImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3AudioEncoderImpl(const torch::TensorOptions& options) {
    block_ = register_module("block", torch::nn::ModuleList());
    conv_in_ = MiniMaxH3AudioConv1d(1, 64, 7, 1, 3, 1, true, options);
    block_->push_back(conv_in_);

    int64_t channels = MiniMaxH3AudioVAEConfig::kEncoderDim;
    for (int64_t stride : MiniMaxH3AudioVAEConfig::kEncoderRates) {
      MiniMaxH3AudioEncoderBlock stage(channels, channels * 2, stride, options);
      block_->push_back(stage);
      stages_.emplace_back(std::move(stage));
      channels *= 2;
    }
    snake_out_ = MiniMaxH3AudioSnake1d(channels, options);
    conv_out_ = MiniMaxH3AudioConv1d(channels,
                                     MiniMaxH3AudioVAEConfig::kLatentDim,
                                     3,
                                     1,
                                     1,
                                     1,
                                     true,
                                     options);
    block_->push_back(snake_out_);
    block_->push_back(conv_out_);
  }

  torch::Tensor forward(const torch::Tensor& input,
                        MiniMaxH3AudioVAETraceContext* trace) const {
    torch::Tensor hidden = conv_in_->forward(input);
    if (trace != nullptr) {
      trace->emit("encoder.block", 0, hidden);
    }
    for (size_t index = 0; index < stages_.size(); ++index) {
      hidden = stages_[index]->forward(hidden);
      if (trace != nullptr) {
        trace->emit("encoder.block", static_cast<int64_t>(index) + 1, hidden);
      }
    }
    hidden = snake_out_->forward(hidden);
    if (trace != nullptr) {
      trace->emit("encoder.block", 6, hidden);
    }
    hidden = conv_out_->forward(hidden);
    if (trace != nullptr) {
      trace->emit("encoder.block", 7, hidden);
    }
    return hidden;
  }

 private:
  torch::nn::ModuleList block_{nullptr};
  MiniMaxH3AudioConv1d conv_in_{nullptr};
  std::vector<MiniMaxH3AudioEncoderBlock> stages_;
  MiniMaxH3AudioSnake1d snake_out_{nullptr};
  MiniMaxH3AudioConv1d conv_out_{nullptr};
};
TORCH_MODULE(MiniMaxH3AudioEncoder);

class MiniMaxH3AudioGeGluMlpImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AudioGeGluMlpImpl(int64_t input_size,
                             int64_t hidden_size,
                             const torch::TensorOptions& options) {
    norm_ =
        register_module("norm", MiniMaxH3AudioLayerNorm(input_size, options));
    w0_ = register_module(
        "w0", MiniMaxH3AudioLinear(input_size, hidden_size, true, options));
    w1_ = register_module(
        "w1", MiniMaxH3AudioLinear(input_size, hidden_size, true, options));
    w2_ = register_module(
        "w2", MiniMaxH3AudioLinear(hidden_size, input_size, true, options));
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    const torch::Tensor hidden = norm_->forward(input);
    const torch::Tensor gate = torch::nn::functional::gelu(
        w0_->forward(hidden),
        torch::nn::functional::GELUFuncOptions().approximate("tanh"));
    return w2_->forward(gate * w1_->forward(hidden));
  }

 private:
  MiniMaxH3AudioLayerNorm norm_{nullptr};
  MiniMaxH3AudioLinear w0_{nullptr};
  MiniMaxH3AudioLinear w1_{nullptr};
  MiniMaxH3AudioLinear w2_{nullptr};
};
TORCH_MODULE(MiniMaxH3AudioGeGluMlp);

class MiniMaxH3AudioCausalAttentionImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3AudioCausalAttentionImpl(
      const torch::TensorOptions& options) {
    constexpr int64_t kInput = MiniMaxH3AudioVAEConfig::kLatentDim;
    qkv_ = register_module(
        "qkv", MiniMaxH3AudioLinear(kInput, 3 * kInput, false, options));
    const torch::TensorOptions fp32 = minimax_h3_audio_fp32_options(options);
    q_bias_ = register_parameter("q_bias", torch::empty({kInput}, fp32));
    v_bias_ = register_parameter("v_bias", torch::empty({kInput}, fp32));
    zero_k_bias_ = register_buffer("zero_k_bias", torch::empty({kInput}, fp32));
    proj_ = register_module(
        "proj",
        MiniMaxH3AudioLinear(MiniMaxH3AudioVAEConfig::kLatentChannels,
                             MiniMaxH3AudioVAEConfig::kLatentChannels,
                             true,
                             options));
  }

  torch::Tensor forward(const torch::Tensor& input,
                        MiniMaxH3AudioVAETraceContext* trace = nullptr) const {
    constexpr int64_t kHeads = MiniMaxH3AudioVAEConfig::kAttentionHeads;
    constexpr int64_t kHeadDim = MiniMaxH3AudioVAEConfig::kAttentionHeadDim;
    const int64_t batch = input.size(0);
    const int64_t sequence = input.size(1);
    torch::Tensor qkv = torch::nn::functional::linear(
        input, qkv_->weight(), torch::cat({q_bias_, zero_k_bias_, v_bias_}));
    torch::Tensor grouped = qkv.view({batch, sequence, 3, kHeads, kHeadDim});
    if (trace != nullptr) {
      trace->emit("pre_block.qkv", -1, qkv);
      trace->emit("pre_block.query", -1, grouped.select(2, 0));
      trace->emit("pre_block.key", -1, grouped.select(2, 1));
      trace->emit("pre_block.value", -1, grouped.select(2, 2));
    }
    std::vector<torch::Tensor> projections =
        grouped.permute({2, 0, 3, 1, 4}).unbind(0);
    torch::Tensor attended =
        torch::scaled_dot_product_attention(projections[0],
                                            projections[1],
                                            projections[2],
                                            torch::nullopt,
                                            /*dropout_p=*/0.0,
                                            /*is_causal=*/true);
    attended = attended.mean(1);
    if (kHeadDim != MiniMaxH3AudioVAEConfig::kLatentChannels) {
      attended = torch::nn::functional::adaptive_avg_pool1d(
          attended,
          torch::nn::functional::AdaptiveAvgPool1dFuncOptions(
              MiniMaxH3AudioVAEConfig::kLatentChannels));
    }
    if (attended.sizes() !=
        std::vector<int64_t>{
            batch, sequence, MiniMaxH3AudioVAEConfig::kLatentChannels}) {
      throw std::logic_error(
          "MiniMax-H3 audio attention projection produced an invalid shape");
    }
    return proj_->forward(attended);
  }

 private:
  MiniMaxH3AudioLinear qkv_{nullptr};
  torch::Tensor q_bias_;
  torch::Tensor v_bias_;
  torch::Tensor zero_k_bias_;
  MiniMaxH3AudioLinear proj_{nullptr};
};
TORCH_MODULE(MiniMaxH3AudioCausalAttention);

class MiniMaxH3AudioAttnProjectionImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3AudioAttnProjectionImpl(
      const torch::TensorOptions& options) {
    norm1_ = register_module(
        "norm1",
        MiniMaxH3AudioLayerNorm(MiniMaxH3AudioVAEConfig::kLatentDim, options));
    attn_ = register_module("attn", MiniMaxH3AudioCausalAttention(options));
    proj_ = register_module(
        "proj",
        MiniMaxH3AudioLinear(MiniMaxH3AudioVAEConfig::kLatentDim,
                             MiniMaxH3AudioVAEConfig::kLatentChannels,
                             true,
                             options));
    norm3_ = register_module(
        "norm3",
        MiniMaxH3AudioLayerNorm(MiniMaxH3AudioVAEConfig::kLatentDim, options));
    norm2_ =
        register_module("norm2",
                        MiniMaxH3AudioLayerNorm(
                            MiniMaxH3AudioVAEConfig::kLatentChannels, options));
    mlp_ = register_module(
        "mlp",
        MiniMaxH3AudioGeGluMlp(MiniMaxH3AudioVAEConfig::kLatentChannels,
                               2 * MiniMaxH3AudioVAEConfig::kLatentChannels,
                               options));
  }

  torch::Tensor forward(const torch::Tensor& input,
                        MiniMaxH3AudioVAETraceContext* trace = nullptr) const {
    torch::Tensor projection = proj_->forward(norm3_->forward(input));
    torch::Tensor attention = attn_->forward(norm1_->forward(input), trace);
    if (trace != nullptr) {
      trace->emit("pre_block.projection_branch", -1, projection);
      trace->emit("pre_block.attention_branch", -1, attention);
    }
    torch::Tensor hidden = projection + attention;
    torch::Tensor mlp = mlp_->forward(norm2_->forward(hidden));
    if (trace != nullptr) {
      trace->emit("pre_block.mlp_branch", -1, mlp);
      trace->emit("pre_block.final", -1, hidden + mlp);
    }
    return hidden + mlp;
  }

 private:
  MiniMaxH3AudioLayerNorm norm1_{nullptr};
  MiniMaxH3AudioCausalAttention attn_{nullptr};
  MiniMaxH3AudioLinear proj_{nullptr};
  MiniMaxH3AudioLayerNorm norm3_{nullptr};
  MiniMaxH3AudioLayerNorm norm2_{nullptr};
  MiniMaxH3AudioGeGluMlp mlp_{nullptr};
};
TORCH_MODULE(MiniMaxH3AudioAttnProjection);

class MiniMaxH3AudioAMPBlockImpl final : public torch::nn::Module {
 public:
  MiniMaxH3AudioAMPBlockImpl(int64_t channels,
                             int64_t kernel_size,
                             const torch::TensorOptions& options) {
    convs1_ = register_module("convs1", torch::nn::ModuleList());
    convs2_ = register_module("convs2", torch::nn::ModuleList());
    activations_ = register_module("activations", torch::nn::ModuleList());
    for (int64_t dilation : MiniMaxH3AudioVAEConfig::kResBlockDilations) {
      MiniMaxH3AudioConv1d conv1(channels,
                                 channels,
                                 kernel_size,
                                 1,
                                 (kernel_size * dilation - dilation) / 2,
                                 dilation,
                                 true,
                                 options);
      MiniMaxH3AudioConv1d conv2(channels,
                                 channels,
                                 kernel_size,
                                 1,
                                 (kernel_size - 1) / 2,
                                 1,
                                 true,
                                 options);
      convs1_->push_back(conv1);
      convs2_->push_back(conv2);
      conv1_layers_.emplace_back(std::move(conv1));
      conv2_layers_.emplace_back(std::move(conv2));
    }
    for (int64_t index = 0; index < 6; ++index) {
      MiniMaxH3AudioActivation1d activation(channels, options);
      activations_->push_back(activation);
      activation_layers_.emplace_back(std::move(activation));
    }
  }

  torch::Tensor forward(const torch::Tensor& input) const {
    torch::Tensor hidden = input;
    for (size_t index = 0; index < conv1_layers_.size(); ++index) {
      torch::Tensor residual = activation_layers_[2 * index]->forward(hidden);
      residual = conv1_layers_[index]->forward(residual);
      residual = activation_layers_[2 * index + 1]->forward(residual);
      residual = conv2_layers_[index]->forward(residual);
      hidden = hidden + residual;
    }
    return hidden;
  }

 private:
  torch::nn::ModuleList convs1_{nullptr};
  torch::nn::ModuleList convs2_{nullptr};
  torch::nn::ModuleList activations_{nullptr};
  std::vector<MiniMaxH3AudioConv1d> conv1_layers_;
  std::vector<MiniMaxH3AudioConv1d> conv2_layers_;
  std::vector<MiniMaxH3AudioActivation1d> activation_layers_;
};
TORCH_MODULE(MiniMaxH3AudioAMPBlock);

class MiniMaxH3AudioBigVGANDecoderImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3AudioBigVGANDecoderImpl(
      const torch::TensorOptions& options) {
    conv_pre_ = register_module(
        "conv_pre",
        MiniMaxH3AudioConv1d(MiniMaxH3AudioVAEConfig::kLatentDim,
                             MiniMaxH3AudioVAEConfig::kDecoderDim,
                             7,
                             1,
                             3,
                             1,
                             true,
                             options));
    ups_ = register_module("ups", torch::nn::ModuleList());
    resblocks_ = register_module("resblocks", torch::nn::ModuleList());
    int64_t input_channels = MiniMaxH3AudioVAEConfig::kDecoderDim;
    for (size_t stage = 0;
         stage < MiniMaxH3AudioVAEConfig::kDecoderRates.size();
         ++stage) {
      const int64_t output_channels = input_channels / 2;
      MiniMaxH3AudioConvTranspose1d upsample(
          input_channels,
          output_channels,
          MiniMaxH3AudioVAEConfig::kDecoderKernels[stage],
          MiniMaxH3AudioVAEConfig::kDecoderRates[stage],
          (MiniMaxH3AudioVAEConfig::kDecoderKernels[stage] -
           MiniMaxH3AudioVAEConfig::kDecoderRates[stage]) /
              2,
          true,
          options);
      torch::nn::ModuleList nested;
      nested->push_back(upsample);
      ups_->push_back(nested);
      upsample_layers_.emplace_back(std::move(upsample));

      for (int64_t kernel : MiniMaxH3AudioVAEConfig::kResBlockKernels) {
        MiniMaxH3AudioAMPBlock block(output_channels, kernel, options);
        resblocks_->push_back(block);
        resblock_layers_.emplace_back(std::move(block));
      }
      input_channels = output_channels;
    }
    activation_post_ = register_module(
        "activation_post", MiniMaxH3AudioActivation1d(input_channels, options));
    conv_post_ = register_module(
        "conv_post",
        MiniMaxH3AudioConv1d(input_channels, 1, 7, 1, 3, 1, false, options));
  }

  torch::Tensor forward(const torch::Tensor& input,
                        MiniMaxH3AudioVAETraceContext* trace) const {
    torch::Tensor hidden = conv_pre_->forward(input);
    if (trace != nullptr) {
      trace->emit("decoder.conv_pre", -1, hidden);
    }
    for (size_t stage = 0; stage < upsample_layers_.size(); ++stage) {
      hidden = upsample_layers_[stage]->forward(hidden);
      if (trace != nullptr) {
        trace->emit("decoder.ups", static_cast<int64_t>(stage), hidden);
      }
      torch::Tensor sum;
      for (size_t kernel = 0;
           kernel < MiniMaxH3AudioVAEConfig::kResBlockKernels.size();
           ++kernel) {
        torch::Tensor block =
            resblock_layers_[stage * 3 + kernel]->forward(hidden);
        sum = sum.defined() ? sum + block : block;
      }
      hidden = sum / 3.0;
      if (trace != nullptr) {
        trace->emit("decoder.stage", static_cast<int64_t>(stage), hidden);
      }
    }
    hidden = activation_post_->forward(hidden);
    if (trace != nullptr) {
      trace->emit("decoder.activation_post", -1, hidden);
    }
    torch::Tensor pre_clamp = conv_post_->forward(hidden);
    if (trace != nullptr) {
      trace->emit("decoder.final_pre_clamp", -1, pre_clamp);
    }
    torch::Tensor clamped = pre_clamp.clamp(-1.0, 1.0);
    if (trace != nullptr) {
      trace->emit("decoder.final_clamped", -1, clamped);
    }
    return clamped;
  }

 private:
  MiniMaxH3AudioConv1d conv_pre_{nullptr};
  torch::nn::ModuleList ups_{nullptr};
  std::vector<MiniMaxH3AudioConvTranspose1d> upsample_layers_;
  torch::nn::ModuleList resblocks_{nullptr};
  std::vector<MiniMaxH3AudioAMPBlock> resblock_layers_;
  MiniMaxH3AudioActivation1d activation_post_{nullptr};
  MiniMaxH3AudioConv1d conv_post_{nullptr};
};
TORCH_MODULE(MiniMaxH3AudioBigVGANDecoder);

class MiniMaxH3AudioDiagonalGaussianDistribution final {
 public:
  MiniMaxH3AudioDiagonalGaussianDistribution(const torch::Tensor& mean,
                                             const torch::Tensor& logs)
      : mean_(mean), logs_(logs), std_(torch::exp(logs)) {
    if (!mean_.defined() || !logs_.defined() ||
        mean_.sizes() != logs_.sizes() || mean_.dim() != 3 ||
        mean_.size(1) != MiniMaxH3AudioVAEConfig::kLatentChannels ||
        mean_.scalar_type() != torch::kFloat32 ||
        logs_.scalar_type() != torch::kFloat32 ||
        mean_.device() != logs_.device()) {
      throw std::invalid_argument(
          "MiniMax-H3 audio posterior mean/logs must be matching FP32 "
          "[B,32,T] tensors");
    }
    minimax_h3_audio_require_finite(mean_, "posterior.mean");
    minimax_h3_audio_require_finite(logs_, "posterior.logs");
    minimax_h3_audio_require_finite(std_, "posterior.std");
  }

  torch::Tensor sample(const torch::Tensor& epsilon) const {
    if (!epsilon.defined() || epsilon.sizes() != mean_.sizes() ||
        epsilon.scalar_type() != torch::kFloat32 ||
        epsilon.device() != mean_.device()) {
      throw std::invalid_argument(
          "MiniMax-H3 audio posterior epsilon must be matching FP32 noise on "
          "the posterior device");
    }
    minimax_h3_audio_require_finite(epsilon, "posterior.epsilon");
    torch::Tensor value = mean_ + std_ * epsilon;
    minimax_h3_audio_require_finite(value, "posterior.sample");
    return value.contiguous();
  }

  torch::Tensor mode() const { return mean_; }
  const torch::Tensor& mean() const { return mean_; }
  const torch::Tensor& logs() const { return logs_; }
  const torch::Tensor& std() const { return std_; }

 private:
  torch::Tensor mean_;
  torch::Tensor logs_;
  torch::Tensor std_;
};

class MiniMaxH3AudioVAESourceLayoutValidator final {
 public:
  static constexpr size_t kExpectedTensorCount = 1087;
  static constexpr int64_t kExpectedElementCount = 151326585;
  static constexpr size_t kExpectedWeightNormCount = 172;
  static constexpr size_t kExpectedKaiserBufferCount = 254;

  static std::vector<MiniMaxH3AudioVAESourceTensorSpec>
  expected_source_tensors() {
    std::vector<MiniMaxH3AudioVAESourceTensorSpec> specs;
    specs.reserve(kExpectedTensorCount);
    auto add = [&specs](std::string name, std::vector<int64_t> shape) {
      specs.push_back({std::move(name), std::move(shape)});
    };
    auto add_weight_norm = [&add](const std::string& prefix,
                                  std::vector<int64_t> weight_shape,
                                  int64_t norm_channels,
                                  std::optional<int64_t> bias_channels) {
      if (bias_channels.has_value()) {
        add(prefix + ".bias", {*bias_channels});
      }
      add(prefix + ".weight_g", {norm_channels, 1, 1});
      add(prefix + ".weight_v", std::move(weight_shape));
    };
    auto add_linear = [&add](const std::string& prefix,
                             int64_t input_size,
                             int64_t output_size,
                             bool bias = true) {
      if (bias) {
        add(prefix + ".bias", {output_size});
      }
      add(prefix + ".weight", {output_size, input_size});
    };
    auto add_layer_norm = [&add](const std::string& prefix, int64_t size) {
      add(prefix + ".bias", {size});
      add(prefix + ".weight", {size});
    };
    auto add_alias_activation = [&add](const std::string& prefix,
                                       int64_t channels) {
      add(prefix + ".act.alpha", {channels});
      add(prefix + ".act.beta", {channels});
      add(prefix + ".downsample.lowpass.filter", {1, 1, 12});
      add(prefix + ".upsample.filter", {1, 1, 12});
    };

    add_weight_norm("encoder.block.0", {64, 1, 7}, 64, 64);
    int64_t channels = MiniMaxH3AudioVAEConfig::kEncoderDim;
    for (size_t stage = 0;
         stage < MiniMaxH3AudioVAEConfig::kEncoderRates.size();
         ++stage) {
      const std::string stage_prefix =
          "encoder.block." + std::to_string(stage + 1) + ".block";
      for (size_t unit = 0;
           unit < MiniMaxH3AudioVAEConfig::kEncoderResidualDilations.size();
           ++unit) {
        const std::string prefix =
            stage_prefix + "." + std::to_string(unit) + ".block";
        add(prefix + ".0.alpha", {1, channels, 1});
        add_weight_norm(
            prefix + ".1", {channels, channels, 7}, channels, channels);
        add(prefix + ".2.alpha", {1, channels, 1});
        add_weight_norm(
            prefix + ".3", {channels, channels, 1}, channels, channels);
      }
      add(stage_prefix + ".3.alpha", {1, channels, 1});
      const int64_t output_channels = channels * 2;
      const int64_t kernel = 2 * MiniMaxH3AudioVAEConfig::kEncoderRates[stage];
      add_weight_norm(stage_prefix + ".4",
                      {output_channels, channels, kernel},
                      output_channels,
                      output_channels);
      channels = output_channels;
    }
    add("encoder.block.6.alpha", {1, channels, 1});
    add_weight_norm("encoder.block.7",
                    {MiniMaxH3AudioVAEConfig::kLatentDim, channels, 3},
                    MiniMaxH3AudioVAEConfig::kLatentDim,
                    MiniMaxH3AudioVAEConfig::kLatentDim);

    add_layer_norm("pre_block.norm1", 2048);
    add_linear("pre_block.attn.qkv", 2048, 6144, false);
    add("pre_block.attn.q_bias", {2048});
    add("pre_block.attn.v_bias", {2048});
    add("pre_block.attn.zero_k_bias", {2048});
    add_linear("pre_block.attn.proj", 32, 32);
    add_linear("pre_block.proj", 2048, 32);
    add_layer_norm("pre_block.norm3", 2048);
    add_layer_norm("pre_block.norm2", 32);
    add_layer_norm("pre_block.mlp.norm", 32);
    add_linear("pre_block.mlp.w0", 32, 64);
    add_linear("pre_block.mlp.w1", 32, 64);
    add_linear("pre_block.mlp.w2", 64, 32);

    add("mean_proj.bias", {32});
    add("mean_proj.weight", {32, 32, 1});
    add("logs_proj.bias", {32});
    add("logs_proj.weight", {32, 32, 1});
    add("dec_in_proj.bias", {2048});
    add("dec_in_proj.weight", {2048, 32, 1});

    add_weight_norm("decoder.conv_pre", {1024, 2048, 7}, 1024, 1024);
    channels = MiniMaxH3AudioVAEConfig::kDecoderDim;
    for (size_t stage = 0;
         stage < MiniMaxH3AudioVAEConfig::kDecoderRates.size();
         ++stage) {
      const int64_t output_channels = channels / 2;
      add_weight_norm("decoder.ups." + std::to_string(stage) + ".0",
                      {channels,
                       output_channels,
                       MiniMaxH3AudioVAEConfig::kDecoderKernels[stage]},
                      channels,
                      output_channels);
      for (size_t kernel_index = 0;
           kernel_index < MiniMaxH3AudioVAEConfig::kResBlockKernels.size();
           ++kernel_index) {
        const size_t block_index =
            stage * MiniMaxH3AudioVAEConfig::kResBlockKernels.size() +
            kernel_index;
        const std::string prefix =
            "decoder.resblocks." + std::to_string(block_index);
        const int64_t kernel =
            MiniMaxH3AudioVAEConfig::kResBlockKernels[kernel_index];
        for (size_t activation = 0; activation < 6; ++activation) {
          add_alias_activation(
              prefix + ".activations." + std::to_string(activation),
              output_channels);
        }
        for (size_t convolution = 0; convolution < 3; ++convolution) {
          add_weight_norm(prefix + ".convs1." + std::to_string(convolution),
                          {output_channels, output_channels, kernel},
                          output_channels,
                          output_channels);
          add_weight_norm(prefix + ".convs2." + std::to_string(convolution),
                          {output_channels, output_channels, kernel},
                          output_channels,
                          output_channels);
        }
      }
      channels = output_channels;
    }
    add_alias_activation("decoder.activation_post", channels);
    add_weight_norm("decoder.conv_post", {1, channels, 7}, 1, std::nullopt);

    size_t weight_norm_count = 0;
    size_t kaiser_count = 0;
    int64_t element_count = 0;
    for (const MiniMaxH3AudioVAESourceTensorSpec& spec : specs) {
      weight_norm_count += spec.name.ends_with(".weight_g") ? 1 : 0;
      kaiser_count += spec.name.ends_with(".filter") ? 1 : 0;
      int64_t tensor_elements = 1;
      for (int64_t dimension : spec.shape) {
        if (dimension <= 0 ||
            tensor_elements > kExpectedElementCount / dimension) {
          throw std::logic_error(
              "MiniMax-H3 audio VAE source inventory has an invalid shape");
        }
        tensor_elements *= dimension;
      }
      element_count += tensor_elements;
    }
    if (specs.size() != kExpectedTensorCount ||
        element_count != kExpectedElementCount ||
        weight_norm_count != kExpectedWeightNormCount ||
        kaiser_count != kExpectedKaiserBufferCount) {
      throw std::logic_error(
          "MiniMax-H3 audio VAE source inventory declaration is invalid");
    }
    return specs;
  }

  static std::unordered_map<std::string, const torch::Tensor*> validate(
      const std::vector<std::unique_ptr<StateDict>>& shards) {
    if (shards.empty()) {
      throw std::invalid_argument(
          "MiniMax-H3 audio VAE source inventory has no shards");
    }
    const std::vector<MiniMaxH3AudioVAESourceTensorSpec> specs =
        expected_source_tensors();
    std::unordered_map<std::string, const MiniMaxH3AudioVAESourceTensorSpec*>
        expected;
    expected.reserve(specs.size());
    for (const MiniMaxH3AudioVAESourceTensorSpec& spec : specs) {
      expected.emplace(spec.name, &spec);
    }

    std::unordered_map<std::string, const torch::Tensor*> source;
    source.reserve(kExpectedTensorCount);
    for (const std::unique_ptr<StateDict>& shard : shards) {
      if (shard == nullptr) {
        throw std::invalid_argument(
            "MiniMax-H3 audio VAE source inventory contains a null shard");
      }
      for (const auto& [name, tensor] : *shard) {
        if (!expected.contains(name)) {
          throw std::invalid_argument(
              "MiniMax-H3 audio VAE source inventory has unknown tensor `" +
              name + "`");
        }
        if (!source.emplace(name, &tensor).second) {
          throw std::invalid_argument(
              "MiniMax-H3 audio VAE source tensor occurs more than once: `" +
              name + "`");
        }
      }
    }
    if (source.size() != kExpectedTensorCount) {
      throw std::invalid_argument(
          "MiniMax-H3 audio VAE source inventory must contain exactly 1087 "
          "tensors");
    }
    for (const MiniMaxH3AudioVAESourceTensorSpec& spec : specs) {
      const auto iterator = source.find(spec.name);
      if (iterator == source.end()) {
        throw std::invalid_argument(
            "MiniMax-H3 audio VAE source inventory is missing tensor `" +
            spec.name + "`");
      }
      const torch::Tensor& tensor = *iterator->second;
      if (!tensor.defined() || tensor.scalar_type() != torch::kFloat32 ||
          tensor.sizes().vec() != spec.shape) {
        throw std::invalid_argument(
            "MiniMax-H3 audio VAE source tensor metadata mismatch for `" +
            spec.name + "`: expected FP32 " +
            minimax_h3_audio_shape_string(spec.shape));
      }
      minimax_h3_audio_require_finite(tensor, spec.name);
    }
    validate_persistent_buffers(source, specs);
    return source;
  }

 private:
  static const std::vector<float>& canonical_kaiser_filter() {
    static const std::vector<float> values = {0.0020289646927267313F,
                                              0.009389465674757957F,
                                              -0.0255434587597847F,
                                              -0.057657383382320404F,
                                              0.12857258319854736F,
                                              0.44320979714393616F,
                                              0.44320979714393616F,
                                              0.12857258319854736F,
                                              -0.057657383382320404F,
                                              -0.0255434587597847F,
                                              0.009389465674757957F,
                                              0.0020289646927267313F};
    return values;
  }

  static void validate_persistent_buffers(
      const std::unordered_map<std::string, const torch::Tensor*>& source,
      const std::vector<MiniMaxH3AudioVAESourceTensorSpec>& specs) {
    size_t filters = 0;
    for (const MiniMaxH3AudioVAESourceTensorSpec& spec : specs) {
      if (!spec.name.ends_with(".filter")) {
        continue;
      }
      ++filters;
      const torch::Tensor& value = *source.at(spec.name);
      const torch::Tensor expected =
          torch::tensor(canonical_kaiser_filter(),
                        torch::TensorOptions().dtype(torch::kFloat32))
              .view({1, 1, MiniMaxH3AudioVAEConfig::kAliasFreeKernel});
      if (!torch::equal(value.to(torch::kCPU), expected)) {
        throw std::invalid_argument(
            "MiniMax-H3 audio VAE Kaiser buffer mismatch for `" + spec.name +
            "`");
      }
    }
    if (filters != kExpectedKaiserBufferCount) {
      throw std::logic_error(
          "MiniMax-H3 audio VAE did not validate all 254 Kaiser buffers");
    }
    const torch::Tensor& zero_k_bias = *source.at("pre_block.attn.zero_k_bias");
    if (zero_k_bias.count_nonzero().item<int64_t>() != 0) {
      throw std::invalid_argument(
          "MiniMax-H3 audio VAE pre_block.attn.zero_k_bias must be all zero");
    }
  }
};

class MiniMaxH3AudioVAEImpl final : public torch::nn::Module {
 public:
  explicit MiniMaxH3AudioVAEImpl(const ModelContext& context)
      : MiniMaxH3AudioVAEImpl(context.get_tensor_options()) {}

  explicit MiniMaxH3AudioVAEImpl(const torch::TensorOptions& options)
      : options_(minimax_h3_audio_fp32_options(options)) {
    encoder_ = register_module("encoder", MiniMaxH3AudioEncoder(options_));
    pre_block_ =
        register_module("pre_block", MiniMaxH3AudioAttnProjection(options_));
    mean_proj_ = register_module(
        "mean_proj", MiniMaxH3AudioConv1d(32, 32, 1, 1, 0, 1, true, options_));
    logs_proj_ = register_module(
        "logs_proj", MiniMaxH3AudioConv1d(32, 32, 1, 1, 0, 1, true, options_));
    dec_in_proj_ = register_module(
        "dec_in_proj",
        MiniMaxH3AudioConv1d(32, 2048, 1, 1, 0, 1, true, options_));
    decoder_ =
        register_module("decoder", MiniMaxH3AudioBigVGANDecoder(options_));
    for (auto& parameter : named_parameters(/*recurse=*/true)) {
      parameter.value().set_requires_grad(false);
    }
  }

  MiniMaxH3AudioDiagonalGaussianDistribution encode(
      const torch::Tensor& waveform,
      const MiniMaxH3AudioVAETraceHook& hook = nullptr) const {
    MiniMaxH3AudioVAETraceContext trace(hook);
    return encode_impl(waveform, &trace);
  }

  torch::Tensor decode(const torch::Tensor& denormalized_latents,
                       const MiniMaxH3AudioVAETraceHook& hook = nullptr) const {
    MiniMaxH3AudioVAETraceContext trace(hook);
    return decode_impl(denormalized_latents, &trace);
  }

  torch::Tensor encode_condition(
      const torch::Tensor& stereo_waveform,
      const MiniMaxH3AudioVAETraceHook& hook = nullptr) const {
    if (!stereo_waveform.defined() || stereo_waveform.dim() != 3 ||
        stereo_waveform.size(0) != MiniMaxH3AudioVAEConfig::kStereoChannels ||
        stereo_waveform.size(1) != MiniMaxH3AudioVAEConfig::kInputChannels) {
      throw std::invalid_argument(
          "MiniMax-H3 audio reference must be stereo-as-batch [2,1,samples]");
    }
    MiniMaxH3AudioVAETraceContext trace(hook);
    MiniMaxH3AudioDiagonalGaussianDistribution posterior =
        encode_impl(stereo_waveform, &trace);
    torch::Tensor normalized = normalize_latents(posterior.mode());
    trace.emit("posterior.normalized_mode", -1, normalized);
    return normalized;
  }

  torch::Tensor decode_normalized(
      const torch::Tensor& normalized_stereo_latents,
      const MiniMaxH3AudioVAETraceHook& hook = nullptr) const {
    validate_latents(normalized_stereo_latents, "normalized decode input");
    if (normalized_stereo_latents.size(0) !=
        MiniMaxH3AudioVAEConfig::kStereoChannels) {
      throw std::invalid_argument(
          "MiniMax-H3 normalized audio decode expects stereo-as-batch "
          "[2,32,T]");
    }
    MiniMaxH3AudioVAETraceContext trace(hook);
    torch::Tensor denormalized = denormalize_latents(normalized_stereo_latents);
    trace.emit("decoder.denormalized_latents", -1, denormalized);
    torch::Tensor mono_batch = decode_impl(denormalized, &trace);
    if (mono_batch.size(0) != 2 || mono_batch.size(1) != 1 ||
        mono_batch.size(2) != normalized_stereo_latents.size(2) *
                                  MiniMaxH3AudioVAEConfig::kHopLength) {
      throw std::logic_error(
          "MiniMax-H3 audio decoder violated the stereo waveform contract");
    }
    torch::Tensor stereo = mono_batch.permute({1, 0, 2}).contiguous();
    trace.emit("decoder.stereo", -1, stereo);
    return stereo;
  }

  torch::Tensor forward(
      const torch::Tensor& waveform,
      bool sample_posterior = false,
      const std::optional<torch::Tensor>& epsilon = std::nullopt,
      const MiniMaxH3AudioVAETraceHook& hook = nullptr) const {
    MiniMaxH3AudioDiagonalGaussianDistribution posterior =
        encode(waveform, hook);
    if (sample_posterior && !epsilon.has_value()) {
      throw std::invalid_argument(
          "MiniMax-H3 audio posterior sampling requires explicit epsilon");
    }
    torch::Tensor latents =
        sample_posterior ? posterior.sample(*epsilon) : posterior.mode();
    return decode(latents, hook);
  }

  static torch::Tensor normalize_latents(const torch::Tensor& latents) {
    validate_latents(latents, "latent normalize input");
    torch::Tensor value = latents.to(torch::kFloat32);
    value = (value - channel_values(value, latent_mean())) /
            channel_values(value, latent_std());
    minimax_h3_audio_require_finite(value, "latent normalize output");
    return value.contiguous();
  }

  static torch::Tensor denormalize_latents(const torch::Tensor& latents) {
    validate_latents(latents, "latent denormalize input");
    torch::Tensor value = latents.to(torch::kFloat32);
    value = value * channel_values(value, latent_std()) +
            channel_values(value, latent_mean());
    minimax_h3_audio_require_finite(value, "latent denormalize output");
    return value.contiguous();
  }

  void load_source_weights(
      const std::vector<std::unique_ptr<StateDict>>& shards) {
    if (loaded_) {
      throw std::logic_error(
          "MiniMax-H3 audio VAE source weights are already loaded");
    }
    const auto source =
        MiniMaxH3AudioVAESourceLayoutValidator::validate(shards);
    const auto specs =
        MiniMaxH3AudioVAESourceLayoutValidator::expected_source_tensors();

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
    constexpr size_t kExpectedNativeTensorCount =
        MiniMaxH3AudioVAESourceLayoutValidator::kExpectedTensorCount -
        MiniMaxH3AudioVAESourceLayoutValidator::kExpectedWeightNormCount;
    if (targets.size() != kExpectedNativeTensorCount) {
      throw std::logic_error(
          "MiniMax-H3 audio VAE native/source tensor counts differ");
    }

    std::unordered_set<std::string> loaded_targets;
    loaded_targets.reserve(targets.size());
    torch::NoGradGuard no_grad;
    for (const MiniMaxH3AudioVAESourceTensorSpec& spec : specs) {
      loaded_source_names_.emplace(spec.name);
      if (spec.name.ends_with(".weight_g")) {
        continue;
      }

      std::string target_name = spec.name;
      torch::Tensor value = *source.at(spec.name);
      if (spec.name.ends_with(".weight_v")) {
        const std::string prefix = spec.name.substr(
            0, spec.name.size() - std::string(".weight_v").size());
        target_name = prefix + ".weight";
      }
      const auto target_iterator = targets.find(target_name);
      if (target_iterator == targets.end()) {
        throw std::logic_error(
            "MiniMax-H3 audio VAE module is missing registered tensor `" +
            target_name + "`");
      }
      torch::Tensor& target = *target_iterator->second;
      if (spec.name.ends_with(".weight_v")) {
        const std::string prefix = spec.name.substr(
            0, spec.name.size() - std::string(".weight_v").size());
        value = fold_weight_norm_dim0(
            source.at(prefix + ".weight_g")->to(target.device()),
            value.to(target.device()));
      }
      if (target.scalar_type() != torch::kFloat32 ||
          target.sizes() != value.sizes()) {
        throw std::logic_error(
            "MiniMax-H3 audio VAE registered tensor metadata mismatch for `" +
            target_name + "`");
      }
      target.copy_(value.to(target.device(), torch::kFloat32));
      if (!loaded_targets.emplace(target_name).second) {
        throw std::logic_error(
            "MiniMax-H3 audio VAE native tensor loaded more than once: `" +
            target_name + "`");
      }
    }
    if (loaded_targets.size() != targets.size()) {
      throw std::logic_error(
          "MiniMax-H3 audio VAE did not populate every native tensor");
    }
    loaded_ = true;
    verify_loaded_weights();
  }

  void load_model(DiTFolderLoader& loader) {
    const std::string weights_file =
        validate_config_files(loader.model_weights_path());
    auto& shards = loader.get_state_dicts();
    if (!shards.empty()) {
      load_source_weights(shards);
      return;
    }
    std::vector<std::unique_ptr<StateDict>> source_shard;
    source_shard.emplace_back(StateDictFromSafeTensor::load(
        loader.model_weights_path() + "/" + weights_file));
    load_source_weights(source_shard);
  }

  void verify_loaded_weights() const {
    const auto specs =
        MiniMaxH3AudioVAESourceLayoutValidator::expected_source_tensors();
    if (!loaded_ || loaded_source_names_.size() != specs.size()) {
      throw std::logic_error(
          "MiniMax-H3 audio VAE does not have all 1087 source tensors loaded");
    }
    for (const MiniMaxH3AudioVAESourceTensorSpec& spec : specs) {
      if (!loaded_source_names_.contains(spec.name)) {
        throw std::logic_error(
            "MiniMax-H3 audio VAE did not load source tensor `" + spec.name +
            "`");
      }
    }

    std::optional<torch::Device> resident_device;
    for (const auto& parameter : named_parameters(/*recurse=*/true)) {
      verify_resident_tensor(
          parameter.key(), parameter.value(), resident_device);
      if (parameter.key().ends_with(".weight_g") ||
          parameter.key().ends_with(".weight_v")) {
        throw std::logic_error(
            "MiniMax-H3 audio VAE must not retain weight-normalization "
            "parameters");
      }
    }
    size_t persistent_buffers = 0;
    for (const auto& buffer : named_buffers(/*recurse=*/true)) {
      verify_resident_tensor(buffer.key(), buffer.value(), resident_device);
      persistent_buffers += buffer.key().ends_with(".filter") ||
                                    buffer.key() == "pre_block.attn.zero_k_bias"
                                ? 1
                                : 0;
    }
    if (persistent_buffers != 255) {
      throw std::logic_error(
          "MiniMax-H3 audio VAE must retain 254 Kaiser filters and "
          "zero_k_bias");
    }
  }

  bool is_loaded() const { return loaded_; }

  static torch::Tensor fold_weight_norm_dim0(const torch::Tensor& weight_g,
                                             const torch::Tensor& weight_v) {
    if (!weight_g.defined() || !weight_v.defined() || weight_g.dim() != 3 ||
        weight_v.dim() != 3 || weight_g.scalar_type() != torch::kFloat32 ||
        weight_v.scalar_type() != torch::kFloat32 ||
        weight_g.device() != weight_v.device() || weight_g.size(1) != 1 ||
        weight_g.size(2) != 1 || weight_g.size(0) != weight_v.size(0)) {
      throw std::invalid_argument(
          "MiniMax-H3 audio weight_g/weight_v are invalid for dimension-0 "
          "weight normalization");
    }
    minimax_h3_audio_require_finite(weight_g, "weight_g");
    minimax_h3_audio_require_finite(weight_v, "weight_v");
    const torch::Tensor norm = weight_v.square().sum({1, 2}, true).sqrt();
    if (!torch::isfinite(norm).all().item<bool>() ||
        torch::le(norm, 0).any().item<bool>()) {
      throw std::invalid_argument(
          "MiniMax-H3 audio weight_v has an invalid dimension-0 norm");
    }
    torch::Tensor folded = at::_weight_norm(weight_v, weight_g, /*dim=*/0);
    minimax_h3_audio_require_finite(folded, "folded weight");
    return folded.contiguous();
  }

 private:
  MiniMaxH3AudioDiagonalGaussianDistribution encode_impl(
      const torch::Tensor& waveform,
      MiniMaxH3AudioVAETraceContext* trace) const {
    verify_loaded_weights();
    validate_waveform(waveform, "encode input");
    if (waveform.device() != dec_in_proj_->weight().device()) {
      throw std::invalid_argument(
          "MiniMax-H3 audio encode input is on the wrong device");
    }
    torch::NoGradGuard no_grad;
    MiniMaxH3AudioAutocastGuard autocast_guard(waveform.device());
    torch::Tensor input = waveform.to(torch::kFloat32);
    const int64_t right_pad =
        (MiniMaxH3AudioVAEConfig::kHopLength -
         input.size(-1) % MiniMaxH3AudioVAEConfig::kHopLength) %
        MiniMaxH3AudioVAEConfig::kHopLength;
    if (right_pad > 0) {
      input = torch::nn::functional::pad(
          input,
          torch::nn::functional::PadFuncOptions({0, right_pad})
              .mode(torch::kConstant)
              .value(0.0));
    }
    if (trace != nullptr) {
      trace->emit("encoder.padded_input", -1, input);
    }
    torch::Tensor hidden = encoder_->forward(input, trace);
    hidden = pre_block_->forward(hidden.transpose(1, 2), trace).transpose(1, 2);
    if (trace != nullptr) {
      trace->emit("encoder.pre_block", -1, hidden);
    }
    torch::Tensor mean = mean_proj_->forward(hidden).contiguous();
    torch::Tensor logs = logs_proj_->forward(hidden).contiguous();
    if (trace != nullptr) {
      trace->emit("posterior.mean", -1, mean);
      trace->emit("posterior.logs", -1, logs);
    }
    return MiniMaxH3AudioDiagonalGaussianDistribution(mean, logs);
  }

  torch::Tensor decode_impl(const torch::Tensor& denormalized_latents,
                            MiniMaxH3AudioVAETraceContext* trace) const {
    verify_loaded_weights();
    validate_latents(denormalized_latents, "decode input");
    if (denormalized_latents.device() != dec_in_proj_->weight().device()) {
      throw std::invalid_argument(
          "MiniMax-H3 audio decode input is on the wrong device");
    }
    torch::NoGradGuard no_grad;
    MiniMaxH3AudioAutocastGuard autocast_guard(denormalized_latents.device());
    torch::Tensor hidden =
        dec_in_proj_->forward(denormalized_latents.to(torch::kFloat32));
    if (trace != nullptr) {
      trace->emit("decoder.dec_in_proj", -1, hidden);
    }
    torch::Tensor decoded = decoder_->forward(hidden, trace);
    if (decoded.scalar_type() != torch::kFloat32 ||
        decoded.size(-1) != denormalized_latents.size(-1) *
                                MiniMaxH3AudioVAEConfig::kHopLength) {
      throw std::logic_error(
          "MiniMax-H3 audio decoder produced an invalid dtype or length");
    }
    minimax_h3_audio_require_finite(decoded, "decoder.raw");
    if (trace != nullptr) {
      trace->emit("decoder.raw", -1, decoded);
    }
    return decoded.contiguous();
  }

  static void validate_waveform(const torch::Tensor& value,
                                std::string_view name) {
    if (!value.defined() || value.dim() != 3 || value.size(0) <= 0 ||
        value.size(1) != MiniMaxH3AudioVAEConfig::kInputChannels ||
        value.size(2) <= 0 || !value.is_floating_point()) {
      throw std::invalid_argument("MiniMax-H3 audio " + std::string(name) +
                                  " must be floating-point [B,1,samples]");
    }
    minimax_h3_audio_require_finite(value, name);
  }

  static void validate_latents(const torch::Tensor& value,
                               std::string_view name) {
    if (!value.defined() || value.dim() != 3 || value.size(0) <= 0 ||
        value.size(1) != MiniMaxH3AudioVAEConfig::kLatentChannels ||
        value.size(2) <= 0 || !value.is_floating_point()) {
      throw std::invalid_argument("MiniMax-H3 audio " + std::string(name) +
                                  " must be floating-point [B,32,T]");
    }
    minimax_h3_audio_require_finite(value, name);
  }

  static torch::Tensor channel_values(const torch::Tensor& reference,
                                      const std::vector<float>& values) {
    return torch::tensor(values,
                         torch::TensorOptions()
                             .dtype(torch::kFloat32)
                             .device(reference.device()))
        .view({1, MiniMaxH3AudioVAEConfig::kLatentChannels, 1});
  }

  static const std::vector<float>& latent_mean() {
    static const std::vector<float> values = {
        -0.020211687488382354F, 0.3876466479950502F,    -0.04398279799186767F,
        -0.28591514936373F,     0.08179686214561671F,   -0.35782641352446604F,
        0.040623809960919084F,  -0.01552534501956604F,  -0.223362481667332F,
        0.1821006842509091F,    0.2941778783780663F,    -0.07901167601970885F,
        -0.056815072777201F,    -0.3699028221860095F,   -0.31616315591624855F,
        0.5905951377425391F,    -0.052139568068853864F, 0.013673160263486295F,
        -0.03691647864630577F,  0.09732660653298163F,   -0.3394662328788498F,
        -0.30685677538541667F,  -0.24504598907458763F,  -0.034698524462007344F,
        0.02868032184767538F,   -0.21217779266454084F,  -0.1678263169941987F,
        0.3221287889040614F,    -0.1223055851554907F,   0.4356604928128464F,
        -0.0502599202236253F,   0.3979258376211797F};
    return values;
  }

  static const std::vector<float>& latent_std() {
    static const std::vector<float> values = {
        1.6895524230479284F, 2.76263727217653F,   1.7945344281264435F,
        1.6801681847309828F, 1.6390226546605453F, 2.7788298348882177F,
        1.7659090095747236F, 1.6199757612137327F, 2.6336525640336896F,
        1.8539356672817833F, 2.5056497896915633F, 1.811019237886178F,
        1.9579657790720237F, 1.6685498243529284F, 1.4922469314453364F,
        3.298670198067373F,  1.9491804496832168F, 1.8720003270431442F,
        1.8334080103291832F, 1.6488070416529093F, 1.6176957696319716F,
        1.9131449234774398F, 1.5695245398428617F, 1.6943659940415912F,
        1.8318420762504692F, 1.5540637421583379F, 1.9344930328968526F,
        1.599198216109855F,  1.718045989838149F,  1.6307219190837705F,
        1.8661226051202384F, 1.5613768203168363F};
    return values;
  }

  static const std::vector<double>& latent_mean_config() {
    static const std::vector<double> values = {
        -0.020211687488382354, 0.3876466479950502,    -0.04398279799186767,
        -0.28591514936373,     0.08179686214561671,   -0.35782641352446604,
        0.040623809960919084,  -0.01552534501956604,  -0.223362481667332,
        0.1821006842509091,    0.2941778783780663,    -0.07901167601970885,
        -0.056815072777201,    -0.3699028221860095,   -0.31616315591624855,
        0.5905951377425391,    -0.052139568068853864, 0.013673160263486295,
        -0.03691647864630577,  0.09732660653298163,   -0.3394662328788498,
        -0.30685677538541667,  -0.24504598907458763,  -0.034698524462007344,
        0.02868032184767538,   -0.21217779266454084,  -0.1678263169941987,
        0.3221287889040614,    -0.1223055851554907,   0.4356604928128464,
        -0.0502599202236253,   0.3979258376211797};
    return values;
  }

  static const std::vector<double>& latent_std_config() {
    static const std::vector<double> values = {
        1.6895524230479284, 2.76263727217653,   1.7945344281264435,
        1.6801681847309828, 1.6390226546605453, 2.7788298348882177,
        1.7659090095747236, 1.6199757612137327, 2.6336525640336896,
        1.8539356672817833, 2.5056497896915633, 1.811019237886178,
        1.9579657790720237, 1.6685498243529284, 1.4922469314453364,
        3.298670198067373,  1.9491804496832168, 1.8720003270431442,
        1.8334080103291832, 1.6488070416529093, 1.6176957696319716,
        1.9131449234774398, 1.5695245398428617, 1.6943659940415912,
        1.8318420762504692, 1.5540637421583379, 1.9344930328968526,
        1.599198216109855,  1.718045989838149,  1.6307219190837705,
        1.8661226051202384, 1.5613768203168363};
    return values;
  }

  static nlohmann::json parse_config(const std::string& path) {
    JsonReader reader;
    if (!reader.parse(path)) {
      throw std::invalid_argument("MiniMax-H3 audio VAE cannot parse config `" +
                                  path + "`");
    }
    return reader.data();
  }

  static void require_exact_keys(
      const nlohmann::json& object,
      const std::string& path,
      const std::unordered_set<std::string>& expected) {
    if (!object.is_object() || object.size() != expected.size()) {
      throw std::invalid_argument("MiniMax-H3 audio VAE config `" + path +
                                  "` has an invalid key set");
    }
    for (const std::string& key : expected) {
      if (!object.contains(key)) {
        throw std::invalid_argument("MiniMax-H3 audio VAE config `" + path +
                                    "` is missing field `" + key + "`");
      }
    }
  }

  static void require_config_value(const nlohmann::json& config,
                                   const std::string& path,
                                   const std::string& key,
                                   const nlohmann::json& expected) {
    if (!config.contains(key) || config.at(key) != expected) {
      throw std::invalid_argument("MiniMax-H3 audio VAE config `" + path +
                                  "` has invalid field `" + key + "`");
    }
  }

  static void require_latent_statistics(const nlohmann::json& config,
                                        const std::string& path) {
    if (!config.contains("latents_mean") || !config.contains("latents_std") ||
        config.at("latents_mean").get<std::vector<double>>() !=
            latent_mean_config() ||
        config.at("latents_std").get<std::vector<double>>() !=
            latent_std_config()) {
      throw std::invalid_argument(
          "MiniMax-H3 audio VAE latent statistics do not match the release "
          "config `" +
          path + "`");
    }
  }

  static void validate_converted_config(const nlohmann::json& config,
                                        const std::string& path) {
    require_exact_keys(config,
                       path,
                       {"_class_name",
                        "_diffusers_version",
                        "encoder_dim",
                        "encoder_rates",
                        "latent_dim",
                        "latent_channels",
                        "decoder_dim",
                        "decoder_rates",
                        "decoder_kernel_sizes",
                        "num_attention_heads",
                        "resblock_kernel_sizes",
                        "resblock_dilation_sizes",
                        "sampling_rate",
                        "latents_mean",
                        "latents_std"});
    if (!config.at("_diffusers_version").is_string()) {
      throw std::invalid_argument(
          "MiniMax-H3 audio VAE converted _diffusers_version must be a string");
    }
    require_config_value(
        config, path, "_class_name", "AutoencoderKLMiniMaxH3Audio");
    require_config_value(config, path, "encoder_dim", 64);
    require_config_value(config, path, "encoder_rates", {2, 4, 4, 5, 5});
    require_config_value(config, path, "latent_dim", 2048);
    require_config_value(config, path, "latent_channels", 32);
    require_config_value(config, path, "decoder_dim", 1024);
    require_config_value(config, path, "decoder_rates", {5, 5, 2, 2, 2, 2, 2});
    require_config_value(
        config, path, "decoder_kernel_sizes", {9, 9, 4, 4, 4, 4, 4});
    require_config_value(config, path, "num_attention_heads", 8);
    require_config_value(config, path, "resblock_kernel_sizes", {3, 7, 11});
    require_config_value(config,
                         path,
                         "resblock_dilation_sizes",
                         {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}});
    require_config_value(config, path, "sampling_rate", 32000);
    require_latent_statistics(config, path);
  }

  static void validate_raw_config(const nlohmann::json& config,
                                  const std::string& path) {
    require_exact_keys(config,
                       path,
                       {"_class_name",
                        "_diffusers_version",
                        "auto_map",
                        "output_channel",
                        "sample_rate",
                        "source_config_path",
                        "source_safetensors_path",
                        "source_metadata_path",
                        "latent_channels",
                        "latents_mean",
                        "latents_std"});
    require_config_value(config, path, "_class_name", "MiniMaxH3AudioVAE");
    require_config_value(config, path, "_diffusers_version", "0.32.2");
    require_config_value(
        config,
        path,
        "auto_map",
        nlohmann::json(
            {{"AutoModel", "minimax_h3_audio_vae.MiniMaxH3AudioVAE"}}));
    require_config_value(config, path, "output_channel", 2);
    require_config_value(config, path, "sample_rate", 32000);
    require_config_value(config, path, "source_config_path", "config.yaml");
    require_config_value(
        config, path, "source_safetensors_path", "model.safetensors");
    require_config_value(config, path, "source_metadata_path", "metadata.json");
    require_config_value(config, path, "latent_channels", 32);
    require_latent_statistics(config, path);
  }

  static void validate_metadata(const std::string& component_path) {
    const std::string path = component_path + "/metadata.json";
    const nlohmann::json root = parse_config(path);
    require_exact_keys(root, path, {"metadata"});
    require_exact_keys(root.at("metadata"), path + ".metadata", {"kwargs"});
    const nlohmann::json& kwargs = root.at("metadata").at("kwargs");
    require_exact_keys(kwargs,
                       path + ".metadata.kwargs",
                       {"attn_proj",
                        "decoder_dim",
                        "decoder_rates",
                        "decoder_type",
                        "encoder_dim",
                        "encoder_rates",
                        "latent_dim",
                        "sample_rate",
                        "vae_latent_channels"});
    require_config_value(kwargs, path, "attn_proj", true);
    require_config_value(kwargs, path, "decoder_dim", 1024);
    require_config_value(kwargs, path, "decoder_rates", {5, 5, 2, 2, 2, 2, 2});
    require_config_value(kwargs, path, "decoder_type", "bigvgan");
    require_config_value(kwargs, path, "encoder_dim", 64);
    require_config_value(kwargs, path, "encoder_rates", {2, 4, 4, 5, 5});
    require_config_value(kwargs, path, "latent_dim", 2048);
    require_config_value(kwargs, path, "sample_rate", 32000);
    require_config_value(kwargs, path, "vae_latent_channels", 32);
  }

  static std::string trim(std::string value) {
    size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin]))) {
      ++begin;
    }
    size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1]))) {
      --end;
    }
    return value.substr(begin, end - begin);
  }

  static void validate_yaml(const std::string& component_path) {
    const std::string path = component_path + "/config.yaml";
    std::ifstream input(path);
    if (!input) {
      throw std::invalid_argument("MiniMax-H3 audio VAE cannot read config `" +
                                  path + "`");
    }
    std::unordered_map<std::string, int64_t> values;
    bool in_model_config = false;
    std::string line;
    while (std::getline(input, line)) {
      const size_t comment = line.find('#');
      if (comment != std::string::npos) {
        line.erase(comment);
      }
      const std::string stripped = trim(line);
      if (stripped.empty()) {
        continue;
      }
      if (stripped == "model_config:") {
        if (in_model_config) {
          throw std::invalid_argument(
              "MiniMax-H3 audio VAE config.yaml repeats model_config");
        }
        in_model_config = true;
        continue;
      }
      if (!in_model_config || line.empty() ||
          !std::isspace(static_cast<unsigned char>(line[0]))) {
        throw std::invalid_argument(
            "MiniMax-H3 audio VAE config.yaml has unsupported structure");
      }
      const size_t separator = stripped.find(':');
      if (separator == std::string::npos ||
          stripped.find(':', separator + 1) != std::string::npos) {
        throw std::invalid_argument(
            "MiniMax-H3 audio VAE config.yaml has an invalid field");
      }
      const std::string key = trim(stripped.substr(0, separator));
      const std::string raw_value = trim(stripped.substr(separator + 1));
      size_t parsed = 0;
      int64_t value = 0;
      try {
        value = std::stoll(raw_value, &parsed);
      } catch (const std::exception&) {
        throw std::invalid_argument(
            "MiniMax-H3 audio VAE config.yaml has a non-integer value");
      }
      if (parsed != raw_value.size() || !values.emplace(key, value).second) {
        throw std::invalid_argument(
            "MiniMax-H3 audio VAE config.yaml has an invalid or duplicate "
            "field");
      }
    }
    const std::unordered_map<std::string, int64_t> expected = {
        {"sr", 32000},
        {"decoder_dim", 1024},
        {"audio_channel", 1},
        {"vae_latent_channels", 32}};
    if (!in_model_config || values != expected) {
      throw std::invalid_argument(
          "MiniMax-H3 audio VAE config.yaml does not match the release");
    }
  }

  static std::string validate_config_files(const std::string& component_path) {
    const std::string path = component_path + "/config.json";
    const nlohmann::json config = parse_config(path);
    if (!config.contains("_class_name") ||
        !config.at("_class_name").is_string()) {
      throw std::invalid_argument(
          "MiniMax-H3 audio VAE config is missing a string _class_name");
    }
    const std::string class_name = config.at("_class_name").get<std::string>();
    if (class_name == "MiniMaxH3AudioVAE") {
      validate_raw_config(config, path);
      validate_metadata(component_path);
      validate_yaml(component_path);
      return "model.safetensors";
    }
    if (class_name == "AutoencoderKLMiniMaxH3Audio") {
      validate_converted_config(config, path);
      return "diffusion_pytorch_model.safetensors";
    }
    throw std::invalid_argument(
        "MiniMax-H3 audio VAE config has an unsupported _class_name");
  }

  static void verify_resident_tensor(
      const std::string& name,
      const torch::Tensor& value,
      std::optional<torch::Device>& resident_device) {
    if (value.scalar_type() != torch::kFloat32) {
      throw std::logic_error("MiniMax-H3 audio VAE tensor is not FP32: `" +
                             name + "`");
    }
    if (!resident_device.has_value()) {
      resident_device = value.device();
    } else if (value.device() != *resident_device) {
      throw std::logic_error(
          "MiniMax-H3 audio VAE is not resident on one device: `" + name + "`");
    }
  }

  torch::TensorOptions options_;
  MiniMaxH3AudioEncoder encoder_{nullptr};
  MiniMaxH3AudioAttnProjection pre_block_{nullptr};
  MiniMaxH3AudioConv1d mean_proj_{nullptr};
  MiniMaxH3AudioConv1d logs_proj_{nullptr};
  MiniMaxH3AudioConv1d dec_in_proj_{nullptr};
  MiniMaxH3AudioBigVGANDecoder decoder_{nullptr};
  std::unordered_set<std::string> loaded_source_names_;
  bool loaded_ = false;
};
TORCH_MODULE(MiniMaxH3AudioVAE);

using AutoencoderKLMiniMaxH3Audio = MiniMaxH3AudioVAE;

}  // namespace xllm
