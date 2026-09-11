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
#include <limits>
#include <stdexcept>
#include <vector>

namespace xllm {

struct MiniMaxH3SigmaSchedule {
  torch::Tensor sigmas;
  torch::Tensor timesteps;

  int64_t point_count() const { return sigmas.numel(); }
  int64_t forward_count() const { return timesteps.numel(); }
};

struct MiniMaxH3DualSigmaSchedule {
  MiniMaxH3SigmaSchedule video;
  MiniMaxH3SigmaSchedule audio;
};

class MiniMaxH3Scheduler final {
 public:
  static constexpr int64_t kBasePointCount = 50;
  static constexpr float kVideoShift = 12.0F;
  static constexpr float kAudioShift = 3.0F;

  static MiniMaxH3SigmaSchedule build(
      int64_t point_count,
      float shift,
      const torch::Device& device = torch::kCPU) {
    if (point_count < 2) {
      throw std::invalid_argument(
          "MiniMax-H3 sigma schedule requires at least two points");
    }
    if (!std::isfinite(shift) || shift <= 0.0F) {
      throw std::invalid_argument(
          "MiniMax-H3 sigma schedule shift must be finite and positive");
    }

    const torch::Tensor base = torch::linspace(
        1.0F, 0.0F, point_count, torch::TensorOptions().dtype(torch::kFloat32));
    const torch::Tensor shifted =
        (shift * base / (1.0F + (shift - 1.0F) * base)).contiguous();
    const float* shifted_values = shifted.const_data_ptr<float>();
    std::vector<float> unique_values;
    unique_values.reserve(static_cast<size_t>(point_count));
    for (int64_t index = 0; index < point_count; ++index) {
      const float value = shifted_values[index];
      if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "MiniMax-H3 sigma schedule contains a non-finite value");
      }
      if (unique_values.empty() || value != unique_values.back()) {
        unique_values.push_back(value);
      }
    }
    if (unique_values.size() < 2 || unique_values.front() != 1.0F ||
        unique_values.back() != 0.0F) {
      throw std::invalid_argument(
          "MiniMax-H3 sigma schedule must descend from one to zero");
    }
    for (size_t index = 1; index < unique_values.size(); ++index) {
      if (!(unique_values[index] < unique_values[index - 1])) {
        throw std::invalid_argument(
            "MiniMax-H3 sigma schedule must be strictly decreasing");
      }
    }

    torch::Tensor sigmas =
        torch::from_blob(unique_values.data(),
                         {static_cast<int64_t>(unique_values.size())},
                         torch::kFloat32)
            .clone()
            .to(device);
    torch::Tensor timesteps = 1.0F - sigmas.slice(0, 0, sigmas.size(0) - 1);
    return {.sigmas = std::move(sigmas), .timesteps = std::move(timesteps)};
  }

  static MiniMaxH3DualSigmaSchedule build_base(
      const torch::Device& device = torch::kCPU) {
    return {.video = build(kBasePointCount, kVideoShift, device),
            .audio = build(kBasePointCount, kAudioShift, device)};
  }

  static torch::Tensor velocity_to_x0(const torch::Tensor& state,
                                      const torch::Tensor& velocity,
                                      const torch::Tensor& timestep) {
    validate_state_pair(state, velocity, "velocity");
    if (!timestep.defined() || timestep.numel() != 1 ||
        !timestep.is_floating_point() ||
        !torch::isfinite(timestep).all().item<bool>()) {
      throw std::invalid_argument(
          "MiniMax-H3 timestep must be one finite floating-point value");
    }
    const double timestep_value = timestep.item<double>();
    if (timestep_value < 0.0 || timestep_value > 1.0) {
      throw std::invalid_argument("MiniMax-H3 timestep must be in [0,1]");
    }
    torch::Tensor sigma_from_timestep =
        1.0F - timestep.to(state.device(), state.scalar_type());
    while (sigma_from_timestep.dim() < state.dim()) {
      sigma_from_timestep.unsqueeze_(-1);
    }
    torch::Tensor denoised = state + sigma_from_timestep * velocity;
    if (!torch::isfinite(denoised).all().item<bool>()) {
      throw std::invalid_argument("MiniMax-H3 x0 contains NaN or Inf");
    }
    return denoised;
  }

  static torch::Tensor step_eta0(const torch::Tensor& state,
                                 const torch::Tensor& denoised,
                                 float sigma_current,
                                 float sigma_next) {
    validate_state_pair(state, denoised, "denoised");
    validate_sigma(sigma_current, "sigma_current");
    validate_sigma(sigma_next, "sigma_next");
    if (sigma_next > sigma_current) {
      throw std::invalid_argument(
          "MiniMax-H3 sigma_next must not exceed sigma_current");
    }
    if (sigma_current == 0.0F) {
      if (sigma_next != 0.0F) {
        throw std::invalid_argument(
            "MiniMax-H3 sigma_next must be zero at terminal sigma");
      }
      return state;
    }

    const torch::ScalarType compute_dtype =
        (state.scalar_type() == torch::kFloat16 ||
         state.scalar_type() == torch::kBFloat16)
            ? torch::kFloat32
            : state.scalar_type();
    const torch::TensorOptions compute_options =
        state.options().dtype(compute_dtype);
    const torch::Tensor current =
        torch::full({}, sigma_current, compute_options);
    const torch::Tensor next = torch::full({}, sigma_next, compute_options);
    const torch::Tensor ratio = next / current;
    torch::Tensor output = ratio * state.to(compute_dtype) +
                           (1.0F - ratio) * denoised.to(compute_dtype);
    output = output.to(state.scalar_type());
    if (!torch::isfinite(output).all().item<bool>()) {
      throw std::invalid_argument(
          "MiniMax-H3 eta-0 scheduler output contains NaN or Inf");
    }
    return output;
  }

 private:
  static void validate_sigma(float sigma, const char* name) {
    if (!std::isfinite(sigma) || sigma < 0.0F) {
      throw std::invalid_argument(std::string("MiniMax-H3 ") + name +
                                  " must be finite and non-negative");
    }
  }

  static void validate_state_pair(const torch::Tensor& state,
                                  const torch::Tensor& other,
                                  const char* other_name) {
    if (!state.defined() || !other.defined() || !state.is_floating_point() ||
        !other.is_floating_point() || state.sizes() != other.sizes() ||
        state.device() != other.device()) {
      throw std::invalid_argument(
          std::string("MiniMax-H3 state and ") + other_name +
          " must be same-shape floating tensors on one device");
    }
    if (!torch::isfinite(state).all().item<bool>() ||
        !torch::isfinite(other).all().item<bool>()) {
      throw std::invalid_argument(std::string("MiniMax-H3 state or ") +
                                  other_name + " contains NaN or Inf");
    }
  }
};

}  // namespace xllm
