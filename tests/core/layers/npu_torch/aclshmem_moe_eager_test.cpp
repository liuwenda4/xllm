/* Copyright 2025-2026 The xLLM Authors.

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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdlib>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "framework/config/eplb_config.h"
#include "framework/config/kernel_config.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/process_group.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/npu_torch/fused_moe.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm::layer {
namespace {

constexpr int32_t kWorldSize = 2;
constexpr int64_t kTokens = 4;
constexpr int64_t kHidden = 37;
constexpr int64_t kIntermediate = 32;
constexpr int64_t kExperts = 4;
constexpr int64_t kLocalExperts = 2;
constexpr int64_t kTopK = 2;

int32_t require_env_int(const char* name) {
  const char* value = std::getenv(name);
  CHECK(value != nullptr) << name << " is required";
  return std::stoi(value);
}

torch::Tensor deterministic_weight(at::IntArrayRef shape,
                                   int64_t expert_id,
                                   int64_t offset) {
  const int64_t elements = std::accumulate(
      shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
  auto values =
      torch::arange(elements, torch::TensorOptions().dtype(torch::kFloat32));
  values = ((values + expert_id * 7 + offset).remainder(17) - 8) * 0.005f;
  return values.reshape(shape).to(torch::kBFloat16);
}

std::unordered_map<std::string, torch::Tensor> make_weights(
    const torch::Device& device,
    std::vector<torch::Tensor>* gate_weights,
    std::vector<torch::Tensor>* up_weights,
    std::vector<torch::Tensor>* down_weights) {
  std::unordered_map<std::string, torch::Tensor> weights;
  for (int64_t expert = 0; expert < kExperts; ++expert) {
    auto gate = deterministic_weight({kIntermediate, kHidden}, expert, 0);
    auto up = deterministic_weight({kIntermediate, kHidden}, expert, 3);
    auto down = deterministic_weight({kHidden, kIntermediate}, expert, 5);
    gate_weights->push_back(gate);
    up_weights->push_back(up);
    down_weights->push_back(down);
    const std::string prefix = "experts." + std::to_string(expert) + ".";
    weights[prefix + "gate_proj.weight"] = gate.to(device);
    weights[prefix + "up_proj.weight"] = up.to(device);
    weights[prefix + "down_proj.weight"] = down.to(device);
  }
  return weights;
}

torch::Tensor make_hidden(int32_t rank, int32_t generation) {
  auto columns = torch::linspace(
      -1.75, 2.25, kHidden, torch::TensorOptions().dtype(torch::kFloat32));
  std::vector<torch::Tensor> rows;
  for (int64_t token = 0; token < kTokens; ++token) {
    const int64_t factor = rank * kTokens + token + generation;
    rows.push_back(columns * factor + (generation - 1) * 0.3125f);
  }
  auto result = torch::stack(rows).to(torch::kBFloat16);
  if (rank == 0 && generation == 1) {
    result[0].zero_();
  }
  return result;
}

torch::Tensor make_ids(int32_t rank, int32_t generation) {
  std::vector<std::vector<int32_t>> routes =
      rank == 0
          ? std::vector<std::vector<int32_t>>{{0, 2}, {1, 3}, {2, 0}, {3, 1}}
          : std::vector<std::vector<int32_t>>{{3, 1}, {2, 0}, {1, 3}, {0, 2}};
  if (generation % 2 == 0) {
    for (auto& row : routes) {
      for (int32_t& expert : row) {
        expert = (expert + kLocalExperts) % kExperts;
      }
    }
  }
  std::vector<int32_t> flattened;
  for (const auto& row : routes) {
    flattened.insert(flattened.end(), row.begin(), row.end());
  }
  return torch::tensor(flattened, torch::TensorOptions().dtype(torch::kInt32))
      .reshape({kTokens, kTopK});
}

torch::Tensor make_route_weights(int32_t rank, int32_t generation) {
  auto weights = torch::tensor(
      {{0.25f, 0.75f}, {-0.5f, 1.25f}, {1.5f, -0.25f}, {0.1f, 0.9f}},
      torch::TensorOptions().dtype(torch::kFloat32));
  return weights + rank * 0.125f + (generation - 1) * 0.0625f;
}

torch::Tensor quant_dequant_reference(const torch::Tensor& input) {
  auto fp32 = input.to(torch::kFloat32);
  auto absmax = std::get<0>(fp32.abs().max(/*dim=*/1));
  auto scale =
      torch::where(absmax == 0, torch::ones_like(absmax), absmax / 127.0);
  auto payload =
      torch::round(fp32 / scale.unsqueeze(1)).clamp(-127, 127).to(torch::kInt8);
  return (payload.to(torch::kFloat32) * scale.unsqueeze(1))
      .to(torch::kBFloat16);
}

torch::Tensor expert_reference(const torch::Tensor& input,
                               const torch::Tensor& gate,
                               const torch::Tensor& up,
                               const torch::Tensor& down) {
  auto gate_out =
      (input.to(torch::kFloat32).unsqueeze(0) * gate.to(torch::kFloat32))
          .sum(/*dim=*/1)
          .to(torch::kBFloat16)
          .to(torch::kFloat32);
  auto up_out =
      (input.to(torch::kFloat32).unsqueeze(0) * up.to(torch::kFloat32))
          .sum(/*dim=*/1)
          .to(torch::kBFloat16)
          .to(torch::kFloat32);
  auto activated =
      (torch::silu(gate_out) * up_out).to(torch::kBFloat16).to(torch::kFloat32);
  return (down.to(torch::kFloat32) * activated.unsqueeze(0))
      .sum(/*dim=*/1)
      .to(torch::kBFloat16);
}

torch::Tensor expected_output(int32_t rank,
                              int32_t generation,
                              const std::vector<torch::Tensor>& gate_weights,
                              const std::vector<torch::Tensor>& up_weights,
                              const std::vector<torch::Tensor>& down_weights) {
  const auto hidden = quant_dequant_reference(make_hidden(rank, generation));
  const auto ids = make_ids(rank, generation);
  const auto route_weights = make_route_weights(rank, generation);
  auto output = torch::zeros({kTokens, kHidden},
                             torch::TensorOptions().dtype(torch::kFloat32));
  for (int64_t token = 0; token < kTokens; ++token) {
    for (int64_t slot = 0; slot < kTopK; ++slot) {
      const int64_t expert = ids[token][slot].item<int32_t>();
      output[token] += expert_reference(hidden[token],
                                        gate_weights[expert],
                                        up_weights[expert],
                                        down_weights[expert])
                           .to(torch::kFloat32) *
                       route_weights[token][slot].item<float>();
    }
  }
  return output.to(torch::kBFloat16);
}

}  // namespace

class AclShmemMoeEagerTestPeer {
 public:
  static bool initialized(const FusedMoEImpl& moe) {
    return moe.aclshmem_moe_resource_ != nullptr;
  }

  static int32_t generation(const FusedMoEImpl& moe) {
    return moe.aclshmem_generation_id_.cpu()[0].item<int32_t>();
  }

  static torch::Tensor expert_counts(const FusedMoEImpl& moe) {
    return moe.aclshmem_expert_token_nums_.cpu();
  }
};

TEST(AclShmemMoeEagerTest, RunsSelectedExpertsThroughExactBucket) {
  const int32_t rank = require_env_int("ACLSHMEM_TEST_RANK");
  const int32_t port = require_env_int("ACLSHMEM_TEST_HCCL_PORT");
  ASSERT_GE(rank, 0);
  ASSERT_LT(rank, kWorldSize);
  ASSERT_GE(Platform::device_count(), kWorldSize);

  Device device_handle(rank);
  device_handle.set_device();
  const torch::Device device = device_handle.unwrap();
  auto ep_group = create_process_group(rank,
                                       kWorldSize,
                                       kWorldSize,
                                       port,
                                       false,
                                       "127.0.0.1",
                                       "aclshmem_moe_eager_test",
                                       device);
  auto tp_group = std::make_unique<ProcessGroup>(0, 1, device);
  ParallelArgs parallel_args(rank, kWorldSize, ep_group.get());
  parallel_args.ep_size() = kWorldSize;
  parallel_args.moe_ep_group_ = ep_group.get();
  parallel_args.moe_tp_group_ = tp_group.get();
  parallel_args.tp_group_ = tp_group.get();

  EPLBConfig::get_instance().expert_parallel_degree(2);
  KernelConfig::get_instance().enable_aclshmem_moe(true);
  ModelArgs model_args;
  model_args.model_type() = "deepseek_v4";
  model_args.n_routed_experts() = kExperts;
  model_args.num_experts_per_tok() = kTopK;
  model_args.n_group() = 1;
  model_args.topk_group() = 1;
  model_args.routed_scaling_factor() = 1.0f;
  model_args.hidden_size() = kHidden;
  model_args.moe_intermediate_size() = kIntermediate;
  model_args.n_shared_experts() = 0;
  model_args.norm_topk_prob() = false;
  model_args.hidden_act() = "silu";
  model_args.scoring_func() = "softmax";
  model_args.topk_method() = "greedy";
  QuantArgs quant_args;
  const auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(device)
                           .requires_grad(false);
  FusedMoE moe(
      FusedMoEImpl(model_args,
                   FusedMoEArgs{.is_gated = true, .skip_gate_load = true},
                   quant_args,
                   parallel_args,
                   options));

  std::vector<torch::Tensor> gate_weights;
  std::vector<torch::Tensor> up_weights;
  std::vector<torch::Tensor> down_weights;
  auto weights =
      make_weights(device, &gate_weights, &up_weights, &down_weights);
  moe->load_state_dict(StateDict(std::move(weights)));

  ModelInputParams input_params;
  input_params.enable_graph = false;
  for (int32_t generation = 1; generation <= 2; ++generation) {
    const auto hidden = make_hidden(rank, generation).to(device);
    const auto ids = make_ids(rank, generation).to(device);
    const auto route_weights = make_route_weights(rank, generation).to(device);
    const auto output = moe->forward_with_selected_experts(
        hidden, route_weights, ids, input_params);
    device_handle.synchronize_default_stream();
    const auto expected = expected_output(
        rank, generation, gate_weights, up_weights, down_weights);
    ASSERT_TRUE(torch::allclose(output.cpu(), expected, 1e-2, 2e-3));
    ASSERT_TRUE(AclShmemMoeEagerTestPeer::initialized(*moe));
    ASSERT_EQ(AclShmemMoeEagerTestPeer::generation(*moe), generation);
    ASSERT_TRUE(torch::equal(AclShmemMoeEagerTestPeer::expert_counts(*moe),
                             torch::tensor({4, 4}, torch::kInt64)));
  }
}

}  // namespace xllm::layer
