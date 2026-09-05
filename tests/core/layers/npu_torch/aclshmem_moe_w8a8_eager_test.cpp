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
#include <iostream>
#include <string>
#include <unordered_map>
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

constexpr int32_t kWorldSize = 16;
constexpr int64_t kTokens = 4;
constexpr int64_t kHidden = 4096;
constexpr int64_t kIntermediate = 2048;
constexpr int64_t kGlobalExperts = 256;
constexpr int64_t kLocalExperts = 16;
constexpr int64_t kTopK = 6;
constexpr int32_t kGenerations = 4;

int32_t require_env_int(const char* name) {
  const char* value = std::getenv(name);
  CHECK(value != nullptr) << name << " is required";
  return std::stoi(value);
}

torch::Tensor make_hidden(int32_t rank, int32_t generation) {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  std::vector<torch::Tensor> rows;
  rows.reserve(kTokens);
  for (int64_t token = 0; token < kTokens; ++token) {
    const float value =
        0.25f * static_cast<float>(1 + ((rank + token + generation) % 4));
    rows.push_back(torch::full({kHidden}, value, options));
  }
  return torch::stack(rows).to(torch::kBFloat16);
}

torch::Tensor make_ids(int32_t rank, int32_t generation) {
  const int32_t destination = (rank + generation) % kWorldSize;
  std::vector<int32_t> flattened;
  flattened.reserve(kTokens * kTopK);
  for (int64_t token = 0; token < kTokens; ++token) {
    (void)token;
    for (int64_t slot = 0; slot < kTopK; ++slot) {
      const int64_t local_slot = slot % (kTopK / 2);
      const int32_t route_rank = slot < (kTopK / 2) ? rank : destination;
      flattened.push_back(route_rank * kLocalExperts + local_slot);
    }
  }
  return torch::tensor(flattened, torch::TensorOptions().dtype(torch::kInt32))
      .reshape({kTokens, kTopK});
}

torch::Tensor make_route_weights() {
  return torch::full({kTokens, kTopK},
                     1.0f / static_cast<float>(kTopK),
                     torch::TensorOptions().dtype(torch::kFloat32));
}

std::unordered_map<std::string, torch::Tensor> make_w8a8_weights(
    int32_t rank,
    const torch::Device& device) {
  const auto fp32_options = torch::TensorOptions().dtype(torch::kFloat32);
  const auto int8_options = torch::TensorOptions().dtype(torch::kInt8);

  // Sparse diagonal weights keep the real H4096/I2048 dimensions while making
  // the CPU golden independent of a large dense matrix multiplication.
  const auto gate = torch::eye(kIntermediate, kHidden, fp32_options)
                        .to(torch::kInt8)
                        .contiguous();
  const auto up = gate.clone();
  const auto down = torch::eye(kHidden, kIntermediate, fp32_options)
                        .to(torch::kInt8)
                        .contiguous();
  const auto gate_scale = torch::ones({kIntermediate}, fp32_options);
  const auto up_scale = torch::ones({kIntermediate}, fp32_options);
  const auto down_scale = torch::ones({kHidden}, fp32_options);

  std::unordered_map<std::string, torch::Tensor> weights;
  const int32_t first_expert = rank * kLocalExperts;
  for (int32_t local_expert = 0; local_expert < kLocalExperts; ++local_expert) {
    const int32_t expert = first_expert + local_expert;
    const std::string prefix = "experts." + std::to_string(expert) + ".";
    weights[prefix + "gate_proj.weight"] = gate.to(device);
    weights[prefix + "up_proj.weight"] = up.to(device);
    weights[prefix + "down_proj.weight"] = down.to(device);
    weights[prefix + "gate_proj.weight_scale"] = gate_scale.to(device);
    weights[prefix + "up_proj.weight_scale"] = up_scale.to(device);
    weights[prefix + "down_proj.weight_scale"] = down_scale.to(device);
  }
  return weights;
}

void configure_w8a8_quant_args(QuantArgs& quant_args, int32_t rank) {
  quant_args.quant_method() = "w8a8_dynamic";
  quant_args.quantize_type() = "w8a8_dynamic";
  auto& quant_descs = quant_args.quant_descs();
  // Resolution probes experts.0 before the sharded expert loader selects the
  // rank-local [rank * local_experts, (rank + 1) * local_experts) tensors.
  quant_descs["experts.0.gate_proj.weight"] = "w8a8_dynamic";
  quant_descs["experts.0.up_proj.weight"] = "w8a8_dynamic";
  quant_descs["experts.0.down_proj.weight"] = "w8a8_dynamic";
  const int32_t first_expert = rank * kLocalExperts;
  for (int32_t local_expert = 0; local_expert < kLocalExperts; ++local_expert) {
    const int32_t expert = first_expert + local_expert;
    const std::string prefix = "experts." + std::to_string(expert) + ".";
    quant_descs[prefix + "gate_proj.weight"] = "w8a8_dynamic";
    quant_descs[prefix + "up_proj.weight"] = "w8a8_dynamic";
    quant_descs[prefix + "down_proj.weight"] = "w8a8_dynamic";
  }
}

torch::Tensor expected_output(int32_t rank, int32_t generation) {
  const auto input = make_hidden(rank, generation).to(torch::kFloat32);
  const auto first_half = input.slice(/*dim=*/1, /*start=*/0, kIntermediate);
  const auto activated = torch::silu(first_half) * first_half;
  auto output = torch::zeros({kTokens, kHidden},
                             torch::TensorOptions().dtype(torch::kFloat32));
  output.slice(/*dim=*/1, /*start=*/0, kIntermediate).copy_(activated);
  return output.to(torch::kBFloat16);
}

}  // namespace

class AclShmemMoeW8A8EagerTestPeer {
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

TEST(AclShmemMoeW8A8EagerTest, RunsRealGeometryThroughExactBucket) {
  const int32_t rank = require_env_int("ACLSHMEM_TEST_RANK");
  const int32_t port = require_env_int("ACLSHMEM_TEST_HCCL_PORT");
  auto marker = [rank](const std::string& phase) {
    std::cerr << "[aclshmem_w8a8] rank=" << rank << " phase=" << phase
              << std::endl;
  };
  marker("test_started");
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
                                       "aclshmem_moe_w8a8_eager_test",
                                       device);
  marker("ep_group_created");
  auto tp_group = std::make_unique<ProcessGroup>(0, 1, device);
  ParallelArgs parallel_args(rank, kWorldSize, ep_group.get());
  parallel_args.ep_size() = kWorldSize;
  parallel_args.moe_ep_group_ = ep_group.get();
  parallel_args.moe_tp_group_ = tp_group.get();
  parallel_args.tp_group_ = tp_group.get();

  // This flag selects the existing xLLM EP2 dispatch/combine control path;
  // the actual transport below still uses all 16 EP ranks.
  EPLBConfig::get_instance().expert_parallel_degree(2);
  KernelConfig::get_instance().enable_aclshmem_moe(true);

  ModelArgs model_args;
  model_args.model_type() = "deepseek_v4";
  model_args.n_routed_experts() = kGlobalExperts;
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
  configure_w8a8_quant_args(quant_args, rank);
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
  marker("fused_moe_constructed");
  moe->load_state_dict(StateDict(make_w8a8_weights(rank, device)));
  marker("weights_loaded");
  device_handle.synchronize_default_stream();
  marker("weights_synchronized");

  ModelInputParams input_params;
  input_params.enable_graph = false;
  const auto expected_counts =
      torch::tensor({8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                    torch::TensorOptions().dtype(torch::kInt64));
  for (int32_t generation = 1; generation <= kGenerations; ++generation) {
    const std::string generation_prefix =
        "generation_" + std::to_string(generation) + "_";
    marker(generation_prefix + "start");
    const auto hidden = make_hidden(rank, generation).to(device);
    const auto ids = make_ids(rank, generation).to(device);
    const auto route_weights = make_route_weights().to(device);
    const auto output = moe->forward_with_selected_experts(
        hidden, route_weights, ids, input_params);
    marker(generation_prefix + "forward_returned");
    device_handle.synchronize_default_stream();
    marker(generation_prefix + "synchronized");

    const auto expected = expected_output(rank, generation);
    const auto output_cpu = output.cpu();
    ASSERT_TRUE(torch::isfinite(output_cpu).all().item<bool>());
    ASSERT_TRUE(torch::allclose(output_cpu, expected, 0.1, 0.05));
    marker(generation_prefix + "golden_pass");
    ASSERT_TRUE(AclShmemMoeW8A8EagerTestPeer::initialized(*moe));
    ASSERT_EQ(AclShmemMoeW8A8EagerTestPeer::generation(*moe), generation);
    ASSERT_TRUE(torch::equal(AclShmemMoeW8A8EagerTestPeer::expert_counts(*moe),
                             expected_counts));
  }
}

}  // namespace xllm::layer
