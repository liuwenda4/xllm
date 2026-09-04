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

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_ACLSHMEM_MOE_COMBINE_BF16_REGISTRY_INC
#error "XLLM_TL_ACLSHMEM_MOE_COMBINE_BF16_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_ACLSHMEM_MOE_COMBINE_BF16_REGISTRY_INC

AclshmemMoeCombineBf16Specialization make_combine_specialization(
    int64_t local_tokens,
    int64_t hidden_size,
    int64_t topk,
    int64_t ep_world_size,
    int64_t local_experts,
    int64_t rank) {
  return make_aclshmem_moe_combine_bf16_specialization(
      AclshmemMoeCombineBf16LocalTokens{static_cast<int32_t>(local_tokens)},
      AclshmemMoeCombineBf16HiddenSize{static_cast<int32_t>(hidden_size)},
      AclshmemMoeCombineBf16Topk{static_cast<int32_t>(topk)},
      AclshmemMoeCombineBf16EpWorldSize{static_cast<int32_t>(ep_world_size)},
      AclshmemMoeCombineBf16LocalExperts{static_cast<int32_t>(local_experts)},
      AclshmemMoeCombineBf16Rank{static_cast<int32_t>(rank)});
}

void check_combine_tensor(const torch::Tensor& tensor,
                          c10::ScalarType dtype,
                          at::IntArrayRef shape,
                          const char* name,
                          const torch::Device& device) {
  CHECK(tensor.defined() && tensor.is_contiguous() &&
        tensor.device().type() == c10::DeviceType::PrivateUse1)
      << name << " must be a contiguous NPU tensor";
  CHECK_EQ(tensor.scalar_type(), dtype) << name << " dtype mismatch";
  CHECK_EQ(tensor.sizes(), shape) << name << " shape mismatch";
  CHECK_EQ(tensor.device(), device) << name << " device mismatch";
}

}  // namespace

bool has_aclshmem_moe_combine_bf16_specialization(int64_t local_tokens,
                                                  int64_t hidden_size,
                                                  int64_t topk,
                                                  int64_t ep_world_size,
                                                  int64_t local_experts,
                                                  int64_t rank) {
  if (local_tokens <= 0 || hidden_size <= 0 || topk <= 0 ||
      ep_world_size <= 1 || local_experts <= 0 || rank < 0 ||
      rank >= ep_world_size || local_tokens > INT32_MAX ||
      hidden_size > INT32_MAX || topk > INT32_MAX ||
      ep_world_size > INT32_MAX || local_experts > INT32_MAX) {
    return false;
  }
  return find_aclshmem_moe_combine_bf16_kernel_entry(
             make_combine_specialization(local_tokens,
                                         hidden_size,
                                         topk,
                                         ep_world_size,
                                         local_experts,
                                         rank)) != nullptr;
}

void aclshmem_moe_combine_bf16(AclShmemMoeCombineBf16Params& params) {
  CHECK_EQ(params.expert_output.dim(), 2);
  CHECK_EQ(params.expand_ids.dim(), 2);
  CHECK_EQ(params.route_weights.dim(), 2);
  const int64_t max_capacity = params.expert_output.size(0);
  const int64_t hidden_size = params.expert_output.size(1);
  const int64_t local_tokens = params.route_weights.size(0);
  const int64_t topk = params.route_weights.size(1);
  CHECK_GT(params.ep_world_size, 1);
  CHECK_GT(params.local_experts, 0);
  CHECK(has_aclshmem_moe_combine_bf16_specialization(local_tokens,
                                                     hidden_size,
                                                     topk,
                                                     params.ep_world_size,
                                                     params.local_experts,
                                                     params.rank));
  CHECK_EQ(max_capacity,
           params.ep_world_size * local_tokens * params.local_experts);

  const torch::Device device = params.expert_output.device();
  check_combine_tensor(params.expert_output,
                       torch::kBFloat16,
                       {max_capacity, hidden_size},
                       "expert_output",
                       device);
  check_combine_tensor(params.expand_ids,
                       torch::kInt32,
                       {max_capacity, 3},
                       "expand_ids",
                       device);
  check_combine_tensor(
      params.active_mask, torch::kInt32, {max_capacity}, "active_mask", device);
  check_combine_tensor(
      params.generation_id, torch::kInt32, {1}, "generation_id", device);
  check_combine_tensor(
      params.iteration_id, torch::kInt32, {1}, "iteration_id", device);
  check_combine_tensor(params.route_weights,
                       torch::kFloat32,
                       {local_tokens, topk},
                       "route_weights",
                       device);
  check_combine_tensor(params.output,
                       torch::kBFloat16,
                       {local_tokens, hidden_size},
                       "output",
                       device);
  CHECK(params.win_payload != nullptr && params.win_status != nullptr &&
        params.win_credit != nullptr)
      << "ACLSHMEM Combine requires payload, status, and credit windows";

  const auto specialization = make_combine_specialization(local_tokens,
                                                          hidden_size,
                                                          topk,
                                                          params.ep_world_size,
                                                          params.local_experts,
                                                          params.rank);
  const auto* entry =
      find_aclshmem_moe_combine_bf16_kernel_entry(specialization);
  CHECK(entry != nullptr);
  const int32_t device_id = params.expert_output.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  entry->fn(reinterpret_cast<uint8_t*>(params.expert_output.data_ptr()),
            reinterpret_cast<uint8_t*>(params.expand_ids.data_ptr()),
            reinterpret_cast<uint8_t*>(params.active_mask.data_ptr()),
            reinterpret_cast<uint8_t*>(params.generation_id.data_ptr()),
            reinterpret_cast<uint8_t*>(params.iteration_id.data_ptr()),
            reinterpret_cast<uint8_t*>(params.route_weights.data_ptr()),
            reinterpret_cast<uint8_t*>(params.win_payload),
            reinterpret_cast<uint8_t*>(params.win_status),
            reinterpret_cast<uint8_t*>(params.win_credit),
            reinterpret_cast<uint8_t*>(params.output.data_ptr()),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
