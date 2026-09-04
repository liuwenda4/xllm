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

#include <algorithm>
#include <array>
#include <cstdint>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_ACLSHMEM_MOE_DISPATCH_INT8_REGISTRY_INC
#error "XLLM_TL_ACLSHMEM_MOE_DISPATCH_INT8_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_ACLSHMEM_MOE_DISPATCH_INT8_REGISTRY_INC

bool is_npu_contiguous(const torch::Tensor& tensor) {
  return tensor.defined() && tensor.is_contiguous() &&
         tensor.device().type() == c10::DeviceType::PrivateUse1;
}

AclshmemMoeDispatchInt8Specialization make_specialization(int64_t local_tokens,
                                                          int64_t hidden_size,
                                                          int64_t topk,
                                                          int64_t ep_world_size,
                                                          int64_t local_experts,
                                                          int64_t rank) {
  return make_aclshmem_moe_dispatch_int8_specialization(
      AclshmemMoeDispatchInt8LocalTokens{static_cast<int32_t>(local_tokens)},
      AclshmemMoeDispatchInt8HiddenSize{static_cast<int32_t>(hidden_size)},
      AclshmemMoeDispatchInt8Topk{static_cast<int32_t>(topk)},
      AclshmemMoeDispatchInt8EpWorldSize{static_cast<int32_t>(ep_world_size)},
      AclshmemMoeDispatchInt8LocalExperts{static_cast<int32_t>(local_experts)},
      AclshmemMoeDispatchInt8Rank{static_cast<int32_t>(rank)});
}

void check_tensor(const torch::Tensor& tensor,
                  c10::ScalarType dtype,
                  at::IntArrayRef shape,
                  const char* name,
                  const torch::Device& device) {
  CHECK(is_npu_contiguous(tensor))
      << name << " must be a contiguous NPU tensor";
  CHECK_EQ(tensor.scalar_type(), dtype) << name << " dtype mismatch";
  CHECK_EQ(tensor.sizes(), shape) << name << " shape mismatch";
  CHECK_EQ(tensor.device(), device) << name << " device mismatch";
}

}  // namespace

bool has_aclshmem_moe_dispatch_int8_specialization(int64_t local_tokens,
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
  return find_aclshmem_moe_dispatch_int8_kernel_entry(
             make_specialization(local_tokens,
                                 hidden_size,
                                 topk,
                                 ep_world_size,
                                 local_experts,
                                 rank)) != nullptr;
}

void aclshmem_moe_dispatch_int8(AclShmemMoeDispatchInt8Params& params) {
  CHECK_EQ(params.payload.dim(), 2);
  CHECK_EQ(params.expert_ids.dim(), 2);
  const int64_t local_tokens = params.payload.size(0);
  const int64_t physical_hidden = params.payload.size(1);
  const int64_t topk = params.expert_ids.size(1);
  const int64_t hidden_size = params.hidden_size;
  CHECK_GT(hidden_size, 0);
  CHECK_GT(physical_hidden, 0);
  CHECK_EQ(physical_hidden,
           std::max<int64_t>(32, (hidden_size + 31) / 32 * 32));
  CHECK_GT(params.ep_world_size, 1);
  CHECK_GT(params.local_experts, 0);
  CHECK(has_aclshmem_moe_dispatch_int8_specialization(local_tokens,
                                                      hidden_size,
                                                      topk,
                                                      params.ep_world_size,
                                                      params.local_experts,
                                                      params.rank));
  const int64_t max_capacity =
      params.ep_world_size * local_tokens * params.local_experts;
  const auto specialization = make_specialization(local_tokens,
                                                  hidden_size,
                                                  topk,
                                                  params.ep_world_size,
                                                  params.local_experts,
                                                  params.rank);
  const auto* entry =
      find_aclshmem_moe_dispatch_int8_kernel_entry(specialization);
  CHECK(entry != nullptr);

  const torch::Device device = params.payload.device();
  check_tensor(params.payload,
               torch::kInt8,
               {local_tokens, physical_hidden},
               "payload",
               device);
  check_tensor(params.scale, torch::kFloat32, {local_tokens}, "scale", device);
  check_tensor(params.expert_ids,
               torch::kInt32,
               {local_tokens, topk},
               "expert_ids",
               device);
  check_tensor(
      params.generation_id, torch::kInt32, {1}, "generation_id", device);
  check_tensor(params.iteration_id, torch::kInt32, {1}, "iteration_id", device);
  check_tensor(params.expand_payload,
               torch::kInt8,
               {max_capacity, physical_hidden},
               "expand_payload",
               device);
  check_tensor(params.expand_scale,
               torch::kFloat32,
               {max_capacity},
               "expand_scale",
               device);
  check_tensor(params.expand_ids,
               torch::kInt32,
               {max_capacity, 3},
               "expand_ids",
               device);
  check_tensor(params.global_prefix,
               torch::kInt32,
               {params.ep_world_size * params.local_experts},
               "global_prefix",
               device);
  check_tensor(params.expert_token_nums,
               torch::kInt64,
               {params.local_experts},
               "expert_token_nums",
               device);
  check_tensor(params.ep_receive_count,
               torch::kInt32,
               {params.local_experts},
               "ep_receive_count",
               device);
  check_tensor(
      params.active_mask, torch::kInt32, {max_capacity}, "active_mask", device);
  check_tensor(params.actual_count, torch::kInt32, {1}, "actual_count", device);
  CHECK(params.win_payload != nullptr && params.win_scale != nullptr &&
        params.win_triplet != nullptr && params.win_status != nullptr &&
        params.win_credit != nullptr)
      << "ACLSHMEM Dispatch requires all symmetric windows";

  const int32_t device_id = params.payload.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  entry->fn(reinterpret_cast<uint8_t*>(params.payload.data_ptr()),
            reinterpret_cast<uint8_t*>(params.scale.data_ptr()),
            reinterpret_cast<uint8_t*>(params.expert_ids.data_ptr()),
            reinterpret_cast<uint8_t*>(params.generation_id.data_ptr()),
            reinterpret_cast<uint8_t*>(params.iteration_id.data_ptr()),
            reinterpret_cast<uint8_t*>(params.win_payload),
            reinterpret_cast<uint8_t*>(params.win_scale),
            reinterpret_cast<uint8_t*>(params.win_triplet),
            reinterpret_cast<uint8_t*>(params.win_status),
            reinterpret_cast<uint8_t*>(params.win_credit),
            reinterpret_cast<uint8_t*>(params.expand_payload.data_ptr()),
            reinterpret_cast<uint8_t*>(params.expand_scale.data_ptr()),
            reinterpret_cast<uint8_t*>(params.expand_ids.data_ptr()),
            reinterpret_cast<uint8_t*>(params.global_prefix.data_ptr()),
            reinterpret_cast<uint8_t*>(params.expert_token_nums.data_ptr()),
            reinterpret_cast<uint8_t*>(params.ep_receive_count.data_ptr()),
            reinterpret_cast<uint8_t*>(params.active_mask.data_ptr()),
            reinterpret_cast<uint8_t*>(params.actual_count.data_ptr()),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
