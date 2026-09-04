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
#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_ACLSHMEM_MOE_DEQUANT_INT8_REGISTRY_INC
#error "XLLM_TL_ACLSHMEM_MOE_DEQUANT_INT8_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_ACLSHMEM_MOE_DEQUANT_INT8_REGISTRY_INC

AclshmemMoeDequantInt8Specialization make_dequant_specialization(
    int64_t max_capacity,
    int64_t hidden_size) {
  return make_aclshmem_moe_dequant_int8_specialization(
      AclshmemMoeDequantInt8MaxCapacity{static_cast<int32_t>(max_capacity)},
      AclshmemMoeDequantInt8HiddenSize{static_cast<int32_t>(hidden_size)});
}

void check_dequant_tensor(const torch::Tensor& tensor,
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

bool has_aclshmem_moe_dequant_int8_specialization(int64_t max_capacity,
                                                  int64_t hidden_size) {
  if (max_capacity <= 0 || hidden_size <= 0 ||
      max_capacity > std::numeric_limits<int32_t>::max() ||
      hidden_size > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  return find_aclshmem_moe_dequant_int8_kernel_entry(
             make_dequant_specialization(max_capacity, hidden_size)) != nullptr;
}

void aclshmem_moe_dequant_int8(const torch::Tensor& payload,
                               const torch::Tensor& scale,
                               const torch::Tensor& active_mask,
                               torch::Tensor& output) {
  CHECK_EQ(payload.dim(), 2);
  CHECK_EQ(output.dim(), 2);
  const int64_t max_capacity = payload.size(0);
  const int64_t physical_hidden = payload.size(1);
  const int64_t hidden_size = output.size(1);
  CHECK_EQ(physical_hidden,
           std::max<int64_t>(32, (hidden_size + 31) / 32 * 32));
  CHECK(
      has_aclshmem_moe_dequant_int8_specialization(max_capacity, hidden_size));

  const torch::Device device = payload.device();
  check_dequant_tensor(payload,
                       torch::kInt8,
                       {max_capacity, physical_hidden},
                       "payload",
                       device);
  check_dequant_tensor(scale, torch::kFloat32, {max_capacity}, "scale", device);
  check_dequant_tensor(
      active_mask, torch::kInt32, {max_capacity}, "active_mask", device);
  check_dequant_tensor(
      output, torch::kBFloat16, {max_capacity, hidden_size}, "output", device);

  const auto* entry = find_aclshmem_moe_dequant_int8_kernel_entry(
      make_dequant_specialization(max_capacity, hidden_size));
  CHECK(entry != nullptr);
  const int32_t device_id = payload.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(payload.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(scale.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(active_mask.data_ptr())),
      reinterpret_cast<uint8_t*>(output.data_ptr()),
      stream);
}

}  // namespace xllm::kernel::npu::tilelang
