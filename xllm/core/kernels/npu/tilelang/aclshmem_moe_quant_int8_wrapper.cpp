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

#ifndef XLLM_TL_ACLSHMEM_MOE_QUANT_INT8_REGISTRY_INC
#error "XLLM_TL_ACLSHMEM_MOE_QUANT_INT8_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_ACLSHMEM_MOE_QUANT_INT8_REGISTRY_INC

AclshmemMoeQuantInt8Specialization make_quant_specialization(
    int64_t local_tokens,
    int64_t hidden_size) {
  return make_aclshmem_moe_quant_int8_specialization(
      AclshmemMoeQuantInt8LocalTokens{static_cast<int32_t>(local_tokens)},
      AclshmemMoeQuantInt8HiddenSize{static_cast<int32_t>(hidden_size)});
}

void check_quant_tensor(const torch::Tensor& tensor,
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

bool has_aclshmem_moe_quant_int8_specialization(int64_t local_tokens,
                                                int64_t hidden_size) {
  if (local_tokens <= 0 || hidden_size <= 0 ||
      local_tokens > std::numeric_limits<int32_t>::max() ||
      hidden_size > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  return find_aclshmem_moe_quant_int8_kernel_entry(
             make_quant_specialization(local_tokens, hidden_size)) != nullptr;
}

void aclshmem_moe_quant_int8(const torch::Tensor& input,
                             torch::Tensor& payload,
                             torch::Tensor& scale) {
  CHECK_EQ(input.dim(), 2);
  const int64_t local_tokens = input.size(0);
  const int64_t hidden_size = input.size(1);
  const int64_t physical_hidden =
      std::max<int64_t>(32, (hidden_size + 31) / 32 * 32);
  CHECK(has_aclshmem_moe_quant_int8_specialization(local_tokens, hidden_size));

  const torch::Device device = input.device();
  check_quant_tensor(
      input, torch::kBFloat16, {local_tokens, hidden_size}, "input", device);
  check_quant_tensor(payload,
                     torch::kInt8,
                     {local_tokens, physical_hidden},
                     "payload",
                     device);
  check_quant_tensor(scale, torch::kFloat32, {local_tokens}, "scale", device);

  const auto* entry = find_aclshmem_moe_quant_int8_kernel_entry(
      make_quant_specialization(local_tokens, hidden_size));
  CHECK(entry != nullptr);
  const int32_t device_id = input.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  entry->fn(reinterpret_cast<uint8_t*>(const_cast<void*>(input.data_ptr())),
            reinterpret_cast<uint8_t*>(payload.data_ptr()),
            reinterpret_cast<uint8_t*>(scale.data_ptr()),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
