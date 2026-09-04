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

#include "framework/parallel_state/aclshmem_moe_capability.h"

#include <limits>

namespace xllm {
namespace {

AclShmemMoeCapability fallback(AclShmemMoeFallbackReason reason,
                               int64_t required_capacity = 0) {
  return {/*supported=*/false, reason, required_capacity};
}

bool multiply_overflows(int64_t lhs, int64_t rhs) {
  return lhs > 0 && rhs > std::numeric_limits<int64_t>::max() / lhs;
}

}  // namespace

AclShmemMoeCapability evaluate_aclshmem_moe_capability(
    const AclShmemMoeCapabilityRequest& request) {
  if (!request.enabled) {
    return fallback(AclShmemMoeFallbackReason::kDisabled);
  }
  if (!request.runtime_available) {
    return fallback(AclShmemMoeFallbackReason::kRuntimeUnavailable);
  }
  if (!request.dispatch_kernel_available) {
    return fallback(AclShmemMoeFallbackReason::kDispatchKernelUnavailable);
  }
  if (!request.combine_kernel_available) {
    return fallback(AclShmemMoeFallbackReason::kCombineKernelUnavailable);
  }
  if (request.graph_requested && !request.graph_variant_available) {
    return fallback(AclShmemMoeFallbackReason::kGraphVariantUnavailable);
  }
  if (!request.expert_backend_available) {
    return fallback(AclShmemMoeFallbackReason::kExpertBackendUnavailable);
  }
  if (request.input_dtype != AclShmemMoeInputDType::kBFloat16) {
    return fallback(AclShmemMoeFallbackReason::kUnsupportedInputDType);
  }
  if (request.local_tokens <= 0 || request.hidden_size <= 0 ||
      request.ep_world_size <= 1 || request.local_experts <= 0) {
    return fallback(AclShmemMoeFallbackReason::kInvalidShape);
  }
  if (request.topk <= 0 || request.topk > 8 ||
      multiply_overflows(request.ep_world_size, request.local_experts) ||
      request.topk > request.ep_world_size * request.local_experts) {
    return fallback(AclShmemMoeFallbackReason::kUnsupportedTopK);
  }
  if (multiply_overflows(request.ep_world_size, request.local_tokens) ||
      multiply_overflows(request.ep_world_size * request.local_tokens,
                         request.local_experts)) {
    return fallback(AclShmemMoeFallbackReason::kInvalidShape);
  }
  const int64_t required_capacity =
      request.ep_world_size * request.local_tokens * request.local_experts;
  if (request.max_capacity != required_capacity) {
    return fallback(AclShmemMoeFallbackReason::kCapacityMismatch,
                    required_capacity);
  }
  return {
      /*supported=*/true, AclShmemMoeFallbackReason::kNone, required_capacity};
}

bool aclshmem_moe_runtime_compiled() {
#if defined(XLLM_HAS_ACLSHMEM)
  return true;
#else
  return false;
#endif
}

const char* aclshmem_moe_fallback_reason_string(
    AclShmemMoeFallbackReason reason) {
  switch (reason) {
    case AclShmemMoeFallbackReason::kNone:
      return "NONE";
    case AclShmemMoeFallbackReason::kDisabled:
      return "DISABLED";
    case AclShmemMoeFallbackReason::kRuntimeUnavailable:
      return "RUNTIME_UNAVAILABLE";
    case AclShmemMoeFallbackReason::kDispatchKernelUnavailable:
      return "DISPATCH_KERNEL_UNAVAILABLE";
    case AclShmemMoeFallbackReason::kCombineKernelUnavailable:
      return "COMBINE_KERNEL_UNAVAILABLE";
    case AclShmemMoeFallbackReason::kGraphVariantUnavailable:
      return "GRAPH_VARIANT_UNAVAILABLE";
    case AclShmemMoeFallbackReason::kExpertBackendUnavailable:
      return "EXPERT_BACKEND_UNAVAILABLE";
    case AclShmemMoeFallbackReason::kUnsupportedInputDType:
      return "UNSUPPORTED_INPUT_DTYPE";
    case AclShmemMoeFallbackReason::kInvalidShape:
      return "INVALID_SHAPE";
    case AclShmemMoeFallbackReason::kUnsupportedTopK:
      return "UNSUPPORTED_TOPK";
    case AclShmemMoeFallbackReason::kCapacityMismatch:
      return "CAPACITY_MISMATCH";
  }
  return "UNKNOWN";
}

}  // namespace xllm
