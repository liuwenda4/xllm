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

#pragma once

#include <cstdint>

namespace xllm {

enum class AclShmemMoeInputDType {
  kBFloat16,
  kFloat16,
  kUnsupported,
};

enum class AclShmemMoeFallbackReason {
  kNone,
  kDisabled,
  kRuntimeUnavailable,
  kDispatchKernelUnavailable,
  kCombineKernelUnavailable,
  kGraphVariantUnavailable,
  kExpertBackendUnavailable,
  kUnsupportedInputDType,
  kInvalidShape,
  kUnsupportedTopK,
  kCapacityMismatch,
};

struct AclShmemMoeCapabilityRequest {
  bool enabled = false;
  bool runtime_available = false;
  bool dispatch_kernel_available = false;
  bool combine_kernel_available = false;
  bool graph_requested = false;
  bool graph_variant_available = false;
  bool expert_backend_available = false;
  bool int8_dispatch = false;
  AclShmemMoeInputDType input_dtype = AclShmemMoeInputDType::kUnsupported;
  int64_t local_tokens = 0;
  int64_t hidden_size = 0;
  int64_t topk = 0;
  int64_t ep_world_size = 0;
  int64_t local_experts = 0;
  int64_t max_capacity = 0;
};

struct AclShmemMoeCapability {
  bool supported = false;
  AclShmemMoeFallbackReason fallback_reason =
      AclShmemMoeFallbackReason::kDisabled;
  int64_t required_capacity = 0;
};

AclShmemMoeCapability evaluate_aclshmem_moe_capability(
    const AclShmemMoeCapabilityRequest& request);

bool aclshmem_moe_runtime_compiled();

const char* aclshmem_moe_fallback_reason_string(
    AclShmemMoeFallbackReason reason);

}  // namespace xllm
