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

#include <gtest/gtest.h>

namespace xllm {
namespace {

AclShmemMoeCapabilityRequest supported_request() {
  AclShmemMoeCapabilityRequest request;
  request.enabled = true;
  request.runtime_available = true;
  request.dispatch_kernel_available = true;
  request.combine_kernel_available = true;
  request.graph_variant_available = true;
  request.expert_backend_available = true;
  request.input_dtype = AclShmemMoeInputDType::kBFloat16;
  request.local_tokens = 4;
  request.hidden_size = 7168;
  request.topk = 2;
  request.ep_world_size = 8;
  request.local_experts = 2;
  request.max_capacity = 64;
  return request;
}

TEST(AclShmemMoeCapabilityTest, AcceptsFixedEagerAndGraphContracts) {
  auto request = supported_request();
  auto capability = evaluate_aclshmem_moe_capability(request);
  EXPECT_TRUE(capability.supported);
  EXPECT_EQ(capability.required_capacity, 64);
  EXPECT_STREQ(aclshmem_moe_fallback_reason_string(capability.fallback_reason),
               "NONE");

  request.graph_requested = true;
  capability = evaluate_aclshmem_moe_capability(request);
  EXPECT_TRUE(capability.supported);
}

TEST(AclShmemMoeCapabilityTest, ReportsRuntimeKernelAndGraphFallbacks) {
  auto request = supported_request();
  request.runtime_available = false;
  EXPECT_EQ(evaluate_aclshmem_moe_capability(request).fallback_reason,
            AclShmemMoeFallbackReason::kRuntimeUnavailable);

  request = supported_request();
  request.dispatch_kernel_available = false;
  EXPECT_EQ(evaluate_aclshmem_moe_capability(request).fallback_reason,
            AclShmemMoeFallbackReason::kDispatchKernelUnavailable);

  request = supported_request();
  request.graph_requested = true;
  request.graph_variant_available = false;
  EXPECT_EQ(evaluate_aclshmem_moe_capability(request).fallback_reason,
            AclShmemMoeFallbackReason::kGraphVariantUnavailable);
}

TEST(AclShmemMoeCapabilityTest, RejectsDtypeTopKAndCapacityMismatch) {
  auto request = supported_request();
  request.input_dtype = AclShmemMoeInputDType::kFloat16;
  EXPECT_EQ(evaluate_aclshmem_moe_capability(request).fallback_reason,
            AclShmemMoeFallbackReason::kUnsupportedInputDType);

  request = supported_request();
  request.topk = 17;
  EXPECT_EQ(evaluate_aclshmem_moe_capability(request).fallback_reason,
            AclShmemMoeFallbackReason::kUnsupportedTopK);

  request = supported_request();
  request.max_capacity = 63;
  const auto capability = evaluate_aclshmem_moe_capability(request);
  EXPECT_EQ(capability.fallback_reason,
            AclShmemMoeFallbackReason::kCapacityMismatch);
  EXPECT_EQ(capability.required_capacity, 64);
}

}  // namespace
}  // namespace xllm
