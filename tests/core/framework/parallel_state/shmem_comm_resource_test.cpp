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

#include "framework/parallel_state/shmem_comm_resource.h"

#include <gtest/gtest.h>

#include <cstdlib>

namespace xllm {
namespace {

ShmemCommSpec valid_spec() {
  return {/*rank=*/0,
          /*world_size=*/2,
          /*device_index=*/0,
          /*local_heap_bytes=*/4096,
          /*ip_port=*/"tcp://127.0.0.1:8899",
          /*windows=*/{{"payload", 513}, {"status", 32}}};
}

TEST(ShmemCommResourceTest, ValidatesAndAlignsWindowBytes) {
  const auto validation = validate_shmem_comm_spec(valid_spec());
  EXPECT_TRUE(validation.valid);
  EXPECT_EQ(validation.required_heap_bytes, 1536);
  EXPECT_TRUE(validation.error.empty());
}

TEST(ShmemCommResourceTest, RejectsDuplicateWindowsAndSmallHeap) {
  auto spec = valid_spec();
  spec.windows.push_back({"payload", 32});
  auto validation = validate_shmem_comm_spec(spec);
  EXPECT_FALSE(validation.valid);
  EXPECT_EQ(validation.error, "SHMEM window names must be unique");

  spec = valid_spec();
  spec.local_heap_bytes = 1024;
  validation = validate_shmem_comm_spec(spec);
  EXPECT_FALSE(validation.valid);
  EXPECT_EQ(validation.required_heap_bytes, 1536);
}

TEST(ShmemCommResourceTest, RejectsInvalidBootstrapAndEmptyWindows) {
  auto spec = valid_spec();
  spec.rank = spec.world_size;
  EXPECT_FALSE(validate_shmem_comm_spec(spec).valid);

  spec = valid_spec();
  spec.ip_port.clear();
  EXPECT_FALSE(validate_shmem_comm_spec(spec).valid);

  spec = valid_spec();
  spec.windows.clear();
  EXPECT_FALSE(validate_shmem_comm_spec(spec).valid);
}

TEST(ShmemCommResourceTest, BuildsFixedInt8MoeWindowLayout) {
  const AclShmemMoeWindowLayoutRequest request{/*local_tokens=*/4,
                                               /*hidden_size=*/37,
                                               /*topk=*/2,
                                               /*ep_world_size=*/8,
                                               /*local_experts=*/2,
                                               /*int8_dispatch=*/true};
  const auto layout = build_aclshmem_moe_window_layout(request);
  ASSERT_TRUE(layout.valid) << layout.error;
  EXPECT_EQ(layout.max_capacity, 64);
  EXPECT_EQ(layout.physical_hidden, 64);
  ASSERT_EQ(layout.windows.size(), 8);
  EXPECT_EQ(layout.windows[0], (ShmemWindowSpec{"dispatch_payload", 4096}));
  EXPECT_EQ(layout.windows[1], (ShmemWindowSpec{"dispatch_scale", 2048}));
  EXPECT_EQ(layout.windows[2], (ShmemWindowSpec{"dispatch_triplet", 2048}));
  EXPECT_EQ(layout.windows[3], (ShmemWindowSpec{"dispatch_status", 512}));
  EXPECT_EQ(layout.windows[4], (ShmemWindowSpec{"dispatch_credit", 512}));
  EXPECT_EQ(layout.windows[5], (ShmemWindowSpec{"combine_payload", 768}));
  EXPECT_EQ(layout.windows[6], (ShmemWindowSpec{"combine_status", 256}));
  EXPECT_EQ(layout.windows[7], (ShmemWindowSpec{"combine_credit", 2048}));
}

TEST(ShmemCommResourceTest, BuildsCommSpecFromExplicitEnvironment) {
  const auto layout =
      build_aclshmem_moe_window_layout({/*local_tokens=*/4,
                                        /*hidden_size=*/37,
                                        /*topk=*/2,
                                        /*ep_world_size=*/2,
                                        /*local_experts=*/2,
                                        /*int8_dispatch=*/true});
  ASSERT_TRUE(layout.valid);

  setenv("XLLM_ACLSHMEM_IP_PORT", "tcp://127.0.0.1:8899", 1);
  setenv("XLLM_ACLSHMEM_HEAP_BYTES", "1048576", 1);
  ShmemCommSpec spec;
  std::string error;
  ASSERT_TRUE(
      build_shmem_comm_spec_from_environment(1, 2, 1, layout, &spec, &error));
  EXPECT_TRUE(error.empty());
  EXPECT_EQ(spec.rank, 1);
  EXPECT_EQ(spec.world_size, 2);
  EXPECT_EQ(spec.local_heap_bytes, 1048576);
  EXPECT_EQ(spec.windows, layout.windows);

  unsetenv("XLLM_ACLSHMEM_HEAP_BYTES");
  EXPECT_FALSE(
      build_shmem_comm_spec_from_environment(1, 2, 1, layout, &spec, &error));
  EXPECT_EQ(error, "XLLM_ACLSHMEM_HEAP_BYTES is not set");
  unsetenv("XLLM_ACLSHMEM_IP_PORT");
}

}  // namespace
}  // namespace xllm
