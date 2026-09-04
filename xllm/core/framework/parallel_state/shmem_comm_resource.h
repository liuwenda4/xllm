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
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {

struct ShmemWindowSpec {
  std::string name;
  uint64_t bytes = 0;

  bool operator==(const ShmemWindowSpec& other) const {
    return name == other.name && bytes == other.bytes;
  }
};

struct ShmemCommSpec {
  int32_t rank = -1;
  int32_t world_size = 0;
  int32_t device_index = -1;
  uint64_t local_heap_bytes = 0;
  std::string ip_port;
  std::vector<ShmemWindowSpec> windows;
};

struct ShmemCommSpecValidation {
  bool valid = false;
  uint64_t required_heap_bytes = 0;
  std::string error;
};

ShmemCommSpecValidation validate_shmem_comm_spec(const ShmemCommSpec& spec);

struct AclShmemMoeWindowLayoutRequest {
  int64_t local_tokens = 0;
  int64_t hidden_size = 0;
  int64_t topk = 0;
  int64_t ep_world_size = 0;
  int64_t local_experts = 0;
  bool int8_dispatch = false;
};

struct AclShmemMoeWindowLayout {
  bool valid = false;
  int64_t max_capacity = 0;
  int64_t physical_hidden = 0;
  std::vector<ShmemWindowSpec> windows;
  std::string error;
};

AclShmemMoeWindowLayout build_aclshmem_moe_window_layout(
    const AclShmemMoeWindowLayoutRequest& request);

bool build_shmem_comm_spec_from_environment(
    int32_t rank,
    int32_t world_size,
    int32_t device_index,
    const AclShmemMoeWindowLayout& layout,
    ShmemCommSpec* spec,
    std::string* error);

class ShmemCommResource final {
 public:
  ~ShmemCommResource();

  ShmemCommResource(const ShmemCommResource&) = delete;
  ShmemCommResource& operator=(const ShmemCommResource&) = delete;
  ShmemCommResource(ShmemCommResource&&) = delete;
  ShmemCommResource& operator=(ShmemCommResource&&) = delete;

  static std::unique_ptr<ShmemCommResource> create(const ShmemCommSpec& spec,
                                                   std::string* error);

  void* window(const std::string& name) const;
  uint64_t window_bytes(const std::string& name) const;
  bool owns_runtime() const;
  int32_t rank() const;
  int32_t world_size() const;

 private:
  class Impl;
  explicit ShmemCommResource(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

class ShmemCommResourceSlot final {
 public:
  std::shared_ptr<ShmemCommResource> acquire(const ShmemCommSpec& spec,
                                             std::string* error);
  void reset();

 private:
  static bool same_key(const ShmemCommSpec& lhs, const ShmemCommSpec& rhs);

  std::mutex mutex_;
  std::optional<ShmemCommSpec> cached_spec_;
  std::shared_ptr<ShmemCommResource> resource_;
};

}  // namespace xllm
