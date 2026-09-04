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

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

#include "shmem.h"

namespace xllm {
namespace {

constexpr uint64_t kWindowAlignment = 512;

uint64_t align_window_bytes(uint64_t bytes) {
  if (bytes == 0 ||
      bytes > std::numeric_limits<uint64_t>::max() - (kWindowAlignment - 1)) {
    return 0;
  }
  return (bytes + kWindowAlignment - 1) / kWindowAlignment * kWindowAlignment;
}

bool checked_multiply(uint64_t lhs, uint64_t rhs, uint64_t* output) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    return false;
  }
  *output = lhs * rhs;
  return true;
}

}  // namespace

ShmemCommSpecValidation validate_shmem_comm_spec(const ShmemCommSpec& spec) {
  auto fail = [](const std::string& message, uint64_t required_heap_bytes = 0) {
    return ShmemCommSpecValidation{
        /*valid=*/false, required_heap_bytes, message};
  };
  if (spec.world_size <= 1 || spec.rank < 0 || spec.rank >= spec.world_size) {
    return fail("invalid SHMEM rank/world size");
  }
  if (spec.device_index < 0) {
    return fail("invalid SHMEM device index");
  }
  if (spec.local_heap_bytes == 0) {
    return fail("SHMEM local heap must be positive");
  }
  if (spec.ip_port.empty() || spec.ip_port.size() >= ACLSHMEM_MAX_IP_PORT_LEN) {
    return fail("SHMEM ip_port is empty or too long");
  }
  if (spec.windows.empty()) {
    return fail("SHMEM window list is empty");
  }
  uint64_t required_bytes = 0;
  std::unordered_map<std::string, bool> names;
  for (const auto& window : spec.windows) {
    const uint64_t aligned_bytes = align_window_bytes(window.bytes);
    if (window.name.empty() || aligned_bytes == 0) {
      return fail("SHMEM window name/size is invalid");
    }
    if (!names.emplace(window.name, true).second) {
      return fail("SHMEM window names must be unique");
    }
    if (required_bytes > std::numeric_limits<uint64_t>::max() - aligned_bytes) {
      return fail("SHMEM window bytes overflow");
    }
    required_bytes += aligned_bytes;
  }
  if (required_bytes > spec.local_heap_bytes) {
    return fail("SHMEM windows exceed the configured local heap",
                required_bytes);
  }
  return {/*valid=*/true, required_bytes, {}};
}

AclShmemMoeWindowLayout build_aclshmem_moe_window_layout(
    const AclShmemMoeWindowLayoutRequest& request) {
  auto fail = [](const std::string& error) {
    AclShmemMoeWindowLayout layout;
    layout.error = error;
    return layout;
  };
  if (request.local_tokens <= 0 || request.hidden_size <= 0 ||
      request.topk <= 0 || request.ep_world_size <= 1 ||
      request.local_experts <= 0 || request.topk > 8) {
    return fail("invalid ACLSHMEM MoE window shape");
  }
  uint64_t global_experts = 0;
  if (!checked_multiply(static_cast<uint64_t>(request.ep_world_size),
                        static_cast<uint64_t>(request.local_experts),
                        &global_experts) ||
      static_cast<uint64_t>(request.topk) > global_experts) {
    return fail("invalid ACLSHMEM MoE expert/top-k geometry");
  }
  uint64_t max_capacity = 0;
  if (!checked_multiply(global_experts,
                        static_cast<uint64_t>(request.local_tokens),
                        &max_capacity) ||
      max_capacity >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return fail("ACLSHMEM MoE capacity overflows int64");
  }

  const int64_t payload_alignment = request.int8_dispatch ? 32 : 16;
  if (request.hidden_size >
      std::numeric_limits<int64_t>::max() - (payload_alignment - 1)) {
    return fail("ACLSHMEM MoE hidden alignment overflows int64");
  }
  const int64_t physical_hidden =
      (request.hidden_size + payload_alignment - 1) / payload_alignment *
      payload_alignment;
  const int64_t combine_hidden = (request.hidden_size + 15) / 16 * 16;
  uint64_t combine_rows = 0;
  if (!checked_multiply(static_cast<uint64_t>(request.local_tokens),
                        static_cast<uint64_t>(request.topk),
                        &combine_rows)) {
    return fail("ACLSHMEM MoE combine rows overflow uint64");
  }

  AclShmemMoeWindowLayout layout;
  layout.max_capacity = static_cast<int64_t>(max_capacity);
  layout.physical_hidden = physical_hidden;
  auto append_window = [&](const char* name,
                           uint64_t first,
                           uint64_t second,
                           uint64_t element_bytes) {
    uint64_t elements = 0;
    uint64_t bytes = 0;
    if (!checked_multiply(first, second, &elements) ||
        !checked_multiply(elements, element_bytes, &bytes)) {
      layout.error = std::string("ACLSHMEM MoE window size overflow: ") + name;
      return false;
    }
    layout.windows.push_back({name, bytes});
    return true;
  };

  const uint64_t payload_element_bytes = request.int8_dispatch ? 1 : 2;
  if (!append_window("dispatch_payload",
                     max_capacity,
                     static_cast<uint64_t>(physical_hidden),
                     payload_element_bytes) ||
      (request.int8_dispatch &&
       !append_window("dispatch_scale", max_capacity, 8, 4)) ||
      !append_window("dispatch_triplet", max_capacity, 8, 4) ||
      !append_window("dispatch_status", global_experts, 8, 4) ||
      !append_window("dispatch_credit", global_experts, 8, 4) ||
      !append_window("combine_payload",
                     combine_rows,
                     static_cast<uint64_t>(combine_hidden),
                     2) ||
      !append_window("combine_status", combine_rows, 8, 4) ||
      !append_window("combine_credit",
                     static_cast<uint64_t>(request.ep_world_size),
                     combine_rows * 8,
                     4)) {
    return layout;
  }
  layout.valid = true;
  return layout;
}

bool build_shmem_comm_spec_from_environment(
    int32_t rank,
    int32_t world_size,
    int32_t device_index,
    const AclShmemMoeWindowLayout& layout,
    ShmemCommSpec* spec,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };
  if (error != nullptr) {
    error->clear();
  }
  if (spec == nullptr) {
    return fail("SHMEM output spec is null");
  }
  if (!layout.valid) {
    return fail("SHMEM MoE window layout is invalid: " + layout.error);
  }
  const char* ip_port = std::getenv("XLLM_ACLSHMEM_IP_PORT");
  if (ip_port == nullptr || ip_port[0] == '\0') {
    return fail("XLLM_ACLSHMEM_IP_PORT is not set");
  }
  const char* heap_bytes = std::getenv("XLLM_ACLSHMEM_HEAP_BYTES");
  if (heap_bytes == nullptr || heap_bytes[0] == '\0') {
    return fail("XLLM_ACLSHMEM_HEAP_BYTES is not set");
  }
  uint64_t parsed_heap_bytes = 0;
  const char* heap_end = heap_bytes + std::strlen(heap_bytes);
  const auto parse_result =
      std::from_chars(heap_bytes, heap_end, parsed_heap_bytes);
  if (parse_result.ec != std::errc() || parse_result.ptr != heap_end ||
      parsed_heap_bytes == 0) {
    return fail("XLLM_ACLSHMEM_HEAP_BYTES must be a positive uint64");
  }

  ShmemCommSpec candidate{/*rank=*/rank,
                          /*world_size=*/world_size,
                          /*device_index=*/device_index,
                          /*local_heap_bytes=*/parsed_heap_bytes,
                          /*ip_port=*/ip_port,
                          /*windows=*/layout.windows};
  const auto validation = validate_shmem_comm_spec(candidate);
  if (!validation.valid) {
    return fail(validation.error);
  }
  *spec = std::move(candidate);
  return true;
}

class ShmemCommResource::Impl final {
 public:
  struct Window {
    void* address = nullptr;
    uint64_t logical_bytes = 0;
  };

  Impl(ShmemCommSpec spec, bool owns_runtime)
      : spec_(std::move(spec)), owns_runtime_(owns_runtime) {}

  ~Impl() {
    if (!allocation_order_.empty()) {
      aclrtSetDevice(spec_.device_index);
      aclrtSynchronizeDevice();
    }
    for (auto iter = allocation_order_.rbegin();
         iter != allocation_order_.rend();
         ++iter) {
      auto window_iter = windows_.find(*iter);
      if (window_iter != windows_.end() &&
          window_iter->second.address != nullptr) {
        aclshmem_free(window_iter->second.address);
      }
    }
    windows_.clear();
    if (owns_runtime_) {
      aclshmem_finalize();
    }
  }

  bool allocate(std::string* error) {
    for (const auto& spec : spec_.windows) {
      const uint64_t aligned_bytes = align_window_bytes(spec.bytes);
      void* address = aclshmem_calloc(1, static_cast<size_t>(aligned_bytes));
      if (address == nullptr) {
        if (error != nullptr) {
          *error = "aclshmem_calloc failed for window '" + spec.name + "'";
        }
        return false;
      }
      windows_.emplace(spec.name, Window{address, spec.bytes});
      allocation_order_.push_back(spec.name);
    }
    return true;
  }

  void* window(const std::string& name) const {
    const auto iter = windows_.find(name);
    return iter == windows_.end() ? nullptr : iter->second.address;
  }

  uint64_t window_bytes(const std::string& name) const {
    const auto iter = windows_.find(name);
    return iter == windows_.end() ? 0 : iter->second.logical_bytes;
  }

  const ShmemCommSpec& spec() const { return spec_; }
  bool owns_runtime() const { return owns_runtime_; }

  int32_t reserve_generation() {
    const int64_t generation =
        next_generation_.fetch_add(1, std::memory_order_relaxed);
    if (generation > std::numeric_limits<int32_t>::max()) {
      throw std::overflow_error("ACLSHMEM generation exhausted int32 range");
    }
    return static_cast<int32_t>(generation);
  }

 private:
  ShmemCommSpec spec_;
  bool owns_runtime_ = false;
  std::unordered_map<std::string, Window> windows_;
  std::vector<std::string> allocation_order_;
  std::atomic<int64_t> next_generation_{1};
};

ShmemCommResource::ShmemCommResource(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

ShmemCommResource::~ShmemCommResource() = default;

std::unique_ptr<ShmemCommResource> ShmemCommResource::create(
    const ShmemCommSpec& spec,
    std::string* error) {
  if (error != nullptr) {
    error->clear();
  }
  const auto validation = validate_shmem_comm_spec(spec);
  if (!validation.valid) {
    if (error != nullptr) {
      *error = validation.error;
    }
    return nullptr;
  }

  int32_t current_device = -1;
  aclError device_result = aclrtGetDevice(&current_device);
  if (device_result != ACL_ERROR_NONE || current_device != spec.device_index) {
    device_result = aclrtSetDevice(spec.device_index);
  }
  if (device_result != ACL_ERROR_NONE) {
    if (error != nullptr) {
      *error = "aclrtSetDevice failed: " +
               std::to_string(static_cast<int32_t>(device_result));
    }
    return nullptr;
  }

  bool owns_runtime = false;
  const int status = aclshmemx_init_status();
  if (status != ACLSHMEM_STATUS_IS_INITIALIZED) {
    const int32_t tls_result = aclshmemx_set_conf_store_tls(false, "", 0);
    if (tls_result != ACLSHMEM_SUCCESS) {
      if (error != nullptr) {
        *error = "aclshmemx_set_conf_store_tls failed: " +
                 std::to_string(tls_result);
      }
      return nullptr;
    }
    aclshmemx_init_attr_t attributes{};
    attributes.my_pe = spec.rank;
    attributes.n_pes = spec.world_size;
    attributes.local_mem_size = spec.local_heap_bytes;
    attributes.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_MTE;
    std::memcpy(
        attributes.ip_port, spec.ip_port.c_str(), spec.ip_port.size() + 1);
    const int init_result =
        aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    if (init_result != ACLSHMEM_SUCCESS) {
      if (error != nullptr) {
        *error = "aclshmemx_init_attr failed: " + std::to_string(init_result);
      }
      return nullptr;
    }
    owns_runtime = true;
  }

  auto impl = std::make_unique<Impl>(spec, owns_runtime);
  if (!impl->allocate(error)) {
    return nullptr;
  }
  return std::unique_ptr<ShmemCommResource>(
      new ShmemCommResource(std::move(impl)));
}

void* ShmemCommResource::window(const std::string& name) const {
  return impl_->window(name);
}

uint64_t ShmemCommResource::window_bytes(const std::string& name) const {
  return impl_->window_bytes(name);
}

bool ShmemCommResource::owns_runtime() const { return impl_->owns_runtime(); }

int32_t ShmemCommResource::rank() const { return impl_->spec().rank; }

int32_t ShmemCommResource::world_size() const {
  return impl_->spec().world_size;
}

int32_t ShmemCommResource::reserve_generation() {
  return impl_->reserve_generation();
}

void ShmemCommResource::barrier_all() const { aclshmem_barrier_all(); }

bool ShmemCommResourceSlot::same_key(const ShmemCommSpec& lhs,
                                     const ShmemCommSpec& rhs) {
  return lhs.rank == rhs.rank && lhs.world_size == rhs.world_size &&
         lhs.device_index == rhs.device_index &&
         lhs.local_heap_bytes == rhs.local_heap_bytes &&
         lhs.ip_port == rhs.ip_port && lhs.windows == rhs.windows;
}

std::shared_ptr<ShmemCommResource> ShmemCommResourceSlot::acquire(
    const ShmemCommSpec& spec,
    std::string* error) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (resource_ != nullptr && cached_spec_.has_value() &&
      same_key(cached_spec_.value(), spec)) {
    if (error != nullptr) {
      error->clear();
    }
    return resource_;
  }

  std::unique_ptr<ShmemCommResource> candidate =
      ShmemCommResource::create(spec, error);
  if (candidate == nullptr) {
    return nullptr;
  }
  cached_spec_ = spec;
  resource_ = std::shared_ptr<ShmemCommResource>(std::move(candidate));
  return resource_;
}

void ShmemCommResourceSlot::reset() {
  std::shared_ptr<ShmemCommResource> released;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    cached_spec_.reset();
    released = std::move(resource_);
  }
}

}  // namespace xllm
