/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/framework/parallel_state/process_group.h"

namespace xllm {

struct CommunicationDomainSpec {
  std::string name;
  std::vector<std::vector<int32_t>> rank_groups;
};

class CommunicationDomain final {
 public:
  CommunicationDomain(std::string name,
                      int32_t global_rank,
                      int32_t world_size,
                      std::vector<std::vector<int32_t>> rank_groups);

  CommunicationDomain(const CommunicationDomain&) = delete;
  CommunicationDomain& operator=(const CommunicationDomain&) = delete;

  const std::string& name() const { return name_; }
  int32_t group_id() const { return group_id_; }
  int32_t local_rank() const { return local_rank_; }
  int32_t size() const { return static_cast<int32_t>(ranks_.size()); }
  int32_t num_groups() const {
    return static_cast<int32_t>(rank_groups_.size());
  }
  const std::vector<int32_t>& ranks() const { return ranks_; }
  const std::vector<std::vector<int32_t>>& rank_groups() const {
    return rank_groups_;
  }

  void bind(std::unique_ptr<ProcessGroup> process_group);
  ProcessGroup* process_group() const { return process_group_.get(); }

 private:
  std::string name_;
  std::vector<std::vector<int32_t>> rank_groups_;
  std::vector<int32_t> ranks_;
  int32_t group_id_ = -1;
  int32_t local_rank_ = -1;
  std::unique_ptr<ProcessGroup> process_group_;
};

class CommunicationDomainSet final {
 public:
  CommunicationDomainSet(int32_t global_rank,
                         int32_t world_size,
                         std::vector<CommunicationDomainSpec> specs);

  CommunicationDomainSet(const CommunicationDomainSet&) = delete;
  CommunicationDomainSet& operator=(const CommunicationDomainSet&) = delete;

  CommunicationDomain& require(std::string_view name);
  const CommunicationDomain& require(std::string_view name) const;
  CommunicationDomain* find(std::string_view name);
  const CommunicationDomain* find(std::string_view name) const;
  const std::vector<std::string>& order() const { return order_; }

 private:
  std::vector<std::string> order_;
  std::unordered_map<std::string, std::unique_ptr<CommunicationDomain>>
      domains_;
};

}  // namespace xllm
