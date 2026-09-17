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

#include "communication_domain.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace xllm {

CommunicationDomain::CommunicationDomain(
    std::string name,
    int32_t global_rank,
    int32_t world_size,
    std::vector<std::vector<int32_t>> rank_groups)
    : name_(std::move(name)), rank_groups_(std::move(rank_groups)) {
  if (name_.empty() || global_rank < 0 || global_rank >= world_size ||
      world_size <= 0 || rank_groups_.empty()) {
    throw std::invalid_argument("communication domain metadata is invalid");
  }

  std::vector<int32_t> rank_occurrences(static_cast<size_t>(world_size), 0);
  const size_t expected_group_size = rank_groups_.front().size();
  if (expected_group_size == 0) {
    throw std::invalid_argument("communication domain group must not be empty");
  }
  for (size_t group_index = 0; group_index < rank_groups_.size();
       ++group_index) {
    const auto& group = rank_groups_[group_index];
    if (group.size() != expected_group_size) {
      throw std::invalid_argument(
          "communication domain groups must have equal sizes");
    }
    std::unordered_set<int32_t> group_members;
    for (size_t local_rank = 0; local_rank < group.size(); ++local_rank) {
      const int32_t rank = group[local_rank];
      if (rank < 0 || rank >= world_size ||
          !group_members.insert(rank).second) {
        throw std::invalid_argument(
            "communication domain contains an invalid or duplicate rank");
      }
      ++rank_occurrences[static_cast<size_t>(rank)];
      if (rank == global_rank) {
        if (group_id_ != -1) {
          throw std::invalid_argument(
              "global rank belongs to multiple communication groups");
        }
        group_id_ = static_cast<int32_t>(group_index);
        local_rank_ = static_cast<int32_t>(local_rank);
        ranks_ = group;
      }
    }
  }
  if (group_id_ < 0 || std::any_of(
                           rank_occurrences.begin(),
                           rank_occurrences.end(),
                           [](int32_t count) { return count != 1; })) {
    throw std::invalid_argument(
        "communication domain groups must partition the world exactly once");
  }
}

void CommunicationDomain::bind(std::unique_ptr<ProcessGroup> process_group) {
  if (process_group == nullptr || process_group_ != nullptr ||
      process_group->rank() != local_rank_ ||
      process_group->world_size() != size()) {
    throw std::invalid_argument(
        "communication domain process group metadata mismatch");
  }
  process_group_ = std::move(process_group);
}

CommunicationDomainSet::CommunicationDomainSet(
    int32_t global_rank,
    int32_t world_size,
    std::vector<CommunicationDomainSpec> specs) {
  if (specs.empty()) {
    throw std::invalid_argument("communication domain set must not be empty");
  }
  order_.reserve(specs.size());
  domains_.reserve(specs.size());
  for (CommunicationDomainSpec& spec : specs) {
    if (spec.name.empty() || domains_.contains(spec.name)) {
      throw std::invalid_argument(
          "communication domain names must be non-empty and unique");
    }
    order_.push_back(spec.name);
    auto domain = std::make_unique<CommunicationDomain>(
        spec.name, global_rank, world_size, std::move(spec.rank_groups));
    domains_.emplace(spec.name, std::move(domain));
  }
}

CommunicationDomain& CommunicationDomainSet::require(std::string_view name) {
  CommunicationDomain* domain = find(name);
  if (domain == nullptr) {
    throw std::out_of_range("communication domain is not registered: " +
                            std::string(name));
  }
  return *domain;
}

const CommunicationDomain& CommunicationDomainSet::require(
    std::string_view name) const {
  const CommunicationDomain* domain = find(name);
  if (domain == nullptr) {
    throw std::out_of_range("communication domain is not registered: " +
                            std::string(name));
  }
  return *domain;
}

CommunicationDomain* CommunicationDomainSet::find(std::string_view name) {
  const auto iterator = domains_.find(std::string(name));
  return iterator == domains_.end() ? nullptr : iterator->second.get();
}

const CommunicationDomain* CommunicationDomainSet::find(
    std::string_view name) const {
  const auto iterator = domains_.find(std::string(name));
  return iterator == domains_.end() ? nullptr : iterator->second.get();
}

}  // namespace xllm
