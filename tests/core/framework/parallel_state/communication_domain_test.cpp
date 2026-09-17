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

#include "core/framework/parallel_state/communication_domain.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace xllm {
namespace {

std::vector<std::vector<int32_t>> tp2_groups() {
  std::vector<std::vector<int32_t>> groups;
  for (int32_t u_rank = 0; u_rank < 8; ++u_rank) {
    groups.push_back({2 * u_rank, 2 * u_rank + 1});
  }
  return groups;
}

std::vector<std::vector<int32_t>> sp8_groups() {
  std::vector<std::vector<int32_t>> groups;
  for (int32_t tp_rank = 0; tp_rank < 2; ++tp_rank) {
    std::vector<int32_t> ranks;
    for (int32_t rank = tp_rank; rank < 16; rank += 2) {
      ranks.push_back(rank);
    }
    groups.push_back(std::move(ranks));
  }
  return groups;
}

TEST(CommunicationDomainTest, SelectsStackedTpAndSpMembership) {
  CommunicationDomainSet domains(
      /*global_rank=*/7,
      /*world_size=*/16,
      {{.name = "tp", .rank_groups = tp2_groups()},
       {.name = "sp", .rank_groups = sp8_groups()},
       {.name = "sp_q", .rank_groups = sp8_groups()}});

  EXPECT_EQ(domains.order(), (std::vector<std::string>{"tp", "sp", "sp_q"}));
  const CommunicationDomain& tp = domains.require("tp");
  EXPECT_EQ(tp.group_id(), 3);
  EXPECT_EQ(tp.local_rank(), 1);
  EXPECT_EQ(tp.ranks(), (std::vector<int32_t>{6, 7}));
  const CommunicationDomain& sp = domains.require("sp");
  EXPECT_EQ(sp.group_id(), 1);
  EXPECT_EQ(sp.local_rank(), 3);
  EXPECT_EQ(sp.ranks(), (std::vector<int32_t>{1, 3, 5, 7, 9, 11, 13, 15}));
  EXPECT_EQ(domains.require("sp_q").ranks(), sp.ranks());
  EXPECT_EQ(domains.find("missing"), nullptr);
  EXPECT_THROW(domains.require("missing"), std::out_of_range);
}

TEST(CommunicationDomainTest, RejectsInvalidPartitionsAndDuplicateNames) {
  EXPECT_THROW(
      CommunicationDomainSet(/*global_rank=*/0,
                             /*world_size=*/4,
                             {{.name = "tp", .rank_groups = {{0, 1}, {2}}}}),
      std::invalid_argument);
  EXPECT_THROW(CommunicationDomainSet(
                   /*global_rank=*/0,
                   /*world_size=*/4,
                   {{.name = "tp", .rank_groups = {{0, 1}, {1, 2}}}}),
               std::invalid_argument);
  EXPECT_THROW(CommunicationDomainSet(
                   /*global_rank=*/0,
                   /*world_size=*/4,
                   {{.name = "tp", .rank_groups = {{0, 1}, {2, 3}}},
                    {.name = "tp", .rank_groups = {{0, 2}, {1, 3}}}}),
               std::invalid_argument);
}

TEST(CommunicationDomainTest, OwnsOneMatchingProcessGroup) {
  CommunicationDomainSet domains(
      /*global_rank=*/3,
      /*world_size=*/4,
      {{.name = "tp", .rank_groups = {{0, 1}, {2, 3}}}});
  CommunicationDomain& domain = domains.require("tp");
  domain.bind(std::make_unique<ProcessGroup>(
      /*rank=*/1, /*world_size=*/2, torch::Device(torch::kCPU)));

  ASSERT_NE(domain.process_group(), nullptr);
  EXPECT_EQ(domain.process_group()->rank(), 1);
  EXPECT_EQ(domain.process_group()->world_size(), 2);
  EXPECT_THROW(domain.bind(std::make_unique<ProcessGroup>(
                   1, 2, torch::Device(torch::kCPU))),
               std::invalid_argument);
}

}  // namespace
}  // namespace xllm
