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

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

#include "core/framework/dit_model_loader.h"
#include "core/platform/device.h"
#include "models/dit/transformers/minimax_h3_blocks.h"
#include "models/dit/transformers/minimax_h3_tp_uaa_resident.h"

namespace xllm {
namespace {

TEST(MiniMaxH3CacheBlockPlanTest, SplitsMiddleAndTailBlocks) {
  const MiniMaxH3CacheBlockPlan no_tail = minimax_h3_cache_block_plan(1, 0);
  EXPECT_EQ(no_tail.middle_start, 1);
  EXPECT_EQ(no_tail.tail_start, 50);
  EXPECT_EQ(no_tail.hit_executed_blocks, 1);
  EXPECT_EQ(no_tail.skipped_blocks, 49);

  const MiniMaxH3CacheBlockPlan tail = minimax_h3_cache_block_plan(16, 16);
  EXPECT_EQ(tail.middle_start, 16);
  EXPECT_EQ(tail.tail_start, 34);
  EXPECT_EQ(tail.hit_executed_blocks, 32);
  EXPECT_EQ(tail.skipped_blocks, 18);

  EXPECT_THROW(minimax_h3_cache_block_plan(0, 0), std::invalid_argument);
  EXPECT_THROW(minimax_h3_cache_block_plan(1, -1), std::invalid_argument);
  EXPECT_THROW(minimax_h3_cache_block_plan(25, 25), std::invalid_argument);
}

TEST(MiniMaxH3SplitQKVTest, MatchesFusedProjectionAtProductionShape) {
  const char* enabled = std::getenv("MINIMAX_H3_SPLIT_QKV_AB");
  const char* checkpoint = std::getenv("MINIMAX_H3_CHECKPOINT");
  if (enabled == nullptr || std::string(enabled) != "1" ||
      checkpoint == nullptr || std::string(checkpoint).empty()) {
    GTEST_SKIP() << "Set MINIMAX_H3_SPLIT_QKV_AB=1 and checkpoint";
  }

  Device rank_device(/*device=*/0);
  rank_device.set_device();
  const torch::Device device = rank_device.unwrap();
  ProcessGroup tp_group(/*rank=*/0, /*world_size=*/2, device);
  MiniMaxH3TPDiTBlock block(
      MiniMaxH3C4Config{},
      &tp_group,
      torch::TensorOptions().device(device).dtype(torch::kBFloat16),
      /*dense_output_projection=*/true);
  auto loader = std::make_unique<DiTModelLoader>(checkpoint);
  ASSERT_TRUE(loader->has_component("transformer"));
  auto transformer_loader = loader->take_component_loader("transformer");
  ASSERT_NE(transformer_loader, nullptr);
  block->load_source_weights(transformer_loader->get_state_dicts(),
                             /*layer=*/0);

  constexpr int64_t kRows = 7520;
  const MiniMaxH3C4Config config;
  torch::manual_seed(20260917);
  const torch::Tensor input =
      (torch::randn({kRows, config.hidden_size}, torch::kFloat32) * 0.1F)
          .to(torch::kBFloat16)
          .to(device);
  const torch::Tensor rope = torch::zeros(
      {kRows, 96},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  torch::NoGradGuard no_grad;
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);

  const auto fused_started = std::chrono::steady_clock::now();
  const MiniMaxH3TPAttentionQKV fused =
      block->attention()->project_local_qkv(input, rope);
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
  const double fused_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - fused_started)
                              .count();

  const auto split_started = std::chrono::steady_clock::now();
  const torch::Tensor query =
      block->attention()->project_local_query(input, rope);
  const torch::Tensor key = block->attention()->project_local_key(input, rope);
  const torch::Tensor value = block->attention()->project_local_value(input);
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
  const double split_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - split_started)
                              .count();

  const double query_max = (query - fused.query).abs().max().item<double>();
  const double key_max = (key - fused.key).abs().max().item<double>();
  const double value_max = (value - fused.value).abs().max().item<double>();
  std::cout << "H3_SPLIT_QKV_AB fused_ms=" << fused_ms
            << " split_ms=" << split_ms << " query_max_abs=" << query_max
            << " key_max_abs=" << key_max << " value_max_abs=" << value_max
            << std::endl;
  EXPECT_TRUE(torch::equal(query, fused.query));
  EXPECT_TRUE(torch::equal(key, fused.key));
  EXPECT_TRUE(torch::equal(value, fused.value));
  const char* hold_seconds = std::getenv("MINIMAX_H3_HOLD_SECONDS");
  if (hold_seconds != nullptr && std::stoi(hold_seconds) > 0) {
    std::this_thread::sleep_for(std::chrono::seconds(std::stoi(hold_seconds)));
  }
}

}  // namespace
}  // namespace xllm
