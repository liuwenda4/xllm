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

#include "core/framework/dit_cache/similarity_residual_cache.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace xllm {
namespace {

TEST(SimilarityResidualCacheTest, AppliesWarmupThresholdAndHitCap) {
  CacheDiTOptions options;
  options.warmup_steps = 2;
  options.residual_diff_threshold = 0.04F;
  options.max_consecutive_hits = 1;
  SimilarityResidualCacheState cache(options);

  torch::Tensor input = torch::zeros({4, 2}, torch::kBFloat16);
  torch::Tensor front = torch::ones({4, 2}, torch::kBFloat16);
  auto decision = cache.decide(0, front - input, /*used_rows=*/3);
  EXPECT_FALSE(decision.hit);
  cache.record_dense(0, front, front + 10);

  torch::Tensor front_one = front + 0.01;
  decision = cache.decide(1, front_one - input, /*used_rows=*/3);
  EXPECT_FALSE(decision.hit);
  cache.record_dense(1, front_one, front_one + 20);

  torch::Tensor front_two = front + 0.011;
  decision = cache.decide(2, front_two - input, /*used_rows=*/3);
  EXPECT_TRUE(decision.hit);
  EXPECT_LT(decision.relative_l1, options.residual_diff_threshold);
  EXPECT_TRUE(torch::equal(cache.apply(2, front_two), front_two + 20));

  torch::Tensor front_three = front + 0.012;
  decision = cache.decide(3, front_three - input, /*used_rows=*/3);
  EXPECT_FALSE(decision.hit);
  cache.record_dense(3, front_three, front_three + 30);

  torch::Tensor front_four = front * 2;
  decision = cache.decide(4, front_four - input, /*used_rows=*/3);
  EXPECT_FALSE(decision.hit);
  cache.record_dense(4, front_four, front_four + 40);

  EXPECT_EQ(cache.cache_hits(), 1);
  EXPECT_EQ(cache.dense_forwards(), 4);
  EXPECT_EQ(cache.similarity_checks(), 2);
  EXPECT_EQ(cache.consecutive_hits(), 0);
}

TEST(SimilarityResidualCacheTest, KeepsDenseProbeAcrossCacheHits) {
  CacheDiTOptions options;
  options.warmup_steps = 0;
  options.residual_diff_threshold = 0.04F;
  options.max_consecutive_hits = 2;
  SimilarityResidualCacheState cache(options);

  torch::Tensor dense_probe = torch::ones({2, 2}, torch::kFloat32);
  EXPECT_FALSE(cache.decide(0, dense_probe, /*used_rows=*/2).hit);
  cache.record_dense(0, dense_probe, dense_probe + 10);

  torch::Tensor cached_probe = dense_probe + 0.03;
  EXPECT_TRUE(cache.decide(1, cached_probe, /*used_rows=*/2).hit);

  torch::Tensor next_probe = dense_probe + 0.06;
  const SimilarityResidualCacheDecision decision =
      cache.decide(2, next_probe, /*used_rows=*/2);
  EXPECT_FALSE(decision.hit);
  EXPECT_GT(decision.relative_l1, options.residual_diff_threshold);
}

TEST(SimilarityResidualCacheTest, PredictsResidualFromDenseHistory) {
  CacheDiTOptions options;
  options.warmup_steps = 0;
  options.residual_diff_threshold = 0.3F;
  options.linear_residual_prediction = true;
  SimilarityResidualCacheState cache(options);

  const torch::Tensor first = torch::ones({2, 2}, torch::kFloat32);
  EXPECT_FALSE(cache.decide(0, first, /*used_rows=*/2).hit);
  cache.record_dense(0, first, first + 10);

  const torch::Tensor second = first * 2;
  EXPECT_FALSE(cache.decide(1, second, /*used_rows=*/2).hit);
  cache.record_dense(1, second, second + 14);

  const torch::Tensor third = second + 0.5;
  const SimilarityResidualCacheDecision decision =
      cache.decide(2, third, /*used_rows=*/2);
  EXPECT_TRUE(decision.hit);
  EXPECT_FLOAT_EQ(decision.prediction_scale, 0.5F);
  EXPECT_TRUE(torch::allclose(cache.apply(2, third), third + 16));
}

TEST(SimilarityResidualCacheTest, StopsCheckingAfterMaximumCachedSteps) {
  CacheDiTOptions options;
  options.warmup_steps = 0;
  options.max_cached_steps = 1;
  SimilarityResidualCacheState cache(options);

  torch::Tensor probe = torch::ones({2, 2}, torch::kFloat32);
  EXPECT_FALSE(cache.decide(0, probe, /*used_rows=*/2).hit);
  cache.record_dense(0, probe, probe + 10);
  EXPECT_TRUE(cache.decide(1, probe, /*used_rows=*/2).hit);
  EXPECT_FALSE(cache.decide(2, probe, /*used_rows=*/2).hit);
  cache.record_dense(2, probe, probe + 20);

  EXPECT_EQ(cache.cache_hits(), 1);
  EXPECT_EQ(cache.similarity_checks(), 1);
}

TEST(SimilarityResidualCacheTest, ExcludesPaddingRowsFromSimilarity) {
  CacheDiTOptions options;
  options.warmup_steps = 0;
  SimilarityResidualCacheState cache(options);

  torch::Tensor first = torch::ones({4, 1}, torch::kBFloat16);
  EXPECT_FALSE(cache.decide(0, first, /*used_rows=*/3).hit);
  cache.record_dense(0, first, first + 5);
  torch::Tensor second = first.clone();
  second[3] = 1000;
  const SimilarityResidualCacheDecision decision =
      cache.decide(1, second, /*used_rows=*/3);

  EXPECT_TRUE(decision.hit);
  EXPECT_FLOAT_EQ(decision.relative_l1, 0.0F);
}

TEST(SimilarityResidualCacheTest, RestrictsSimilarityToSelectedRows) {
  CacheDiTOptions options;
  options.warmup_steps = 0;
  SimilarityResidualCacheState cache(options);

  torch::Tensor first = torch::ones({4, 1}, torch::kFloat32);
  EXPECT_FALSE(cache.decide(0, first, /*used_rows=*/4).hit);
  cache.record_dense(0, first, first + 5);

  torch::Tensor second = first.clone();
  second.index_put_({1}, 100);
  second.index_put_({2}, 100);
  const torch::Tensor selected = torch::tensor({0, 3}, torch::kInt64);
  const SimilarityResidualCacheDecision decision =
      cache.decide(1,
                   second,
                   /*used_rows=*/4,
                   /*consensus_group=*/nullptr,
                   selected);

  EXPECT_TRUE(decision.hit);
  EXPECT_FLOAT_EQ(decision.relative_l1, 0.0F);
}

TEST(SimilarityResidualCacheTest, RejectsInvalidOptionsAndTensorChanges) {
  CacheDiTOptions invalid;
  invalid.residual_diff_threshold = 0.0F;
  EXPECT_THROW(
      {
        SimilarityResidualCacheState invalid_cache{invalid};
        (void)invalid_cache;
      },
      std::invalid_argument);

  invalid = CacheDiTOptions{};
  invalid.max_cached_steps = -2;
  EXPECT_THROW(
      {
        SimilarityResidualCacheState invalid_cache{invalid};
        (void)invalid_cache;
      },
      std::invalid_argument);

  invalid = CacheDiTOptions{};
  invalid.back_blocks = -1;
  EXPECT_THROW(
      {
        SimilarityResidualCacheState invalid_cache{invalid};
        (void)invalid_cache;
      },
      std::invalid_argument);

  invalid = CacheDiTOptions{};
  invalid.front_blocks = 0;
  EXPECT_THROW(
      {
        SimilarityResidualCacheState invalid_cache{invalid};
        (void)invalid_cache;
      },
      std::invalid_argument);

  SimilarityResidualCacheState cache{CacheDiTOptions{}};
  torch::Tensor first = torch::ones({2, 2}, torch::kBFloat16);
  EXPECT_FALSE(cache.decide(0, first, /*used_rows=*/2).hit);
  EXPECT_THROW(cache.decide(1,
                            torch::ones({3, 2}, torch::kBFloat16),
                            /*used_rows=*/2),
               std::invalid_argument);
  EXPECT_THROW(cache.apply(1, first), std::logic_error);
  cache.record_dense(0, first, first + 1);
  EXPECT_THROW(cache.decide(1,
                            first,
                            /*used_rows=*/2,
                            /*consensus_group=*/nullptr,
                            torch::tensor({2}, torch::kInt64)),
               std::invalid_argument);
  EXPECT_THROW(cache.decide(1,
                            first,
                            /*used_rows=*/2,
                            /*consensus_group=*/nullptr,
                            torch::empty({0}, torch::kInt64)),
               std::invalid_argument);
}

}  // namespace
}  // namespace xllm
