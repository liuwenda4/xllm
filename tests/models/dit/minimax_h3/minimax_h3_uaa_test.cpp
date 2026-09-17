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

#include "models/dit/utils/minimax_h3_uaa.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/process_group.h"
#include "core/platform/device.h"

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#endif

namespace xllm {
namespace {

class RecordingWork final : public c10d::Work {
 public:
  explicit RecordingWork(bool* waited) : waited_(waited) {}

  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    (void)timeout;
    *waited_ = true;
    return true;
  }

 private:
  bool* waited_;
};

class RecordingUAAProcessGroup final : public ProcessGroup {
 public:
  RecordingUAAProcessGroup()
      : ProcessGroup(/*rank=*/0,
                     /*world_size=*/kMiniMaxH3UaaSize,
                     torch::Device(torch::kCPU)) {}

  void all_to_all_single(torch::Tensor output,
                         torch::Tensor input,
                         std::vector<int64_t> output_split_sizes,
                         std::vector<int64_t> input_split_sizes,
                         bool async_op,
                         c10::intrusive_ptr<c10d::Work>* async_work) override {
    EXPECT_TRUE(async_op);
    EXPECT_NE(async_work, nullptr);
    const std::vector<int64_t> expected_splits(
        kMiniMaxH3UaaSize, input.size(0) / kMiniMaxH3UaaSize);
    EXPECT_EQ(output_split_sizes, expected_splits);
    EXPECT_EQ(input_split_sizes, expected_splits);
    output.copy_(input);
    *async_work = c10::make_intrusive<RecordingWork>(&waited_);
  }

  bool waited() const { return waited_; }

 private:
  bool waited_ = false;
};

class CountingWork final : public c10d::Work {
 public:
  explicit CountingWork(int64_t* waits) : waits_(waits) {}

  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    (void)timeout;
    ++*waits_;
    return true;
  }

 private:
  int64_t* waits_;
};

class CountingUAAProcessGroup final : public ProcessGroup {
 public:
  CountingUAAProcessGroup()
      : ProcessGroup(/*rank=*/0,
                     /*world_size=*/kMiniMaxH3UaaSize,
                     torch::Device(torch::kCPU)) {}

  void all_to_all_single(torch::Tensor output,
                         torch::Tensor input,
                         std::vector<int64_t> output_split_sizes,
                         std::vector<int64_t> input_split_sizes,
                         bool async_op,
                         c10::intrusive_ptr<c10d::Work>* async_work) override {
    EXPECT_TRUE(async_op);
    EXPECT_NE(async_work, nullptr);
    EXPECT_EQ(output_split_sizes, input_split_sizes);
    output.copy_(input);
    ++launches_;
    *async_work = c10::make_intrusive<CountingWork>(&waits_);
  }

  int64_t launches() const { return launches_; }
  int64_t waits() const { return waits_; }

 private:
  int64_t launches_ = 0;
  int64_t waits_ = 0;
};

torch::Tensor make_layout_input(int64_t source,
                                int64_t local_sequence,
                                int64_t head_dim) {
  return torch::arange(local_sequence * kMiniMaxH3UaaLogicalHeads * head_dim,
                       torch::kInt64)
             .reshape(
                 {1, local_sequence, kMiniMaxH3UaaLogicalHeads, head_dim}) +
         source * 1000000;
}

TEST(MiniMaxH3UAALayoutTest, PadsOnlyPhysicalDummyHeads) {
  const torch::Tensor input = make_layout_input(/*source=*/3,
                                                /*local_sequence=*/2,
                                                /*head_dim=*/3);
  const torch::Tensor padded = minimax_h3_uaa_pad_heads(input);

  ASSERT_EQ(padded.sizes().vec(),
            (std::vector<int64_t>{1, 2, kMiniMaxH3UaaPaddedHeads, 3}));
  EXPECT_TRUE(
      torch::equal(padded.slice(2, 0, kMiniMaxH3UaaLogicalHeads), input));
  EXPECT_EQ(
      torch::count_nonzero(
          padded.slice(2, kMiniMaxH3UaaLogicalHeads, kMiniMaxH3UaaPaddedHeads))
          .item<int64_t>(),
      0);
  EXPECT_EQ(minimax_h3_uaa_real_heads_for_rank(6), 4);
  EXPECT_EQ(minimax_h3_uaa_real_heads_for_rank(7), 0);
  EXPECT_THROW(minimax_h3_uaa_real_heads_for_rank(8), std::invalid_argument);
}

TEST(MiniMaxH3UAALayoutTest, ForwardSliceAndInverseRoundTripAreExact) {
  constexpr int64_t kLocalSequence = 3;
  constexpr int64_t kHeadDim = 2;
  std::vector<torch::Tensor> original;
  std::vector<torch::Tensor> forward_send;
  for (int64_t source = 0; source < kMiniMaxH3UaaSize; ++source) {
    original.emplace_back(make_layout_input(source, kLocalSequence, kHeadDim));
    forward_send.emplace_back(
        minimax_h3_uaa_prepare_forward_send(original.back()));
  }

  std::vector<torch::Tensor> forward_output;
  for (int64_t destination = 0; destination < kMiniMaxH3UaaSize;
       ++destination) {
    std::vector<torch::Tensor> receive_chunks;
    std::vector<torch::Tensor> expected_chunks;
    for (int64_t source = 0; source < kMiniMaxH3UaaSize; ++source) {
      receive_chunks.emplace_back(forward_send[source].narrow(
          0, destination * kLocalSequence, kLocalSequence));
      if (destination == kMiniMaxH3UaaSize - 1) {
        expected_chunks.emplace_back(
            torch::zeros({1, kLocalSequence, kMiniMaxH3UaaLocalHeads, kHeadDim},
                         original[source].options()));
      } else {
        expected_chunks.emplace_back(original[source].slice(
            2,
            destination * kMiniMaxH3UaaLocalHeads,
            (destination + 1) * kMiniMaxH3UaaLocalHeads));
      }
    }
    torch::Tensor receive = torch::cat(receive_chunks, 0);
    torch::Tensor output = minimax_h3_uaa_finish_forward_receive(
        receive, /*batch=*/1, kLocalSequence, kHeadDim);
    const torch::Tensor expected = torch::cat(expected_chunks, 1);
    EXPECT_TRUE(torch::equal(output, expected))
        << "forward destination=" << destination;
    EXPECT_TRUE(minimax_h3_uaa_dummy_heads_are_zero(
        output, minimax_h3_uaa_real_heads_for_rank(destination)));
    forward_output.emplace_back(std::move(output));
  }

  std::vector<torch::Tensor> inverse_send;
  for (const torch::Tensor& output : forward_output) {
    inverse_send.emplace_back(minimax_h3_uaa_prepare_inverse_send(output));
  }
  for (int64_t destination = 0; destination < kMiniMaxH3UaaSize;
       ++destination) {
    std::vector<torch::Tensor> receive_chunks;
    for (int64_t source = 0; source < kMiniMaxH3UaaSize; ++source) {
      receive_chunks.emplace_back(inverse_send[source].narrow(
          0, destination * kLocalSequence, kLocalSequence));
    }
    const torch::Tensor restored =
        minimax_h3_uaa_finish_inverse_receive(torch::cat(receive_chunks, 0),
                                              /*batch=*/1,
                                              kLocalSequence,
                                              kHeadDim);
    EXPECT_TRUE(torch::equal(restored, original[destination]))
        << "inverse destination=" << destination;
  }
}

TEST(MiniMaxH3UAALayoutTest, RejectsInvalidShapes) {
  EXPECT_THROW(minimax_h3_uaa_pad_heads(torch::zeros({1, 2, 27, 128})),
               std::invalid_argument);
  EXPECT_THROW(
      minimax_h3_uaa_prepare_inverse_send(torch::zeros({1, 15, 4, 128})),
      std::invalid_argument);
  EXPECT_THROW(minimax_h3_uaa_finish_forward_receive(
                   torch::zeros({15, 1, 4, 128}), 1, 2, 128),
               std::invalid_argument);
  EXPECT_THROW(minimax_h3_uaa_finish_inverse_receive(
                   torch::zeros({15, 1, 4, 128}), 1, 2, 128),
               std::invalid_argument);
}

TEST(MiniMaxH3UAAContextTest, RetainsBuffersWaitsAndFinishesOnce) {
  RecordingUAAProcessGroup process_group;
  const torch::Tensor input =
      torch::randn({1, 2, kMiniMaxH3UaaLogicalHeads, kMiniMaxH3UaaHeadDim},
                   torch::TensorOptions().dtype(torch::kBFloat16));

  MiniMaxH3UAAForwardContext forward =
      minimax_h3_uaa_launch_forward(input, &process_group);
  ASSERT_TRUE(forward.send_buffer().defined());
  ASSERT_TRUE(forward.receive_buffer().defined());
  EXPECT_FALSE(process_group.waited());
  const torch::Tensor u_output = forward.finish();
  EXPECT_TRUE(process_group.waited());
  EXPECT_FALSE(forward.send_buffer().defined());
  EXPECT_FALSE(forward.receive_buffer().defined());
  EXPECT_THROW(forward.finish(), std::logic_error);

  RecordingUAAProcessGroup inverse_process_group;
  MiniMaxH3UAAInverseContext inverse =
      minimax_h3_uaa_launch_inverse(u_output, &inverse_process_group);
  ASSERT_TRUE(inverse.send_buffer().defined());
  EXPECT_FALSE(inverse_process_group.waited());
  const torch::Tensor restored = inverse.finish();
  EXPECT_TRUE(inverse_process_group.waited());
  EXPECT_EQ(restored.sizes(), input.sizes());
  EXPECT_THROW(inverse.finish(), std::logic_error);
}

TEST(MiniMaxH3UAAContextTest, AbandonedContextWaitsBeforeReleasingBuffers) {
  RecordingUAAProcessGroup process_group;
  const torch::Tensor input =
      torch::zeros({1, 2, kMiniMaxH3UaaLogicalHeads, kMiniMaxH3UaaHeadDim},
                   torch::TensorOptions().dtype(torch::kBFloat16));
  {
    MiniMaxH3UAAForwardContext forward =
        minimax_h3_uaa_launch_forward(input, &process_group);
    EXPECT_FALSE(process_group.waited());
  }
  EXPECT_TRUE(process_group.waited());

  RecordingUAAProcessGroup forward_process_group;
  const torch::Tensor u_output =
      minimax_h3_uaa_launch_forward(input, &forward_process_group).finish();
  RecordingUAAProcessGroup inverse_process_group;
  {
    MiniMaxH3UAAInverseContext inverse =
        minimax_h3_uaa_launch_inverse(u_output, &inverse_process_group);
    EXPECT_FALSE(inverse_process_group.waited());
  }
  EXPECT_TRUE(inverse_process_group.waited());
}

TEST(MiniMaxH3UAAWorkspaceTest, ReusesBuffersAndLaunchesQkvBeforeWaiting) {
  constexpr int64_t kLocalSequence = 2;
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
  MiniMaxH3UAAWorkspace workspace;
  workspace.reserve(kLocalSequence, options);
  ASSERT_EQ(workspace.capacity(), kLocalSequence);
  ASSERT_EQ(workspace.allocated_bytes(),
            (MiniMaxH3UAAWorkspace::kForwardSlots +
             MiniMaxH3UAAWorkspace::kReceiveSlots) *
                kMiniMaxH3UaaSize * kLocalSequence * kMiniMaxH3UaaLocalHeads *
                kMiniMaxH3UaaHeadDim * 2);
  const void* send_address = workspace.send(0, kLocalSequence).data_ptr();
  const void* receive_address = workspace.receive(0, kLocalSequence).data_ptr();

  const torch::Tensor input = torch::randn(
      {1, kLocalSequence, kMiniMaxH3UaaLogicalHeads, kMiniMaxH3UaaHeadDim},
      options);
  CountingUAAProcessGroup process_group;
  MiniMaxH3UAAForwardContext query =
      minimax_h3_uaa_launch_forward_into(input,
                                         &process_group,
                                         workspace.send(0, kLocalSequence),
                                         workspace.receive(0, kLocalSequence));
  MiniMaxH3UAAForwardContext key =
      minimax_h3_uaa_launch_forward_into(input,
                                         &process_group,
                                         workspace.send(1, kLocalSequence),
                                         workspace.receive(1, kLocalSequence));
  MiniMaxH3UAAForwardContext value =
      minimax_h3_uaa_launch_forward_into(input,
                                         &process_group,
                                         workspace.send(2, kLocalSequence),
                                         workspace.receive(2, kLocalSequence));
  EXPECT_EQ(process_group.launches(), 3);
  EXPECT_EQ(process_group.waits(), 0);

  const torch::Tensor query_u = query.finish();
  const torch::Tensor key_u = key.finish();
  const torch::Tensor value_u = value.finish();
  EXPECT_EQ(process_group.waits(), 3);
  EXPECT_TRUE(torch::equal(query_u, key_u));
  EXPECT_TRUE(torch::equal(query_u, value_u));

  MiniMaxH3UAAInverseContext inverse = minimax_h3_uaa_launch_inverse_into(
      query_u,
      &process_group,
      workspace.send(0, kLocalSequence),
      workspace.receive(MiniMaxH3UAAWorkspace::kInverseReceiveSlot,
                        kLocalSequence));
  const torch::Tensor restored = inverse.finish();
  EXPECT_EQ(process_group.launches(), 4);
  EXPECT_EQ(process_group.waits(), 4);
  EXPECT_TRUE(torch::equal(restored, input));

  workspace.reserve(kLocalSequence, options);
  EXPECT_EQ(workspace.send(0, kLocalSequence).data_ptr(), send_address);
  EXPECT_EQ(workspace.receive(0, kLocalSequence).data_ptr(), receive_address);
}

TEST(MiniMaxH3UAAWorkspaceTest, OutPackingMatchesAllocatingHelpers) {
  constexpr int64_t kLocalSequence = 3;
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
  const torch::Tensor input = torch::randn(
      {1, kLocalSequence, kMiniMaxH3UaaLogicalHeads, kMiniMaxH3UaaHeadDim},
      options);
  MiniMaxH3UAAWorkspace workspace;
  workspace.reserve(kLocalSequence, options);
  torch::Tensor send = workspace.send(0, kLocalSequence);
  minimax_h3_uaa_prepare_forward_send_out(input, send);
  EXPECT_TRUE(torch::equal(send, minimax_h3_uaa_prepare_forward_send(input)));

  const torch::Tensor global = torch::randn({1,
                                             kLocalSequence * kMiniMaxH3UaaSize,
                                             kMiniMaxH3UaaLocalHeads,
                                             kMiniMaxH3UaaHeadDim},
                                            options);
  minimax_h3_uaa_prepare_inverse_send_out(global, send);
  EXPECT_TRUE(torch::equal(send, minimax_h3_uaa_prepare_inverse_send(global)));
}

#if defined(USE_NPU)

struct UAAHcclEnvironment {
  int32_t rank;
  int32_t local_rank;
  int32_t world_size;
  int32_t port;
  bool production;
  int32_t hold_seconds;
};

std::optional<UAAHcclEnvironment> uaa_hccl_environment() {
  const char* enabled = std::getenv("MINIMAX_H3_UAA_HCCL");
  if (enabled == nullptr || std::string(enabled) != "1") {
    return std::nullopt;
  }
  const char* rank = std::getenv("RANK");
  const char* local_rank = std::getenv("LOCAL_RANK");
  const char* world_size = std::getenv("WORLD_SIZE");
  const char* port = std::getenv("MINIMAX_H3_HCCL_PORT");
  if (rank == nullptr || local_rank == nullptr || world_size == nullptr ||
      port == nullptr) {
    throw std::invalid_argument(
        "MiniMax-H3 UAA HCCL requires RANK, LOCAL_RANK, WORLD_SIZE, and "
        "MINIMAX_H3_HCCL_PORT");
  }
  const char* production = std::getenv("MINIMAX_H3_UAA_PRODUCTION");
  const char* hold_seconds = std::getenv("MINIMAX_H3_HOLD_SECONDS");
  return UAAHcclEnvironment{
      .rank = std::stoi(rank),
      .local_rank = std::stoi(local_rank),
      .world_size = std::stoi(world_size),
      .port = std::stoi(port),
      .production = production != nullptr && std::string(production) == "1",
      .hold_seconds = hold_seconds == nullptr ? 0 : std::stoi(hold_seconds)};
}

double relative_l2(const torch::Tensor& actual, const torch::Tensor& expected) {
  const torch::Tensor actual_fp32 = actual.to(torch::kCPU).to(torch::kFloat32);
  const torch::Tensor expected_fp32 =
      expected.to(torch::kCPU).to(torch::kFloat32);
  return (actual_fp32 - expected_fp32).norm().item<double>() /
         std::max(expected_fp32.norm().item<double>(), 1e-12);
}

std::vector<torch::Tensor> small_qkv(int64_t global_sequence) {
  torch::manual_seed(20260915);
  std::vector<torch::Tensor> result;
  for (int64_t index = 0; index < 3; ++index) {
    result.emplace_back((torch::randn({1,
                                       global_sequence,
                                       kMiniMaxH3UaaLogicalHeads,
                                       kMiniMaxH3UaaHeadDim},
                                      torch::kFloat32) *
                         0.1)
                            .to(torch::kBFloat16));
  }
  return result;
}

double max_rank_value(ProcessGroup* group,
                      double value,
                      const torch::Device& device) {
  const torch::Tensor local = torch::tensor(
      {value}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  return group->allgather_base_sync(local).max().item<double>();
}

TEST(MiniMaxH3UAAHcclTest, ForwardAttentionInverseAndProductionShape) {
  const std::optional<UAAHcclEnvironment> environment = uaa_hccl_environment();
  if (!environment.has_value()) {
    GTEST_SKIP() << "Set MINIMAX_H3_UAA_HCCL=1 under an eight-rank launcher";
  }
  ASSERT_EQ(environment->world_size, kMiniMaxH3UaaSize);
  ASSERT_GE(environment->rank, 0);
  ASSERT_LT(environment->rank, environment->world_size);
  ASSERT_GE(environment->local_rank, 0);
  ASSERT_LT(environment->local_rank, environment->world_size);

  Device rank_device(environment->local_rank);
  rank_device.set_device();
  const torch::Device device = rank_device.unwrap();
  auto u_group = create_process_group(environment->rank,
                                      environment->world_size,
                                      environment->world_size,
                                      environment->port,
                                      /*trans=*/false,
                                      "127.0.0.1",
                                      "minimax_h3_u8_uaa",
                                      device);
  ASSERT_NE(u_group, nullptr);
  torch::NoGradGuard no_grad;
  bool passed = true;

  constexpr int64_t kSmallLocalSequence = 32;
  constexpr int64_t kSmallGlobalSequence =
      kSmallLocalSequence * kMiniMaxH3UaaSize;
  const std::vector<torch::Tensor> qkv = small_qkv(kSmallGlobalSequence);
  std::vector<torch::Tensor> local_qkv;
  for (const torch::Tensor& value : qkv) {
    local_qkv.emplace_back(value
                               .narrow(1,
                                       environment->rank * kSmallLocalSequence,
                                       kSmallLocalSequence)
                               .to(device));
  }
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);

  MiniMaxH3UAAForwardContext allocating_q =
      minimax_h3_uaa_launch_forward(local_qkv[0], u_group.get());
  MiniMaxH3UAAForwardContext allocating_k =
      minimax_h3_uaa_launch_forward(local_qkv[1], u_group.get());
  MiniMaxH3UAAForwardContext allocating_v =
      minimax_h3_uaa_launch_forward(local_qkv[2], u_group.get());
  const std::vector<torch::Tensor> allocating_u_qkv = {
      allocating_q.finish(), allocating_k.finish(), allocating_v.finish()};

  MiniMaxH3UAAWorkspace small_workspace;
  small_workspace.reserve(kSmallLocalSequence, local_qkv[0].options());
  MiniMaxH3UAAForwardContext q_context = minimax_h3_uaa_launch_forward_into(
      local_qkv[0],
      u_group.get(),
      small_workspace.send(0, kSmallLocalSequence),
      small_workspace.receive(0, kSmallLocalSequence));
  MiniMaxH3UAAForwardContext k_context = minimax_h3_uaa_launch_forward_into(
      local_qkv[1],
      u_group.get(),
      small_workspace.send(1, kSmallLocalSequence),
      small_workspace.receive(1, kSmallLocalSequence));
  MiniMaxH3UAAForwardContext v_context = minimax_h3_uaa_launch_forward_into(
      local_qkv[2],
      u_group.get(),
      small_workspace.send(2, kSmallLocalSequence),
      small_workspace.receive(2, kSmallLocalSequence));
  std::vector<torch::Tensor> u_qkv = {
      q_context.finish(), k_context.finish(), v_context.finish()};
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
  const int64_t real_heads =
      minimax_h3_uaa_real_heads_for_rank(environment->rank);
  bool allocating_concurrent_exact = true;
  bool workspace_concurrent_exact = true;
  for (int64_t index = 0; index < 3; ++index) {
    torch::Tensor expected =
        torch::zeros({1,
                      kSmallGlobalSequence,
                      kMiniMaxH3UaaLocalHeads,
                      kMiniMaxH3UaaHeadDim},
                     torch::TensorOptions().dtype(torch::kBFloat16));
    if (real_heads > 0) {
      expected.slice(2, 0, real_heads)
          .copy_(qkv[index].slice(
              2,
              environment->rank * kMiniMaxH3UaaLocalHeads,
              environment->rank * kMiniMaxH3UaaLocalHeads + real_heads));
    }
    allocating_concurrent_exact =
        allocating_concurrent_exact &&
        torch::equal(allocating_u_qkv[index].to(torch::kCPU), expected);
    workspace_concurrent_exact =
        workspace_concurrent_exact &&
        torch::equal(u_qkv[index].to(torch::kCPU), expected);
    passed =
        passed && minimax_h3_uaa_dummy_heads_are_zero(u_qkv[index], real_heads);
  }
  passed = passed && allocating_concurrent_exact && workspace_concurrent_exact;

  const torch::Tensor actual_attention =
      torch::scaled_dot_product_attention(u_qkv[0].permute({0, 2, 1, 3}),
                                          u_qkv[1].permute({0, 2, 1, 3}),
                                          u_qkv[2].permute({0, 2, 1, 3}),
                                          torch::nullopt,
                                          /*dropout_p=*/0.0,
                                          /*is_causal=*/false);
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
  torch::Tensor expected_attention = torch::zeros(
      {1, kMiniMaxH3UaaLocalHeads, kSmallGlobalSequence, kMiniMaxH3UaaHeadDim},
      torch::TensorOptions().dtype(torch::kBFloat16));
  if (real_heads > 0) {
    const int64_t start = environment->rank * kMiniMaxH3UaaLocalHeads;
    expected_attention.slice(1, 0, real_heads)
        .copy_(torch::scaled_dot_product_attention(
            qkv[0].slice(2, start, start + real_heads).permute({0, 2, 1, 3}),
            qkv[1].slice(2, start, start + real_heads).permute({0, 2, 1, 3}),
            qkv[2].slice(2, start, start + real_heads).permute({0, 2, 1, 3}),
            torch::nullopt,
            /*dropout_p=*/0.0,
            /*is_causal=*/false));
  }
  const double attention_relative_l2 =
      relative_l2(actual_attention, expected_attention);
  passed = passed && std::isfinite(attention_relative_l2) &&
           attention_relative_l2 <= 0.002;
  passed = passed && minimax_h3_uaa_dummy_heads_are_zero(
                         actual_attention.permute({0, 2, 1, 3}), real_heads);

  const torch::Tensor restored =
      minimax_h3_uaa_launch_inverse_into(
          u_qkv[0],
          u_group.get(),
          small_workspace.send(0, kSmallLocalSequence),
          small_workspace.receive(MiniMaxH3UAAWorkspace::kInverseReceiveSlot,
                                  kSmallLocalSequence))
          .finish();
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
  passed = passed &&
           torch::equal(restored.to(torch::kCPU), local_qkv[0].to(torch::kCPU));

  double production_forward_ms = 0.0;
  double production_inverse_ms = 0.0;
  int64_t peak_allocated_bytes = 0;
  if (environment->production) {
    constexpr int64_t kProductionLocalSequence = 7520;
    c10_npu::NPUCachingAllocator::resetPeakStats(environment->local_rank);
    const torch::Tensor production_input =
        (torch::arange(kMiniMaxH3UaaLogicalHeads, torch::kFloat32) +
         environment->rank * 64)
            .reshape({1, 1, kMiniMaxH3UaaLogicalHeads, 1})
            .expand({1,
                     kProductionLocalSequence,
                     kMiniMaxH3UaaLogicalHeads,
                     kMiniMaxH3UaaHeadDim})
            .to(torch::kBFloat16)
            .contiguous()
            .to(device);
    ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
    MiniMaxH3UAAWorkspace production_workspace;
    production_workspace.reserve(kProductionLocalSequence,
                                 production_input.options());
    const auto forward_start = std::chrono::steady_clock::now();
    MiniMaxH3UAAForwardContext production_q =
        minimax_h3_uaa_launch_forward_into(
            production_input,
            u_group.get(),
            production_workspace.send(0, kProductionLocalSequence),
            production_workspace.receive(0, kProductionLocalSequence));
    MiniMaxH3UAAForwardContext production_k =
        minimax_h3_uaa_launch_forward_into(
            production_input,
            u_group.get(),
            production_workspace.send(1, kProductionLocalSequence),
            production_workspace.receive(1, kProductionLocalSequence));
    MiniMaxH3UAAForwardContext production_v =
        minimax_h3_uaa_launch_forward_into(
            production_input,
            u_group.get(),
            production_workspace.send(2, kProductionLocalSequence),
            production_workspace.receive(2, kProductionLocalSequence));
    const torch::Tensor production_u = production_q.finish();
    const torch::Tensor production_k_u = production_k.finish();
    const torch::Tensor production_v_u = production_v.finish();
    ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
    production_forward_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - forward_start)
            .count();
    passed = passed &&
             production_u.sizes().vec() ==
                 std::vector<int64_t>(
                     {1, 60160, kMiniMaxH3UaaLocalHeads, kMiniMaxH3UaaHeadDim});
    passed =
        passed && minimax_h3_uaa_dummy_heads_are_zero(production_u, real_heads);
    passed = passed && torch::equal(production_u, production_k_u) &&
             torch::equal(production_u, production_v_u);
    passed = passed && torch::isfinite(production_u).all().item<bool>();

    const auto inverse_start = std::chrono::steady_clock::now();
    const torch::Tensor production_restored =
        minimax_h3_uaa_launch_inverse_into(
            production_u,
            u_group.get(),
            production_workspace.send(0, kProductionLocalSequence),
            production_workspace.receive(
                MiniMaxH3UAAWorkspace::kInverseReceiveSlot,
                kProductionLocalSequence))
            .finish();
    ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
    production_inverse_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - inverse_start)
            .count();
    passed = passed && torch::equal(production_restored.to(torch::kCPU),
                                    production_input.to(torch::kCPU));
    peak_allocated_bytes =
        c10_npu::NPUCachingAllocator::getDeviceStats(environment->local_rank)
            .allocated_bytes[static_cast<size_t>(
                c10_npu::NPUCachingAllocator::StatType::AGGREGATE)]
            .peak;
  }

  torch::Tensor failure =
      torch::tensor({passed ? 0 : 1},
                    torch::TensorOptions().dtype(torch::kInt32).device(device));
  u_group->allreduce(failure);
  const int32_t failure_count = failure.item<int32_t>();
  const double max_forward_ms =
      max_rank_value(u_group.get(), production_forward_ms, device);
  const double max_inverse_ms =
      max_rank_value(u_group.get(), production_inverse_ms, device);
  const double max_peak_allocated_bytes = max_rank_value(
      u_group.get(), static_cast<double>(peak_allocated_bytes), device);

  std::cout << "C8C_UAA_RANK rank=" << environment->rank
            << " real_heads=" << real_heads
            << " allocating_concurrent_exact=" << allocating_concurrent_exact
            << " workspace_concurrent_exact=" << workspace_concurrent_exact
            << " attention_relative_l2=" << attention_relative_l2
            << " production=" << environment->production
            << " forward_ms=" << production_forward_ms
            << " inverse_ms=" << production_inverse_ms
            << " peak_allocated_bytes=" << peak_allocated_bytes
            << " local_pass=" << passed << std::endl;
  if (environment->rank == 0) {
    std::cout << "C8C_UAA_AGGREGATE failure_count=" << failure_count
              << " max_rank_forward_ms=" << max_forward_ms
              << " max_rank_inverse_ms=" << max_inverse_ms
              << " max_rank_peak_allocated_bytes="
              << static_cast<int64_t>(max_peak_allocated_bytes) << std::endl;
    if (failure_count == 0) {
      std::cout << "H3_U8_ADVANCED_UAA=PASS" << std::endl;
    }
  }
  if (environment->hold_seconds > 0) {
    std::this_thread::sleep_for(
        std::chrono::seconds(environment->hold_seconds));
  }
  EXPECT_EQ(failure_count, 0);
}

#endif

}  // namespace
}  // namespace xllm
