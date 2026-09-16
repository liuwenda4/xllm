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

#include "models/dit/transformers/minimax_h3_tp_uaa_resident.h"

#include <gtest/gtest.h>
#include <openssl/evp.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/process_group.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/platform/device.h"
#include "models/dit/pipelines/pipeline_minimax_h3.h"

namespace xllm {
namespace {

struct ResidentEnvironment {
  int32_t global_rank;
  int32_t local_rank;
  int32_t local_world_size;
  int32_t world_size;
  int32_t port;
  int32_t hold_seconds;
  std::filesystem::path checkpoint;
  std::filesystem::path golden;
};

std::optional<ResidentEnvironment> resident_environment() {
  const char* enabled = std::getenv("MINIMAX_H3_TP_UAA_RESIDENT_HCCL");
  if (enabled == nullptr || std::string(enabled) != "1") {
    return std::nullopt;
  }
  const char* rank = std::getenv("RANK");
  const char* local_rank = std::getenv("LOCAL_RANK");
  const char* local_world_size = std::getenv("LOCAL_WORLD_SIZE");
  const char* world_size = std::getenv("WORLD_SIZE");
  const char* port = std::getenv("MINIMAX_H3_HCCL_PORT");
  const char* checkpoint = std::getenv("MINIMAX_H3_CHECKPOINT");
  const char* golden = std::getenv("MINIMAX_H3_TRAJECTORY_GOLDEN");
  const char* hold = std::getenv("MINIMAX_H3_LEASE_HOLD_SECONDS");
  if (rank == nullptr || local_rank == nullptr || local_world_size == nullptr ||
      world_size == nullptr || port == nullptr || checkpoint == nullptr ||
      golden == nullptr) {
    throw std::invalid_argument(
        "MiniMax-H3 resident HCCL environment is incomplete");
  }
  return ResidentEnvironment{
      .global_rank = std::stoi(rank),
      .local_rank = std::stoi(local_rank),
      .local_world_size = std::stoi(local_world_size),
      .world_size = std::stoi(world_size),
      .port = std::stoi(port),
      .hold_seconds = hold == nullptr ? 0 : std::stoi(hold),
      .checkpoint = checkpoint,
      .golden = golden};
}

std::filesystem::path resolve_golden(std::filesystem::path path) {
  if (std::filesystem::is_directory(path)) {
    path /= "minimax_h3_trajectory_reference.safetensors";
  }
  return path;
}

std::string file_sha256(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::invalid_argument("Cannot open C8E artifact: " + path.string());
  }
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context ||
      EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("Cannot initialize C8E artifact SHA256");
  }
  std::array<char, 1 << 20> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0 &&
        EVP_DigestUpdate(context.get(), buffer.data(), count) != 1) {
      throw std::runtime_error("Cannot update C8E artifact SHA256");
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("Cannot read C8E artifact for SHA256");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1 ||
      digest_size != 32) {
    throw std::runtime_error("Cannot finalize C8E artifact SHA256");
  }
  std::ostringstream encoded;
  encoded << std::hex << std::setfill('0');
  for (unsigned int index = 0; index < digest_size; ++index) {
    encoded << std::setw(2) << static_cast<int>(digest[index]);
  }
  return encoded.str();
}

struct ProcessGroups {
  std::unique_ptr<ProcessGroup> tp;
  std::unique_ptr<ProcessGroup> u;
};

ProcessGroups create_groups(
    const ResidentEnvironment& environment,
    const MiniMaxH3TPUAAParallelCoordinates& coordinates,
    const torch::Device& device) {
  return {.tp = create_process_group(environment.global_rank,
                                     coordinates.tp_rank,
                                     coordinates.tp_group_ranks,
                                     environment.world_size,
                                     kMiniMaxH3TPSize,
                                     environment.port + coordinates.u_rank,
                                     "127.0.0.1",
                                     "minimax_h3_c8e_tp_group",
                                     device),
          .u = create_process_group(environment.global_rank,
                                    coordinates.u_rank,
                                    coordinates.u_group_ranks,
                                    environment.world_size,
                                    kMiniMaxH3UaaSize,
                                    environment.port + 8 + coordinates.tp_rank,
                                    "127.0.0.1",
                                    "minimax_h3_c8e_u_group",
                                    device)};
}

torch::Tensor pad_rows(const torch::Tensor& input, int64_t rows) {
  std::vector<int64_t> shape = input.sizes().vec();
  shape[0] = rows;
  torch::Tensor output = torch::zeros(shape, input.options());
  output.narrow(0, 0, input.size(0)).copy_(input);
  return output;
}

struct ResidentCase {
  torch::Tensor hidden;
  torch::Tensor time_embedding;
  torch::Tensor combined_indices;
  torch::Tensor rope_frequencies;
  torch::Tensor global_cu_seqlens;
  std::vector<torch::Tensor> expected_outputs;
  int64_t local_valid_rows;
};

ResidentCase load_resident_case(const ResidentEnvironment& environment,
                                int32_t u_rank,
                                const torch::Device& device) {
  const std::filesystem::path path = resolve_golden(environment.golden);
  if (file_sha256(path) !=
      "3df46cb6b656ee7d0fc63b9285b6ef077a2a63d9bb532fccbfcb02cf65ebf04f") {
    throw std::runtime_error("MiniMax-H3 C5 trajectory SHA256 mismatch");
  }
  const std::unique_ptr<StateDict> golden =
      StateDictFromSafeTensor::load(path.string());
  if (golden == nullptr) {
    throw std::runtime_error("Failed to load MiniMax-H3 C5 trajectory");
  }
  const torch::Tensor cu_seqlens = golden->get_tensor("layout.cu_seqlens");
  const int64_t global_rows =
      cu_seqlens.select(0, cu_seqlens.numel() - 1).item<int64_t>();
  if (global_rows != 64 || global_rows % kMiniMaxH3UaaSize != 0) {
    throw std::runtime_error("MiniMax-H3 C5 aligned rows mismatch");
  }
  const int64_t used_rows =
      golden->get_tensor("step_000.packed_hidden").size(0);
  const int64_t local_rows = global_rows / kMiniMaxH3UaaSize;
  const int64_t start = u_rank * local_rows;
  const int64_t local_valid_rows =
      std::max<int64_t>(0, std::min(local_rows, used_rows - start));

  ResidentCase result{
      .hidden =
          pad_rows(golden->get_tensor("step_000.packed_hidden"), global_rows)
              .narrow(0, start, local_rows)
              .to(device)
              .contiguous(),
      .time_embedding =
          golden->get_tensor("step_000.time_embedding").to(device).contiguous(),
      .combined_indices =
          pad_rows(golden->get_tensor("step_000.combined_indices"), global_rows)
              .narrow(0, start, local_rows)
              .to(device)
              .contiguous(),
      .rope_frequencies =
          pad_rows(golden->get_tensor("step_000.rope_frequencies"), global_rows)
              .narrow(0, start, local_rows)
              .to(device)
              .contiguous(),
      .global_cu_seqlens = cu_seqlens.clone(),
      .expected_outputs = {},
      .local_valid_rows = local_valid_rows};
  result.expected_outputs.reserve(kMiniMaxH3ResidentBlockCount);
  for (int64_t layer = 0; layer < kMiniMaxH3ResidentBlockCount; ++layer) {
    std::ostringstream name;
    name << "step_000.block_" << std::setw(2) << std::setfill('0') << layer
         << ".output";
    const torch::Tensor expected = golden->get_tensor(name.str());
    if (local_valid_rows == 0) {
      result.expected_outputs.emplace_back(torch::empty(
          {0, expected.size(1)}, expected.options().device(device)));
    } else {
      result.expected_outputs.emplace_back(
          expected.narrow(0, start, local_valid_rows).to(device).contiguous());
    }
  }
  if (Device(device).synchronize_default_stream() != 0) {
    throw std::runtime_error("MiniMax-H3 C8E case H2D synchronization failed");
  }
  return result;
}

struct Metrics {
  double relative_l2;
  double cosine;
};

Metrics metrics(const torch::Tensor& actual, const torch::Tensor& expected) {
  if (actual.numel() == 0 && expected.numel() == 0) {
    return {.relative_l2 = 0.0, .cosine = 1.0};
  }
  const torch::Tensor actual_fp64 = actual.to(torch::kCPU).to(torch::kFloat64);
  const torch::Tensor expected_fp64 =
      expected.to(torch::kCPU).to(torch::kFloat64);
  const double actual_norm = actual_fp64.norm().item<double>();
  const double expected_norm = expected_fp64.norm().item<double>();
  return {
      .relative_l2 = (actual_fp64 - expected_fp64).norm().item<double>() /
                     std::max(expected_norm, 1e-12),
      .cosine =
          actual_fp64.flatten().dot(expected_fp64.flatten()).item<double>() /
          std::max(actual_norm * expected_norm, 1e-12)};
}

int64_t global_max_int64(int64_t value,
                         ProcessGroup* tp_group,
                         ProcessGroup* u_group,
                         const torch::Device& device) {
  const torch::Tensor local = torch::tensor(
      {value}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  const torch::Tensor tp_max = tp_group->allgather_base_sync(local).max();
  return u_group->allgather_base_sync(tp_max.reshape({1}))
      .max()
      .item<int64_t>();
}

bool verify_groups(const MiniMaxH3TPUAAParallelCoordinates& coordinates,
                   ProcessGroup* tp_group,
                   ProcessGroup* u_group,
                   const torch::Device& device) {
  const torch::Tensor rank =
      torch::tensor({coordinates.global_rank},
                    torch::TensorOptions().dtype(torch::kInt32).device(device));
  const torch::Tensor tp_ranks =
      tp_group->allgather_base_sync(rank).to(torch::kCPU).flatten();
  const torch::Tensor u_ranks =
      u_group->allgather_base_sync(rank).to(torch::kCPU).flatten();
  return tp_group->rank() == coordinates.tp_rank &&
         u_group->rank() == coordinates.u_rank &&
         torch::equal(
             tp_ranks,
             torch::tensor(coordinates.tp_group_ranks, torch::kInt32)) &&
         torch::equal(u_ranks,
                      torch::tensor(coordinates.u_group_ranks, torch::kInt32));
}

TEST(MiniMaxH3TPUAAResidentHcclTest, LoadsAndRunsFiftyLayers) {
  const std::optional<ResidentEnvironment> environment = resident_environment();
  if (!environment.has_value()) {
    GTEST_SKIP() << "Set MINIMAX_H3_TP_UAA_RESIDENT_HCCL=1 on 16 ranks";
  }
  ASSERT_EQ(environment->world_size, kMiniMaxH3TPUAAWorldSize);
  ASSERT_EQ(environment->local_world_size, kMiniMaxH3TPUAAWorldSize);
  ASSERT_EQ(environment->local_rank, environment->global_rank);
  ASSERT_GE(environment->hold_seconds, 0);
  ASSERT_LE(environment->hold_seconds, 60);

  const MiniMaxH3TPUAAParallelCoordinates coordinates =
      minimax_h3_tp_uaa_coordinates(environment->global_rank);
  Device rank_device(environment->local_rank);
  rank_device.set_device();
  const torch::Device device = rank_device.unwrap();
  ProcessGroups groups = create_groups(*environment, coordinates, device);
  ASSERT_NE(groups.tp, nullptr);
  ASSERT_NE(groups.u, nullptr);
  ASSERT_TRUE(
      verify_groups(coordinates, groups.tp.get(), groups.u.get(), device));
  torch::NoGradGuard no_grad;

  ResidentCase resident_case =
      load_resident_case(*environment, coordinates.u_rank, device);
  Device::empty_cache(environment->local_rank);
  const int64_t total_bytes = rank_device.total_memory();
  const int64_t free_before_bytes = rank_device.free_memory();
  c10_npu::NPUCachingAllocator::resetPeakStats(environment->local_rank);

  MiniMaxH3TPUAAResidentTransformer resident(
      MiniMaxH3C4Config{},
      groups.tp.get(),
      groups.u.get(),
      torch::TensorOptions().device(device).dtype(torch::kBFloat16));
  auto loader = std::make_unique<DiTModelLoader>(environment->checkpoint);
  ASSERT_TRUE(loader->has_component("transformer"));
  auto transformer_loader = loader->take_component_loader("transformer");
  ASSERT_NE(transformer_loader, nullptr);
  resident->load_source_weights(transformer_loader->get_state_dicts());
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
  transformer_loader.reset();
  loader.reset();

  const int64_t local_resident_bytes = resident->resident_module_bytes();
  const int64_t local_free_after_bytes = rank_device.free_memory();
  const auto& load_stats =
      c10_npu::NPUCachingAllocator::getDeviceStats(environment->local_rank);
  const int64_t local_load_peak_allocated_bytes =
      load_stats
          .allocated_bytes[static_cast<size_t>(
              c10_npu::NPUCachingAllocator::StatType::AGGREGATE)]
          .peak;
  const int64_t max_resident_bytes = global_max_int64(
      local_resident_bytes, groups.tp.get(), groups.u.get(), device);
  const int64_t max_load_peak_allocated_bytes = global_max_int64(
      local_load_peak_allocated_bytes, groups.tp.get(), groups.u.get(), device);
  const int64_t min_free_after_bytes = -global_max_int64(
      -local_free_after_bytes, groups.tp.get(), groups.u.get(), device);
  const int64_t min_free_drop_bytes =
      -global_max_int64(-(free_before_bytes - local_free_after_bytes),
                        groups.tp.get(),
                        groups.u.get(),
                        device);
  constexpr int64_t kExpectedResidentBytes = 43839692800LL;
  constexpr int64_t kRequiredFreeBytes = 8LL * 1024 * 1024 * 1024;
  bool passed = resident->loaded_block_count() == 50 &&
                resident->all_tensors_on(device) &&
                local_resident_bytes == kExpectedResidentBytes &&
                max_resident_bytes == kExpectedResidentBytes &&
                min_free_drop_bytes >= kExpectedResidentBytes &&
                min_free_after_bytes >= kRequiredFreeBytes;

  std::vector<double> local_relative_l2(50, 0.0);
  std::vector<double> local_cosine(50, 1.0);
  std::vector<bool> layer_seen(50, false);
  int64_t observed_layers = 0;
  const torch::Tensor output = resident->forward(
      resident_case.hidden,
      resident_case.time_embedding,
      resident_case.combined_indices,
      resident_case.rope_frequencies,
      resident_case.global_cu_seqlens,
      [&](int64_t layer, const MiniMaxH3ResidualBranchTrace& trace) {
        if (layer < 0 || layer >= 50 ||
            layer_seen[static_cast<size_t>(layer)]) {
          passed = false;
          return;
        }
        layer_seen[static_cast<size_t>(layer)] = true;
        ++observed_layers;
        const torch::Tensor actual =
            trace.output.narrow(0, 0, resident_case.local_valid_rows);
        const Metrics value = metrics(
            actual, resident_case.expected_outputs[static_cast<size_t>(layer)]);
        local_relative_l2[static_cast<size_t>(layer)] = value.relative_l2;
        local_cosine[static_cast<size_t>(layer)] = value.cosine;
        passed = passed && torch::isfinite(trace.output).all().item<bool>();
        std::cout << "H3-C8E rank=" << environment->global_rank
                  << " layer=" << layer << " relative_l2=" << value.relative_l2
                  << " cosine=" << value.cosine << std::endl;
      });
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
  passed = passed && observed_layers == 50 &&
           std::all_of(layer_seen.begin(), layer_seen.end(), [](bool seen) {
             return seen;
           });
  const Metrics final_metrics =
      metrics(output.narrow(0, 0, resident_case.local_valid_rows),
              resident_case.expected_outputs.back());
  passed = passed && torch::isfinite(output).all().item<bool>();

  std::vector<double> teacher_relative_l2(50, 0.0);
  std::vector<double> teacher_cosine(50, 1.0);
  for (int64_t layer = 0; layer < 50; ++layer) {
    torch::Tensor teacher_input;
    if (layer == 0) {
      teacher_input = resident_case.hidden;
    } else {
      teacher_input = torch::zeros_like(resident_case.hidden);
      if (resident_case.local_valid_rows > 0) {
        teacher_input.narrow(0, 0, resident_case.local_valid_rows)
            .copy_(
                resident_case.expected_outputs[static_cast<size_t>(layer - 1)]);
      }
    }
    const MiniMaxH3ResidualBranchTrace teacher_trace =
        resident->block_at(layer)->forward(teacher_input,
                                           resident_case.time_embedding,
                                           resident_case.combined_indices,
                                           resident_case.rope_frequencies,
                                           resident_case.global_cu_seqlens);
    const Metrics teacher_value = metrics(
        teacher_trace.output.narrow(0, 0, resident_case.local_valid_rows),
        resident_case.expected_outputs[static_cast<size_t>(layer)]);
    teacher_relative_l2[static_cast<size_t>(layer)] = teacher_value.relative_l2;
    teacher_cosine[static_cast<size_t>(layer)] = teacher_value.cosine;
    passed = passed && teacher_value.relative_l2 <= 0.005 &&
             teacher_value.cosine >= 0.99999 &&
             torch::isfinite(teacher_trace.output).all().item<bool>();
    std::cout << "H3-C8E-TEACHER rank=" << environment->global_rank
              << " layer=" << layer
              << " relative_l2=" << teacher_value.relative_l2
              << " cosine=" << teacher_value.cosine << std::endl;
  }

  const auto& forward_stats =
      c10_npu::NPUCachingAllocator::getDeviceStats(environment->local_rank);
  const int64_t local_forward_peak_allocated_bytes =
      forward_stats
          .allocated_bytes[static_cast<size_t>(
              c10_npu::NPUCachingAllocator::StatType::AGGREGATE)]
          .peak;
  const int64_t max_forward_peak_allocated_bytes =
      global_max_int64(local_forward_peak_allocated_bytes,
                       groups.tp.get(),
                       groups.u.get(),
                       device);
  passed = passed && max_forward_peak_allocated_bytes < total_bytes;

  std::vector<float> encoded_metrics;
  encoded_metrics.reserve(200);
  for (int64_t layer = 0; layer < 50; ++layer) {
    encoded_metrics.push_back(
        static_cast<float>(local_relative_l2[static_cast<size_t>(layer)]));
    encoded_metrics.push_back(
        static_cast<float>(-local_cosine[static_cast<size_t>(layer)]));
    encoded_metrics.push_back(
        static_cast<float>(teacher_relative_l2[static_cast<size_t>(layer)]));
    encoded_metrics.push_back(
        static_cast<float>(-teacher_cosine[static_cast<size_t>(layer)]));
  }
  const torch::Tensor local_metrics =
      torch::tensor(
          encoded_metrics,
          torch::TensorOptions().dtype(torch::kFloat32).device(device))
          .reshape({50, 4});
  const torch::Tensor tp_metrics =
      std::get<0>(groups.tp->allgather_base_sync(local_metrics).max(/*dim=*/0));
  const torch::Tensor global_metrics =
      std::get<0>(groups.u->allgather_base_sync(tp_metrics).max(/*dim=*/0))
          .to(torch::kCPU);
  const double max_layer_relative_l2 =
      global_metrics.select(1, 0).max().item<double>();
  const double min_layer_cosine =
      -global_metrics.select(1, 1).max().item<double>();
  const double max_teacher_relative_l2 =
      global_metrics.select(1, 2).max().item<double>();
  const double min_teacher_cosine =
      -global_metrics.select(1, 3).max().item<double>();

  const torch::Tensor tp_outputs =
      groups.tp->allgather_base_sync(output.contiguous());
  const bool cross_tp_exact =
      torch::equal(tp_outputs.select(0, 0).to(torch::kCPU),
                   tp_outputs.select(0, 1).to(torch::kCPU));
  passed =
      passed && cross_tp_exact && torch::isfinite(output).all().item<bool>();

  torch::Tensor failures =
      torch::tensor({passed ? 0 : 1},
                    torch::TensorOptions().dtype(torch::kInt32).device(device));
  groups.tp->allreduce(failures);
  groups.u->allreduce(failures);
  const int32_t global_failures = failures.item<int32_t>();

  std::cout << "C8E_RESIDENT_RANK rank=" << environment->global_rank
            << " resident_bytes=" << local_resident_bytes
            << " free_before_bytes=" << free_before_bytes
            << " free_after_bytes=" << local_free_after_bytes
            << " cross_tp_exact=" << cross_tp_exact << " local_pass=" << passed
            << std::endl;
  if (environment->global_rank == 0) {
    for (int64_t layer = 0; layer < 50; ++layer) {
      std::cout << "C8E_RESIDENT_LAYER layer=" << layer << " max_relative_l2="
                << global_metrics[layer][0].item<double>()
                << " min_cosine=" << -global_metrics[layer][1].item<double>()
                << " teacher_max_relative_l2="
                << global_metrics[layer][2].item<double>()
                << " teacher_min_cosine="
                << -global_metrics[layer][3].item<double>() << std::endl;
    }
    std::cout << "C8E_RESIDENT_AGGREGATE global_failures=" << global_failures
              << " max_layer_relative_l2=" << max_layer_relative_l2
              << " min_layer_cosine=" << min_layer_cosine
              << " max_teacher_relative_l2=" << max_teacher_relative_l2
              << " min_teacher_cosine=" << min_teacher_cosine
              << " max_resident_bytes=" << max_resident_bytes
              << " max_load_peak_allocated_bytes="
              << max_load_peak_allocated_bytes
              << " max_forward_peak_allocated_bytes="
              << max_forward_peak_allocated_bytes
              << " min_free_after_bytes=" << min_free_after_bytes
              << " min_free_drop_bytes=" << min_free_drop_bytes
              << " observed_layers=" << observed_layers << " profile_enabled=0"
              << std::endl;
    if (global_failures == 0) {
      std::cout << "H3_TP2_U8_RESIDENT_LOAD=PASS" << std::endl;
      std::cout << "H3_TP2_U8_RESIDENT_HBM=PASS" << std::endl;
      std::cout << "H3_TP2_U8_50_LAYER_FORWARD=PASS" << std::endl;
      std::cout << "H3_TP2_U8_50_LAYER_FINITE=PASS" << std::endl;
      std::cout << "H3_TP2_U8_50_LAYER_TEACHER_FORCED=PASS" << std::endl;
    }
  }
  std::this_thread::sleep_for(std::chrono::seconds(environment->hold_seconds));
  EXPECT_EQ(global_failures, 0);
}

}  // namespace
}  // namespace xllm
