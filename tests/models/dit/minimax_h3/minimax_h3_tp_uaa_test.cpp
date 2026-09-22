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
#include "models/dit/schedulers/minimax_h3_scheduler.h"
#include "models/dit/transformers/minimax_h3_tp_uaa_block.h"

namespace xllm {
namespace {

TEST(MiniMaxH3TPUAAGroupMapTest, CoversOrthogonalTP2AndU8Groups) {
  std::vector<int32_t> seen(kMiniMaxH3TPUAAWorldSize, 0);
  for (int32_t global_rank = 0; global_rank < kMiniMaxH3TPUAAWorldSize;
       ++global_rank) {
    const MiniMaxH3TPUAAParallelCoordinates coordinates =
        minimax_h3_tp_uaa_coordinates(global_rank);
    EXPECT_EQ(coordinates.tp_rank, global_rank % 2);
    EXPECT_EQ(coordinates.u_rank, global_rank / 2);
    EXPECT_EQ(coordinates.tp_group_ranks,
              (std::vector<int32_t>{2 * coordinates.u_rank,
                                    2 * coordinates.u_rank + 1}));
    ASSERT_EQ(coordinates.u_group_ranks.size(), kMiniMaxH3UaaSize);
    for (int32_t rank : coordinates.u_group_ranks) {
      EXPECT_EQ(rank % 2, coordinates.tp_rank);
      ++seen[rank];
    }
  }
  EXPECT_EQ(seen, std::vector<int32_t>(kMiniMaxH3TPUAAWorldSize, 8));
  EXPECT_THROW(minimax_h3_tp_uaa_coordinates(-1), std::invalid_argument);
  EXPECT_THROW(minimax_h3_tp_uaa_coordinates(16), std::invalid_argument);
}

struct HcclEnvironment {
  int32_t global_rank;
  int32_t local_rank;
  int32_t local_world_size;
  int32_t world_size;
  int32_t port;
  std::string test_case;
  std::filesystem::path checkpoint;
  std::filesystem::path c4_golden;
  std::filesystem::path production_prepared;
};

std::optional<HcclEnvironment> hccl_environment() {
  const char* enabled = std::getenv("MINIMAX_H3_TP_UAA_HCCL");
  if (enabled == nullptr || std::string(enabled) != "1") {
    return std::nullopt;
  }
  const char* rank = std::getenv("RANK");
  const char* local_rank = std::getenv("LOCAL_RANK");
  const char* local_world_size = std::getenv("LOCAL_WORLD_SIZE");
  const char* world_size = std::getenv("WORLD_SIZE");
  const char* port = std::getenv("MINIMAX_H3_HCCL_PORT");
  const char* test_case = std::getenv("MINIMAX_H3_TP_UAA_CASE");
  const char* checkpoint = std::getenv("MINIMAX_H3_CHECKPOINT");
  const char* c4_golden = std::getenv("MINIMAX_H3_BLOCK_GOLDEN");
  const char* production = std::getenv("MINIMAX_H3_REF2VA_PRODUCTION_PREPARED");
  if (rank == nullptr || local_rank == nullptr || local_world_size == nullptr ||
      world_size == nullptr || port == nullptr || test_case == nullptr ||
      checkpoint == nullptr || c4_golden == nullptr) {
    throw std::invalid_argument(
        "MiniMax-H3 TP2 x U8 HCCL environment is incomplete");
  }
  if (std::string(test_case) != "c4" &&
      std::string(test_case) != "production") {
    throw std::invalid_argument(
        "MiniMax-H3 TP2 x U8 case must be c4 or production");
  }
  if (std::string(test_case) == "production" && production == nullptr) {
    throw std::invalid_argument(
        "MiniMax-H3 production TP2 x U8 requires the C7 prepared fixture");
  }
  return HcclEnvironment{
      .global_rank = std::stoi(rank),
      .local_rank = std::stoi(local_rank),
      .local_world_size = std::stoi(local_world_size),
      .world_size = std::stoi(world_size),
      .port = std::stoi(port),
      .test_case = test_case,
      .checkpoint = checkpoint,
      .c4_golden = c4_golden,
      .production_prepared = production == nullptr ? "" : production};
}

std::filesystem::path resolve_safetensors(std::filesystem::path path,
                                          const std::string& filename) {
  if (std::filesystem::is_directory(path)) {
    path /= filename;
  }
  return path;
}

std::string file_sha256(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::invalid_argument("Cannot open C8D artifact: " + path.string());
  }
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context ||
      EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("Cannot initialize C8D artifact SHA256");
  }
  std::array<char, 1 << 20> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0 &&
        EVP_DigestUpdate(context.get(), buffer.data(), count) != 1) {
      throw std::runtime_error("Cannot update C8D artifact SHA256");
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("Cannot read C8D artifact for SHA256");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1 ||
      digest_size != 32) {
    throw std::runtime_error("Cannot finalize C8D artifact SHA256");
  }
  std::ostringstream encoded;
  encoded << std::hex << std::setfill('0');
  for (unsigned int index = 0; index < digest_size; ++index) {
    encoded << std::setw(2) << static_cast<int>(digest[index]);
  }
  return encoded.str();
}

bool tensors_within_one_ulp(const torch::Tensor& actual,
                            const torch::Tensor& expected) {
  if (actual.sizes() != expected.sizes() ||
      actual.scalar_type() != torch::kFloat64 ||
      expected.scalar_type() != torch::kFloat64) {
    return false;
  }
  const torch::Tensor actual_values = actual.to(torch::kCPU).contiguous();
  const torch::Tensor expected_values = expected.to(torch::kCPU).contiguous();
  const double* actual_data = actual_values.data_ptr<double>();
  const double* expected_data = expected_values.data_ptr<double>();
  for (int64_t index = 0; index < actual_values.numel(); ++index) {
    const double value = actual_data[index];
    const double reference = expected_data[index];
    if (value != reference && value != std::nextafter(reference, -INFINITY) &&
        value != std::nextafter(reference, INFINITY)) {
      return false;
    }
  }
  return true;
}

struct ProcessGroups {
  std::unique_ptr<ProcessGroup> tp;
  std::unique_ptr<ProcessGroup> u;
  std::unique_ptr<ProcessGroup> q_u;
  std::unique_ptr<ProcessGroup> k_u;
  std::unique_ptr<ProcessGroup> v_u;
};

ProcessGroups create_groups(
    const HcclEnvironment& environment,
    const MiniMaxH3TPUAAParallelCoordinates& coordinates,
    const torch::Device& device) {
  ProcessGroups groups;
  groups.tp = create_process_group(environment.global_rank,
                                   coordinates.tp_rank,
                                   coordinates.tp_group_ranks,
                                   environment.world_size,
                                   kMiniMaxH3TPSize,
                                   environment.port + coordinates.u_rank,
                                   "127.0.0.1",
                                   "minimax_h3_c8d_tp_group",
                                   device);
  groups.u = create_process_group(environment.global_rank,
                                  coordinates.u_rank,
                                  coordinates.u_group_ranks,
                                  environment.world_size,
                                  kMiniMaxH3UaaSize,
                                  environment.port + 8 + coordinates.tp_rank,
                                  "127.0.0.1",
                                  "minimax_h3_c8d_u_group",
                                  device);
  groups.q_u = create_process_group(environment.global_rank,
                                    coordinates.u_rank,
                                    coordinates.u_group_ranks,
                                    environment.world_size,
                                    kMiniMaxH3UaaSize,
                                    environment.port + 10 + coordinates.tp_rank,
                                    "127.0.0.1",
                                    "minimax_h3_c8d_q_u_group",
                                    device);
  groups.k_u = create_process_group(environment.global_rank,
                                    coordinates.u_rank,
                                    coordinates.u_group_ranks,
                                    environment.world_size,
                                    kMiniMaxH3UaaSize,
                                    environment.port + 12 + coordinates.tp_rank,
                                    "127.0.0.1",
                                    "minimax_h3_c8d_k_u_group",
                                    device);
  groups.v_u = create_process_group(environment.global_rank,
                                    coordinates.u_rank,
                                    coordinates.u_group_ranks,
                                    environment.world_size,
                                    kMiniMaxH3UaaSize,
                                    environment.port + 14 + coordinates.tp_rank,
                                    "127.0.0.1",
                                    "minimax_h3_c8d_v_u_group",
                                    device);
  return groups;
}

bool verify_groups(const MiniMaxH3TPUAAParallelCoordinates& coordinates,
                   ProcessGroup* tp_group,
                   ProcessGroup* u_group,
                   const torch::Device& device) {
  if (tp_group == nullptr || u_group == nullptr) {
    return false;
  }
  const bool metadata_matches = tp_group->rank() == coordinates.tp_rank &&
                                u_group->rank() == coordinates.u_rank &&
                                tp_group->world_size() == kMiniMaxH3TPSize &&
                                u_group->world_size() == kMiniMaxH3UaaSize;
  const torch::Tensor rank_tensor =
      torch::tensor({coordinates.global_rank},
                    torch::TensorOptions().dtype(torch::kInt32).device(device));
  const torch::Tensor tp_ranks =
      tp_group->allgather_base_sync(rank_tensor).to(torch::kCPU).flatten();
  const torch::Tensor u_ranks =
      u_group->allgather_base_sync(rank_tensor).to(torch::kCPU).flatten();
  return metadata_matches &&
         torch::equal(
             tp_ranks,
             torch::tensor(coordinates.tp_group_ranks, torch::kInt32)) &&
         torch::equal(u_ranks,
                      torch::tensor(coordinates.u_group_ranks, torch::kInt32));
}

torch::Tensor pad_rows(const torch::Tensor& input, int64_t rows) {
  if (!input.defined() || input.dim() < 1 || rows < input.size(0)) {
    throw std::invalid_argument("MiniMax-H3 C8D row padding mismatch");
  }
  std::vector<int64_t> shape = input.sizes().vec();
  shape[0] = rows;
  torch::Tensor output = torch::zeros(shape, input.options());
  output.narrow(0, 0, input.size(0)).copy_(input);
  return output;
}

struct BlockCase {
  torch::Tensor hidden;
  torch::Tensor time_embedding;
  torch::Tensor combined_indices;
  torch::Tensor rope_frequencies;
  torch::Tensor global_cu_seqlens;
  torch::Tensor expected_adaln;
  torch::Tensor expected_attention_input;
  torch::Tensor expected_attention_delta;
  torch::Tensor expected_mlp_input;
  torch::Tensor expected_mlp_delta;
  torch::Tensor expected_output;
  int64_t global_sequence;
  int64_t global_used_rows;
  int64_t local_valid_rows;
};

torch::Tensor expected_local_rows(const torch::Tensor& input,
                                  int64_t start,
                                  int64_t count,
                                  const torch::Device& device) {
  if (count == 0) {
    std::vector<int64_t> shape = input.sizes().vec();
    shape[0] = 0;
    return torch::empty(shape, input.options().device(device));
  }
  return input.narrow(0, start, count).to(device).contiguous();
}

BlockCase load_c4_case(const HcclEnvironment& environment,
                       int32_t u_rank,
                       const torch::Device& device) {
  const std::filesystem::path path = resolve_safetensors(
      environment.c4_golden, "minimax_h3_block_reference.safetensors");
  if (file_sha256(path) !=
      "0c6318351ba28dc5ef1e70cb27332dffa59fb933aeb53d346010edeac48a2b14") {
    throw std::runtime_error("MiniMax-H3 C4 Golden SHA256 mismatch");
  }
  const std::unique_ptr<StateDict> golden =
      StateDictFromSafeTensor::load(path.string());
  if (golden == nullptr) {
    throw std::runtime_error("Failed to load MiniMax-H3 C4 Golden");
  }
  const torch::Tensor cu_seqlens = golden->get_tensor("layout.cu_seqlens");
  const int64_t global_sequence =
      cu_seqlens.select(0, cu_seqlens.numel() - 1).item<int64_t>();
  if (global_sequence % kMiniMaxH3UaaSize != 0) {
    throw std::runtime_error("MiniMax-H3 C4 rows do not divide U8");
  }
  const int64_t local_sequence = global_sequence / kMiniMaxH3UaaSize;
  const int64_t start = u_rank * local_sequence;
  const int64_t used_rows = golden->get_tensor("packed_hidden").size(0);
  const int64_t local_valid_rows =
      std::max<int64_t>(0, std::min(local_sequence, used_rows - start));

  const torch::Tensor hidden =
      pad_rows(golden->get_tensor("packed_hidden"), global_sequence)
          .narrow(0, start, local_sequence)
          .to(device)
          .contiguous();
  const torch::Tensor combined =
      pad_rows(golden->get_tensor("combined_indices"), global_sequence)
          .narrow(0, start, local_sequence)
          .to(device)
          .contiguous();
  const torch::Tensor rope =
      pad_rows(golden->get_tensor("rope_frequencies"), global_sequence)
          .narrow(0, start, local_sequence)
          .to(device)
          .contiguous();
  BlockCase result = {
      .hidden = hidden,
      .time_embedding =
          golden->get_tensor("time_embedding").to(device).contiguous(),
      .combined_indices = combined,
      .rope_frequencies = rope,
      .global_cu_seqlens = cu_seqlens.clone(),
      .expected_adaln =
          golden->get_tensor("block0.adaln").to(device).contiguous(),
      .expected_attention_input =
          expected_local_rows(golden->get_tensor("block0.attention_input"),
                              start,
                              local_valid_rows,
                              device),
      .expected_attention_delta =
          expected_local_rows(golden->get_tensor("block0.attention_delta"),
                              start,
                              local_valid_rows,
                              device),
      .expected_mlp_input =
          expected_local_rows(golden->get_tensor("block0.mlp_input"),
                              start,
                              local_valid_rows,
                              device),
      .expected_mlp_delta =
          expected_local_rows(golden->get_tensor("block0.mlp_delta"),
                              start,
                              local_valid_rows,
                              device),
      .expected_output = expected_local_rows(
          golden->get_tensor("block0.output"), start, local_valid_rows, device),
      .global_sequence = global_sequence,
      .global_used_rows = used_rows,
      .local_valid_rows = local_valid_rows};
  if (Device(device).synchronize_default_stream() != 0) {
    throw std::runtime_error("MiniMax-H3 C4 case H2D synchronization failed");
  }
  return result;
}

BlockCase load_production_case(const HcclEnvironment& environment,
                               int32_t u_rank,
                               const torch::Device& device) {
  const std::filesystem::path path =
      resolve_safetensors(environment.production_prepared,
                          "minimax_h3_ref2va_production_prepared.safetensors");
  if (file_sha256(path) !=
      "8bdb0ffe3a091bafb16ec09c7f176830d1797ba46d2ca48f53c708ce92277bea") {
    throw std::runtime_error(
        "MiniMax-H3 C7 production prepared SHA256 mismatch");
  }
  const std::unique_ptr<StateDict> prepared =
      StateDictFromSafeTensor::load(path.string());
  if (prepared == nullptr) {
    throw std::runtime_error("Failed to load C7 production prepared fixture");
  }

  auto loader = std::make_unique<DiTModelLoader>(environment.checkpoint);
  DiTModelContext context(
      ParallelArgs(0, 1, nullptr),
      loader->get_model_args(),
      loader->get_quant_args(),
      torch::TensorOptions().device(device).dtype(torch::kBFloat16),
      DiTCacheConfig(),
      loader->get_model_type());
  MiniMaxH3Pipeline pipeline(context);
  pipeline->load_model(std::move(loader));
  pipeline->load_c4_probe();

  const torch::Tensor condition =
      prepared->get_tensor("condition.official_hf.hidden").unsqueeze(0);
  const torch::Tensor tags =
      prepared->get_tensor("condition.text_token_tags").unsqueeze(0);
  const H3TargetLatents target = {.audio_t = 207,
                                  .audio_channels = 2,
                                  .latent_t = 37,
                                  .latent_h = 48,
                                  .latent_w = 84};
  const std::vector<H3ReferenceBlock> references = {
      {.kind = H3ReferenceBlockKind::IMAGE,
       .latent_t = 1,
       .latent_h = 128,
       .latent_w = 346}};
  const H3PackedLayout layout = minimax_h3_build_ref2va_packed_layout(
      condition, tags, target, references, /*sequence_length=*/60160);
  const MiniMaxH3DualSigmaSchedule schedule = MiniMaxH3Scheduler::build_base();
  const MiniMaxH3RowTimestepPlan plan = minimax_h3_build_row_timestep_plan(
      layout,
      schedule.video.timesteps[0].item<float>(),
      schedule.audio.timesteps[0].item<float>());
  const auto require_exact = [](const torch::Tensor& actual,
                                const torch::Tensor& expected,
                                const std::string& name) {
    if (!torch::equal(actual.to(torch::kCPU), expected.to(torch::kCPU))) {
      throw std::runtime_error("MiniMax-H3 production prepared mismatch: " +
                               name);
    }
  };
  if (layout.used_length !=
          prepared->get_tensor("layout.used_rows").item<int64_t>() ||
      layout.aligned_length !=
          prepared->get_tensor("layout.aligned_rows").item<int64_t>()) {
    throw std::runtime_error(
        "MiniMax-H3 production prepared row geometry mismatch");
  }
  if (!tensors_within_one_ulp(
          layout.position_ids.narrow(0, 0, layout.used_length),
          prepared->get_tensor("layout.position_ids"))) {
    throw std::runtime_error(
        "MiniMax-H3 production prepared mismatch: position_ids");
  }
  require_exact(layout.token_tags.narrow(0, 0, layout.used_length),
                prepared->get_tensor("layout.token_tags"),
                "token_tags");
  require_exact(layout.text_pos,
                prepared->get_tensor("layout.text_indices"),
                "text_indices");
  require_exact(layout.img_pos,
                prepared->get_tensor("layout.video_indices"),
                "video_indices");
  require_exact(layout.audio_pos,
                prepared->get_tensor("layout.audio_indices"),
                "audio_indices");
  require_exact(layout.update_mask,
                prepared->get_tensor("layout.video_update_mask"),
                "video_update_mask");
  require_exact(layout.audio_update_mask,
                prepared->get_tensor("layout.audio_update_mask"),
                "audio_update_mask");
  require_exact(plan.unique_timesteps,
                prepared->get_tensor("schedule.forward_001.unique_timesteps"),
                "forward_001.unique_timesteps");
  require_exact(plan.inverse_indices.narrow(0, 0, layout.used_length),
                prepared->get_tensor("schedule.forward_001.timestep_indices"),
                "forward_001.timestep_indices");
  const MiniMaxH3C4Trace dense =
      pipeline->probe_c4(layout,
                         prepared->get_tensor("initial.video_rows"),
                         prepared->get_tensor("initial.audio_rows"),
                         plan.unique_timesteps,
                         plan.inverse_indices);
  Device(device).synchronize_default_stream();

  if (layout.aligned_length % kMiniMaxH3UaaSize != 0) {
    throw std::runtime_error("MiniMax-H3 production rows do not divide U8");
  }
  const int64_t local_sequence = layout.aligned_length / kMiniMaxH3UaaSize;
  const int64_t start = u_rank * local_sequence;
  const int64_t local_valid_rows = std::max<int64_t>(
      0, std::min(local_sequence, layout.used_length - start));
  BlockCase result = {
      .hidden = dense.packed_hidden.narrow(0, start, local_sequence).clone(),
      .time_embedding = dense.time_embedding.clone(),
      .combined_indices =
          dense.combined_indices.narrow(0, start, local_sequence).clone(),
      .rope_frequencies =
          dense.rope_frequencies.narrow(0, start, local_sequence).clone(),
      .global_cu_seqlens = layout.cu_seqlens.clone(),
      .expected_adaln = dense.block0.adaln_parameters.clone(),
      .expected_attention_input =
          dense.block0.attention_input.narrow(0, start, local_valid_rows)
              .clone(),
      .expected_attention_delta =
          dense.block0.attention_delta.narrow(0, start, local_valid_rows)
              .clone(),
      .expected_mlp_input =
          dense.block0.mlp_input.narrow(0, start, local_valid_rows).clone(),
      .expected_mlp_delta =
          dense.block0.mlp_delta.narrow(0, start, local_valid_rows).clone(),
      .expected_output =
          dense.block0.output.narrow(0, start, local_valid_rows).clone(),
      .global_sequence = layout.aligned_length,
      .global_used_rows = layout.used_length,
      .local_valid_rows = local_valid_rows};
  if (Device(device).synchronize_default_stream() != 0) {
    throw std::runtime_error(
        "MiniMax-H3 production case materialization failed");
  }
  return result;
}

TEST(MiniMaxH3TPUAAProductionFixtureTest, MaterializesAttestedDenseOracle) {
  const char* enabled = std::getenv("MINIMAX_H3_TP_UAA_PRODUCTION_FIXTURE");
  if (enabled == nullptr || std::string(enabled) != "1") {
    GTEST_SKIP() << "Set MINIMAX_H3_TP_UAA_PRODUCTION_FIXTURE=1 on one NPU";
  }
  const char* checkpoint = std::getenv("MINIMAX_H3_CHECKPOINT");
  const char* prepared = std::getenv("MINIMAX_H3_REF2VA_PRODUCTION_PREPARED");
  ASSERT_NE(checkpoint, nullptr);
  ASSERT_NE(prepared, nullptr);
  Device device_wrapper(/*device_index=*/0);
  device_wrapper.set_device();
  const torch::Device device = device_wrapper.unwrap();
  const HcclEnvironment environment{.global_rank = 0,
                                    .local_rank = 0,
                                    .local_world_size = 1,
                                    .world_size = 1,
                                    .port = 0,
                                    .test_case = "production",
                                    .checkpoint = checkpoint,
                                    .c4_golden = "",
                                    .production_prepared = prepared};
  torch::NoGradGuard no_grad;
  const BlockCase block_case =
      load_production_case(environment, /*u_rank=*/0, device);
  EXPECT_EQ(block_case.hidden.sizes().vec(),
            (std::vector<int64_t>{7520, 5376}));
  EXPECT_EQ(block_case.rope_frequencies.sizes().vec(),
            (std::vector<int64_t>{7520, 96}));
  EXPECT_EQ(block_case.global_sequence, 60160);
  EXPECT_EQ(block_case.global_used_rows, 60132);
  EXPECT_EQ(block_case.local_valid_rows, 7520);
  EXPECT_TRUE(torch::isfinite(block_case.expected_output).all().item<bool>());
  ASSERT_EQ(device_wrapper.synchronize_default_stream(), 0);
  ASSERT_FALSE(HasFailure());
  std::cout << "H3_TP2_U8_PRODUCTION_FIXTURE=PASS" << std::endl;
}

MiniMaxH3TPUAADiTBlock load_combined_block(const HcclEnvironment& environment,
                                           ProcessGroup* tp_group,
                                           ProcessGroup* u_group,
                                           ProcessGroup* q_u_group,
                                           ProcessGroup* k_u_group,
                                           ProcessGroup* v_u_group,
                                           const torch::Device& device) {
  auto loader = std::make_unique<DiTModelLoader>(environment.checkpoint);
  if (!loader->has_component("transformer")) {
    throw std::runtime_error("MiniMax-H3 checkpoint has no transformer");
  }
  auto transformer_loader = loader->take_component_loader("transformer");
  if (transformer_loader == nullptr) {
    throw std::runtime_error("MiniMax-H3 transformer loader is unavailable");
  }
  MiniMaxH3TPUAADiTBlock block(
      MiniMaxH3C4Config{},
      tp_group,
      u_group,
      torch::TensorOptions().device(device).dtype(torch::kBFloat16),
      /*uaa_workspace=*/nullptr,
      q_u_group,
      k_u_group,
      v_u_group);
  block->load_source_weights(transformer_loader->get_state_dicts(),
                             /*layer_index=*/0);
  return block;
}

struct ComparisonMetrics {
  double relative_l2;
  double cosine;
  double max_abs;
};

ComparisonMetrics comparison_metrics(const torch::Tensor& actual,
                                     const torch::Tensor& expected) {
  if (actual.numel() == 0 && expected.numel() == 0) {
    return {.relative_l2 = 0.0, .cosine = 1.0, .max_abs = 0.0};
  }
  const torch::Tensor actual_fp64 = actual.to(torch::kCPU).to(torch::kFloat64);
  const torch::Tensor expected_fp64 =
      expected.to(torch::kCPU).to(torch::kFloat64);
  const torch::Tensor difference = actual_fp64 - expected_fp64;
  const double expected_norm = expected_fp64.norm().item<double>();
  const double actual_norm = actual_fp64.norm().item<double>();
  const double denominator = std::max(expected_norm, 1e-12);
  const double cosine_denominator =
      std::max(actual_norm * expected_norm, 1e-12);
  return {
      .relative_l2 = difference.norm().item<double>() / denominator,
      .cosine =
          actual_fp64.flatten().dot(expected_fp64.flatten()).item<double>() /
          cosine_denominator,
      .max_abs = difference.abs().max().item<double>()};
}

double global_max(double value,
                  ProcessGroup* tp_group,
                  ProcessGroup* u_group,
                  const torch::Device& device) {
  const torch::Tensor local = torch::tensor(
      {value}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  const torch::Tensor tp_max = tp_group->allgather_base_sync(local).max();
  return u_group->allgather_base_sync(tp_max.reshape({1})).max().item<double>();
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

double percentile(std::vector<double> values, double quantile) {
  std::sort(values.begin(), values.end());
  const double position = quantile * static_cast<double>(values.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return values[lower] + (values[upper] - values[lower]) * fraction;
}

void synchronize_ranks(ProcessGroup* tp_group,
                       ProcessGroup* u_group,
                       const torch::Device& device) {
  torch::Tensor value = torch::zeros(
      {1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  tp_group->allreduce(value);
  u_group->allreduce(value);
}

int64_t module_bytes(const torch::nn::Module& module) {
  int64_t bytes = 0;
  for (const auto& item : module.named_parameters(/*recurse=*/true)) {
    bytes += item.value().numel() *
             static_cast<int64_t>(item.value().element_size());
  }
  for (const auto& item : module.named_buffers(/*recurse=*/true)) {
    bytes += item.value().numel() *
             static_cast<int64_t>(item.value().element_size());
  }
  return bytes;
}

torch::Tensor diagnostic_fp32_fc2(MiniMaxH3TPMLP mlp,
                                  const torch::Tensor& input,
                                  ProcessGroup* tp_group) {
  const std::vector<torch::Tensor> up_gate =
      torch::nn::functional::linear(input, mlp->fc1_weight()).chunk(2, -1);
  const torch::Tensor activation = up_gate[0] * torch::silu(up_gate[1]);
  const torch::Tensor partial = torch::nn::functional::linear(
      activation.to(torch::kFloat32), mlp->fc2_weight().to(torch::kFloat32));
  return minimax_h3_tp_fp32_all_reduce(partial, tp_group);
}

bool verify_case_contract(const BlockCase& block_case,
                          ProcessGroup* tp_group,
                          ProcessGroup* u_group,
                          const torch::Device& device) {
  const torch::Tensor boundaries =
      block_case.global_cu_seqlens.to(torch::kInt64).to(torch::kCPU);
  const int64_t* values = boundaries.data_ptr<int64_t>();
  int64_t boundary_fingerprint = 0;
  for (int64_t index = 0; index < boundaries.numel(); ++index) {
    boundary_fingerprint += (index + 1) * values[index];
  }
  const torch::Tensor local =
      torch::tensor({block_case.hidden.size(0),
                     block_case.global_sequence,
                     boundaries.numel(),
                     boundary_fingerprint},
                    torch::TensorOptions().dtype(torch::kInt64).device(device));
  const auto group_matches = [&local](ProcessGroup* group) {
    const torch::Tensor gathered =
        group->allgather_base_sync(local).to(torch::kCPU);
    for (int64_t rank = 1; rank < gathered.size(0); ++rank) {
      if (!torch::equal(gathered.select(0, 0), gathered.select(0, rank))) {
        return false;
      }
    }
    return true;
  };
  const bool tp_matches = group_matches(tp_group);
  const bool u_matches = group_matches(u_group);
  const auto tensor_is_replicated = [](const torch::Tensor& value,
                                       ProcessGroup* group) {
    const torch::Tensor gathered =
        group->allgather_base_sync(value.contiguous()).to(torch::kCPU);
    for (int64_t rank = 1; rank < gathered.size(0); ++rank) {
      if (!torch::equal(gathered.select(0, 0), gathered.select(0, rank))) {
        return false;
      }
    }
    return true;
  };
  const bool tp_hidden = tensor_is_replicated(block_case.hidden, tp_group);
  const bool tp_indices =
      tensor_is_replicated(block_case.combined_indices, tp_group);
  const bool tp_rope =
      tensor_is_replicated(block_case.rope_frequencies, tp_group);
  const bool tp_time =
      tensor_is_replicated(block_case.time_embedding, tp_group);
  const bool u_time = tensor_is_replicated(block_case.time_embedding, u_group);
  const torch::Tensor device_boundaries =
      boundaries.to(torch::TensorOptions().device(device).dtype(torch::kInt64));
  const bool tp_boundaries = tensor_is_replicated(device_boundaries, tp_group);
  const bool u_boundaries = tensor_is_replicated(device_boundaries, u_group);
  const bool indices_in_range =
      block_case.combined_indices.min().item<int64_t>() >= 0 &&
      block_case.combined_indices.max().item<int64_t>() <
          3 * block_case.time_embedding.size(0);
  return tp_matches && u_matches && tp_hidden && tp_indices && tp_rope &&
         tp_time && u_time && tp_boundaries && u_boundaries && indices_in_range;
}

TEST(MiniMaxH3TPUAAHcclTest, MatchesDenseBlockAndProfilesProduction) {
  const std::optional<HcclEnvironment> environment = hccl_environment();
  if (!environment.has_value()) {
    GTEST_SKIP() << "Set MINIMAX_H3_TP_UAA_HCCL=1 under a 16-rank launcher";
  }
  ASSERT_EQ(environment->world_size, kMiniMaxH3TPUAAWorldSize);
  ASSERT_EQ(environment->local_world_size, kMiniMaxH3TPUAAWorldSize);
  ASSERT_EQ(environment->local_rank, environment->global_rank);
  ASSERT_GE(environment->global_rank, 0);
  ASSERT_LT(environment->global_rank, environment->world_size);
  ASSERT_GE(environment->local_rank, 0);
  ASSERT_LT(environment->local_rank, environment->world_size);

  const MiniMaxH3TPUAAParallelCoordinates coordinates =
      minimax_h3_tp_uaa_coordinates(environment->global_rank);
  Device rank_device(environment->local_rank);
  rank_device.set_device();
  const torch::Device device = rank_device.unwrap();
  ProcessGroups groups = create_groups(*environment, coordinates, device);
  ASSERT_NE(groups.tp, nullptr);
  ASSERT_NE(groups.u, nullptr);
  ASSERT_NE(groups.q_u, nullptr);
  ASSERT_NE(groups.k_u, nullptr);
  ASSERT_NE(groups.v_u, nullptr);
  bool passed =
      verify_groups(coordinates, groups.tp.get(), groups.u.get(), device);
  passed = passed && groups.q_u->rank() == coordinates.u_rank &&
           groups.k_u->rank() == coordinates.u_rank &&
           groups.v_u->rank() == coordinates.u_rank &&
           groups.q_u->world_size() == kMiniMaxH3UaaSize &&
           groups.k_u->world_size() == kMiniMaxH3UaaSize &&
           groups.v_u->world_size() == kMiniMaxH3UaaSize &&
           groups.q_u.get() != groups.k_u.get() &&
           groups.q_u.get() != groups.v_u.get() &&
           groups.k_u.get() != groups.v_u.get();

  torch::NoGradGuard no_grad;
  BlockCase block_case =
      environment->test_case == "production"
          ? load_production_case(*environment, coordinates.u_rank, device)
          : load_c4_case(*environment, coordinates.u_rank, device);
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
  const bool case_contract =
      verify_case_contract(block_case, groups.tp.get(), groups.u.get(), device);
  passed = passed && case_contract;
  torch::Tensor contract_failures =
      torch::tensor({passed ? 0 : 1},
                    torch::TensorOptions().dtype(torch::kInt32).device(device));
  groups.tp->allreduce(contract_failures);
  groups.u->allreduce(contract_failures);
  ASSERT_EQ(contract_failures.item<int32_t>(), 0)
      << "TP2 x U8 group or input contract mismatch";
  Device::empty_cache(environment->local_rank);
  MiniMaxH3TPUAADiTBlock block = load_combined_block(*environment,
                                                     groups.tp.get(),
                                                     groups.u.get(),
                                                     groups.q_u.get(),
                                                     groups.k_u.get(),
                                                     groups.v_u.get(),
                                                     device);
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);

  MiniMaxH3TPUAAAttentionDiagnostics diagnostics;
  MiniMaxH3ResidualBranchTrace actual =
      block->forward(block_case.hidden,
                     block_case.time_embedding,
                     block_case.combined_indices,
                     block_case.rope_frequencies,
                     block_case.global_cu_seqlens,
                     &diagnostics);
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);

  const int64_t real_heads =
      minimax_h3_uaa_real_heads_for_rank(coordinates.u_rank);
  passed =
      passed &&
      minimax_h3_uaa_dummy_heads_are_zero(diagnostics.query_u, real_heads) &&
      minimax_h3_uaa_dummy_heads_are_zero(diagnostics.key_u, real_heads) &&
      minimax_h3_uaa_dummy_heads_are_zero(diagnostics.value_u, real_heads) &&
      minimax_h3_uaa_dummy_heads_are_zero(diagnostics.attention_u, real_heads);
  passed = passed && torch::isfinite(actual.output).all().item<bool>();

  const auto compare = [&](const std::string& name,
                           const torch::Tensor& value,
                           const torch::Tensor& expected,
                           double relative_l2_limit,
                           double cosine_limit,
                           double max_abs_limit) {
    if (!value.defined() || !expected.defined() ||
        value.sizes() != expected.sizes() ||
        value.scalar_type() != expected.scalar_type() ||
        !torch::isfinite(value).all().item<bool>()) {
      std::cout << "H3-C8D rank=" << environment->global_rank
                << " node=" << name << " metadata_or_finite=FAIL" << std::endl;
      passed = false;
      return ComparisonMetrics{
          .relative_l2 = INFINITY, .cosine = -INFINITY, .max_abs = INFINITY};
    }
    const ComparisonMetrics metrics = comparison_metrics(value, expected);
    std::cout << "H3-C8D rank=" << environment->global_rank
              << " case=" << environment->test_case << " node=" << name
              << " relative_l2=" << metrics.relative_l2
              << " cosine=" << metrics.cosine << " max_abs=" << metrics.max_abs
              << std::endl;
    const bool node_passed = max_abs_limit == 0.0
                                 ? metrics.max_abs == 0.0
                                 : metrics.relative_l2 <= relative_l2_limit &&
                                       metrics.cosine >= cosine_limit &&
                                       metrics.max_abs <= max_abs_limit;
    passed = passed && node_passed;
    return metrics;
  };

  const ComparisonMetrics adaln_metrics = compare("adaln",
                                                  actual.adaln_parameters,
                                                  block_case.expected_adaln,
                                                  /*relative_l2_limit=*/1e-6,
                                                  /*cosine_limit=*/0.999999999,
                                                  /*max_abs_limit=*/0.0);
  const auto used_rows = [&](const torch::Tensor& value) {
    return value.narrow(0, 0, block_case.local_valid_rows);
  };
  const ComparisonMetrics attention_input_metrics =
      compare("attention_input",
              used_rows(actual.attention_input),
              block_case.expected_attention_input,
              /*relative_l2_limit=*/1e-6,
              /*cosine_limit=*/0.999999999,
              /*max_abs_limit=*/0.0);
  const ComparisonMetrics attention_metrics =
      compare("attention_delta",
              used_rows(actual.attention_delta),
              block_case.expected_attention_delta,
              /*relative_l2_limit=*/0.002,
              /*cosine_limit=*/0.999999,
              /*max_abs_limit=*/4.0);
  const ComparisonMetrics mlp_input_metrics =
      compare("mlp_input",
              used_rows(actual.mlp_input),
              block_case.expected_mlp_input,
              /*relative_l2_limit=*/0.002,
              /*cosine_limit=*/0.999999,
              /*max_abs_limit=*/2.0);
  const ComparisonMetrics mlp_metrics = compare("mlp_delta",
                                                used_rows(actual.mlp_delta),
                                                block_case.expected_mlp_delta,
                                                /*relative_l2_limit=*/0.003,
                                                /*cosine_limit=*/0.999995,
                                                /*max_abs_limit=*/8.0);
  const ComparisonMetrics output_metrics = compare("output",
                                                   used_rows(actual.output),
                                                   block_case.expected_output,
                                                   /*relative_l2_limit=*/0.003,
                                                   /*cosine_limit=*/0.999995,
                                                   /*max_abs_limit=*/8.0);

  if (environment->test_case == "production") {
    static const std::array<std::string, 3> kDiagnosticMlpNodes = {
        "diagnostic_mlp_dense_input",
        "diagnostic_mlp_fp32_fc2_actual_input",
        "diagnostic_mlp_fp32_fc2_dense_input"};
    MiniMaxH3TPMLP tp_mlp = block->tp_block()->mlp();
    if (!tp_mlp->has_sharded_fc2()) {
      for (const std::string& name : kDiagnosticMlpNodes) {
        std::cout << "H3-C8D rank=" << environment->global_rank
                  << " case=production node=" << name
                  << " status=UNAVAILABLE_SHARDED_FC2_NOT_ALLOCATED"
                  << std::endl;
      }
    } else {
      const torch::Tensor gate =
          actual.gate_mlp.narrow(0, 0, block_case.local_valid_rows);
      const torch::Tensor baseline_dense_input_delta =
          gate * tp_mlp->forward(block_case.expected_mlp_input);
      const torch::Tensor fp32_actual_input_delta =
          gate * diagnostic_fp32_fc2(
                     tp_mlp, used_rows(actual.mlp_input), groups.tp.get());
      const torch::Tensor fp32_dense_input_delta =
          gate * diagnostic_fp32_fc2(
                     tp_mlp, block_case.expected_mlp_input, groups.tp.get());
      const std::array<torch::Tensor, 3> values = {baseline_dense_input_delta,
                                                   fp32_actual_input_delta,
                                                   fp32_dense_input_delta};
      for (size_t index = 0; index < values.size(); ++index) {
        const ComparisonMetrics metrics =
            comparison_metrics(values[index], block_case.expected_mlp_delta);
        std::cout << "H3-C8D rank=" << environment->global_rank
                  << " case=production node=" << kDiagnosticMlpNodes[index]
                  << " relative_l2=" << metrics.relative_l2
                  << " cosine=" << metrics.cosine
                  << " max_abs=" << metrics.max_abs << std::endl;
      }
    }
  }

  const torch::Tensor tp_outputs =
      groups.tp->allgather_base_sync(actual.output.contiguous());
  const bool cross_tp_exact =
      torch::equal(tp_outputs.select(0, 0).to(torch::kCPU),
                   tp_outputs.select(0, 1).to(torch::kCPU));
  passed = passed && cross_tp_exact;

  const double max_attention_relative_l2 = global_max(
      attention_metrics.relative_l2, groups.tp.get(), groups.u.get(), device);
  const double max_mlp_relative_l2 = global_max(
      mlp_metrics.relative_l2, groups.tp.get(), groups.u.get(), device);
  const double max_output_relative_l2 = global_max(
      output_metrics.relative_l2, groups.tp.get(), groups.u.get(), device);
  (void)adaln_metrics;
  (void)attention_input_metrics;
  (void)mlp_input_metrics;

  std::vector<double> max_rank_iterations;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  int64_t max_peak_allocated_bytes = 0;
  int64_t max_resident_module_bytes = 0;
  const char* profile_value = std::getenv("MINIMAX_H3_TP_UAA_PROFILE");
  const bool profile_enabled =
      profile_value != nullptr && std::string(profile_value) == "1";
  if (environment->test_case == "production" && profile_enabled) {
    actual = MiniMaxH3ResidualBranchTrace{};
    diagnostics = MiniMaxH3TPUAAAttentionDiagnostics{};
    Device::empty_cache(environment->local_rank);
    block->validate_forward_inputs(block_case.hidden,
                                   block_case.time_embedding,
                                   block_case.combined_indices,
                                   block_case.rope_frequencies,
                                   block_case.global_cu_seqlens);
    for (int32_t warmup = 0; warmup < 3; ++warmup) {
      const torch::Tensor warmup_output =
          block->forward_output_only_assuming_validated(
              block_case.hidden,
              block_case.time_embedding,
              block_case.combined_indices,
              block_case.rope_frequencies,
              block_case.global_cu_seqlens);
      (void)warmup_output;
    }
    ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
    c10_npu::NPUCachingAllocator::resetPeakStats(environment->local_rank);
    for (int32_t iteration = 0; iteration < 10; ++iteration) {
      synchronize_ranks(groups.tp.get(), groups.u.get(), device);
      ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
      const auto start = std::chrono::steady_clock::now();
      const torch::Tensor measured =
          block->forward_output_only_assuming_validated(
              block_case.hidden,
              block_case.time_embedding,
              block_case.combined_indices,
              block_case.rope_frequencies,
              block_case.global_cu_seqlens);
      ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
      const double local_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - start)
                                  .count();
      max_rank_iterations.push_back(
          global_max(local_ms, groups.tp.get(), groups.u.get(), device));
      passed = passed && torch::isfinite(measured).all().item<bool>();
    }
    p50_ms = percentile(max_rank_iterations, 0.50);
    p95_ms = percentile(max_rank_iterations, 0.95);
    const auto& stats =
        c10_npu::NPUCachingAllocator::getDeviceStats(environment->local_rank);
    const int64_t local_peak =
        stats
            .allocated_bytes[static_cast<size_t>(
                c10_npu::NPUCachingAllocator::StatType::AGGREGATE)]
            .peak;
    max_peak_allocated_bytes =
        global_max_int64(local_peak, groups.tp.get(), groups.u.get(), device);
    max_resident_module_bytes = global_max_int64(
        module_bytes(*block), groups.tp.get(), groups.u.get(), device);
    constexpr int64_t kPeakAllocatedLimit = 10LL * 1024 * 1024 * 1024;
    passed = passed && max_peak_allocated_bytes < kPeakAllocatedLimit;
  }

  torch::Tensor failures =
      torch::tensor({passed ? 0 : 1},
                    torch::TensorOptions().dtype(torch::kInt32).device(device));
  groups.tp->allreduce(failures);
  groups.u->allreduce(failures);
  const int32_t global_failures = failures.item<int32_t>();

  std::cout << "C8D_TP2_U8_RANK rank=" << environment->global_rank
            << " tp_rank=" << coordinates.tp_rank
            << " u_rank=" << coordinates.u_rank
            << " local_rows=" << block_case.hidden.size(0)
            << " local_valid_rows=" << block_case.local_valid_rows
            << " real_heads=" << real_heads
            << " cross_tp_exact=" << cross_tp_exact << " local_pass=" << passed
            << std::endl;
  if (environment->global_rank == 0) {
    std::cout << "C8D_TP2_U8_AGGREGATE case=" << environment->test_case
              << " global_failures=" << global_failures
              << " max_attention_relative_l2=" << max_attention_relative_l2
              << " max_mlp_relative_l2=" << max_mlp_relative_l2
              << " max_output_relative_l2=" << max_output_relative_l2
              << " split_qkv=1 validated_once=1"
              << " profile_enabled=" << profile_enabled << " p50_ms=" << p50_ms
              << " p95_ms=" << p95_ms
              << " max_peak_allocated_bytes=" << max_peak_allocated_bytes
              << " max_resident_module_bytes=" << max_resident_module_bytes
              << std::endl;
    if (global_failures == 0) {
      std::cout << (environment->test_case == "production"
                        ? "H3_TP2_U8_PRODUCTION_BLOCK=PASS"
                        : "H3_TP2_U8_C4_BLOCK=PASS")
                << std::endl;
    }
  }
  const char* hold_value = std::getenv("MINIMAX_H3_LEASE_HOLD_SECONDS");
  if (hold_value != nullptr) {
    const int32_t hold_seconds = std::stoi(hold_value);
    if (hold_seconds < 0 || hold_seconds > 60) {
      throw std::invalid_argument(
          "MINIMAX_H3_LEASE_HOLD_SECONDS must be in [0,60]");
    }
    std::this_thread::sleep_for(std::chrono::seconds(hold_seconds));
  }
  EXPECT_EQ(global_failures, 0);
}

}  // namespace
}  // namespace xllm
