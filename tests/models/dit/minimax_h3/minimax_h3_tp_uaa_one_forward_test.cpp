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

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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
#include "models/dit/transformers/minimax_h3_tp_uaa_denoiser.h"

namespace xllm {
namespace {

struct Environment {
  int32_t global_rank;
  int32_t local_rank;
  int32_t local_world_size;
  int32_t world_size;
  int32_t port;
  int32_t hold_seconds;
  std::filesystem::path checkpoint;
  std::filesystem::path prepared;
  std::filesystem::path official;
};

std::optional<Environment> environment() {
  const char* enabled = std::getenv("MINIMAX_H3_TP_UAA_ONE_FORWARD_HCCL");
  if (enabled == nullptr || std::string(enabled) != "1") {
    return std::nullopt;
  }
  const char* rank = std::getenv("RANK");
  const char* local_rank = std::getenv("LOCAL_RANK");
  const char* local_world_size = std::getenv("LOCAL_WORLD_SIZE");
  const char* world_size = std::getenv("WORLD_SIZE");
  const char* port = std::getenv("MINIMAX_H3_HCCL_PORT");
  const char* checkpoint = std::getenv("MINIMAX_H3_CHECKPOINT");
  const char* prepared = std::getenv("MINIMAX_H3_REF2VA_PRODUCTION_PREPARED");
  const char* official = std::getenv("MINIMAX_H3_REF2VA_PRODUCTION_OFFICIAL");
  const char* hold = std::getenv("MINIMAX_H3_LEASE_HOLD_SECONDS");
  if (rank == nullptr || local_rank == nullptr || local_world_size == nullptr ||
      world_size == nullptr || port == nullptr || checkpoint == nullptr ||
      prepared == nullptr || official == nullptr) {
    throw std::invalid_argument(
        "MiniMax-H3 one-forward environment is incomplete");
  }
  return Environment{.global_rank = std::stoi(rank),
                     .local_rank = std::stoi(local_rank),
                     .local_world_size = std::stoi(local_world_size),
                     .world_size = std::stoi(world_size),
                     .port = std::stoi(port),
                     .hold_seconds = hold == nullptr ? 0 : std::stoi(hold),
                     .checkpoint = checkpoint,
                     .prepared = prepared,
                     .official = official};
}

std::filesystem::path resolve(std::filesystem::path path,
                              const std::string& filename) {
  if (std::filesystem::is_directory(path)) {
    path /= filename;
  }
  return path;
}

std::array<unsigned char, 32> file_digest(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::invalid_argument("Cannot open one-forward artifact");
  }
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context ||
      EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("Cannot initialize one-forward SHA256");
  }
  std::array<char, 1 << 20> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0 &&
        EVP_DigestUpdate(context.get(), buffer.data(), count) != 1) {
      throw std::runtime_error("Cannot update one-forward SHA256");
    }
  }
  std::array<unsigned char, 32> digest{};
  unsigned int size = 0;
  if (!input.eof() ||
      EVP_DigestFinal_ex(context.get(), digest.data(), &size) != 1 ||
      size != digest.size()) {
    throw std::runtime_error("Cannot finalize one-forward SHA256");
  }
  return digest;
}

std::string encoded_digest(const std::array<unsigned char, 32>& digest) {
  std::ostringstream encoded;
  encoded << std::hex << std::setfill('0');
  for (unsigned char value : digest) {
    encoded << std::setw(2) << static_cast<int>(value);
  }
  return encoded.str();
}

std::array<unsigned char, 32> tensor_digest(const torch::Tensor& value) {
  const torch::Tensor cpu = value.to(torch::kCPU).contiguous();
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context ||
      EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), cpu.const_data_ptr(), cpu.nbytes()) !=
          1) {
    throw std::runtime_error("Cannot hash one-forward tensor");
  }
  std::array<unsigned char, 32> digest{};
  unsigned int size = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &size) != 1 ||
      size != digest.size()) {
    throw std::runtime_error("Cannot finalize one-forward tensor hash");
  }
  return digest;
}

struct Groups {
  std::unique_ptr<ProcessGroup> tp;
  std::unique_ptr<ProcessGroup> u;
};

Groups create_groups(const Environment& env,
                     const MiniMaxH3TPUAAParallelCoordinates& coordinates,
                     const torch::Device& device) {
  return {.tp = create_process_group(env.global_rank,
                                     coordinates.tp_rank,
                                     coordinates.tp_group_ranks,
                                     env.world_size,
                                     kMiniMaxH3TPSize,
                                     env.port + coordinates.u_rank,
                                     "127.0.0.1",
                                     "minimax_h3_c8f_tp_group",
                                     device),
          .u = create_process_group(env.global_rank,
                                    coordinates.u_rank,
                                    coordinates.u_group_ranks,
                                    env.world_size,
                                    kMiniMaxH3UaaSize,
                                    env.port + 8 + coordinates.tp_rank,
                                    "127.0.0.1",
                                    "minimax_h3_c8f_u_group",
                                    device)};
}

struct Metrics {
  double relative_l2;
  double cosine;
  double max_abs;
};

Metrics metrics(const torch::Tensor& actual, const torch::Tensor& expected) {
  const torch::Tensor actual_fp64 = actual.to(torch::kCPU).to(torch::kFloat64);
  const torch::Tensor expected_fp64 =
      expected.to(torch::kCPU).to(torch::kFloat64);
  const torch::Tensor difference = actual_fp64 - expected_fp64;
  const double actual_norm = actual_fp64.norm().item<double>();
  const double expected_norm = expected_fp64.norm().item<double>();
  return {
      .relative_l2 =
          difference.norm().item<double>() / std::max(expected_norm, 1e-12),
      .cosine =
          actual_fp64.flatten().dot(expected_fp64.flatten()).item<double>() /
          std::max(actual_norm * expected_norm, 1e-12),
      .max_abs = difference.abs().max().item<double>()};
}

bool digest_consensus(const std::array<unsigned char, 32>& digest,
                      ProcessGroup* tp_group,
                      ProcessGroup* u_group,
                      const torch::Device& device) {
  const torch::Tensor local =
      torch::from_blob(
          const_cast<unsigned char*>(digest.data()), {32}, torch::kUInt8)
          .clone()
          .to(device);
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
  return tp_matches && u_matches;
}

TEST(MiniMaxH3TPUAAOneForwardHcclTest, MatchesProductionForwardOne) {
  const std::optional<Environment> env = environment();
  if (!env.has_value()) {
    GTEST_SKIP() << "Set MINIMAX_H3_TP_UAA_ONE_FORWARD_HCCL=1 on 16 ranks";
  }
  ASSERT_EQ(env->world_size, kMiniMaxH3TPUAAWorldSize);
  ASSERT_EQ(env->local_world_size, kMiniMaxH3TPUAAWorldSize);
  ASSERT_EQ(env->local_rank, env->global_rank);

  const std::filesystem::path prepared_path = resolve(
      env->prepared, "minimax_h3_ref2va_production_prepared.safetensors");
  const std::filesystem::path official_path =
      resolve(env->official,
              "minimax_h3_ref2va_production_official_hf_full.safetensors");
  ASSERT_EQ(encoded_digest(file_digest(prepared_path)),
            "8bdb0ffe3a091bafb16ec09c7f176830d1797ba46d2ca48f53c708ce92277bea");
  ASSERT_EQ(encoded_digest(file_digest(official_path)),
            "459248e62ed7c49899d7ceda6f7d7b8ec4fa8eca6f477f86eaa148aae36d7e93");
  const std::unique_ptr<StateDict> prepared =
      StateDictFromSafeTensor::load(prepared_path.string());
  const std::unique_ptr<StateDict> official =
      StateDictFromSafeTensor::load(official_path.string());
  ASSERT_NE(prepared, nullptr);
  ASSERT_NE(official, nullptr);

  Device rank_device(env->local_rank);
  rank_device.set_device();
  const torch::Device device = rank_device.unwrap();
  const MiniMaxH3TPUAAParallelCoordinates coordinates =
      minimax_h3_tp_uaa_coordinates(env->global_rank);
  Groups groups = create_groups(*env, coordinates, device);
  ASSERT_NE(groups.tp, nullptr);
  ASSERT_NE(groups.u, nullptr);
  torch::NoGradGuard no_grad;

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

  MiniMaxH3TPUAAResidentDenoiser denoiser(
      MiniMaxH3C4Config{},
      groups.tp.get(),
      groups.u.get(),
      torch::TensorOptions().device(device).dtype(torch::kBFloat16));
  auto loader = std::make_unique<DiTModelLoader>(env->checkpoint);
  ASSERT_TRUE(loader->has_component("transformer"));
  auto transformer_loader = loader->take_component_loader("transformer");
  ASSERT_NE(transformer_loader, nullptr);
  denoiser->load_source_weights(transformer_loader->get_state_dicts());
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
  transformer_loader.reset();
  loader.reset();

  const torch::Tensor initial_video =
      prepared->get_tensor("initial.video_rows").to(device).contiguous();
  const torch::Tensor initial_audio =
      prepared->get_tensor("initial.audio_rows").to(device).contiguous();
  const MiniMaxH3TPUAAOneForwardOutput output =
      denoiser->forward(layout,
                        initial_video,
                        initial_audio,
                        plan.unique_timesteps,
                        plan.inverse_indices);
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);

  const torch::Tensor video_x0 = MiniMaxH3Scheduler::velocity_to_x0(
      initial_video,
      output.final_output.raw_selected_video_logits,
      schedule.video.timesteps[0]);
  const torch::Tensor audio_x0 = MiniMaxH3Scheduler::velocity_to_x0(
      initial_audio,
      output.final_output.raw_selected_audio_logits,
      schedule.audio.timesteps[0]);
  torch::Tensor video_after =
      MiniMaxH3Scheduler::step_eta0(initial_video,
                                    video_x0,
                                    schedule.video.sigmas[0].item<float>(),
                                    schedule.video.sigmas[1].item<float>());
  torch::Tensor audio_after =
      MiniMaxH3Scheduler::step_eta0(initial_audio,
                                    audio_x0,
                                    schedule.audio.sigmas[0].item<float>(),
                                    schedule.audio.sigmas[1].item<float>());
  const torch::Tensor image_update = layout.update_mask.to(device);
  const torch::Tensor audio_update = layout.audio_update_mask.to(device);
  video_after.index_put_({~image_update}, initial_video.index({~image_update}));
  audio_after.index_put_({~audio_update}, initial_audio.index({~audio_update}));
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);

  const Metrics video_metrics = metrics(
      video_after, official->get_tensor("trajectory.forward_001.video_rows"));
  const Metrics audio_metrics = metrics(
      audio_after, official->get_tensor("trajectory.forward_001.audio_rows"));
  const bool video_consensus = digest_consensus(
      tensor_digest(video_after), groups.tp.get(), groups.u.get(), device);
  const bool audio_consensus = digest_consensus(
      tensor_digest(audio_after), groups.tp.get(), groups.u.get(), device);
  bool passed =
      torch::isfinite(video_after).all().item<bool>() &&
      torch::isfinite(audio_after).all().item<bool>() &&
      video_metrics.relative_l2 <= 0.01 && video_metrics.cosine >= 0.99995 &&
      audio_metrics.relative_l2 <= 0.01 && audio_metrics.cosine >= 0.99995 &&
      video_consensus && audio_consensus;
  torch::Tensor failures =
      torch::tensor({passed ? 0 : 1},
                    torch::TensorOptions().dtype(torch::kInt32).device(device));
  groups.tp->allreduce(failures);
  groups.u->allreduce(failures);
  const int32_t global_failures = failures.item<int32_t>();

  std::cout << "C8F_ONE_FORWARD_RANK rank=" << env->global_rank
            << " video_relative_l2=" << video_metrics.relative_l2
            << " video_cosine=" << video_metrics.cosine
            << " video_max_abs=" << video_metrics.max_abs
            << " audio_relative_l2=" << audio_metrics.relative_l2
            << " audio_cosine=" << audio_metrics.cosine
            << " audio_max_abs=" << audio_metrics.max_abs
            << " local_pass=" << passed << std::endl;
  if (env->global_rank == 0) {
    std::cout << "C8F_ONE_FORWARD_AGGREGATE global_failures=" << global_failures
              << " profile_enabled=0" << std::endl;
    if (global_failures == 0) {
      std::cout << "H3_TP2_U8_PRODUCTION_ONE_FORWARD=PASS" << std::endl;
    }
  }
  std::this_thread::sleep_for(std::chrono::seconds(env->hold_seconds));
  EXPECT_EQ(global_failures, 0);
}

}  // namespace
}  // namespace xllm
