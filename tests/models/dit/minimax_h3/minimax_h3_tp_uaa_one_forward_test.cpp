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
#include <nlohmann/json.hpp>
#include <numeric>
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

constexpr char kMiniMaxH3OracleArchiveSha256[] =
    "617f7eeff91e97a8e42601bf068cb716bb3925fd0fac74ada57b430a86a6613c";
constexpr char kMiniMaxH3FullOracleArchiveSha256[] =
    "7fb6563bbd5cfcd39ebb8282593694ed3fad214cc2e4bbcdc3f440578dad0594";
constexpr char kMiniMaxH3CheckpointManifestDigest[] =
    "092289ee588832268bb2ce82dbe15263802a8bf70954625d781fe89df47ea068";

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
  std::filesystem::path oracle;
  std::string oracle_sha256;
};

std::optional<Environment> environment(
    const char* enable_name = "MINIMAX_H3_TP_UAA_ONE_FORWARD_HCCL") {
  const char* enabled = std::getenv(enable_name);
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
  const char* oracle = std::getenv("MINIMAX_H3_REF2VA_PRODUCTION_ORACLE");
  const char* oracle_sha256 = std::getenv("MINIMAX_H3_ORACLE_SHA256");
  const char* hold = std::getenv("MINIMAX_H3_LEASE_HOLD_SECONDS");
  if (rank == nullptr || local_rank == nullptr || local_world_size == nullptr ||
      world_size == nullptr || port == nullptr || checkpoint == nullptr ||
      prepared == nullptr || official == nullptr) {
    throw std::invalid_argument(
        "MiniMax-H3 one-forward environment is incomplete");
  }
  if ((oracle == nullptr) != (oracle_sha256 == nullptr)) {
    throw std::invalid_argument(
        "MiniMax-H3 Oracle path and SHA256 must be provided together");
  }
  return Environment{
      .global_rank = std::stoi(rank),
      .local_rank = std::stoi(local_rank),
      .local_world_size = std::stoi(local_world_size),
      .world_size = std::stoi(world_size),
      .port = std::stoi(port),
      .hold_seconds = hold == nullptr ? 0 : std::stoi(hold),
      .checkpoint = checkpoint,
      .prepared = prepared,
      .official = official,
      .oracle = oracle == nullptr ? "" : oracle,
      .oracle_sha256 = oracle_sha256 == nullptr ? "" : oracle_sha256};
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

TEST(MiniMaxH3FinalLayerTest, ProjectsUsedRowsAndZerosAlignedPadding) {
  const MiniMaxH3C4Config config;
  MiniMaxH3FinalLayer layer(
      config,
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kBFloat16));
  torch::NoGradGuard no_grad;
  for (auto& parameter : layer->named_parameters(/*recurse=*/true)) {
    parameter.value().zero_();
    if (parameter.key() == "video_out.bias" ||
        parameter.key() == "audio_out.bias") {
      parameter.value().fill_(1.0F);
    }
  }
  H3PackedLayout layout;
  layout.used_length = 3;
  layout.aligned_length = 4;
  layout.img_pos = torch::tensor({1}, torch::kInt64);
  layout.audio_pos = torch::tensor({2}, torch::kInt64);
  layout.update_mask = torch::tensor({true}, torch::kBool);
  layout.audio_update_mask = torch::tensor({true}, torch::kBool);
  const torch::Tensor input =
      torch::zeros({4, config.hidden_size}, torch::kBFloat16);
  const torch::Tensor time_embedding =
      torch::zeros({1, config.time_embed_dim}, torch::kFloat32);
  const torch::Tensor inverse_indices = torch::zeros({4}, torch::kInt64);

  const MiniMaxH3FinalOutput output =
      layer->forward(input, time_embedding, inverse_indices, layout);
  EXPECT_TRUE(
      torch::equal(output.all_video_logits.narrow(0, 0, layout.used_length),
                   torch::ones({layout.used_length, config.video_patch_dim},
                               torch::kFloat32)));
  EXPECT_TRUE(torch::equal(
      output.all_audio_logits.narrow(0, 0, layout.used_length),
      torch::ones({layout.used_length, config.audio_dim}, torch::kFloat32)));
  EXPECT_EQ(output.all_video_logits.index({3}).count_nonzero().item<int64_t>(),
            0);
  EXPECT_EQ(output.all_audio_logits.index({3}).count_nonzero().item<int64_t>(),
            0);
  EXPECT_TRUE(
      torch::equal(output.raw_selected_video_logits,
                   torch::ones({1, config.video_patch_dim}, torch::kFloat32)));
  EXPECT_TRUE(
      torch::equal(output.raw_selected_audio_logits,
                   torch::ones({1, config.audio_dim}, torch::kFloat32)));

  layout.used_length = 5;
  EXPECT_THROW(layer->forward(input, time_embedding, inverse_indices, layout),
               std::invalid_argument);
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
  std::unique_ptr<StateDict> oracle;
  if (!env->oracle.empty()) {
    ASSERT_TRUE(std::filesystem::is_directory(env->oracle));
    const std::filesystem::path oracle_path = resolve(
        env->oracle,
        "minimax_h3_ref2va_production_official_hf_smoke_oracle.safetensors");
    const std::filesystem::path oracle_manifest_path =
        resolve(env->oracle,
                "minimax_h3_ref2va_production_official_hf_smoke_oracle.json");
    ASSERT_EQ(env->oracle_sha256, kMiniMaxH3OracleArchiveSha256);
    ASSERT_EQ(encoded_digest(file_digest(oracle_path)),
              kMiniMaxH3OracleArchiveSha256);
    std::ifstream oracle_manifest_input(oracle_manifest_path);
    ASSERT_TRUE(oracle_manifest_input);
    const nlohmann::json oracle_manifest =
        nlohmann::json::parse(oracle_manifest_input);
    ASSERT_EQ(oracle_manifest.at("schema").get<std::string>(),
              "xllm.minimax_h3.ref2va_production_official_oracle/v1");
    ASSERT_EQ(oracle_manifest.at("condition_backend").get<std::string>(),
              "official_hf");
    ASSERT_EQ(oracle_manifest.at("artifact").at("sha256").get<std::string>(),
              kMiniMaxH3OracleArchiveSha256);
    ASSERT_EQ(
        oracle_manifest.at("prepared").at("artifact_sha256").get<std::string>(),
        "8bdb0ffe3a091bafb16ec09c7f176830d1797ba46d2ca48f53c708ce92277bea");
    ASSERT_EQ(oracle_manifest.at("source").at("revision").get<std::string>(),
              "d30c748f5f5d0925a5af14dc0e6a6de983025e63");
    ASSERT_EQ(oracle_manifest.at("checkpoint").at("digest").get<std::string>(),
              kMiniMaxH3CheckpointManifestDigest);
    ASSERT_EQ(oracle_manifest.at("oracle").at("selected_forwards"),
              nlohmann::json::array({1}));
    oracle = StateDictFromSafeTensor::load(oracle_path.string());
    ASSERT_NE(oracle, nullptr);
  }

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
  bool oracle_passed = true;
  if (oracle != nullptr) {
    const Metrics transformer_hidden_metrics =
        metrics(output.full_transformer_output.narrow(0, 0, layout.used_length),
                oracle->get_tensor("oracle.forward_001.transformer_hidden"));
    const Metrics final_activation_metrics =
        metrics(output.final_output.activation.narrow(0, 0, layout.used_length),
                oracle->get_tensor("oracle.forward_001.final_activation"));
    const Metrics raw_video_velocity_metrics =
        metrics(output.final_output.raw_selected_video_logits,
                oracle->get_tensor("oracle.forward_001.raw_video_velocity"));
    const Metrics raw_audio_velocity_metrics =
        metrics(output.final_output.raw_selected_audio_logits,
                oracle->get_tensor("oracle.forward_001.raw_audio_velocity"));
    const Metrics video_x0_metrics =
        metrics(video_x0.index({image_update}),
                oracle->get_tensor("oracle.forward_001.target_video_x0"));
    const Metrics audio_x0_metrics = metrics(
        audio_x0, oracle->get_tensor("oracle.forward_001.target_audio_x0"));
    const Metrics oracle_video_rows_metrics = metrics(
        video_after, oracle->get_tensor("oracle.forward_001.video_rows_after"));
    const Metrics oracle_audio_rows_metrics = metrics(
        audio_after, oracle->get_tensor("oracle.forward_001.audio_rows_after"));
    oracle_passed = transformer_hidden_metrics.max_abs == 0.0 &&
                    final_activation_metrics.max_abs == 0.0 &&
                    raw_video_velocity_metrics.max_abs == 0.0 &&
                    raw_audio_velocity_metrics.max_abs == 0.0 &&
                    video_x0_metrics.max_abs == 0.0 &&
                    audio_x0_metrics.max_abs == 0.0 &&
                    oracle_video_rows_metrics.max_abs == 0.0 &&
                    oracle_audio_rows_metrics.max_abs == 0.0;
    std::cout
        << "H3_ORACLE_BOUNDARY rank=" << env->global_rank
        << " forward=1 transformer_hidden_relative_l2="
        << transformer_hidden_metrics.relative_l2
        << " transformer_hidden_max_abs=" << transformer_hidden_metrics.max_abs
        << " final_activation_relative_l2="
        << final_activation_metrics.relative_l2
        << " final_activation_max_abs=" << final_activation_metrics.max_abs
        << " raw_video_velocity_relative_l2="
        << raw_video_velocity_metrics.relative_l2
        << " raw_video_velocity_max_abs=" << raw_video_velocity_metrics.max_abs
        << " raw_audio_velocity_relative_l2="
        << raw_audio_velocity_metrics.relative_l2
        << " raw_audio_velocity_max_abs=" << raw_audio_velocity_metrics.max_abs
        << " video_x0_relative_l2=" << video_x0_metrics.relative_l2
        << " video_x0_max_abs=" << video_x0_metrics.max_abs
        << " audio_x0_relative_l2=" << audio_x0_metrics.relative_l2
        << " audio_x0_max_abs=" << audio_x0_metrics.max_abs
        << " video_rows_relative_l2=" << oracle_video_rows_metrics.relative_l2
        << " video_rows_max_abs=" << oracle_video_rows_metrics.max_abs
        << " audio_rows_relative_l2=" << oracle_audio_rows_metrics.relative_l2
        << " audio_rows_max_abs=" << oracle_audio_rows_metrics.max_abs
        << " local_pass=" << oracle_passed << std::endl;
  }

  const torch::Tensor official_video_after_one =
      official->get_tensor("trajectory.forward_001.video_rows")
          .to(device)
          .contiguous();
  const torch::Tensor official_audio_after_one =
      official->get_tensor("trajectory.forward_001.audio_rows")
          .to(device)
          .contiguous();
  const MiniMaxH3RowTimestepPlan plan_two = minimax_h3_build_row_timestep_plan(
      layout,
      schedule.video.timesteps[1].item<float>(),
      schedule.audio.timesteps[1].item<float>());
  const MiniMaxH3FinalOutput teacher_forced_two =
      denoiser->forward_output_only(layout,
                                    official_video_after_one,
                                    official_audio_after_one,
                                    plan_two.unique_timesteps,
                                    plan_two.inverse_indices);
  const torch::Tensor teacher_video_target =
      official_video_after_one.index({image_update});
  const torch::Tensor teacher_audio_target =
      official_audio_after_one.index({audio_update});
  const torch::Tensor teacher_video_x0 = MiniMaxH3Scheduler::velocity_to_x0(
      teacher_video_target,
      teacher_forced_two.raw_selected_video_logits.index({image_update}),
      schedule.video.timesteps[1]);
  const torch::Tensor teacher_audio_x0 = MiniMaxH3Scheduler::velocity_to_x0(
      teacher_audio_target,
      teacher_forced_two.raw_selected_audio_logits.index({audio_update}),
      schedule.audio.timesteps[1]);
  torch::Tensor teacher_video_after = official_video_after_one.clone();
  torch::Tensor teacher_audio_after = official_audio_after_one.clone();
  teacher_video_after.index_put_(
      {image_update},
      MiniMaxH3Scheduler::step_eta0(teacher_video_target,
                                    teacher_video_x0,
                                    schedule.video.sigmas[1].item<float>(),
                                    schedule.video.sigmas[2].item<float>()));
  teacher_audio_after.index_put_(
      {audio_update},
      MiniMaxH3Scheduler::step_eta0(teacher_audio_target,
                                    teacher_audio_x0,
                                    schedule.audio.sigmas[1].item<float>(),
                                    schedule.audio.sigmas[2].item<float>()));
  const Metrics teacher_video_metrics =
      metrics(teacher_video_after,
              official->get_tensor("trajectory.forward_002.video_rows"));
  const Metrics teacher_audio_metrics =
      metrics(teacher_audio_after,
              official->get_tensor("trajectory.forward_002.audio_rows"));
  const bool teacher_video_consensus =
      digest_consensus(tensor_digest(teacher_video_after),
                       groups.tp.get(),
                       groups.u.get(),
                       device);
  const bool teacher_audio_consensus =
      digest_consensus(tensor_digest(teacher_audio_after),
                       groups.tp.get(),
                       groups.u.get(),
                       device);
  const bool teacher_forced_pass = teacher_video_metrics.relative_l2 <= 0.01 &&
                                   teacher_video_metrics.cosine >= 0.99995 &&
                                   teacher_audio_metrics.relative_l2 <= 0.01 &&
                                   teacher_audio_metrics.cosine >= 0.99995 &&
                                   teacher_video_consensus &&
                                   teacher_audio_consensus;
  const bool video_consensus = digest_consensus(
      tensor_digest(video_after), groups.tp.get(), groups.u.get(), device);
  const bool audio_consensus = digest_consensus(
      tensor_digest(audio_after), groups.tp.get(), groups.u.get(), device);
  bool passed = torch::isfinite(video_after).all().item<bool>() &&
                torch::isfinite(audio_after).all().item<bool>() &&
                video_metrics.relative_l2 <= 0.01 &&
                video_metrics.cosine >= 0.99995 &&
                audio_metrics.relative_l2 <= 0.01 &&
                audio_metrics.cosine >= 0.99995 && video_consensus &&
                audio_consensus && teacher_forced_pass && oracle_passed;
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
  std::cout << "C8_ORACLE_BOOTSTRAP rank=" << env->global_rank
            << " forward=2 teacher_video_relative_l2="
            << teacher_video_metrics.relative_l2
            << " teacher_video_cosine=" << teacher_video_metrics.cosine
            << " teacher_video_max_abs=" << teacher_video_metrics.max_abs
            << " teacher_audio_relative_l2="
            << teacher_audio_metrics.relative_l2
            << " teacher_audio_cosine=" << teacher_audio_metrics.cosine
            << " teacher_audio_max_abs=" << teacher_audio_metrics.max_abs
            << " local_pass=" << teacher_forced_pass << std::endl;
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

TEST(MiniMaxH3TPUAATrajectoryHcclTest, MatchesProductionCheckpoints) {
  const std::optional<Environment> env =
      environment("MINIMAX_H3_TP_UAA_TRAJECTORY_HCCL");
  if (!env.has_value()) {
    GTEST_SKIP() << "Set MINIMAX_H3_TP_UAA_TRAJECTORY_HCCL=1 on 16 ranks";
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
  std::unique_ptr<StateDict> trajectory_oracle;
  if (!env->oracle.empty()) {
    ASSERT_TRUE(std::filesystem::is_directory(env->oracle));
    const std::filesystem::path oracle_path = resolve(
        env->oracle,
        "minimax_h3_ref2va_production_official_hf_full_oracle.safetensors");
    const std::filesystem::path oracle_manifest_path =
        resolve(env->oracle,
                "minimax_h3_ref2va_production_official_hf_full_oracle.json");
    ASSERT_EQ(env->oracle_sha256, kMiniMaxH3FullOracleArchiveSha256);
    ASSERT_EQ(encoded_digest(file_digest(oracle_path)),
              kMiniMaxH3FullOracleArchiveSha256);
    std::ifstream oracle_manifest_input(oracle_manifest_path);
    ASSERT_TRUE(oracle_manifest_input);
    const nlohmann::json oracle_manifest =
        nlohmann::json::parse(oracle_manifest_input);
    ASSERT_EQ(oracle_manifest.at("schema").get<std::string>(),
              "xllm.minimax_h3.ref2va_production_official_oracle/v1");
    ASSERT_EQ(oracle_manifest.at("condition_backend").get<std::string>(),
              "official_hf");
    ASSERT_EQ(oracle_manifest.at("artifact").at("sha256").get<std::string>(),
              kMiniMaxH3FullOracleArchiveSha256);
    ASSERT_EQ(
        oracle_manifest.at("prepared").at("artifact_sha256").get<std::string>(),
        "8bdb0ffe3a091bafb16ec09c7f176830d1797ba46d2ca48f53c708ce92277bea");
    ASSERT_EQ(oracle_manifest.at("source").at("revision").get<std::string>(),
              "d30c748f5f5d0925a5af14dc0e6a6de983025e63");
    ASSERT_EQ(oracle_manifest.at("checkpoint").at("digest").get<std::string>(),
              kMiniMaxH3CheckpointManifestDigest);
    std::vector<int64_t> expected_forwards(49);
    std::iota(expected_forwards.begin(), expected_forwards.end(), 1);
    ASSERT_EQ(oracle_manifest.at("oracle").at("selected_forwards"),
              expected_forwards);
    trajectory_oracle = StateDictFromSafeTensor::load(oracle_path.string());
    ASSERT_NE(trajectory_oracle, nullptr);
  }

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
  const std::vector<int64_t> checkpoints = {1, 2, 4, 8, 49};
  size_t observed = 0;
  size_t oracle_observed = 0;
  bool passed = true;
  bool oracle_passed = true;
  const MiniMaxH3TPUAATrajectoryOutput output = denoiser->run_base_trajectory(
      layout,
      initial_video,
      initial_audio,
      [&](const MiniMaxH3TPUAABoundary& boundary) {
        const int64_t forward_index = boundary.forward_index;
        if (trajectory_oracle != nullptr) {
          std::ostringstream oracle_prefix;
          oracle_prefix << "oracle.forward_" << std::setw(3)
                        << std::setfill('0') << forward_index;
          const auto exact = [&](const torch::Tensor& actual,
                                 const std::string& suffix) {
            return metrics(actual,
                           trajectory_oracle->get_tensor(oracle_prefix.str() +
                                                         suffix))
                       .max_abs == 0.0;
          };
          const bool forward_oracle_passed =
              exact(boundary.video_rows_before, ".input_video_rows") &&
              exact(boundary.audio_rows_before, ".input_audio_rows") &&
              exact(boundary.raw_video_velocity, ".raw_video_velocity") &&
              exact(boundary.raw_audio_velocity, ".raw_audio_velocity") &&
              exact(boundary.video_x0, ".target_video_x0") &&
              exact(boundary.audio_x0, ".target_audio_x0") &&
              exact(boundary.video_rows_after, ".video_rows_after") &&
              exact(boundary.audio_rows_after, ".audio_rows_after");
          oracle_passed = oracle_passed && forward_oracle_passed;
          ++oracle_observed;
          if (env->global_rank == 0) {
            std::cout << "H3_ORACLE_TRAJECTORY forward=" << forward_index
                      << " exact=" << forward_oracle_passed << std::endl;
          }
        }
        if (observed >= checkpoints.size() ||
            forward_index != checkpoints[observed]) {
          return;
        }
        std::ostringstream prefix;
        prefix << "trajectory.forward_" << std::setw(3) << std::setfill('0')
               << forward_index;
        const Metrics video_metrics =
            metrics(boundary.video_rows_after,
                    official->get_tensor(prefix.str() + ".video_rows"));
        const Metrics audio_metrics =
            metrics(boundary.audio_rows_after,
                    official->get_tensor(prefix.str() + ".audio_rows"));
        const bool video_consensus =
            digest_consensus(tensor_digest(boundary.video_rows_after),
                             groups.tp.get(),
                             groups.u.get(),
                             device);
        const bool audio_consensus =
            digest_consensus(tensor_digest(boundary.audio_rows_after),
                             groups.tp.get(),
                             groups.u.get(),
                             device);
        passed =
            passed && video_metrics.max_abs == 0.0 &&
            audio_metrics.max_abs == 0.0 && video_consensus &&
            audio_consensus &&
            torch::isfinite(boundary.video_rows_after).all().item<bool>() &&
            torch::isfinite(boundary.audio_rows_after).all().item<bool>();
        ++observed;
        std::cout << "C8G_TRAJECTORY rank=" << env->global_rank
                  << " forward=" << forward_index
                  << " video_relative_l2=" << video_metrics.relative_l2
                  << " video_cosine=" << video_metrics.cosine
                  << " audio_relative_l2=" << audio_metrics.relative_l2
                  << " audio_cosine=" << audio_metrics.cosine << std::endl;
      });
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
  const bool trajectory_passed = passed && observed == checkpoints.size() &&
                                 output.transformer_forwards == 49 &&
                                 output.block_forwards == 2450 &&
                                 (trajectory_oracle == nullptr ||
                                  (oracle_observed == 49 && oracle_passed));

  denoiser = nullptr;
  Device::empty_cache(env->local_rank);
  const torch::Tensor image_update = layout.update_mask.to(device);
  const torch::Tensor audio_update = layout.audio_update_mask.to(device);
  const torch::Tensor final_video_latent = minimax_h3_unpatchify_video_tokens(
      output.video_rows.index({image_update}).contiguous(),
      {.channels = MiniMaxH3TransformerConfig::kVideoLatentDim,
       .temporal = target.latent_t,
       .height = target.latent_h,
       .width = target.latent_w});
  const torch::Tensor final_audio_latent = minimax_h3_unpack_audio_tokens(
      output.audio_rows.index({audio_update}).contiguous(),
      target.audio_channels,
      target.audio_t);

  auto media_loader = std::make_unique<DiTModelLoader>(env->checkpoint);
  DiTModelContext media_context(
      ParallelArgs(0, 1, nullptr),
      media_loader->get_model_args(),
      media_loader->get_quant_args(),
      torch::TensorOptions().device(device).dtype(torch::kBFloat16),
      DiTCacheConfig(),
      media_loader->get_model_type());
  MiniMaxH3Pipeline media_pipeline(media_context);
  media_pipeline->load_model(std::move(media_loader));
  media_pipeline->load_c6a_video_vae();
  media_pipeline->load_c6b_audio_vae();
  const torch::Tensor decoded_video =
      media_pipeline->c6a_video_vae()->decode_normalized(final_video_latent);
  const torch::Tensor decoded_audio =
      media_pipeline->c6b_audio_vae()->decode_normalized(final_audio_latent);
  ASSERT_EQ(rank_device.synchronize_default_stream(), 0);
  const torch::Tensor official_video =
      official->get_tensor("decoded.video_uint8").to(torch::kFloat32) / 255.0F;
  const torch::Tensor official_audio =
      official->get_tensor("decoded.audio_float32");
  const Metrics video_media_metrics = metrics(decoded_video, official_video);
  const Metrics audio_media_metrics = metrics(decoded_audio, official_audio);
  const bool media_passed =
      decoded_video.sizes().vec() ==
          std::vector<int64_t>({1, 3, 124, 768, 1344}) &&
      decoded_audio.sizes().vec() == std::vector<int64_t>({1, 2, 165600}) &&
      torch::isfinite(decoded_video).all().item<bool>() &&
      torch::isfinite(decoded_audio).all().item<bool>() &&
      decoded_video.min().item<float>() >= 0.0F &&
      decoded_video.max().item<float>() <= 1.0F &&
      decoded_video.std().item<float>() > 0.01F &&
      decoded_audio.min().item<float>() >= -1.0F &&
      decoded_audio.max().item<float>() <= 1.0F &&
      video_media_metrics.relative_l2 <= 0.005 &&
      video_media_metrics.cosine >= 0.99999 &&
      audio_media_metrics.relative_l2 <= 0.001 &&
      audio_media_metrics.cosine >= 0.999999 &&
      audio_media_metrics.max_abs <= 0.002;
  std::cout << "C8G_MEDIA rank=" << env->global_rank
            << " video_relative_l2=" << video_media_metrics.relative_l2
            << " video_cosine=" << video_media_metrics.cosine
            << " video_max_abs=" << video_media_metrics.max_abs
            << " audio_relative_l2=" << audio_media_metrics.relative_l2
            << " audio_cosine=" << audio_media_metrics.cosine
            << " audio_max_abs=" << audio_media_metrics.max_abs
            << " local_pass=" << media_passed << std::endl;
  torch::Tensor failures =
      torch::tensor({trajectory_passed ? 0 : 1, media_passed ? 0 : 1},
                    torch::TensorOptions().dtype(torch::kInt32).device(device));
  groups.tp->allreduce(failures);
  groups.u->allreduce(failures);
  const int32_t trajectory_failures = failures[0].item<int32_t>();
  const int32_t media_failures = failures[1].item<int32_t>();
  if (env->global_rank == 0) {
    std::cout << "C8G_TRAJECTORY_AGGREGATE global_failures="
              << trajectory_failures << " observed_checkpoints=" << observed
              << " observed_oracle_forwards=" << oracle_observed
              << " profile_enabled=0" << std::endl;
    std::cout << "C8G_MEDIA_AGGREGATE global_failures=" << media_failures
              << " profile_enabled=0" << std::endl;
    if (trajectory_failures == 0) {
      std::cout << (trajectory_oracle == nullptr
                        ? "H3_TP2_U8_49_FORWARD_SELECTED_CHECKPOINTS=PASS"
                        : "H3_TP2_U8_49_FORWARD_TRAJECTORY=PASS")
                << std::endl;
    }
    if (media_failures == 0) {
      std::cout << "H3_TP2_U8_PRODUCTION_MEDIA=PASS" << std::endl;
    }
  }
  std::this_thread::sleep_for(std::chrono::seconds(env->hold_seconds));
  EXPECT_EQ(trajectory_failures, 0);
  EXPECT_EQ(media_failures, 0);
}

}  // namespace
}  // namespace xllm
