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
#include <torch/fft.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "framework/state_dict/state_dict.h"
#include "models/dit/pipelines/pipeline_minimax_h3.h"
#include "models/dit/schedulers/minimax_h3_scheduler.h"

namespace xllm {
namespace {

void expect_invalid_argument_contains(const std::function<void()>& operation,
                                      const std::string& expected) {
  try {
    operation();
    FAIL() << "Expected std::invalid_argument containing: " << expected;
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find(expected), std::string::npos)
        << error.what();
  } catch (const std::exception& error) {
    FAIL() << "Expected std::invalid_argument, got: " << error.what();
  }
}

nlohmann::json original_transformer_config() {
  return {
      {"_class_name", "MiniMaxH3DiTModel"},
      {"hidden_size", 5376},
      {"num_layers", 50},
      {"token_refiner_num_layers", 2},
      {"num_attention_heads", 56},
      {"attention_head_dim", 128},
      {"ffn_hidden_size", 14336},
      {"latents_dim", 24},
      {"audio_latents_dim", 32},
      {"patch_size", {1, 2, 2}},
      {"text_dim", 5120},
      {"timestep_input_dim", 256},
      {"time_embed_hidden_size", 5376},
      {"time_embed_dim", 2688},
      {"adaln_out_features", 96768},
      {"final_adaln_out_features", 10752},
      {"rope_inv_freq_len", 16},
      {"norm_eps", 1e-5},
      {"qk_norm_eps", 1e-5},
      {"final_norm_eps", 1e-5},
  };
}

nlohmann::json converted_transformer_config() {
  return {
      {"_class_name", "MiniMaxH3Transformer3DModel"},
      {"hidden_size", 5376},
      {"num_layers", 50},
      {"num_refiner_layers", 2},
      {"num_attention_heads", 56},
      {"attention_head_dim", 128},
      {"ffn_dim", 14336},
      {"in_channels", 24},
      {"audio_in_channels", 32},
      {"patch_size", {1, 2, 2}},
      {"text_dim", 5120},
      {"freq_dim", 256},
      {"time_embed_hidden_dim", 5376},
      {"time_embed_dim", 2688},
      {"rope_freq_dim", 16},
      {"rope_theta", 10000.0},
      {"norm_eps", 1e-5},
      {"qk_norm_eps", 1e-5},
      {"final_norm_eps", 1e-5},
  };
}

ModelArgs load_transformer_args(const std::string& class_name,
                                const nlohmann::json& config) {
  JsonReader reader;
  EXPECT_TRUE(reader.parse_text(config.dump()));
  ModelArgs args;
  const bool loaded =
      class_name == "MiniMaxH3DiTModel"
          ? MiniMaxH3TransformerConfig::load_original_model_args(reader, &args)
          : MiniMaxH3TransformerConfig::load_converted_model_args(reader,
                                                                  &args);
  EXPECT_TRUE(loaded);
  return args;
}

TEST(MiniMaxH3ConfigTest, MapsOriginalAndConvertedFieldNames) {
  const auto original =
      load_transformer_args("MiniMaxH3DiTModel", original_transformer_config());
  EXPECT_EQ(original.model_type(), "MiniMaxH3DiTModel");
  EXPECT_EQ(original.h3_num_layers(), 50);
  EXPECT_EQ(original.h3_token_refiner_num_layers(), 2);
  EXPECT_EQ(original.h3_hidden_size(), 5376);
  EXPECT_EQ(original.h3_num_attention_heads(), 56);
  EXPECT_EQ(original.h3_attention_head_dim(), 128);
  EXPECT_EQ(original.h3_ffn_hidden_size(), 14336);
  EXPECT_EQ(original.h3_video_latent_dim(), 24);
  EXPECT_EQ(original.h3_audio_latent_dim(), 32);
  EXPECT_EQ(original.h3_patch_size(), (std::vector<int64_t>{1, 2, 2}));
  EXPECT_EQ(original.h3_text_dim(), 5120);
  EXPECT_EQ(original.h3_timestep_input_dim(), 256);
  EXPECT_EQ(original.h3_time_embed_hidden_size(), 5376);
  EXPECT_EQ(original.h3_time_embed_dim(), 2688);
  EXPECT_EQ(original.h3_adaln_out_features(), 96768);
  EXPECT_EQ(original.h3_final_adaln_out_features(), 10752);
  EXPECT_EQ(original.h3_rope_inv_freq_len(), 16);
  EXPECT_DOUBLE_EQ(original.h3_rope_theta(), 10000.0);
  EXPECT_DOUBLE_EQ(original.h3_norm_eps(), 1e-5);
  EXPECT_DOUBLE_EQ(original.h3_qk_norm_eps(), 1e-5);
  EXPECT_DOUBLE_EQ(original.h3_final_norm_eps(), 1e-5);

  const auto converted = load_transformer_args("MiniMaxH3Transformer3DModel",
                                               converted_transformer_config());
  EXPECT_EQ(converted.model_type(), "MiniMaxH3Transformer3DModel");
  EXPECT_EQ(converted.h3_token_refiner_num_layers(), 2);
  EXPECT_EQ(converted.h3_ffn_hidden_size(), 14336);
  EXPECT_EQ(converted.h3_video_latent_dim(), 24);
  EXPECT_EQ(converted.h3_audio_latent_dim(), 32);
  EXPECT_EQ(converted.h3_timestep_input_dim(), 256);
  EXPECT_EQ(converted.h3_time_embed_hidden_size(), 5376);
  EXPECT_EQ(converted.h3_rope_inv_freq_len(), 16);
}

TEST(MiniMaxH3ConfigTest, RejectsMissingRequiredCheckpointField) {
  auto config = original_transformer_config();
  config.erase("hidden_size");
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(config.dump()));
  ModelArgs args;
  expect_invalid_argument_contains(
      [&] {
        MiniMaxH3TransformerConfig::load_original_model_args(reader, &args);
      },
      "missing required field `hidden_size`");
}

TEST(MiniMaxH3ConfigTest, RejectsEveryOriginalInvariantMismatch) {
  const std::vector<std::pair<std::string, nlohmann::json>> wrong_values = {
      {"_class_name", "OtherTransformer"},
      {"num_layers", 49},
      {"token_refiner_num_layers", 3},
      {"hidden_size", 4096},
      {"num_attention_heads", 48},
      {"attention_head_dim", 64},
      {"ffn_hidden_size", 14000},
      {"latents_dim", 16},
      {"audio_latents_dim", 16},
      {"patch_size", {2, 2, 2}},
      {"text_dim", 4096},
      {"timestep_input_dim", 128},
      {"time_embed_hidden_size", 4096},
      {"time_embed_dim", 2048},
      {"adaln_out_features", 100},
      {"final_adaln_out_features", 100},
      {"rope_inv_freq_len", 8},
      {"rope_theta", 5000.0},
      {"norm_eps", 1e-6},
      {"qk_norm_eps", 1e-6},
      {"final_norm_eps", 1e-6},
  };
  for (const auto& [field, wrong_value] : wrong_values) {
    SCOPED_TRACE(field);
    auto config = original_transformer_config();
    config[field] = wrong_value;
    JsonReader reader;
    ASSERT_TRUE(reader.parse_text(config.dump()));
    ModelArgs args;
    expect_invalid_argument_contains(
        [&] {
          MiniMaxH3TransformerConfig::load_original_model_args(reader, &args);
        },
        "invariant");
  }
}

TEST(MiniMaxH3ConfigTest, RejectsConvertedInvariantMismatch) {
  auto config = converted_transformer_config();
  config["num_refiner_layers"] = 1;
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(config.dump()));
  ModelArgs args;
  expect_invalid_argument_contains(
      [&] {
        MiniMaxH3TransformerConfig::load_converted_model_args(reader, &args);
      },
      "token_refiner_num_layers");
}

struct SyntheticLayoutOptions {
  std::optional<std::string> omitted_key;
  std::optional<std::string> wrong_shape_key;
  std::optional<std::string> wrong_dtype_key;
  bool add_unknown = false;
  bool add_duplicate = false;
};

std::vector<std::unique_ptr<StateDict>> synthetic_source_layout(
    const SyntheticLayoutOptions& options = {}) {
  constexpr size_t kShardCount = 4;
  std::vector<std::unordered_map<std::string, torch::Tensor>> tensors(
      kShardCount);
  const auto meta_device = c10::Device(c10::DeviceType::Meta);
  const auto specs = MiniMaxH3SourceLayoutValidator::expected_source_tensors();
  for (size_t index = 0; index < specs.size(); ++index) {
    const auto& spec = specs[index];
    if (options.omitted_key == spec.name) {
      continue;
    }
    auto shape = spec.shape;
    if (options.wrong_shape_key == spec.name) {
      ++shape.back();
    }
    const auto dtype =
        options.wrong_dtype_key == spec.name ? torch::kFloat16 : spec.dtype;
    tensors[index % kShardCount].emplace(
        spec.name,
        torch::empty(shape,
                     torch::TensorOptions().dtype(dtype).device(meta_device)));
  }
  if (options.add_unknown) {
    tensors[0].emplace("unknown.weight",
                       torch::empty({1},
                                    torch::TensorOptions()
                                        .dtype(torch::kBFloat16)
                                        .device(meta_device)));
  }
  if (options.add_duplicate) {
    const auto& spec = specs.front();
    tensors[1].emplace(
        spec.name,
        torch::empty(
            spec.shape,
            torch::TensorOptions().dtype(spec.dtype).device(meta_device)));
  }

  std::vector<std::unique_ptr<StateDict>> shards;
  shards.reserve(kShardCount);
  for (auto& shard : tensors) {
    shards.push_back(std::make_unique<StateDict>(std::move(shard)));
  }
  return shards;
}

TEST(MiniMaxH3SourceLayoutTest, ValidatesExact535TensorMetadataUnion) {
  const auto specs = MiniMaxH3SourceLayoutValidator::expected_source_tensors();
  ASSERT_EQ(specs.size(), 535);
  EXPECT_TRUE(std::any_of(specs.begin(), specs.end(), [](const auto& spec) {
    return spec.name == "rope.inv_freq" &&
           spec.shape == std::vector<int64_t>({16}) &&
           spec.dtype == torch::kFloat32;
  }));

  const auto shards = synthetic_source_layout();
  const auto summary = MiniMaxH3SourceLayoutValidator::validate(shards);
  EXPECT_EQ(summary.shard_count, 4);
  EXPECT_EQ(summary.tensor_count, 535);
  EXPECT_EQ(summary.bfloat16_tensor_count, 522);
  EXPECT_EQ(summary.float32_tensor_count, 13);
  EXPECT_EQ(summary.transformer_layer_count, 50);
  EXPECT_EQ(summary.token_refiner_layer_count, 2);
}

TEST(MiniMaxH3SourceLayoutTest, RejectsMissingRequiredRopeTensor) {
  const auto shards =
      synthetic_source_layout({.omitted_key = std::string("rope.inv_freq")});
  expect_invalid_argument_contains(
      [&] { MiniMaxH3SourceLayoutValidator::validate(shards); },
      "rope.inv_freq");
}

TEST(MiniMaxH3SourceLayoutTest, RejectsUnknownTensor) {
  const auto shards = synthetic_source_layout({.add_unknown = true});
  expect_invalid_argument_contains(
      [&] { MiniMaxH3SourceLayoutValidator::validate(shards); },
      "unknown.weight");
}

TEST(MiniMaxH3SourceLayoutTest, RejectsDuplicateAcrossShards) {
  const auto shards = synthetic_source_layout({.add_duplicate = true});
  expect_invalid_argument_contains(
      [&] { MiniMaxH3SourceLayoutValidator::validate(shards); },
      "duplicate tensor `video_patch_proj.weight`");
}

TEST(MiniMaxH3SourceLayoutTest, RejectsWrongShape) {
  const auto shards = synthetic_source_layout(
      {.wrong_shape_key = std::string("blocks.49.mlp.fc2.weight")});
  expect_invalid_argument_contains(
      [&] { MiniMaxH3SourceLayoutValidator::validate(shards); },
      "shape mismatch");
}

TEST(MiniMaxH3SourceLayoutTest, RejectsWrongDtype) {
  const auto shards = synthetic_source_layout(
      {.wrong_dtype_key = std::string("condition_proj.weight")});
  expect_invalid_argument_contains(
      [&] { MiniMaxH3SourceLayoutValidator::validate(shards); },
      "dtype mismatch");
}

TEST(MiniMaxH3SourceLayoutTest, ValidatesPinnedRealCheckpointWhenRequested) {
  const char* checkpoint_root = std::getenv("MINIMAX_H3_CHECKPOINT");
  if (checkpoint_root == nullptr || std::string(checkpoint_root).empty()) {
    GTEST_SKIP() << "Set MINIMAX_H3_CHECKPOINT for the pinned mmap audit";
  }
  auto loader = std::make_unique<DiTModelLoader>(std::string(checkpoint_root));
  EXPECT_EQ(loader->get_model_type(), "MiniMaxH3Pipeline");
  DiTModelContext context(
      ParallelArgs(0, 1, nullptr),
      loader->get_model_args(),
      loader->get_quant_args(),
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kBFloat16),
      DiTCacheConfig(),
      loader->get_model_type());
  MiniMaxH3Pipeline pipeline(context);
  pipeline->load_model(std::move(loader));
  ASSERT_TRUE(pipeline->is_loaded());
  ASSERT_TRUE(pipeline->source_layout_summary().has_value());
  const auto& summary = *pipeline->source_layout_summary();
  EXPECT_EQ(summary.shard_count, 13);
  EXPECT_EQ(summary.tensor_count, 535);
  EXPECT_EQ(summary.bfloat16_tensor_count, 522);
  EXPECT_EQ(summary.float32_tensor_count, 13);
}

nlohmann::json ref2va_model_index() {
  return {
      {"_class_name", "MiniMaxH3Pipeline"},
      {"_diffusers_version", "0.32.2"},
      {"text_encoder", {"transformers", "MiniMaxH3Qwen3VLHFEncoder"}},
      {"tokenizer", {"transformers", "Qwen2TokenizerFast"}},
      {"video_vae", {"diffusers", "MiniMaxH3VideoVAE"}},
      {"audio_vae", {"diffusers", "MiniMaxH3AudioVAE"}},
      {"scheduler", nullptr},
      {"transformer", {"diffusers", "MiniMaxH3DiTModel"}},
      {"processor", {"transformers", "Qwen3VLProcessor"}},
      {"_minimax_h3",
       {{"schema_version", 1},
        {"partition", "ref2va"},
        {"tasks", {"ref2va"}},
        {"task_aliases", nlohmann::json::object()},
        {"sigma_shift_scales", {{"video", 12.0}, {"audio", 3.0}}}}},
  };
}

TEST(MiniMaxH3ModelIndexTest, AcceptsExactRef2VAPartitionMetadata) {
  const auto summary = MiniMaxH3ModelIndexContract::parse(ref2va_model_index());
  EXPECT_EQ(summary.class_name, "MiniMaxH3Pipeline");
  EXPECT_EQ(summary.schema_version, 1);
  EXPECT_EQ(summary.partition, "ref2va");
  EXPECT_EQ(summary.tasks, (std::vector<std::string>{"ref2va"}));
  EXPECT_DOUBLE_EQ(summary.video_sigma_shift, 12.0);
  EXPECT_DOUBLE_EQ(summary.audio_sigma_shift, 3.0);
  EXPECT_EQ(summary.component_classes.size(), 6);
  EXPECT_EQ(summary.component_classes.at("transformer"), "MiniMaxH3DiTModel");
}

TEST(MiniMaxH3ModelIndexTest, RejectsWrongOrIncompleteMetadata) {
  struct Mutation {
    std::string expected;
    std::function<void(nlohmann::json&)> apply;
  };
  const std::vector<Mutation> mutations = {
      {"_class_name",
       [](auto& json) { json["_class_name"] = "OtherPipeline"; }},
      {"schema_version",
       [](auto& json) { json["_minimax_h3"]["schema_version"] = 2; }},
      {"partition",
       [](auto& json) { json["_minimax_h3"]["partition"] = "fl2va"; }},
      {"tasks", [](auto& json) { json["_minimax_h3"]["tasks"] = {"fl2va"}; }},
      {"video",
       [](auto& json) {
         json["_minimax_h3"]["sigma_shift_scales"]["video"] = 6.0;
       }},
      {"scheduler",
       [](auto& json) {
         json["scheduler"] = {"diffusers", "MiniMaxH3Scheduler"};
       }},
      {"missing required key `audio_vae`",
       [](auto& json) { json.erase("audio_vae"); }},
      {"unknown key `transformer_2`",
       [](auto& json) {
         json["transformer_2"] = {"diffusers", "MiniMaxH3DiTModel"};
       }},
      {"unsupported component descriptor",
       [](auto& json) {
         json["transformer"] = {"diffusers", "OtherTransformer"};
       }},
  };

  for (const auto& mutation : mutations) {
    SCOPED_TRACE(mutation.expected);
    auto model_index = ref2va_model_index();
    mutation.apply(model_index);
    expect_invalid_argument_contains(
        [&] { MiniMaxH3ModelIndexContract::parse(model_index); },
        mutation.expected);
  }
}

TEST(MiniMaxH3DryRunTest, TracesAlignedVideoAudioAndConditionGeometry) {
  const auto trace = MiniMaxH3PipelineImpl::dry_run_shape_trace(
      {.height = 768,
       .width = 1344,
       .requested_num_frames = 124,
       .condition_shape = {2, 77, 5120},
       .token_tags_shape = {2, 77}});
  EXPECT_EQ(trace.batch_size, 2);
  EXPECT_EQ(trace.condition_token_count, 77);
  EXPECT_EQ(trace.aligned_num_frames, 124);
  EXPECT_EQ(trace.video_latent_channels, 24);
  EXPECT_EQ(trace.video_latent_frames, 37);
  EXPECT_EQ(trace.video_latent_height, 48);
  EXPECT_EQ(trace.video_latent_width, 84);
  EXPECT_EQ(trace.video_rows_per_sample, 37296);
  EXPECT_EQ(trace.audio_channels, 2);
  EXPECT_EQ(trace.audio_latent_channels, 32);
  EXPECT_EQ(trace.audio_latents_per_channel, 207);
  EXPECT_EQ(trace.audio_rows_per_sample, 414);
  EXPECT_EQ(trace.minimum_packed_rows_per_sample, 37787);

  const auto four_seconds = MiniMaxH3PipelineImpl::dry_run_shape_trace(
      {.height = 768,
       .width = 1344,
       .requested_num_frames = 96,
       .condition_shape = {1, 1, 5120},
       .token_tags_shape = {1, 1}});
  EXPECT_EQ(four_seconds.aligned_num_frames, 107);
  EXPECT_EQ(four_seconds.video_latent_frames, 32);
  EXPECT_EQ(four_seconds.audio_latents_per_channel, 178);
}

TEST(MiniMaxH3DryRunTest, RejectsInvalidGeometryDurationAndConditionShapes) {
  MiniMaxH3DryRunShapeInput input{.height = 768,
                                  .width = 1344,
                                  .requested_num_frames = 124,
                                  .condition_shape = {1, 8, 5120},
                                  .token_tags_shape = {1, 8}};

  auto invalid = input;
  invalid.width = 1280;
  expect_invalid_argument_contains(
      [&] { MiniMaxH3PipelineImpl::dry_run_shape_trace(invalid); }, "1344x768");

  invalid = input;
  invalid.requested_num_frames = 95;
  expect_invalid_argument_contains(
      [&] { MiniMaxH3PipelineImpl::dry_run_shape_trace(invalid); },
      "between 4 and 15 seconds");

  invalid = input;
  invalid.requested_num_frames = 346;
  expect_invalid_argument_contains(
      [&] { MiniMaxH3PipelineImpl::dry_run_shape_trace(invalid); },
      "alignment");

  invalid = input;
  invalid.condition_shape = {1, 8, 4096};
  expect_invalid_argument_contains(
      [&] { MiniMaxH3PipelineImpl::dry_run_shape_trace(invalid); },
      "[B,N,5120]");

  invalid = input;
  invalid.token_tags_shape = {1, 7};
  expect_invalid_argument_contains(
      [&] { MiniMaxH3PipelineImpl::dry_run_shape_trace(invalid); }, "[B,N]");
}

torch::Tensor h3_condition_hidden(int64_t tokens) {
  return torch::arange(tokens * 5120, torch::kFloat32)
      .remainder(257)
      .sub(128)
      .reshape({1, tokens, 5120})
      .to(torch::kBFloat16)
      .contiguous();
}

H3TargetLatents h3_target() {
  return {.audio_t = 2,
          .audio_channels = 2,
          .latent_t = 2,
          .latent_h = 4,
          .latent_w = 6};
}

H3ReferenceBlock h3_image(int64_t height = 4, int64_t width = 4) {
  return {.kind = H3ReferenceBlockKind::IMAGE,
          .latent_h = height,
          .latent_w = width};
}

H3ReferenceBlock h3_audio(int64_t temporal = 2) {
  return {.kind = H3ReferenceBlockKind::AUDIO, .ref_audio_t = temporal};
}

H3ReferenceBlock h3_video(int64_t temporal = 3,
                          int64_t height = 4,
                          int64_t width = 4) {
  return {.kind = H3ReferenceBlockKind::VIDEO,
          .ref_audio_t = 0,
          .latent_t = temporal,
          .latent_h = height,
          .latent_w = width};
}

H3ReferenceBlock h3_video_audio(int64_t audio_temporal = 2,
                                int64_t video_temporal = 3,
                                int64_t height = 4,
                                int64_t width = 4) {
  return {.kind = H3ReferenceBlockKind::VIDEO_AUDIO,
          .ref_audio_t = audio_temporal,
          .latent_t = video_temporal,
          .latent_h = height,
          .latent_w = width};
}

H3PackedLayout h3_layout(
    const std::vector<H3ReferenceBlock>& references,
    const std::vector<int64_t>& tags = {0, 1, 1},
    std::optional<int64_t> sequence_length = std::nullopt) {
  const torch::Tensor condition_tags =
      torch::tensor(tags, torch::kInt64).reshape({1, -1});
  return minimax_h3_build_ref2va_packed_layout(
      h3_condition_hidden(static_cast<int64_t>(tags.size())),
      condition_tags,
      h3_target(),
      references,
      sequence_length);
}

void expect_slice(const H3Slice& slice, int64_t start, int64_t stop) {
  EXPECT_EQ(slice.start, start);
  EXPECT_EQ(slice.stop, stop);
}

bool within_one_ulp(double actual, double expected) {
  return actual == expected || actual == std::nextafter(expected, -INFINITY) ||
         actual == std::nextafter(expected, INFINITY);
}

void expect_positions_within_one_ulp(
    const torch::Tensor& actual,
    const std::vector<std::array<double, 3>>& expected) {
  ASSERT_EQ(actual.scalar_type(), torch::kFloat64);
  ASSERT_TRUE(actual.device().is_cpu());
  ASSERT_EQ(actual.dim(), 2);
  ASSERT_EQ(actual.size(0), static_cast<int64_t>(expected.size()));
  ASSERT_EQ(actual.size(1), 3);
  const auto values = actual.accessor<double, 2>();
  for (int64_t row = 0; row < actual.size(0); ++row) {
    for (int64_t axis = 0; axis < 3; ++axis) {
      SCOPED_TRACE("row=" + std::to_string(row) +
                   " axis=" + std::to_string(axis));
      EXPECT_TRUE(within_one_ulp(values[row][axis],
                                 expected[static_cast<size_t>(row)][axis]));
    }
  }
}

TEST(MiniMaxH3PackedTokensTest, PatchifyAndAudioPackUseGoldenRowOrder) {
  const torch::Tensor video =
      torch::arange(32, torch::kInt64).reshape({1, 2, 2, 2, 4}).contiguous();
  const torch::Tensor expected_video_rows =
      torch::tensor(
          {0, 1, 4,  5,  16, 17, 20, 21, 2,  3,  6,  7,  18, 19, 22, 23,
           8, 9, 12, 13, 24, 25, 28, 29, 10, 11, 14, 15, 26, 27, 30, 31},
          torch::kInt64)
          .reshape({4, 8});
  const torch::Tensor video_rows = minimax_h3_patchify_video_latent(video);
  EXPECT_TRUE(torch::equal(video_rows, expected_video_rows));
  EXPECT_TRUE(video_rows.is_contiguous());
  EXPECT_TRUE(torch::equal(
      minimax_h3_unpatchify_video_tokens(
          video_rows, {.channels = 2, .temporal = 2, .height = 2, .width = 4}),
      video));

  const torch::Tensor audio =
      torch::arange(24, torch::kInt64).reshape({2, 3, 4}).contiguous();
  const torch::Tensor expected_audio_rows =
      torch::tensor({0,  4,  8,  1,  5,  9,  2,  6,  10, 3,  7,  11,
                     12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23},
                    torch::kInt64)
          .reshape({8, 3});
  const torch::Tensor audio_rows = minimax_h3_pack_audio_latent(audio);
  EXPECT_TRUE(torch::equal(audio_rows, expected_audio_rows));
  EXPECT_TRUE(audio_rows.is_contiguous());
  EXPECT_TRUE(
      torch::equal(minimax_h3_unpack_audio_tokens(
                       audio_rows, /*audio_channels=*/2, /*temporal=*/4),
                   audio));
}

const torch::Tensor* find_raw_tensor(const StateDict& state_dict,
                                     const std::string& key) {
  for (const auto& [name, tensor] : state_dict) {
    if (name == key) {
      return &tensor;
    }
  }
  return nullptr;
}

void expect_golden_tensor(const StateDict& golden,
                          const std::string& key,
                          const torch::Tensor& actual,
                          bool allow_one_ulp = false) {
  SCOPED_TRACE(key);
  const torch::Tensor* expected = find_raw_tensor(golden, key);
  ASSERT_NE(expected, nullptr) << "Golden tensor key is missing";
  ASSERT_TRUE(actual.defined());
  ASSERT_EQ(actual.scalar_type(), expected->scalar_type())
      << "Golden tensor dtype mismatch";
  ASSERT_EQ(actual.sizes().vec(), expected->sizes().vec())
      << "Golden tensor shape mismatch";
  ASSERT_TRUE(actual.device().is_cpu());
  ASSERT_TRUE(expected->device().is_cpu());
  if (!allow_one_ulp) {
    EXPECT_TRUE(torch::equal(actual, *expected));
    return;
  }

  ASSERT_EQ(actual.scalar_type(), torch::kFloat64);
  const torch::Tensor actual_values = actual.contiguous();
  const torch::Tensor expected_values = expected->contiguous();
  const double* actual_data = actual_values.data_ptr<double>();
  const double* expected_data = expected_values.data_ptr<double>();
  for (int64_t index = 0; index < actual_values.numel(); ++index) {
    SCOPED_TRACE("element=" + std::to_string(index));
    EXPECT_TRUE(within_one_ulp(actual_data[index], expected_data[index]))
        << "actual=" << actual_data[index]
        << " expected=" << expected_data[index];
  }
}

struct H3GoldenCase {
  std::string name;
  std::vector<int64_t> tags;
  std::vector<H3ReferenceBlock> references;
  std::optional<int64_t> sequence_length;
};

TEST(MiniMaxH3PackingGoldenTest, MatchesPinnedVllmOmniReferenceWhenRequested) {
  const char* golden_path = std::getenv("MINIMAX_H3_PACKING_GOLDEN");
  if (golden_path == nullptr || std::string(golden_path).empty()) {
    GTEST_SKIP() << "Set MINIMAX_H3_PACKING_GOLDEN to the generated "
                    "safetensors file";
  }
  const std::unique_ptr<StateDict> golden =
      StateDictFromSafeTensor::load(golden_path);
  ASSERT_NE(golden, nullptr);
  ASSERT_EQ(golden->size(), 91);

  const std::vector<H3GoldenCase> cases = {
      {.name = "one_image", .tags = {0, 1, 1}, .references = {h3_image()}},
      {.name = "image_audio",
       .tags = {1, 0, 1},
       .references = {h3_image(), h3_audio()}},
      {.name = "video", .tags = {1, 1, 0}, .references = {h3_video()}},
      {.name = "video_audio",
       .tags = {0, 1, 0},
       .references = {h3_video_audio()},
       .sequence_length = 128},
      {.name = "mixed",
       .tags = {0, 1, 0, 1},
       .references = {h3_image(/*height=*/2, /*width=*/4),
                      h3_audio(/*temporal=*/3),
                      h3_video(/*temporal=*/2, /*height=*/2, /*width=*/4),
                      h3_video_audio(/*audio_temporal=*/2,
                                     /*video_temporal=*/3,
                                     /*height=*/4,
                                     /*width=*/2)}}};

  for (const H3GoldenCase& golden_case : cases) {
    SCOPED_TRACE(golden_case.name);
    const torch::Tensor condition_hidden =
        h3_condition_hidden(static_cast<int64_t>(golden_case.tags.size()));
    const torch::Tensor condition_tags =
        torch::tensor(golden_case.tags, torch::kInt64).reshape({1, -1});
    const H3PackedLayout layout =
        minimax_h3_build_ref2va_packed_layout(condition_hidden,
                                              condition_tags,
                                              h3_target(),
                                              golden_case.references,
                                              golden_case.sequence_length);
    const torch::Tensor seq_len =
        torch::tensor({layout.aligned_length}, torch::kInt64).reshape({});
    const torch::Tensor latent_grid =
        torch::tensor({layout.target_latent_grid.temporal,
                       layout.target_latent_grid.height,
                       layout.target_latent_grid.width},
                      torch::kInt64);
    const torch::Tensor video_row_start =
        torch::tensor({layout.target_video_slice.start}, torch::kInt64)
            .reshape({});
    const std::vector<std::pair<std::string, torch::Tensor>> fields = {
        {"seq_len", seq_len},
        {"condition_hidden", layout.condition_hidden},
        {"condition_tags", condition_tags},
        {"input_ids", layout.input_ids},
        {"image_mask", layout.image_mask},
        {"audio_mask", layout.audio_mask},
        {"img_pos", layout.img_pos},
        {"audio_pos", layout.audio_pos},
        {"text_pos", layout.text_pos},
        {"update_mask", layout.update_mask},
        {"audio_update_mask", layout.audio_update_mask},
        {"position_ids", layout.position_ids},
        {"token_tags", layout.token_tags},
        {"cu_seqlens", layout.cu_seqlens},
        {"document_id", layout.document_id},
        {"latent_grid", latent_grid},
        {"video_row_start", video_row_start}};
    for (const auto& [field, actual] : fields) {
      expect_golden_tensor(*golden,
                           golden_case.name + "." + field,
                           actual,
                           field == "position_ids");
    }
    EXPECT_EQ(seq_len.item<int64_t>(), layout.aligned_length);
  }

  const torch::Tensor video_latent =
      torch::arange(2 * 3 * 2 * 4 * 6, torch::kFloat32)
          .reshape({2, 3, 2, 4, 6});
  const torch::Tensor video_rows =
      minimax_h3_patchify_video_latent(video_latent);
  const torch::Tensor video_roundtrip = minimax_h3_unpatchify_video_tokens(
      video_rows, {.channels = 3, .temporal = 2, .height = 4, .width = 6});
  expect_golden_tensor(*golden, "transforms.video_latent", video_latent);
  expect_golden_tensor(*golden, "transforms.video_rows", video_rows);
  expect_golden_tensor(*golden, "transforms.video_roundtrip", video_roundtrip);

  const torch::Tensor audio_latent =
      torch::arange(2 * 3 * 4, torch::kFloat32).reshape({2, 3, 4});
  const torch::Tensor audio_rows = minimax_h3_pack_audio_latent(audio_latent);
  const torch::Tensor audio_roundtrip = minimax_h3_unpack_audio_tokens(
      audio_rows, /*audio_channels=*/2, /*temporal=*/4);
  expect_golden_tensor(*golden, "transforms.audio_latent", audio_latent);
  expect_golden_tensor(*golden, "transforms.audio_rows", audio_rows);
  expect_golden_tensor(*golden, "transforms.audio_roundtrip", audio_roundtrip);
}

TEST(MiniMaxH3PackingGoldenTest, ConsumesAttestedRealConditionBundle) {
  const char* bundle_path = std::getenv("MINIMAX_H3_CONDITION_BUNDLE");
  if (bundle_path == nullptr || std::string(bundle_path).empty()) {
    GTEST_SKIP() << "Set MINIMAX_H3_CONDITION_BUNDLE to condition.safetensors";
  }
  const auto bundle = StateDictFromSafeTensor::load(bundle_path);
  ASSERT_NE(bundle, nullptr);
  const torch::Tensor hidden = bundle->get_tensor("prompt_embeds");
  const torch::Tensor tags = bundle->get_tensor("text_token_tags");
  ASSERT_TRUE(hidden.defined());
  ASSERT_TRUE(tags.defined());

  DiTForwardInput input;
  input.prompt_embeds = hidden.unsqueeze(0).contiguous();
  input.text_token_tags = tags.unsqueeze(0).contiguous();
  const H3PackedLayout layout =
      MiniMaxH3PipelineImpl::dry_run_packed_layout(input,
                                                   {.audio_t = 207,
                                                    .audio_channels = 2,
                                                    .latent_t = 37,
                                                    .latent_h = 48,
                                                    .latent_w = 84},
                                                   {h3_image()});

  EXPECT_EQ(layout.text_slice.stop, 11350);
  EXPECT_EQ(layout.used_length, 49064);
  EXPECT_EQ(layout.aligned_length, 49088);
  EXPECT_TRUE(torch::equal(layout.token_tags.slice(0, 0, 11350), tags));
  EXPECT_TRUE(
      layout.token_tags.slice(0, 49064, 49088).eq(-1).all().item<bool>());
  EXPECT_TRUE(
      layout.document_id.slice(0, 49064, 49088).eq(1).all().item<bool>());
  EXPECT_TRUE(torch::equal(layout.cu_seqlens,
                           torch::tensor({0, 49064, 49088}, torch::kInt32)));
}

TEST(MiniMaxH3PackedTokensTest, RejectsInvalidRanksDivisibilityAndRows) {
  expect_invalid_argument_contains(
      [] { minimax_h3_patchify_video_latent(torch::zeros({1, 2, 3, 4})); },
      "rank 5");
  expect_invalid_argument_contains(
      [] { minimax_h3_patchify_video_latent(torch::zeros({1, 2, 1, 3, 4})); },
      "divisible");
  expect_invalid_argument_contains(
      [] {
        minimax_h3_unpatchify_video_tokens(
            torch::zeros({4, 7}),
            {.channels = 2, .temporal = 2, .height = 2, .width = 4});
      },
      "row dimension");
  expect_invalid_argument_contains(
      [] {
        minimax_h3_unpatchify_video_tokens(
            torch::zeros({3, 8}),
            {.channels = 2, .temporal = 2, .height = 2, .width = 4});
      },
      "row count");
  expect_invalid_argument_contains(
      [] { minimax_h3_pack_audio_latent(torch::zeros({2, 3})); }, "rank 3");
  expect_invalid_argument_contains(
      [] {
        minimax_h3_unpack_audio_tokens(
            torch::zeros({7, 3}), /*audio_channels=*/2, /*temporal=*/4);
      },
      "row count");
}

TEST(MiniMaxH3PackingTest, OneImageMatchesAllGoldenFields) {
  DiTForwardInput input;
  input.prompt_embeds = h3_condition_hidden(/*tokens=*/3);
  input.text_token_tags = torch::tensor({{0, 1, 1}}, torch::kInt64);
  const H3PackedLayout layout = MiniMaxH3PipelineImpl::dry_run_packed_layout(
      input, h3_target(), {h3_image()});

  EXPECT_TRUE(layout.condition_hidden.is_same(input.prompt_embeds));
  EXPECT_EQ(layout.used_length, 23);
  EXPECT_EQ(layout.aligned_length, 64);
  expect_slice(layout.text_slice, 0, 3);
  expect_slice(layout.target_audio_slice, 7, 11);
  expect_slice(layout.target_video_slice, 11, 23);
  expect_slice(layout.padding_slice, 23, 64);
  EXPECT_EQ(layout.target_temporal_origin, 4.0);
  EXPECT_EQ(layout.target_latent_grid.temporal, 2);
  EXPECT_EQ(layout.target_latent_grid.height, 2);
  EXPECT_EQ(layout.target_latent_grid.width, 3);

  std::vector<int64_t> expected_ids(64, kMiniMaxH3PadId);
  std::fill(expected_ids.begin(), expected_ids.begin() + 3, kMiniMaxH3TextId);
  std::fill(expected_ids.begin() + 3,
            expected_ids.begin() + 7,
            kMiniMaxH3ImageVideoConditionId);
  std::fill(
      expected_ids.begin() + 7, expected_ids.begin() + 11, kMiniMaxH3AudioId);
  expected_ids[7] = kMiniMaxH3AudioFirstId;
  std::fill(
      expected_ids.begin() + 11, expected_ids.begin() + 23, kMiniMaxH3VideoId);
  expected_ids[11] = kMiniMaxH3VideoFirstId;
  expected_ids[22] = kMiniMaxH3VideoLastId;
  EXPECT_TRUE(torch::equal(layout.input_ids,
                           torch::tensor(expected_ids, torch::kInt64)));

  const torch::Tensor expected_img_pos = torch::tensor(
      {3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22},
      torch::kInt64);
  const torch::Tensor expected_audio_pos =
      torch::tensor({7, 8, 9, 10}, torch::kInt64);
  EXPECT_TRUE(torch::equal(layout.img_pos, expected_img_pos));
  EXPECT_TRUE(torch::equal(layout.audio_pos, expected_audio_pos));
  EXPECT_TRUE(
      torch::equal(layout.text_pos, torch::tensor({0, 1, 2}, torch::kInt64)));
  EXPECT_TRUE(torch::equal(torch::nonzero(layout.image_mask).flatten(),
                           expected_img_pos));
  EXPECT_TRUE(torch::equal(torch::nonzero(layout.audio_mask).flatten(),
                           expected_audio_pos));
  EXPECT_FALSE(layout.update_mask.slice(0, 0, 4).any().item<bool>());
  EXPECT_TRUE(layout.update_mask.slice(0, 4, 16).all().item<bool>());
  EXPECT_TRUE(layout.audio_update_mask.all().item<bool>());

  std::vector<std::array<double, 3>> expected_positions = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {2.0, 0.0, 0.0},
      {3.0, 0.0, 0.0},
      {3.0, 0.0, 16.0},
      {3.0, 16.0, 0.0},
      {3.0, 16.0, 16.0},
      {4.0, 0.0, -3.5959179422654266},
      {5.0, 0.0, -3.5959179422654266},
      {4.0, 0.0, 22.531972647421814},
      {5.0, 0.0, 22.531972647421814}};
  const std::array<double, 2> target_height = {2.9360547051563817, 16.0};
  const std::array<double, 3> target_width = {
      -3.5959179422654266, 9.468027352578194, 22.531972647421814};
  for (double temporal : {4.0, 5.666666666666667}) {
    for (double height : target_height) {
      for (double width : target_width) {
        expected_positions.emplace_back(
            std::array<double, 3>{temporal, height, width});
      }
    }
  }
  expect_positions_within_one_ulp(
      layout.position_ids.slice(0, 0, layout.used_length), expected_positions);
  EXPECT_EQ(layout.position_ids[3][0].item<double>(), 3.0);
  EXPECT_EQ(layout.position_ids[7][0].item<double>(), 4.0);
  EXPECT_EQ(layout.position_ids[11][0].item<double>(), 4.0);
  EXPECT_EQ(layout.position_ids[17][0].item<double>(), 5.666666666666667);
  EXPECT_EQ(
      layout.position_ids.slice(0, 23, 64).count_nonzero().item<int64_t>(), 0);

  std::vector<int64_t> expected_tags(64, -1);
  expected_tags[0] = 0;
  expected_tags[1] = 1;
  expected_tags[2] = 1;
  std::fill(expected_tags.begin() + 3, expected_tags.begin() + 7, 0);
  std::fill(expected_tags.begin() + 7, expected_tags.begin() + 11, 2);
  std::fill(expected_tags.begin() + 11, expected_tags.begin() + 23, 0);
  EXPECT_TRUE(torch::equal(layout.token_tags,
                           torch::tensor(expected_tags, torch::kInt64)));
  EXPECT_EQ(layout.cu_seqlens.scalar_type(), torch::kInt32);
  EXPECT_TRUE(torch::equal(layout.cu_seqlens,
                           torch::tensor({0, 23, 64}, torch::kInt32)));
  EXPECT_EQ(layout.document_id.scalar_type(), torch::kInt32);
  EXPECT_EQ(layout.document_id.slice(0, 0, 23).count_nonzero().item<int64_t>(),
            0);
  EXPECT_TRUE(layout.document_id.slice(0, 23, 64).eq(1).all().item<bool>());

  ASSERT_EQ(layout.reference_blocks.size(), 1);
  const H3PhysicalReferenceBlock& image = layout.reference_blocks[0];
  EXPECT_EQ(image.packed_offset, 3);
  expect_slice(image.row_slice, 3, 7);
  EXPECT_FALSE(image.audio_slice.has_value());
  ASSERT_TRUE(image.visual_slice.has_value());
  expect_slice(*image.visual_slice, 3, 7);
  EXPECT_EQ(image.temporal_origin, 3.0);
  EXPECT_EQ(image.temporal_extent, 1.0);
  ASSERT_EQ(layout.video_spans.size(), 1);
  EXPECT_EQ(layout.video_spans[0].role, H3VideoSpanRole::TARGET);
  EXPECT_EQ(layout.video_spans[0].start, 11);
}

TEST(MiniMaxH3PackingTest, ImageAndStandaloneAudioAdvanceInPhysicalOrder) {
  const H3PackedLayout layout = h3_layout({h3_image(), h3_audio()});

  EXPECT_EQ(layout.used_length, 27);
  ASSERT_EQ(layout.reference_blocks.size(), 2);
  expect_slice(layout.reference_blocks[0].row_slice, 3, 7);
  expect_slice(*layout.reference_blocks[0].visual_slice, 3, 7);
  EXPECT_EQ(layout.reference_blocks[0].temporal_origin, 3.0);
  expect_slice(layout.reference_blocks[1].row_slice, 7, 11);
  expect_slice(*layout.reference_blocks[1].audio_slice, 7, 11);
  EXPECT_EQ(layout.reference_blocks[1].temporal_origin, 4.0);
  EXPECT_EQ(layout.reference_blocks[1].temporal_extent, 2.0);
  expect_slice(layout.target_audio_slice, 11, 15);
  expect_slice(layout.target_video_slice, 15, 27);
  EXPECT_EQ(layout.target_temporal_origin, 6.0);
  EXPECT_TRUE(layout.input_ids.slice(0, 7, 11)
                  .eq(kMiniMaxH3AudioReferenceConditionId)
                  .all()
                  .item<bool>());
  EXPECT_TRUE(torch::equal(
      layout.audio_pos,
      torch::tensor({7, 8, 9, 10, 11, 12, 13, 14}, torch::kInt64)));
  EXPECT_FALSE(layout.audio_update_mask.slice(0, 0, 4).any().item<bool>());
  EXPECT_TRUE(layout.audio_update_mask.slice(0, 4, 8).all().item<bool>());
  EXPECT_EQ(layout.position_ids[7][0].item<double>(), 4.0);
  EXPECT_EQ(layout.position_ids[8][0].item<double>(), 5.0);
  EXPECT_EQ(layout.position_ids[9][0].item<double>(), 4.0);
  EXPECT_EQ(layout.position_ids[11][0].item<double>(), 6.0);
  EXPECT_TRUE(within_one_ulp(layout.position_ids[7][2].item<double>(),
                             -3.5959179422654266));
  EXPECT_TRUE(within_one_ulp(layout.position_ids[9][2].item<double>(),
                             22.531972647421814));
  ASSERT_EQ(layout.video_spans.size(), 1);
  EXPECT_EQ(layout.video_spans[0].role, H3VideoSpanRole::TARGET);
}

TEST(MiniMaxH3PackingTest, VideoUsesSequentialSpanAndHasNoAudioRows) {
  const H3PackedLayout layout = h3_layout({h3_video()});

  EXPECT_EQ(layout.used_length, 31);
  ASSERT_EQ(layout.reference_blocks.size(), 1);
  const H3PhysicalReferenceBlock& video = layout.reference_blocks[0];
  ASSERT_TRUE(video.audio_slice.has_value());
  ASSERT_TRUE(video.visual_slice.has_value());
  expect_slice(*video.audio_slice, 3, 3);
  expect_slice(*video.visual_slice, 3, 15);
  expect_slice(video.row_slice, 3, 15);
  EXPECT_EQ(video.temporal_origin, 3.0);
  EXPECT_EQ(video.temporal_extent, 15.0);
  EXPECT_EQ(layout.target_temporal_origin, 18.0);
  EXPECT_EQ(layout.position_ids[3][0].item<double>(), 3.0);
  EXPECT_EQ(layout.position_ids[7][0].item<double>(), 4.666666666666667);
  EXPECT_EQ(layout.position_ids[11][0].item<double>(), 11.333333333333334);
  EXPECT_EQ(layout.position_ids[15][0].item<double>(), 18.0);
  EXPECT_TRUE(torch::equal(layout.audio_pos,
                           torch::tensor({15, 16, 17, 18}, torch::kInt64)));
  ASSERT_EQ(layout.video_spans.size(), 2);
  EXPECT_EQ(layout.video_spans[0].start, 3);
  EXPECT_EQ(layout.video_spans[0].latent_grid.temporal, 3);
  EXPECT_EQ(layout.video_spans[0].latent_grid.height, 2);
  EXPECT_EQ(layout.video_spans[0].latent_grid.width, 2);
  EXPECT_EQ(layout.video_spans[0].role, H3VideoSpanRole::REFERENCE);
  EXPECT_EQ(layout.video_spans[1].start, 19);
  EXPECT_EQ(layout.video_spans[1].role, H3VideoSpanRole::TARGET);
}

TEST(MiniMaxH3PackingTest, VideoAudioSharesOriginAndHonorsExplicitAlignment) {
  const H3PackedLayout layout =
      h3_layout({h3_video_audio()}, {0, 1, 0}, /*sequence_length=*/128);

  EXPECT_EQ(layout.used_length, 35);
  EXPECT_EQ(layout.aligned_length, 128);
  const H3PhysicalReferenceBlock& video_audio = layout.reference_blocks[0];
  expect_slice(*video_audio.audio_slice, 3, 7);
  expect_slice(*video_audio.visual_slice, 7, 19);
  expect_slice(video_audio.row_slice, 3, 19);
  EXPECT_EQ(video_audio.temporal_origin, 3.0);
  EXPECT_EQ(layout.position_ids[3][0].item<double>(), 3.0);
  EXPECT_EQ(layout.position_ids[5][0].item<double>(), 3.0);
  EXPECT_EQ(layout.position_ids[7][0].item<double>(), 3.0);
  EXPECT_EQ(layout.target_temporal_origin, 18.0);
  expect_slice(layout.target_audio_slice, 19, 23);
  expect_slice(layout.target_video_slice, 23, 35);
  expect_slice(layout.padding_slice, 35, 128);
  EXPECT_TRUE(layout.input_ids.slice(0, 3, 7)
                  .eq(kMiniMaxH3AudioReferenceConditionId)
                  .all()
                  .item<bool>());
  EXPECT_TRUE(layout.input_ids.slice(0, 7, 19)
                  .eq(kMiniMaxH3ImageVideoConditionId)
                  .all()
                  .item<bool>());
  EXPECT_TRUE(torch::equal(layout.cu_seqlens,
                           torch::tensor({0, 35, 128}, torch::kInt32)));
  EXPECT_TRUE(layout.document_id.slice(0, 35, 128).eq(1).all().item<bool>());
  EXPECT_TRUE(layout.token_tags.slice(0, 35, 128).eq(-1).all().item<bool>());
  EXPECT_EQ(
      layout.position_ids.slice(0, 35, 128).count_nonzero().item<int64_t>(), 0);
  ASSERT_EQ(layout.video_spans.size(), 2);
  EXPECT_EQ(layout.video_spans[0].start, 7);
  EXPECT_EQ(layout.video_spans[1].start, 23);
}

TEST(MiniMaxH3PackingTest, MixedReferencesPreserveSlicesOriginsAndVideoSpans) {
  const std::vector<H3ReferenceBlock> references = {
      h3_image(/*height=*/2, /*width=*/4),
      h3_audio(/*temporal=*/3),
      h3_video(/*temporal=*/2, /*height=*/2, /*width=*/4),
      h3_video_audio(/*audio_temporal=*/2,
                     /*video_temporal=*/3,
                     /*height=*/4,
                     /*width=*/2)};
  const H3PackedLayout layout = h3_layout(references, {0, 1, 0, 1});

  EXPECT_EQ(layout.used_length, 42);
  ASSERT_EQ(layout.reference_blocks.size(), 4);
  expect_slice(layout.reference_blocks[0].row_slice, 4, 6);
  expect_slice(layout.reference_blocks[1].row_slice, 6, 12);
  expect_slice(layout.reference_blocks[2].row_slice, 12, 16);
  expect_slice(*layout.reference_blocks[2].audio_slice, 12, 12);
  expect_slice(*layout.reference_blocks[2].visual_slice, 12, 16);
  expect_slice(layout.reference_blocks[3].row_slice, 16, 26);
  expect_slice(*layout.reference_blocks[3].audio_slice, 16, 20);
  expect_slice(*layout.reference_blocks[3].visual_slice, 20, 26);
  EXPECT_EQ(layout.reference_blocks[0].temporal_origin, 4.0);
  EXPECT_EQ(layout.reference_blocks[1].temporal_origin, 5.0);
  EXPECT_EQ(layout.reference_blocks[2].temporal_origin, 8.0);
  EXPECT_EQ(layout.reference_blocks[3].temporal_origin, 16.333333333333336);
  EXPECT_EQ(layout.target_temporal_origin, 31.333333333333336);
  EXPECT_EQ(layout.position_ids[14][0].item<double>(), 9.666666666666666);
  EXPECT_EQ(layout.position_ids[20][0].item<double>(), 16.333333333333336);
  EXPECT_EQ(layout.position_ids[22][0].item<double>(), 18.000000000000004);
  EXPECT_EQ(layout.position_ids[26][0].item<double>(), 31.333333333333336);
  expect_slice(layout.target_audio_slice, 26, 30);
  expect_slice(layout.target_video_slice, 30, 42);
  EXPECT_TRUE(torch::equal(
      layout.img_pos,
      torch::tensor({4,  5,  12, 13, 14, 15, 20, 21, 22, 23, 24, 25,
                     30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41},
                    torch::kInt64)));
  EXPECT_TRUE(torch::equal(
      layout.audio_pos,
      torch::tensor({6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 26, 27, 28, 29},
                    torch::kInt64)));
  ASSERT_EQ(layout.video_spans.size(), 3);
  EXPECT_EQ(layout.video_spans[0].start, 12);
  EXPECT_EQ(layout.video_spans[0].latent_grid.temporal, 2);
  EXPECT_EQ(layout.video_spans[0].latent_grid.height, 1);
  EXPECT_EQ(layout.video_spans[0].latent_grid.width, 2);
  EXPECT_EQ(layout.video_spans[1].start, 20);
  EXPECT_EQ(layout.video_spans[1].latent_grid.temporal, 3);
  EXPECT_EQ(layout.video_spans[1].latent_grid.height, 2);
  EXPECT_EQ(layout.video_spans[1].latent_grid.width, 1);
  EXPECT_EQ(layout.video_spans[2].start, 30);
  EXPECT_EQ(layout.video_spans[2].role, H3VideoSpanRole::TARGET);
}

TEST(MiniMaxH3PackingTest, RejectsInvalidReferenceAndTargetSemantics) {
  expect_invalid_argument_contains([] { h3_layout({}); },
                                   "at least one visual reference");
  expect_invalid_argument_contains([] { h3_layout({h3_audio()}); },
                                   "audio-only");

  H3ReferenceBlock invalid = h3_image();
  invalid.kind = static_cast<H3ReferenceBlockKind>(99);
  expect_invalid_argument_contains([&] { h3_layout({invalid}); },
                                   "kind is unsupported");

  invalid = h3_image();
  invalid.latent_h = 0;
  expect_invalid_argument_contains([&] { h3_layout({invalid}); }, "positive");
  invalid = h3_image();
  invalid.latent_w = -2;
  expect_invalid_argument_contains([&] { h3_layout({invalid}); }, "positive");
  invalid = h3_image();
  invalid.latent_h = 3;
  expect_invalid_argument_contains([&] { h3_layout({invalid}); },
                                   "divisible by 2");
  invalid = h3_video();
  invalid.latent_t = 0;
  expect_invalid_argument_contains([&] { h3_layout({invalid}); }, "positive");
  invalid = h3_audio(/*temporal=*/0);
  expect_invalid_argument_contains([&] { h3_layout({h3_image(), invalid}); },
                                   "positive");

  invalid = h3_video();
  invalid.ref_audio_t = 1;
  expect_invalid_argument_contains([&] { h3_layout({invalid}); },
                                   "must be 0 for video kind");
  invalid = h3_video_audio();
  invalid.ref_audio_t = 0;
  expect_invalid_argument_contains([&] { h3_layout({invalid}); }, "positive");

  H3TargetLatents target = h3_target();
  target.audio_channels = 1;
  expect_invalid_argument_contains(
      [&] {
        minimax_h3_build_ref2va_packed_layout(
            h3_condition_hidden(3),
            torch::tensor({{0, 1, 1}}, torch::kInt64),
            target,
            {h3_image()});
      },
      "stereo");
  target = h3_target();
  target.audio_t = -1;
  expect_invalid_argument_contains(
      [&] {
        minimax_h3_build_ref2va_packed_layout(
            h3_condition_hidden(3),
            torch::tensor({{0, 1, 1}}, torch::kInt64),
            target,
            {h3_image()});
      },
      "positive");
  target = h3_target();
  target.latent_t = 0;
  expect_invalid_argument_contains(
      [&] {
        minimax_h3_build_ref2va_packed_layout(
            h3_condition_hidden(3),
            torch::tensor({{0, 1, 1}}, torch::kInt64),
            target,
            {h3_image()});
      },
      "positive");
  target = h3_target();
  target.latent_w = 5;
  expect_invalid_argument_contains(
      [&] {
        minimax_h3_build_ref2va_packed_layout(
            h3_condition_hidden(3),
            torch::tensor({{0, 1, 1}}, torch::kInt64),
            target,
            {h3_image()});
      },
      "divisible by 2");
}

TEST(MiniMaxH3PackingTest, RejectsBadConditionAbiAndSequenceLength) {
  const torch::Tensor hidden = h3_condition_hidden(/*tokens=*/3);
  const torch::Tensor tags = torch::tensor({{0, 1, 1}}, torch::kInt64);
  const auto build = [&](const torch::Tensor& candidate_hidden,
                         const torch::Tensor& candidate_tags) {
    return minimax_h3_build_ref2va_packed_layout(
        candidate_hidden, candidate_tags, h3_target(), {h3_image()});
  };

  expect_invalid_argument_contains([&] { build(torch::Tensor(), tags); },
                                   "condition hidden");
  expect_invalid_argument_contains(
      [&] { build(hidden.to(torch::kFloat32), tags); }, "condition hidden");
  expect_invalid_argument_contains(
      [&] { build(torch::zeros({2, 3, 5120}, torch::kBFloat16), tags); },
      "condition hidden");
  expect_invalid_argument_contains(
      [&] { build(torch::zeros({1, 3, 5119}, torch::kBFloat16), tags); },
      "condition hidden");
  const torch::Tensor noncontiguous_hidden =
      torch::zeros({1, 3, 10240}, torch::kBFloat16)
          .slice(/*dim=*/2, /*start=*/0, /*end=*/10240, /*step=*/2);
  ASSERT_FALSE(noncontiguous_hidden.is_contiguous());
  expect_invalid_argument_contains([&] { build(noncontiguous_hidden, tags); },
                                   "condition hidden");

  expect_invalid_argument_contains([&] { build(hidden, torch::Tensor()); },
                                   "condition tags");
  expect_invalid_argument_contains(
      [&] { build(hidden, tags.to(torch::kInt32)); }, "condition tags");
  expect_invalid_argument_contains(
      [&] { build(hidden, torch::tensor({0, 1, 1}, torch::kInt64)); },
      "condition tags");
  expect_invalid_argument_contains(
      [&] { build(hidden, torch::tensor({{0, 1}}, torch::kInt64)); },
      "condition tags");
  expect_invalid_argument_contains(
      [&] { build(hidden, torch::tensor({{0, 2, 1}}, torch::kInt64)); },
      "values must be 0 or 1");
  const torch::Tensor noncontiguous_tags =
      torch::zeros({1, 6}, torch::kInt64)
          .slice(/*dim=*/1, /*start=*/0, /*end=*/6, /*step=*/2);
  ASSERT_FALSE(noncontiguous_tags.is_contiguous());
  expect_invalid_argument_contains([&] { build(hidden, noncontiguous_tags); },
                                   "condition tags");

  expect_invalid_argument_contains(
      [&] { h3_layout({h3_image()}, {0, 1, 1}, /*sequence_length=*/0); },
      "smaller than used");
  expect_invalid_argument_contains(
      [&] { h3_layout({h3_image()}, {0, 1, 1}, /*sequence_length=*/65); },
      "divisible by 64");
}

TEST(MiniMaxH3SchedulerTest, BuildsExactBaseDualSchedule) {
  const MiniMaxH3DualSigmaSchedule schedule = MiniMaxH3Scheduler::build_base();
  ASSERT_EQ(schedule.video.point_count(), 50);
  ASSERT_EQ(schedule.video.forward_count(), 49);
  ASSERT_EQ(schedule.audio.point_count(), 50);
  ASSERT_EQ(schedule.audio.forward_count(), 49);
  EXPECT_EQ(schedule.video.sigmas.scalar_type(), torch::kFloat32);
  EXPECT_EQ(schedule.audio.sigmas.scalar_type(), torch::kFloat32);
  EXPECT_EQ(schedule.video.sigmas[0].item<float>(), 1.0F);
  EXPECT_EQ(schedule.video.sigmas[1].item<float>(), 0.998266875743866F);
  EXPECT_EQ(schedule.video.sigmas[24].item<float>(), 0.9259259700775146F);
  EXPECT_EQ(schedule.video.sigmas[48].item<float>(), 0.20000000298023224F);
  EXPECT_EQ(schedule.video.sigmas[49].item<float>(), 0.0F);
  EXPECT_EQ(schedule.audio.sigmas[1].item<float>(), 0.9931034445762634F);
  EXPECT_EQ(schedule.audio.sigmas[24].item<float>(), 0.7575757503509521F);
  EXPECT_EQ(schedule.audio.sigmas[48].item<float>(), 0.05882352963089943F);
  EXPECT_EQ(schedule.audio.sigmas[49].item<float>(), 0.0F);
  EXPECT_TRUE((schedule.video.sigmas.slice(0, 1) <
               schedule.video.sigmas.slice(0, 0, -1))
                  .all()
                  .item<bool>());
  EXPECT_TRUE(torch::equal(schedule.video.timesteps,
                           1.0F - schedule.video.sigmas.slice(
                                      0, 0, schedule.video.point_count() - 1)));
}

TEST(MiniMaxH3SchedulerTest, UsesDataWardVelocityAndEtaZeroBlend) {
  const torch::Tensor state = torch::tensor({2.0F}, torch::kFloat32);
  const torch::Tensor velocity = torch::tensor({3.0F}, torch::kFloat32);
  const torch::Tensor timestep = torch::tensor(0.25F, torch::kFloat32);
  const torch::Tensor denoised =
      MiniMaxH3Scheduler::velocity_to_x0(state, velocity, timestep);
  EXPECT_TRUE(torch::equal(denoised, torch::tensor({4.25F})));
  const torch::Tensor next =
      MiniMaxH3Scheduler::step_eta0(state, denoised, 0.75F, 0.25F);
  EXPECT_TRUE(torch::allclose(next, torch::tensor({3.5F}), 0.0, 0.0));
}

TEST(MiniMaxH3SchedulerTest, RejectsMalformedInputs) {
  expect_invalid_argument_contains([] { MiniMaxH3Scheduler::build(1, 12.0F); },
                                   "at least two");
  expect_invalid_argument_contains([] { MiniMaxH3Scheduler::build(50, 0.0F); },
                                   "positive");
  expect_invalid_argument_contains(
      [] {
        MiniMaxH3Scheduler::velocity_to_x0(
            torch::ones({2}), torch::ones({3}), torch::tensor(0.0F));
      },
      "same-shape");
  expect_invalid_argument_contains(
      [] {
        MiniMaxH3Scheduler::step_eta0(
            torch::ones({1}), torch::ones({1}), 0.25F, 0.5F);
      },
      "must not exceed");
}

TEST(MiniMaxH3DenoiserTest, BuildsAnchorAwareRowTimestepPlan) {
  const H3PackedLayout layout = h3_layout({h3_image(), h3_audio()}, {1, 0, 1});
  const MiniMaxH3RowTimestepPlan plan =
      minimax_h3_build_row_timestep_plan(layout, 0.25F, 0.5F);

  EXPECT_EQ(layout.used_length, 27);
  EXPECT_EQ(plan.row_timesteps.sizes().vec(), (std::vector<int64_t>{64}));
  EXPECT_EQ(plan.row_timesteps.scalar_type(), torch::kFloat32);
  EXPECT_TRUE(torch::equal(
      plan.unique_timesteps,
      torch::tensor({0.25F, 0.5F, 0.999F, 1.0F}, torch::kFloat32)));
  EXPECT_TRUE(
      torch::equal(plan.unique_timesteps.index_select(0, plan.inverse_indices),
                   plan.row_timesteps));
  EXPECT_TRUE(
      plan.row_timesteps.index({layout.img_pos.index({~layout.update_mask})})
          .eq(0.999F)
          .all()
          .item<bool>());
  EXPECT_TRUE(plan.row_timesteps
                  .index({layout.audio_pos.index({~layout.audio_update_mask})})
                  .eq(1.0F)
                  .all()
                  .item<bool>());
  EXPECT_TRUE(
      plan.row_timesteps.slice(0, layout.used_length, layout.aligned_length)
          .eq(0.25F)
          .all()
          .item<bool>());
}

TEST(MiniMaxH3DenoiserTest, RejectsOutOfRangeMediaPosition) {
  H3PackedLayout layout = h3_layout({h3_image(), h3_audio()}, {1, 0, 1});
  layout.img_pos = layout.img_pos.clone();
  layout.img_pos[0] = layout.used_length;
  expect_invalid_argument_contains(
      [&layout] { minimax_h3_build_row_timestep_plan(layout, 0.25F, 0.5F); },
      "outside used rows");
}

TEST(MiniMaxH3DenoiserTest, RegistersOneStreamingBlockAndFixedWeights) {
  MiniMaxH3StreamingDenoiser denoiser(
      torch::TensorOptions().device(torch::kMeta).dtype(torch::kBFloat16));
  const auto parameters = denoiser->named_parameters(/*recurse=*/true);
  const auto buffers = denoiser->named_buffers(/*recurse=*/true);
  EXPECT_EQ(parameters.size() + buffers.size(), 45);
  size_t block_tensors = 0;
  for (const auto& parameter : parameters) {
    if (parameter.key().starts_with("streaming_block.")) {
      ++block_tensors;
    }
  }
  EXPECT_EQ(block_tensors, MiniMaxH3StreamingDenoiserImpl::kBlockTensorCount);
  EXPECT_EQ(denoiser->loaded_fixed_tensor_count(), 0);
  EXPECT_EQ(denoiser->loaded_block_index(), -1);
}

TEST(MiniMaxH3VideoVAEHelpersTest, DeclaresExactFP32SourceModuleTree) {
  const auto specs =
      MiniMaxH3VideoVAESourceLayoutValidator::expected_source_tensors();
  ASSERT_EQ(specs.size(), 560);
  MiniMaxH3VideoVAE vae(
      torch::TensorOptions().device(torch::kMeta).dtype(torch::kBFloat16));
  const auto parameters = vae->named_parameters(/*recurse=*/true);
  const auto buffers = vae->named_buffers(/*recurse=*/true);
  EXPECT_EQ(parameters.size() + buffers.size(), 560);
  for (const auto& parameter : parameters) {
    EXPECT_EQ(parameter.value().scalar_type(), torch::kFloat32)
        << parameter.key();
  }
  for (const auto& buffer : buffers) {
    EXPECT_EQ(buffer.value().scalar_type(), torch::kFloat32) << buffer.key();
  }
}

TEST(MiniMaxH3VideoVAEHelpersTest, CausalConvDoesNotReadFutureFrames) {
  torch::NoGradGuard no_grad;
  MiniMaxH3VAECausalConv3d conv(
      /*input_channels=*/1,
      /*output_channels=*/1,
      /*kernel_size=*/3,
      torch::TensorOptions().dtype(torch::kFloat32));
  auto parameters = conv->named_parameters(/*recurse=*/true);
  parameters["weight"].fill_(1.0);
  parameters["bias"].zero_();
  const torch::Tensor input =
      torch::arange(4 * 4 * 4, torch::kFloat32).view({1, 1, 4, 4, 4});
  torch::Tensor changed = input.clone();
  changed.select(2, 3).add_(1000.0);
  const torch::Tensor baseline = conv->forward(input);
  const torch::Tensor modified = conv->forward(changed);
  EXPECT_TRUE(torch::equal(baseline.slice(2, 0, 3), modified.slice(2, 0, 3)));
  EXPECT_FALSE(torch::equal(baseline.select(2, 3), modified.select(2, 3)));
}

TEST(MiniMaxH3VideoVAEHelpersTest, GroupNormIsolatesTemporalFrames) {
  torch::NoGradGuard no_grad;
  MiniMaxH3VAEGroupNorm norm(
      /*channels=*/32, torch::TensorOptions().dtype(torch::kFloat32));
  auto parameters = norm->named_parameters(/*recurse=*/true);
  parameters["weight"].fill_(1.0);
  parameters["bias"].zero_();
  torch::manual_seed(19);
  const torch::Tensor input = torch::randn({1, 32, 2, 3, 3});
  torch::Tensor changed = input.clone();
  changed.select(2, 1).mul_(17.0).add_(31.0);
  const torch::Tensor baseline = norm->forward(input);
  const torch::Tensor modified = norm->forward(changed);
  EXPECT_TRUE(torch::equal(baseline.select(2, 0), modified.select(2, 0)));
}

TEST(MiniMaxH3VideoVAEHelpersTest, PosteriorClampAndNormalizeRoundTrip) {
  torch::Tensor mean =
      torch::arange(24, torch::kFloat32).view({1, 24, 1, 1, 1}) / 16.0;
  torch::Tensor raw_logvar =
      torch::linspace(-40.0, 30.0, 24).view({1, 24, 1, 1, 1});
  MiniMaxH3VAEDiagonalGaussianDistribution posterior(
      torch::cat({mean, raw_logvar}, 1));
  EXPECT_EQ(posterior.logvar().min().item<float>(), -30.0F);
  EXPECT_EQ(posterior.logvar().max().item<float>(), 20.0F);
  const torch::Tensor epsilon = torch::full_like(mean, 0.25);
  const torch::Tensor sample = posterior.sample(epsilon);
  EXPECT_TRUE(torch::equal(
      sample,
      mean + torch::exp(0.5 * raw_logvar.clamp(-30.0, 20.0)) * epsilon));
  const torch::Tensor normalized =
      MiniMaxH3VideoVAEImpl::normalize_latents(sample);
  const torch::Tensor restored =
      MiniMaxH3VideoVAEImpl::denormalize_latents(normalized);
  EXPECT_TRUE(torch::allclose(restored, sample, 1e-5, 1e-5));
}

TEST(MiniMaxH3AudioVAEHelpersTest, DeclaresExactSourceAndNativeInventories) {
  const auto specs =
      MiniMaxH3AudioVAESourceLayoutValidator::expected_source_tensors();
  ASSERT_EQ(specs.size(), 1087);
  MiniMaxH3AudioVAE vae(
      torch::TensorOptions().device(torch::kMeta).dtype(torch::kBFloat16));
  const auto parameters = vae->named_parameters(/*recurse=*/true);
  const auto buffers = vae->named_buffers(/*recurse=*/true);
  EXPECT_EQ(parameters.size() + buffers.size(), 915);
  EXPECT_EQ(parameters.size(), 660);
  EXPECT_EQ(buffers.size(), 255);
  for (const auto& parameter : parameters) {
    EXPECT_EQ(parameter.value().scalar_type(), torch::kFloat32)
        << parameter.key();
    EXPECT_FALSE(parameter.key().ends_with(".weight_g"));
    EXPECT_FALSE(parameter.key().ends_with(".weight_v"));
  }
  size_t filters = 0;
  for (const auto& buffer : buffers) {
    EXPECT_EQ(buffer.value().scalar_type(), torch::kFloat32) << buffer.key();
    filters += buffer.key().ends_with(".filter") ? 1 : 0;
  }
  EXPECT_EQ(filters, 254);
}

TEST(MiniMaxH3AudioVAEHelpersTest, FoldsWeightNormAlongOutputDimension) {
  const torch::Tensor weight_v =
      torch::tensor({{{3.0F, 4.0F}}, {{0.0F, 5.0F}}});
  const torch::Tensor weight_g = torch::tensor({2.0F, 3.0F}).view({2, 1, 1});
  const torch::Tensor folded =
      MiniMaxH3AudioVAEImpl::fold_weight_norm_dim0(weight_g, weight_v);
  const torch::Tensor expected =
      weight_v / weight_v.square().sum({1, 2}, true).sqrt() * weight_g;
  EXPECT_TRUE(torch::equal(folded, expected));
}

TEST(MiniMaxH3AudioVAEHelpersTest, SnakeVariantsUseCheckpointEquations) {
  torch::NoGradGuard no_grad;
  const torch::Tensor input =
      torch::tensor({{{-1.0F, 0.25F, 2.0F}}}, torch::kFloat32);
  MiniMaxH3AudioSnake1d snake(
      /*channels=*/1, torch::TensorOptions().dtype(torch::kFloat32));
  snake->named_parameters()["alpha"].fill_(2.0F);
  const torch::Tensor expected_snake =
      input + torch::sin(2.0F * input).square() / (2.0F + 1e-9F);
  EXPECT_TRUE(torch::equal(snake->forward(input), expected_snake));

  MiniMaxH3AudioSnakeBeta snake_beta(
      /*channels=*/1, torch::TensorOptions().dtype(torch::kFloat32));
  snake_beta->named_parameters()["alpha"].fill_(std::log(2.0F));
  snake_beta->named_parameters()["beta"].fill_(std::log(3.0F));
  const torch::Tensor expected_beta =
      input + torch::sin(2.0F * input).square() / (3.0F + 1e-9F);
  EXPECT_TRUE(torch::equal(snake_beta->forward(input), expected_beta));
}

TEST(MiniMaxH3AudioVAEHelpersTest, PosteriorUsesLogStandardDeviationAndMode) {
  const torch::Tensor mean =
      torch::arange(64, torch::kFloat32).view({2, 32, 1}) / 16.0F;
  const torch::Tensor logs = torch::full_like(mean, std::log(4.0F));
  const torch::Tensor epsilon = torch::full_like(mean, 0.25F);
  MiniMaxH3AudioDiagonalGaussianDistribution posterior(mean, logs);
  EXPECT_TRUE(torch::equal(posterior.mode(), mean));
  EXPECT_TRUE(torch::allclose(posterior.std(), torch::full_like(mean, 4.0F)));
  EXPECT_TRUE(torch::allclose(posterior.sample(epsilon), mean + 1.0F));

  const torch::Tensor normalized =
      MiniMaxH3AudioVAEImpl::normalize_latents(mean);
  const torch::Tensor restored =
      MiniMaxH3AudioVAEImpl::denormalize_latents(normalized);
  EXPECT_TRUE(torch::allclose(restored, mean, 1e-6, 1e-6));
}

TEST(MiniMaxH3AudioVAEHelpersTest, RejectsInvalidWeightsPosteriorAndStereoAbi) {
  expect_invalid_argument_contains(
      [] {
        MiniMaxH3AudioVAEImpl::fold_weight_norm_dim0(torch::ones({1, 1, 1}),
                                                     torch::zeros({2, 1, 3}));
      },
      "weight_g/weight_v");
  expect_invalid_argument_contains(
      [] {
        MiniMaxH3AudioVAEImpl::fold_weight_norm_dim0(torch::ones({2, 1, 1}),
                                                     torch::zeros({2, 1, 3}));
      },
      "invalid dimension-0 norm");
  expect_invalid_argument_contains(
      [] {
        MiniMaxH3AudioDiagonalGaussianDistribution(torch::zeros({2, 32, 4}),
                                                   torch::zeros({2, 31, 4}));
      },
      "matching FP32");
  MiniMaxH3AudioVAE vae(
      torch::TensorOptions().device(torch::kMeta).dtype(torch::kFloat32));
  expect_invalid_argument_contains(
      [&vae] { vae->encode_condition(torch::zeros({1, 1, 800})); },
      "stereo-as-batch");
  expect_invalid_argument_contains(
      [&vae] { vae->decode_normalized(torch::zeros({1, 32, 8})); },
      "stereo-as-batch");
}

TEST(MiniMaxH3BlockHelpersTest, ReordersGroupedQKVToAllQAllKAllV) {
  const torch::Tensor grouped =
      torch::arange(12, torch::kBFloat16).view({12, 1});
  const torch::Tensor reordered =
      minimax_h3_reorder_grouped_qkv(grouped, /*num_heads=*/2, /*head_dim=*/2);
  const torch::Tensor expected =
      torch::tensor({0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11}, torch::kBFloat16)
          .view({12, 1});
  EXPECT_TRUE(torch::equal(reordered, expected));
  EXPECT_EQ(reordered.scalar_type(), grouped.scalar_type());
}

TEST(MiniMaxH3BlockHelpersTest, ReordersGateUpRowsToOfficialUpGateLayout) {
  const torch::Tensor gate_up =
      torch::tensor({1, 2, 3, 4, 5, 6, 7, 8}, torch::kBFloat16).view({4, 2});
  const torch::Tensor actual = minimax_h3_reorder_gate_up_to_up_gate(gate_up);
  const torch::Tensor expected =
      torch::tensor({5, 6, 7, 8, 1, 2, 3, 4}, torch::kBFloat16).view({4, 2});
  EXPECT_TRUE(torch::equal(actual, expected));
}

TEST(MiniMaxH3BlockHelpersTest, RMSNormAccumulatesInFP32AndReturnsBF16) {
  const torch::Tensor input =
      torch::tensor({{1.0, 2.0, 3.0, 4.0}}, torch::kBFloat16);
  const torch::Tensor weight =
      torch::tensor({1.0, 1.5, 0.5, 2.0}, torch::kBFloat16);
  const torch::Tensor actual = minimax_h3_rms_norm(input, weight, 1e-5);
  const torch::Tensor fp32 = input.to(torch::kFloat32);
  const torch::Tensor expected =
      (fp32 * torch::rsqrt(fp32.pow(2).mean(-1, true) + 1e-5) *
       weight.to(torch::kFloat32))
          .to(torch::kBFloat16);
  EXPECT_EQ(actual.scalar_type(), torch::kBFloat16);
  EXPECT_TRUE(torch::equal(actual, expected));
}

TEST(MiniMaxH3BlockHelpersTest, TimestepUsesHalfAsExactDenominator) {
  const torch::Tensor timestep = torch::tensor({1.0}, torch::kFloat32);
  const torch::Tensor actual =
      minimax_h3_timestep_embedding(timestep, /*embedding_size=*/8);
  const torch::Tensor frequency =
      torch::exp(-std::log(10000.0) * torch::arange(4, torch::kFloat32) / 4.0);
  const torch::Tensor expected =
      torch::cat({torch::cos(frequency), torch::sin(frequency)}).view({1, 8});
  EXPECT_TRUE(torch::allclose(actual, expected, 0.0, 0.0));
}

TEST(MiniMaxH3BlockHelpersTest, RopeRotatesFirst96AndPreservesFinal32) {
  const torch::Tensor position_ids =
      torch::tensor({{1.5707963267948966, 0.5, -0.25}}, torch::kFloat64);
  const torch::Tensor inv_freq = torch::linspace(1.0, 0.1, 16);
  const torch::Tensor frequencies =
      minimax_h3_rope_frequencies(position_ids, inv_freq);
  ASSERT_EQ(frequencies.sizes().vec(), (std::vector<int64_t>{1, 96}));
  EXPECT_TRUE(
      torch::equal(frequencies.slice(1, 0, 48), frequencies.slice(1, 48, 96)));

  const torch::Tensor input = torch::arange(128, torch::kFloat32)
                                  .to(torch::kBFloat16)
                                  .view({1, 1, 128});
  const torch::Tensor output = minimax_h3_apply_rope(input, frequencies);
  EXPECT_FALSE(torch::equal(output.slice(2, 0, 96), input.slice(2, 0, 96)));
  EXPECT_TRUE(torch::equal(output.slice(2, 96, 128), input.slice(2, 96, 128)));
}

TEST(MiniMaxH3BlockHelpersTest, AdaLNUsesTimestepMajorModalityRows) {
  const torch::Tensor inverse = torch::tensor({0, 1, 0, 1}, torch::kInt64);
  const torch::Tensor tags = torch::tensor({0, 1, 2, -1}, torch::kInt64);
  const torch::Tensor combined =
      minimax_h3_combined_adaln_indices(inverse, tags);
  EXPECT_TRUE(
      torch::equal(combined, torch::tensor({0, 4, 2, 3}, torch::kInt64)));

  const torch::Tensor parameters =
      torch::arange(2 * 3 * 6 * 2, torch::kFloat32).view({2, 3, 6, 2});
  const torch::Tensor actual =
      minimax_h3_select_adaln_parameter(parameters, 4, combined);
  const torch::Tensor expected = torch::stack({parameters[0][0][4],
                                               parameters[1][1][4],
                                               parameters[0][2][4],
                                               parameters[1][0][4]});
  EXPECT_TRUE(torch::equal(actual, expected));
}

TEST(MiniMaxH3BlockHelpersTest, SegmentedSDPAIsolatesDocuments) {
  torch::manual_seed(17);
  const torch::Tensor query = torch::randn({6, 2, 4});
  const torch::Tensor key = torch::randn({6, 2, 4});
  const torch::Tensor value = torch::randn({6, 2, 4});
  const torch::Tensor cu_seqlens = torch::tensor({0, 3, 6}, torch::kInt32);
  const torch::Tensor baseline =
      minimax_h3_segmented_sdpa(query, key, value, cu_seqlens);
  const torch::Tensor changed = minimax_h3_segmented_sdpa(
      torch::cat({query.slice(0, 0, 3), query.slice(0, 3, 6) + 1000}, 0),
      torch::cat({key.slice(0, 0, 3), key.slice(0, 3, 6) - 1000}, 0),
      torch::cat({value.slice(0, 0, 3), value.slice(0, 3, 6) + 1000}, 0),
      cu_seqlens);
  EXPECT_TRUE(torch::equal(baseline.slice(0, 0, 3), changed.slice(0, 0, 3)));
}

std::vector<std::unique_ptr<StateDict>> synthetic_c4_source(
    const std::optional<std::string>& omitted = std::nullopt,
    const std::optional<std::string>& unknown = std::nullopt) {
  std::unordered_map<std::string, torch::Tensor> tensors;
  const c10::Device meta_device(c10::DeviceType::Meta);
  for (const MiniMaxH3C4SourceTensorSpec& spec :
       MiniMaxH3C4SourceLayoutValidator::expected_source_tensors()) {
    if (omitted == spec.name) {
      continue;
    }
    tensors.emplace(
        spec.name,
        torch::empty(
            spec.shape,
            torch::TensorOptions().dtype(spec.dtype).device(meta_device)));
  }
  if (unknown.has_value()) {
    tensors.emplace(*unknown,
                    torch::empty({1},
                                 torch::TensorOptions()
                                     .dtype(torch::kBFloat16)
                                     .device(meta_device)));
  }
  std::vector<std::unique_ptr<StateDict>> shards;
  shards.emplace_back(std::make_unique<StateDict>(std::move(tensors)));
  return shards;
}

TEST(MiniMaxH3C4SourceTest, RequiresExactSelectedTensorInventory) {
  const auto specs =
      MiniMaxH3C4SourceLayoutValidator::expected_source_tensors();
  ASSERT_EQ(specs.size(), 45);
  EXPECT_EQ(
      MiniMaxH3C4SourceLayoutValidator::validate(synthetic_c4_source()).size(),
      45);
  expect_invalid_argument_contains(
      [] {
        MiniMaxH3C4SourceLayoutValidator::validate(
            synthetic_c4_source("blocks.0.mlp.fc2.weight"));
      },
      "missing tensor `blocks.0.mlp.fc2.weight`");
  expect_invalid_argument_contains(
      [] {
        MiniMaxH3C4SourceLayoutValidator::validate(
            synthetic_c4_source(std::nullopt, "blocks.0.unknown.weight"));
      },
      "unknown selected tensor");

  auto duplicate = synthetic_c4_source();
  std::unordered_map<std::string, torch::Tensor> duplicate_tensor;
  duplicate_tensor.emplace(
      "rope.inv_freq",
      torch::empty({16},
                   torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(c10::Device(c10::DeviceType::Meta))));
  duplicate.emplace_back(
      std::make_unique<StateDict>(std::move(duplicate_tensor)));
  expect_invalid_argument_contains(
      [&duplicate] { MiniMaxH3C4SourceLayoutValidator::validate(duplicate); },
      "loaded more than once");
}

TEST(MiniMaxH3C4SourceTest, RegisteredModuleTreePreservesMixedDtypes) {
  const torch::TensorOptions options =
      torch::TensorOptions().device(torch::kMeta).dtype(torch::kBFloat16);
  MiniMaxH3C4Harness harness(options);
  std::unordered_map<std::string, torch::ScalarType> expected;
  for (const MiniMaxH3C4SourceTensorSpec& spec :
       MiniMaxH3C4SourceLayoutValidator::expected_source_tensors()) {
    expected.emplace(spec.name, spec.dtype);
  }
  const auto parameters = harness->named_parameters(/*recurse=*/true);
  const auto buffers = harness->named_buffers(/*recurse=*/true);
  ASSERT_EQ(parameters.size() + buffers.size(), 45);
  for (const auto& parameter : parameters) {
    ASSERT_TRUE(expected.contains(parameter.key())) << parameter.key();
    EXPECT_EQ(parameter.value().scalar_type(), expected.at(parameter.key()));
  }
  for (const auto& buffer : buffers) {
    ASSERT_TRUE(expected.contains(buffer.key())) << buffer.key();
    EXPECT_EQ(buffer.value().scalar_type(), expected.at(buffer.key()));
  }
}

struct H3ComparisonThreshold {
  double relative_l2;
  double minimum_cosine;
};

struct H3ComparisonMetrics {
  double relative_l2;
  double cosine;
  double max_abs;
};

H3ComparisonMetrics h3_comparison_metrics(const torch::Tensor& actual,
                                          const torch::Tensor& expected) {
  const torch::Tensor actual_fp64 = actual.to(torch::kCPU).to(torch::kFloat64);
  const torch::Tensor expected_fp64 =
      expected.to(torch::kCPU).to(torch::kFloat64);
  const torch::Tensor difference = actual_fp64 - expected_fp64;
  const double expected_norm = expected_fp64.norm().item<double>();
  const double denominator =
      std::max(expected_norm, std::numeric_limits<double>::min());
  const double actual_norm = actual_fp64.norm().item<double>();
  double cosine = 1.0;
  if (actual_norm > 0.0 && expected_norm > 0.0) {
    cosine = (actual_fp64.flatten().dot(expected_fp64.flatten()) /
              (actual_norm * expected_norm))
                 .item<double>();
  }
  return {.relative_l2 = difference.norm().item<double>() / denominator,
          .cosine = cosine,
          .max_abs = difference.abs().max().item<double>()};
}

void validate_h3_audio_golden_manifest(
    const std::filesystem::path& golden_path) {
  const std::filesystem::path manifest_path =
      golden_path.parent_path() / "minimax_h3_audio_vae_reference.json";
  JsonReader reader;
  if (!reader.parse(manifest_path.string())) {
    throw std::invalid_argument(
        "MiniMax-H3 C6b cannot parse the Golden manifest");
  }
  const nlohmann::json& manifest = reader.data();
  const auto fail = [](const std::string& field) {
    throw std::invalid_argument(
        "MiniMax-H3 C6b Golden manifest failed attestation at `" + field + "`");
  };
  if (manifest.value("schema", "") !=
      "xllm.minimax_h3.audio_vae_reference/v1") {
    fail("schema");
  }
  if (manifest.value("status", "") != "OFFICIAL_H3_C6B_AUDIO_VAE_GOLDEN") {
    fail("status");
  }
  if (manifest.at("source").value("revision", "") !=
      "d30c748f5f5d0925a5af14dc0e6a6de983025e63") {
    fail("source.revision");
  }
  if (manifest.at("checkpoint").value("tensor_count", 0) != 1087 ||
      manifest.at("checkpoint").value("tensor_dtype", "") != "float32") {
    fail("checkpoint");
  }
  const std::string diffusers_path =
      manifest.at("runtime").value("diffusers_path", "");
  if (!diffusers_path.starts_with(
          "/data/workspace/lwd/minimax/diffusers-reference/src/")) {
    fail("runtime.diffusers_path");
  }
  if (manifest.at("runtime").value("attention_backend", "") != "_native_math") {
    fail("runtime.attention_backend");
  }
  if (manifest.at("production_smoke").value("status", "") != "PASS" ||
      manifest.at("production_smoke").at("pipeline_stereo").at("shape") !=
          nlohmann::json({1, 2, 165600})) {
    fail("production_smoke");
  }
  const std::string artifact_digest =
      manifest.at("artifact").value("sha256", "");
  if (manifest.at("artifact").value("path", "") !=
          golden_path.filename().string() ||
      artifact_digest !=
          "08359fff731dc76ff821fe819458db3ecb41a40012d05a6d22397f3b8003ac45") {
    fail("artifact");
  }
}

bool compare_h3_golden_node(const std::string& name,
                            const torch::Tensor& actual,
                            const StateDict& golden,
                            const H3ComparisonThreshold& threshold) {
  const torch::Tensor* expected = find_raw_tensor(golden, name);
  if (expected == nullptr) {
    ADD_FAILURE() << "Golden tensor is missing for node " << name;
    return false;
  }
  if (!actual.defined() || actual.sizes() != expected->sizes() ||
      actual.scalar_type() != expected->scalar_type()) {
    ADD_FAILURE() << "Node metadata mismatch for " << name;
    return false;
  }
  if (!torch::isfinite(actual).all().item<bool>() ||
      !torch::isfinite(*expected).all().item<bool>()) {
    ADD_FAILURE() << "NaN or Inf at node " << name;
    return false;
  }
  const H3ComparisonMetrics metrics = h3_comparison_metrics(actual, *expected);
  std::cout << "H3-C4 node=" << name << " relative_l2=" << metrics.relative_l2
            << " cosine=" << metrics.cosine << " max_abs=" << metrics.max_abs
            << std::endl;
  if (metrics.relative_l2 > threshold.relative_l2 ||
      metrics.cosine < threshold.minimum_cosine) {
    ADD_FAILURE() << "First divergent H3-C4 node: " << name
                  << " relative_l2=" << metrics.relative_l2 << " > "
                  << threshold.relative_l2 << ", cosine=" << metrics.cosine
                  << " < " << threshold.minimum_cosine
                  << ", max_abs=" << metrics.max_abs;
    return false;
  }
  return true;
}

TEST(MiniMaxH3BlockGoldenTest, MatchesOfficialUsedRowsAndIsolatesPadding) {
  const char* golden_value = std::getenv("MINIMAX_H3_BLOCK_GOLDEN");
  const char* checkpoint_value = std::getenv("MINIMAX_H3_CHECKPOINT");
  if (golden_value == nullptr || std::string(golden_value).empty() ||
      checkpoint_value == nullptr || std::string(checkpoint_value).empty()) {
    GTEST_SKIP() << "Set MINIMAX_H3_BLOCK_GOLDEN and MINIMAX_H3_CHECKPOINT for "
                    "the real NPU C4 gate";
  }
  std::filesystem::path golden_path(golden_value);
  if (std::filesystem::is_directory(golden_path)) {
    golden_path /= "minimax_h3_block_reference.safetensors";
  }
  const std::unique_ptr<StateDict> golden =
      StateDictFromSafeTensor::load(golden_path.string());
  ASSERT_NE(golden, nullptr);

  auto loader = std::make_unique<DiTModelLoader>(checkpoint_value);
  DiTModelContext context(ParallelArgs(0, 1, nullptr),
                          loader->get_model_args(),
                          loader->get_quant_args(),
                          torch::TensorOptions()
                              .device(torch::Device("npu:0"))
                              .dtype(torch::kBFloat16),
                          DiTCacheConfig(),
                          loader->get_model_type());
  MiniMaxH3Pipeline pipeline(context);
  pipeline->load_model(std::move(loader));
  pipeline->load_c4_probe();
  pipeline->c4_harness()->verify_loaded_weights();

  const H3PackedLayout layout = h3_layout({h3_image()});
  const torch::Tensor video_rows = golden->get_tensor("input.video_rows");
  const torch::Tensor audio_rows = golden->get_tensor("input.audio_rows");
  const torch::Tensor timesteps = golden->get_tensor("input.timesteps");
  const torch::Tensor inverse_indices =
      golden->get_tensor("input.inverse_indices");
  torch::NoGradGuard no_grad;
  const MiniMaxH3C4Trace trace = pipeline->probe_c4(
      layout, video_rows, audio_rows, timesteps, inverse_indices);
  const int64_t used = layout.used_length;
  const auto used_rows = [used](const torch::Tensor& tensor) {
    return tensor.slice(0, 0, used);
  };

  const std::unordered_map<std::string, H3ComparisonThreshold> thresholds = {
      {"condition_projection", {1e-6, 0.999999999}},
      {"video_embedding", {1e-6, 0.999999999}},
      {"audio_embedding", {1e-6, 0.999999999}},
      {"time_embedding", {1e-6, 0.999999999}},
      {"rope_frequencies", {1e-6, 0.999999999}},
      {"token_refiner.block0.attention_delta", {0.002, 0.999999}},
      {"token_refiner.block0.mlp_delta", {0.004, 0.99999}},
      {"token_refiner.block0.output", {0.004, 0.99999}},
      {"token_refiner.block1.attention_delta", {0.006, 0.99998}},
      {"token_refiner.block1.mlp_delta", {0.006, 0.99998}},
      {"token_refiner.block1.output", {0.004, 0.99999}},
      {"refined_condition", {0.004, 0.99999}},
      {"packed_hidden", {0.003, 0.999995}},
      {"block0.attention_delta", {0.002, 0.999999}},
      {"block0.mlp_delta", {0.003, 0.999995}},
      {"block0.output", {0.003, 0.999995}},
      {"final_activation", {0.003, 0.999995}},
      {"all_video_logits", {0.002, 0.999999}},
      {"all_audio_logits", {0.001, 0.999999}},
      {"selected_video_logits", {0.003, 0.999995}},
      {"selected_audio_logits", {0.001, 0.999999}},
  };
  const std::vector<std::pair<std::string, torch::Tensor>> projection_nodes = {
      {"condition_projection", trace.condition_projection},
      {"video_embedding", trace.video_embedding},
      {"audio_embedding", trace.audio_embedding},
      {"time_embedding", trace.time_embedding},
      {"rope_frequencies", used_rows(trace.rope_frequencies)},
  };
  for (const auto& [name, actual] : projection_nodes) {
    if (!compare_h3_golden_node(name, actual, *golden, thresholds.at(name))) {
      return;
    }
  }
  for (size_t index = 0; index < trace.token_refiner_blocks.size(); ++index) {
    const MiniMaxH3ResidualBranchTrace& block =
        trace.token_refiner_blocks[index];
    const std::string prefix =
        "token_refiner.block" + std::to_string(index) + ".";
    for (const auto& [suffix, actual] :
         std::vector<std::pair<std::string, torch::Tensor>>{
             {"attention_delta", block.attention_delta},
             {"mlp_delta", block.mlp_delta},
             {"output", block.output}}) {
      const std::string name = prefix + suffix;
      if (!compare_h3_golden_node(name, actual, *golden, thresholds.at(name))) {
        return;
      }
    }
  }
  if (!compare_h3_golden_node("refined_condition",
                              trace.refined_condition,
                              *golden,
                              thresholds.at("refined_condition"))) {
    return;
  }
  if (!compare_h3_golden_node("packed_hidden",
                              used_rows(trace.packed_hidden),
                              *golden,
                              thresholds.at("packed_hidden"))) {
    return;
  }
  for (const auto& [name, actual] :
       std::vector<std::pair<std::string, torch::Tensor>>{
           {"block0.attention_delta", used_rows(trace.block0.attention_delta)},
           {"block0.mlp_delta", used_rows(trace.block0.mlp_delta)},
           {"block0.output", used_rows(trace.block0.output)}}) {
    if (!compare_h3_golden_node(name, actual, *golden, thresholds.at(name))) {
      return;
    }
  }
  for (const auto& [name, actual] :
       std::vector<std::pair<std::string, torch::Tensor>>{
           {"final_activation", used_rows(trace.final_output.activation)},
           {"all_video_logits", used_rows(trace.final_output.all_video_logits)},
           {"all_audio_logits", used_rows(trace.final_output.all_audio_logits)},
           {"selected_video_logits", trace.final_output.selected_video_logits},
           {"selected_audio_logits",
            trace.final_output.selected_audio_logits}}) {
    if (!compare_h3_golden_node(name, actual, *golden, thresholds.at(name))) {
      return;
    }
  }

  torch::Tensor changed_padding = trace.packed_hidden.clone();
  changed_padding.slice(0, used, layout.aligned_length).fill_(17.0);
  const MiniMaxH3ResidualBranchTrace changed =
      pipeline->c4_harness()->block0()->forward(changed_padding,
                                                trace.time_embedding,
                                                trace.combined_indices,
                                                trace.rope_frequencies,
                                                layout.cu_seqlens);
  ASSERT_TRUE(torch::isfinite(changed.output).all().item<bool>());
  ASSERT_TRUE(
      torch::equal(used_rows(trace.block0.output), used_rows(changed.output)))
      << "Changing [used,aligned) padding affected a real packed row";
  std::cout << "G5_SINGLE_DIT_BLOCK=PASS" << std::endl;
}

TEST(MiniMaxH3C5GoldenTest, MatchesOfficialFullBaseTrajectory) {
  const char* golden_value = std::getenv("MINIMAX_H3_TRAJECTORY_GOLDEN");
  const char* checkpoint_value = std::getenv("MINIMAX_H3_CHECKPOINT");
  if (golden_value == nullptr || std::string(golden_value).empty() ||
      checkpoint_value == nullptr || std::string(checkpoint_value).empty()) {
    GTEST_SKIP() << "Set MINIMAX_H3_TRAJECTORY_GOLDEN and "
                    "MINIMAX_H3_CHECKPOINT for the real NPU C5 Gate";
  }
  std::filesystem::path golden_path(golden_value);
  if (std::filesystem::is_directory(golden_path)) {
    golden_path /= "minimax_h3_trajectory_reference.safetensors";
  }
  const std::unique_ptr<StateDict> golden =
      StateDictFromSafeTensor::load(golden_path.string());
  ASSERT_NE(golden, nullptr);

  const MiniMaxH3DualSigmaSchedule schedule = MiniMaxH3Scheduler::build_base();
  ASSERT_TRUE(torch::equal(schedule.video.sigmas,
                           golden->get_tensor("schedule.video_sigmas")));
  ASSERT_TRUE(torch::equal(schedule.audio.sigmas,
                           golden->get_tensor("schedule.audio_sigmas")));
  ASSERT_TRUE(torch::equal(schedule.video.timesteps,
                           golden->get_tensor("schedule.video_timesteps")));
  ASSERT_TRUE(torch::equal(schedule.audio.timesteps,
                           golden->get_tensor("schedule.audio_timesteps")));

  auto loader = std::make_unique<DiTModelLoader>(checkpoint_value);
  DiTModelContext context(ParallelArgs(0, 1, nullptr),
                          loader->get_model_args(),
                          loader->get_quant_args(),
                          torch::TensorOptions()
                              .device(torch::Device("npu:0"))
                              .dtype(torch::kBFloat16),
                          DiTCacheConfig(),
                          loader->get_model_type());
  MiniMaxH3Pipeline pipeline(context);
  pipeline->load_model(std::move(loader));
  pipeline->load_c5_denoiser();
  ASSERT_EQ(pipeline->c5_denoiser()->loaded_fixed_tensor_count(), 35);

  const H3PackedLayout layout = h3_layout({h3_image(), h3_audio()}, {1, 0, 1});
  const torch::Tensor initial_video =
      golden->get_tensor("input.initial_video_rows");
  const torch::Tensor initial_audio =
      golden->get_tensor("input.initial_audio_rows");
  bool gate_pass = true;
  const auto report = [&golden, &gate_pass](const std::string& name,
                                            const torch::Tensor& actual) {
    const torch::Tensor* expected = find_raw_tensor(*golden, name);
    EXPECT_NE(expected, nullptr) << name;
    if (expected == nullptr) {
      gate_pass = false;
      return;
    }
    EXPECT_EQ(actual.sizes(), expected->sizes()) << name;
    EXPECT_EQ(actual.scalar_type(), expected->scalar_type()) << name;
    EXPECT_TRUE(torch::isfinite(actual).all().item<bool>()) << name;
    if (actual.sizes() != expected->sizes() ||
        actual.scalar_type() != expected->scalar_type()) {
      gate_pass = false;
      return;
    }
    const torch::Tensor actual_cpu = actual.to(torch::kCPU);
    const H3ComparisonMetrics metrics =
        h3_comparison_metrics(actual_cpu, *expected);
    std::cout << "H3-C5 node=" << name << " relative_l2=" << metrics.relative_l2
              << " cosine=" << metrics.cosine << " max_abs=" << metrics.max_abs
              << std::endl;
    if (!torch::equal(actual_cpu, *expected)) {
      ADD_FAILURE() << "H3-C5 node is not bit-exact: " << name;
      gate_pass = false;
    }
  };
  const MiniMaxH3LayerObserver layer_observer =
      [&report](int64_t step,
                int64_t layer,
                const MiniMaxH3ResidualBranchTrace& trace) {
        std::ostringstream name;
        name << "step_" << std::setw(3) << std::setfill('0') << step
             << ".block_" << std::setw(2) << layer << ".output";
        report(name.str(), trace.output);
        if (step == 0 && layer == 0) {
          report("step_000.block_00.adaln", trace.adaln_parameters);
          report("step_000.block_00.shift_msa_selected", trace.shift_msa);
          report("step_000.block_00.scale_msa_selected", trace.scale_msa);
          report("step_000.block_00.gate_msa_selected", trace.gate_msa);
          report("step_000.block_00.shift_mlp_selected", trace.shift_mlp);
          report("step_000.block_00.scale_mlp_selected", trace.scale_mlp);
          report("step_000.block_00.gate_mlp_selected", trace.gate_mlp);
          report("step_000.block_00.norm1_output", trace.norm1_output);
          report("step_000.block_00.attention_input", trace.attention_input);
          report("step_000.block_00.attention_output", trace.attention_output);
          report("step_000.block_00.attention_delta", trace.attention_delta);
          report("step_000.block_00.norm2_output", trace.norm2_output);
          report("step_000.block_00.mlp_input", trace.mlp_input);
          report("step_000.block_00.mlp_output", trace.mlp_output);
          report("step_000.block_00.mlp_delta", trace.mlp_delta);
        }
      };
  const MiniMaxH3StepObserver step_observer =
      [&golden, &report, &gate_pass](const MiniMaxH3TrajectoryStep& trace) {
        std::ostringstream prefix;
        prefix << "step_" << std::setw(3) << std::setfill('0') << trace.step
               << ".";
        const std::string base = prefix.str();
        const bool row_times_exact =
            torch::equal(trace.timestep_plan.row_timesteps,
                         golden->get_tensor(base + "row_timesteps"));
        const bool unique_times_exact =
            torch::equal(trace.timestep_plan.unique_timesteps,
                         golden->get_tensor(base + "unique_timesteps"));
        const bool inverse_exact =
            torch::equal(trace.timestep_plan.inverse_indices,
                         golden->get_tensor(base + "inverse_indices"));
        EXPECT_TRUE(row_times_exact);
        EXPECT_TRUE(unique_times_exact);
        EXPECT_TRUE(inverse_exact);
        gate_pass =
            gate_pass && row_times_exact && unique_times_exact && inverse_exact;
        report(base + "video_velocity", trace.denoiser.video_velocity);
        report(base + "audio_velocity", trace.denoiser.audio_velocity);
        report(base + "video_x0", trace.video_x0);
        report(base + "audio_x0", trace.audio_x0);
        report(base + "video_rows_after", trace.video_rows_after);
        report(base + "audio_rows_after", trace.audio_rows_after);
        if (trace.step == 0) {
          report("step_000.packed_hidden", trace.denoiser.packed_hidden);
          report("step_000.condition_projection",
                 trace.denoiser.condition_projection);
          report("step_000.refined_condition",
                 trace.denoiser.refined_condition);
          report("step_000.video_embedding", trace.denoiser.video_embedding);
          report("step_000.audio_embedding", trace.denoiser.audio_embedding);
          report("step_000.time_embedding", trace.denoiser.time_embedding);
          report("step_000.rope_frequencies", trace.denoiser.rope_frequencies);
          const bool combined_exact =
              torch::equal(trace.denoiser.combined_indices.to(torch::kCPU),
                           golden->get_tensor("step_000.combined_indices"));
          EXPECT_TRUE(combined_exact);
          gate_pass = gate_pass && combined_exact;
          report("step_000.final_activation", trace.denoiser.final_activation);
        }
      };

  const MiniMaxH3TrajectoryOutput output = pipeline->probe_c5_trajectory(
      layout, initial_video, initial_audio, layer_observer, step_observer);
  ASSERT_EQ(output.transformer_forwards, 49);
  ASSERT_EQ(output.block_forwards, 2450);
  ASSERT_EQ(pipeline->c5_denoiser()->loaded_block_index(), 49);
  report("step_048.video_rows_after", output.video_rows);
  report("step_048.audio_rows_after", output.audio_rows);
  ASSERT_TRUE(gate_pass);
  std::cout << "G6_FULL_DENOISE=PASS" << std::endl;
}

TEST(MiniMaxH3C5DiagnosticTest, StreamingBlockMatchesC4ResidentBlock) {
  const char* golden_value = std::getenv("MINIMAX_H3_TRAJECTORY_GOLDEN");
  const char* checkpoint_value = std::getenv("MINIMAX_H3_CHECKPOINT");
  if (golden_value == nullptr || std::string(golden_value).empty() ||
      checkpoint_value == nullptr || std::string(checkpoint_value).empty()) {
    GTEST_SKIP() << "Set C5 Golden and checkpoint paths for block loader A/B";
  }
  std::filesystem::path golden_path(golden_value);
  if (std::filesystem::is_directory(golden_path)) {
    golden_path /= "minimax_h3_trajectory_reference.safetensors";
  }
  const std::unique_ptr<StateDict> golden =
      StateDictFromSafeTensor::load(golden_path.string());
  ASSERT_NE(golden, nullptr);
  auto loader = std::make_unique<DiTModelLoader>(checkpoint_value);
  DiTModelContext context(ParallelArgs(0, 1, nullptr),
                          loader->get_model_args(),
                          loader->get_quant_args(),
                          torch::TensorOptions()
                              .device(torch::Device("npu:0"))
                              .dtype(torch::kBFloat16),
                          DiTCacheConfig(),
                          loader->get_model_type());
  MiniMaxH3Pipeline pipeline(context);
  pipeline->load_model(std::move(loader));
  pipeline->load_c4_probe();
  pipeline->load_c5_denoiser();

  const H3PackedLayout layout = h3_layout({h3_image(), h3_audio()}, {1, 0, 1});
  const torch::Tensor video = golden->get_tensor("input.initial_video_rows");
  const torch::Tensor audio = golden->get_tensor("input.initial_audio_rows");
  const torch::Tensor timesteps =
      golden->get_tensor("step_000.unique_timesteps");
  const torch::Tensor inverse = golden->get_tensor("step_000.inverse_indices");
  const MiniMaxH3C4Trace preparation =
      pipeline->probe_c4(layout, video, audio, timesteps, inverse);
  torch::Tensor hidden = torch::zeros(
      {layout.aligned_length, MiniMaxH3TransformerConfig::kHiddenSize},
      preparation.packed_hidden.options());
  hidden.slice(0, 0, layout.used_length)
      .copy_(golden->get_tensor("step_000.packed_hidden").to(hidden.device()));
  const MiniMaxH3ResidualBranchTrace resident =
      pipeline->c4_harness()->block0()->forward(hidden,
                                                preparation.time_embedding,
                                                preparation.combined_indices,
                                                preparation.rope_frequencies,
                                                layout.cu_seqlens);
  const MiniMaxH3ResidualBranchTrace streaming =
      pipeline->probe_c5_streaming_block(0,
                                         hidden,
                                         preparation.time_embedding,
                                         preparation.combined_indices,
                                         preparation.rope_frequencies,
                                         layout.cu_seqlens);
  EXPECT_TRUE(
      torch::equal(resident.attention_input, streaming.attention_input));
  EXPECT_TRUE(
      torch::equal(resident.attention_output, streaming.attention_output));
  EXPECT_TRUE(
      torch::equal(resident.attention_delta, streaming.attention_delta));
  EXPECT_TRUE(torch::equal(resident.mlp_input, streaming.mlp_input));
  EXPECT_TRUE(torch::equal(resident.mlp_output, streaming.mlp_output));
  EXPECT_TRUE(torch::equal(resident.mlp_delta, streaming.mlp_delta));
  EXPECT_TRUE(torch::equal(resident.output, streaming.output));
  std::cout << "H3_C5_STREAMING_BLOCK_LOAD_AB=PASS" << std::endl;
}

TEST(MiniMaxH3C6aGoldenTest, MatchesOfficialTiledVideoVAEFixture) {
  const char* golden_value = std::getenv("MINIMAX_H3_VIDEO_VAE_GOLDEN");
  const char* checkpoint_value = std::getenv("MINIMAX_H3_CHECKPOINT");
  if (golden_value == nullptr || std::string(golden_value).empty() ||
      checkpoint_value == nullptr || std::string(checkpoint_value).empty()) {
    GTEST_SKIP() << "Set MINIMAX_H3_VIDEO_VAE_GOLDEN and "
                    "MINIMAX_H3_CHECKPOINT for the real NPU C6a Gate";
  }
  std::filesystem::path golden_path(golden_value);
  if (std::filesystem::is_directory(golden_path)) {
    golden_path /= "minimax_h3_video_vae_reference.safetensors";
  }
  const std::unique_ptr<StateDict> golden =
      StateDictFromSafeTensor::load(golden_path.string());
  ASSERT_NE(golden, nullptr);

  auto loader = std::make_unique<DiTModelLoader>(checkpoint_value);
  DiTModelContext context(ParallelArgs(0, 1, nullptr),
                          loader->get_model_args(),
                          loader->get_quant_args(),
                          torch::TensorOptions()
                              .device(torch::Device("npu:0"))
                              .dtype(torch::kBFloat16),
                          DiTCacheConfig(),
                          loader->get_model_type());
  MiniMaxH3Pipeline pipeline(context);
  pipeline->load_model(std::move(loader));
  pipeline->load_c6a_video_vae();
  MiniMaxH3VideoVAE vae = pipeline->c6a_video_vae();
  ASSERT_TRUE(vae->is_loaded());

  bool gate_pass = true;
  const auto report = [&golden, &gate_pass](const std::string& name,
                                            const torch::Tensor& actual) {
    const torch::Tensor* expected = find_raw_tensor(*golden, name);
    EXPECT_NE(expected, nullptr) << name;
    if (expected == nullptr) {
      gate_pass = false;
      return;
    }
    EXPECT_EQ(actual.sizes(), expected->sizes()) << name;
    EXPECT_EQ(actual.scalar_type(), expected->scalar_type()) << name;
    EXPECT_TRUE(torch::isfinite(actual).all().item<bool>()) << name;
    if (actual.sizes() != expected->sizes() ||
        actual.scalar_type() != expected->scalar_type()) {
      gate_pass = false;
      return;
    }
    const H3ComparisonMetrics metrics =
        h3_comparison_metrics(actual, *expected);
    std::cout << "H3-C6a node=" << name
              << " relative_l2=" << metrics.relative_l2
              << " cosine=" << metrics.cosine << " max_abs=" << metrics.max_abs
              << std::endl;
    H3ComparisonThreshold threshold{0.0, 1.0};
    if (name.ends_with(".rope_cos") || name.ends_with(".rope_sin")) {
      threshold = {1e-7, 0.999999999};
    } else if (name.ends_with(".block_00_output")) {
      threshold = {0.0004, 0.9999999};
    } else if (name.ends_with(".block_35_output")) {
      threshold = {0.0008, 0.9999995};
    } else if (name == "decoder.raw") {
      threshold = {0.001, 0.999999};
    } else if (name == "decoder.postprocessed") {
      threshold = {0.0005, 0.999999};
    }
    const bool passed = threshold.relative_l2 == 0.0
                            ? torch::equal(actual.to(torch::kCPU), *expected)
                            : metrics.relative_l2 <= threshold.relative_l2 &&
                                  metrics.cosine >= threshold.minimum_cosine;
    if (!passed) {
      ADD_FAILURE() << "H3-C6a node failed Gate: " << name
                    << " relative_l2=" << metrics.relative_l2
                    << " threshold=" << threshold.relative_l2
                    << " cosine=" << metrics.cosine
                    << " minimum=" << threshold.minimum_cosine;
      gate_pass = false;
    }
  };

  int64_t encoder_tile = 0;
  int64_t decoder_tile = 0;
  int64_t decoder_rope = 0;
  int64_t decoder_block0 = 0;
  int64_t decoder_block35 = 0;
  const MiniMaxH3VideoVAETraceHook hook =
      [&](std::string_view name, int64_t index, const torch::Tensor& value) {
        if (name == "encoder.quant_conv") {
          std::ostringstream key;
          key << "encoder.tile_call_" << std::setw(2) << std::setfill('0')
              << encoder_tile++ << ".moments";
          report(key.str(), value);
        } else if (name == "decoder.post_quant_conv") {
          std::ostringstream key;
          key << "decoder.tile_call_" << std::setw(2) << std::setfill('0')
              << decoder_tile++ << ".post_quant";
          report(key.str(), value);
        } else if (name == "decoder.rope.position_ids") {
          std::ostringstream key;
          key << "decoder.tile_call_" << std::setw(2) << std::setfill('0')
              << decoder_rope << ".rope_position_ids";
          report(key.str(), value);
        } else if (name == "decoder.rope.cos") {
          std::ostringstream key;
          key << "decoder.tile_call_" << std::setw(2) << std::setfill('0')
              << decoder_rope << ".rope_cos";
          report(key.str(), value);
        } else if (name == "decoder.rope.sin") {
          std::ostringstream key;
          key << "decoder.tile_call_" << std::setw(2) << std::setfill('0')
              << decoder_rope++ << ".rope_sin";
          report(key.str(), value);
        } else if (name == "decoder.block" && index == 0) {
          std::ostringstream key;
          key << "decoder.tile_call_" << std::setw(2) << std::setfill('0')
              << decoder_block0++ << ".block_00_output";
          report(key.str(), value);
        } else if (name == "decoder.block" && index == 35) {
          std::ostringstream key;
          key << "decoder.tile_call_" << std::setw(2) << std::setfill('0')
              << decoder_block35++ << ".block_35_output";
          report(key.str(), value);
        }
      };

  const torch::Tensor input = golden->get_tensor("input.imagenet_normalized")
                                  .to(torch::Device("npu:0"));
  MiniMaxH3VAEDiagonalGaussianDistribution posterior = vae->encode(input, hook);
  report("encoder.final_moments", posterior.parameters());
  report("posterior.mean", posterior.mean());
  report("posterior.raw_logvar", posterior.raw_logvar());
  report("posterior.logvar", posterior.logvar());
  report("posterior.std", posterior.std());
  const torch::Tensor epsilon =
      golden->get_tensor("posterior.epsilon").to(input.device());
  const torch::Tensor sample = posterior.sample(epsilon);
  report("posterior.sample", sample);
  const torch::Tensor rounded_fp16 = sample.to(torch::kFloat16);
  const torch::Tensor rounded_fp32 = rounded_fp16.to(torch::kFloat32);
  report("posterior.rounded_fp16", rounded_fp16);
  report("posterior.rounded_fp32", rounded_fp32);
  const torch::Tensor normalized =
      MiniMaxH3VideoVAEImpl::normalize_latents(rounded_fp32);
  report("posterior.normalized", normalized);

  const torch::Tensor independent =
      golden->get_tensor("decoder.independent_normalized_latent")
          .to(input.device());
  const torch::Tensor denormalized =
      MiniMaxH3VideoVAEImpl::denormalize_latents(independent);
  report("decoder.denormalized_latent", denormalized);
  const torch::Tensor decoded = vae->decode(denormalized, hook);
  report("decoder.raw", decoded);
  const torch::Tensor postprocessed =
      MiniMaxH3VideoVAEImpl::imagenet_postprocess(decoded);
  report("decoder.postprocessed", postprocessed);

  gate_pass = gate_pass && encoder_tile == 12 && decoder_tile == 8 &&
              decoder_rope == 8 && decoder_block0 == 8 && decoder_block35 == 8;
  ASSERT_TRUE(gate_pass);
  std::cout << "H3_C6A_VIDEO_VAE=PASS" << std::endl;
}

TEST(MiniMaxH3C6aTargetTest, DecodesProductionGeometryWithRealWeights) {
  const char* checkpoint_value = std::getenv("MINIMAX_H3_CHECKPOINT");
  const char* smoke_value = std::getenv("MINIMAX_H3_VIDEO_VAE_TARGET_SMOKE");
  if (checkpoint_value == nullptr || std::string(checkpoint_value).empty() ||
      smoke_value == nullptr || std::string(smoke_value) != "1") {
    GTEST_SKIP() << "Set checkpoint and MINIMAX_H3_VIDEO_VAE_TARGET_SMOKE=1";
  }
  auto loader = std::make_unique<DiTModelLoader>(checkpoint_value);
  DiTModelContext context(ParallelArgs(0, 1, nullptr),
                          loader->get_model_args(),
                          loader->get_quant_args(),
                          torch::TensorOptions()
                              .device(torch::Device("npu:0"))
                              .dtype(torch::kBFloat16),
                          DiTCacheConfig(),
                          loader->get_model_type());
  MiniMaxH3Pipeline pipeline(context);
  pipeline->load_model(std::move(loader));
  pipeline->load_c6a_video_vae();
  MiniMaxH3VideoVAE vae = pipeline->c6a_video_vae();

  constexpr int64_t kElements = 24 * 37 * 48 * 84;
  torch::Tensor normalized = torch::arange(kElements, torch::kFloat32)
                                 .remainder(257)
                                 .sub(128)
                                 .div(128)
                                 .view({1, 24, 37, 48, 84})
                                 .to(torch::Device("npu:0"));
  int64_t post_quant_calls = 0;
  int64_t block0_calls = 0;
  int64_t block35_calls = 0;
  const MiniMaxH3VideoVAETraceHook hook =
      [&](std::string_view name, int64_t index, const torch::Tensor&) {
        if (name == "decoder.post_quant_conv") {
          ++post_quant_calls;
        } else if (name == "decoder.block" && index == 0) {
          ++block0_calls;
        } else if (name == "decoder.block" && index == 35) {
          ++block35_calls;
        }
      };
  const torch::Tensor decoded = vae->decode_normalized(normalized, hook);
  EXPECT_EQ(decoded.sizes().vec(),
            (std::vector<int64_t>{1, 3, 124, 768, 1344}));
  EXPECT_EQ(decoded.scalar_type(), torch::kFloat32);
  EXPECT_TRUE(torch::isfinite(decoded).all().item<bool>());
  EXPECT_GE(decoded.min().item<float>(), 0.0F);
  EXPECT_LE(decoded.max().item<float>(), 1.0F);
  EXPECT_EQ(post_quant_calls, 196);
  EXPECT_EQ(block0_calls, 196);
  EXPECT_EQ(block35_calls, 196);
  std::cout << "H3_C6A_TARGET_GEOMETRY=PASS" << std::endl;
}

TEST(MiniMaxH3C6bGoldenTest, MatchesOfficialAudioVAEFixture) {
  const char* golden_value = std::getenv("MINIMAX_H3_AUDIO_VAE_GOLDEN");
  const char* checkpoint_value = std::getenv("MINIMAX_H3_CHECKPOINT");
  if (golden_value == nullptr || std::string(golden_value).empty() ||
      checkpoint_value == nullptr || std::string(checkpoint_value).empty()) {
    GTEST_SKIP() << "Set MINIMAX_H3_AUDIO_VAE_GOLDEN and "
                    "MINIMAX_H3_CHECKPOINT for the real NPU C6b Gate";
  }
  std::filesystem::path golden_path(golden_value);
  if (std::filesystem::is_directory(golden_path)) {
    golden_path /= "minimax_h3_audio_vae_reference.safetensors";
  }
  validate_h3_audio_golden_manifest(golden_path);
  const std::unique_ptr<StateDict> golden =
      StateDictFromSafeTensor::load(golden_path.string());
  ASSERT_NE(golden, nullptr);

  auto loader = std::make_unique<DiTModelLoader>(checkpoint_value);
  DiTModelContext context(ParallelArgs(0, 1, nullptr),
                          loader->get_model_args(),
                          loader->get_quant_args(),
                          torch::TensorOptions()
                              .device(torch::Device("npu:0"))
                              .dtype(torch::kBFloat16),
                          DiTCacheConfig(),
                          loader->get_model_type());
  MiniMaxH3Pipeline pipeline(context);
  pipeline->load_model(std::move(loader));
  pipeline->load_c6b_audio_vae();
  MiniMaxH3AudioVAE vae = pipeline->c6b_audio_vae();
  ASSERT_TRUE(vae->is_loaded());

  bool gate_pass = true;
  const auto report = [&golden, &gate_pass](const std::string& name,
                                            const torch::Tensor& actual) {
    const torch::Tensor* expected = find_raw_tensor(*golden, name);
    EXPECT_NE(expected, nullptr) << name;
    if (expected == nullptr) {
      gate_pass = false;
      return;
    }
    EXPECT_EQ(actual.sizes(), expected->sizes()) << name;
    EXPECT_EQ(actual.scalar_type(), expected->scalar_type()) << name;
    EXPECT_TRUE(torch::isfinite(actual).all().item<bool>()) << name;
    if (actual.sizes() != expected->sizes() ||
        actual.scalar_type() != expected->scalar_type()) {
      gate_pass = false;
      return;
    }
    const H3ComparisonMetrics metrics =
        h3_comparison_metrics(actual, *expected);
    std::cout << "H3-C6b node=" << name
              << " relative_l2=" << metrics.relative_l2
              << " cosine=" << metrics.cosine << " max_abs=" << metrics.max_abs
              << std::endl;
    static const std::unordered_set<std::string> kExactNodes = {
        "encoder.input_padded",
        "encoder.block_00",
        "decoder.denormalized_latent",
        "decoder.dec_in",
        "decoder.conv_pre",
        "decoder.stage_0.upsample",
    };
    H3ComparisonThreshold threshold{0.0002, 0.999999};
    if (name == "posterior.std") {
      threshold = {0.0003, 0.999999};
    } else if (name.starts_with("decoder.stage_") ||
               name == "decoder.final_pre_clamp" ||
               name == "decoder.final_clamped" || name == "pipeline.stereo") {
      threshold = {0.0008, 0.999999};
    }
    bool passed = kExactNodes.contains(name)
                      ? torch::equal(actual.to(torch::kCPU), *expected)
                      : metrics.relative_l2 <= threshold.relative_l2 &&
                            metrics.cosine >= threshold.minimum_cosine;
    if (name == "decoder.final_pre_clamp" || name == "decoder.final_clamped" ||
        name == "pipeline.stereo") {
      passed = passed && metrics.max_abs <= 0.0015;
    }
    if (!passed) {
      ADD_FAILURE() << "H3-C6b node failed Gate: " << name
                    << " relative_l2=" << metrics.relative_l2
                    << " threshold=" << threshold.relative_l2
                    << " cosine=" << metrics.cosine
                    << " minimum=" << threshold.minimum_cosine;
      gate_pass = false;
    }
  };

  std::unordered_map<std::string, int64_t> trace_counts;
  const MiniMaxH3AudioVAETraceHook hook =
      [&report, &trace_counts](
          std::string_view name, int64_t index, const torch::Tensor& value) {
        ++trace_counts[std::string(name) + ":" + std::to_string(index)];
        if (name == "encoder.padded_input") {
          report("encoder.input_padded", value);
        } else if (name == "encoder.block") {
          std::ostringstream key;
          key << "encoder.block_" << std::setw(2) << std::setfill('0') << index;
          report(key.str(), value);
        } else if (name.starts_with("pre_block.")) {
          report(std::string(name), value);
        } else if (name == "posterior.mean") {
          report("posterior.mean_head", value);
        } else if (name == "posterior.logs") {
          report("posterior.logs_head", value);
        } else if (name == "decoder.denormalized_latents") {
          report("decoder.denormalized_latent", value);
        } else if (name == "decoder.dec_in_proj") {
          report("decoder.dec_in", value);
        } else if (name == "decoder.conv_pre") {
          report("decoder.conv_pre", value);
        } else if (name == "decoder.ups") {
          report("decoder.stage_" + std::to_string(index) + ".upsample", value);
        } else if (name == "decoder.stage") {
          report("decoder.stage_" + std::to_string(index) + ".average", value);
        } else if (name == "decoder.final_pre_clamp") {
          report("decoder.final_pre_clamp", value);
        } else if (name == "decoder.final_clamped") {
          report("decoder.final_clamped", value);
        } else if (name == "decoder.stereo") {
          report("pipeline.stereo", value);
        }
      };

  const torch::Device device("npu:0");
  const torch::Tensor network_input =
      golden->get_tensor("preprocess.mono_network_batch").to(device);
  MiniMaxH3AudioDiagonalGaussianDistribution posterior =
      vae->encode(network_input, hook);
  report("posterior.mean", posterior.mean());
  report("posterior.logs", posterior.logs());
  report("posterior.std", posterior.std());
  report("posterior.mode", posterior.mode());
  const torch::Tensor epsilon =
      golden->get_tensor("posterior.epsilon").to(device);
  report("posterior.sample", posterior.sample(epsilon));
  const torch::Tensor normalized_mode =
      MiniMaxH3AudioVAEImpl::normalize_latents(posterior.mode());
  report("posterior.normalized_mode", normalized_mode);
  const torch::Tensor rows =
      normalized_mode.transpose(1, 2).reshape({-1, 32}).contiguous();
  report("posterior.normalized_channel_major_rows", rows);

  const torch::Tensor normalized_decode =
      golden->get_tensor("decoder.independent_normalized_latent").to(device);
  const torch::Tensor stereo = vae->decode_normalized(normalized_decode, hook);
  report("pipeline.stereo", stereo);
  std::unordered_map<std::string, int64_t> expected_trace_counts = {
      {"encoder.padded_input:-1", 1},
      {"pre_block.qkv:-1", 1},
      {"pre_block.query:-1", 1},
      {"pre_block.key:-1", 1},
      {"pre_block.value:-1", 1},
      {"pre_block.projection_branch:-1", 1},
      {"pre_block.attention_branch:-1", 1},
      {"pre_block.mlp_branch:-1", 1},
      {"pre_block.final:-1", 1},
      {"encoder.pre_block:-1", 1},
      {"posterior.mean:-1", 1},
      {"posterior.logs:-1", 1},
      {"decoder.denormalized_latents:-1", 1},
      {"decoder.dec_in_proj:-1", 1},
      {"decoder.conv_pre:-1", 1},
      {"decoder.activation_post:-1", 1},
      {"decoder.final_pre_clamp:-1", 1},
      {"decoder.final_clamped:-1", 1},
      {"decoder.raw:-1", 1},
      {"decoder.stereo:-1", 1},
  };
  for (int64_t index = 0; index < 8; ++index) {
    expected_trace_counts.emplace("encoder.block:" + std::to_string(index), 1);
  }
  for (int64_t index = 0; index < 7; ++index) {
    expected_trace_counts.emplace("decoder.ups:" + std::to_string(index), 1);
    expected_trace_counts.emplace("decoder.stage:" + std::to_string(index), 1);
  }
  if (trace_counts != expected_trace_counts) {
    ADD_FAILURE() << "H3-C6b trace event inventory is incomplete";
    gate_pass = false;
  }
  const torch::Tensor actual_waveform =
      stereo.to(torch::kCPU).to(torch::kFloat64);
  const torch::Tensor expected_waveform =
      golden->get_tensor("pipeline.stereo").to(torch::kFloat64);
  const torch::Tensor actual_centered =
      actual_waveform - actual_waveform.mean(-1, true);
  const torch::Tensor expected_centered =
      expected_waveform - expected_waveform.mean(-1, true);
  const torch::Tensor correlation =
      (actual_centered * expected_centered).sum(-1) /
      (actual_centered.square().sum(-1).sqrt() *
       expected_centered.square().sum(-1).sqrt());
  const double minimum_correlation = correlation.min().item<double>();
  const torch::Tensor rms_ratio = actual_waveform.square().mean(-1).sqrt() /
                                  expected_waveform.square().mean(-1).sqrt();
  const double minimum_rms_ratio = rms_ratio.min().item<double>();
  const double maximum_rms_ratio = rms_ratio.max().item<double>();
  std::cout << "H3-C6b waveform correlation=" << minimum_correlation
            << " rms_ratio_min=" << minimum_rms_ratio
            << " rms_ratio_max=" << maximum_rms_ratio << std::endl;
  if (minimum_correlation < 0.99999 || minimum_rms_ratio < 0.999 ||
      maximum_rms_ratio > 1.001) {
    ADD_FAILURE() << "H3-C6b waveform correlation or RMS ratio failed";
    gate_pass = false;
  }
  for (int64_t fft_size : {512, 1024, 2048}) {
    const int64_t hop = fft_size / 4;
    const torch::Tensor window = torch::hann_window(fft_size, torch::kFloat64);
    const torch::Tensor actual_frames =
        actual_waveform.unfold(-1, fft_size, hop) * window;
    const torch::Tensor expected_frames =
        expected_waveform.unfold(-1, fft_size, hop) * window;
    const torch::Tensor actual_magnitude =
        torch::abs(torch::fft::rfft(actual_frames));
    const torch::Tensor expected_magnitude =
        torch::abs(torch::fft::rfft(expected_frames));
    const double spectral_cosine =
        (actual_magnitude.flatten().dot(expected_magnitude.flatten()) /
         (actual_magnitude.norm() * expected_magnitude.norm()))
            .item<double>();
    std::cout << "H3-C6b spectral_cosine fft=" << fft_size
              << " value=" << spectral_cosine << std::endl;
    if (spectral_cosine < 0.9999) {
      ADD_FAILURE() << "H3-C6b spectral cosine failed for FFT " << fft_size;
      gate_pass = false;
    }
  }
  torch::npu::synchronize();
  ASSERT_TRUE(gate_pass);
  std::cout << "H3_C6B_AUDIO_VAE=PASS" << std::endl;
  std::cout << "G7_DUAL_VAE=PASS" << std::endl;
}

TEST(MiniMaxH3C6bTargetTest, DecodesProductionAudioGeometryWithRealWeights) {
  const char* checkpoint_value = std::getenv("MINIMAX_H3_CHECKPOINT");
  const char* golden_value = std::getenv("MINIMAX_H3_AUDIO_VAE_GOLDEN");
  const char* smoke_value = std::getenv("MINIMAX_H3_AUDIO_VAE_TARGET_SMOKE");
  if (checkpoint_value == nullptr || std::string(checkpoint_value).empty() ||
      golden_value == nullptr || std::string(golden_value).empty() ||
      smoke_value == nullptr || std::string(smoke_value) != "1") {
    GTEST_SKIP() << "Set checkpoint, C6b Golden, and "
                    "MINIMAX_H3_AUDIO_VAE_TARGET_SMOKE=1";
  }
  std::filesystem::path golden_path(golden_value);
  if (std::filesystem::is_directory(golden_path)) {
    golden_path /= "minimax_h3_audio_vae_reference.safetensors";
  }
  validate_h3_audio_golden_manifest(golden_path);
  const std::unique_ptr<StateDict> golden =
      StateDictFromSafeTensor::load(golden_path.string());
  ASSERT_NE(golden, nullptr);
  auto loader = std::make_unique<DiTModelLoader>(checkpoint_value);
  DiTModelContext context(ParallelArgs(0, 1, nullptr),
                          loader->get_model_args(),
                          loader->get_quant_args(),
                          torch::TensorOptions()
                              .device(torch::Device("npu:0"))
                              .dtype(torch::kBFloat16),
                          DiTCacheConfig(),
                          loader->get_model_type());
  MiniMaxH3Pipeline pipeline(context);
  pipeline->load_model(std::move(loader));
  pipeline->load_c6b_audio_vae();
  MiniMaxH3AudioVAE vae = pipeline->c6b_audio_vae();

  const torch::Tensor normalized =
      golden->get_tensor("production.normalized_latent")
          .to(torch::Device("npu:0"));
  const torch::Tensor stereo = vae->decode_normalized(normalized);
  EXPECT_EQ(stereo.sizes().vec(), (std::vector<int64_t>{1, 2, 165600}));
  EXPECT_EQ(stereo.scalar_type(), torch::kFloat32);
  EXPECT_TRUE(torch::isfinite(stereo).all().item<bool>());
  EXPECT_GE(stereo.min().item<float>(), -1.0F);
  EXPECT_LE(stereo.max().item<float>(), 1.0F);
  const torch::Tensor expected =
      golden->get_tensor("production.pipeline_stereo");
  const H3ComparisonMetrics metrics = h3_comparison_metrics(stereo, expected);
  std::cout << "H3-C6b target relative_l2=" << metrics.relative_l2
            << " cosine=" << metrics.cosine << " max_abs=" << metrics.max_abs
            << std::endl;
  EXPECT_LE(metrics.relative_l2, 0.001);
  EXPECT_GE(metrics.cosine, 0.999999);
  EXPECT_LE(metrics.max_abs, 0.002);
  const torch::Tensor actual_fp64 = stereo.to(torch::kCPU).to(torch::kFloat64);
  const torch::Tensor actual_centered =
      actual_fp64 - actual_fp64.mean(-1, true);
  const torch::Tensor expected_fp64 = expected.to(torch::kFloat64);
  const torch::Tensor expected_centered =
      expected_fp64 - expected_fp64.mean(-1, true);
  const double minimum_correlation =
      ((actual_centered * expected_centered).sum(-1) /
       (actual_centered.square().sum(-1).sqrt() *
        expected_centered.square().sum(-1).sqrt()))
          .min()
          .item<double>();
  std::cout << "H3-C6b target correlation=" << minimum_correlation << std::endl;
  EXPECT_GE(minimum_correlation, 0.99999);
  torch::npu::synchronize();
  std::cout << "H3_C6B_TARGET_GEOMETRY=PASS" << std::endl;
}

TEST(MiniMaxH3PipelineTest, ForwardFailsInsteadOfReturningFakeMedia) {
  try {
    MiniMaxH3PipelineImpl::throw_forward_unavailable();
    FAIL() << "Expected H3-C6b forward failure";
  } catch (const std::logic_error& error) {
    EXPECT_NE(std::string(error.what()).find("no production Ref2VA"),
              std::string::npos);
  }
}

}  // namespace
}  // namespace xllm
