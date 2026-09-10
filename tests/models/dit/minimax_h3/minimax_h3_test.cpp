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

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "models/dit/pipelines/pipeline_minimax_h3.h"

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

TEST(MiniMaxH3PipelineTest, ForwardFailsInsteadOfReturningFakeMedia) {
  try {
    MiniMaxH3PipelineImpl::throw_forward_unavailable();
    FAIL() << "Expected H3-C2 forward failure";
  } catch (const std::logic_error& error) {
    EXPECT_NE(std::string(error.what()).find("has no denoiser"),
              std::string::npos);
  }
}

}  // namespace
}  // namespace xllm
