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

#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/util/json_reader.h"
#include "models/model_registry.h"

namespace xllm {

struct MiniMaxH3TransformerConfig {
  static constexpr int64_t kNumLayers = 50;
  static constexpr int64_t kNumRefinerLayers = 2;
  static constexpr int64_t kHiddenSize = 5376;
  static constexpr int64_t kNumAttentionHeads = 56;
  static constexpr int64_t kAttentionHeadDim = 128;
  static constexpr int64_t kFfnHiddenSize = 14336;
  static constexpr int64_t kVideoLatentDim = 24;
  static constexpr int64_t kAudioLatentDim = 32;
  static constexpr int64_t kTextDim = 5120;
  static constexpr int64_t kTimestepInputDim = 256;
  static constexpr int64_t kTimeEmbedHiddenSize = 5376;
  static constexpr int64_t kTimeEmbedDim = 2688;
  static constexpr int64_t kAdalnOutFeatures = 96768;
  static constexpr int64_t kFinalAdalnOutFeatures = 10752;
  static constexpr int64_t kRopeInvFreqLen = 16;
  static constexpr double kRopeTheta = 10000.0;
  static constexpr double kNormEps = 1e-5;

  int64_t num_layers = kNumLayers;
  int64_t num_refiner_layers = kNumRefinerLayers;
  int64_t hidden_size = kHiddenSize;
  int64_t num_attention_heads = kNumAttentionHeads;
  int64_t attention_head_dim = kAttentionHeadDim;
  int64_t ffn_hidden_size = kFfnHiddenSize;
  int64_t video_latent_dim = kVideoLatentDim;
  int64_t audio_latent_dim = kAudioLatentDim;
  std::vector<int64_t> patch_size = {1, 2, 2};
  int64_t text_dim = kTextDim;
  int64_t timestep_input_dim = kTimestepInputDim;
  int64_t time_embed_hidden_size = kTimeEmbedHiddenSize;
  int64_t time_embed_dim = kTimeEmbedDim;
  int64_t adaln_out_features = kAdalnOutFeatures;
  int64_t final_adaln_out_features = kFinalAdalnOutFeatures;
  int64_t rope_inv_freq_len = kRopeInvFreqLen;
  double rope_theta = kRopeTheta;
  double norm_eps = kNormEps;
  double qk_norm_eps = kNormEps;
  double final_norm_eps = kNormEps;

  static MiniMaxH3TransformerConfig from_model_args(const ModelArgs& args) {
    MiniMaxH3TransformerConfig config;
    config.num_layers = args.h3_num_layers();
    config.num_refiner_layers = args.h3_token_refiner_num_layers();
    config.hidden_size = args.h3_hidden_size();
    config.num_attention_heads = args.h3_num_attention_heads();
    config.attention_head_dim = args.h3_attention_head_dim();
    config.ffn_hidden_size = args.h3_ffn_hidden_size();
    config.video_latent_dim = args.h3_video_latent_dim();
    config.audio_latent_dim = args.h3_audio_latent_dim();
    config.patch_size = args.h3_patch_size();
    config.text_dim = args.h3_text_dim();
    config.timestep_input_dim = args.h3_timestep_input_dim();
    config.time_embed_hidden_size = args.h3_time_embed_hidden_size();
    config.time_embed_dim = args.h3_time_embed_dim();
    config.adaln_out_features = args.h3_adaln_out_features();
    config.final_adaln_out_features = args.h3_final_adaln_out_features();
    config.rope_inv_freq_len = args.h3_rope_inv_freq_len();
    config.rope_theta = args.h3_rope_theta();
    config.norm_eps = args.h3_norm_eps();
    config.qk_norm_eps = args.h3_qk_norm_eps();
    config.final_norm_eps = args.h3_final_norm_eps();
    config.validate();
    return config;
  }

  static bool load_original_model_args(const JsonReader& json,
                                       ModelArgs* args) {
    return load_model_args(json, args, /*converted=*/false);
  }

  static bool load_converted_model_args(const JsonReader& json,
                                        ModelArgs* args) {
    return load_model_args(json, args, /*converted=*/true);
  }

  void validate() const {
    require_equal("num_layers", num_layers, kNumLayers);
    require_equal(
        "token_refiner_num_layers", num_refiner_layers, kNumRefinerLayers);
    require_equal("hidden_size", hidden_size, kHiddenSize);
    require_equal(
        "num_attention_heads", num_attention_heads, kNumAttentionHeads);
    require_equal("attention_head_dim", attention_head_dim, kAttentionHeadDim);
    require_equal("ffn_hidden_size", ffn_hidden_size, kFfnHiddenSize);
    require_equal("latents_dim", video_latent_dim, kVideoLatentDim);
    require_equal("audio_latents_dim", audio_latent_dim, kAudioLatentDim);
    require_equal("patch_size", patch_size, std::vector<int64_t>({1, 2, 2}));
    require_equal("text_dim", text_dim, kTextDim);
    require_equal("timestep_input_dim", timestep_input_dim, kTimestepInputDim);
    require_equal(
        "time_embed_hidden_size", time_embed_hidden_size, kTimeEmbedHiddenSize);
    require_equal("time_embed_dim", time_embed_dim, kTimeEmbedDim);
    require_equal("adaln_out_features", adaln_out_features, kAdalnOutFeatures);
    require_equal("final_adaln_out_features",
                  final_adaln_out_features,
                  kFinalAdalnOutFeatures);
    require_equal("rope_inv_freq_len", rope_inv_freq_len, kRopeInvFreqLen);
    require_equal("rope_theta", rope_theta, kRopeTheta);
    require_equal("norm_eps", norm_eps, kNormEps);
    require_equal("qk_norm_eps", qk_norm_eps, kNormEps);
    require_equal("final_norm_eps", final_norm_eps, kNormEps);
  }

 private:
  template <typename T>
  static std::string format_value(const T& value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
  }

  static std::string format_value(const std::vector<int64_t>& value) {
    std::ostringstream stream;
    stream << "[";
    for (size_t i = 0; i < value.size(); ++i) {
      if (i > 0) {
        stream << ",";
      }
      stream << value[i];
    }
    stream << "]";
    return stream.str();
  }

  template <typename T>
  static void require_equal(const std::string& field,
                            const T& actual,
                            const T& expected) {
    if (actual != expected) {
      throw std::invalid_argument("MiniMax-H3 transformer config invariant `" +
                                  field + "` failed: expected " +
                                  format_value(expected) + ", got " +
                                  format_value(actual));
    }
  }

  template <typename T>
  static T require_value(const JsonReader& json, const std::string& field) {
    const auto value = json.value<T>(field);
    if (!value.has_value()) {
      throw std::invalid_argument(
          "MiniMax-H3 transformer config is missing required field `" + field +
          "`");
    }
    return *value;
  }

  static bool load_model_args(const JsonReader& json,
                              ModelArgs* args,
                              bool converted) {
    if (args == nullptr) {
      throw std::invalid_argument(
          "MiniMax-H3 transformer config requires non-null ModelArgs");
    }

    const std::string expected_class =
        converted ? "MiniMaxH3Transformer3DModel" : "MiniMaxH3DiTModel";
    const std::string class_name =
        require_value<std::string>(json, "_class_name");
    require_equal("_class_name", class_name, expected_class);

    args->model_type() = expected_class;
    args->h3_num_layers() = require_value<int64_t>(json, "num_layers");
    args->h3_token_refiner_num_layers() = require_value<int64_t>(
        json, converted ? "num_refiner_layers" : "token_refiner_num_layers");
    args->h3_hidden_size() = require_value<int64_t>(json, "hidden_size");
    args->h3_num_attention_heads() =
        require_value<int64_t>(json, "num_attention_heads");
    args->h3_attention_head_dim() =
        require_value<int64_t>(json, "attention_head_dim");
    args->h3_ffn_hidden_size() =
        require_value<int64_t>(json, converted ? "ffn_dim" : "ffn_hidden_size");
    args->h3_video_latent_dim() =
        require_value<int64_t>(json, converted ? "in_channels" : "latents_dim");
    args->h3_audio_latent_dim() = require_value<int64_t>(
        json, converted ? "audio_in_channels" : "audio_latents_dim");
    args->h3_patch_size() =
        require_value<std::vector<int64_t>>(json, "patch_size");
    args->h3_text_dim() = require_value<int64_t>(json, "text_dim");
    args->h3_timestep_input_dim() = require_value<int64_t>(
        json, converted ? "freq_dim" : "timestep_input_dim");
    args->h3_time_embed_hidden_size() = require_value<int64_t>(
        json, converted ? "time_embed_hidden_dim" : "time_embed_hidden_size");
    args->h3_time_embed_dim() = require_value<int64_t>(json, "time_embed_dim");
    args->h3_adaln_out_features() =
        converted ? kAdalnOutFeatures
                  : require_value<int64_t>(json, "adaln_out_features");
    args->h3_final_adaln_out_features() =
        converted ? kFinalAdalnOutFeatures
                  : require_value<int64_t>(json, "final_adaln_out_features");
    args->h3_rope_inv_freq_len() = require_value<int64_t>(
        json, converted ? "rope_freq_dim" : "rope_inv_freq_len");
    args->h3_rope_theta() = json.value_or<double>("rope_theta", kRopeTheta);
    args->h3_norm_eps() = require_value<double>(json, "norm_eps");
    args->h3_qk_norm_eps() = require_value<double>(json, "qk_norm_eps");
    args->h3_final_norm_eps() = require_value<double>(json, "final_norm_eps");

    from_model_args(*args);
    return true;
  }
};

struct MiniMaxH3SourceTensorSpec {
  std::string name;
  std::vector<int64_t> shape;
  torch::ScalarType dtype;
};

struct MiniMaxH3SourceLayoutSummary {
  size_t shard_count = 0;
  size_t tensor_count = 0;
  size_t bfloat16_tensor_count = 0;
  size_t float32_tensor_count = 0;
  int64_t transformer_layer_count = 0;
  int64_t token_refiner_layer_count = 0;
};

class MiniMaxH3SourceLayoutValidator {
 public:
  static constexpr size_t kExpectedTensorCount = 535;
  static constexpr size_t kExpectedBFloat16TensorCount = 522;
  static constexpr size_t kExpectedFloat32TensorCount = 13;

  static std::vector<MiniMaxH3SourceTensorSpec> expected_source_tensors() {
    constexpr int64_t hidden = MiniMaxH3TransformerConfig::kHiddenSize;
    constexpr int64_t head_dim = MiniMaxH3TransformerConfig::kAttentionHeadDim;
    constexpr int64_t inner = MiniMaxH3TransformerConfig::kNumAttentionHeads *
                              MiniMaxH3TransformerConfig::kAttentionHeadDim;
    constexpr int64_t ffn = MiniMaxH3TransformerConfig::kFfnHiddenSize;
    constexpr int64_t time = MiniMaxH3TransformerConfig::kTimeEmbedDim;

    std::vector<MiniMaxH3SourceTensorSpec> specs;
    specs.reserve(kExpectedTensorCount);
    auto add = [&specs](std::string name,
                        std::vector<int64_t> shape,
                        torch::ScalarType dtype) {
      specs.push_back({std::move(name), std::move(shape), dtype});
    };

    add("video_patch_proj.weight", {hidden, 96}, torch::kFloat32);
    add("video_patch_proj.bias", {hidden}, torch::kFloat32);
    add("audio_patch_proj.weight", {hidden, 32}, torch::kFloat32);
    add("audio_patch_proj.bias", {hidden}, torch::kFloat32);
    add("condition_proj.weight", {hidden, 5120}, torch::kBFloat16);
    add("condition_proj.bias", {hidden}, torch::kBFloat16);
    add("time_embedder.proj_in.weight", {hidden, 256}, torch::kFloat32);
    add("time_embedder.proj_in.bias", {hidden}, torch::kFloat32);
    add("time_embedder.proj_out.weight", {time, hidden}, torch::kFloat32);
    add("time_embedder.proj_out.bias", {time}, torch::kFloat32);
    add("rope.inv_freq", {16}, torch::kFloat32);
    add("token_refiner.final_norm.weight", {hidden}, torch::kBFloat16);
    add("final_layer.norm.weight", {hidden}, torch::kBFloat16);
    add("final_layer.adaln_proj.linear.weight",
        {10752, time},
        torch::kBFloat16);
    add("final_layer.adaln_proj.linear.bias", {10752}, torch::kBFloat16);
    add("final_layer.video_out.weight", {96, hidden}, torch::kFloat32);
    add("final_layer.video_out.bias", {96}, torch::kFloat32);
    add("final_layer.audio_out.weight", {32, hidden}, torch::kFloat32);
    add("final_layer.audio_out.bias", {32}, torch::kFloat32);

    auto add_common_block = [&add](const std::string& prefix) {
      add(prefix + ".norm1.weight", {hidden}, torch::kBFloat16);
      add(prefix + ".norm2.weight", {hidden}, torch::kBFloat16);
      add(prefix + ".attn.q_norm.weight", {head_dim}, torch::kBFloat16);
      add(prefix + ".attn.k_norm.weight", {head_dim}, torch::kBFloat16);
      add(prefix + ".attn.qkv_proj.weight",
          {3 * inner, hidden},
          torch::kBFloat16);
      add(prefix + ".attn.out_proj.weight", {hidden, inner}, torch::kBFloat16);
      add(prefix + ".mlp.fc1.weight", {2 * ffn, hidden}, torch::kBFloat16);
      add(prefix + ".mlp.fc2.weight", {hidden, ffn}, torch::kBFloat16);
    };

    for (int64_t layer = 0;
         layer < MiniMaxH3TransformerConfig::kNumRefinerLayers;
         ++layer) {
      add_common_block("token_refiner.blocks." + std::to_string(layer));
    }
    for (int64_t layer = 0; layer < MiniMaxH3TransformerConfig::kNumLayers;
         ++layer) {
      const std::string prefix = "blocks." + std::to_string(layer);
      add_common_block(prefix);
      add(prefix + ".adaln_proj.linear.weight",
          {MiniMaxH3TransformerConfig::kAdalnOutFeatures, time},
          torch::kBFloat16);
      add(prefix + ".adaln_proj.linear.bias",
          {MiniMaxH3TransformerConfig::kAdalnOutFeatures},
          torch::kBFloat16);
    }

    return specs;
  }

  static MiniMaxH3SourceLayoutSummary validate(
      const std::vector<std::unique_ptr<StateDict>>& shards) {
    std::vector<const StateDict*> shard_views;
    shard_views.reserve(shards.size());
    for (const auto& shard : shards) {
      if (shard == nullptr) {
        throw std::invalid_argument(
            "MiniMax-H3 transformer source layout contains a null shard");
      }
      shard_views.push_back(shard.get());
    }
    return validate(shard_views);
  }

  static MiniMaxH3SourceLayoutSummary validate(
      const std::vector<const StateDict*>& shards) {
    struct LocatedTensor {
      const torch::Tensor* tensor;
      size_t shard_index;
    };

    std::unordered_map<std::string, LocatedTensor> actual;
    for (size_t shard_index = 0; shard_index < shards.size(); ++shard_index) {
      const StateDict* shard = shards[shard_index];
      if (shard == nullptr) {
        throw std::invalid_argument(
            "MiniMax-H3 transformer source layout contains a null shard");
      }
      for (const auto& [name, tensor] : *shard) {
        auto [iterator, inserted] =
            actual.emplace(name, LocatedTensor{&tensor, shard_index});
        if (!inserted) {
          throw std::invalid_argument(
              "MiniMax-H3 transformer source layout duplicate tensor `" + name +
              "` in shards " + std::to_string(iterator->second.shard_index) +
              " and " + std::to_string(shard_index));
        }
      }
    }

    const auto specs = expected_source_tensors();
    std::unordered_map<std::string, const MiniMaxH3SourceTensorSpec*> expected;
    expected.reserve(specs.size());
    for (const auto& spec : specs) {
      expected.emplace(spec.name, &spec);
    }

    std::vector<std::string> missing;
    std::vector<std::string> unknown;
    for (const auto& spec : specs) {
      if (!actual.contains(spec.name)) {
        missing.push_back(spec.name);
      }
    }
    for (const auto& [name, located] : actual) {
      (void)located;
      if (!expected.contains(name)) {
        unknown.push_back(name);
      }
    }
    if (!missing.empty() || !unknown.empty() ||
        actual.size() != kExpectedTensorCount) {
      throw std::invalid_argument(
          "MiniMax-H3 transformer source layout key mismatch: expected " +
          std::to_string(kExpectedTensorCount) + ", got " +
          std::to_string(actual.size()) + ", missing=" + format_names(missing) +
          ", unknown=" + format_names(unknown));
    }

    MiniMaxH3SourceLayoutSummary summary;
    summary.shard_count = shards.size();
    summary.tensor_count = actual.size();
    summary.transformer_layer_count = MiniMaxH3TransformerConfig::kNumLayers;
    summary.token_refiner_layer_count =
        MiniMaxH3TransformerConfig::kNumRefinerLayers;

    for (const auto& spec : specs) {
      const torch::Tensor& tensor = *actual.at(spec.name).tensor;
      if (!tensor.defined()) {
        throw std::invalid_argument("MiniMax-H3 transformer source tensor `" +
                                    spec.name + "` is undefined");
      }
      if (tensor.sizes().vec() != spec.shape) {
        throw std::invalid_argument("MiniMax-H3 transformer source tensor `" +
                                    spec.name + "` shape mismatch: expected " +
                                    format_shape(spec.shape) + ", got " +
                                    format_shape(tensor.sizes().vec()));
      }
      if (tensor.scalar_type() != spec.dtype) {
        throw std::invalid_argument("MiniMax-H3 transformer source tensor `" +
                                    spec.name + "` dtype mismatch: expected " +
                                    c10::toString(spec.dtype) + ", got " +
                                    c10::toString(tensor.scalar_type()));
      }
      if (spec.dtype == torch::kBFloat16) {
        ++summary.bfloat16_tensor_count;
      } else if (spec.dtype == torch::kFloat32) {
        ++summary.float32_tensor_count;
      }
    }

    if (summary.bfloat16_tensor_count != kExpectedBFloat16TensorCount ||
        summary.float32_tensor_count != kExpectedFloat32TensorCount) {
      throw std::logic_error(
          "MiniMax-H3 internal source layout dtype-count contract is invalid");
    }
    return summary;
  }

 private:
  static std::string format_shape(const std::vector<int64_t>& shape) {
    std::ostringstream stream;
    stream << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) {
        stream << ",";
      }
      stream << shape[i];
    }
    stream << "]";
    return stream.str();
  }

  static std::string format_names(const std::vector<std::string>& names) {
    if (names.empty()) {
      return "[]";
    }
    constexpr size_t kMaxReportedNames = 5;
    std::ostringstream stream;
    stream << "[";
    const size_t count = std::min(names.size(), kMaxReportedNames);
    for (size_t i = 0; i < count; ++i) {
      if (i > 0) {
        stream << ",";
      }
      stream << names[i];
    }
    if (names.size() > count) {
      stream << ",...";
    }
    stream << "]";
    return stream.str();
  }
};

#if !defined(XLLM_MINIMAX_H3_DISABLE_REGISTRATION)
namespace {

REGISTER_MODEL_ARGS_LOADER_WITH_VARNAME(
    minimax_h3_dit_original,
    MiniMaxH3DiTModel,
    MiniMaxH3TransformerConfig::load_original_model_args);

REGISTER_MODEL_ARGS_LOADER_WITH_VARNAME(
    minimax_h3_dit_converted,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerConfig::load_converted_model_args);

REGISTER_MODEL_ARGS_WITH_VARNAME(minimax_h3_video_vae, MiniMaxH3VideoVAE, [&] {
  const auto latent_channels = json.value<int64_t>("latent_channels");
  if (!latent_channels.has_value()) {
    throw std::invalid_argument(
        "MiniMax-H3 video VAE config is missing latent_channels");
  }
  args->h3_video_latent_dim() = *latent_channels;
  if (args->h3_video_latent_dim() != 24) {
    throw std::invalid_argument(
        "MiniMax-H3 video VAE latent_channels "
        "must be 24");
  }
});

REGISTER_MODEL_ARGS_WITH_VARNAME(minimax_h3_audio_vae, MiniMaxH3AudioVAE, [&] {
  const auto latent_channels = json.value<int64_t>("latent_channels");
  if (!latent_channels.has_value()) {
    throw std::invalid_argument(
        "MiniMax-H3 audio VAE config is missing latent_channels");
  }
  args->h3_audio_latent_dim() = *latent_channels;
  if (args->h3_audio_latent_dim() != 32) {
    throw std::invalid_argument(
        "MiniMax-H3 audio VAE latent_channels "
        "must be 32");
  }
});

REGISTER_MODEL_ARGS_WITH_VARNAME(minimax_h3_text_encoder,
                                 MiniMaxH3Qwen3VLHFEncoder,
                                 [&] {
                                   const auto hidden_size = json.value<int64_t>(
                                       "text_config.hidden_size");
                                   if (!hidden_size.has_value()) {
                                     throw std::invalid_argument(
                                         "MiniMax-H3 text encoder config is "
                                         "missing text_config.hidden_size");
                                   }
                                   args->h3_text_dim() = *hidden_size;
                                   if (args->h3_text_dim() != 5120) {
                                     throw std::invalid_argument(
                                         "MiniMax-H3 text encoder hidden_size "
                                         "must be 5120");
                                   }
                                 });

}  // namespace
#endif

}  // namespace xllm
