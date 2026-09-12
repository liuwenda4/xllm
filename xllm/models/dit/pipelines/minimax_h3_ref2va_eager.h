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

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "models/dit/autoencoders/autoencoder_kl_minimax_h3.h"
#include "models/dit/autoencoders/autoencoder_kl_minimax_h3_audio.h"
#include "models/dit/transformers/minimax_h3_denoiser.h"
#include "models/dit/utils/minimax_h3_packing.h"

namespace xllm {

inline constexpr std::string_view kMiniMaxH3ConditionSchemaV1 =
    "xllm.minimax_h3.text_conditioning/v1";

enum class MiniMaxH3Ref2VASourceBackend : int32_t {
  OFFICIAL_HF = 0,
  XLLM_NATIVE = 1,
};

inline std::string_view minimax_h3_ref2va_source_backend_name(
    MiniMaxH3Ref2VASourceBackend backend) {
  switch (backend) {
    case MiniMaxH3Ref2VASourceBackend::OFFICIAL_HF:
      return "official_hf";
    case MiniMaxH3Ref2VASourceBackend::XLLM_NATIVE:
      return "xllm_native";
    default:
      throw std::invalid_argument(
          "MiniMax-H3 Ref2VA condition source backend is unsupported");
  }
}

struct MiniMaxH3Ref2VACondition {
  std::string schema;
  MiniMaxH3Ref2VASourceBackend source_backend =
      static_cast<MiniMaxH3Ref2VASourceBackend>(-1);
  torch::Tensor hidden;
  torch::Tensor token_tags;
};

struct MiniMaxH3Ref2VAEncodedReference {
  H3ReferenceBlockKind kind = static_cast<H3ReferenceBlockKind>(-1);
  torch::Tensor normalized_visual_latent;
  torch::Tensor normalized_audio_latent;
  torch::Tensor visual_anchor_noise;
};

struct MiniMaxH3Ref2VAPreparedTarget {
  torch::Tensor initial_normalized_video_latent;
  torch::Tensor initial_normalized_audio_latent;
};

struct MiniMaxH3Ref2VAEagerInput {
  MiniMaxH3Ref2VACondition condition;
  std::vector<MiniMaxH3Ref2VAEncodedReference> references;
  MiniMaxH3Ref2VAPreparedTarget target;
  std::optional<int64_t> sequence_length;
};

struct MiniMaxH3Ref2VACompactRows {
  torch::Tensor video_rows;
  torch::Tensor audio_rows;
};

struct MiniMaxH3Ref2VAOutputGeometry {
  int64_t video_frames = 0;
  int64_t video_height = 0;
  int64_t video_width = 0;
  int64_t video_fps = 24;
  int64_t audio_samples = 0;
  int64_t audio_sample_rate = MiniMaxH3AudioVAEConfig::kSampleRate;
};

struct MiniMaxH3Ref2VAEagerCounters {
  int64_t reference_blocks = 0;
  int64_t visual_references = 0;
  int64_t audio_references = 0;
  int64_t transformer_forwards = 0;
  int64_t block_forwards = 0;
  int64_t video_vae_decodes = 0;
  int64_t audio_vae_decodes = 0;
};

struct MiniMaxH3Ref2VAEagerOutput {
  std::string condition_schema;
  MiniMaxH3Ref2VASourceBackend source_backend =
      static_cast<MiniMaxH3Ref2VASourceBackend>(-1);
  H3PackedLayout layout;
  MiniMaxH3Ref2VACompactRows initial_compact_rows;
  MiniMaxH3Ref2VACompactRows final_compact_rows;
  torch::Tensor final_normalized_video_latent;
  torch::Tensor final_normalized_audio_latent;
  torch::Tensor video;
  torch::Tensor audio;
  MiniMaxH3Ref2VAOutputGeometry geometry;
  MiniMaxH3Ref2VAEagerCounters counters;
};

class MiniMaxH3Ref2VAEagerRunner final {
 public:
  static MiniMaxH3Ref2VAEagerOutput run(
      const MiniMaxH3Ref2VAEagerInput& input,
      const std::vector<std::unique_ptr<StateDict>>& transformer_shards,
      MiniMaxH3StreamingDenoiser& denoiser,
      MiniMaxH3VideoVAE& video_vae,
      MiniMaxH3AudioVAE& audio_vae) {
    if (!denoiser || !video_vae || !audio_vae) {
      throw std::invalid_argument(
          "MiniMax-H3 Ref2VA eager runner requires non-null native modules");
    }
    if (denoiser->loaded_fixed_tensor_count() !=
        MiniMaxH3StreamingDenoiserImpl::kFixedTensorCount) {
      throw std::logic_error(
          "MiniMax-H3 Ref2VA eager denoiser fixed weights are not loaded");
    }
    video_vae->verify_loaded_weights();
    audio_vae->verify_loaded_weights();
    MiniMaxH3SourceLayoutValidator::validate(transformer_shards);

    const torch::Device device = require_module_device(
        *denoiser.ptr(), "streaming denoiser", /*require_npu=*/true);
    const torch::Device video_vae_device =
        require_module_device(*video_vae.ptr(), "video VAE", true);
    const torch::Device audio_vae_device =
        require_module_device(*audio_vae.ptr(), "audio VAE", true);
    if (video_vae_device != device || audio_vae_device != device) {
      throw std::invalid_argument(
          "MiniMax-H3 Ref2VA denoiser and VAEs must reside on one NPU");
    }

    validate_condition(input.condition, device);
    validate_target(input.target, device);
    const int64_t output_frames = expected_video_frames(
        input.target.initial_normalized_video_latent.size(2));

    std::vector<H3ReferenceBlock> reference_blocks;
    std::vector<torch::Tensor> reference_video_rows;
    std::vector<torch::Tensor> reference_audio_rows;
    reference_blocks.reserve(input.references.size());
    reference_video_rows.reserve(input.references.size() + 1);
    reference_audio_rows.reserve(input.references.size() + 1);
    int64_t visual_reference_count = 0;
    int64_t audio_reference_count = 0;
    int64_t standalone_audio_reference_count = 0;
    int64_t image_reference_count = 0;
    int64_t video_reference_count = 0;
    if (input.references.empty() || input.references.size() > 12) {
      throw std::invalid_argument(
          "MiniMax-H3 Ref2VA requires between 1 and 12 references");
    }
    for (size_t index = 0; index < input.references.size(); ++index) {
      const MiniMaxH3Ref2VAEncodedReference& reference =
          input.references[index];
      H3ReferenceBlock block = validate_reference(reference, index, device);
      reference_blocks.emplace_back(block);
      if (block.kind == H3ReferenceBlockKind::IMAGE) {
        ++image_reference_count;
      } else if (block.kind == H3ReferenceBlockKind::VIDEO ||
                 block.kind == H3ReferenceBlockKind::VIDEO_AUDIO) {
        ++video_reference_count;
      }
      if (block.kind == H3ReferenceBlockKind::AUDIO) {
        ++standalone_audio_reference_count;
      }
      if (has_visual(block.kind)) {
        torch::Tensor anchor =
            kVisualAnchorTimestep * reference.normalized_visual_latent +
            (1.0F - kVisualAnchorTimestep) * reference.visual_anchor_noise;
        require_finite(anchor, reference_name(index, "visual anchor"));
        reference_video_rows.emplace_back(
            minimax_h3_patchify_video_latent(anchor));
        ++visual_reference_count;
      }
      if (has_audio(block.kind)) {
        reference_audio_rows.emplace_back(
            minimax_h3_pack_audio_latent(reference.normalized_audio_latent));
        ++audio_reference_count;
      }
    }
    if (visual_reference_count == 0) {
      throw std::invalid_argument(
          "MiniMax-H3 Ref2VA requires at least one visual reference");
    }
    if (image_reference_count > 9 || video_reference_count > 3 ||
        standalone_audio_reference_count > 3) {
      throw std::invalid_argument(
          "MiniMax-H3 Ref2VA reference modality count exceeds its limit");
    }

    const torch::Tensor& initial_video_latent =
        input.target.initial_normalized_video_latent;
    const torch::Tensor& initial_audio_latent =
        input.target.initial_normalized_audio_latent;
    const H3TargetLatents target = {
        .audio_t = initial_audio_latent.size(2),
        .audio_channels = initial_audio_latent.size(0),
        .latent_t = initial_video_latent.size(2),
        .latent_h = initial_video_latent.size(3),
        .latent_w = initial_video_latent.size(4),
    };
    H3PackedLayout layout =
        minimax_h3_build_ref2va_packed_layout(input.condition.hidden,
                                              input.condition.token_tags,
                                              target,
                                              reference_blocks,
                                              input.sequence_length);

    reference_video_rows.emplace_back(
        minimax_h3_patchify_video_latent(initial_video_latent));
    reference_audio_rows.emplace_back(
        minimax_h3_pack_audio_latent(initial_audio_latent));
    MiniMaxH3Ref2VACompactRows initial_rows = {
        .video_rows = torch::cat(reference_video_rows, 0).contiguous(),
        .audio_rows = torch::cat(reference_audio_rows, 0).contiguous(),
    };
    validate_compact_rows(layout, initial_rows, device, "initial");

    MiniMaxH3TrajectoryOutput trajectory =
        denoiser->run_base_trajectory(transformer_shards,
                                      layout,
                                      initial_rows.video_rows,
                                      initial_rows.audio_rows,
                                      /*layer_observer=*/nullptr,
                                      /*step_observer=*/nullptr);
    if (trajectory.transformer_forwards != kTransformerForwards ||
        trajectory.block_forwards !=
            kTransformerForwards * MiniMaxH3TransformerConfig::kNumLayers) {
      throw std::logic_error(
          "MiniMax-H3 Ref2VA eager trajectory violated forward counters");
    }
    MiniMaxH3Ref2VACompactRows final_rows = {
        .video_rows = std::move(trajectory.video_rows),
        .audio_rows = std::move(trajectory.audio_rows),
    };
    validate_compact_rows(layout, final_rows, device, "final");

    const torch::Tensor image_update = layout.update_mask.to(device);
    const torch::Tensor audio_update = layout.audio_update_mask.to(device);
    torch::Tensor target_video_rows =
        final_rows.video_rows.index({image_update}).contiguous();
    torch::Tensor target_audio_rows =
        final_rows.audio_rows.index({audio_update}).contiguous();
    torch::Tensor final_video_latent = minimax_h3_unpatchify_video_tokens(
        target_video_rows,
        {.channels = MiniMaxH3TransformerConfig::kVideoLatentDim,
         .temporal = target.latent_t,
         .height = target.latent_h,
         .width = target.latent_w});
    torch::Tensor final_audio_latent = minimax_h3_unpack_audio_tokens(
        target_audio_rows, target.audio_channels, target.audio_t);
    require_exact_tensor(final_video_latent,
                         {1,
                          MiniMaxH3TransformerConfig::kVideoLatentDim,
                          target.latent_t,
                          target.latent_h,
                          target.latent_w},
                         torch::kFloat32,
                         device,
                         "final normalized video latent");
    require_exact_tensor(
        final_audio_latent,
        {2, MiniMaxH3TransformerConfig::kAudioLatentDim, target.audio_t},
        torch::kFloat32,
        device,
        "final normalized audio latent");

    torch::Tensor video = video_vae->decode_normalized(final_video_latent);
    torch::Tensor audio = audio_vae->decode_normalized(final_audio_latent);
    const int64_t output_height =
        target.latent_h * MiniMaxH3VideoVAEConfig::kSpatialRatio;
    const int64_t output_width =
        target.latent_w * MiniMaxH3VideoVAEConfig::kSpatialRatio;
    const int64_t output_samples =
        target.audio_t * MiniMaxH3AudioVAEConfig::kHopLength;
    require_exact_tensor(video,
                         {1, 3, output_frames, output_height, output_width},
                         torch::kFloat32,
                         device,
                         "decoded video");
    require_exact_tensor(audio,
                         {1, 2, output_samples},
                         torch::kFloat32,
                         device,
                         "decoded audio");
    require_range(video, 0.0F, 1.0F, "decoded video");
    require_range(audio, -1.0F, 1.0F, "decoded audio");

    return {
        .condition_schema = input.condition.schema,
        .source_backend = input.condition.source_backend,
        .layout = std::move(layout),
        .initial_compact_rows = std::move(initial_rows),
        .final_compact_rows = std::move(final_rows),
        .final_normalized_video_latent = std::move(final_video_latent),
        .final_normalized_audio_latent = std::move(final_audio_latent),
        .video = std::move(video),
        .audio = std::move(audio),
        .geometry = {.video_frames = output_frames,
                     .video_height = output_height,
                     .video_width = output_width,
                     .video_fps = kVideoFps,
                     .audio_samples = output_samples,
                     .audio_sample_rate = MiniMaxH3AudioVAEConfig::kSampleRate},
        .counters =
            {
                .reference_blocks =
                    static_cast<int64_t>(input.references.size()),
                .visual_references = visual_reference_count,
                .audio_references = audio_reference_count,
                .transformer_forwards = trajectory.transformer_forwards,
                .block_forwards = trajectory.block_forwards,
                .video_vae_decodes = 1,
                .audio_vae_decodes = 1,
            },
    };
  }

 private:
  static constexpr int64_t kVideoFps = 24;
  static constexpr int64_t kTransformerForwards = 49;
  static constexpr float kVisualAnchorTimestep = 0.999F;

  static bool has_visual(H3ReferenceBlockKind kind) {
    return kind == H3ReferenceBlockKind::IMAGE ||
           kind == H3ReferenceBlockKind::VIDEO ||
           kind == H3ReferenceBlockKind::VIDEO_AUDIO;
  }

  static bool has_audio(H3ReferenceBlockKind kind) {
    return kind == H3ReferenceBlockKind::AUDIO ||
           kind == H3ReferenceBlockKind::VIDEO_AUDIO;
  }

  static std::string reference_name(size_t index, std::string_view field) {
    return "reference[" + std::to_string(index) + "]." + std::string(field);
  }

  static torch::Device require_module_device(const torch::nn::Module& module,
                                             std::string_view name,
                                             bool require_npu) {
    std::optional<torch::Device> resident_device;
    const auto inspect = [&](const torch::Tensor& tensor) {
      if (!tensor.defined()) {
        throw std::logic_error("MiniMax-H3 Ref2VA " + std::string(name) +
                               " contains an undefined resident tensor");
      }
      if (!resident_device.has_value()) {
        resident_device = tensor.device();
      } else if (tensor.device() != *resident_device) {
        throw std::invalid_argument("MiniMax-H3 Ref2VA " + std::string(name) +
                                    " spans more than one device");
      }
    };
    for (const auto& parameter : module.named_parameters(/*recurse=*/true)) {
      inspect(parameter.value());
    }
    for (const auto& buffer : module.named_buffers(/*recurse=*/true)) {
      inspect(buffer.value());
    }
    if (!resident_device.has_value()) {
      throw std::logic_error("MiniMax-H3 Ref2VA " + std::string(name) +
                             " has no resident tensors");
    }
    if (require_npu &&
        resident_device->type() != c10::DeviceType::PrivateUse1) {
      throw std::invalid_argument("MiniMax-H3 Ref2VA " + std::string(name) +
                                  " must reside on an NPU");
    }
    return *resident_device;
  }

  static void require_finite(const torch::Tensor& tensor,
                             const std::string& name) {
    if (!torch::isfinite(tensor).all().item<bool>()) {
      throw std::invalid_argument("MiniMax-H3 Ref2VA " + name +
                                  " contains NaN or Inf");
    }
  }

  static void require_exact_tensor(const torch::Tensor& tensor,
                                   const std::vector<int64_t>& shape,
                                   torch::ScalarType dtype,
                                   const torch::Device& device,
                                   std::string_view name) {
    if (!tensor.defined() || tensor.sizes().vec() != shape ||
        tensor.scalar_type() != dtype || tensor.device() != device ||
        !tensor.is_contiguous()) {
      throw std::invalid_argument(
          "MiniMax-H3 Ref2VA " + std::string(name) +
          " has an invalid shape, dtype, device, or memory layout");
    }
    if (tensor.is_floating_point()) {
      require_finite(tensor, std::string(name));
    }
  }

  static void require_range(const torch::Tensor& tensor,
                            float minimum,
                            float maximum,
                            std::string_view name) {
    if (tensor.min().item<float>() < minimum ||
        tensor.max().item<float>() > maximum) {
      throw std::runtime_error("MiniMax-H3 Ref2VA " + std::string(name) +
                               " is outside its output range");
    }
  }

  static void validate_condition(const MiniMaxH3Ref2VACondition& condition,
                                 const torch::Device& device) {
    if (condition.schema != kMiniMaxH3ConditionSchemaV1) {
      throw std::invalid_argument(
          "MiniMax-H3 Ref2VA condition schema must be " +
          std::string(kMiniMaxH3ConditionSchemaV1));
    }
    (void)minimax_h3_ref2va_source_backend_name(condition.source_backend);
    if (!condition.hidden.defined() || condition.hidden.dim() != 3 ||
        condition.hidden.size(0) != 1 || condition.hidden.size(1) <= 0 ||
        condition.hidden.size(2) != MiniMaxH3TransformerConfig::kTextDim) {
      throw std::invalid_argument(
          "MiniMax-H3 Ref2VA condition hidden must have shape [1,N,5120]");
    }
    require_exact_tensor(condition.hidden,
                         condition.hidden.sizes().vec(),
                         torch::kBFloat16,
                         device,
                         "condition hidden");
    require_exact_tensor(condition.token_tags,
                         {1, condition.hidden.size(1)},
                         torch::kInt64,
                         device,
                         "condition token tags");
    if (!torch::logical_or(condition.token_tags == 0, condition.token_tags == 1)
             .all()
             .item<bool>()) {
      throw std::invalid_argument(
          "MiniMax-H3 Ref2VA condition token tags must contain only 0 or 1");
    }
  }

  static void validate_target(const MiniMaxH3Ref2VAPreparedTarget& target,
                              const torch::Device& device) {
    if (!target.initial_normalized_video_latent.defined() ||
        target.initial_normalized_video_latent.dim() != 5 ||
        target.initial_normalized_video_latent.size(0) != 1 ||
        target.initial_normalized_video_latent.size(1) !=
            MiniMaxH3TransformerConfig::kVideoLatentDim ||
        target.initial_normalized_video_latent.size(2) <= 0 ||
        target.initial_normalized_video_latent.size(3) <= 0 ||
        target.initial_normalized_video_latent.size(4) <= 0) {
      throw std::invalid_argument(
          "MiniMax-H3 Ref2VA target video must have shape [1,24,T,H,W]");
    }
    require_exact_tensor(target.initial_normalized_video_latent,
                         target.initial_normalized_video_latent.sizes().vec(),
                         torch::kFloat32,
                         device,
                         "target initial normalized video latent");
    if (target.initial_normalized_video_latent.size(3) % 2 != 0 ||
        target.initial_normalized_video_latent.size(4) % 2 != 0) {
      throw std::invalid_argument(
          "MiniMax-H3 Ref2VA target video latent H and W must be even");
    }
    if (!target.initial_normalized_audio_latent.defined() ||
        target.initial_normalized_audio_latent.dim() != 3 ||
        target.initial_normalized_audio_latent.size(0) != 2 ||
        target.initial_normalized_audio_latent.size(1) !=
            MiniMaxH3TransformerConfig::kAudioLatentDim ||
        target.initial_normalized_audio_latent.size(2) <= 0) {
      throw std::invalid_argument(
          "MiniMax-H3 Ref2VA target audio must have shape [2,32,T]");
    }
    require_exact_tensor(target.initial_normalized_audio_latent,
                         target.initial_normalized_audio_latent.sizes().vec(),
                         torch::kFloat32,
                         device,
                         "target initial normalized audio latent");
  }

  static H3ReferenceBlock validate_reference(
      const MiniMaxH3Ref2VAEncodedReference& reference,
      size_t index,
      const torch::Device& device) {
    const bool visual = has_visual(reference.kind);
    const bool audio = has_audio(reference.kind);
    if (!visual && !audio) {
      throw std::invalid_argument("MiniMax-H3 Ref2VA " +
                                  reference_name(index, "kind") +
                                  " is unsupported");
    }
    if (visual) {
      if (!reference.normalized_visual_latent.defined() ||
          reference.normalized_visual_latent.dim() != 5 ||
          reference.normalized_visual_latent.size(0) != 1 ||
          reference.normalized_visual_latent.size(1) !=
              MiniMaxH3TransformerConfig::kVideoLatentDim ||
          reference.normalized_visual_latent.size(2) <= 0 ||
          reference.normalized_visual_latent.size(3) <= 0 ||
          reference.normalized_visual_latent.size(4) <= 0) {
        throw std::invalid_argument("MiniMax-H3 Ref2VA " +
                                    reference_name(index, "visual latent") +
                                    " must have shape [1,24,T,H,W]");
      }
      require_exact_tensor(reference.normalized_visual_latent,
                           reference.normalized_visual_latent.sizes().vec(),
                           torch::kFloat32,
                           device,
                           reference_name(index, "normalized visual latent"));
      require_exact_tensor(reference.visual_anchor_noise,
                           reference.normalized_visual_latent.sizes().vec(),
                           torch::kFloat32,
                           device,
                           reference_name(index, "visual anchor noise"));
      if (reference.normalized_visual_latent.size(3) % 2 != 0 ||
          reference.normalized_visual_latent.size(4) % 2 != 0) {
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA reference visual latent H and W must be even");
      }
      if (reference.kind == H3ReferenceBlockKind::IMAGE &&
          reference.normalized_visual_latent.size(2) != 1) {
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA image reference must have one latent frame");
      }
    } else if (reference.normalized_visual_latent.defined() ||
               reference.visual_anchor_noise.defined()) {
      throw std::invalid_argument(
          "MiniMax-H3 Ref2VA audio reference must not carry visual tensors");
    }
    if (audio) {
      if (!reference.normalized_audio_latent.defined() ||
          reference.normalized_audio_latent.dim() != 3 ||
          reference.normalized_audio_latent.size(0) != 2 ||
          reference.normalized_audio_latent.size(1) !=
              MiniMaxH3TransformerConfig::kAudioLatentDim ||
          reference.normalized_audio_latent.size(2) <= 0) {
        throw std::invalid_argument("MiniMax-H3 Ref2VA " +
                                    reference_name(index, "audio latent") +
                                    " must have shape [2,32,T]");
      }
      require_exact_tensor(reference.normalized_audio_latent,
                           reference.normalized_audio_latent.sizes().vec(),
                           torch::kFloat32,
                           device,
                           reference_name(index, "normalized audio latent"));
    } else if (reference.normalized_audio_latent.defined()) {
      throw std::invalid_argument(
          "MiniMax-H3 Ref2VA visual-only reference must not carry audio");
    }

    H3ReferenceBlock block;
    block.kind = reference.kind;
    if (audio) {
      block.ref_audio_t = reference.normalized_audio_latent.size(2);
    }
    if (visual) {
      block.latent_t = reference.normalized_visual_latent.size(2);
      block.latent_h = reference.normalized_visual_latent.size(3);
      block.latent_w = reference.normalized_visual_latent.size(4);
    }
    return block;
  }

  static int64_t expected_video_frames(int64_t latent_frames) {
    constexpr int64_t kTokenDrop = MiniMaxH3VideoVAEConfig::kTokenDrop;
    constexpr int64_t kTokensPerChunk =
        MiniMaxH3VideoVAEConfig::kTokensPerChunk;
    if (latent_frames < 7 ||
        (latent_frames + kTokenDrop) % kTokensPerChunk != 0) {
      throw std::invalid_argument(
          "MiniMax-H3 Ref2VA target video latent T must decode exactly to "
          "17*n+5 frames");
    }
    const int64_t chunks = (latent_frames + kTokenDrop) / kTokensPerChunk - 1;
    return chunks * MiniMaxH3VideoVAEConfig::kClipLength +
           MiniMaxH3VideoVAEConfig::kFrameOverlap;
  }

  static void validate_compact_rows(const H3PackedLayout& layout,
                                    const MiniMaxH3Ref2VACompactRows& rows,
                                    const torch::Device& device,
                                    std::string_view stage) {
    require_exact_tensor(rows.video_rows,
                         {layout.img_pos.numel(),
                          MiniMaxH3TransformerConfig::kVideoLatentDim * 4},
                         torch::kFloat32,
                         device,
                         std::string(stage) + " compact video rows");
    require_exact_tensor(
        rows.audio_rows,
        {layout.audio_pos.numel(), MiniMaxH3TransformerConfig::kAudioLatentDim},
        torch::kFloat32,
        device,
        std::string(stage) + " compact audio rows");
  }
};

inline MiniMaxH3Ref2VAEagerOutput minimax_h3_run_ref2va_eager(
    const MiniMaxH3Ref2VAEagerInput& input,
    const std::vector<std::unique_ptr<StateDict>>& transformer_shards,
    MiniMaxH3StreamingDenoiser& denoiser,
    MiniMaxH3VideoVAE& video_vae,
    MiniMaxH3AudioVAE& audio_vae) {
  return MiniMaxH3Ref2VAEagerRunner::run(
      input, transformer_shards, denoiser, video_vae, audio_vae);
}

}  // namespace xllm
