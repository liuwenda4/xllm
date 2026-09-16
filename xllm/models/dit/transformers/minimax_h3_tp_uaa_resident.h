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

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "models/dit/transformers/minimax_h3_tp_uaa_block.h"

namespace xllm {

inline constexpr int64_t kMiniMaxH3ResidentBlockCount = 50;

using MiniMaxH3TPUAAResidentObserver =
    std::function<void(int64_t, const MiniMaxH3ResidualBranchTrace&)>;

class MiniMaxH3TPUAAResidentTransformerImpl final : public torch::nn::Module {
 public:
  MiniMaxH3TPUAAResidentTransformerImpl(const MiniMaxH3C4Config& config,
                                        ProcessGroup* tp_group,
                                        ProcessGroup* u_group,
                                        const torch::TensorOptions& options)
      : config_(config) {
    blocks_ = register_module("blocks", torch::nn::ModuleList());
    block_layers_.reserve(kMiniMaxH3ResidentBlockCount);
    for (int64_t layer = 0; layer < kMiniMaxH3ResidentBlockCount; ++layer) {
      MiniMaxH3TPUAADiTBlock block(config, tp_group, u_group, options);
      blocks_->push_back(block);
      block_layers_.emplace_back(std::move(block));
    }
  }

  void load_source_weights(
      const std::vector<std::unique_ptr<StateDict>>& shards) {
    if (loaded_block_count_ != 0) {
      throw std::logic_error(
          "MiniMax-H3 resident transformer weights are already loaded");
    }
    validate_source_weights(shards);
    for (int64_t layer = 0; layer < kMiniMaxH3ResidentBlockCount; ++layer) {
      block_layers_[static_cast<size_t>(layer)]->load_source_weights(shards,
                                                                     layer);
      ++loaded_block_count_;
    }
  }

  torch::Tensor forward(
      const torch::Tensor& input,
      const torch::Tensor& time_embedding,
      const torch::Tensor& combined_indices,
      const torch::Tensor& rope_frequencies,
      const torch::Tensor& global_cu_seqlens,
      const MiniMaxH3TPUAAResidentObserver& observer = nullptr) {
    if (loaded_block_count_ != kMiniMaxH3ResidentBlockCount) {
      throw std::logic_error(
          "MiniMax-H3 resident transformer does not have all 50 blocks");
    }
    torch::Tensor hidden = input;
    for (int64_t layer = 0; layer < kMiniMaxH3ResidentBlockCount; ++layer) {
      if (observer) {
        MiniMaxH3ResidualBranchTrace trace =
            block_layers_[static_cast<size_t>(layer)]->forward(
                hidden,
                time_embedding,
                combined_indices,
                rope_frequencies,
                global_cu_seqlens);
        hidden = trace.output;
        observer(layer, trace);
      } else {
        hidden = block_layers_[static_cast<size_t>(layer)]->forward_output_only(
            hidden,
            time_embedding,
            combined_indices,
            rope_frequencies,
            global_cu_seqlens);
      }
    }
    return hidden;
  }

  int64_t loaded_block_count() const { return loaded_block_count_; }

  MiniMaxH3TPUAADiTBlock block_at(int64_t layer) const {
    if (layer < 0 || layer >= kMiniMaxH3ResidentBlockCount) {
      throw std::out_of_range("MiniMax-H3 resident block index out of range");
    }
    return block_layers_[static_cast<size_t>(layer)];
  }

  int64_t resident_module_bytes() const {
    int64_t bytes = 0;
    for (const auto& item : named_parameters(/*recurse=*/true)) {
      bytes += item.value().numel() *
               static_cast<int64_t>(item.value().element_size());
    }
    for (const auto& item : named_buffers(/*recurse=*/true)) {
      bytes += item.value().numel() *
               static_cast<int64_t>(item.value().element_size());
    }
    return bytes;
  }

  bool all_tensors_on(const torch::Device& device) const {
    for (const auto& item : named_parameters(/*recurse=*/true)) {
      if (item.value().device() != device) {
        return false;
      }
    }
    for (const auto& item : named_buffers(/*recurse=*/true)) {
      if (item.value().device() != device) {
        return false;
      }
    }
    return true;
  }

 private:
  void validate_source_weights(
      const std::vector<std::unique_ptr<StateDict>>& shards) const {
    const int64_t inner =
        config_.num_attention_heads * config_.attention_head_dim;
    const std::array<std::pair<std::string, std::vector<int64_t>>, 10> specs = {
        std::pair{"norm1.weight", std::vector<int64_t>{config_.hidden_size}},
        std::pair{"norm2.weight", std::vector<int64_t>{config_.hidden_size}},
        std::pair{"attn.q_norm.weight",
                  std::vector<int64_t>{config_.attention_head_dim}},
        std::pair{"attn.k_norm.weight",
                  std::vector<int64_t>{config_.attention_head_dim}},
        std::pair{"attn.qkv_proj.weight",
                  std::vector<int64_t>{3 * inner, config_.hidden_size}},
        std::pair{"attn.out_proj.weight",
                  std::vector<int64_t>{config_.hidden_size, inner}},
        std::pair{"mlp.fc1.weight",
                  std::vector<int64_t>{2 * config_.ffn_hidden_size,
                                       config_.hidden_size}},
        std::pair{
            "mlp.fc2.weight",
            std::vector<int64_t>{config_.hidden_size, config_.ffn_hidden_size}},
        std::pair{"adaln_proj.linear.weight",
                  std::vector<int64_t>{18 * config_.hidden_size,
                                       config_.time_embed_dim}},
        std::pair{"adaln_proj.linear.bias",
                  std::vector<int64_t>{18 * config_.hidden_size}},
    };
    for (int64_t layer = 0; layer < kMiniMaxH3ResidentBlockCount; ++layer) {
      const std::string prefix = "blocks." + std::to_string(layer) + ".";
      for (const auto& [suffix, shape] : specs) {
        const std::string name = prefix + suffix;
        torch::Tensor found;
        for (const std::unique_ptr<StateDict>& shard : shards) {
          if (shard == nullptr) {
            throw std::invalid_argument(
                "MiniMax-H3 resident source contains a null shard");
          }
          const torch::Tensor candidate = shard->get_tensor(name);
          if (!candidate.defined()) {
            continue;
          }
          if (found.defined()) {
            throw std::invalid_argument(
                "MiniMax-H3 resident source tensor is duplicated: `" + name +
                "`");
          }
          found = candidate;
        }
        if (!found.defined() || found.sizes().vec() != shape ||
            found.scalar_type() != torch::kBFloat16) {
          throw std::invalid_argument(
              "MiniMax-H3 resident source tensor metadata mismatch: `" + name +
              "`");
        }
      }
    }
  }

  MiniMaxH3C4Config config_;
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<MiniMaxH3TPUAADiTBlock> block_layers_;
  int64_t loaded_block_count_ = 0;
};
TORCH_MODULE(MiniMaxH3TPUAAResidentTransformer);

}  // namespace xllm
