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
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/process_group.h"

namespace xllm {

inline constexpr int64_t kMiniMaxH3UaaSize = 8;
inline constexpr int64_t kMiniMaxH3UaaLogicalHeads = 28;
inline constexpr int64_t kMiniMaxH3UaaPaddedHeads = 32;
inline constexpr int64_t kMiniMaxH3UaaLocalHeads = 4;
inline constexpr int64_t kMiniMaxH3UaaHeadDim = 128;

inline torch::Tensor minimax_h3_uaa_pad_heads(const torch::Tensor& input) {
  if (!input.defined() || input.dim() != 4 || input.size(0) <= 0 ||
      input.size(1) <= 0 || input.size(2) != kMiniMaxH3UaaLogicalHeads ||
      input.size(3) <= 0) {
    throw std::invalid_argument(
        "MiniMax-H3 UAA input must be [B,S,28,D] with B,S,D > 0");
  }
  torch::Tensor padded = torch::zeros(
      {input.size(0), input.size(1), kMiniMaxH3UaaPaddedHeads, input.size(3)},
      input.options());
  padded.slice(2, 0, kMiniMaxH3UaaLogicalHeads).copy_(input);
  return padded;
}

inline torch::Tensor minimax_h3_uaa_prepare_forward_send(
    const torch::Tensor& input) {
  torch::Tensor padded = minimax_h3_uaa_pad_heads(input);
  return padded
      .reshape({input.size(0),
                input.size(1),
                kMiniMaxH3UaaSize,
                kMiniMaxH3UaaLocalHeads,
                input.size(3)})
      .permute({2, 1, 0, 3, 4})
      .contiguous()
      .flatten(0, 1);
}

inline void minimax_h3_uaa_prepare_forward_send_out(const torch::Tensor& input,
                                                    torch::Tensor& send) {
  if (!input.defined() || input.dim() != 4 || input.size(0) != 1 ||
      input.size(1) <= 0 || input.size(2) != kMiniMaxH3UaaLogicalHeads ||
      input.size(3) != kMiniMaxH3UaaHeadDim || !send.defined() ||
      send.sizes() != torch::IntArrayRef({kMiniMaxH3UaaSize * input.size(1),
                                          input.size(0),
                                          kMiniMaxH3UaaLocalHeads,
                                          input.size(3)}) ||
      send.options().dtype() != input.options().dtype() ||
      send.device() != input.device()) {
    throw std::invalid_argument(
        "MiniMax-H3 UAA forward workspace metadata mismatch");
  }
  send.zero_();
  for (int64_t destination = 0; destination < kMiniMaxH3UaaSize - 1;
       ++destination) {
    send.narrow(0, destination * input.size(1), input.size(1))
        .copy_(input
                   .slice(2,
                          destination * kMiniMaxH3UaaLocalHeads,
                          (destination + 1) * kMiniMaxH3UaaLocalHeads)
                   .permute({1, 0, 2, 3}));
  }
}

inline torch::Tensor minimax_h3_uaa_finish_forward_receive(
    const torch::Tensor& receive,
    int64_t batch,
    int64_t local_sequence,
    int64_t head_dim) {
  const int64_t global_sequence = local_sequence * kMiniMaxH3UaaSize;
  if (!receive.defined() || receive.dim() != 4 || batch <= 0 ||
      local_sequence <= 0 || head_dim <= 0 ||
      receive.sizes() !=
          torch::IntArrayRef(
              {global_sequence, batch, kMiniMaxH3UaaLocalHeads, head_dim})) {
    throw std::invalid_argument(
        "MiniMax-H3 UAA forward receive buffer metadata mismatch");
  }
  return receive.permute({1, 0, 2, 3}).contiguous();
}

inline torch::Tensor minimax_h3_uaa_prepare_inverse_send(
    const torch::Tensor& input) {
  if (!input.defined() || input.dim() != 4 || input.size(0) <= 0 ||
      input.size(1) <= 0 || input.size(1) % kMiniMaxH3UaaSize != 0 ||
      input.size(2) != kMiniMaxH3UaaLocalHeads || input.size(3) <= 0) {
    throw std::invalid_argument(
        "MiniMax-H3 UAA inverse input must be [B,S_global,4,D]");
  }
  return input.permute({1, 0, 2, 3}).contiguous();
}

inline void minimax_h3_uaa_prepare_inverse_send_out(const torch::Tensor& input,
                                                    torch::Tensor& send) {
  if (!input.defined() || input.dim() != 4 || input.size(0) != 1 ||
      input.size(1) <= 0 || input.size(1) % kMiniMaxH3UaaSize != 0 ||
      input.size(2) != kMiniMaxH3UaaLocalHeads ||
      input.size(3) != kMiniMaxH3UaaHeadDim || !send.defined() ||
      send.sizes() !=
          torch::IntArrayRef(
              {input.size(1), input.size(0), input.size(2), input.size(3)}) ||
      send.scalar_type() != input.scalar_type() ||
      send.device() != input.device()) {
    throw std::invalid_argument(
        "MiniMax-H3 UAA inverse workspace metadata mismatch");
  }
  send.copy_(input.permute({1, 0, 2, 3}));
}

inline torch::Tensor minimax_h3_uaa_finish_inverse_receive(
    const torch::Tensor& receive,
    int64_t batch,
    int64_t local_sequence,
    int64_t head_dim) {
  if (!receive.defined() || receive.dim() != 4 || batch <= 0 ||
      local_sequence <= 0 || head_dim <= 0 ||
      receive.sizes() != torch::IntArrayRef({kMiniMaxH3UaaSize * local_sequence,
                                             batch,
                                             kMiniMaxH3UaaLocalHeads,
                                             head_dim})) {
    throw std::invalid_argument(
        "MiniMax-H3 UAA inverse receive buffer metadata mismatch");
  }
  torch::Tensor padded =
      receive
          .reshape({kMiniMaxH3UaaSize,
                    local_sequence,
                    batch,
                    kMiniMaxH3UaaLocalHeads,
                    head_dim})
          .permute({2, 1, 0, 3, 4})
          .contiguous()
          .reshape({batch, local_sequence, kMiniMaxH3UaaPaddedHeads, head_dim});
  return padded.slice(2, 0, kMiniMaxH3UaaLogicalHeads).contiguous();
}

inline int64_t minimax_h3_uaa_real_heads_for_rank(int64_t u_rank) {
  if (u_rank < 0 || u_rank >= kMiniMaxH3UaaSize) {
    throw std::invalid_argument("MiniMax-H3 UAA rank must be in [0,8)");
  }
  const int64_t first_head = u_rank * kMiniMaxH3UaaLocalHeads;
  return std::max<int64_t>(
      0,
      std::min<int64_t>(kMiniMaxH3UaaLocalHeads,
                        kMiniMaxH3UaaLogicalHeads - first_head));
}

inline bool minimax_h3_uaa_dummy_heads_are_zero(const torch::Tensor& value,
                                                int64_t real_heads) {
  if (!value.defined() || value.dim() != 4 || real_heads < 0 ||
      real_heads > value.size(2)) {
    throw std::invalid_argument("MiniMax-H3 UAA dummy check expects [B,S,H,D]");
  }
  torch::Tensor dummy = value.slice(2, real_heads, value.size(2));
  return dummy.numel() == 0 || torch::count_nonzero(dummy).item<int64_t>() == 0;
}

class MiniMaxH3UAAForwardContext final {
 public:
  MiniMaxH3UAAForwardContext(torch::Tensor send,
                             torch::Tensor receive,
                             c10::intrusive_ptr<c10d::Work> work,
                             int64_t batch,
                             int64_t local_sequence,
                             int64_t head_dim)
      : send_(std::move(send)),
        receive_(std::move(receive)),
        work_(std::move(work)),
        batch_(batch),
        local_sequence_(local_sequence),
        head_dim_(head_dim) {}

  MiniMaxH3UAAForwardContext(const MiniMaxH3UAAForwardContext&) = delete;
  MiniMaxH3UAAForwardContext& operator=(const MiniMaxH3UAAForwardContext&) =
      delete;
  MiniMaxH3UAAForwardContext(MiniMaxH3UAAForwardContext&&) = default;
  MiniMaxH3UAAForwardContext& operator=(MiniMaxH3UAAForwardContext&&) = delete;

  ~MiniMaxH3UAAForwardContext() {
    if (work_ != nullptr) {
      try {
        work_->wait();
      } catch (...) {
      }
    }
  }

  torch::Tensor finish() {
    if (finished_ || work_ == nullptr) {
      throw std::logic_error(
          "MiniMax-H3 UAA forward context is already finished");
    }
    if (!work_->wait()) {
      throw std::runtime_error("MiniMax-H3 UAA forward HCCL wait failed");
    }
    torch::Tensor output = minimax_h3_uaa_finish_forward_receive(
        receive_, batch_, local_sequence_, head_dim_);
    finished_ = true;
    work_.reset();
    send_ = torch::Tensor();
    receive_ = torch::Tensor();
    return output;
  }

  const torch::Tensor& send_buffer() const { return send_; }
  const torch::Tensor& receive_buffer() const { return receive_; }

 private:
  torch::Tensor send_;
  torch::Tensor receive_;
  c10::intrusive_ptr<c10d::Work> work_;
  int64_t batch_;
  int64_t local_sequence_;
  int64_t head_dim_;
  bool finished_ = false;
};

class MiniMaxH3UAAInverseContext final {
 public:
  MiniMaxH3UAAInverseContext(torch::Tensor send,
                             torch::Tensor receive,
                             c10::intrusive_ptr<c10d::Work> work,
                             int64_t batch,
                             int64_t local_sequence,
                             int64_t head_dim)
      : send_(std::move(send)),
        receive_(std::move(receive)),
        work_(std::move(work)),
        batch_(batch),
        local_sequence_(local_sequence),
        head_dim_(head_dim) {}

  MiniMaxH3UAAInverseContext(const MiniMaxH3UAAInverseContext&) = delete;
  MiniMaxH3UAAInverseContext& operator=(const MiniMaxH3UAAInverseContext&) =
      delete;
  MiniMaxH3UAAInverseContext(MiniMaxH3UAAInverseContext&&) = default;
  MiniMaxH3UAAInverseContext& operator=(MiniMaxH3UAAInverseContext&&) = delete;

  ~MiniMaxH3UAAInverseContext() {
    if (work_ != nullptr) {
      try {
        work_->wait();
      } catch (...) {
      }
    }
  }

  torch::Tensor finish() {
    if (finished_ || work_ == nullptr) {
      throw std::logic_error(
          "MiniMax-H3 UAA inverse context is already finished");
    }
    if (!work_->wait()) {
      throw std::runtime_error("MiniMax-H3 UAA inverse HCCL wait failed");
    }
    torch::Tensor output = minimax_h3_uaa_finish_inverse_receive(
        receive_, batch_, local_sequence_, head_dim_);
    finished_ = true;
    work_.reset();
    send_ = torch::Tensor();
    receive_ = torch::Tensor();
    return output;
  }

  const torch::Tensor& send_buffer() const { return send_; }
  const torch::Tensor& receive_buffer() const { return receive_; }

 private:
  torch::Tensor send_;
  torch::Tensor receive_;
  c10::intrusive_ptr<c10d::Work> work_;
  int64_t batch_;
  int64_t local_sequence_;
  int64_t head_dim_;
  bool finished_ = false;
};

class MiniMaxH3UAAWorkspace final {
 public:
  static constexpr int64_t kForwardSlots = 3;
  static constexpr int64_t kReceiveSlots = 4;
  static constexpr int64_t kInverseReceiveSlot = 3;

  void reserve(int64_t local_sequence, const torch::TensorOptions& options) {
    if (local_sequence <= 0 || options.dtype_opt() != torch::kBFloat16) {
      throw std::invalid_argument(
          "MiniMax-H3 UAA workspace requires positive BF16 capacity");
    }
    if (send_storage_.defined()) {
      if (send_storage_.device() != options.device() ||
          send_storage_.scalar_type() != options.dtype().toScalarType()) {
        throw std::invalid_argument(
            "MiniMax-H3 UAA workspace device or dtype changed");
      }
      if (capacity_ >= local_sequence) {
        return;
      }
    }
    capacity_ = local_sequence;
    const std::vector<int64_t> buffer_shape = {kMiniMaxH3UaaSize * capacity_,
                                               1,
                                               kMiniMaxH3UaaLocalHeads,
                                               kMiniMaxH3UaaHeadDim};
    std::vector<int64_t> send_shape = {kForwardSlots};
    send_shape.insert(
        send_shape.end(), buffer_shape.begin(), buffer_shape.end());
    std::vector<int64_t> receive_shape = {kReceiveSlots};
    receive_shape.insert(
        receive_shape.end(), buffer_shape.begin(), buffer_shape.end());
    send_storage_ = torch::empty(send_shape, options);
    receive_storage_ = torch::empty(receive_shape, options);
  }

  torch::Tensor send(int64_t slot, int64_t local_sequence) const {
    return active_buffer(send_storage_, kForwardSlots, slot, local_sequence);
  }

  torch::Tensor receive(int64_t slot, int64_t local_sequence) const {
    return active_buffer(receive_storage_, kReceiveSlots, slot, local_sequence);
  }

  int64_t capacity() const { return capacity_; }
  int64_t allocated_bytes() const {
    if (!send_storage_.defined()) {
      return 0;
    }
    return (send_storage_.numel() + receive_storage_.numel()) *
           static_cast<int64_t>(send_storage_.element_size());
  }

 private:
  torch::Tensor active_buffer(const torch::Tensor& storage,
                              int64_t slots,
                              int64_t slot,
                              int64_t local_sequence) const {
    if (!storage.defined() || slot < 0 || slot >= slots ||
        local_sequence <= 0 || local_sequence > capacity_) {
      throw std::invalid_argument(
          "MiniMax-H3 UAA workspace request is out of range");
    }
    return storage.select(0, slot).narrow(
        0, 0, kMiniMaxH3UaaSize * local_sequence);
  }

  torch::Tensor send_storage_;
  torch::Tensor receive_storage_;
  int64_t capacity_ = 0;
};

inline void minimax_h3_validate_uaa_runtime_input(const torch::Tensor& input,
                                                  ProcessGroup* u_group) {
  if (u_group == nullptr || u_group->world_size() != kMiniMaxH3UaaSize ||
      u_group->rank() < 0 || u_group->rank() >= kMiniMaxH3UaaSize) {
    throw std::invalid_argument(
        "MiniMax-H3 advanced UAA requires an exact U8 process group");
  }
  if (!input.defined() || input.dim() != 4 ||
      input.device() != u_group->device() ||
      input.scalar_type() != torch::kBFloat16 || input.size(0) != 1 ||
      input.size(3) != kMiniMaxH3UaaHeadDim) {
    throw std::invalid_argument(
        "MiniMax-H3 advanced UAA requires NPU BF16 [1,S,H,128]");
  }
}

inline MiniMaxH3UAAForwardContext minimax_h3_uaa_launch_forward(
    const torch::Tensor& input,
    ProcessGroup* u_group) {
  minimax_h3_validate_uaa_runtime_input(input, u_group);
  if (input.size(2) != kMiniMaxH3UaaLogicalHeads || input.size(1) <= 0) {
    throw std::invalid_argument(
        "MiniMax-H3 UAA forward requires [1,S_local,28,128]");
  }
  torch::Tensor send = minimax_h3_uaa_prepare_forward_send(input);
  torch::Tensor receive = torch::empty({kMiniMaxH3UaaSize * input.size(1),
                                        input.size(0),
                                        kMiniMaxH3UaaLocalHeads,
                                        input.size(3)},
                                       input.options());
  std::vector<int64_t> splits(kMiniMaxH3UaaSize, input.size(1));
  c10::intrusive_ptr<c10d::Work> work;
  u_group->all_to_all_single(receive,
                             send,
                             splits,
                             splits,
                             /*async_op=*/true,
                             &work);
  if (work == nullptr) {
    throw std::runtime_error(
        "MiniMax-H3 UAA forward did not receive an HCCL Work handle");
  }
  return MiniMaxH3UAAForwardContext(std::move(send),
                                    std::move(receive),
                                    std::move(work),
                                    input.size(0),
                                    input.size(1),
                                    input.size(3));
}

inline MiniMaxH3UAAForwardContext minimax_h3_uaa_launch_forward_into(
    const torch::Tensor& input,
    ProcessGroup* u_group,
    torch::Tensor send,
    torch::Tensor receive) {
  minimax_h3_validate_uaa_runtime_input(input, u_group);
  if (input.size(2) != kMiniMaxH3UaaLogicalHeads || input.size(1) <= 0) {
    throw std::invalid_argument(
        "MiniMax-H3 UAA forward requires [1,S_local,28,128]");
  }
  minimax_h3_uaa_prepare_forward_send_out(input, send);
  const std::vector<int64_t> splits(kMiniMaxH3UaaSize, input.size(1));
  c10::intrusive_ptr<c10d::Work> work;
  u_group->all_to_all_single(receive,
                             send,
                             splits,
                             splits,
                             /*async_op=*/true,
                             &work);
  if (work == nullptr) {
    throw std::runtime_error(
        "MiniMax-H3 UAA forward did not receive an HCCL Work handle");
  }
  return MiniMaxH3UAAForwardContext(std::move(send),
                                    std::move(receive),
                                    std::move(work),
                                    input.size(0),
                                    input.size(1),
                                    input.size(3));
}

inline MiniMaxH3UAAInverseContext minimax_h3_uaa_launch_inverse(
    const torch::Tensor& input,
    ProcessGroup* u_group) {
  minimax_h3_validate_uaa_runtime_input(input, u_group);
  if (input.size(2) != kMiniMaxH3UaaLocalHeads || input.size(1) <= 0 ||
      input.size(1) % kMiniMaxH3UaaSize != 0) {
    throw std::invalid_argument(
        "MiniMax-H3 UAA inverse requires [1,S_global,4,128]");
  }
  const int64_t local_sequence = input.size(1) / kMiniMaxH3UaaSize;
  torch::Tensor send = minimax_h3_uaa_prepare_inverse_send(input);
  torch::Tensor receive = torch::empty({kMiniMaxH3UaaSize * local_sequence,
                                        input.size(0),
                                        kMiniMaxH3UaaLocalHeads,
                                        input.size(3)},
                                       input.options());
  std::vector<int64_t> splits(kMiniMaxH3UaaSize, local_sequence);
  c10::intrusive_ptr<c10d::Work> work;
  u_group->all_to_all_single(receive,
                             send,
                             splits,
                             splits,
                             /*async_op=*/true,
                             &work);
  if (work == nullptr) {
    throw std::runtime_error(
        "MiniMax-H3 UAA inverse did not receive an HCCL Work handle");
  }
  return MiniMaxH3UAAInverseContext(std::move(send),
                                    std::move(receive),
                                    std::move(work),
                                    input.size(0),
                                    local_sequence,
                                    input.size(3));
}

inline MiniMaxH3UAAInverseContext minimax_h3_uaa_launch_inverse_into(
    const torch::Tensor& input,
    ProcessGroup* u_group,
    torch::Tensor send,
    torch::Tensor receive) {
  minimax_h3_validate_uaa_runtime_input(input, u_group);
  if (input.size(2) != kMiniMaxH3UaaLocalHeads || input.size(1) <= 0 ||
      input.size(1) % kMiniMaxH3UaaSize != 0) {
    throw std::invalid_argument(
        "MiniMax-H3 UAA inverse requires [1,S_global,4,128]");
  }
  const int64_t local_sequence = input.size(1) / kMiniMaxH3UaaSize;
  minimax_h3_uaa_prepare_inverse_send_out(input, send);
  const std::vector<int64_t> splits(kMiniMaxH3UaaSize, local_sequence);
  c10::intrusive_ptr<c10d::Work> work;
  u_group->all_to_all_single(receive,
                             send,
                             splits,
                             splits,
                             /*async_op=*/true,
                             &work);
  if (work == nullptr) {
    throw std::runtime_error(
        "MiniMax-H3 UAA inverse did not receive an HCCL Work handle");
  }
  return MiniMaxH3UAAInverseContext(std::move(send),
                                    std::move(receive),
                                    std::move(work),
                                    input.size(0),
                                    local_sequence,
                                    input.size(3));
}

}  // namespace xllm
