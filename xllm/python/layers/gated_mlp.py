# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tensor-parallel gated MLP shared by Python model implementations."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
)


class GatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int,
        dtype: torch.dtype,
        device: torch.device,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        if intermediate_size % tp_size:
            raise ValueError("intermediate_size must be divisible by tp_size")
        local_intermediate_size = intermediate_size // tp_size
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * local_intermediate_size,
            tp_size,
            dtype=dtype,
            device=device,
        )
        self.down_proj = RowParallelLinear(
            local_intermediate_size,
            hidden_size,
            tp_size,
            dtype=dtype,
            device=device,
            reduce_results=reduce_results,
        )
        self.split_gate_up = False
        self.diagnostic_callback: Callable[[str, torch.Tensor], None] | None = None

    def load_weights(
        self,
        state: ScopedWeightLoader,
        context: ParallelLoadContext,
    ) -> None:
        state.load_fused(
            self.gate_up_proj.weight,
            ["gate_proj.weight", "up_proj.weight"],
            "{gate,up}_proj.weight",
            context.tp_rank,
            context.tp_size,
        )
        state.load_tensor(
            self.down_proj.weight,
            "down_proj.weight",
            dim=1,
            rank=context.tp_rank,
            world_size=context.tp_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.split_gate_up:
            weight = self.gate_up_proj.weight
            local_intermediate_size = weight.shape[0] // 2
            gate_up = torch.cat(
                [
                    torch.nn.functional.linear(hidden_states, weight[:local_intermediate_size]),
                    torch.nn.functional.linear(hidden_states, weight[local_intermediate_size:]),
                ],
                dim=-1,
            )
        else:
            gate_up = self.gate_up_proj(hidden_states)
        callback = self.diagnostic_callback
        if callback is not None:
            gate, up = gate_up.chunk(2, dim=-1)
            callback("gate_projection", gate)
            callback("up_projection", up)
        activated = kernels.silu_and_mul(gate_up)
        if callback is not None:
            callback("silu_and_mul", activated)
        output = self.down_proj(activated)
        if callback is not None:
            callback("down_projection", output)
        return output
