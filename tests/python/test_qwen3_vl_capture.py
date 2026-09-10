# Copyright 2026 The xLLM Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.models.aux_hidden_capture import AuxHiddenCapture
from xllm.python.models.qwen3_vl import Qwen3VLModel


class _Embedding(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        values = input_ids.to(torch.float32)
        return torch.stack((values, values + 10.0), dim=-1)


class _ResidualLayer(nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = delta

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        mrope_section: list[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions, cos_sin_cache, cos, sin, mrope_section
        residual = hidden if residual is None else hidden + residual
        return torch.full_like(hidden, self.delta), residual


class _FinalNorm(nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return (hidden if residual is None else hidden + residual), residual


class _IdentityNorm(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _ConstantBranch(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, hidden: torch.Tensor, *args: object) -> torch.Tensor:
        template = args[0] if args else hidden
        assert isinstance(template, torch.Tensor)
        return torch.full_like(template, self.value)


class _ExactLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = _IdentityNorm()
        self.self_attn = _ConstantBranch(1.0)
        self.post_attention_layernorm = _IdentityNorm()
        self.mlp = _ConstantBranch(2.0)


class _Rotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("cos_sin_cache", torch.empty(0), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.zeros((*positions.shape, 1), dtype=torch.float32)
        return values, values


def _model(layers_to_capture: tuple[int, ...]) -> Qwen3VLModel:
    model = Qwen3VLModel.__new__(Qwen3VLModel)
    nn.Module.__init__(model)
    model.embed_tokens = _Embedding()
    model.rotary = _Rotary()
    model.mrope_section = []
    model.layers = nn.ModuleList([_ResidualLayer(1.0), _ResidualLayer(2.0)])
    model.norm = _FinalNorm()
    model.aux_hidden_capture = AuxHiddenCapture(layers_to_capture)
    model.exact_residual = False
    model._inputs_embeds = None
    model.deepstack_input_embeds = None
    return model


def test_capture_includes_deepstack_and_precedes_final_norm() -> None:
    model = _model((0,))
    embedded = model.embed_tokens(torch.tensor([1, 2]))
    deepstack = torch.full_like(embedded, 100.0)
    model.deepstack_input_embeds = [deepstack]

    output = model(torch.tensor([1, 2]), torch.tensor([0, 1]))

    assert isinstance(output, tuple)
    final_hidden, captured = output
    torch.testing.assert_close(captured, embedded + 101.0)
    torch.testing.assert_close(final_hidden, embedded + 103.0)


def test_capture_uses_premerged_multimodal_embeddings() -> None:
    model = _model((0,))
    inputs_embeds = torch.tensor([[20.0, 21.0], [30.0, 31.0]])
    model._inputs_embeds = inputs_embeds

    _, captured = model(torch.tensor([1, 2]), torch.tensor([0, 1]))

    torch.testing.assert_close(captured, inputs_embeds + 1.0)


def test_disabled_capture_preserves_tensor_return_type() -> None:
    model = _model(())

    output = model(torch.tensor([1, 2]), torch.tensor([0, 1]))

    assert isinstance(output, torch.Tensor)


def test_exact_residual_mode_matches_official_deepstack_order() -> None:
    model = _model((0,))
    model.layers = nn.ModuleList([_ExactLayer(), _ExactLayer()])
    model.norm = _IdentityNorm()
    model.exact_residual = True
    embedded = model.embed_tokens(torch.tensor([1, 2]))
    model.deepstack_input_embeds = [torch.full_like(embedded, 100.0)]

    final_hidden, captured = model(torch.tensor([1, 2]), torch.tensor([0, 1]))

    torch.testing.assert_close(captured, embedded + 103.0)
    torch.testing.assert_close(final_hidden, embedded + 106.0)
