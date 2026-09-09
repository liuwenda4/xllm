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

from types import SimpleNamespace

import pytest
import torch

from tools.minimax_h3_weight_ledger import (
    _expected_transformer_dtype,
    _reorder_interleaved_qkv,
    _swap_halves,
    _validate_exact_keys,
    _validate_tensor_metadata,
    _validate_variant,
)


def test_qkv_deinterleave_split_preserves_head_order() -> None:
    weight = torch.arange(12).reshape(12, 1)

    query, key, value = _reorder_interleaved_qkv(weight, num_heads=2, head_dim=2)

    assert query.flatten().tolist() == [0, 1, 6, 7]
    assert key.flatten().tolist() == [2, 3, 8, 9]
    assert value.flatten().tolist() == [4, 5, 10, 11]


def test_qkv_deinterleave_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="QKV row mismatch"):
        _reorder_interleaved_qkv(torch.zeros(11, 1), num_heads=2, head_dim=2)


def test_swap_halves_preserves_rows_without_transpose() -> None:
    weight = torch.arange(12).reshape(4, 3)

    swapped = _swap_halves(weight)

    assert swapped.tolist() == [weight[2].tolist(), weight[3].tolist(), weight[0].tolist(), weight[1].tolist()]


def test_swap_halves_rejects_odd_rows() -> None:
    with pytest.raises(ValueError, match="odd first dimension"):
        _swap_halves(torch.zeros(3, 2))


def test_dropped_rope_keeps_fp32_source_contract() -> None:
    converter = SimpleNamespace(
        MINIMAX_H3_TRANSFORMER_DROPPED_KEYS=("rope.inv_freq",),
        MINIMAX_H3_FP32_SOURCE_PREFIXES=("video_patch_proj.",),
    )

    assert _expected_transformer_dtype("rope.inv_freq", converter) == "F32"


def test_exact_key_coverage_rejects_missing_and_unexpected() -> None:
    with pytest.raises(ValueError, match=r"missing=\['b'\].*unexpected=\['c'\]"):
        _validate_exact_keys({"a", "c"}, {"a", "b"}, "fixture")


@pytest.mark.parametrize(
    ("metadata", "shape", "dtype", "message"),
    [
        ({"shape": [2, 3], "dtype": "BF16", "nbytes": 12}, [3, 2], "BF16", "shape mismatch"),
        ({"shape": [2, 3], "dtype": "F32", "nbytes": 24}, [2, 3], "BF16", "dtype mismatch"),
        ({"shape": [2, 3], "dtype": "BF16", "nbytes": 10}, [2, 3], "BF16", "byte-size mismatch"),
    ],
)
def test_tensor_contract_rejects_shape_dtype_and_size_drift(
    metadata: dict[str, object], shape: list[int], dtype: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_tensor_metadata(metadata, shape, dtype, "fixture")


def test_variant_rejects_fl2va_for_ref2va() -> None:
    model_index = {"_minimax_h3": {"partition": "fl2va"}}

    with pytest.raises(ValueError, match="expected ref2va, got fl2va"):
        _validate_variant(model_index, "ref2va")


def test_variant_accepts_ref2va() -> None:
    _validate_variant({"_minimax_h3": {"partition": "ref2va"}}, "ref2va")
