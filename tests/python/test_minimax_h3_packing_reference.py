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

import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file

from tools.minimax_h3_packing_reference import (
    DEFAULT_SOURCE_ROOT,
    PINNED_REVISION,
    _build_reference,
    _normalize_layout,
    _write_reference,
)


def _raw_layout() -> dict[str, object]:
    return {
        "seq_len": torch.tensor(4),
        "input_ids": torch.tensor([-5, -5, -11, -1]),
        "image_mask": torch.tensor([False, False, True, False]),
        "audio_mask": torch.zeros(4, dtype=torch.bool),
        "img_pos": torch.tensor([2]),
        "audio_pos": torch.empty(0, dtype=torch.int64),
        "text_pos": torch.tensor([0, 1]),
        "update_mask": torch.tensor([False]),
        "audio_update_mask": torch.empty(0, dtype=torch.bool),
        "img_position_ids": torch.zeros((4, 3), dtype=torch.float64),
        "token_tags": torch.tensor([1, 1, 0, -1]),
        "cu_seqlens": torch.tensor([0, 3, 4], dtype=torch.int32),
        "document_id": torch.tensor([0, 0, 0, 1], dtype=torch.int32),
        "latent_grid": torch.tensor([1, 1, 1]),
        "video_row_start": torch.tensor(2),
        "video_spans": ({"start": 2, "latent_grid": (1, 1, 1), "role": "target"},),
    }


def test_normalize_layout_overlays_c1_tags_and_renames_positions() -> None:
    tensors, metadata = _normalize_layout(_raw_layout(), torch.tensor([[0, 1]], dtype=torch.int64))

    assert torch.equal(tensors["token_tags"], torch.tensor([0, 1, 0, -1]))
    assert "position_ids" in tensors
    assert "img_position_ids" not in tensors
    assert metadata == {
        "used_length": 3,
        "aligned_length": 4,
        "video_spans": [{"start": 2, "latent_grid": [1, 1, 1], "role": "target"}],
    }


@pytest.mark.parametrize(
    "tags",
    [
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([[0, 2]], dtype=torch.int64),
        torch.tensor([[0, 1]], dtype=torch.int32),
    ],
)
def test_normalize_layout_rejects_invalid_condition_tags(tags: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="condition_tags"):
        _normalize_layout(_raw_layout(), tags)


def test_write_reference_roundtrips_safetensors_and_hashes_artifact(tmp_path) -> None:
    archive = {
        "case.scalar": torch.tensor(7, dtype=torch.int64),
        "case.hidden": torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).to(torch.bfloat16),
    }
    manifest = {"schema": "test", "tensor_archive": "minimax_h3_packing_reference.safetensors"}

    manifest_path = _write_reference(tmp_path, manifest, archive)
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensor_path = tmp_path / persisted["tensor_archive"]
    loaded = load_file(str(tensor_path), device="cpu")

    assert set(loaded) == set(archive)
    for key, expected in archive.items():
        assert loaded[key].dtype == expected.dtype
        assert loaded[key].shape == expected.shape
        assert torch.equal(loaded[key], expected)
    assert persisted["artifacts"] == {
        tensor_path.name: {"sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest()}
    }


@pytest.mark.skipif(not DEFAULT_SOURCE_ROOT.is_dir(), reason="pinned vllm-omni checkout is unavailable")
def test_pinned_generator_loads_without_vllm_package() -> None:
    manifest, archive = _build_reference(DEFAULT_SOURCE_ROOT)

    assert manifest["golden"]["revision"] == PINNED_REVISION
    assert manifest["tensor_archive"].endswith(".safetensors")
    assert set(manifest["cases"]) == {"one_image", "image_audio", "video", "video_audio", "mixed"}
    assert manifest["cases"]["one_image"]["used_length"] == 23
    assert manifest["cases"]["video_audio"]["aligned_length"] == 128
    assert torch.equal(archive["one_image.token_tags"][:3], torch.tensor([0, 1, 1]))
    assert torch.equal(archive["transforms.video_latent"], archive["transforms.video_roundtrip"])
    assert torch.equal(archive["transforms.audio_latent"], archive["transforms.audio_roundtrip"])
