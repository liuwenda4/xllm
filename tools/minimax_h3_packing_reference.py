#!/usr/bin/env python3
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
"""Generate deterministic MiniMax-H3 packing tensors from the pinned Golden."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import types
import uuid
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

PINNED_REVISION = "e540dfb72f0439be08c26fe61912d8df572fdc7c"
DEFAULT_SOURCE_ROOT = Path("/data/workspace/lwd/vllm-omni")
SOURCE_FILES = {
    "packed_sequence.py": "41d8354fdd8fbc8ccb84f271a3a18b61e7ae8883c2938422cc3be2e6f5576e26",
    "packed_tokens.py": "20994e723054071e34acc41398e9d1636c226802e49894ac9162f5a808feb06f",
}
TENSOR_FIELDS = (
    "seq_len",
    "input_ids",
    "image_mask",
    "audio_mask",
    "img_pos",
    "audio_pos",
    "text_pos",
    "update_mask",
    "audio_update_mask",
    "img_position_ids",
    "token_tags",
    "cu_seqlens",
    "document_id",
    "latent_grid",
    "video_row_start",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths(source_root: Path) -> dict[str, Path]:
    base = source_root / "vllm_omni" / "diffusion" / "models" / "minimax_h3"
    return {name: base / name for name in SOURCE_FILES}


def _source_metadata(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    paths = _source_paths(source_root)
    files = []
    for name, expected_digest in SOURCE_FILES.items():
        path = paths[name]
        if not path.is_file():
            raise ValueError(f"Golden source file does not exist: {path}")
        digest = _sha256(path)
        if digest != expected_digest:
            raise ValueError(f"Golden source digest mismatch for {name}: {digest}")
        files.append({"path": str(path.relative_to(source_root)), "sha256": digest})
    revision = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PINNED_REVISION:
        raise ValueError(f"Golden source revision mismatch: {revision}")
    return {"revision": revision, "source_root": str(source_root), "files": files}


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot create import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_golden_modules(source_root: Path) -> tuple[types.ModuleType, types.ModuleType]:
    paths = _source_paths(source_root.resolve())
    return (
        _load_module("_minimax_h3_packed_sequence_golden", paths["packed_sequence.py"]),
        _load_module("_minimax_h3_packed_tokens_golden", paths["packed_tokens.py"]),
    )


def _cases() -> dict[str, dict[str, Any]]:
    target = {"latent_t": 2, "latent_h": 4, "latent_w": 6, "audio_t": 2, "audio_channel": 2}
    return {
        "one_image": {
            **target,
            "condition_tags": [0, 1, 1],
            "ref_blocks": [{"kind": "image", "latent_h": 4, "latent_w": 4}],
        },
        "image_audio": {
            **target,
            "condition_tags": [1, 0, 1],
            "ref_blocks": [
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {"kind": "audio", "ref_audio_t": 2},
            ],
        },
        "video": {
            **target,
            "condition_tags": [1, 1, 0],
            "ref_blocks": [
                {
                    "kind": "video",
                    "ref_audio_t": 0,
                    "latent_t": 3,
                    "latent_h": 4,
                    "latent_w": 4,
                }
            ],
        },
        "video_audio": {
            **target,
            "condition_tags": [0, 1, 0],
            "seq_len": 128,
            "ref_blocks": [
                {
                    "kind": "video_audio",
                    "ref_audio_t": 2,
                    "latent_t": 3,
                    "latent_h": 4,
                    "latent_w": 4,
                }
            ],
        },
        "mixed": {
            **target,
            "condition_tags": [0, 1, 0, 1],
            "ref_blocks": [
                {"kind": "image", "latent_h": 2, "latent_w": 4},
                {"kind": "audio", "ref_audio_t": 3},
                {
                    "kind": "video",
                    "ref_audio_t": 0,
                    "latent_t": 2,
                    "latent_h": 2,
                    "latent_w": 4,
                },
                {
                    "kind": "video_audio",
                    "ref_audio_t": 2,
                    "latent_t": 3,
                    "latent_h": 4,
                    "latent_w": 2,
                },
            ],
        },
    }


def _condition_hidden(text_len: int) -> torch.Tensor:
    values = torch.arange(text_len * 5120, dtype=torch.float32)
    return values.remainder(257).sub(128).reshape(1, text_len, 5120).to(torch.bfloat16).contiguous()


def _normalize_layout(
    raw: dict[str, object], condition_tags: torch.Tensor
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if condition_tags.dtype != torch.int64 or condition_tags.ndim != 2 or condition_tags.shape[0] != 1:
        raise ValueError("condition_tags must be int64 with shape [1,N]")
    if not bool(torch.logical_or(condition_tags == 0, condition_tags == 1).all().item()):
        raise ValueError("condition_tags values must be 0 or 1")
    tensors: dict[str, torch.Tensor] = {}
    for field in TENSOR_FIELDS:
        value = raw.get(field)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Golden layout is missing tensor field {field!r}")
        tensors["position_ids" if field == "img_position_ids" else field] = value.detach().cpu().contiguous()
    text_len = int(condition_tags.shape[1])
    token_tags = tensors["token_tags"].clone()
    if text_len > token_tags.numel():
        raise ValueError("condition tags exceed the packed sequence")
    token_tags[:text_len].copy_(condition_tags[0])
    tensors["token_tags"] = token_tags

    spans = raw.get("video_spans")
    if not isinstance(spans, tuple):
        raise ValueError("Golden layout video_spans must be a tuple")
    normalized_spans = []
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("Golden layout video span must be an object")
        normalized_spans.append(
            {
                "start": int(span["start"]),
                "latent_grid": [int(value) for value in span["latent_grid"]],
                "role": str(span["role"]),
            }
        )
    metadata = {
        "used_length": int(tensors["cu_seqlens"][1].item()),
        "aligned_length": int(tensors["seq_len"].item()),
        "video_spans": normalized_spans,
    }
    return tensors, metadata


def _tensor_digest(tensor: torch.Tensor) -> str:
    byte_view = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "sha256": _tensor_digest(tensor),
    }


def _build_reference(source_root: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    source = _source_metadata(source_root)
    packed_sequence, packed_tokens = _load_golden_modules(source_root)
    archive: dict[str, torch.Tensor] = {}
    case_metadata: dict[str, Any] = {}
    for name, case in _cases().items():
        condition_tags = torch.tensor([case["condition_tags"]], dtype=torch.int64)
        hidden = _condition_hidden(condition_tags.shape[1])
        kwargs = {key: value for key, value in case.items() if key != "condition_tags"}
        kwargs["text_len"] = int(condition_tags.shape[1])
        raw = packed_sequence.minimax_h3_packed_sequence_ref2va_blocks(**kwargs)
        tensors, normalized = _normalize_layout(raw, condition_tags)
        tensors["condition_hidden"] = hidden
        tensors["condition_tags"] = condition_tags
        for field, tensor in tensors.items():
            archive[f"{name}.{field}"] = tensor
        case_metadata[name] = {
            "input": case,
            **normalized,
            "tensors": {field: _tensor_summary(tensor) for field, tensor in tensors.items()},
        }

    video = torch.arange(2 * 3 * 2 * 4 * 6, dtype=torch.float32).reshape(2, 3, 2, 4, 6)
    video_rows = packed_tokens.minimax_h3_patchify_video_latent(video, patch_size=(1, 2, 2))
    video_roundtrip = packed_tokens.minimax_h3_unpatchify_video_tokens(
        video_rows,
        latent_shape=(2, 2, 3, 3),
        patch_size=(1, 2, 2),
    )
    audio = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    audio_rows = packed_tokens.minimax_h3_pack_audio_latent(audio)
    audio_roundtrip = packed_tokens.minimax_h3_unpack_audio_tokens(audio_rows, audio_t=8, audio_channel=2)
    transforms = {
        "video_latent": video,
        "video_rows": video_rows,
        "video_roundtrip": video_roundtrip,
        "audio_latent": audio,
        "audio_rows": audio_rows,
        "audio_roundtrip": audio_roundtrip,
    }
    for field, tensor in transforms.items():
        archive[f"transforms.{field}"] = tensor

    manifest = {
        "schema": "xllm.minimax_h3.packing_reference/v1",
        "golden": source,
        "normalization": {
            "condition_tag_overlay": "C1 tags replace Golden text defaults",
            "position_field": "img_position_ids renamed to position_ids",
        },
        "cases": case_metadata,
        "transforms": {field: _tensor_summary(tensor) for field, tensor in transforms.items()},
        "tensor_archive": "minimax_h3_packing_reference.safetensors",
    }
    return manifest, archive


def _write_reference(output_dir: Path, manifest: dict[str, Any], archive: dict[str, torch.Tensor]) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_name = manifest.get("tensor_archive")
    if not isinstance(tensor_name, str) or Path(tensor_name).name != tensor_name:
        raise ValueError("tensor_archive must be a file name")

    tensor_path = output_dir / tensor_name
    manifest_path = output_dir / "minimax_h3_packing_reference.json"
    suffix = f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    tensor_temp = output_dir / f".{tensor_name}{suffix}"
    manifest_temp = output_dir / f".{manifest_path.name}{suffix}"
    try:
        save_file(archive, str(tensor_temp))
        manifest["artifacts"] = {tensor_name: {"sha256": _sha256(tensor_temp)}}
        with manifest_temp.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, allow_nan=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tensor_temp, tensor_path)
        os.replace(manifest_temp, manifest_path)
    finally:
        tensor_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)
    return manifest_path


def main() -> None:
    args = _parse_args()
    manifest, archive = _build_reference(args.source_root)
    manifest_path = _write_reference(args.output_dir, manifest, archive)
    print(manifest_path)


if __name__ == "__main__":
    main()
