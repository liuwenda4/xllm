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
"""Prepare and run the pinned MiniMax-H3 C7 production Ref2VA reference."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger
from tools.minimax_h3_ref2va_e2e_reference import (
    AUDIO_SCHEDULER_CHECKPOINT_SHA256,
    AUDIO_VAE_CHECKPOINT_SHA256,
    DIFFUSERS_SOURCE_SHA256,
    ROOT_CHECKPOINT_SHA256,
    TRANSFORMER_CHECKPOINT_SHA256,
    VIDEO_SCHEDULER_CHECKPOINT_SHA256,
    VIDEO_VAE_CHECKPOINT_SHA256,
    PackedLayout,
    ReferenceSpec,
    _activate_diffusers_source,
    _checkpoint_manifest,
    _clear_npu,
    _denormalize_latents,
    _memory_stats,
    _pack_video_latents,
    _parameter_dtype,
    _place_transformer,
    _set_native_attention,
    _unpack_audio_rows,
    _unpack_video_rows,
)

PINNED_DIFFUSERS_REVISION = "d30c748f5f5d0925a5af14dc0e6a6de983025e63"
DEFAULT_DIFFUSERS_SOURCE = Path("/data/workspace/lwd/minimax/diffusers-reference")
DEFAULT_CHECKPOINT_ROOT = Path("/data/workspace/lwd/minimax/checkpoints/Ref2VA-diffusers-d30c748f")
DEFAULT_OFFICIAL_CONDITION = Path(
    "/data/workspace/lwd/minimax/artifacts/h3/20260910T155745Z-h3-c1/condition-cache/"
    "2b464f60aba38f5831325a27a9a7acb2ed62589aaa1bcec1c37574e38146a7a6/condition.safetensors"
)
DEFAULT_NATIVE_CONDITION = Path(
    "/data/workspace/lwd/minimax/artifacts/h3/20260912T121822Z-h3-c7/native-qwen-debug/"
    "20260912T161052Z-first-divergence/condition-downstream-c1/condition_downstream.safetensors"
)
DEFAULT_NATIVE_SUMMARY = Path(
    "/data/workspace/lwd/minimax/artifacts/h3/20260912T121822Z-h3-c7/native-qwen-debug/"
    "20260912T161052Z-first-divergence/condition-downstream-c1/summary.json"
)
DEFAULT_REFERENCE_IMAGE = Path("/data/workspace/lwd/minimax/artifacts/h1/20260909T083601Z-h1/inputs/reference.png")
DEFAULT_OUTPUT_DIR = Path("/data/workspace/lwd/minimax/artifacts/h3/20260912T121822Z-h3-c7/production")

PREPARED_ARCHIVE_NAME = "minimax_h3_ref2va_production_prepared.safetensors"
PREPARED_MANIFEST_NAME = "minimax_h3_ref2va_production_prepared.json"
PREPARED_SCHEMA = "xllm.minimax_h3.ref2va_production_prepared/v1"
EXECUTION_SCHEMA = "xllm.minimax_h3.ref2va_production_execution/v1"

OFFICIAL_CONDITION_SHA256 = "524652e82e9b20db4ede5460347b4ce309dce33f7db983a8cc7ae2e7cfe4f496"
OFFICIAL_CONDITION_MANIFEST_SHA256 = "9a6a0e2b1291658b26e55188e2136d257e45dbea77ef8cdb8f634be66df52c6b"
OFFICIAL_CONDITION_CACHE_KEY = "2b464f60aba38f5831325a27a9a7acb2ed62589aaa1bcec1c37574e38146a7a6"
OFFICIAL_HIDDEN_SHA256 = "c6c68cc020b976843aafde9b115eb50a84bea89d16bd8ee7fba70a2ef249a1b5"
CONDITION_TAGS_SHA256 = "fc50e76d6420e4d0768f49284dcfabc29f4a1986a4cb246a8e01c87b5c250137"
NATIVE_CONDITION_SHA256 = "77a8fcc41b741887276da8407bb32a91d1e18da98d41c520e58ce7bf34d47408"
NATIVE_SUMMARY_SHA256 = "9c0bf7244f6889638ec7c63cb91016406293c34ae1b235695ec94ead5f7bffb2"
NATIVE_HIDDEN_SHA256 = "c6c68cc020b976843aafde9b115eb50a84bea89d16bd8ee7fba70a2ef249a1b5"
REFERENCE_IMAGE_SHA256 = "b32290cdabb1fe08982ecca971b3df7e14b22c661b171db610c1fa66198881b9"

CONDITION_SHAPE = (11350, 5120)
CONDITION_TAGS_SHAPE = (11350,)
SOURCE_IMAGE_SIZE = (3616, 1336)
RESIZED_IMAGE_SIZE = (5536, 2048)
RESIZED_PIXELS_SHAPE = (1, 3, 1, 2048, 5536)
VISUAL_LATENT_SHAPE = (1, 24, 1, 128, 346)
TARGET_VIDEO_SHAPE = (1, 24, 37, 48, 84)
TARGET_AUDIO_SHAPE = (2, 32, 207)
TARGET_AUDIO_ROWS_SHAPE = (414, 32)
DECODED_VIDEO_SHAPE = (1, 3, 124, 768, 1344)
DECODED_AUDIO_SHAPE = (1, 2, 165600)

PATCH_SIZE = (1, 2, 2)
AUDIO_CHANNELS = 2
VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2
VIDEO_FPS = 24
AUDIO_SAMPLE_RATE = 32000
REFERENCE_SHORT_EDGE = 2048
CANVAS_MULTIPLE = 32
TARGET_NUM_FRAMES = 124
TARGET_HEIGHT = 768
TARGET_WIDTH = 1344
VISUAL_ANCHOR_TIMESTEP = 0.999
POSTERIOR_SEED = 42
REQUEST_SEED = 42
SIGMA_POINTS = 50
TRANSFORMER_FORWARDS = 49
TRANSFORMER_LAYERS = 50
SNAPSHOT_FORWARDS = (1, 2, 4, 8, 49)
REQUEST_DRAW_ORDER = ("visual_anchor_noise", "target_video_initial", "target_audio_rows")

TEXT_ROWS = CONDITION_SHAPE[0]
IMAGE_ROWS = 11072
TARGET_AUDIO_ROWS = 414
TARGET_VIDEO_ROWS = 37296
USED_ROWS = 60132
ALIGNED_ROWS = 60160
VIDEO_COMPACT_ROWS = IMAGE_ROWS + TARGET_VIDEO_ROWS
AUDIO_COMPACT_ROWS = TARGET_AUDIO_ROWS
VIDEO_PATCH_WIDTH = 96

PREPARED_CONDITION_KEYS = {
    "official_hf": "condition.official_hf.hidden",
    "xllm_native": "condition.xllm_native.hidden",
}
COMMON_EXECUTION_KEYS = {
    "condition.text_token_tags",
    "initial.video_rows",
    "initial.audio_rows",
    "layout.position_ids",
    "layout.token_tags",
    "layout.video_indices",
    "layout.audio_indices",
    "layout.text_indices",
    "layout.video_update_mask",
    "layout.audio_update_mask",
    "layout.aligned_valid_mask",
    "layout.used_rows",
    "layout.aligned_rows",
    "layout.num_condition_video_rows",
    "layout.num_condition_audio_rows",
    "layout.ranges",
    "schedule.video_sigmas",
    "schedule.audio_sigmas",
    "schedule.video_timesteps",
    "schedule.audio_timesteps",
}

PRODUCTION_DIFFUSERS_SOURCE_SHA256 = {
    **DIFFUSERS_SOURCE_SHA256,
    "src/diffusers/image_processor.py": "d7558c34e9dfd36ea2a8721fc670615ca868de551ba232992a7f5382924b98f8",
    "src/diffusers/modular_pipelines/minimax_h3/before_encoder.py": (
        "03612baa8b983d058884d2c1740e57342279a1124002553f78cec98abcfc7c28"
    ),
    "src/diffusers/modular_pipelines/minimax_h3/denoise.py": (
        "bf0224f3ac8f3bba8366599143f60d78bd14e9faf77cdb207a844549bb8c1dc4"
    ),
    "src/diffusers/modular_pipelines/minimax_h3/encoders.py": (
        "fea751a889752ba58f1528acbc23b223e827ea707a75e2ae0c43d4a53ce758af"
    ),
    "src/diffusers/utils/export_utils.py": "881e6c39011013236d0027479ded42f3c07e310b8b32f664dbcfe8438db32d6e",
    "src/diffusers/utils/torch_utils.py": "64877dee5d2d9374a0cd89e889a2c6be82d902578ceea2b1dc4831747dd4568e",
}
PRODUCTION_CHECKPOINT_SHA256 = {
    "transformer_ref": TRANSFORMER_CHECKPOINT_SHA256,
    "vae": VIDEO_VAE_CHECKPOINT_SHA256,
    "audio_vae": AUDIO_VAE_CHECKPOINT_SHA256,
    "scheduler": VIDEO_SCHEDULER_CHECKPOINT_SHA256,
    "audio_scheduler": AUDIO_SCHEDULER_CHECKPOINT_SHA256,
}


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...] | None
    dtype: torch.dtype


@dataclass(frozen=True)
class RequestNoise:
    visual_anchor_noise: torch.Tensor
    target_video_initial: torch.Tensor
    target_audio_rows: torch.Tensor


RandnTensor = Callable[..., torch.Tensor]
LayoutBuilder = Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]]
RowTimestepBuilder = Callable[..., tuple[torch.Tensor, torch.Tensor]]


def _prepared_tensor_specs() -> dict[str, TensorSpec]:
    specs = {
        "condition.official_hf.hidden": TensorSpec(CONDITION_SHAPE, torch.bfloat16),
        "condition.xllm_native.hidden": TensorSpec(CONDITION_SHAPE, torch.bfloat16),
        "condition.text_token_tags": TensorSpec(CONDITION_TAGS_SHAPE, torch.int64),
        "input.reference_resized_uint8": TensorSpec(RESIZED_PIXELS_SHAPE, torch.uint8),
        "input.visual_latent_normalized": TensorSpec(VISUAL_LATENT_SHAPE, torch.float32),
        "input.visual_anchor_noise": TensorSpec(VISUAL_LATENT_SHAPE, torch.float32),
        "input.visual_anchor_noised": TensorSpec(VISUAL_LATENT_SHAPE, torch.float32),
        "input.target_video_initial": TensorSpec(TARGET_VIDEO_SHAPE, torch.float32),
        "input.target_audio_initial_rows": TensorSpec(TARGET_AUDIO_ROWS_SHAPE, torch.float32),
        "input.target_audio_initial_latent": TensorSpec(TARGET_AUDIO_SHAPE, torch.float32),
        "initial.video_rows": TensorSpec((VIDEO_COMPACT_ROWS, VIDEO_PATCH_WIDTH), torch.float32),
        "initial.audio_rows": TensorSpec(TARGET_AUDIO_ROWS_SHAPE, torch.float32),
        "layout.position_ids": TensorSpec((USED_ROWS, 3), torch.float64),
        "layout.token_tags": TensorSpec((USED_ROWS,), torch.int64),
        "layout.video_indices": TensorSpec((VIDEO_COMPACT_ROWS,), torch.int64),
        "layout.audio_indices": TensorSpec((AUDIO_COMPACT_ROWS,), torch.int64),
        "layout.text_indices": TensorSpec((TEXT_ROWS,), torch.int64),
        "layout.video_update_mask": TensorSpec((VIDEO_COMPACT_ROWS,), torch.bool),
        "layout.audio_update_mask": TensorSpec((AUDIO_COMPACT_ROWS,), torch.bool),
        "layout.aligned_valid_mask": TensorSpec((ALIGNED_ROWS,), torch.bool),
        "layout.used_rows": TensorSpec((), torch.int64),
        "layout.aligned_rows": TensorSpec((), torch.int64),
        "layout.num_condition_video_rows": TensorSpec((), torch.int64),
        "layout.num_condition_audio_rows": TensorSpec((), torch.int64),
        "layout.ranges": TensorSpec((4, 2), torch.int64),
        "schedule.video_sigmas": TensorSpec((SIGMA_POINTS,), torch.float32),
        "schedule.audio_sigmas": TensorSpec((SIGMA_POINTS,), torch.float32),
        "schedule.video_timesteps": TensorSpec((TRANSFORMER_FORWARDS,), torch.float32),
        "schedule.audio_timesteps": TensorSpec((TRANSFORMER_FORWARDS,), torch.float32),
    }
    for forward_number in range(1, TRANSFORMER_FORWARDS + 1):
        prefix = f"schedule.forward_{forward_number:03d}"
        specs[f"{prefix}.unique_timesteps"] = TensorSpec(None, torch.float32)
        specs[f"{prefix}.timestep_indices"] = TensorSpec((USED_ROWS,), torch.int64)
    return specs


PREPARED_TENSOR_SPECS = _prepared_tensor_specs()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare and attest all common production inputs")
    prepare.add_argument("--diffusers-source", type=Path, default=DEFAULT_DIFFUSERS_SOURCE)
    prepare.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    prepare.add_argument("--official-condition", type=Path, default=DEFAULT_OFFICIAL_CONDITION)
    prepare.add_argument("--native-condition", type=Path, default=DEFAULT_NATIVE_CONDITION)
    prepare.add_argument("--native-summary", type=Path, default=DEFAULT_NATIVE_SUMMARY)
    prepare.add_argument("--reference-image", type=Path, default=DEFAULT_REFERENCE_IMAGE)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare.add_argument("--vae-device", default="npu:0")

    execute = subparsers.add_parser("execute", help="Run one backend from an attested prepared artifact")
    execute.add_argument("--diffusers-source", type=Path, default=DEFAULT_DIFFUSERS_SOURCE)
    execute.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    execute.add_argument("--prepared-artifact", type=Path, default=DEFAULT_OUTPUT_DIR / PREPARED_ARCHIVE_NAME)
    execute.add_argument("--prepared-manifest", type=Path, default=DEFAULT_OUTPUT_DIR / PREPARED_MANIFEST_NAME)
    execute.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    execute.add_argument("--condition-backend", choices=tuple(PREPARED_CONDITION_KEYS), required=True)
    execute.add_argument("--execution-mode", choices=("smoke", "full"), default="smoke")
    execute.add_argument("--devices", nargs=2, default=("npu:0", "npu:1"), metavar=("FIRST", "SECOND"))
    execute.add_argument("--vae-device", default="npu:0")
    execute.add_argument("--write-media", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _with_digest(value: dict[str, Any]) -> dict[str, Any]:
    value["digest"] = _canonical_digest(value)
    return value


def _validate_section_digest(value: Any, name: str) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("digest"), str):
        raise ValueError(f"prepared manifest {name} must carry a canonical digest")
    expected = dict(value)
    digest = expected.pop("digest")
    if _canonical_digest(expected) != digest:
        raise ValueError(f"prepared manifest {name} canonical digest mismatch")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous()
    byte_view = value.reshape(-1).view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    chunk_size = 16 * 1024 * 1024
    for offset in range(0, byte_view.numel(), chunk_size):
        digest.update(memoryview(byte_view[offset : offset + chunk_size].numpy()))
    return digest.hexdigest()


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().to("cpu").contiguous()
    flat = value.reshape(-1)
    finite = True
    count = 0
    total = 0.0
    total_squared = 0.0
    minimum = math.inf
    maximum = -math.inf
    chunk_size = 1_048_576
    for offset in range(0, flat.numel(), chunk_size):
        chunk = flat[offset : offset + chunk_size]
        if chunk.is_floating_point() and not bool(torch.isfinite(chunk).all().item()):
            finite = False
            chunk = chunk[torch.isfinite(chunk)]
        numeric = chunk.to(torch.float64)
        if numeric.numel():
            count += numeric.numel()
            total += float(numeric.sum().item())
            total_squared += float(numeric.square().sum().item())
            minimum = min(minimum, float(numeric.min().item()))
            maximum = max(maximum, float(numeric.max().item()))
    mean = total / count if count else None
    variance = max(0.0, total_squared / count - mean * mean) if count else None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "numel": value.numel(),
        "finite": finite,
        "min": minimum if count else None,
        "max": maximum if count else None,
        "mean": mean,
        "std": math.sqrt(variance) if variance is not None else None,
        "sha256": _tensor_sha256(value),
    }


def _validate_tensor_contract(tensors: Mapping[str, torch.Tensor], specs: Mapping[str, TensorSpec]) -> None:
    if set(tensors) != set(specs):
        raise ValueError(
            "prepared tensor key mismatch: "
            f"missing={sorted(set(specs) - set(tensors))}, extra={sorted(set(tensors) - set(specs))}"
        )
    for name, spec in specs.items():
        value = tensors[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"prepared value {name!r} is not a tensor")
        if spec.shape is not None and tuple(value.shape) != spec.shape:
            raise ValueError(f"prepared tensor {name!r} has shape {tuple(value.shape)}, expected {spec.shape}")
        if spec.shape is None and (value.ndim != 1 or not 1 <= value.numel() <= 4):
            raise ValueError(f"prepared tensor {name!r} must contain one to four unique timesteps")
        if value.dtype != spec.dtype or not value.is_contiguous():
            raise ValueError(f"prepared tensor {name!r} must be contiguous {spec.dtype}, got {value.dtype}")
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"prepared tensor {name!r} contains NaN or Inf")


def _source_manifest(diffusers_source: Path) -> dict[str, Any]:
    root = diffusers_source.resolve()
    files = {}
    for relative, expected_sha256 in sorted(PRODUCTION_DIFFUSERS_SOURCE_SHA256.items()):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"pinned Diffusers source file does not exist: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"pinned Diffusers source digest mismatch for {path}: {actual_sha256}")
        files[relative] = {"path": str(path), "size": path.stat().st_size, "sha256": actual_sha256}
    generator_path = Path(__file__).resolve()
    return _with_digest(
        {
            "path": str(root),
            "revision": PINNED_DIFFUSERS_REVISION,
            "revision_verification": "pinned_file_sha256",
            "files": files,
            "generator": {
                "path": str(generator_path),
                "size": generator_path.stat().st_size,
                "sha256": _sha256(generator_path),
            },
        }
    )


def _reference_resize_geometry(
    width: int,
    height: int,
    *,
    short_edge: int = REFERENCE_SHORT_EDGE,
    multiple: int = CANVAS_MULTIPLE,
) -> tuple[int, int]:
    if width <= 0 or height <= 0 or short_edge <= 0 or multiple <= 0:
        raise ValueError("reference dimensions, short edge, and canvas multiple must be positive")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"reference image must be within 1:4 and 4:1, got {width}x{height}")
    scale = short_edge / min(width, height)
    target_height = max(multiple, round(height * scale / multiple) * multiple)
    target_width = max(multiple, round(width * scale / multiple) * multiple)
    return target_height, target_width


def _geometry_manifest() -> dict[str, Any]:
    return {
        "reference_source_size_wh": list(SOURCE_IMAGE_SIZE),
        "reference_resized_size_wh": list(RESIZED_IMAGE_SIZE),
        "reference_pixels_shape": list(RESIZED_PIXELS_SHAPE),
        "reference_short_edge": REFERENCE_SHORT_EDGE,
        "canvas_multiple": CANVAS_MULTIPLE,
        "visual_latent_shape": list(VISUAL_LATENT_SHAPE),
        "target": {
            "num_frames": TARGET_NUM_FRAMES,
            "height": TARGET_HEIGHT,
            "width": TARGET_WIDTH,
            "fps": VIDEO_FPS,
            "video_latent_shape": list(TARGET_VIDEO_SHAPE),
            "decoded_video_shape": list(DECODED_VIDEO_SHAPE),
            "audio_latent_shape": list(TARGET_AUDIO_SHAPE),
            "audio_rows_shape": list(TARGET_AUDIO_ROWS_SHAPE),
            "decoded_audio_shape": list(DECODED_AUDIO_SHAPE),
            "audio_sample_rate": AUDIO_SAMPLE_RATE,
        },
        "patch_size": list(PATCH_SIZE),
        "packed_order": ["text", "image", "target_audio", "target_video"],
        "row_counts": {
            "text": TEXT_ROWS,
            "image": IMAGE_ROWS,
            "target_audio": TARGET_AUDIO_ROWS,
            "target_video": TARGET_VIDEO_ROWS,
            "used": USED_ROWS,
            "aligned": ALIGNED_ROWS,
        },
        "ranges": {
            "text": [0, TEXT_ROWS],
            "image": [TEXT_ROWS, TEXT_ROWS + IMAGE_ROWS],
            "target_audio": [TEXT_ROWS + IMAGE_ROWS, TEXT_ROWS + IMAGE_ROWS + TARGET_AUDIO_ROWS],
            "target_video": [TEXT_ROWS + IMAGE_ROWS + TARGET_AUDIO_ROWS, USED_ROWS],
        },
    }


def _torch_randn_tensor(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def _draw_request_noise(
    *,
    seed: int = REQUEST_SEED,
    visual_shape: tuple[int, ...] = VISUAL_LATENT_SHAPE,
    video_shape: tuple[int, ...] = TARGET_VIDEO_SHAPE,
    audio_rows_shape: tuple[int, ...] = TARGET_AUDIO_ROWS_SHAPE,
    draw_order: Sequence[str] = REQUEST_DRAW_ORDER,
    randn_tensor: RandnTensor = _torch_randn_tensor,
    device: str | torch.device = "cpu",
) -> RequestNoise:
    if tuple(draw_order) != REQUEST_DRAW_ORDER and (
        len(draw_order) != len(REQUEST_DRAW_ORDER) or set(draw_order) != set(REQUEST_DRAW_ORDER)
    ):
        raise ValueError(f"request draw order must be a permutation of {REQUEST_DRAW_ORDER}")
    shapes = {
        "visual_anchor_noise": visual_shape,
        "target_video_initial": video_shape,
        "target_audio_rows": audio_rows_shape,
    }
    generator = torch.Generator(device="cpu").manual_seed(seed)
    drawn = {
        name: randn_tensor(
            shapes[name],
            generator=generator,
            device=torch.device(device),
            dtype=torch.float32,
        )
        for name in draw_order
    }
    return RequestNoise(**drawn)


def _prove_pinned_randn_equivalence(randn_tensor: RandnTensor) -> dict[str, Any]:
    shape = (2, 3, 5)
    pinned = _draw_request_noise(
        visual_shape=shape,
        video_shape=shape,
        audio_rows_shape=shape,
        randn_tensor=randn_tensor,
        device="cpu",
    )
    direct = _draw_request_noise(
        visual_shape=shape,
        video_shape=shape,
        audio_rows_shape=shape,
        randn_tensor=_torch_randn_tensor,
        device="cpu",
    )
    fields = REQUEST_DRAW_ORDER
    if any(not torch.equal(getattr(pinned, name), getattr(direct, name)) for name in fields):
        raise ValueError("pinned randn_tensor CPU semantics do not match sequential torch.randn draws")
    return {
        "proved": True,
        "proof_shape": list(shape),
        "pinned_source": "diffusers.utils.torch_utils.randn_tensor",
        "equivalent_to": "sequential torch.randn from one CPU generator",
    }


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"failed to read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _attest_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    sha256: str,
) -> dict[str, Any]:
    if tuple(tensor.shape) != shape or tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype} {shape}, got {tensor.dtype} {tuple(tensor.shape)}")
    value = tensor.contiguous()
    if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains NaN or Inf")
    actual_sha256 = _tensor_sha256(value)
    if actual_sha256 != sha256:
        raise ValueError(f"{name} tensor digest mismatch: {actual_sha256}")
    return {"shape": list(shape), "dtype": str(dtype).removeprefix("torch."), "sha256": actual_sha256}


def _load_and_attest_conditions(
    official_path: Path,
    native_path: Path,
    native_summary_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from safetensors import safe_open

    official_path = official_path.resolve()
    native_path = native_path.resolve()
    native_summary_path = native_summary_path.resolve()
    official_manifest_path = official_path.parent / "manifest.json"
    fixed_files = (
        (official_path, OFFICIAL_CONDITION_SHA256, "official condition"),
        (official_manifest_path, OFFICIAL_CONDITION_MANIFEST_SHA256, "official condition manifest"),
        (native_path, NATIVE_CONDITION_SHA256, "native downstream archive"),
        (native_summary_path, NATIVE_SUMMARY_SHA256, "native downstream summary"),
    )
    for path, expected_sha256, description in fixed_files:
        if not path.is_file():
            raise ValueError(f"{description} does not exist: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"{description} digest mismatch: {actual_sha256}")

    official_manifest = _load_json(official_manifest_path, "official condition manifest")
    if official_manifest.get("backend") != "official_hf":
        raise ValueError("official condition manifest backend mismatch")
    if official_manifest.get("cache_key") != OFFICIAL_CONDITION_CACHE_KEY:
        raise ValueError("official condition cache key mismatch")
    cache_inputs = official_manifest.get("cache_key_inputs", {})
    checkpoint_identity = cache_inputs.get("checkpoint_identity", {}).get("values", {})
    if checkpoint_identity.get("diffusers_revision") != PINNED_DIFFUSERS_REVISION:
        raise ValueError("official condition Diffusers revision mismatch")
    if cache_inputs.get("ordered_references") != [{"type": "image", "sha256": REFERENCE_IMAGE_SHA256}]:
        raise ValueError("official condition ordered image provenance mismatch")
    if (
        official_manifest.get("artifacts", {}).get("condition.safetensors", {}).get("sha256")
        != OFFICIAL_CONDITION_SHA256
    ):
        raise ValueError("official condition manifest artifact digest mismatch")

    with safe_open(str(official_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"prompt_embeds", "text_token_tags"}:
            raise ValueError("official condition archive has unexpected tensor keys")
        official_hidden = handle.get_tensor("prompt_embeds").contiguous()
        official_tags = handle.get_tensor("text_token_tags").contiguous()
    official_hidden_meta = _attest_tensor(
        official_hidden,
        name="official condition hidden",
        shape=CONDITION_SHAPE,
        dtype=torch.bfloat16,
        sha256=OFFICIAL_HIDDEN_SHA256,
    )
    tags_meta = _attest_tensor(
        official_tags,
        name="condition tags",
        shape=CONDITION_TAGS_SHAPE,
        dtype=torch.int64,
        sha256=CONDITION_TAGS_SHA256,
    )
    if not set(official_tags.unique().tolist()).issubset({VIDEO_TAG, TEXT_TAG}):
        raise ValueError("condition tags contain a modality outside video/text")

    native_summary = _load_json(native_summary_path, "native downstream summary")
    if native_summary.get("status") != "H3_NATIVE_DOWNSTREAM_REPORT":
        raise ValueError("native downstream summary status mismatch")
    if Path(native_summary.get("tensor_archive", "")).resolve() != native_path:
        raise ValueError("native downstream summary archive path mismatch")
    comparisons = native_summary.get("comparisons")
    comparison = next(
        (entry for entry in comparisons or [] if isinstance(entry, dict) and entry.get("name") == "layer_49_condition"),
        None,
    )
    if not isinstance(comparison, dict):
        raise ValueError("native downstream summary lacks layer_49_condition attestation")
    if comparison.get("actual_sha256") != NATIVE_HIDDEN_SHA256:
        raise ValueError("native downstream summary hidden digest mismatch")
    if comparison.get("actual_shape") != list(CONDITION_SHAPE) or comparison.get("actual_dtype") != "torch.bfloat16":
        raise ValueError("native downstream summary hidden ABI mismatch")

    with safe_open(str(native_path), framework="pt", device="cpu") as handle:
        required = {"native_layer_49_condition", "text_token_tags"}
        if not required.issubset(handle.keys()):
            raise ValueError("native downstream archive lacks required condition tensors")
        native_hidden = handle.get_tensor("native_layer_49_condition").contiguous()
        native_tags = handle.get_tensor("text_token_tags").contiguous()
    native_hidden_meta = _attest_tensor(
        native_hidden,
        name="native condition hidden",
        shape=CONDITION_SHAPE,
        dtype=torch.bfloat16,
        sha256=NATIVE_HIDDEN_SHA256,
    )
    _attest_tensor(
        native_tags,
        name="native condition tags",
        shape=CONDITION_TAGS_SHAPE,
        dtype=torch.int64,
        sha256=CONDITION_TAGS_SHA256,
    )
    if not torch.equal(native_tags, official_tags):
        raise ValueError("native and official condition tags are not byte-identical")

    inputs = {
        "official_hf": official_hidden,
        "xllm_native": native_hidden,
        "text_token_tags": official_tags,
    }
    attestation = _with_digest(
        {
            "official_condition": {
                "path": str(official_path),
                "sha256": OFFICIAL_CONDITION_SHA256,
                "manifest_path": str(official_manifest_path),
                "manifest_sha256": OFFICIAL_CONDITION_MANIFEST_SHA256,
                "cache_key": OFFICIAL_CONDITION_CACHE_KEY,
                "backend": "official_hf",
                "tensor": official_hidden_meta,
            },
            "native_condition": {
                "path": str(native_path),
                "sha256": NATIVE_CONDITION_SHA256,
                "summary_path": str(native_summary_path),
                "summary_sha256": NATIVE_SUMMARY_SHA256,
                "summary_status": native_summary["status"],
                "backend": "xllm_native",
                "tensor": native_hidden_meta,
            },
            "common_tags": tags_meta,
        }
    )
    return inputs, attestation


def _prepare_reference_pixels(image_path: Path, image_processor: Any) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    image_path = image_path.resolve()
    if not image_path.is_file() or _sha256(image_path) != REFERENCE_IMAGE_SHA256:
        raise ValueError("reference image is missing or its SHA256 does not match the C7 contract")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    if image.size != SOURCE_IMAGE_SIZE:
        raise ValueError(f"reference image size is {image.size}, expected {SOURCE_IMAGE_SIZE}")
    target_height, target_width = _reference_resize_geometry(*image.size)
    if (target_width, target_height) != RESIZED_IMAGE_SIZE:
        raise ValueError("reference resize geometry does not match the pinned production contract")
    resized = image_processor.resize(image, height=target_height, width=target_width)
    if resized.mode != "RGB" or resized.size != RESIZED_IMAGE_SIZE:
        raise ValueError("pinned VaeImageProcessor did not produce the expected RGB reference canvas")
    pixels = torch.from_numpy(np.array(resized, dtype=np.uint8, copy=True)).permute(2, 0, 1)[None, :, None].contiguous()
    if tuple(pixels.shape) != RESIZED_PIXELS_SHAPE or pixels.dtype != torch.uint8:
        raise ValueError("resized reference pixel tensor ABI mismatch")
    return pixels


def _build_production_layout(builder: LayoutBuilder, tags: torch.Tensor, visual_latent: torch.Tensor) -> PackedLayout:
    values = builder(
        tags,
        [ReferenceSpec("image")],
        [visual_latent],
        [],
        TARGET_VIDEO_SHAPE[2],
        TARGET_VIDEO_SHAPE[3],
        TARGET_VIDEO_SHAPE[4],
        TARGET_AUDIO_SHAPE[2],
        PATCH_SIZE,
        AUDIO_CHANNELS,
        AUDIO_TAG,
        VIDEO_TAG,
    )
    layout = PackedLayout(*values)
    _validate_production_layout(layout, tags)
    return layout


def _validate_production_layout(layout: PackedLayout, text_token_tags: torch.Tensor) -> None:
    image_start = TEXT_ROWS
    audio_start = image_start + IMAGE_ROWS
    video_start = audio_start + TARGET_AUDIO_ROWS
    expected_video_indices = torch.cat((torch.arange(image_start, audio_start), torch.arange(video_start, USED_ROWS)))
    expected_audio_indices = torch.arange(audio_start, video_start)
    expected_tags = torch.cat(
        (
            text_token_tags,
            torch.full((IMAGE_ROWS,), VIDEO_TAG),
            torch.full((TARGET_AUDIO_ROWS,), AUDIO_TAG),
            torch.full((TARGET_VIDEO_ROWS,), VIDEO_TAG),
        )
    )
    checks = (
        (layout.sequence_length == USED_ROWS, "used row count"),
        (tuple(layout.position_ids.shape) == (USED_ROWS, 3), "position IDs shape"),
        (layout.position_ids.dtype == torch.float64, "position IDs dtype"),
        (torch.equal(layout.text_indices, torch.arange(TEXT_ROWS)), "text row order"),
        (torch.equal(layout.video_indices, expected_video_indices), "video row order"),
        (torch.equal(layout.audio_indices, expected_audio_indices), "audio row order"),
        (torch.equal(layout.token_tags, expected_tags), "packed modality tags"),
        (layout.num_condition_video_rows == IMAGE_ROWS, "image anchor row count"),
        (layout.num_condition_audio_rows == 0, "audio anchor row count"),
    )
    failed = [name for passed, name in checks if not passed]
    if failed:
        raise ValueError(f"pinned Diffusers production layout mismatch: {', '.join(failed)}")


def _build_row_plans(
    builder: RowTimestepBuilder,
    layout: PackedLayout,
    video_timesteps: torch.Tensor,
    audio_timesteps: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if video_timesteps.numel() != TRANSFORMER_FORWARDS or audio_timesteps.numel() != TRANSFORMER_FORWARDS:
        raise ValueError("production schedules must each drive exactly 49 forwards")
    plans = []
    for video_timestep, audio_timestep in zip(video_timesteps, audio_timesteps, strict=True):
        unique, inverse = builder(
            layout.video_indices,
            layout.audio_indices,
            layout.num_condition_video_rows,
            layout.num_condition_audio_rows,
            layout.text_indices.numel(),
            float(video_timestep),
            float(audio_timestep),
            max(float(video_timestep), VISUAL_ANCHOR_TIMESTEP),
            1.0,
        )
        if unique.dtype != torch.float32 or inverse.dtype != torch.int64 or tuple(inverse.shape) != (USED_ROWS,):
            raise ValueError("pinned Diffusers row-timestep plan ABI mismatch")
        if inverse.numel() and int(inverse.max().item()) >= unique.numel():
            raise ValueError("pinned Diffusers row-timestep plan contains an invalid index")
        plans.append((unique.contiguous(), inverse.contiguous()))
    return plans


def _layout_archive(layout: PackedLayout) -> dict[str, torch.Tensor]:
    video_mask = torch.zeros(VIDEO_COMPACT_ROWS, dtype=torch.bool)
    video_mask[IMAGE_ROWS:] = True
    audio_mask = torch.ones(AUDIO_COMPACT_ROWS, dtype=torch.bool)
    aligned_valid_mask = torch.zeros(ALIGNED_ROWS, dtype=torch.bool)
    aligned_valid_mask[:USED_ROWS] = True
    return {
        "layout.position_ids": layout.position_ids.contiguous(),
        "layout.token_tags": layout.token_tags.contiguous(),
        "layout.video_indices": layout.video_indices.contiguous(),
        "layout.audio_indices": layout.audio_indices.contiguous(),
        "layout.text_indices": layout.text_indices.contiguous(),
        "layout.video_update_mask": video_mask,
        "layout.audio_update_mask": audio_mask,
        "layout.aligned_valid_mask": aligned_valid_mask,
        "layout.used_rows": torch.tensor(USED_ROWS, dtype=torch.int64),
        "layout.aligned_rows": torch.tensor(ALIGNED_ROWS, dtype=torch.int64),
        "layout.num_condition_video_rows": torch.tensor(IMAGE_ROWS, dtype=torch.int64),
        "layout.num_condition_audio_rows": torch.tensor(0, dtype=torch.int64),
        "layout.ranges": torch.tensor(
            [
                [0, TEXT_ROWS],
                [TEXT_ROWS, TEXT_ROWS + IMAGE_ROWS],
                [TEXT_ROWS + IMAGE_ROWS, TEXT_ROWS + IMAGE_ROWS + TARGET_AUDIO_ROWS],
                [TEXT_ROWS + IMAGE_ROWS + TARGET_AUDIO_ROWS, USED_ROWS],
            ],
            dtype=torch.int64,
        ),
    }


def _schedule_archive(
    video_scheduler: Any,
    audio_scheduler: Any,
    row_plans: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    archive = {
        "schedule.video_sigmas": video_scheduler.sigmas.detach().cpu().contiguous(),
        "schedule.audio_sigmas": audio_scheduler.sigmas.detach().cpu().contiguous(),
        "schedule.video_timesteps": video_scheduler.timesteps.detach().cpu().contiguous(),
        "schedule.audio_timesteps": audio_scheduler.timesteps.detach().cpu().contiguous(),
    }
    for forward_number, (unique, inverse) in enumerate(row_plans, 1):
        prefix = f"schedule.forward_{forward_number:03d}"
        archive[f"{prefix}.unique_timesteps"] = unique.detach().cpu().contiguous()
        archive[f"{prefix}.timestep_indices"] = inverse.detach().cpu().contiguous()
    return archive


def _validate_prepared_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != PREPARED_SCHEMA or manifest.get("status") != "C7_PRODUCTION_PREPARED":
        raise ValueError("prepared manifest schema or status mismatch")
    if manifest.get("gate_evaluation") != {"gate": "G8", "status": "NOT_EVALUATED"}:
        raise ValueError("prepared artifact must explicitly record that G8 was not evaluated")
    source = manifest.get("source")
    checkpoint = manifest.get("checkpoint")
    inputs = manifest.get("inputs")
    for value, name in ((source, "source"), (checkpoint, "checkpoint"), (inputs, "inputs")):
        _validate_section_digest(value, name)
    if source.get("revision") != PINNED_DIFFUSERS_REVISION:
        raise ValueError("prepared Diffusers revision mismatch")
    source_files = source.get("files")
    if (
        not isinstance(source_files, dict)
        or {name: entry.get("sha256") for name, entry in source_files.items() if isinstance(entry, dict)}
        != PRODUCTION_DIFFUSERS_SOURCE_SHA256
    ):
        raise ValueError("prepared pinned Diffusers source file set or digest mismatch")
    modular_model = checkpoint.get("modular_model_index")
    if (
        not isinstance(modular_model, dict)
        or modular_model.get("sha256") != ROOT_CHECKPOINT_SHA256["modular_model_index.json"]
    ):
        raise ValueError("prepared checkpoint root digest mismatch")
    for component_name, expected_files in PRODUCTION_CHECKPOINT_SHA256.items():
        component = checkpoint.get(component_name)
        files = component.get("files") if isinstance(component, dict) else None
        actual_files = (
            {name: entry.get("sha256") for name, entry in files.items() if isinstance(entry, dict)}
            if isinstance(files, dict)
            else None
        )
        if actual_files != expected_files:
            raise ValueError(f"prepared checkpoint component {component_name!r} file digest mismatch")
    official = inputs.get("official_condition", {})
    native = inputs.get("native_condition", {})
    common_tags = inputs.get("common_tags", {})
    if (
        official.get("sha256") != OFFICIAL_CONDITION_SHA256
        or official.get("manifest_sha256") != OFFICIAL_CONDITION_MANIFEST_SHA256
        or official.get("cache_key") != OFFICIAL_CONDITION_CACHE_KEY
        or official.get("backend") != "official_hf"
        or official.get("tensor", {}).get("sha256") != OFFICIAL_HIDDEN_SHA256
        or official.get("tensor", {}).get("shape") != list(CONDITION_SHAPE)
        or official.get("tensor", {}).get("dtype") != "bfloat16"
    ):
        raise ValueError("prepared official condition provenance mismatch")
    if (
        native.get("sha256") != NATIVE_CONDITION_SHA256
        or native.get("summary_sha256") != NATIVE_SUMMARY_SHA256
        or native.get("summary_status") != "H3_NATIVE_DOWNSTREAM_REPORT"
        or native.get("backend") != "xllm_native"
        or native.get("tensor", {}).get("sha256") != NATIVE_HIDDEN_SHA256
        or native.get("tensor", {}).get("shape") != list(CONDITION_SHAPE)
        or native.get("tensor", {}).get("dtype") != "bfloat16"
    ):
        raise ValueError("prepared native condition provenance mismatch")
    if (
        common_tags.get("sha256") != CONDITION_TAGS_SHA256
        or common_tags.get("shape") != list(CONDITION_TAGS_SHAPE)
        or common_tags.get("dtype") != "int64"
    ):
        raise ValueError("prepared common condition tags provenance mismatch")
    image = manifest.get("image", {})
    if image.get("sha256") != REFERENCE_IMAGE_SHA256 or image.get("source_size_wh") != list(SOURCE_IMAGE_SIZE):
        raise ValueError("prepared reference image provenance mismatch")
    if manifest.get("geometry") != _geometry_manifest():
        raise ValueError("prepared production geometry mismatch")
    generators = manifest.get("generators", {})
    if generators.get("posterior") != {"device": "cpu", "seed": POSTERIOR_SEED, "scope": "fresh"}:
        raise ValueError("prepared posterior generator contract mismatch")
    request = generators.get("request", {})
    if request.get("device") != "cpu" or request.get("seed") != REQUEST_SEED:
        raise ValueError("prepared request generator seed or device mismatch")
    if request.get("draw_order") != list(REQUEST_DRAW_ORDER):
        raise ValueError("prepared request generator draw order mismatch")
    schedule = manifest.get("schedule", {})
    if schedule != {
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "sigma_points": SIGMA_POINTS,
        "transformer_forwards": TRANSFORMER_FORWARDS,
        "visual_anchor_timestep": VISUAL_ANCHOR_TIMESTEP,
    }:
        raise ValueError("prepared scheduler contract mismatch")
    backends = manifest.get("condition_backends")
    expected_backends = {
        name: {
            "tensor_key": key,
            "hidden_sha256": OFFICIAL_HIDDEN_SHA256 if name == "official_hf" else NATIVE_HIDDEN_SHA256,
        }
        for name, key in PREPARED_CONDITION_KEYS.items()
    }
    if backends != expected_backends:
        raise ValueError("prepared condition backend map mismatch")
    declarations = manifest.get("tensors")
    if not isinstance(declarations, dict) or set(declarations) != set(PREPARED_TENSOR_SPECS):
        raise ValueError("prepared manifest tensor declarations do not match the production contract")
    for name, spec in PREPARED_TENSOR_SPECS.items():
        declaration = declarations[name]
        if declaration.get("dtype") != str(spec.dtype).removeprefix("torch."):
            raise ValueError(f"prepared manifest tensor {name!r} dtype mismatch")
        shape = declaration.get("shape")
        if spec.shape is not None and shape != list(spec.shape):
            raise ValueError(f"prepared manifest tensor {name!r} shape mismatch")
        if spec.shape is None and (not isinstance(shape, list) or len(shape) != 1 or not 1 <= shape[0] <= 4):
            raise ValueError(f"prepared manifest tensor {name!r} unique timestep shape mismatch")
    condition_declarations = {
        "condition.official_hf.hidden": OFFICIAL_HIDDEN_SHA256,
        "condition.xllm_native.hidden": NATIVE_HIDDEN_SHA256,
        "condition.text_token_tags": CONDITION_TAGS_SHA256,
    }
    for name, expected_sha256 in condition_declarations.items():
        if declarations[name].get("sha256") != expected_sha256:
            raise ValueError(f"prepared manifest tensor {name!r} digest mismatch")


def _validate_artifact_pair(
    archive_path: Path,
    manifest: Mapping[str, Any],
    *,
    retain_names: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("path") != archive_path.name:
        raise ValueError("artifact filename does not match its manifest")
    if artifact.get("size") != archive_path.stat().st_size or artifact.get("sha256") != _sha256(archive_path):
        raise ValueError("artifact size or SHA256 does not match its manifest")
    expected_tensors = manifest.get("tensors")
    if not isinstance(expected_tensors, dict):
        raise ValueError("artifact manifest does not contain tensor metadata")
    retained: dict[str, torch.Tensor] = {}
    with safe_open(str(archive_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected_tensors):
            raise ValueError("artifact tensor keys do not match its manifest")
        for name in sorted(handle.keys()):
            value = handle.get_tensor(name).contiguous()
            if _tensor_summary(value) != expected_tensors[name]:
                raise ValueError(f"artifact tensor {name!r} metadata or SHA256 mismatch")
            if retain_names is not None and name in retain_names:
                retained[name] = value.clone()
    return retained


def _condition_tensor_key(backend: str, manifest: Mapping[str, Any] | None = None) -> str:
    if backend not in PREPARED_CONDITION_KEYS:
        raise ValueError(f"unsupported condition backend {backend!r}; no fallback is permitted")
    key = PREPARED_CONDITION_KEYS[backend]
    if manifest is not None:
        backend_manifest = manifest.get("condition_backends", {}).get(backend)
        if backend_manifest != {
            "tensor_key": key,
            "hidden_sha256": OFFICIAL_HIDDEN_SHA256 if backend == "official_hf" else NATIVE_HIDDEN_SHA256,
        }:
            raise ValueError(f"prepared artifact backend provenance mismatch for {backend}")
    return key


def _write_artifact_pair(
    output_dir: Path,
    archive_name: str,
    manifest_name: str,
    tensors: Mapping[str, torch.Tensor],
    manifest: dict[str, Any],
) -> Path:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    archive_temp = output_dir / f".{archive_name}{suffix}"
    manifest_temp = output_dir / f".{manifest_name}{suffix}"
    try:
        persisted = {name: value.detach().to("cpu").contiguous().clone() for name, value in sorted(tensors.items())}
        save_file(persisted, str(archive_temp))
        archive_fd = os.open(archive_temp, os.O_RDONLY)
        try:
            os.fsync(archive_fd)
        finally:
            os.close(archive_fd)
        manifest["artifact"] = {
            "path": archive_name,
            "size": archive_temp.stat().st_size,
            "sha256": _sha256(archive_temp),
        }
        with manifest_temp.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(archive_temp, output_dir / archive_name)
        os.replace(manifest_temp, output_dir / manifest_name)
        directory_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        archive_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)
    return output_dir / manifest_name


def _validate_npu_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "npu" or device.index is None or str(device) != value:
        raise ValueError(f"C7 requires an explicitly indexed NPU device, got {value!r}")
    return device


def _runtime_manifest(torch_npu: Any, diffusers: Any, devices: Sequence[torch.device]) -> dict[str, Any]:
    return _with_digest(
        {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "diffusers": diffusers.__version__,
            "diffusers_path": str(Path(diffusers.__file__).resolve()),
            "devices": [{"device": str(device), "name": torch.npu.get_device_name(device)} for device in devices],
            "network": "disabled by local_files_only and pinned local source/checkpoints",
        }
    )


def _prepare(args: argparse.Namespace) -> Path:
    diffusers_source = args.diffusers_source.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    output_dir = args.output_dir.resolve()
    vae_device = _validate_npu_device(args.vae_device)

    source_started = time.perf_counter()
    source = _source_manifest(diffusers_source)
    source_seconds = time.perf_counter() - source_started
    _activate_diffusers_source(diffusers_source)

    import diffusers
    import torch_npu
    from diffusers import AutoencoderKLMiniMaxH3, MiniMaxH3Scheduler
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3Ref2VAPrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from diffusers.utils.torch_utils import randn_tensor

    expected_source = (diffusers_source / "src").resolve()
    if expected_source not in Path(diffusers.__file__).resolve().parents:
        raise RuntimeError("Diffusers imported outside the pinned source tree")
    if not torch.npu.is_available() or vae_device.index >= torch.npu.device_count():
        raise RuntimeError(f"requested preparation device is unavailable: {vae_device}")
    torch.npu.set_device(vae_device)
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats(vae_device)

    checkpoint_started = time.perf_counter()
    checkpoint = _checkpoint_manifest(checkpoint_root)
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    condition_started = time.perf_counter()
    conditions, condition_attestation = _load_and_attest_conditions(
        args.official_condition, args.native_condition, args.native_summary
    )
    condition_seconds = time.perf_counter() - condition_started
    randn_equivalence = _prove_pinned_randn_equivalence(randn_tensor)

    image_processor = VaeImageProcessor(vae_scale_factor=16)
    if image_processor.config.resample != "lanczos":
        raise ValueError("pinned VaeImageProcessor is not configured for Lanczos resizing")
    pixels = _prepare_reference_pixels(args.reference_image, image_processor)

    logger.info(f"Loading pinned production Video VAE on {vae_device}")
    vae_load_started = time.perf_counter()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        checkpoint_root / "vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    vae.eval()
    vae.requires_grad_(False)
    _assert_fp32_model(vae, "video VAE")
    vae.enable_tiling(
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_sample_min_overlap_height=64,
        tile_sample_min_overlap_width=64,
    )
    vae.to(vae_device)
    torch.npu.synchronize(vae_device)
    vae_load_seconds = time.perf_counter() - vae_load_started
    encode_started = time.perf_counter()
    with torch.inference_mode():
        visual_latent = encode_vae_condition(
            vae,
            pixels.to(vae_device),
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            POSTERIOR_SEED,
        ).contiguous()
    torch.npu.synchronize(vae_device)
    encode_seconds = time.perf_counter() - encode_started
    if tuple(visual_latent.shape) != VISUAL_LATENT_SHAPE or visual_latent.dtype != torch.float32:
        raise ValueError(
            f"official Video VAE reference latent is {visual_latent.dtype} {tuple(visual_latent.shape)}, "
            f"expected float32 {VISUAL_LATENT_SHAPE}"
        )
    vae_memory = _memory_stats((vae_device,))[str(vae_device)]
    del vae
    _clear_npu((vae_device,))

    request_noise = _draw_request_noise(randn_tensor=randn_tensor, device=vae_device)
    video_scheduler = MiniMaxH3Scheduler.from_pretrained(checkpoint_root / "scheduler", local_files_only=True)
    audio_scheduler = MiniMaxH3Scheduler.from_pretrained(checkpoint_root / "audio_scheduler", local_files_only=True)
    video_scheduler.set_timesteps(SIGMA_POINTS)
    audio_scheduler.set_timesteps(SIGMA_POINTS)
    visual_anchor = video_scheduler.scale_noise(
        visual_latent.to(vae_device),
        VISUAL_ANCHOR_TIMESTEP,
        request_noise.visual_anchor_noise,
    )
    visual_anchor = visual_anchor.detach().cpu().contiguous()
    request_noise = RequestNoise(
        request_noise.visual_anchor_noise.detach().cpu().contiguous(),
        request_noise.target_video_initial.detach().cpu().contiguous(),
        request_noise.target_audio_rows.detach().cpu().contiguous(),
    )
    _clear_npu((vae_device,))

    layout = _build_production_layout(
        MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence,
        conditions["text_token_tags"],
        visual_latent,
    )
    row_plans = _build_row_plans(
        MiniMaxH3SetTimestepsStep.build_row_timesteps,
        layout,
        video_scheduler.timesteps,
        audio_scheduler.timesteps,
    )
    video_rows = torch.cat(
        (
            _pack_video_latents(visual_anchor, PATCH_SIZE),
            _pack_video_latents(request_noise.target_video_initial, PATCH_SIZE),
        )
    ).contiguous()
    if not torch.equal(
        video_rows[:IMAGE_ROWS],
        patchify_video_latents(visual_anchor, PATCH_SIZE),
    ):
        raise ValueError("local visual patch order does not match pinned Diffusers")
    target_audio_latent = _unpack_audio_rows(request_noise.target_audio_rows, TARGET_AUDIO_SHAPE[2])

    archive: dict[str, torch.Tensor] = {
        "condition.official_hf.hidden": conditions["official_hf"],
        "condition.xllm_native.hidden": conditions["xllm_native"],
        "condition.text_token_tags": conditions["text_token_tags"],
        "input.reference_resized_uint8": pixels,
        "input.visual_latent_normalized": visual_latent,
        "input.visual_anchor_noise": request_noise.visual_anchor_noise,
        "input.visual_anchor_noised": visual_anchor,
        "input.target_video_initial": request_noise.target_video_initial,
        "input.target_audio_initial_rows": request_noise.target_audio_rows,
        "input.target_audio_initial_latent": target_audio_latent,
        "initial.video_rows": video_rows,
        "initial.audio_rows": request_noise.target_audio_rows,
    }
    archive.update(_layout_archive(layout))
    archive.update(_schedule_archive(video_scheduler, audio_scheduler, row_plans))
    _validate_tensor_contract(archive, PREPARED_TENSOR_SPECS)

    inputs = dict(condition_attestation)
    inputs.pop("digest")
    inputs = _with_digest(inputs)
    image = {
        "path": str(args.reference_image.resolve()),
        "sha256": REFERENCE_IMAGE_SHA256,
        "mode": "RGB",
        "source_size_wh": list(SOURCE_IMAGE_SIZE),
        "resized_size_wh": list(RESIZED_IMAGE_SIZE),
        "resize": "pinned VaeImageProcessor PIL Lanczos",
    }
    manifest = {
        "schema": PREPARED_SCHEMA,
        "status": "C7_PRODUCTION_PREPARED",
        "gate_evaluation": {"gate": "G8", "status": "NOT_EVALUATED"},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "checkpoint": checkpoint,
        "inputs": inputs,
        "image": image,
        "geometry": _geometry_manifest(),
        "condition_backends": {
            name: {
                "tensor_key": key,
                "hidden_sha256": OFFICIAL_HIDDEN_SHA256 if name == "official_hf" else NATIVE_HIDDEN_SHA256,
            }
            for name, key in PREPARED_CONDITION_KEYS.items()
        },
        "generators": {
            "posterior": {"device": "cpu", "seed": POSTERIOR_SEED, "scope": "fresh"},
            "request": {
                "device": "cpu",
                "seed": REQUEST_SEED,
                "draw_order": list(REQUEST_DRAW_ORDER),
                "randn_semantics": "pinned diffusers randn_tensor; CPU draw then transfer to NPU",
            },
            "randn_equivalence": randn_equivalence,
        },
        "schedule": {
            "video_shift": float(video_scheduler.config.shift),
            "audio_shift": float(audio_scheduler.config.shift),
            "sigma_points": SIGMA_POINTS,
            "transformer_forwards": TRANSFORMER_FORWARDS,
            "visual_anchor_timestep": VISUAL_ANCHOR_TIMESTEP,
        },
        "contracts": {
            "preparation_scope": "condition/image attestation, one real tiled Video VAE encode, request RNG, layout, schedules; no transformer",
            "posterior": "fresh CPU generator seed 42; sample -> FP16 -> FP32; per-channel normalization in pinned encode_vae_condition",
            "request_rng": "one CPU generator seed 42 consumed visual anchor, target video, target audio rows in order",
            "packing": "used rows only [text | image | target audio | target video]; external alignment recorded separately",
            "backend_isolation": "both hidden tensors and one byte-identical tag tensor; every non-condition input is common",
            "atomic_output": "fsynced safetensors rename precedes fsynced manifest rename and directory fsync",
        },
        "execution": {
            "device": str(vae_device),
            "video_vae": {
                "weights_dtype": "float32",
                "tiling": {"tile_sample": [256, 256], "minimum_overlap": [64, 64]},
                "memory": vae_memory,
            },
            "timings_seconds": {
                "source_attestation": source_seconds,
                "checkpoint_attestation": checkpoint_seconds,
                "condition_attestation": condition_seconds,
                "video_vae_load": vae_load_seconds,
                "video_vae_encode": encode_seconds,
            },
            "runtime": _runtime_manifest(torch_npu, diffusers, (vae_device,)),
        },
        "tensors": {name: _tensor_summary(tensor) for name, tensor in sorted(archive.items())},
    }
    _validate_prepared_manifest(manifest)
    return _write_artifact_pair(
        output_dir,
        PREPARED_ARCHIVE_NAME,
        PREPARED_MANIFEST_NAME,
        archive,
        manifest,
    )


def _verify_prepared_against_runtime(
    manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    if manifest["source"] != source:
        raise ValueError("current pinned Diffusers source does not match the prepared artifact")
    if manifest["checkpoint"] != checkpoint:
        raise ValueError("current converted checkpoint does not match the prepared artifact")


def _verify_scheduler_archive(tensors: Mapping[str, torch.Tensor], video_scheduler: Any, audio_scheduler: Any) -> None:
    pairs = {
        "schedule.video_sigmas": video_scheduler.sigmas,
        "schedule.audio_sigmas": audio_scheduler.sigmas,
        "schedule.video_timesteps": video_scheduler.timesteps,
        "schedule.audio_timesteps": audio_scheduler.timesteps,
    }
    for name, runtime_value in pairs.items():
        if not torch.equal(tensors[name], runtime_value.detach().cpu()):
            raise ValueError(f"runtime scheduler does not match prepared tensor {name!r}")


def _runtime_layout(tensors: Mapping[str, torch.Tensor]) -> PackedLayout:
    layout = PackedLayout(
        tensors["layout.position_ids"],
        tensors["layout.token_tags"],
        tensors["layout.video_indices"],
        tensors["layout.audio_indices"],
        tensors["layout.text_indices"],
        int(tensors["layout.num_condition_video_rows"].item()),
        int(tensors["layout.num_condition_audio_rows"].item()),
    )
    _validate_production_layout(layout, tensors["condition.text_token_tags"])
    expected_video_mask = torch.zeros(VIDEO_COMPACT_ROWS, dtype=torch.bool)
    expected_video_mask[IMAGE_ROWS:] = True
    expected_aligned_mask = torch.zeros(ALIGNED_ROWS, dtype=torch.bool)
    expected_aligned_mask[:USED_ROWS] = True
    expected_ranges = torch.tensor(
        [
            [0, TEXT_ROWS],
            [TEXT_ROWS, TEXT_ROWS + IMAGE_ROWS],
            [TEXT_ROWS + IMAGE_ROWS, TEXT_ROWS + IMAGE_ROWS + TARGET_AUDIO_ROWS],
            [TEXT_ROWS + IMAGE_ROWS + TARGET_AUDIO_ROWS, USED_ROWS],
        ],
        dtype=torch.int64,
    )
    scalar_values = {
        "layout.used_rows": USED_ROWS,
        "layout.aligned_rows": ALIGNED_ROWS,
        "layout.num_condition_video_rows": IMAGE_ROWS,
        "layout.num_condition_audio_rows": 0,
    }
    if any(int(tensors[name].item()) != expected for name, expected in scalar_values.items()):
        raise ValueError("prepared layout scalar values do not match the production contract")
    if not torch.equal(tensors["layout.video_update_mask"], expected_video_mask):
        raise ValueError("prepared video update mask does not preserve exactly the image rows")
    if not bool(tensors["layout.audio_update_mask"].all().item()):
        raise ValueError("prepared audio update mask must select every target audio row")
    if not torch.equal(tensors["layout.aligned_valid_mask"], expected_aligned_mask):
        raise ValueError("prepared aligned valid-row mask mismatch")
    if not torch.equal(tensors["layout.ranges"], expected_ranges):
        raise ValueError("prepared packed ranges mismatch")
    return layout


def _verify_pinned_layout(
    tensors: Mapping[str, torch.Tensor],
    builder: LayoutBuilder,
) -> PackedLayout:
    runtime = _runtime_layout(tensors)
    expected = _build_production_layout(
        builder,
        tensors["condition.text_token_tags"],
        torch.empty(VISUAL_LATENT_SHAPE, dtype=torch.float32),
    )
    for name in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"):
        if not torch.equal(getattr(runtime, name), getattr(expected, name)):
            raise ValueError(f"prepared layout {name} does not byte-match pinned Diffusers")
    return runtime


def _verify_pinned_row_plans(
    tensors: Mapping[str, torch.Tensor],
    layout: PackedLayout,
    builder: RowTimestepBuilder,
) -> None:
    expected = _build_row_plans(
        builder,
        layout,
        tensors["schedule.video_timesteps"],
        tensors["schedule.audio_timesteps"],
    )
    for forward_number, (unique, inverse) in enumerate(expected, 1):
        prefix = f"schedule.forward_{forward_number:03d}"
        if not torch.equal(tensors[f"{prefix}.unique_timesteps"], unique) or not torch.equal(
            tensors[f"{prefix}.timestep_indices"], inverse
        ):
            raise ValueError(f"prepared row-timestep plan {forward_number} does not byte-match pinned Diffusers")


def _eager_transformer_forward(
    model: Any,
    layout: PackedLayout,
    condition_hidden: torch.Tensor,
    video_rows: torch.Tensor,
    audio_rows: torch.Tensor,
    unique_timesteps: torch.Tensor,
    timestep_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    first = next(model.context_embedder.parameters()).device
    last = next(model.norm_out.parameters()).device
    video_indices_first = layout.video_indices.to(first)
    audio_indices_first = layout.audio_indices.to(first)
    text_indices_first = layout.text_indices.to(first)

    video_embeds = model.proj_in(video_rows.to(first, _parameter_dtype(model.proj_in)).unsqueeze(0))
    audio_embeds = model.audio_proj_in(audio_rows.to(first, _parameter_dtype(model.audio_proj_in)).unsqueeze(0))
    text_embeds = model.context_embedder(condition_hidden.unsqueeze(0).to(_parameter_dtype(model.context_embedder)))
    text_embeds = model.token_refiner(text_embeds)
    hidden = text_embeds.new_zeros((1, layout.sequence_length, text_embeds.shape[-1]))
    hidden = hidden.index_copy(1, text_indices_first, text_embeds)
    hidden = hidden.index_copy(1, video_indices_first, video_embeds.to(text_embeds.dtype))
    hidden = hidden.index_copy(1, audio_indices_first, audio_embeds.to(text_embeds.dtype))

    time_frequency = model.time_proj(unique_timesteps.to(first))
    time_embedding = model.time_embedder(time_frequency.to(_parameter_dtype(model.time_embedder)))
    rotary = model.rope(layout.position_ids.to(first))
    adaln_indices = timestep_indices.to(first) * 3 + layout.token_tags.to(first)
    for block in model.transformer_blocks:
        device = next(block.parameters()).device
        hidden = block(
            hidden.to(device),
            time_embedding.to(device),
            adaln_indices.to(device),
            tuple(value.to(device) for value in rotary),
        )

    hidden = hidden.to(last)
    activation = model.norm_out(hidden, time_embedding.to(last), timestep_indices.to(last)).to(
        _parameter_dtype(model.proj_out)
    )
    video_velocity = model.proj_out(activation).index_select(1, layout.video_indices.to(last)).squeeze(0).to(first)
    audio_velocity = (
        model.audio_proj_out(activation).index_select(1, layout.audio_indices.to(last)).squeeze(0).to(first)
    )
    return video_velocity, audio_velocity


def _replace_generated_rows(rows: torch.Tensor, generated: torch.Tensor, condition_rows: int) -> torch.Tensor:
    if not 0 <= condition_rows < rows.shape[0]:
        raise ValueError(f"invalid compact condition row count {condition_rows} for {rows.shape[0]} rows")
    if tuple(generated.shape) != tuple(rows[condition_rows:].shape) or generated.dtype != rows.dtype:
        raise ValueError("generated replacement rows have an unexpected shape or dtype")
    if condition_rows == 0:
        return generated.contiguous()
    output = torch.cat((rows[:condition_rows], generated), dim=0).contiguous()
    if not torch.equal(output[:condition_rows], rows[:condition_rows]):
        raise ValueError("conditioning anchor changed while replacing generated rows")
    return output


def _validate_execution_archive(archive: Mapping[str, torch.Tensor], forwards: int) -> None:
    expected: dict[str, TensorSpec] = {}
    for forward_number in SNAPSHOT_FORWARDS:
        if forward_number <= forwards:
            expected[f"trajectory.forward_{forward_number:03d}.video_rows"] = TensorSpec(
                (VIDEO_COMPACT_ROWS, VIDEO_PATCH_WIDTH), torch.float32
            )
            expected[f"trajectory.forward_{forward_number:03d}.audio_rows"] = TensorSpec(
                TARGET_AUDIO_ROWS_SHAPE, torch.float32
            )
    if forwards == TRANSFORMER_FORWARDS:
        expected.update(
            {
                "final.target_video_rows": TensorSpec((TARGET_VIDEO_ROWS, VIDEO_PATCH_WIDTH), torch.float32),
                "final.target_audio_rows": TensorSpec(TARGET_AUDIO_ROWS_SHAPE, torch.float32),
                "final.target_video_latent": TensorSpec(TARGET_VIDEO_SHAPE, torch.float32),
                "final.target_audio_latent": TensorSpec(TARGET_AUDIO_SHAPE, torch.float32),
                "decoded.video_uint8": TensorSpec(DECODED_VIDEO_SHAPE, torch.uint8),
                "decoded.audio_float32": TensorSpec(DECODED_AUDIO_SHAPE, torch.float32),
                "decoded.audio_sample_rate": TensorSpec((), torch.int64),
            }
        )
    _validate_tensor_contract(archive, expected)
    if forwards == TRANSFORMER_FORWARDS and int(archive["decoded.audio_sample_rate"].item()) != AUDIO_SAMPLE_RATE:
        raise ValueError("decoded audio sample rate tensor mismatch")


def _assert_fp32_model(model: Any, name: str) -> None:
    bad_parameters = [
        parameter_name for parameter_name, value in model.named_parameters() if value.dtype != torch.float32
    ]
    bad_buffers = [
        buffer_name
        for buffer_name, value in model.named_buffers()
        if value.is_floating_point() and value.dtype != torch.float32
    ]
    if bad_parameters or bad_buffers:
        raise ValueError(f"{name} is not wholly FP32; parameters={bad_parameters[:5]}, buffers={bad_buffers[:5]}")


def _run_transformer(
    model: Any,
    layout: PackedLayout,
    tensors: Mapping[str, torch.Tensor],
    condition_hidden: torch.Tensor,
    video_scheduler: Any,
    audio_scheduler: Any,
    devices: tuple[torch.device, torch.device],
    forwards: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, list[float]]:
    first = devices[0]
    video_rows = tensors["initial.video_rows"].to(first)
    audio_rows = tensors["initial.audio_rows"].to(first)
    condition_hidden = condition_hidden.to(first)
    video_anchor = video_rows[:IMAGE_ROWS].clone()
    archive: dict[str, torch.Tensor] = {}
    forward_seconds = []

    for index in range(forwards):
        forward_number = index + 1
        prefix = f"schedule.forward_{forward_number:03d}"
        started = time.perf_counter()
        video_velocity, audio_velocity = _eager_transformer_forward(
            model,
            layout,
            condition_hidden,
            video_rows,
            audio_rows,
            tensors[f"{prefix}.unique_timesteps"].to(first),
            tensors[f"{prefix}.timestep_indices"].to(first),
        )
        next_video = video_scheduler.step(
            video_velocity[IMAGE_ROWS:].float(),
            video_scheduler.timesteps[index],
            video_rows[IMAGE_ROWS:],
            return_dict=False,
        )[0]
        next_audio = audio_scheduler.step(
            audio_velocity.float(),
            audio_scheduler.timesteps[index],
            audio_rows,
            return_dict=False,
        )[0]
        video_rows = _replace_generated_rows(video_rows, next_video, IMAGE_ROWS)
        audio_rows = _replace_generated_rows(audio_rows, next_audio, 0)
        if not torch.equal(video_rows[:IMAGE_ROWS], video_anchor):
            raise ValueError(f"visual conditioning anchor changed after forward {forward_number}")
        for device in devices:
            torch.npu.synchronize(device)
        forward_seconds.append(time.perf_counter() - started)
        if forward_number in SNAPSHOT_FORWARDS:
            archive[f"trajectory.forward_{forward_number:03d}.video_rows"] = video_rows.detach().cpu().contiguous()
            archive[f"trajectory.forward_{forward_number:03d}.audio_rows"] = audio_rows.detach().cpu().contiguous()
        logger.info(f"Completed production transformer forward {forward_number}/{forwards}")
    return archive, video_rows.detach().cpu().contiguous(), audio_rows.detach().cpu().contiguous(), forward_seconds


def _decode_video(
    model_class: type[Any],
    checkpoint_path: Path,
    normalized_latent: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    torch.npu.set_device(device)
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model = model_class.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    _assert_fp32_model(model, "video VAE")
    model.enable_tiling(
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_sample_min_overlap_height=64,
        tile_sample_min_overlap_width=64,
    )
    model.to(device)
    torch.npu.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    denormalized = _denormalize_latents(
        normalized_latent.to(device), tuple(model.config.latents_mean), tuple(model.config.latents_std)
    )
    decode_started = time.perf_counter()
    with torch.autocast(device_type="npu", dtype=torch.float16):
        decoded = model.decode(denormalized, return_dict=False)[0]
    pixel_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=device).view(1, 3, 1, 1, 1)
    pixel_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=device).view(1, 3, 1, 1, 1)
    video_uint8 = (decoded.float() * pixel_std + pixel_mean).clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
    torch.npu.synchronize(device)
    decode_seconds = time.perf_counter() - decode_started
    if tuple(video_uint8.shape) != DECODED_VIDEO_SHAPE:
        raise ValueError(f"decoded video shape {tuple(video_uint8.shape)} does not match {DECODED_VIDEO_SHAPE}")
    execution = {
        "device": str(device),
        "weights_dtype": "float32",
        "compute": "NPU FP16 autocast",
        "tiling": {"tile_sample": [256, 256], "minimum_overlap": [64, 64]},
        "load_seconds": load_seconds,
        "decode_seconds": decode_seconds,
        "memory": _memory_stats((device,))[str(device)],
    }
    del model, denormalized, decoded
    _clear_npu((device,))
    return video_uint8.contiguous(), execution


def _decode_audio(
    model_class: type[Any],
    checkpoint_path: Path,
    normalized_latent: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    torch.npu.set_device(device)
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model = model_class.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    _assert_fp32_model(model, "audio VAE")
    model.pre_block.attn.set_attention_backend("_native_math")
    model.to(device)
    torch.npu.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    denormalized = _denormalize_latents(
        normalized_latent.to(device), tuple(model.config.latents_mean), tuple(model.config.latents_std)
    )
    decode_started = time.perf_counter()
    decoded = model.decode(denormalized, return_dict=False)[0]
    audio = decoded.float().permute(1, 0, 2).cpu().contiguous()
    torch.npu.synchronize(device)
    decode_seconds = time.perf_counter() - decode_started
    if tuple(audio.shape) != DECODED_AUDIO_SHAPE:
        raise ValueError(f"decoded audio shape {tuple(audio.shape)} does not match {DECODED_AUDIO_SHAPE}")
    execution = {
        "device": str(device),
        "weights_dtype": "float32",
        "compute": "native Diffusers eager FP32; no autocast",
        "attention_backend": "_native_math",
        "load_seconds": load_seconds,
        "decode_seconds": decode_seconds,
        "memory": _memory_stats((device,))[str(device)],
    }
    del model, denormalized, decoded
    _clear_npu((device,))
    return audio, execution


def _write_media(
    output_dir: Path,
    basename: str,
    video_uint8: torch.Tensor,
    audio: torch.Tensor,
) -> dict[str, Any]:
    if importlib.util.find_spec("av") is None:
        raise RuntimeError("--write-media requires PyAV; refusing to omit requested real media")
    from diffusers.utils.export_utils import encode_video

    output_dir.mkdir(parents=True, exist_ok=True)
    media_name = f"{basename}.mp4"
    media_path = output_dir / media_name
    temp_path = output_dir / f".{basename}.{os.getpid()}.{uuid.uuid4().hex}.tmp.mp4"
    try:
        frames = video_uint8[0].permute(1, 2, 3, 0).contiguous()
        encode_video(
            frames,
            fps=VIDEO_FPS,
            output_path=str(temp_path),
            audio=audio[0],
            audio_sample_rate=AUDIO_SAMPLE_RATE,
        )
        media_fd = os.open(temp_path, os.O_RDONLY)
        try:
            os.fsync(media_fd)
        finally:
            os.close(media_fd)
        metadata = {"path": media_name, "size": temp_path.stat().st_size, "sha256": _sha256(temp_path)}
        os.replace(temp_path, media_path)
        return metadata
    finally:
        temp_path.unlink(missing_ok=True)


def _execution_names(backend: str, mode: str) -> tuple[str, str, str]:
    stem = f"minimax_h3_ref2va_production_{backend}_{mode}"
    return f"{stem}.safetensors", f"{stem}.json", stem


def _execute(args: argparse.Namespace) -> Path:
    if args.write_media and args.execution_mode != "full":
        raise ValueError("--write-media is only valid for a full 49-forward execution")
    devices = tuple(_validate_npu_device(value) for value in args.devices)
    if devices[0] == devices[1]:
        raise ValueError("production transformer execution requires two distinct explicitly indexed NPUs")
    vae_device = _validate_npu_device(args.vae_device)
    if vae_device not in devices:
        raise ValueError("the sequential decoder device must be one of the two transformer devices")

    prepared_path = args.prepared_artifact.resolve()
    prepared_manifest_path = args.prepared_manifest.resolve()
    if not prepared_path.is_file() or not prepared_manifest_path.is_file():
        raise ValueError("prepared artifact and manifest must both exist before transformer execution")
    prepared_manifest = _load_json(prepared_manifest_path, "prepared manifest")
    _validate_prepared_manifest(prepared_manifest)
    condition_key = _condition_tensor_key(args.condition_backend, prepared_manifest)
    retain_names = set(COMMON_EXECUTION_KEYS)
    retain_names.add(condition_key)
    for forward_number in range(1, TRANSFORMER_FORWARDS + 1):
        prefix = f"schedule.forward_{forward_number:03d}"
        retain_names.add(f"{prefix}.unique_timesteps")
        retain_names.add(f"{prefix}.timestep_indices")
    prepared_tensors = _validate_artifact_pair(prepared_path, prepared_manifest, retain_names=retain_names)

    diffusers_source = args.diffusers_source.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    source_started = time.perf_counter()
    source = _source_manifest(diffusers_source)
    source_seconds = time.perf_counter() - source_started
    checkpoint_started = time.perf_counter()
    checkpoint = _checkpoint_manifest(checkpoint_root)
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    _verify_prepared_against_runtime(prepared_manifest, source, checkpoint)
    _activate_diffusers_source(diffusers_source)

    import diffusers
    import torch_npu
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Scheduler
    from diffusers.models.transformers import MiniMaxH3Transformer3DModel
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3Ref2VAPrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
    )

    expected_source = (diffusers_source / "src").resolve()
    if expected_source not in Path(diffusers.__file__).resolve().parents:
        raise RuntimeError("Diffusers imported outside the pinned source tree")
    if not torch.npu.is_available() or max(device.index for device in devices) >= torch.npu.device_count():
        raise RuntimeError("the requested two-NPU production placement is unavailable")
    for device in devices:
        torch.npu.set_device(device)
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats(device)

    video_scheduler = MiniMaxH3Scheduler.from_pretrained(checkpoint_root / "scheduler", local_files_only=True)
    audio_scheduler = MiniMaxH3Scheduler.from_pretrained(checkpoint_root / "audio_scheduler", local_files_only=True)
    video_scheduler.set_timesteps(SIGMA_POINTS, device=devices[0])
    audio_scheduler.set_timesteps(SIGMA_POINTS, device=devices[0])
    _verify_scheduler_archive(prepared_tensors, video_scheduler, audio_scheduler)
    layout = _verify_pinned_layout(
        prepared_tensors,
        MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence,
    )
    _verify_pinned_row_plans(
        prepared_tensors,
        layout,
        MiniMaxH3SetTimestepsStep.build_row_timesteps,
    )

    logger.info(f"Loading pinned 50-layer Ref2VA transformer for backend {args.condition_backend}")
    load_started = time.perf_counter()
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        checkpoint_root / "transformer_ref",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    attention_module_count = _set_native_attention(model)
    if attention_module_count != TRANSFORMER_LAYERS + 2:
        raise ValueError(f"expected 52 native attention modules, found {attention_module_count}")
    placement = _place_transformer(model, devices)
    for device in devices:
        torch.npu.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    forwards = 1 if args.execution_mode == "smoke" else TRANSFORMER_FORWARDS
    run_started = time.perf_counter()
    with torch.inference_mode():
        archive, final_video_rows, final_audio_rows, per_forward_seconds = _run_transformer(
            model,
            layout,
            prepared_tensors,
            prepared_tensors[condition_key],
            video_scheduler,
            audio_scheduler,
            devices,
            forwards,
        )
    transformer_seconds = time.perf_counter() - run_started
    transformer_memory = _memory_stats(devices)
    del model
    del prepared_tensors
    gc.collect()
    _clear_npu(devices)

    video_decode = None
    audio_decode = None
    media = {"requested": args.write_media, "written": False}
    if args.execution_mode == "full":
        target_video_rows = final_video_rows[IMAGE_ROWS:].contiguous()
        target_audio_rows = final_audio_rows.contiguous()
        target_video_latent = _unpack_video_rows(target_video_rows, TARGET_VIDEO_SHAPE, PATCH_SIZE)
        target_audio_latent = _unpack_audio_rows(target_audio_rows, TARGET_AUDIO_SHAPE[2])
        archive.update(
            {
                "final.target_video_rows": target_video_rows,
                "final.target_audio_rows": target_audio_rows,
                "final.target_video_latent": target_video_latent,
                "final.target_audio_latent": target_audio_latent,
            }
        )
        logger.info("Transformer released; decoding production video with the pinned Video VAE")
        with torch.inference_mode():
            decoded_video, video_decode = _decode_video(
                AutoencoderKLMiniMaxH3, checkpoint_root / "vae", target_video_latent, vae_device
            )
        archive["decoded.video_uint8"] = decoded_video
        logger.info("Video VAE released; decoding production audio in FP32")
        with torch.inference_mode():
            decoded_audio, audio_decode = _decode_audio(
                AutoencoderKLMiniMaxH3Audio, checkpoint_root / "audio_vae", target_audio_latent, vae_device
            )
        archive["decoded.audio_float32"] = decoded_audio
        archive["decoded.audio_sample_rate"] = torch.tensor(AUDIO_SAMPLE_RATE, dtype=torch.int64)
        if args.write_media:
            _, _, basename = _execution_names(args.condition_backend, args.execution_mode)
            media = {
                "requested": True,
                "written": True,
                **_write_media(args.output_dir.resolve(), basename, decoded_video, decoded_audio),
            }

    _validate_execution_archive(archive, forwards)

    archive_name, manifest_name, _ = _execution_names(args.condition_backend, args.execution_mode)
    status = "C7_ONE_FORWARD_ATTENTION_HBM_SMOKE_COMPLETE" if forwards == 1 else "C7_49_FORWARD_REFERENCE_COMPLETE"
    manifest = {
        "schema": EXECUTION_SCHEMA,
        "status": status,
        "gate_evaluation": {
            "gate": "G8",
            "status": "NOT_EVALUATED",
            "reason": "a smoke or one Python condition backend is not a cross-backend C++ Gate result",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition_backend": args.condition_backend,
        "backend_provenance": prepared_manifest["condition_backends"][args.condition_backend],
        "prepared": {
            "artifact_path": str(prepared_path),
            "artifact_sha256": prepared_manifest["artifact"]["sha256"],
            "manifest_path": str(prepared_manifest_path),
            "manifest_sha256": _sha256(prepared_manifest_path),
            "common_inputs": "all tensors except the selected condition hidden are shared byte-for-byte",
        },
        "source": source,
        "checkpoint": checkpoint,
        "geometry": _geometry_manifest(),
        "runtime": _runtime_manifest(torch_npu, diffusers, devices),
        "execution": {
            "mode": args.execution_mode,
            "placement": placement,
            "attention_backend": "native",
            "attention_module_count": attention_module_count,
            "transformer_layers": TRANSFORMER_LAYERS,
            "transformer_forwards": forwards,
            "transformer_block_forwards": forwards * TRANSFORMER_LAYERS,
            "snapshot_forwards": [value for value in SNAPSHOT_FORWARDS if value <= forwards],
            "full_residual_cpu_exports": 0,
            "memory": {
                "transformer": transformer_memory,
                "video_vae": None if video_decode is None else video_decode["memory"],
                "audio_vae": None if audio_decode is None else audio_decode["memory"],
            },
            "timings_seconds": {
                "source_attestation": source_seconds,
                "checkpoint_attestation": checkpoint_seconds,
                "transformer_load_and_placement": load_seconds,
                "transformer_total": transformer_seconds,
                "per_forward": per_forward_seconds,
                "video_vae_load": None if video_decode is None else video_decode["load_seconds"],
                "video_vae_decode": None if video_decode is None else video_decode["decode_seconds"],
                "audio_vae_load": None if audio_decode is None else audio_decode["load_seconds"],
                "audio_vae_decode": None if audio_decode is None else audio_decode["decode_seconds"],
            },
        },
        "contracts": {
            "condition_selection": "exactly one prepared backend tensor; no fallback",
            "attention": "pinned Diffusers native full self-attention over 60,132 used rows",
            "placement": "50 transformer blocks split contiguously 25/25 over two explicit NPUs",
            "anchors": "11,072 image rows checked byte-exact after every scheduler update",
            "snapshots": "compact video/audio rows after forwards 1, 2, 4, 8, and 49 when reached",
            "residual_capture": "the 60,132x5,376 hidden state remains device-resident and is never hashed or copied to CPU",
            "video_decode": "pinned FP32 Video VAE weights under NPU FP16 autocast",
            "audio_decode": "pinned Audio VAE in FP32 without autocast",
        },
        "media": media,
        "tensors": {name: _tensor_summary(tensor) for name, tensor in sorted(archive.items())},
    }
    return _write_artifact_pair(args.output_dir.resolve(), archive_name, manifest_name, archive, manifest)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = _prepare(args) if args.command == "prepare" else _execute(args)
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("MiniMax-H3 C7 production reference failed")
        raise
