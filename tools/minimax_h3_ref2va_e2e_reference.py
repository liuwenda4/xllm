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
"""Generate the pinned MiniMax-H3 C7 real-weight Ref2VA image+audio Golden."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger

PINNED_DIFFUSERS_REVISION = "d30c748f5f5d0925a5af14dc0e6a6de983025e63"
DEFAULT_DIFFUSERS_SOURCE = Path("/data/workspace/lwd/minimax/diffusers-reference")
DEFAULT_CHECKPOINT_ROOT = Path("/data/workspace/lwd/minimax/checkpoints/Ref2VA-diffusers-d30c748f")
DEFAULT_C3_PACKING_GOLDEN = Path(
    "/data/workspace/lwd/minimax/artifacts/h3/20260910T223223Z-h3-c3/minimax_h3_packing_reference.safetensors"
)
C3_PACKING_SHA256 = "6a0687b8727c0073ed323ddfff71fca58f500d6c47c28b32fe498e5e8d58abbf"
DEFAULT_OUTPUT_DIR = Path("/data/workspace/lwd/minimax/artifacts/h3/20260912T121822Z-h3-c7")

ARCHIVE_NAME = "minimax_h3_ref2va_e2e_reference.safetensors"
MANIFEST_NAME = "minimax_h3_ref2va_e2e_reference.json"
MEDIA_NAME = "minimax_h3_ref2va_e2e_reference.mp4"

CONDITION_SHAPE = (1, 3, 5120)
CONDITION_TAGS = (1, 0, 1)
VISUAL_REFERENCE_SHAPE = (1, 24, 1, 16, 16)
AUDIO_REFERENCE_SHAPE = (2, 32, 4)
TARGET_VIDEO_SHAPE = (1, 24, 7, 16, 16)
TARGET_AUDIO_SHAPE = (2, 32, 8)
DECODED_VIDEO_SHAPE = (1, 3, 22, 256, 256)
DECODED_AUDIO_SHAPE = (1, 2, 6400)
PATCH_SIZE = (1, 2, 2)
VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2
AUDIO_CHANNELS = 2
VIDEO_SIGMA_POINTS = 50
AUDIO_SIGMA_POINTS = 50
TRANSFORMER_FORWARDS = 49
TRANSFORMER_LAYERS = 50
VISUAL_ANCHOR_TIMESTEP = 0.999
AUDIO_ANCHOR_TIMESTEP = 1.0
SNAPSHOT_FORWARDS = (1, 2, 4, 8, 49)
VIDEO_FPS = 24
AUDIO_SAMPLE_RATE = 32000

DIFFUSERS_SOURCE_SHA256 = {
    "src/diffusers/models/attention.py": "3c61df6cc4832149eb654c1e82220f4a6b91daca13741c957c4e0faff7810adf",
    "src/diffusers/models/attention_dispatch.py": ("265dd891dc563578a3d808c9cc01faea893fbedaa71e4e3ed27e8f0196ce7d02"),
    "src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py": (
        "4c3c9745ee27d16ff343c4998244bad41cd8f4213f0029cf7ce11ebb6d72ca1b"
    ),
    "src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py": (
        "b56c5bb95a7dc99ace03df192544615c89ec01ac3b134a7f1dd7348b03731d57"
    ),
    "src/diffusers/models/autoencoders/vae.py": "8e6abad3bd7b7806dd9c6c451b2438641ef728d7884e6bfed37b867f719b98fc",
    "src/diffusers/models/embeddings.py": "4eb810f715786eb1f24f2a4641e529817f7c951b9caf2f7925811329a03ec796",
    "src/diffusers/models/normalization.py": "e92ebbb130082578f3cff3361891a1adaa32331347e7bb09e4c70b208a0a6059",
    "src/diffusers/models/transformers/transformer_minimax_h3.py": (
        "1926b1bc15a5bebda05e3dc8cde1b3955d56641ba7f8f9d6e90c8f78c66ae30c"
    ),
    "src/diffusers/modular_pipelines/minimax_h3/before_denoise.py": (
        "530b007c1d689c3ee1fc1690527f5253522d2da6b44dd326bec99faaf9f72fff"
    ),
    "src/diffusers/modular_pipelines/minimax_h3/decoders.py": (
        "db553956502537613d17f83a5e1ac44f880b46514d254115676c0efe49ca0776"
    ),
    "src/diffusers/modular_pipelines/minimax_h3/modular_pipeline.py": (
        "9d5284ac8390f97d3e5eb3e0a50eddf25e04e7a12ff7fe063409b3cbd333b0ff"
    ),
    "src/diffusers/modular_pipelines/minimax_h3/references.py": (
        "9d20d0031ca1bc98b4556c73845601f3f69f995d702a954c160265303da160cb"
    ),
    "src/diffusers/schedulers/scheduling_minimax_h3.py": (
        "307d5bf755337ef00c47237f9ac8be116e627d26e1df3b5f0bd504a80f9de8dd"
    ),
}

ROOT_CHECKPOINT_SHA256 = {
    "modular_model_index.json": "83eff350057df4556d8596985046032b4266fb244a4f276eb3b70de988040858"
}
TRANSFORMER_CHECKPOINT_SHA256 = {
    "config.json": "0b8bcb003cca6d8fdc3c0732466772975a83b28419c0e9b2777b1731cf8a122d",
    "diffusion_pytorch_model-00001-of-00013.safetensors": (
        "c0e8e5f0817473cb8606e5de651f177ffa5a5f6b2cc6d4c1e09c553cd50b8a03"
    ),
    "diffusion_pytorch_model-00002-of-00013.safetensors": (
        "f1d275f2c0fde80de8bf3ff711765f577e6b6d80ba1ef10a72cd1959972d1fb6"
    ),
    "diffusion_pytorch_model-00003-of-00013.safetensors": (
        "6677035fa565a3000c94f7eea904c2bb1c7b0977fef6368a0f834c2e60766491"
    ),
    "diffusion_pytorch_model-00004-of-00013.safetensors": (
        "d1b1e34a0f1cd865ecc60fb688080b963d3ed8e7ebd607202177cbb43311100c"
    ),
    "diffusion_pytorch_model-00005-of-00013.safetensors": (
        "0383f5d1624d88499a9d02499241f437a777ba45270ddf17cfcd8a9ec5284a1f"
    ),
    "diffusion_pytorch_model-00006-of-00013.safetensors": (
        "822eddf70df028c2b9a893cf2380458ae348cdeaa1cee342d204a9910aa9d8d0"
    ),
    "diffusion_pytorch_model-00007-of-00013.safetensors": (
        "82dcc4026d82deee64b655df61d081990363273ca1fe3e7470f42d095563785b"
    ),
    "diffusion_pytorch_model-00008-of-00013.safetensors": (
        "df64aa42794aa8c4bb4a8bb5ee4bb17e48e81b07a4e03417b670eb014d427430"
    ),
    "diffusion_pytorch_model-00009-of-00013.safetensors": (
        "f6b0ed5c006dffe10a2d4e56ba6c01ff9bb2ef40c1c5f7d752f14150a54fb886"
    ),
    "diffusion_pytorch_model-00010-of-00013.safetensors": (
        "835cb211bc3f138c800360a5f4100a64fa09b0f6e9591da3b31fbfb03111033c"
    ),
    "diffusion_pytorch_model-00011-of-00013.safetensors": (
        "951659296a8b4243bb5f9c0ca0e572e8afb3d2263b3351e858923dc15956ded2"
    ),
    "diffusion_pytorch_model-00012-of-00013.safetensors": (
        "6cf0ab1662e5c8a3f999a80e403aeb1ab28068a2632d71f3b9f73879daabe26c"
    ),
    "diffusion_pytorch_model-00013-of-00013.safetensors": (
        "e266c789c2afc02cf496e45d2094aa0e2852b231daaa6a89a2ae4a4a85512446"
    ),
    "diffusion_pytorch_model.safetensors.index.json": (
        "c63df740da3b3b87f239b40df74185d027ca16e9291e7fa95997511e0c65fa34"
    ),
}
VIDEO_VAE_CHECKPOINT_SHA256 = {
    "config.json": "7cb686ac348f60f82214935854665e47da85fc7d95f12ad8ee98448ca44a8cfc",
    "diffusion_pytorch_model-00001-of-00002.safetensors": (
        "16696561a757bc356508a8fe22de04254945056905613ce5131b9586165b4198"
    ),
    "diffusion_pytorch_model-00002-of-00002.safetensors": (
        "0f52e2f980d2290b40d63cef66e9aa41cc3a683477c8c8571755715df67673bf"
    ),
    "diffusion_pytorch_model.safetensors.index.json": (
        "d8f364f0f7c924aafd621ddf5bbb83fd7eac8165462a843d129d2af82b9f5a6e"
    ),
}
AUDIO_VAE_CHECKPOINT_SHA256 = {
    "config.json": "0867acbc29fb725e8b1a58d9ad1454792a31d23c67402febbfa080f8b073cbfc",
    "diffusion_pytorch_model.safetensors": "52c59e67ba8de5477c81bfbced0327aabf500f1bfdeefd5ee754529241cb26cb",
}
VIDEO_SCHEDULER_CHECKPOINT_SHA256 = {
    "scheduler_config.json": "3a3f0eb45d59da4766f9bd334f12c271fc6b74515c70582c873c69cf099c23e9"
}
AUDIO_SCHEDULER_CHECKPOINT_SHA256 = {
    "scheduler_config.json": "3a471c54bfa4e4b5510ea8f91ea1d54d9a47c85d92b0faebe830e4f355d08caf"
}


@dataclass(frozen=True)
class ReferenceSpec:
    kind: str
    has_audio: bool = False


@dataclass(frozen=True)
class ExplicitInputs:
    condition_hidden: torch.Tensor
    condition_tags: torch.Tensor
    visual_reference: torch.Tensor
    visual_noise: torch.Tensor
    audio_reference: torch.Tensor
    target_video_initial: torch.Tensor
    target_audio_initial: torch.Tensor


@dataclass(frozen=True)
class PackedLayout:
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    num_condition_video_rows: int
    num_condition_audio_rows: int

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])


LayoutBuilder = Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]]
RowTimestepBuilder = Callable[..., tuple[torch.Tensor, torch.Tensor]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusers-source", type=Path, default=DEFAULT_DIFFUSERS_SOURCE)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--c3-packing-golden", type=Path, default=DEFAULT_C3_PACKING_GOLDEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--devices", nargs=2, default=("npu:0", "npu:1"), metavar=("FIRST", "SECOND"))
    parser.add_argument("--vae-device", default="npu:0")
    parser.add_argument("--write-media", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().to("cpu").contiguous()
    finite = torch.isfinite(value) if value.is_floating_point() else torch.ones_like(value, dtype=torch.bool)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "numel": value.numel(),
        "finite": bool(finite.all().item()),
        "sha256": _tensor_digest(value),
    }


def _to_archive(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").contiguous().clone()


def _validate_source(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"pinned source file does not exist: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"source digest mismatch for {path}: {actual_sha256}")
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": actual_sha256}


def _source_manifest(diffusers_source: Path) -> dict[str, Any]:
    source_root = diffusers_source.resolve()
    files = {
        relative: _validate_source(source_root / relative, expected)
        for relative, expected in sorted(DIFFUSERS_SOURCE_SHA256.items())
    }
    source = {
        "path": str(source_root),
        "revision": PINNED_DIFFUSERS_REVISION,
        "revision_verification": "pinned_file_sha256",
        "files": files,
        "generator": {
            "path": str(Path(__file__).resolve()),
            "size": Path(__file__).stat().st_size,
            "sha256": _sha256(Path(__file__)),
        },
    }
    source["digest"] = _canonical_digest(source)
    return source


def _activate_diffusers_source(diffusers_source: Path) -> None:
    source_package = (diffusers_source.resolve() / "src").resolve()
    if not source_package.is_dir():
        raise ValueError(f"pinned Diffusers src directory does not exist: {source_package}")
    loaded = sys.modules.get("diffusers")
    if loaded is not None:
        loaded_path_value = getattr(loaded, "__file__", None)
        loaded_path = Path(loaded_path_value).resolve() if loaded_path_value else None
        if loaded_path is None or source_package not in loaded_path.parents:
            raise RuntimeError(f"Diffusers was imported from outside the pinned source tree: {loaded_path}")
    source_value = str(source_package)
    sys.path[:] = [entry for entry in sys.path if entry != source_value]
    sys.path.insert(0, source_value)


def _component_checkpoint_manifest(
    component_path: Path,
    expected_files: dict[str, str],
    *,
    index_name: str | None = None,
    expected_tensor_count: int | None = None,
) -> dict[str, Any]:
    from safetensors import safe_open

    root = component_path.resolve()
    if not root.is_dir():
        raise ValueError(f"checkpoint component directory does not exist: {root}")
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    if actual_names != set(expected_files):
        raise ValueError(
            f"checkpoint file set mismatch for {root}: missing={sorted(set(expected_files) - actual_names)}, "
            f"extra={sorted(actual_names - set(expected_files))}"
        )
    logger.info(f"Hashing {len(expected_files)} pinned checkpoint files under {root}")
    files = {}
    for name, expected in sorted(expected_files.items()):
        path = root / name
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"checkpoint digest mismatch for {path}: {actual}")
        files[name] = {"size": path.stat().st_size, "sha256": actual}

    tensor_count = None
    total_size = None
    if index_name is not None:
        with (root / index_name).open(encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"checkpoint index has no weight map: {root / index_name}")
        indexed_shards = set(weight_map.values())
        present_shards = {name for name in expected_files if name.endswith(".safetensors")}
        if indexed_shards != present_shards:
            raise ValueError(f"checkpoint index shard set mismatch for {root}")
        tensor_count = len(weight_map)
        total_size = int(index.get("metadata", {}).get("total_size", 0))
    elif expected_tensor_count is not None:
        tensor_paths = [root / name for name in expected_files if name.endswith(".safetensors")]
        tensor_count = 0
        for path in tensor_paths:
            with safe_open(path, framework="pt", device="cpu") as handle:
                tensor_count += len(list(handle.keys()))
    if expected_tensor_count is not None and tensor_count != expected_tensor_count:
        raise ValueError(f"checkpoint tensor count for {root} is {tensor_count}, expected {expected_tensor_count}")

    result: dict[str, Any] = {"path": str(root), "files": files}
    if tensor_count is not None:
        result["tensor_count"] = tensor_count
    if total_size is not None:
        result["total_size"] = total_size
    result["digest"] = _canonical_digest(result)
    return result


def _checkpoint_manifest(checkpoint_root: Path) -> dict[str, Any]:
    root = checkpoint_root.resolve()
    if not root.is_dir():
        raise ValueError(f"checkpoint root does not exist: {root}")
    root_file = root / "modular_model_index.json"
    root_entry = _validate_source(root_file, ROOT_CHECKPOINT_SHA256[root_file.name])
    checkpoint = {
        "path": str(root),
        "modular_model_index": root_entry,
        "transformer_ref": _component_checkpoint_manifest(
            root / "transformer_ref",
            TRANSFORMER_CHECKPOINT_SHA256,
            index_name="diffusion_pytorch_model.safetensors.index.json",
            expected_tensor_count=638,
        ),
        "vae": _component_checkpoint_manifest(
            root / "vae",
            VIDEO_VAE_CHECKPOINT_SHA256,
            index_name="diffusion_pytorch_model.safetensors.index.json",
        ),
        "audio_vae": _component_checkpoint_manifest(
            root / "audio_vae", AUDIO_VAE_CHECKPOINT_SHA256, expected_tensor_count=1087
        ),
        "scheduler": _component_checkpoint_manifest(root / "scheduler", VIDEO_SCHEDULER_CHECKPOINT_SHA256),
        "audio_scheduler": _component_checkpoint_manifest(root / "audio_scheduler", AUDIO_SCHEDULER_CHECKPOINT_SHA256),
    }
    checkpoint["digest"] = _canonical_digest(checkpoint)
    return checkpoint


def _normalized_video(
    shape: tuple[int, int, int, int, int], coefficients: tuple[int, ...], offset: int
) -> torch.Tensor:
    batch, channels, frames, height, width = shape
    b = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1, 1, 1)
    c = torch.arange(channels, dtype=torch.float32).view(1, channels, 1, 1, 1)
    t = torch.arange(frames, dtype=torch.float32).view(1, 1, frames, 1, 1)
    y = torch.arange(height, dtype=torch.float32).view(1, 1, 1, height, 1)
    x = torch.arange(width, dtype=torch.float32).view(1, 1, 1, 1, width)
    values = coefficients[0] * b + coefficients[1] * c + coefficients[2] * t
    values = values + coefficients[3] * y + coefficients[4] * x + offset
    return values.remainder(257.0).sub(128.0).div(128.0).contiguous()


def _normalized_audio(shape: tuple[int, int, int], coefficients: tuple[int, ...], offset: int) -> torch.Tensor:
    batch, channels, frames = shape
    b = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1)
    c = torch.arange(channels, dtype=torch.float32).view(1, channels, 1)
    t = torch.arange(frames, dtype=torch.float32).view(1, 1, frames)
    values = coefficients[0] * b + coefficients[1] * c + coefficients[2] * t + offset
    return values.remainder(257.0).sub(128.0).div(128.0).contiguous()


def _prepare_explicit_inputs() -> ExplicitInputs:
    condition_values = torch.arange(torch.tensor(CONDITION_SHAPE).prod().item(), dtype=torch.float32)
    condition_hidden = condition_values.remainder(257).sub(128).reshape(CONDITION_SHAPE).to(torch.bfloat16)
    inputs = ExplicitInputs(
        condition_hidden=condition_hidden.contiguous(),
        condition_tags=torch.tensor([CONDITION_TAGS], dtype=torch.int64),
        visual_reference=_normalized_video(VISUAL_REFERENCE_SHAPE, (71, 53, 29, 11, 7), 0),
        visual_noise=_normalized_video(VISUAL_REFERENCE_SHAPE, (67, 97, 31, 17, 13), 19),
        audio_reference=_normalized_audio(AUDIO_REFERENCE_SHAPE, (97, 53, 29), 0),
        target_video_initial=_normalized_video(TARGET_VIDEO_SHAPE, (79, 89, 43, 17, 5), 23),
        target_audio_initial=_normalized_audio(TARGET_AUDIO_SHAPE, (101, 47, 31), 13),
    )
    _validate_explicit_inputs(inputs)
    return inputs


def _validate_explicit_inputs(inputs: ExplicitInputs) -> None:
    expected = {
        "condition_hidden": (CONDITION_SHAPE, torch.bfloat16),
        "condition_tags": ((1, len(CONDITION_TAGS)), torch.int64),
        "visual_reference": (VISUAL_REFERENCE_SHAPE, torch.float32),
        "visual_noise": (VISUAL_REFERENCE_SHAPE, torch.float32),
        "audio_reference": (AUDIO_REFERENCE_SHAPE, torch.float32),
        "target_video_initial": (TARGET_VIDEO_SHAPE, torch.float32),
        "target_audio_initial": (TARGET_AUDIO_SHAPE, torch.float32),
    }
    for name, (shape, dtype) in expected.items():
        value = getattr(inputs, name)
        if tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous():
            raise ValueError(
                f"explicit input {name!r} must be contiguous {dtype} {shape}, got {value.dtype} {tuple(value.shape)}"
            )
        if value.is_floating_point() and not bool(value.isfinite().all().item()):
            raise ValueError(f"explicit input {name!r} contains NaN or Inf")
    if tuple(inputs.condition_tags[0].tolist()) != CONDITION_TAGS:
        raise ValueError(f"condition tags must be {CONDITION_TAGS}")


def _mix_visual_anchor(
    clean: torch.Tensor, noise: torch.Tensor, timestep: float = VISUAL_ANCHOR_TIMESTEP
) -> torch.Tensor:
    if clean.shape != noise.shape or clean.dtype != torch.float32 or noise.dtype != torch.float32:
        raise ValueError("visual anchor and explicit noise must be same-shaped float32 tensors")
    if not 0.0 <= timestep <= 1.0:
        raise ValueError(f"visual anchor timestep must be in [0,1], got {timestep}")
    timestep_tensor = torch.tensor(timestep, dtype=clean.dtype, device=clean.device)
    return (timestep_tensor * clean + (1.0 - timestep_tensor) * noise).contiguous()


def _pack_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int] = PATCH_SIZE) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError(
            f"video latents must have shape [batch,channels,frames,height,width], got {tuple(latents.shape)}"
        )
    patch_t, patch_h, patch_w = patch_size
    batch, channels, frames, height, width = latents.shape
    if min(patch_size) <= 0 or frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"video latent shape {tuple(latents.shape)} is not divisible by patch {patch_size}")
    rows = latents.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    return rows.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def _unpack_video_rows(
    rows: torch.Tensor,
    latent_shape: tuple[int, int, int, int, int],
    patch_size: tuple[int, int, int] = PATCH_SIZE,
) -> torch.Tensor:
    batch, channels, frames, height, width = latent_shape
    patch_t, patch_h, patch_w = patch_size
    expected_rows = batch * (frames // patch_t) * (height // patch_h) * (width // patch_w)
    expected_width = channels * patch_t * patch_h * patch_w
    if tuple(rows.shape) != (expected_rows, expected_width):
        raise ValueError(f"video rows have shape {tuple(rows.shape)}, expected {(expected_rows, expected_width)}")
    value = rows.reshape(
        batch,
        frames // patch_t,
        height // patch_h,
        width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    return value.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(latent_shape).contiguous()


def _pack_audio_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 3 or latents.shape[0] != AUDIO_CHANNELS or latents.shape[1] != 32:
        raise ValueError(f"audio latents must have shape [2,32,frames], got {tuple(latents.shape)}")
    return latents.permute(0, 2, 1).reshape(-1, latents.shape[1]).contiguous()


def _unpack_audio_rows(rows: torch.Tensor, num_frames: int) -> torch.Tensor:
    expected_shape = (AUDIO_CHANNELS * num_frames, 32)
    if tuple(rows.shape) != expected_shape:
        raise ValueError(f"audio rows have shape {tuple(rows.shape)}, expected {expected_shape}")
    return rows.reshape(AUDIO_CHANNELS, num_frames, 32).permute(0, 2, 1).contiguous()


def _validate_references(
    references: Sequence[ReferenceSpec],
    visual_latents: Sequence[torch.Tensor],
    audio_rows: Sequence[torch.Tensor],
) -> None:
    if not references or len(references) > 12:
        raise ValueError("Ref2VA requires between 1 and 12 references")
    allowed = {"image", "video", "audio"}
    kinds = [reference.kind for reference in references]
    if any(kind not in allowed for kind in kinds):
        raise ValueError(f"invalid reference kind in {kinds}")
    if kinds.count("image") > 9 or kinds.count("video") > 3 or kinds.count("audio") > 3:
        raise ValueError("reference modality count exceeds the Ref2VA limit")
    if not any(kind in ("image", "video") for kind in kinds):
        raise ValueError("audio references require at least one image or video reference")
    if any(reference.has_audio for reference in references if reference.kind == "image"):
        raise ValueError("an image reference cannot carry audio")
    if any(not reference.has_audio for reference in references if reference.kind == "audio"):
        raise ValueError("an audio reference must carry audio")

    expected_visual = sum(reference.kind in ("image", "video") for reference in references)
    expected_audio = sum(
        reference.kind == "audio" or (reference.kind == "video" and reference.has_audio) for reference in references
    )
    if len(visual_latents) != expected_visual or len(audio_rows) != expected_audio:
        raise ValueError(
            f"reference latent count mismatch: expected visual/audio {expected_visual}/{expected_audio}, "
            f"got {len(visual_latents)}/{len(audio_rows)}"
        )
    for value in visual_latents:
        if value.ndim != 5 or value.shape[0] != 1 or value.shape[1] != 24:
            raise ValueError(f"invalid visual reference latent shape {tuple(value.shape)}")
    for value in audio_rows:
        if value.ndim != 2 or value.shape[1] != 32 or value.shape[0] % AUDIO_CHANNELS:
            raise ValueError(f"invalid audio reference row shape {tuple(value.shape)}")


def _c7_references() -> tuple[ReferenceSpec, ReferenceSpec]:
    return ReferenceSpec("image"), ReferenceSpec("audio", has_audio=True)


def _as_layout(
    values: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int],
) -> PackedLayout:
    return PackedLayout(*values)


def _build_c7_layout(builder: LayoutBuilder, inputs: ExplicitInputs) -> PackedLayout:
    references = _c7_references()
    visual_latents = [inputs.visual_reference]
    audio_rows = [_pack_audio_latents(inputs.audio_reference)]
    _validate_references(references, visual_latents, audio_rows)
    layout = _as_layout(
        builder(
            inputs.condition_tags[0],
            list(references),
            visual_latents,
            audio_rows,
            TARGET_VIDEO_SHAPE[2],
            TARGET_VIDEO_SHAPE[3],
            TARGET_VIDEO_SHAPE[4],
            TARGET_AUDIO_SHAPE[2],
            PATCH_SIZE,
            AUDIO_CHANNELS,
            AUDIO_TAG,
            VIDEO_TAG,
        )
    )
    _validate_c7_layout(layout)
    return layout


def _validate_c7_layout(layout: PackedLayout) -> None:
    expected_video_indices = torch.cat((torch.arange(3, 67), torch.arange(91, 539)))
    expected_audio_indices = torch.arange(67, 91)
    expected_tags = torch.cat(
        (
            torch.tensor(CONDITION_TAGS),
            torch.full((64,), VIDEO_TAG),
            torch.full((8 + 16,), AUDIO_TAG),
            torch.full((448,), VIDEO_TAG),
        )
    )
    if layout.sequence_length != 539 or tuple(layout.position_ids.shape) != (539, 3):
        raise ValueError(f"C7 packed sequence must have 539 rows, got {tuple(layout.position_ids.shape)}")
    if layout.position_ids.dtype != torch.float64:
        raise ValueError("C7 position IDs must be float64")
    if not torch.equal(layout.text_indices, torch.arange(3)):
        raise ValueError("C7 text rows are not first")
    if not torch.equal(layout.video_indices, expected_video_indices):
        raise ValueError("C7 video row ordering is not [image reference | target video]")
    if not torch.equal(layout.audio_indices, expected_audio_indices):
        raise ValueError("C7 audio row ordering is not [audio reference | target audio]")
    if not torch.equal(layout.token_tags, expected_tags):
        raise ValueError("C7 packed modality tags do not match the C3 image_audio contract")
    if layout.num_condition_video_rows != 64 or layout.num_condition_audio_rows != 8:
        raise ValueError("C7 reference row counts must be 64 visual and 8 audio")


def _validate_c3_equivalence(builder: LayoutBuilder, packing_golden: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    if not packing_golden.is_file():
        raise ValueError(f"C3 packing Golden does not exist: {packing_golden}")
    packing_sha256 = _sha256(packing_golden)
    if packing_sha256 != C3_PACKING_SHA256:
        raise ValueError(f"C3 packing Golden digest mismatch: {packing_sha256}")
    archive = load_file(str(packing_golden), device="cpu")
    references = list(_c7_references())
    official = _as_layout(
        builder(
            torch.tensor(CONDITION_TAGS, dtype=torch.int64),
            references,
            [torch.empty((1, 24, 1, 4, 4))],
            [torch.empty((4, 32))],
            2,
            4,
            6,
            2,
            PATCH_SIZE,
            AUDIO_CHANNELS,
            AUDIO_TAG,
            VIDEO_TAG,
        )
    )
    comparisons = {
        "position_ids": "position_ids",
        "token_tags": "token_tags",
        "video_indices": "img_pos",
        "audio_indices": "audio_pos",
        "text_indices": "text_pos",
    }
    for field_name, c3_name in comparisons.items():
        value = getattr(official, field_name)
        expected = archive[f"image_audio.{c3_name}"][: value.shape[0]]
        if not torch.equal(value, expected):
            raise ValueError(f"pinned Diffusers {field_name} is not exactly equivalent to C3 image_audio")
    if (official.num_condition_video_rows, official.num_condition_audio_rows) != (4, 4):
        raise ValueError("pinned Diffusers C3-equivalent anchor counts are not 4/4")
    return {
        "path": str(packing_golden.resolve()),
        "sha256": packing_sha256,
        "case": "image_audio",
        "used_length": 27,
        "aligned_length": 64,
        "diffusers_used_rows_exact": True,
    }


def _target_update_mask(row_count: int, condition_rows: int) -> torch.Tensor:
    if row_count <= 0 or condition_rows < 0 or condition_rows >= row_count:
        raise ValueError(f"invalid target mask geometry: rows={row_count}, condition_rows={condition_rows}")
    mask = torch.zeros(row_count, dtype=torch.bool)
    mask[condition_rows:] = True
    return mask


def _extract_target_rows(rows: torch.Tensor, update_mask: torch.Tensor) -> torch.Tensor:
    if update_mask.dtype != torch.bool or update_mask.ndim != 1 or update_mask.numel() != rows.shape[0]:
        raise ValueError("update mask must be a one-dimensional bool tensor matching the compact rows")
    if not bool(update_mask.any().item()) or bool(update_mask.all().item()):
        raise ValueError("update mask must contain both anchors and target rows")
    return rows[update_mask].contiguous()


def _replace_target_rows(rows: torch.Tensor, target_rows: torch.Tensor, update_mask: torch.Tensor) -> torch.Tensor:
    expected = _extract_target_rows(rows, update_mask)
    if target_rows.shape != expected.shape or target_rows.dtype != rows.dtype:
        raise ValueError(f"replacement target rows must have shape {tuple(expected.shape)} and dtype {rows.dtype}")
    output = rows.clone()
    output[update_mask] = target_rows
    if not torch.equal(output[~update_mask], rows[~update_mask]):
        raise ValueError("anchor rows changed while replacing target rows")
    return output


def _initial_compact_rows(
    inputs: ExplicitInputs, visual_anchor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    video_rows = torch.cat((_pack_video_latents(visual_anchor), _pack_video_latents(inputs.target_video_initial)))
    audio_rows = torch.cat(
        (_pack_audio_latents(inputs.audio_reference), _pack_audio_latents(inputs.target_audio_initial))
    )
    video_update_mask = _target_update_mask(video_rows.shape[0], 64)
    audio_update_mask = _target_update_mask(audio_rows.shape[0], 8)
    if tuple(video_rows.shape) != (512, 96) or tuple(audio_rows.shape) != (24, 32):
        raise ValueError(
            f"compact C7 rows have unexpected shapes {tuple(video_rows.shape)} and {tuple(audio_rows.shape)}"
        )
    return video_rows, audio_rows, video_update_mask, audio_update_mask


def _build_row_plans(
    builder: RowTimestepBuilder,
    layout: PackedLayout,
    video_timesteps: torch.Tensor,
    audio_timesteps: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if video_timesteps.numel() != TRANSFORMER_FORWARDS or audio_timesteps.numel() != TRANSFORMER_FORWARDS:
        raise ValueError("C7 native schedules must each drive exactly 49 forwards")
    plans = []
    for video_timestep, audio_timestep in zip(video_timesteps, audio_timesteps, strict=True):
        plans.append(
            builder(
                layout.video_indices,
                layout.audio_indices,
                layout.num_condition_video_rows,
                layout.num_condition_audio_rows,
                layout.text_indices.numel(),
                float(video_timestep),
                float(audio_timestep),
                max(float(video_timestep), VISUAL_ANCHOR_TIMESTEP),
                AUDIO_ANCHOR_TIMESTEP,
            )
        )
    return plans


def _set_native_attention(model: Any) -> int:
    from diffusers.models.attention import AttentionModuleMixin

    attention_modules = [module for module in model.modules() if isinstance(module, AttentionModuleMixin)]
    model.set_attention_backend("native")
    if any(module.processor._attention_backend.value != "native" for module in attention_modules):
        raise ValueError("C7 failed to select native attention for every attention module")
    return len(attention_modules)


def _place_transformer(model: Any, devices: tuple[torch.device, torch.device]) -> dict[str, Any]:
    if len(model.transformer_blocks) != TRANSFORMER_LAYERS:
        raise ValueError(f"C7 requires a 50-layer transformer, found {len(model.transformer_blocks)}")
    first, second = devices
    first_modules = (
        "context_embedder",
        "token_refiner",
        "time_proj",
        "time_embedder",
        "rope",
        "proj_in",
        "audio_proj_in",
    )
    last_modules = ("norm_out", "proj_out", "audio_proj_out")
    for name in first_modules:
        model.get_submodule(name).to(first)
    for name in last_modules:
        model.get_submodule(name).to(second)
    block_devices = []
    for index, block in enumerate(model.transformer_blocks):
        device = first if index < TRANSFORMER_LAYERS // 2 else second
        block.to(device)
        block_devices.append(str(device))
    return {
        "strategy": "typed_eager_contiguous_pipeline",
        "device_handoff_after_layer": 24,
        "first_device_modules": list(first_modules),
        "last_device_modules": list(last_modules),
        "block_devices": block_devices,
    }


def _parameter_dtype(module: Any) -> torch.dtype:
    return next(module.parameters()).dtype


def _eager_transformer_forward(
    model: Any,
    layout: PackedLayout,
    condition_hidden: torch.Tensor,
    video_rows: torch.Tensor,
    audio_rows: torch.Tensor,
    unique_timesteps: torch.Tensor,
    timestep_indices: torch.Tensor,
    forward_number: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    first = next(model.context_embedder.parameters()).device
    last = next(model.norm_out.parameters()).device
    condition = condition_hidden.to(first)
    video_indices_first = layout.video_indices.to(first)
    audio_indices_first = layout.audio_indices.to(first)
    text_indices_first = layout.text_indices.to(first)

    video_embeds = model.proj_in(video_rows.to(first, _parameter_dtype(model.proj_in)).unsqueeze(0))
    audio_embeds = model.audio_proj_in(audio_rows.to(first, _parameter_dtype(model.audio_proj_in)).unsqueeze(0))
    text_embeds = model.context_embedder(condition.to(_parameter_dtype(model.context_embedder)))
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

    final_layer = _to_archive(hidden.squeeze(0))
    final_layer_digest = {
        "forward": forward_number,
        "schedule_index": forward_number - 1,
        "layer": TRANSFORMER_LAYERS - 1,
        "shape": list(final_layer.shape),
        "dtype": str(final_layer.dtype).removeprefix("torch."),
        "sha256": _tensor_digest(final_layer),
    }

    hidden = hidden.to(last)
    time_embedding = time_embedding.to(last)
    timestep_indices_last = timestep_indices.to(last)
    activation = model.norm_out(hidden, time_embedding, timestep_indices_last).to(_parameter_dtype(model.proj_out))
    all_video = model.proj_out(activation)
    all_audio = model.audio_proj_out(activation)
    video_velocity = all_video.index_select(1, layout.video_indices.to(last)).squeeze(0).to(first)
    audio_velocity = all_audio.index_select(1, layout.audio_indices.to(last)).squeeze(0).to(first)
    return video_velocity, audio_velocity, final_layer_digest


def _run_denoise(
    model: Any,
    layout: PackedLayout,
    inputs: ExplicitInputs,
    video_rows_cpu: torch.Tensor,
    audio_rows_cpu: torch.Tensor,
    video_update_mask_cpu: torch.Tensor,
    audio_update_mask_cpu: torch.Tensor,
    video_scheduler: Any,
    audio_scheduler: Any,
    row_plans: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    first = next(model.context_embedder.parameters()).device
    video_rows = video_rows_cpu.to(first)
    audio_rows = audio_rows_cpu.to(first)
    video_update_mask = video_update_mask_cpu.to(first)
    audio_update_mask = audio_update_mask_cpu.to(first)
    video_anchor = video_rows[~video_update_mask].clone()
    audio_anchor = audio_rows[~audio_update_mask].clone()
    archive: dict[str, torch.Tensor] = {}
    final_layer_digests = []

    for index, ((unique, inverse), video_timestep, audio_timestep) in enumerate(
        zip(row_plans, video_scheduler.timesteps, audio_scheduler.timesteps, strict=True)
    ):
        forward_number = index + 1
        archive[f"schedule.forward_{forward_number:03d}.unique_timesteps"] = _to_archive(unique)
        archive[f"schedule.forward_{forward_number:03d}.timestep_indices"] = _to_archive(inverse)
        video_velocity, audio_velocity, layer_digest = _eager_transformer_forward(
            model,
            layout,
            inputs.condition_hidden,
            video_rows,
            audio_rows,
            unique,
            inverse,
            forward_number,
        )
        next_video_target = video_scheduler.step(
            video_velocity[video_update_mask].float(),
            video_timestep,
            video_rows[video_update_mask],
            return_dict=False,
        )[0]
        next_audio_target = audio_scheduler.step(
            audio_velocity[audio_update_mask].float(),
            audio_timestep,
            audio_rows[audio_update_mask],
            return_dict=False,
        )[0]
        video_rows = _replace_target_rows(video_rows, next_video_target, video_update_mask)
        audio_rows = _replace_target_rows(audio_rows, next_audio_target, audio_update_mask)
        if not torch.equal(video_rows[~video_update_mask], video_anchor) or not torch.equal(
            audio_rows[~audio_update_mask], audio_anchor
        ):
            raise ValueError(f"conditioning anchor changed after forward {forward_number}")
        if forward_number in SNAPSHOT_FORWARDS:
            archive[f"trajectory.forward_{forward_number:03d}.video_rows"] = _to_archive(video_rows)
            archive[f"trajectory.forward_{forward_number:03d}.audio_rows"] = _to_archive(audio_rows)
        final_layer_digests.append(layer_digest)
        logger.info(f"Completed pinned C7 transformer forward {forward_number}/{TRANSFORMER_FORWARDS}")

    return archive, final_layer_digests, _to_archive(video_rows), _to_archive(audio_rows)


def _channel_values(reference: torch.Tensor, values: Sequence[float]) -> torch.Tensor:
    if reference.ndim not in (3, 5) or reference.shape[1] != len(values):
        raise ValueError(f"channel values do not match tensor shape {list(reference.shape)}")
    shape = (1, -1, *(1 for _ in range(reference.ndim - 2)))
    return torch.tensor(values, dtype=torch.float32, device=reference.device).view(shape)


def _denormalize_latents(latents: torch.Tensor, means: Sequence[float], stds: Sequence[float]) -> torch.Tensor:
    value = latents.float()
    mean = _channel_values(value, means)
    std = _channel_values(value, stds)
    if bool((std <= 0).any().item()):
        raise ValueError("latent standard deviations must be positive")
    return (value * std + mean).contiguous()


def _memory_stats(devices: Sequence[torch.device]) -> dict[str, dict[str, int]]:
    return {
        str(device): {
            "allocated": int(torch.npu.memory_allocated(device)),
            "reserved": int(torch.npu.memory_reserved(device)),
            "peak_allocated": int(torch.npu.max_memory_allocated(device)),
            "peak_reserved": int(torch.npu.max_memory_reserved(device)),
        }
        for device in devices
    }


def _clear_npu(devices: Sequence[torch.device]) -> None:
    gc.collect()
    for device in devices:
        torch.npu.set_device(device)
        torch.npu.empty_cache()


def _decode_video(
    model_class: type[Any], checkpoint_path: Path, normalized_latents: torch.Tensor, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    torch.npu.set_device(device)
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = model_class.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    model.enable_tiling(
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_sample_min_overlap_height=64,
        tile_sample_min_overlap_width=64,
    )
    model.to(device)
    load_seconds = time.perf_counter() - started
    denormalized = _denormalize_latents(
        normalized_latents.to(device), tuple(model.config.latents_mean), tuple(model.config.latents_std)
    )
    decode_started = time.perf_counter()
    with torch.autocast(device_type="npu", dtype=torch.float16):
        decoded_raw = model.decode(denormalized, return_dict=False)[0]
    pixel_mean = _channel_values(decoded_raw, (0.485, 0.456, 0.406))
    pixel_std = _channel_values(decoded_raw, (0.229, 0.224, 0.225))
    decoded = (decoded_raw.float() * pixel_std + pixel_mean).clamp(0.0, 1.0)
    torch.npu.synchronize(device)
    decode_seconds = time.perf_counter() - decode_started
    tensors = {
        "decode.video_denormalized_latent": _to_archive(denormalized),
        "decode.video_raw": _to_archive(decoded_raw),
        "decoded.video": _to_archive(decoded),
    }
    execution = {
        "device": str(device),
        "weights_dtype": "float32",
        "compute": "native Diffusers eager with NPU FP16 autocast",
        "tiling": {"tile_sample": [256, 256], "minimum_overlap": [64, 64]},
        "load_seconds": load_seconds,
        "decode_seconds": decode_seconds,
        "memory": _memory_stats((device,))[str(device)],
    }
    del model, denormalized, decoded_raw, decoded
    _clear_npu((device,))
    return tensors, execution


def _decode_audio(
    model_class: type[Any], checkpoint_path: Path, normalized_latents: torch.Tensor, device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    torch.npu.set_device(device)
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = model_class.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    load_seconds = time.perf_counter() - started
    denormalized = _denormalize_latents(
        normalized_latents.to(device), tuple(model.config.latents_mean), tuple(model.config.latents_std)
    )
    decode_started = time.perf_counter()
    decoded_raw = model.decode(denormalized, return_dict=False)[0]
    decoded = decoded_raw.float().permute(1, 0, 2)
    torch.npu.synchronize(device)
    decode_seconds = time.perf_counter() - decode_started
    tensors = {
        "decode.audio_denormalized_latent": _to_archive(denormalized),
        "decode.audio_raw": _to_archive(decoded_raw),
        "decoded.audio": _to_archive(decoded),
        "decoded.audio_sampling_rate": torch.tensor(AUDIO_SAMPLE_RATE, dtype=torch.int64),
    }
    execution = {
        "device": str(device),
        "weights_dtype": "float32",
        "compute": "native Diffusers eager FP32; no autocast",
        "load_seconds": load_seconds,
        "decode_seconds": decode_seconds,
        "memory": _memory_stats((device,))[str(device)],
    }
    del model, denormalized, decoded_raw, decoded
    _clear_npu((device,))
    return tensors, execution


def _validate_archive(archive: dict[str, torch.Tensor]) -> None:
    expected_shapes = {
        "input.condition_hidden": CONDITION_SHAPE,
        "input.condition_tags": (1, 3),
        "input.visual_reference_normalized": VISUAL_REFERENCE_SHAPE,
        "input.visual_reference_noise": VISUAL_REFERENCE_SHAPE,
        "input.visual_reference_noised": VISUAL_REFERENCE_SHAPE,
        "input.audio_reference_normalized": AUDIO_REFERENCE_SHAPE,
        "input.target_video_initial_normalized": TARGET_VIDEO_SHAPE,
        "input.target_audio_initial_normalized": TARGET_AUDIO_SHAPE,
        "layout.position_ids": (539, 3),
        "layout.token_tags": (539,),
        "layout.video_indices": (512,),
        "layout.audio_indices": (24,),
        "layout.text_indices": (3,),
        "layout.video_update_mask": (512,),
        "layout.audio_update_mask": (24,),
        "layout.sequence_length": (),
        "layout.num_condition_video_rows": (),
        "layout.num_condition_audio_rows": (),
        "layout.reference_and_target_ranges": (5, 2),
        "initial.video_compact_rows": (512, 96),
        "initial.audio_compact_rows": (24, 32),
        "final.target_video_rows": (448, 96),
        "final.target_audio_rows": (16, 32),
        "final.target_video_latent": TARGET_VIDEO_SHAPE,
        "final.target_audio_latent": TARGET_AUDIO_SHAPE,
        "decoded.video": DECODED_VIDEO_SHAPE,
        "decoded.audio": DECODED_AUDIO_SHAPE,
    }
    for forward_number in SNAPSHOT_FORWARDS:
        expected_shapes[f"trajectory.forward_{forward_number:03d}.video_rows"] = (512, 96)
        expected_shapes[f"trajectory.forward_{forward_number:03d}.audio_rows"] = (24, 32)
    for name, expected_shape in expected_shapes.items():
        if name not in archive or tuple(archive[name].shape) != expected_shape:
            actual = tuple(archive[name].shape) if name in archive else None
            raise ValueError(f"C7 tensor {name!r} has shape {actual}, expected {expected_shape}")
    for name, tensor in archive.items():
        if tensor.is_floating_point() and not bool(tensor.isfinite().all().item()):
            raise ValueError(f"C7 tensor {name!r} contains NaN or Inf")


def _runtime_manifest(
    torch_npu: Any,
    diffusers: Any,
    devices: Sequence[torch.device],
    diffusers_source: Path = DEFAULT_DIFFUSERS_SOURCE,
) -> dict[str, Any]:
    imported_path = Path(diffusers.__file__).resolve()
    expected_source = (diffusers_source / "src").resolve()
    runtime = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "diffusers": diffusers.__version__,
        "diffusers_path": str(imported_path),
        "pinned_diffusers_import": expected_source in imported_path.parents,
        "devices": [{"device": str(device), "name": torch.npu.get_device_name(device)} for device in devices],
        "attention_backend": "native",
        "execution": "typed eager; inference_mode; no compile; no random draws",
    }
    runtime["digest"] = _canonical_digest(runtime)
    return runtime


def _write_optional_media(
    requested: bool, output_dir: Path, video: torch.Tensor, audio: torch.Tensor
) -> dict[str, Any]:
    if not requested:
        return {"requested": False, "written": False}
    if importlib.util.find_spec("av") is None:
        return {"requested": True, "written": False, "reason": "PyAV is unavailable"}
    try:
        from diffusers.utils.export_utils import encode_video
    except ImportError as error:
        return {"requested": True, "written": False, "reason": str(error)}

    frames = video[0].permute(1, 2, 3, 0).clamp(0, 1).mul(255).round().to(torch.uint8)
    soundtrack = audio[0]
    path = output_dir / MEDIA_NAME
    encode_video(
        frames,
        fps=VIDEO_FPS,
        output_path=str(path),
        audio=soundtrack,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
    )
    return {"requested": True, "written": True, "path": path.name, "size": path.stat().st_size, "sha256": _sha256(path)}


def _write_output(output_dir: Path, archive: dict[str, torch.Tensor], manifest: dict[str, Any]) -> Path:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    archive_temp = output_dir / f".{ARCHIVE_NAME}{suffix}"
    manifest_temp = output_dir / f".{MANIFEST_NAME}{suffix}"
    try:
        persisted = {name: value.detach().to("cpu").contiguous().clone() for name, value in sorted(archive.items())}
        save_file(persisted, str(archive_temp))
        archive_fd = os.open(archive_temp, os.O_RDONLY)
        try:
            os.fsync(archive_fd)
        finally:
            os.close(archive_fd)
        manifest["artifact"] = {
            "path": ARCHIVE_NAME,
            "size": archive_temp.stat().st_size,
            "sha256": _sha256(archive_temp),
        }
        with manifest_temp.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(archive_temp, output_dir / ARCHIVE_NAME)
        os.replace(manifest_temp, output_dir / MANIFEST_NAME)
        directory_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        archive_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)
    return output_dir / MANIFEST_NAME


def main() -> None:
    args = _parse_args()
    diffusers_source = args.diffusers_source.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    packing_golden = args.c3_packing_golden.resolve()
    output_dir = args.output_dir.resolve()
    if len(set(args.devices)) != 2 or any(not value.startswith("npu:") for value in args.devices):
        raise ValueError("C7 requires two distinct explicitly indexed NPU devices")
    devices = tuple(torch.device(value) for value in args.devices)
    vae_device = torch.device(args.vae_device)
    if vae_device not in devices:
        raise ValueError("the sequential VAE device must be one of the two transformer devices")

    source_started = time.perf_counter()
    source = _source_manifest(diffusers_source)
    source_seconds = time.perf_counter() - source_started
    _activate_diffusers_source(diffusers_source)

    import diffusers
    import torch_npu
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio, MiniMaxH3Scheduler
    from diffusers.models.transformers import MiniMaxH3Transformer3DModel
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3Ref2VAPrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )

    expected_source = (diffusers_source / "src").resolve()
    if expected_source not in Path(diffusers.__file__).resolve().parents:
        raise RuntimeError(f"Diffusers did not import from pinned source {expected_source}")
    if not torch.npu.is_available() or torch.npu.device_count() < 2:
        raise RuntimeError("C7 needs at least two available NPUs")
    for device in devices:
        torch.npu.set_device(device)
        torch.npu.empty_cache()

    checkpoint_started = time.perf_counter()
    checkpoint = _checkpoint_manifest(checkpoint_root)
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    layout_builder = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence
    row_timestep_builder = MiniMaxH3SetTimestepsStep.build_row_timesteps
    c3_packing = _validate_c3_equivalence(layout_builder, packing_golden)
    inputs = _prepare_explicit_inputs()
    layout = _build_c7_layout(layout_builder, inputs)

    video_scheduler = MiniMaxH3Scheduler.from_pretrained(checkpoint_root / "scheduler", local_files_only=True)
    audio_scheduler = MiniMaxH3Scheduler.from_pretrained(checkpoint_root / "audio_scheduler", local_files_only=True)
    video_scheduler.set_timesteps(VIDEO_SIGMA_POINTS, device=devices[0])
    audio_scheduler.set_timesteps(AUDIO_SIGMA_POINTS, device=devices[0])
    if video_scheduler.sigmas.numel() != VIDEO_SIGMA_POINTS or audio_scheduler.sigmas.numel() != AUDIO_SIGMA_POINTS:
        raise ValueError("native scheduler sigma point count is not exactly 50")
    row_plans = _build_row_plans(
        row_timestep_builder,
        layout,
        video_scheduler.timesteps,
        audio_scheduler.timesteps,
    )

    official_visual_anchor = video_scheduler.scale_noise(
        inputs.visual_reference.to(devices[0]),
        VISUAL_ANCHOR_TIMESTEP,
        inputs.visual_noise.to(devices[0]),
    )
    expected_visual_anchor = video_scheduler.scale_noise(
        inputs.visual_reference,
        VISUAL_ANCHOR_TIMESTEP,
        inputs.visual_noise,
    )
    if not torch.equal(expected_visual_anchor, _mix_visual_anchor(inputs.visual_reference, inputs.visual_noise)):
        raise ValueError("explicit visual anchor mixing does not match the pinned native scheduler")
    video_rows, audio_rows, video_update_mask, audio_update_mask = _initial_compact_rows(
        inputs, _to_archive(official_visual_anchor)
    )
    if not torch.equal(video_rows[:64], patchify_video_latents(official_visual_anchor, PATCH_SIZE).to("cpu")):
        raise ValueError("local C7 patch ordering does not match pinned Diffusers")

    archive: dict[str, torch.Tensor] = {
        "input.condition_hidden": _to_archive(inputs.condition_hidden),
        "input.condition_tags": _to_archive(inputs.condition_tags),
        "input.visual_reference_normalized": _to_archive(inputs.visual_reference),
        "input.visual_reference_noise": _to_archive(inputs.visual_noise),
        "input.visual_reference_noised": _to_archive(official_visual_anchor),
        "input.audio_reference_normalized": _to_archive(inputs.audio_reference),
        "input.target_video_initial_normalized": _to_archive(inputs.target_video_initial),
        "input.target_audio_initial_normalized": _to_archive(inputs.target_audio_initial),
        "layout.position_ids": _to_archive(layout.position_ids),
        "layout.token_tags": _to_archive(layout.token_tags),
        "layout.video_indices": _to_archive(layout.video_indices),
        "layout.audio_indices": _to_archive(layout.audio_indices),
        "layout.text_indices": _to_archive(layout.text_indices),
        "layout.video_update_mask": _to_archive(video_update_mask),
        "layout.audio_update_mask": _to_archive(audio_update_mask),
        "layout.sequence_length": torch.tensor(layout.sequence_length, dtype=torch.int64),
        "layout.num_condition_video_rows": torch.tensor(layout.num_condition_video_rows, dtype=torch.int64),
        "layout.num_condition_audio_rows": torch.tensor(layout.num_condition_audio_rows, dtype=torch.int64),
        "layout.reference_and_target_ranges": torch.tensor(
            [[0, 3], [3, 67], [67, 75], [75, 91], [91, 539]], dtype=torch.int64
        ),
        "initial.video_compact_rows": _to_archive(video_rows),
        "initial.audio_compact_rows": _to_archive(audio_rows),
        "schedule.video_sigmas": _to_archive(video_scheduler.sigmas),
        "schedule.audio_sigmas": _to_archive(audio_scheduler.sigmas),
        "schedule.video_timesteps": _to_archive(video_scheduler.timesteps),
        "schedule.audio_timesteps": _to_archive(audio_scheduler.timesteps),
    }

    logger.info(f"Loading pinned 50-layer Ref2VA transformer from {checkpoint_root / 'transformer_ref'}")
    transformer_load_started = time.perf_counter()
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
        raise ValueError(f"C7 expected 52 native attention modules, found {attention_module_count}")
    for device in devices:
        torch.npu.reset_peak_memory_stats(device)
    placement = _place_transformer(model, devices)
    for device in devices:
        torch.npu.synchronize(device)
    transformer_load_seconds = time.perf_counter() - transformer_load_started

    logger.info("Running all 49 C7 full-transformer forwards")
    denoise_started = time.perf_counter()
    with torch.inference_mode():
        trajectory, final_layer_digests, final_video_rows, final_audio_rows = _run_denoise(
            model,
            layout,
            inputs,
            video_rows,
            audio_rows,
            video_update_mask,
            audio_update_mask,
            video_scheduler,
            audio_scheduler,
            row_plans,
        )
    for device in devices:
        torch.npu.synchronize(device)
    denoise_seconds = time.perf_counter() - denoise_started
    transformer_memory = _memory_stats(devices)
    archive.update(trajectory)

    final_target_video_rows = _extract_target_rows(final_video_rows, video_update_mask)
    final_target_audio_rows = _extract_target_rows(final_audio_rows, audio_update_mask)
    final_video_latent = _unpack_video_rows(final_target_video_rows, TARGET_VIDEO_SHAPE)
    final_audio_latent = _unpack_audio_rows(final_target_audio_rows, TARGET_AUDIO_SHAPE[2])
    archive.update(
        {
            "final.target_video_rows": final_target_video_rows,
            "final.target_audio_rows": final_target_audio_rows,
            "final.target_video_latent": final_video_latent,
            "final.target_audio_latent": final_audio_latent,
        }
    )

    del model, row_plans, trajectory, final_video_rows, final_audio_rows, official_visual_anchor
    _clear_npu(devices)
    logger.info("Transformer released; loading and running the native video VAE")
    with torch.inference_mode():
        video_tensors, video_execution = _decode_video(
            AutoencoderKLMiniMaxH3, checkpoint_root / "vae", final_video_latent, vae_device
        )
    archive.update(video_tensors)
    logger.info("Video VAE released; loading and running the native audio VAE")
    with torch.inference_mode():
        audio_tensors, audio_execution = _decode_audio(
            AutoencoderKLMiniMaxH3Audio, checkpoint_root / "audio_vae", final_audio_latent, vae_device
        )
    archive.update(audio_tensors)
    _validate_archive(archive)

    output_dir.mkdir(parents=True, exist_ok=True)
    optional_media = _write_optional_media(
        args.write_media, output_dir, archive["decoded.video"], archive["decoded.audio"]
    )
    runtime = _runtime_manifest(torch_npu, diffusers, devices, diffusers_source)
    if expected_source not in Path(runtime["diffusers_path"]).parents:
        raise RuntimeError("runtime attestation is not from the pinned Diffusers source")
    manifest = {
        "schema": "xllm.minimax_h3.ref2va_e2e_reference/v1",
        "status": "OFFICIAL_H3_C7_TYPED_EAGER_COMPOSITE_GOLDEN",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "checkpoint": checkpoint,
        "runtime": runtime,
        "c3_packing_contract": c3_packing,
        "fixture": {
            "reference_order": ["image", "audio"],
            "condition_shape": list(CONDITION_SHAPE),
            "condition_dtype": "bfloat16",
            "condition_tags": list(CONDITION_TAGS),
            "visual_reference_shape": list(VISUAL_REFERENCE_SHAPE),
            "audio_reference_shape": list(AUDIO_REFERENCE_SHAPE),
            "target_video_shape": list(TARGET_VIDEO_SHAPE),
            "target_audio_shape": list(TARGET_AUDIO_SHAPE),
            "packed_sequence_length": layout.sequence_length,
            "num_condition_video_rows": layout.num_condition_video_rows,
            "num_condition_audio_rows": layout.num_condition_audio_rows,
            "compact_video_rows": list(video_rows.shape),
            "compact_audio_rows": list(audio_rows.shape),
            "decoded_video_shape": list(DECODED_VIDEO_SHAPE),
            "decoded_audio_shape": list(DECODED_AUDIO_SHAPE),
        },
        "contracts": {
            "explicit_inputs": "analytic CPU tensors only; no torch random API or generator is consulted",
            "model_outputs": "all trajectory, final latent, and decoded tensors come from pinned real checkpoint execution",
            "packing": "native Diffusers [text | image reference | audio reference | target audio | target video]",
            "visual_anchor": "scheduler.scale_noise(clean, 0.999, explicit_noise), held byte-exact for all forwards",
            "audio_anchor": "clean normalized reference rows at timestep 1.0, held byte-exact for all forwards",
            "target_update": "only rows selected by explicit compact update masks are passed to scheduler.step",
            "snapshots": "compact rows captured after scheduler updates following forwards 1, 2, 4, 8, and 49",
            "video_decode": "per-channel denormalize, native VAE decode under NPU FP16 autocast, ImageNet reverse-normalize and clamp [0,1]",
            "audio_decode": "per-channel denormalize, native FP32 mono-batch VAE decode, permute to [1,2,samples]",
            "atomic_output": "fsynced safetensors rename precedes fsynced manifest rename and directory fsync",
        },
        "schedule": {
            "video_shift": float(video_scheduler.config.shift),
            "audio_shift": float(audio_scheduler.config.shift),
            "video_sigma_points": video_scheduler.sigmas.numel(),
            "audio_sigma_points": audio_scheduler.sigmas.numel(),
            "transformer_forwards": TRANSFORMER_FORWARDS,
        },
        "execution": {
            "placement": placement,
            "attention_backend": "native",
            "attention_module_count": attention_module_count,
            "transformer_layers": TRANSFORMER_LAYERS,
            "transformer_forwards": TRANSFORMER_FORWARDS,
            "transformer_block_forwards": TRANSFORMER_LAYERS * TRANSFORMER_FORWARDS,
            "snapshot_forwards": list(SNAPSHOT_FORWARDS),
            "final_layer_digest_count": len(final_layer_digests),
            "final_layer_digests": final_layer_digests,
            "vae_residency": "sequential after transformer release; video VAE released before audio VAE load",
            "memory": {
                "transformer": transformer_memory,
                "video_vae": video_execution["memory"],
                "audio_vae": audio_execution["memory"],
            },
            "timings_seconds": {
                "source_attestation": source_seconds,
                "checkpoint_attestation": checkpoint_seconds,
                "transformer_load_and_placement": transformer_load_seconds,
                "denoise_49_forwards": denoise_seconds,
                "video_vae_load": video_execution["load_seconds"],
                "video_vae_decode": video_execution["decode_seconds"],
                "audio_vae_load": audio_execution["load_seconds"],
                "audio_vae_decode": audio_execution["decode_seconds"],
            },
        },
        "optional_media": optional_media,
        "tensors": {name: _tensor_summary(tensor) for name, tensor in sorted(archive.items())},
    }
    if len(final_layer_digests) != TRANSFORMER_FORWARDS:
        raise ValueError("C7 did not capture one final-layer digest per transformer forward")
    manifest_path = _write_output(output_dir, archive, manifest)
    print(manifest_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("MiniMax-H3 C7 Ref2VA end-to-end Golden generation failed")
        raise
