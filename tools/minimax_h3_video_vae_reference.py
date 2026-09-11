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
"""Generate the pinned MiniMax-H3 C6a video VAE Golden on one NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger

PINNED_DIFFUSERS_REVISION = "d30c748f5f5d0925a5af14dc0e6a6de983025e63"
DEFAULT_DIFFUSERS_SOURCE = Path("/data/workspace/lwd/minimax/diffusers-reference")
DEFAULT_VAE_PATH = Path("/data/workspace/lwd/minimax/checkpoints/Ref2VA-diffusers-d30c748f/vae")
ARCHIVE_NAME = "minimax_h3_video_vae_reference.safetensors"
MANIFEST_NAME = "minimax_h3_video_vae_reference.json"

INPUT_SHAPE = (1, 3, 39, 288, 288)
LATENT_SHAPE = (1, 24, 12, 18, 18)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ENCODE_SEED = 42
TILE_SIZE = 256
TILE_MIN_OVERLAP = 64
SPATIAL_COMPRESSION_RATIO = 16
TEMPORAL_COMPRESSION_RATIO = 4
CLIP_LENGTH = 17
TOKEN_DROP = 3
POSTERIOR_LOGVAR_MIN = -30.0
POSTERIOR_LOGVAR_MAX = 20.0

DIFFUSERS_SOURCE_SHA256 = {
    "src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py": (
        "4c3c9745ee27d16ff343c4998244bad41cd8f4213f0029cf7ce11ebb6d72ca1b"
    ),
    "src/diffusers/models/autoencoders/vae.py": ("8e6abad3bd7b7806dd9c6c451b2438641ef728d7884e6bfed37b867f719b98fc"),
    "src/diffusers/modular_pipelines/minimax_h3/decoders.py": (
        "db553956502537613d17f83a5e1ac44f880b46514d254115676c0efe49ca0776"
    ),
    "src/diffusers/modular_pipelines/minimax_h3/encoders.py": (
        "fea751a889752ba58f1528acbc23b223e827ea707a75e2ae0c43d4a53ce758af"
    ),
}

CHECKPOINT_SHA256 = {
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusers-source", type=Path, default=DEFAULT_DIFFUSERS_SOURCE)
    parser.add_argument("--vae-path", type=Path, default=DEFAULT_VAE_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
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


def _validate_source(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"pinned source file does not exist: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"source digest mismatch for {path}: {actual_sha256}")
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": actual_sha256,
    }


def _source_manifest(diffusers_source: Path) -> dict[str, Any]:
    source_root = diffusers_source.resolve()
    files = {
        relative_path: _validate_source(source_root / relative_path, expected_sha256)
        for relative_path, expected_sha256 in sorted(DIFFUSERS_SOURCE_SHA256.items())
    }
    source = {
        "path": str(source_root),
        "revision": PINNED_DIFFUSERS_REVISION,
        "revision_short": PINNED_DIFFUSERS_REVISION[:8],
        "revision_verification": "pinned_file_sha256",
        "files": files,
    }
    source["digest"] = _canonical_digest(source)
    return source


def _checkpoint_manifest(vae_path: Path) -> dict[str, Any]:
    vae_path = vae_path.resolve()
    index_path = vae_path / "diffusion_pytorch_model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(f"converted VAE index does not exist: {index_path}")
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("converted VAE index has no weight_map")
    indexed_shards = set(weight_map.values())
    expected_shards = {name for name in CHECKPOINT_SHA256 if name.endswith(".safetensors")}
    if indexed_shards != expected_shards:
        raise ValueError(f"converted VAE shard set mismatch: {sorted(indexed_shards)}")

    logger.info(f"Hashing {len(CHECKPOINT_SHA256)} converted VAE files")
    files = {}
    for name, expected_sha256 in sorted(CHECKPOINT_SHA256.items()):
        path = vae_path / name
        if not path.is_file():
            raise ValueError(f"converted VAE file does not exist: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"checkpoint digest mismatch for {path}: {actual_sha256}")
        files[name] = {"size": path.stat().st_size, "sha256": actual_sha256}
    checkpoint = {
        "path": str(vae_path),
        "format": "diffusers_sharded_safetensors",
        "tensor_count": len(weight_map),
        "total_size": int(index.get("metadata", {}).get("total_size", 0)),
        "files": files,
    }
    checkpoint["digest"] = _canonical_digest(checkpoint)
    return checkpoint


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
    if source_value not in sys.path:
        sys.path.insert(0, source_value)


def _channel_values(reference: Any, values: tuple[float, ...]) -> Any:
    import torch

    if reference.ndim != 5 or reference.shape[1] != len(values):
        raise ValueError(f"channel values do not match tensor shape {list(reference.shape)}")
    return torch.tensor(values, dtype=torch.float32, device=reference.device).view(1, -1, 1, 1, 1)


def _deterministic_imagenet_input(
    num_frames: int = INPUT_SHAPE[2], height: int = INPUT_SHAPE[3], width: int = INPUT_SHAPE[4]
) -> Any:
    import torch

    channels = torch.arange(3, dtype=torch.float32).view(1, 3, 1, 1, 1)
    frames = torch.arange(num_frames, dtype=torch.float32).view(1, 1, num_frames, 1, 1)
    rows = torch.arange(height, dtype=torch.float32).view(1, 1, 1, height, 1)
    columns = torch.arange(width, dtype=torch.float32).view(1, 1, 1, 1, width)
    pixels = (channels * 29.0 + frames * 17.0 + rows * 7.0 + columns * 3.0).remainder(256.0).div(255.0)
    pixel_mean = _channel_values(pixels, IMAGENET_MEAN)
    pixel_std = _channel_values(pixels, IMAGENET_STD)
    return ((pixels - pixel_mean) / pixel_std).contiguous()


def _deterministic_normalized_latent(
    num_frames: int = LATENT_SHAPE[2], height: int = LATENT_SHAPE[3], width: int = LATENT_SHAPE[4]
) -> Any:
    import torch

    channels = torch.arange(24, dtype=torch.float32).view(1, 24, 1, 1, 1)
    frames = torch.arange(num_frames, dtype=torch.float32).view(1, 1, num_frames, 1, 1)
    rows = torch.arange(height, dtype=torch.float32).view(1, 1, 1, height, 1)
    columns = torch.arange(width, dtype=torch.float32).view(1, 1, 1, 1, width)
    values = channels * 53.0 + frames * 29.0 + rows * 11.0 + columns * 7.0
    return values.remainder(257.0).sub(128.0).div(128.0).contiguous()


def _deterministic_epsilon(shape: tuple[int, ...], seed: int = ENCODE_SEED) -> Any:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu").contiguous()


def _posterior_math(moments: Any, epsilon: Any) -> dict[str, Any]:
    import torch

    if moments.dtype != torch.float32 or epsilon.dtype != torch.float32:
        raise ValueError("posterior moments and explicit epsilon must be float32")
    if moments.ndim != 5 or moments.shape[1] % 2 != 0:
        raise ValueError(f"invalid posterior moments shape: {list(moments.shape)}")
    mean, raw_logvar = moments.chunk(2, dim=1)
    if epsilon.shape != mean.shape:
        raise ValueError(f"epsilon shape {list(epsilon.shape)} does not match posterior mean {list(mean.shape)}")
    logvar = raw_logvar.clamp(POSTERIOR_LOGVAR_MIN, POSTERIOR_LOGVAR_MAX)
    std = torch.exp(0.5 * logvar)
    return {
        "mean": mean,
        "raw_logvar": raw_logvar,
        "logvar": logvar,
        "std": std,
        "sample": mean + std * epsilon,
    }


def _normalize_latents(latents: Any, latents_mean: tuple[float, ...], latents_std: tuple[float, ...]) -> Any:
    value = latents.float()
    mean = _channel_values(value, latents_mean)
    std = _channel_values(value, latents_std)
    if bool((std <= 0).any().item()):
        raise ValueError("latent standard deviations must be positive")
    return ((value - mean) / std).contiguous()


def _denormalize_latents(latents: Any, latents_mean: tuple[float, ...], latents_std: tuple[float, ...]) -> Any:
    value = latents.float()
    mean = _channel_values(value, latents_mean)
    std = _channel_values(value, latents_std)
    if bool((std <= 0).any().item()):
        raise ValueError("latent standard deviations must be positive")
    return (value * std + mean).contiguous()


def _temporal_plan(
    pixel_frames: int = INPUT_SHAPE[2],
    latent_frames: int = LATENT_SHAPE[2],
    clip_length: int = CLIP_LENGTH,
    temporal_ratio: int = TEMPORAL_COMPRESSION_RATIO,
    token_drop: int = TOKEN_DROP,
) -> dict[str, Any]:
    if min(pixel_frames, latent_frames, clip_length, temporal_ratio) <= 0 or token_drop < 0:
        raise ValueError("temporal geometry values must be positive and token_drop must be nonnegative")
    padded_pixel_frames = math.ceil(pixel_frames / clip_length) * clip_length
    encoder_chunks = padded_pixel_frames // clip_length
    tokens_chunk_size = math.ceil(clip_length / temporal_ratio)
    encoded_frames = encoder_chunks * tokens_chunk_size - token_drop
    if encoded_frames != latent_frames:
        raise ValueError(f"encoder geometry maps {pixel_frames} pixel frames to {encoded_frames}, not {latent_frames}")

    encoder_clip_plan = []
    for index in range(encoder_chunks):
        start = index * clip_length
        stop = (index + 1) * clip_length
        encoder_clip_plan.append(
            {
                "index": index,
                "padded_input_range": [start, stop],
                "source_input_range": [min(start, pixel_frames), min(stop, pixel_frames)],
                "repeated_tail_frames": max(stop - max(start, pixel_frames), 0),
                "moments_token_range": [index * tokens_chunk_size, (index + 1) * tokens_chunk_size],
            }
        )

    frame_pre_padding = (-clip_length) % temporal_ratio
    token_overlap = (-token_drop) % tokens_chunk_size
    frame_overlap = max(token_overlap * temporal_ratio - frame_pre_padding, 0)
    num_tokens = latent_frames + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk_size
    decoder_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
    padded_latent_frames = latent_frames + pad_tokens
    decoder_clip_plan = []
    for index in range(decoder_chunks):
        start = index * tokens_chunk_size
        stop = start + tokens_chunk_size + token_overlap
        decoder_clip_plan.append(
            {
                "index": index,
                "latent_input_range": [start, stop],
                "decoded_frames_before_trim": (tokens_chunk_size + token_overlap) * temporal_ratio,
                "primary_frames_after_pre_padding": clip_length,
                "trailing_overlap_frames": frame_overlap,
            }
        )

    pad_frames = 0
    if pad_tokens > 0:
        intra_tail = clip_length % temporal_ratio
        for offset in range(pad_tokens):
            is_chunk_tail = (padded_latent_frames - pad_tokens + offset) % tokens_chunk_size == 0
            pad_frames += intra_tail if intra_tail and is_chunk_tail else temporal_ratio
    decoded_frames = decoder_chunks * clip_length + int(token_drop > 0) * frame_overlap - pad_frames
    if decoded_frames != pixel_frames:
        raise ValueError(f"decoder geometry maps {latent_frames} latent frames to {decoded_frames}, not {pixel_frames}")

    return {
        "encoder": {
            "input_frames": pixel_frames,
            "clip_length": clip_length,
            "temporal_compression_ratio": temporal_ratio,
            "padded_input_frames": padded_pixel_frames,
            "repeated_tail_frames": padded_pixel_frames - pixel_frames,
            "chunk_count": encoder_chunks,
            "clip_plan": encoder_clip_plan,
            "tokens_per_clip_before_drop": tokens_chunk_size,
            "concatenated_moments_frames": encoder_chunks * tokens_chunk_size,
            "trailing_tokens_dropped": token_drop,
            "output_latent_frames": encoded_frames,
        },
        "decoder": {
            "input_latent_frames": latent_frames,
            "temporal_compression_ratio": temporal_ratio,
            "tokens_chunk_size": tokens_chunk_size,
            "token_overlap": token_overlap,
            "frame_pre_padding": frame_pre_padding,
            "frame_overlap": frame_overlap,
            "pad_tokens": pad_tokens,
            "padded_latent_frames": padded_latent_frames,
            "chunk_count": decoder_chunks,
            "clip_plan": decoder_clip_plan,
            "trailing_pad_frames_removed": pad_frames,
            "output_frames": decoded_frames,
        },
    }


def _tile_plan(
    length: int = INPUT_SHAPE[3],
    tile_size: int = TILE_SIZE,
    min_overlap: int = TILE_MIN_OVERLAP,
    spatial_ratio: int = SPATIAL_COMPRESSION_RATIO,
) -> dict[str, Any]:
    if min(length, tile_size, spatial_ratio) <= 0 or min_overlap < 0 or min_overlap >= tile_size:
        raise ValueError("invalid spatial tile geometry")
    if length % spatial_ratio != 0 or tile_size % spatial_ratio != 0 or min_overlap % spatial_ratio != 0:
        raise ValueError("spatial tile geometry must be latent aligned")
    if tile_size >= length:
        starts, lengths, overlaps = [0], [length], []
    else:
        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) < length:
            num_tiles += 1
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        if remaining % spatial_ratio != 0:
            raise ValueError("tile slack cannot be distributed in latent-aligned steps")
        for index in range(remaining // spatial_ratio):
            overlaps[index % (num_tiles - 1)] += spatial_ratio
        starts = [0]
        for index, overlap in enumerate(overlaps):
            starts.append(starts[index] + tile_size - overlap)
        lengths = [tile_size] * num_tiles
    if starts[-1] + lengths[-1] != length:
        raise ValueError("tile plan does not cover the axis exactly")
    return {
        "sample_length": length,
        "sample_starts": starts,
        "sample_lengths": lengths,
        "sample_overlaps": overlaps,
        "spatial_compression_ratio": spatial_ratio,
        "latent_length": length // spatial_ratio,
        "latent_starts": [start // spatial_ratio for start in starts],
        "latent_lengths": [tile_length // spatial_ratio for tile_length in lengths],
        "latent_overlaps": [overlap // spatial_ratio for overlap in overlaps],
    }


def _fixture_plan() -> dict[str, Any]:
    temporal = _temporal_plan()
    height = _tile_plan(INPUT_SHAPE[3])
    width = _tile_plan(INPUT_SHAPE[4])
    tiles_per_clip = len(height["sample_starts"]) * len(width["sample_starts"])
    tile_order = []
    for row, (sample_y, latent_y) in enumerate(zip(height["sample_starts"], height["latent_starts"])):
        for column, (sample_x, latent_x) in enumerate(zip(width["sample_starts"], width["latent_starts"])):
            tile_order.append(
                {
                    "tile_in_clip": len(tile_order),
                    "row": row,
                    "column": column,
                    "sample_y_range": [sample_y, sample_y + height["sample_lengths"][row]],
                    "sample_x_range": [sample_x, sample_x + width["sample_lengths"][column]],
                    "latent_y_range": [latent_y, latent_y + height["latent_lengths"][row]],
                    "latent_x_range": [latent_x, latent_x + width["latent_lengths"][column]],
                }
            )

    encoder_call_order = []
    for temporal_chunk in range(temporal["encoder"]["chunk_count"]):
        for tile in tile_order:
            encoder_call_order.append({"call": len(encoder_call_order), "temporal_chunk": temporal_chunk, **tile})
    decoder_call_order = []
    for temporal_chunk in range(temporal["decoder"]["chunk_count"]):
        for tile in tile_order:
            decoder_call_order.append({"call": len(decoder_call_order), "temporal_chunk": temporal_chunk, **tile})
    return {
        "id": "h3-c6a",
        "input_shape": list(INPUT_SHAPE),
        "independent_decode_normalized_latent_shape": list(LATENT_SHAPE),
        "temporal": temporal,
        "spatial": {
            "tile_size": [TILE_SIZE, TILE_SIZE],
            "minimum_overlap": [TILE_MIN_OVERLAP, TILE_MIN_OVERLAP],
            "height": height,
            "width": width,
            "tiles_per_clip": tiles_per_clip,
            "tile_order_within_clip": tile_order,
        },
        "encoder_tile_call_order": encoder_call_order,
        "decoder_tile_call_order": decoder_call_order,
        "expected_calls": {
            "encoder_quant_conv": temporal["encoder"]["chunk_count"] * tiles_per_clip,
            "decoder_post_quant_conv": temporal["decoder"]["chunk_count"] * tiles_per_clip,
            "decoder_rope": temporal["decoder"]["chunk_count"] * tiles_per_clip,
            "decoder_block_0": temporal["decoder"]["chunk_count"] * tiles_per_clip,
            "decoder_block_35": temporal["decoder"]["chunk_count"] * tiles_per_clip,
        },
    }


def _to_cpu(tensor: Any) -> Any:
    return tensor.detach().to("cpu").contiguous()


def _register_capture_hooks(model: Any, archive: dict[str, Any]) -> tuple[list[Any], dict[str, int]]:
    counters = {
        "encoder_quant_conv": 0,
        "decoder_post_quant_conv": 0,
        "decoder_rope": 0,
        "decoder_block_0": 0,
        "decoder_block_35": 0,
    }
    if len(model.decoder.transformer_blocks) != 36:
        raise ValueError(f"C6a requires 36 decoder blocks, found {len(model.decoder.transformer_blocks)}")

    def _encoder_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        index = counters["encoder_quant_conv"]
        archive[f"encoder.tile_call_{index:02d}.moments"] = _to_cpu(output)
        counters["encoder_quant_conv"] += 1

    def _post_quant_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        index = counters["decoder_post_quant_conv"]
        archive[f"decoder.tile_call_{index:02d}.post_quant"] = _to_cpu(output)
        counters["decoder_post_quant_conv"] += 1

    def _rope_hook(_module: Any, inputs: tuple[Any, ...], output: tuple[Any, Any]) -> None:
        index = counters["decoder_rope"]
        archive[f"decoder.tile_call_{index:02d}.rope_position_ids"] = _to_cpu(inputs[0])
        archive[f"decoder.tile_call_{index:02d}.rope_cos"] = _to_cpu(output[0])
        archive[f"decoder.tile_call_{index:02d}.rope_sin"] = _to_cpu(output[1])
        counters["decoder_rope"] += 1

    def _block_0_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        index = counters["decoder_block_0"]
        archive[f"decoder.tile_call_{index:02d}.block_00_output"] = _to_cpu(output)
        counters["decoder_block_0"] += 1

    def _block_35_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        index = counters["decoder_block_35"]
        archive[f"decoder.tile_call_{index:02d}.block_35_output"] = _to_cpu(output)
        counters["decoder_block_35"] += 1

    handles = [
        model.quant_conv.register_forward_hook(_encoder_hook),
        model.post_quant_conv.register_forward_hook(_post_quant_hook),
        model.decoder.rope.register_forward_hook(_rope_hook),
        model.decoder.transformer_blocks[0].register_forward_hook(_block_0_hook),
        model.decoder.transformer_blocks[35].register_forward_hook(_block_35_hook),
    ]
    return handles, counters


def _validate_execution(archive: dict[str, Any], counters: dict[str, int], fixture: dict[str, Any]) -> None:
    expected_calls = fixture["expected_calls"]
    if counters != expected_calls:
        raise ValueError(f"fixture did not execute the required chunk/tile plan: {counters} != {expected_calls}")
    expected_shapes = {
        "input.imagenet_normalized": INPUT_SHAPE,
        "encoder.final_moments": (1, 48, *LATENT_SHAPE[2:]),
        "posterior.mean": LATENT_SHAPE,
        "posterior.logvar": LATENT_SHAPE,
        "posterior.epsilon": LATENT_SHAPE,
        "posterior.sample": LATENT_SHAPE,
        "posterior.rounded_fp32": LATENT_SHAPE,
        "posterior.normalized": LATENT_SHAPE,
        "decoder.independent_normalized_latent": LATENT_SHAPE,
        "decoder.denormalized_latent": LATENT_SHAPE,
        "decoder.raw": INPUT_SHAPE,
        "decoder.postprocessed": INPUT_SHAPE,
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(archive[name].shape)
        if actual_shape != expected_shape:
            raise ValueError(f"Golden tensor {name!r} has shape {actual_shape}, expected {expected_shape}")
    for name, tensor in archive.items():
        if tensor.is_floating_point() and not bool(tensor.isfinite().all().item()):
            raise ValueError(f"Golden tensor {name!r} contains NaN or Inf")


def _run_reference(
    model: Any, device: Any, fixture: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, float], dict[str, int]]:
    import torch

    archive: dict[str, Any] = {}
    handles, counters = _register_capture_hooks(model, archive)
    input_video = _deterministic_imagenet_input()
    archive["input.imagenet_normalized"] = input_video
    timings: dict[str, float] = {}
    try:
        encoder_started = time.perf_counter()
        posterior = model.encode(input_video.to(device), return_dict=False)[0]
        torch.npu.synchronize(device)
        timings["encoder"] = time.perf_counter() - encoder_started

        moments = posterior.parameters
        epsilon_cpu = _deterministic_epsilon(tuple(posterior.mean.shape))
        posterior_values = _posterior_math(moments, epsilon_cpu.to(device))
        if not torch.equal(posterior.mean, posterior_values["mean"]) or not torch.equal(
            posterior.logvar, posterior_values["logvar"]
        ):
            raise ValueError("explicit posterior clamp does not match pinned Diffusers")
        sampled = posterior_values["sample"]
        rounded_fp16 = sampled.to(torch.float16)
        rounded_fp32 = rounded_fp16.to(torch.float32)
        normalized = _normalize_latents(
            rounded_fp32,
            tuple(model.config.latents_mean),
            tuple(model.config.latents_std),
        )

        archive["encoder.final_moments"] = _to_cpu(moments)
        archive["posterior.mean"] = _to_cpu(posterior_values["mean"])
        archive["posterior.raw_logvar"] = _to_cpu(posterior_values["raw_logvar"])
        archive["posterior.logvar"] = _to_cpu(posterior_values["logvar"])
        archive["posterior.std"] = _to_cpu(posterior_values["std"])
        archive["posterior.epsilon"] = epsilon_cpu
        archive["posterior.sample"] = _to_cpu(sampled)
        archive["posterior.rounded_fp16"] = _to_cpu(rounded_fp16)
        archive["posterior.rounded_fp32"] = _to_cpu(rounded_fp32)
        archive["posterior.normalized"] = _to_cpu(normalized)

        normalized_decode_latent = _deterministic_normalized_latent()
        archive["decoder.independent_normalized_latent"] = normalized_decode_latent
        denormalized_decode_latent = _denormalize_latents(
            normalized_decode_latent.to(device),
            tuple(model.config.latents_mean),
            tuple(model.config.latents_std),
        )
        archive["decoder.denormalized_latent"] = _to_cpu(denormalized_decode_latent)

        decoder_started = time.perf_counter()
        with torch.autocast(device_type="npu", dtype=torch.float16):
            decoded = model.decode(denormalized_decode_latent, return_dict=False)[0]
        torch.npu.synchronize(device)
        timings["decoder"] = time.perf_counter() - decoder_started
        archive["decoder.raw"] = _to_cpu(decoded)

        postprocess_started = time.perf_counter()
        pixel_mean = _channel_values(decoded, IMAGENET_MEAN)
        pixel_std = _channel_values(decoded, IMAGENET_STD)
        postprocessed = (decoded.float() * pixel_std + pixel_mean).clamp(0.0, 1.0)
        archive["decoder.postprocessed"] = _to_cpu(postprocessed)
        timings["postprocess"] = time.perf_counter() - postprocess_started
    finally:
        for handle in handles:
            handle.remove()
    _validate_execution(archive, counters, fixture)
    return archive, timings, counters


def _tensor_digest(tensor: Any) -> str:
    import torch

    value = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(value.view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    value = tensor.detach().to("cpu").contiguous()
    finite = torch.isfinite(value) if value.is_floating_point() else torch.ones_like(value, dtype=torch.bool)
    finite_values = value[finite].float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "numel": value.numel(),
        "finite": bool(finite.all().item()),
        "min": float(finite_values.min().item()) if finite_values.numel() else None,
        "max": float(finite_values.max().item()) if finite_values.numel() else None,
        "mean": float(finite_values.mean().item()) if finite_values.numel() else None,
        "std": float(finite_values.std(correction=0).item()) if finite_values.numel() else None,
        "sha256": _tensor_digest(value),
    }


def _runtime_manifest(torch: Any, torch_npu: Any, diffusers: Any, device: Any) -> dict[str, Any]:
    runtime = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "diffusers": diffusers.__version__,
        "device": str(device),
        "device_count": torch.npu.device_count(),
        "device_name": torch.npu.get_device_name(device),
    }
    runtime["digest"] = _canonical_digest(runtime)
    return runtime


def _write_output(output_dir: Path, archive: dict[str, Any], manifest: dict[str, Any]) -> Path:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    archive_temp = output_dir / f".{ARCHIVE_NAME}{suffix}"
    manifest_temp = output_dir / f".{MANIFEST_NAME}{suffix}"
    try:
        save_file({name: value.contiguous() for name, value in sorted(archive.items())}, str(archive_temp))
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
    vae_path = args.vae_path.resolve()
    output_dir = args.output_dir.resolve()
    if not diffusers_source.is_dir() or not vae_path.is_dir():
        raise ValueError("pinned Diffusers source and converted VAE directories must exist")
    if not args.device.startswith("npu:"):
        raise ValueError("the H3-C6a Golden must run on one explicitly indexed NPU")

    source_started = time.perf_counter()
    source = _source_manifest(diffusers_source)
    source_seconds = time.perf_counter() - source_started
    _activate_diffusers_source(diffusers_source)

    import diffusers
    import torch
    import torch_npu
    from diffusers import AutoencoderKLMiniMaxH3

    if not torch.npu.is_available():
        raise RuntimeError("NPU runtime is unavailable")
    device = torch.device(args.device)
    torch.npu.set_device(device)
    torch.npu.empty_cache()
    checkpoint_started = time.perf_counter()
    checkpoint = _checkpoint_manifest(vae_path)
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    fixture = _fixture_plan()

    logger.info(f"Loading pinned MiniMax-H3 video VAE from {vae_path}")
    load_started = time.perf_counter()
    model = AutoencoderKLMiniMaxH3.from_pretrained(
        vae_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    model.enable_tiling(
        tile_sample_min_height=TILE_SIZE,
        tile_sample_min_width=TILE_SIZE,
        tile_sample_min_overlap_height=TILE_MIN_OVERLAP,
        tile_sample_min_overlap_width=TILE_MIN_OVERLAP,
    )
    model.to(device)
    torch.npu.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    torch.npu.reset_peak_memory_stats(device)

    logger.info("Running H3-C6a encoder and independent decoder Gates")
    run_started = time.perf_counter()
    with torch.inference_mode():
        archive, timings, counters = _run_reference(model, device, fixture)
    torch.npu.synchronize(device)
    run_seconds = time.perf_counter() - run_started

    timings.update(
        {
            "source_attestation": source_seconds,
            "checkpoint_attestation": checkpoint_seconds,
            "model_load_and_device_transfer": load_seconds,
            "reference_total": run_seconds,
        }
    )
    manifest = {
        "schema": "xllm.minimax_h3.video_vae_reference/v1",
        "status": "OFFICIAL_H3_C6A_VIDEO_VAE_GOLDEN",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "checkpoint": checkpoint,
        "runtime": _runtime_manifest(torch, torch_npu, diffusers, device),
        "fixture": fixture,
        "contracts": {
            "input": ("FP32 ImageNet-normalized analytic RGB: pixel=(29*c+17*t+7*y+3*x)%256/255"),
            "posterior": {
                "epsilon": f"explicit FP32 CPU torch.randn seed {ENCODE_SEED}",
                "logvar_clamp": [POSTERIOR_LOGVAR_MIN, POSTERIOR_LOGVAR_MAX],
                "sample": "mean + exp(0.5 * clamped_logvar) * epsilon",
                "rounding": "sample -> FP16 -> FP32 before per-channel normalization",
            },
            "decoder_gate": ("independent FP32 normalized latent; value=(53*c+29*t+11*y+7*x)%257/128-1"),
            "decoder_compute": "FP16 NPU autocast over pinned FP32 VAE weights",
            "postprocess": "raw.float32 * ImageNet std + ImageNet mean, clamped to [0,1]",
            "normalization": {
                "imagenet_mean": list(IMAGENET_MEAN),
                "imagenet_std": list(IMAGENET_STD),
                "latents_mean": list(model.config.latents_mean),
                "latents_std": list(model.config.latents_std),
            },
        },
        "execution": {
            "actual_calls": counters,
            "timings_seconds": timings,
            "peak_npu_memory_allocated": int(torch.npu.max_memory_allocated(device)),
            "peak_npu_memory_reserved": int(torch.npu.max_memory_reserved(device)),
            "final_npu_memory_allocated": int(torch.npu.memory_allocated(device)),
            "final_npu_memory_reserved": int(torch.npu.memory_reserved(device)),
        },
        "tensors": {name: _tensor_summary(tensor) for name, tensor in sorted(archive.items())},
    }
    manifest_path = _write_output(output_dir, archive, manifest)
    print(manifest_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("MiniMax-H3 C6a video VAE Golden generation failed")
        raise
