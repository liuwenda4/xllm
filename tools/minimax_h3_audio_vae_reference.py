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
"""Generate the pinned MiniMax-H3 C6b audio VAE Golden on one NPU."""

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
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger

PINNED_DIFFUSERS_REVISION = "d30c748f5f5d0925a5af14dc0e6a6de983025e63"
DEFAULT_DIFFUSERS_SOURCE = Path("/data/workspace/lwd/minimax/diffusers-reference")
DEFAULT_VAE_PATH = Path("/data/workspace/lwd/minimax/checkpoints/Ref2VA-diffusers-d30c748f/audio_vae")
ARCHIVE_NAME = "minimax_h3_audio_vae_reference.safetensors"
MANIFEST_NAME = "minimax_h3_audio_vae_reference.json"

INPUT_SAMPLE_RATE = 16000
VAE_SAMPLE_RATE = 32000
INPUT_SAMPLES = 3201
RESAMPLED_SAMPLES = 6402
HOP_LENGTH = 800
PADDED_SAMPLES = 7200
ENCODE_LATENT_FRAMES = 9
DECODE_LATENT_FRAMES = 8
PRODUCTION_LATENT_FRAMES = 207
LATENT_CHANNELS = 32
STEREO_CHANNELS = 2
POSTERIOR_SEED = 314159

INPUT_SHAPE = (STEREO_CHANNELS, INPUT_SAMPLES)
RESAMPLED_SHAPE = (STEREO_CHANNELS, RESAMPLED_SAMPLES)
NETWORK_INPUT_SHAPE = (STEREO_CHANNELS, 1, RESAMPLED_SAMPLES)
PADDED_INPUT_SHAPE = (STEREO_CHANNELS, 1, PADDED_SAMPLES)
POSTERIOR_SHAPE = (STEREO_CHANNELS, LATENT_CHANNELS, ENCODE_LATENT_FRAMES)
DECODE_LATENT_SHAPE = (STEREO_CHANNELS, LATENT_CHANNELS, DECODE_LATENT_FRAMES)
DECODE_OUTPUT_SHAPE = (STEREO_CHANNELS, 1, DECODE_LATENT_FRAMES * HOP_LENGTH)
PIPELINE_OUTPUT_SHAPE = (1, STEREO_CHANNELS, DECODE_LATENT_FRAMES * HOP_LENGTH)
PRODUCTION_LATENT_SHAPE = (STEREO_CHANNELS, LATENT_CHANNELS, PRODUCTION_LATENT_FRAMES)
PRODUCTION_PIPELINE_SHAPE = (1, STEREO_CHANNELS, PRODUCTION_LATENT_FRAMES * HOP_LENGTH)

DIFFUSERS_SOURCE_SHA256 = {
    "scripts/convert_minimax_h3_to_diffusers.py": "86f61f62934d1eccf2a76ec57b6d072e2c23767c6e49d96aa1cf24d40422b42c",
    "src/diffusers/models/attention.py": "3c61df6cc4832149eb654c1e82220f4a6b91daca13741c957c4e0faff7810adf",
    "src/diffusers/models/attention_dispatch.py": ("265dd891dc563578a3d808c9cc01faea893fbedaa71e4e3ed27e8f0196ce7d02"),
    "src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py": (
        "b56c5bb95a7dc99ace03df192544615c89ec01ac3b134a7f1dd7348b03731d57"
    ),
    "src/diffusers/modular_pipelines/minimax_h3/before_encoder.py": (
        "03612baa8b983d058884d2c1740e57342279a1124002553f78cec98abcfc7c28"
    ),
    "src/diffusers/modular_pipelines/minimax_h3/decoders.py": (
        "db553956502537613d17f83a5e1ac44f880b46514d254115676c0efe49ca0776"
    ),
    "src/diffusers/modular_pipelines/minimax_h3/encoders.py": (
        "fea751a889752ba58f1528acbc23b223e827ea707a75e2ae0c43d4a53ce758af"
    ),
}

CHECKPOINT_SHA256 = {
    "config.json": "0867acbc29fb725e8b1a58d9ad1454792a31d23c67402febbfa080f8b073cbfc",
    "diffusion_pytorch_model.safetensors": "52c59e67ba8de5477c81bfbced0327aabf500f1bfdeefd5ee754529241cb26cb",
}
EXPECTED_CHECKPOINT_TENSORS = 1087


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
    from safetensors import safe_open

    checkpoint_root = vae_path.resolve()
    logger.info(f"Hashing {len(CHECKPOINT_SHA256)} converted audio VAE files")
    files = {}
    for name, expected_sha256 in sorted(CHECKPOINT_SHA256.items()):
        path = checkpoint_root / name
        if not path.is_file():
            raise ValueError(f"converted audio VAE file does not exist: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"checkpoint digest mismatch for {path}: {actual_sha256}")
        files[name] = {"size": path.stat().st_size, "sha256": actual_sha256}

    with (checkpoint_root / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    expected_config = {
        "_class_name": "AutoencoderKLMiniMaxH3Audio",
        "sampling_rate": VAE_SAMPLE_RATE,
        "latent_channels": LATENT_CHANNELS,
        "latent_dim": 2048,
        "encoder_rates": [2, 4, 4, 5, 5],
        "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"converted audio VAE config {key!r} is {config.get(key)!r}, expected {expected!r}")
    if len(config.get("latents_mean", ())) != LATENT_CHANNELS or len(config.get("latents_std", ())) != LATENT_CHANNELS:
        raise ValueError("converted audio VAE config must contain 32 latent means and 32 latent standard deviations")
    if any(value <= 0 for value in config["latents_std"]):
        raise ValueError("converted audio VAE latent standard deviations must be positive")

    weights_path = checkpoint_root / "diffusion_pytorch_model.safetensors"
    tensor_data_bytes = 0
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            tensor_slice = handle.get_slice(key)
            if tensor_slice.get_dtype() != "F32":
                raise ValueError(f"converted audio VAE tensor {key!r} is not FP32")
            tensor_data_bytes += math.prod(tensor_slice.get_shape()) * 4
    if len(keys) != EXPECTED_CHECKPOINT_TENSORS:
        raise ValueError(f"converted audio VAE has {len(keys)} tensors, expected {EXPECTED_CHECKPOINT_TENSORS}")

    checkpoint = {
        "path": str(checkpoint_root),
        "format": "diffusers_safetensors",
        "tensor_count": len(keys),
        "tensor_dtype": "float32",
        "tensor_data_bytes": tensor_data_bytes,
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
    sys.path[:] = [entry for entry in sys.path if entry != source_value]
    sys.path.insert(0, source_value)


def _deterministic_stereo_waveform(num_samples: int = INPUT_SAMPLES, sample_rate: int = INPUT_SAMPLE_RATE) -> Any:
    import torch

    if num_samples <= 0 or sample_rate <= 0:
        raise ValueError("waveform sample count and sample rate must be positive")
    time_axis = torch.arange(num_samples, dtype=torch.float32) / float(sample_rate)
    left = 0.55 * torch.sin(2.0 * math.pi * 233.0 * time_axis)
    left += 0.20 * torch.cos(2.0 * math.pi * 617.0 * time_axis + 0.25)
    right = 0.45 * torch.cos(2.0 * math.pi * 347.0 * time_axis)
    right += 0.25 * torch.sin(2.0 * math.pi * 109.0 * time_axis + 0.50)
    return torch.stack((left, right)).contiguous()


def _official_audio_preprocess(
    waveform: Any,
    sample_rate: int,
    target_sample_rate: int,
    max_duration: float,
    resample_factory: Callable[[int, int], Callable[[Any], Any]] | None = None,
) -> Any:
    """Mirror MiniMax-H3 setup: native-rate truncation, mono duplication, then one resample."""
    import torch

    if sample_rate <= 0 or target_sample_rate <= 0 or max_duration < 0:
        raise ValueError("sample rates must be positive and max_duration must be nonnegative")
    value = torch.as_tensor(waveform)
    if value.ndim != 2 or value.shape[0] not in (1, 2):
        raise ValueError(
            f"a reference soundtrack must be a [channels, samples] mono or stereo waveform, got {tuple(value.shape)}"
        )
    value = value.to(torch.float32)[:, : int(max_duration * sample_rate)]
    if value.shape[0] != STEREO_CHANNELS:
        value = value.expand(STEREO_CHANNELS, -1).contiguous()
    if sample_rate == target_sample_rate:
        return value

    if resample_factory is None:
        try:
            import torchaudio
        except ImportError as error:
            raise ImportError(
                f"resampling the C6b waveform from {sample_rate} Hz to {target_sample_rate} Hz needs torchaudio"
            ) from error
        resample_factory = lambda source, target: torchaudio.transforms.Resample(source, target)
    resampled = resample_factory(sample_rate, target_sample_rate)(value)
    return resampled.to(torch.float32).contiguous()


def _reshape_mono_batch_and_pad(waveform: Any, hop_length: int = HOP_LENGTH) -> tuple[Any, Any, int]:
    import torch.nn.functional as torch_functional

    if waveform.ndim != 2 or waveform.shape[0] != STEREO_CHANNELS:
        raise ValueError(f"stereo waveform must have shape [2, samples], got {tuple(waveform.shape)}")
    if hop_length <= 0:
        raise ValueError("hop length must be positive")
    network_input = waveform[:, None, :].contiguous()
    right_pad = math.ceil(network_input.shape[-1] / hop_length) * hop_length - network_input.shape[-1]
    padded = torch_functional.pad(network_input, (0, right_pad)) if right_pad else network_input
    return network_input, padded.contiguous(), right_pad


def _deterministic_epsilon(shape: tuple[int, ...], seed: int = POSTERIOR_SEED) -> Any:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu").contiguous()


def _posterior_math(mean: Any, logs: Any, epsilon: Any) -> dict[str, Any]:
    import torch

    if mean.dtype != torch.float32 or logs.dtype != torch.float32 or epsilon.dtype != torch.float32:
        raise ValueError("posterior mean, log standard deviation, and epsilon must be float32")
    if mean.ndim != 3 or logs.shape != mean.shape or epsilon.shape != mean.shape:
        raise ValueError(
            f"posterior tensors must share a [batch, channels, frames] shape: "
            f"{tuple(mean.shape)}, {tuple(logs.shape)}, {tuple(epsilon.shape)}"
        )
    std = torch.exp(logs)
    return {
        "mean": mean,
        "logs": logs,
        "std": std,
        "mode": mean,
        "sample": mean + std * epsilon,
    }


def _channel_values(reference: Any, values: tuple[float, ...] | list[float]) -> Any:
    import torch

    if reference.ndim != 3 or reference.shape[1] != len(values):
        raise ValueError(f"channel values do not match tensor shape {list(reference.shape)}")
    return torch.tensor(values, dtype=torch.float32, device=reference.device).view(1, -1, 1)


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


def _channel_major_rows(latents: Any) -> Any:
    if latents.ndim != 3:
        raise ValueError(f"audio latents must have shape [batch, channels, frames], got {tuple(latents.shape)}")
    return latents.transpose(1, 2).reshape(-1, latents.shape[1]).contiguous()


def _deterministic_normalized_latent(num_frames: int = DECODE_LATENT_FRAMES) -> Any:
    import torch

    if num_frames <= 0:
        raise ValueError("latent frame count must be positive")
    batches = torch.arange(STEREO_CHANNELS, dtype=torch.float32).view(STEREO_CHANNELS, 1, 1)
    channels = torch.arange(LATENT_CHANNELS, dtype=torch.float32).view(1, LATENT_CHANNELS, 1)
    frames = torch.arange(num_frames, dtype=torch.float32).view(1, 1, num_frames)
    values = batches * 97.0 + channels * 53.0 + frames * 29.0
    return values.remainder(257.0).sub(128.0).div(128.0).contiguous()


def _audio_geometry(
    latent_frames: int, sample_rate: int = VAE_SAMPLE_RATE, hop_length: int = HOP_LENGTH
) -> dict[str, Any]:
    if latent_frames <= 0 or sample_rate <= 0 or hop_length <= 0:
        raise ValueError("audio geometry values must be positive")
    latent_rate = sample_rate / hop_length
    output_samples = latent_frames * hop_length
    return {
        "latent_frames": latent_frames,
        "sample_rate": sample_rate,
        "hop_length": hop_length,
        "latent_rate_hz": latent_rate,
        "output_samples": output_samples,
        "duration_seconds": output_samples / sample_rate,
    }


def _expected_call_counts() -> dict[str, int]:
    counts = {
        "preprocess.torchaudio_resample": 1,
        "encoder.input": 1,
        "pre_block.qkv_capture_recompute": 1,
        "pre_block.projection_branch": 1,
        "pre_block.attention_branch": 1,
        "pre_block.mlp_branch": 1,
        "pre_block.final": 1,
        "posterior.mean_head": 1,
        "posterior.logs_head": 1,
        "decoder.dec_in": 1,
        "decoder.conv_pre": 1,
        "decoder.final_pre_clamp": 1,
        "decoder.final_clamped": 1,
    }
    for index in range(8):
        counts[f"encoder.block_{index}"] = 1
    for stage in range(7):
        counts[f"decoder.upsample_{stage}"] = 1
        counts[f"decoder.stage_average_{stage}"] = 1
        for kernel in range(3):
            counts[f"decoder.resblock_{stage}_{kernel}"] = 1
    return counts


def _fixture_manifest() -> dict[str, Any]:
    return {
        "id": "h3-c6b",
        "input": {
            "shape": list(INPUT_SHAPE),
            "sample_rate": INPUT_SAMPLE_RATE,
            "max_duration_seconds": INPUT_SAMPLES / INPUT_SAMPLE_RATE,
            "native_rate_truncation_samples": INPUT_SAMPLES,
            "resampled_shape": list(RESAMPLED_SHAPE),
            "resampled_sample_rate": VAE_SAMPLE_RATE,
            "mono_network_batch_shape": list(NETWORK_INPUT_SHAPE),
            "right_pad_samples": PADDED_SAMPLES - RESAMPLED_SAMPLES,
            "padded_network_shape": list(PADDED_INPUT_SHAPE),
        },
        "encoder": {
            "posterior_shape": list(POSTERIOR_SHAPE),
            "posterior_parameterization": "mean_and_log_standard_deviation",
            "posterior_seed": POSTERIOR_SEED,
            "normalized_channel_major_rows_shape": [STEREO_CHANNELS * ENCODE_LATENT_FRAMES, LATENT_CHANNELS],
        },
        "decoder": {
            "normalized_latent_shape": list(DECODE_LATENT_SHAPE),
            "vae_output_shape": list(DECODE_OUTPUT_SHAPE),
            "pipeline_stereo_shape": list(PIPELINE_OUTPUT_SHAPE),
            "geometry": _audio_geometry(DECODE_LATENT_FRAMES),
        },
        "production_smoke": {
            "normalized_latent_shape": list(PRODUCTION_LATENT_SHAPE),
            "vae_output_shape": [STEREO_CHANNELS, 1, PRODUCTION_LATENT_FRAMES * HOP_LENGTH],
            "pipeline_stereo_shape": list(PRODUCTION_PIPELINE_SHAPE),
            "geometry": _audio_geometry(PRODUCTION_LATENT_FRAMES),
            "expected_calls": {"model.decode": 1},
            "archive_policy": "summaries_only",
        },
        "expected_calls": _expected_call_counts(),
    }


def _to_cpu(tensor: Any) -> Any:
    return tensor.detach().to("cpu").contiguous()


def _register_capture_hooks(model: Any, archive: dict[str, Any]) -> tuple[list[Any], dict[str, int]]:
    import torch
    import torch.nn.functional as torch_functional

    if len(model.encoder.block) != 8:
        raise ValueError(f"C6b requires 8 encoder entries, found {len(model.encoder.block)}")
    if model.decoder.num_upsamples != 7 or model.decoder.num_kernels != 3:
        raise ValueError(
            "C6b requires 7 decoder upsampling stages and 3 AMP kernels per stage, got "
            f"{model.decoder.num_upsamples} and {model.decoder.num_kernels}"
        )
    if len(model.decoder.resblocks) != 21:
        raise ValueError(f"C6b requires 21 decoder residual blocks, found {len(model.decoder.resblocks)}")

    counters = {name: 0 for name in _expected_call_counts()}

    def _capture(name: str, tensor: Any) -> None:
        if name in archive:
            raise ValueError(f"capture {name!r} executed more than once")
        archive[name] = _to_cpu(tensor)

    def _simple_hook(counter_name: str, archive_name: str) -> Callable[[Any, tuple[Any, ...], Any], None]:
        def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            _capture(archive_name, output)
            counters[counter_name] += 1

        return _hook

    def _encoder_input_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        _capture("encoder.input_padded", inputs[0])
        counters["encoder.input"] += 1

    def _attention_pre_hook(module: Any, inputs: tuple[Any, ...]) -> None:
        hidden_states = inputs[0]
        bias = torch.cat((module.q_bias, module.zero_k_bias, module.v_bias))
        qkv = torch_functional.linear(hidden_states, module.qkv.weight, bias)
        batch_size, seq_len, _ = qkv.shape
        query, key, value = (
            qkv.reshape(batch_size, seq_len, 3, module.num_heads, module.head_dim).permute(2, 0, 1, 3, 4).unbind(0)
        )
        _capture("pre_block.qkv", qkv)
        _capture("pre_block.query", query)
        _capture("pre_block.key", key)
        _capture("pre_block.value", value)
        counters["pre_block.qkv_capture_recompute"] += 1

    def _resblock_hook(stage: int, kernel: int) -> Callable[[Any, tuple[Any, ...], Any], None]:
        def _hook(_module: Any, _inputs: tuple[Any, ...], _output: Any) -> None:
            counters[f"decoder.resblock_{stage}_{kernel}"] += 1

        return _hook

    def _stage_average_pre_hook(stage: int) -> Callable[[Any, tuple[Any, ...]], None]:
        def _hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            _capture(f"decoder.stage_{stage}.average", inputs[0])
            counters[f"decoder.stage_average_{stage}"] += 1

        return _hook

    handles = [model.encoder.register_forward_pre_hook(_encoder_input_hook)]
    for index, block in enumerate(model.encoder.block):
        handles.append(
            block.register_forward_hook(_simple_hook(f"encoder.block_{index}", f"encoder.block_{index:02d}"))
        )
    handles += [
        model.pre_block.attn.register_forward_pre_hook(_attention_pre_hook),
        model.pre_block.proj.register_forward_hook(
            _simple_hook("pre_block.projection_branch", "pre_block.projection_branch")
        ),
        model.pre_block.attn.register_forward_hook(
            _simple_hook("pre_block.attention_branch", "pre_block.attention_branch")
        ),
        model.pre_block.mlp.register_forward_hook(_simple_hook("pre_block.mlp_branch", "pre_block.mlp_branch")),
        model.pre_block.register_forward_hook(_simple_hook("pre_block.final", "pre_block.final")),
        model.mean_proj.register_forward_hook(_simple_hook("posterior.mean_head", "posterior.mean_head")),
        model.logs_proj.register_forward_hook(_simple_hook("posterior.logs_head", "posterior.logs_head")),
        model.dec_in_proj.register_forward_hook(_simple_hook("decoder.dec_in", "decoder.dec_in")),
        model.decoder.conv_pre.register_forward_hook(_simple_hook("decoder.conv_pre", "decoder.conv_pre")),
    ]
    for stage, upsample in enumerate(model.decoder.ups):
        if stage > 0:
            handles.append(upsample[0].register_forward_pre_hook(_stage_average_pre_hook(stage - 1)))
        handles.append(
            upsample[0].register_forward_hook(
                _simple_hook(f"decoder.upsample_{stage}", f"decoder.stage_{stage}.upsample")
            )
        )
        for kernel in range(model.decoder.num_kernels):
            index = stage * model.decoder.num_kernels + kernel
            handles.append(model.decoder.resblocks[index].register_forward_hook(_resblock_hook(stage, kernel)))
    handles += [
        model.decoder.activation_post.register_forward_pre_hook(_stage_average_pre_hook(6)),
        model.decoder.conv_post.register_forward_hook(
            _simple_hook("decoder.final_pre_clamp", "decoder.final_pre_clamp")
        ),
        model.decoder.register_forward_hook(_simple_hook("decoder.final_clamped", "decoder.final_clamped")),
    ]
    return handles, counters


def _validate_execution(archive: dict[str, Any], counters: dict[str, int], fixture: dict[str, Any]) -> None:
    import torch

    if counters != fixture["expected_calls"]:
        raise ValueError(f"C6b did not execute the required calls: {counters} != {fixture['expected_calls']}")
    expected_shapes = {
        "input.stereo_16khz": INPUT_SHAPE,
        "preprocess.stereo_32khz": RESAMPLED_SHAPE,
        "preprocess.mono_network_batch": NETWORK_INPUT_SHAPE,
        "preprocess.padded_network_input": PADDED_INPUT_SHAPE,
        "encoder.input_padded": PADDED_INPUT_SHAPE,
        "encoder.block_00": (2, 64, 7200),
        "encoder.block_01": (2, 128, 3600),
        "encoder.block_02": (2, 256, 900),
        "encoder.block_03": (2, 512, 225),
        "encoder.block_04": (2, 1024, 45),
        "encoder.block_05": (2, 2048, 9),
        "encoder.block_06": (2, 2048, 9),
        "encoder.block_07": (2, 2048, 9),
        "pre_block.qkv": (2, 9, 6144),
        "pre_block.query": (2, 9, 8, 256),
        "pre_block.key": (2, 9, 8, 256),
        "pre_block.value": (2, 9, 8, 256),
        "pre_block.projection_branch": (2, 9, 32),
        "pre_block.attention_branch": (2, 9, 32),
        "pre_block.mlp_branch": (2, 9, 32),
        "pre_block.final": (2, 9, 32),
        "posterior.mean_head": POSTERIOR_SHAPE,
        "posterior.logs_head": POSTERIOR_SHAPE,
        "posterior.mean": POSTERIOR_SHAPE,
        "posterior.logs": POSTERIOR_SHAPE,
        "posterior.std": POSTERIOR_SHAPE,
        "posterior.mode": POSTERIOR_SHAPE,
        "posterior.epsilon": POSTERIOR_SHAPE,
        "posterior.sample": POSTERIOR_SHAPE,
        "posterior.normalized_mode": POSTERIOR_SHAPE,
        "posterior.normalized_channel_major_rows": (18, 32),
        "decoder.independent_normalized_latent": DECODE_LATENT_SHAPE,
        "decoder.denormalized_latent": DECODE_LATENT_SHAPE,
        "decoder.dec_in": (2, 2048, 8),
        "decoder.conv_pre": (2, 1024, 8),
        "decoder.final_pre_clamp": DECODE_OUTPUT_SHAPE,
        "decoder.final_clamped": DECODE_OUTPUT_SHAPE,
        "pipeline.stereo": PIPELINE_OUTPUT_SHAPE,
    }
    stage_shapes = [
        (2, 512, 40),
        (2, 256, 200),
        (2, 128, 400),
        (2, 64, 800),
        (2, 32, 1600),
        (2, 16, 3200),
        (2, 8, 6400),
    ]
    for stage, shape in enumerate(stage_shapes):
        expected_shapes[f"decoder.stage_{stage}.upsample"] = shape
        expected_shapes[f"decoder.stage_{stage}.average"] = shape
    if set(archive) != set(expected_shapes):
        missing = sorted(set(expected_shapes) - set(archive))
        extra = sorted(set(archive) - set(expected_shapes))
        raise ValueError(f"Golden tensor set mismatch; missing={missing}, extra={extra}")
    for name, expected_shape in expected_shapes.items():
        tensor = archive[name]
        actual_shape = tuple(tensor.shape)
        if actual_shape != expected_shape:
            raise ValueError(f"Golden tensor {name!r} has shape {actual_shape}, expected {expected_shape}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"Golden tensor {name!r} has dtype {tensor.dtype}, expected torch.float32")
        if not bool(tensor.isfinite().all().item()):
            raise ValueError(f"Golden tensor {name!r} contains NaN or Inf")
    if not torch.equal(archive["encoder.input_padded"], archive["preprocess.padded_network_input"]):
        raise ValueError("captured encoder input does not match explicit official right-padding")
    if not bool((archive["preprocess.padded_network_input"][..., RESAMPLED_SAMPLES:] == 0).all().item()):
        raise ValueError("encoder right-padding is not all zero")
    if not torch.equal(archive["posterior.mean"], archive["posterior.mode"]):
        raise ValueError("audio posterior mode is not exactly its mean")
    if not torch.equal(
        archive["posterior.normalized_channel_major_rows"],
        _channel_major_rows(archive["posterior.normalized_mode"]),
    ):
        raise ValueError("normalized posterior rows are not channel-major")
    if not torch.equal(archive["pipeline.stereo"], archive["decoder.final_clamped"].permute(1, 0, 2)):
        raise ValueError("pipeline stereo output does not match the official mono-batch permutation")


def _run_reference(
    model: Any, device: Any, fixture: dict[str, Any], torchaudio: Any
) -> tuple[dict[str, Any], dict[str, float], dict[str, int]]:
    import torch

    archive: dict[str, Any] = {}
    handles, counters = _register_capture_hooks(model, archive)
    timings: dict[str, float] = {}

    def _counted_resample_factory(source_rate: int, target_rate: int) -> Callable[[Any], Any]:
        transform = torchaudio.transforms.Resample(source_rate, target_rate)

        def _resample(value: Any) -> Any:
            counters["preprocess.torchaudio_resample"] += 1
            return transform(value)

        return _resample

    try:
        preprocess_started = time.perf_counter()
        input_waveform = _deterministic_stereo_waveform()
        resampled = _official_audio_preprocess(
            input_waveform,
            sample_rate=INPUT_SAMPLE_RATE,
            target_sample_rate=VAE_SAMPLE_RATE,
            max_duration=INPUT_SAMPLES / INPUT_SAMPLE_RATE,
            resample_factory=_counted_resample_factory,
        )
        network_input, padded_network_input, right_pad = _reshape_mono_batch_and_pad(resampled)
        if tuple(resampled.shape) != RESAMPLED_SHAPE:
            raise ValueError(f"torchaudio produced shape {tuple(resampled.shape)}, expected {RESAMPLED_SHAPE}")
        if right_pad != PADDED_SAMPLES - RESAMPLED_SAMPLES:
            raise ValueError(f"audio VAE right-padding is {right_pad}, expected 798")
        archive["input.stereo_16khz"] = input_waveform
        archive["preprocess.stereo_32khz"] = resampled
        archive["preprocess.mono_network_batch"] = network_input
        archive["preprocess.padded_network_input"] = padded_network_input
        timings["official_preprocess"] = time.perf_counter() - preprocess_started

        torch.npu.synchronize(device)
        encoder_started = time.perf_counter()
        posterior = model.encode(network_input.to(device), return_dict=False)[0]
        torch.npu.synchronize(device)
        timings["encoder_with_captures"] = time.perf_counter() - encoder_started

        posterior_started = time.perf_counter()
        epsilon_cpu = _deterministic_epsilon(tuple(posterior.mean.shape))
        posterior_values = _posterior_math(posterior.mean, posterior.logs, epsilon_cpu.to(device))
        if not torch.equal(posterior.mode(), posterior_values["mode"]):
            raise ValueError("explicit posterior mode does not match pinned Diffusers")
        if not torch.equal(posterior.std, posterior_values["std"]):
            raise ValueError("explicit exp(log_std) does not match pinned Diffusers")
        archive["posterior.mean"] = _to_cpu(posterior_values["mean"])
        archive["posterior.logs"] = _to_cpu(posterior_values["logs"])
        archive["posterior.std"] = _to_cpu(posterior_values["std"])
        archive["posterior.mode"] = _to_cpu(posterior_values["mode"])
        archive["posterior.epsilon"] = epsilon_cpu
        archive["posterior.sample"] = _to_cpu(posterior_values["sample"])
        torch.npu.synchronize(device)
        timings["posterior_explicit_sample"] = time.perf_counter() - posterior_started

        normalize_started = time.perf_counter()
        latents_mean = tuple(model.config.latents_mean)
        latents_std = tuple(model.config.latents_std)
        normalized_mode = _normalize_latents(archive["posterior.mode"], latents_mean, latents_std)
        archive["posterior.normalized_mode"] = normalized_mode
        archive["posterior.normalized_channel_major_rows"] = _channel_major_rows(normalized_mode)
        timings["pipeline_mode_normalization_and_row_pack"] = time.perf_counter() - normalize_started

        normalized_decode_latent = _deterministic_normalized_latent()
        archive["decoder.independent_normalized_latent"] = normalized_decode_latent
        denormalized_decode_latent = _denormalize_latents(
            normalized_decode_latent.to(device), latents_mean, latents_std
        )
        archive["decoder.denormalized_latent"] = _to_cpu(denormalized_decode_latent)

        torch.npu.synchronize(device)
        decoder_started = time.perf_counter()
        decoded = model.decode(denormalized_decode_latent, return_dict=False)[0]
        torch.npu.synchronize(device)
        timings["decoder_fp32_with_captures"] = time.perf_counter() - decoder_started
        if not torch.equal(archive["decoder.final_clamped"], _to_cpu(decoded)):
            raise ValueError("decoder return value does not match the captured final clamp")

        pipeline_started = time.perf_counter()
        archive["pipeline.stereo"] = _to_cpu(decoded.float().permute(1, 0, 2))
        timings["pipeline_stereo_permutation"] = time.perf_counter() - pipeline_started
    finally:
        for handle in handles:
            handle.remove()

    _validate_execution(archive, counters, fixture)
    return archive, timings, counters


def _run_production_smoke(model: Any, device: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    normalized = _deterministic_normalized_latent(PRODUCTION_LATENT_FRAMES)
    if tuple(normalized.shape) != PRODUCTION_LATENT_SHAPE:
        raise ValueError(f"production latent has shape {tuple(normalized.shape)}, expected {PRODUCTION_LATENT_SHAPE}")
    latents_mean = tuple(model.config.latents_mean)
    latents_std = tuple(model.config.latents_std)
    denormalized = _denormalize_latents(normalized.to(device), latents_mean, latents_std)
    actual_calls = {"model.decode": 0}

    def _count_decode(_module: Any, _inputs: tuple[Any, ...], _output: Any) -> None:
        actual_calls["model.decode"] += 1

    handle = model.decoder.register_forward_hook(_count_decode)
    torch.npu.synchronize(device)
    started = time.perf_counter()
    try:
        decoded = model.decode(denormalized, return_dict=False)[0]
        stereo = decoded.float().permute(1, 0, 2)
        torch.npu.synchronize(device)
    finally:
        handle.remove()
    elapsed = time.perf_counter() - started
    if actual_calls != {"model.decode": 1}:
        raise ValueError(f"production smoke call count mismatch: {actual_calls}")
    if tuple(stereo.shape) != PRODUCTION_PIPELINE_SHAPE:
        raise ValueError(f"production decode has shape {tuple(stereo.shape)}, expected {PRODUCTION_PIPELINE_SHAPE}")
    for name, tensor in (
        ("normalized latent", normalized),
        ("denormalized latent", denormalized),
        ("VAE output", decoded),
        ("pipeline stereo", stereo),
    ):
        if tensor.dtype != torch.float32:
            raise ValueError(f"production {name} has dtype {tensor.dtype}, expected torch.float32")
        if not bool(tensor.isfinite().all().item()):
            raise ValueError(f"production {name} contains NaN or Inf")
    summary = {
        "status": "PASS",
        "actual_calls": actual_calls,
        "timing_seconds": elapsed,
        "geometry": _audio_geometry(PRODUCTION_LATENT_FRAMES),
        "normalized_latent": _tensor_summary(normalized),
        "denormalized_latent": _tensor_summary(denormalized),
        "pipeline_stereo": _tensor_summary(stereo),
    }
    tensors = {
        "production.normalized_latent": _to_cpu(normalized),
        "production.pipeline_stereo": _to_cpu(stereo),
    }
    return summary, tensors


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


def _assert_model_fp32(model: Any) -> None:
    import torch

    bad_parameters = [name for name, value in model.named_parameters() if value.dtype != torch.float32]
    bad_buffers = [
        name for name, value in model.named_buffers() if value.is_floating_point() and value.dtype != torch.float32
    ]
    if bad_parameters or bad_buffers:
        raise ValueError(
            f"audio VAE is not wholly FP32; bad parameters={bad_parameters[:5]}, bad buffers={bad_buffers[:5]}"
        )


def _runtime_manifest(torch: Any, torch_npu: Any, torchaudio: Any, diffusers: Any, device: Any) -> dict[str, Any]:
    runtime = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "torchaudio": torchaudio.__version__,
        "diffusers": diffusers.__version__,
        "diffusers_path": str(Path(diffusers.__file__).resolve()),
        "device": str(device),
        "device_count": torch.npu.device_count(),
        "device_name": torch.npu.get_device_name(device),
        "attention_backend": "_native_math",
        "inference_mode": True,
        "autocast": False,
        "offload": False,
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
        save_file(
            {name: value.detach().to("cpu").contiguous().clone() for name, value in sorted(archive.items())},
            str(archive_temp),
        )
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
        raise ValueError("pinned Diffusers source and converted audio VAE directories must exist")

    source_started = time.perf_counter()
    source = _source_manifest(diffusers_source)
    source_seconds = time.perf_counter() - source_started
    _activate_diffusers_source(diffusers_source)

    import diffusers
    import torch
    import torch_npu

    try:
        import torchaudio
    except ImportError as error:
        raise ImportError("H3-C6b generation requires torchaudio for the official 16 kHz to 32 kHz resample") from error
    from diffusers import AutoencoderKLMiniMaxH3Audio

    expected_diffusers_root = (diffusers_source / "src").resolve()
    imported_diffusers_path = Path(diffusers.__file__).resolve()
    if expected_diffusers_root not in imported_diffusers_path.parents:
        raise RuntimeError(
            f"Diffusers imported from {imported_diffusers_path}, expected pinned source under {expected_diffusers_root}"
        )

    device = torch.device(args.device)
    if device.type != "npu" or device.index is None:
        raise ValueError("the H3-C6b Golden must run on one explicitly indexed NPU")
    if not torch.npu.is_available():
        raise RuntimeError("NPU runtime is unavailable")
    torch.npu.set_device(device)
    torch.npu.empty_cache()

    checkpoint_started = time.perf_counter()
    checkpoint = _checkpoint_manifest(vae_path)
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    fixture = _fixture_manifest()

    logger.info(f"Loading pinned MiniMax-H3 audio VAE from {vae_path}")
    load_started = time.perf_counter()
    model = AutoencoderKLMiniMaxH3Audio.from_pretrained(
        vae_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    model.pre_block.attn.set_attention_backend("_native_math")
    _assert_model_fp32(model)
    if model.hop_length != HOP_LENGTH:
        raise ValueError(f"audio VAE hop length is {model.hop_length}, expected {HOP_LENGTH}")
    if model.config.sampling_rate != VAE_SAMPLE_RATE:
        raise ValueError(f"audio VAE sampling rate is {model.config.sampling_rate}, expected {VAE_SAMPLE_RATE}")
    model.to(device)
    torch.npu.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    memory_after_load = {
        "allocated": int(torch.npu.memory_allocated(device)),
        "reserved": int(torch.npu.memory_reserved(device)),
    }
    torch.npu.reset_peak_memory_stats(device)

    logger.info("Running H3-C6b real FP32 encoder and independent decoder")
    run_started = time.perf_counter()
    with torch.inference_mode():
        archive, timings, counters = _run_reference(model, device, fixture, torchaudio)
        production_smoke, production_tensors = _run_production_smoke(model, device)
        archive.update(production_tensors)
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
        "schema": "xllm.minimax_h3.audio_vae_reference/v1",
        "status": "OFFICIAL_H3_C6B_AUDIO_VAE_GOLDEN",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "checkpoint": checkpoint,
        "runtime": _runtime_manifest(torch, torch_npu, torchaudio, diffusers, device),
        "fixture": fixture,
        "contracts": {
            "input": (
                "distinct FP32 stereo: left=.55*sin(2*pi*233*t)+.20*cos(2*pi*617*t+.25); "
                "right=.45*cos(2*pi*347*t)+.25*sin(2*pi*109*t+.50)"
            ),
            "preprocess": (
                "official native-rate [:int(max_duration*sample_rate)] truncation; mono expand before exactly one "
                "torchaudio.transforms.Resample; stereo channels become mono-network batch; encode right-pads to hop"
            ),
            "posterior": {
                "parameterization": "mean, log_std; std=exp(log_std) with no log-variance factor or clamp",
                "mode": "mean",
                "epsilon": f"explicit FP32 CPU torch.randn seed {POSTERIOR_SEED}",
                "sample": "mean + exp(log_std) * epsilon",
            },
            "normalization": {
                "formula": "(mode - channel_mean) / channel_std",
                "latents_mean": list(model.config.latents_mean),
                "latents_std": list(model.config.latents_std),
                "row_order": "channel-major stereo batch: all left frames, then all right frames",
            },
            "decoder_gate": "independent normalized FP32 latent: ((97*b+53*c+29*t)%257-128)/128",
            "decoder_compute": "real FP32 NPU inference; no autocast and no offload",
            "pipeline_stereo": "mono VAE [2,1,samples].float().permute(1,0,2) -> [1,2,samples]",
            "qkv_capture": (
                "exact F.linear recomputation in the attention forward pre-hook because the pinned processor invokes "
                "functional F.linear instead of the qkv module; counted separately from network module calls"
            ),
        },
        "execution": {
            "actual_calls": counters,
            "timings_seconds": timings,
            "memory_after_load": memory_after_load,
            "peak_npu_memory_allocated": int(torch.npu.max_memory_allocated(device)),
            "peak_npu_memory_reserved": int(torch.npu.max_memory_reserved(device)),
            "final_npu_memory_allocated": int(torch.npu.memory_allocated(device)),
            "final_npu_memory_reserved": int(torch.npu.memory_reserved(device)),
        },
        "production_smoke": production_smoke,
        "tensors": {name: _tensor_summary(tensor) for name, tensor in sorted(archive.items())},
    }
    manifest_path = _write_output(output_dir, archive, manifest)
    print(manifest_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("MiniMax-H3 C6b audio VAE Golden generation failed")
        raise
