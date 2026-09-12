# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.ndimage import convolve1d

PRODUCTION_DIR = Path("/data/workspace/lwd/minimax/artifacts/h3/20260912T121822Z-h3-c7/production")
DEFAULT_OFFICIAL_ARCHIVE = PRODUCTION_DIR / "minimax_h3_ref2va_production_official_hf_full.safetensors"
DEFAULT_OFFICIAL_MANIFEST = DEFAULT_OFFICIAL_ARCHIVE.with_suffix(".json")
DEFAULT_NATIVE_ARCHIVE = PRODUCTION_DIR / "minimax_h3_ref2va_production_xllm_native_full.safetensors"
DEFAULT_NATIVE_MANIFEST = DEFAULT_NATIVE_ARCHIVE.with_suffix(".json")
DEFAULT_OUTPUT_REPORT = PRODUCTION_DIR / "native_vs_official_media_report.json"

EXECUTION_SCHEMA = "xllm.minimax_h3.ref2va_production_execution/v1"
PREPARED_SCHEMA = "xllm.minimax_h3.ref2va_production_prepared/v1"
REPORT_SCHEMA = "xllm.minimax_h3.ref2va_media_comparison/v1"
EXECUTION_STATUS = "C7_49_FORWARD_REFERENCE_COMPLETE"
PINNED_SOURCE_DIGEST = "434923bdaa8b4022c46e2d5c6d7af7d745250fcf0bda502feb39f9fc66a55b08"
PINNED_CHECKPOINT_DIGEST = "092289ee588832268bb2ce82dbe15263802a8bf70954625d781fe89df47ea068"
SNAPSHOT_FORWARDS = (1, 2, 4, 8, 49)
VIDEO_SHAPE = (1, 3, 124, 768, 1344)
AUDIO_SHAPE = (1, 2, 165600)
VIDEO_FRAMES = 124
VIDEO_WIDTH = 1344
VIDEO_HEIGHT = 768
VIDEO_FPS = 24
AUDIO_CHANNELS = 2
AUDIO_SAMPLES = 165600
AUDIO_SAMPLE_RATE = 32000
SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5
SSIM_C1 = (0.01 * 255.0) ** 2
SSIM_C2 = (0.03 * 255.0) ** 2
STFT_FFT_SIZES = (512, 1024, 2048)

THRESHOLDS = {
    "declaration": "predeclared C7 media thresholds",
    "video": {
        "ssim_minimum": 0.97,
        "psnr_db_minimum": 34.0,
    },
    "audio": {
        "minimum_centered_pearson": 0.98,
        "minimum_stft_magnitude_spectral_cosine": 0.98,
        "rms_ratio_minimum": 0.95,
        "rms_ratio_maximum": 1.05,
        "stft_fft_sizes": list(STFT_FFT_SIZES),
        "stft_hop_divisor": 4,
        "stft_window": "periodic Hann",
        "stft_center": False,
    },
    "media_structure": {
        "video_codec": "h264",
        "video_frames": VIDEO_FRAMES,
        "video_width": VIDEO_WIDTH,
        "video_height": VIDEO_HEIGHT,
        "video_fps": VIDEO_FPS,
        "audio_codec": "aac",
        "audio_channels": AUDIO_CHANNELS,
        "audio_layout": "stereo",
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
        "stream_duration_tolerance": "one corresponding stream time-base tick",
        "decoded_aac_padding": "nonnegative and less than one 1024-sample AAC frame",
        "container_duration_tolerance_seconds": 1.0 / 1_000_000.0,
    },
}


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype


def _execution_tensor_specs() -> dict[str, TensorSpec]:
    specs: dict[str, TensorSpec] = {}
    for forward in SNAPSHOT_FORWARDS:
        specs[f"trajectory.forward_{forward:03d}.audio_rows"] = TensorSpec((414, 32), torch.float32)
        specs[f"trajectory.forward_{forward:03d}.video_rows"] = TensorSpec((48368, 96), torch.float32)
    specs.update(
        {
            "decoded.audio_float32": TensorSpec(AUDIO_SHAPE, torch.float32),
            "decoded.audio_sample_rate": TensorSpec((), torch.int64),
            "decoded.video_uint8": TensorSpec(VIDEO_SHAPE, torch.uint8),
            "final.target_audio_latent": TensorSpec((2, 32, 207), torch.float32),
            "final.target_audio_rows": TensorSpec((414, 32), torch.float32),
            "final.target_video_latent": TensorSpec((1, 24, 37, 48, 84), torch.float32),
            "final.target_video_rows": TensorSpec((37296, 96), torch.float32),
        }
    )
    return specs


EXECUTION_TENSOR_SPECS = _execution_tensor_specs()

EXPECTED_GEOMETRY = {
    "canvas_multiple": 32,
    "packed_order": ["text", "image", "target_audio", "target_video"],
    "patch_size": [1, 2, 2],
    "ranges": {
        "image": [11350, 22422],
        "target_audio": [22422, 22836],
        "target_video": [22836, 60132],
        "text": [0, 11350],
    },
    "reference_pixels_shape": [1, 3, 1, 2048, 5536],
    "reference_resized_size_wh": [5536, 2048],
    "reference_short_edge": 2048,
    "reference_source_size_wh": [3616, 1336],
    "row_counts": {
        "aligned": 60160,
        "image": 11072,
        "target_audio": 414,
        "target_video": 37296,
        "text": 11350,
        "used": 60132,
    },
    "target": {
        "audio_latent_shape": [2, 32, 207],
        "audio_rows_shape": [414, 32],
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
        "decoded_audio_shape": list(AUDIO_SHAPE),
        "decoded_video_shape": list(VIDEO_SHAPE),
        "fps": VIDEO_FPS,
        "height": VIDEO_HEIGHT,
        "num_frames": VIDEO_FRAMES,
        "video_latent_shape": [1, 24, 37, 48, 84],
        "width": VIDEO_WIDTH,
    },
    "visual_latent_shape": [1, 24, 1, 128, 346],
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare attested MiniMax-H3 C7 official and native production media")
    parser.add_argument("--official-archive", type=Path, default=DEFAULT_OFFICIAL_ARCHIVE)
    parser.add_argument("--official-manifest", type=Path, default=DEFAULT_OFFICIAL_MANIFEST)
    parser.add_argument("--native-archive", type=Path, default=DEFAULT_NATIVE_ARCHIVE)
    parser.add_argument("--native-manifest", type=Path, default=DEFAULT_NATIVE_MANIFEST)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    parser.add_argument("--video-frame-chunk", type=int, default=1)
    args = parser.parse_args(argv)
    if args.video_frame_chunk <= 0:
        parser.error("--video-frame-chunk must be positive")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{description} does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=lambda constant: (_ for _ in ()).throw(ValueError(constant)))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{description} is not strict JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _validate_canonical_section(section: Any, name: str) -> None:
    if not isinstance(section, dict) or not _is_sha256(section.get("digest")):
        raise ValueError(f"execution manifest {name} must carry a canonical SHA256 digest")
    body = dict(section)
    declared = body.pop("digest")
    if _canonical_digest(body) != declared:
        raise ValueError(f"execution manifest {name} canonical digest mismatch")


def _validate_tensor_declarations(declarations: Any) -> None:
    if not isinstance(declarations, dict) or set(declarations) != set(EXECUTION_TENSOR_SPECS):
        actual_keys = set(declarations) if isinstance(declarations, dict) else set()
        raise ValueError(
            "execution manifest tensor key mismatch: "
            f"missing={sorted(set(EXECUTION_TENSOR_SPECS) - actual_keys)}, "
            f"extra={sorted(actual_keys - set(EXECUTION_TENSOR_SPECS))}"
        )
    summary_keys = {"shape", "dtype", "numel", "finite", "min", "max", "mean", "std", "sha256"}
    for name, spec in EXECUTION_TENSOR_SPECS.items():
        summary = declarations[name]
        if not isinstance(summary, dict) or set(summary) != summary_keys:
            raise ValueError(f"execution manifest tensor {name!r} summary schema mismatch")
        expected_dtype = str(spec.dtype).removeprefix("torch.")
        expected_numel = math.prod(spec.shape)
        if summary["shape"] != list(spec.shape) or summary["dtype"] != expected_dtype:
            raise ValueError(f"execution manifest tensor {name!r} shape or dtype mismatch")
        if summary["numel"] != expected_numel or summary["finite"] is not True:
            raise ValueError(f"execution manifest tensor {name!r} count or finiteness mismatch")
        if not _is_sha256(summary["sha256"]):
            raise ValueError(f"execution manifest tensor {name!r} has an invalid SHA256")
        numeric_summary = [summary[field] for field in ("min", "max", "mean", "std")]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in numeric_summary):
            raise ValueError(f"execution manifest tensor {name!r} has invalid numeric summary fields")


def _validate_execution_manifest(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    archive_path: Path,
    expected_backend: str,
) -> Path:
    if manifest.get("schema") != EXECUTION_SCHEMA:
        raise ValueError(f"{expected_backend} execution manifest schema mismatch")
    if manifest.get("status") != EXECUTION_STATUS:
        raise ValueError(f"{expected_backend} execution manifest is not a completed full production run")
    if manifest.get("condition_backend") != expected_backend:
        raise ValueError(f"{expected_backend} execution manifest condition backend mismatch")

    execution = manifest.get("execution")
    expected_execution = {
        "mode": "full",
        "transformer_layers": 50,
        "transformer_forwards": 49,
        "transformer_block_forwards": 2450,
        "snapshot_forwards": list(SNAPSHOT_FORWARDS),
    }
    if not isinstance(execution, dict) or any(execution.get(key) != value for key, value in expected_execution.items()):
        raise ValueError(f"{expected_backend} execution manifest full-mode forward contract mismatch")
    timings = execution.get("timings_seconds") if isinstance(execution, dict) else None
    per_forward = timings.get("per_forward") if isinstance(timings, dict) else None
    if (
        not isinstance(per_forward, list)
        or len(per_forward) != 49
        or not all(type(value) in (int, float) and math.isfinite(value) and value >= 0.0 for value in per_forward)
    ):
        raise ValueError(f"{expected_backend} execution manifest does not attest 49 finite forward timings")

    provenance = manifest.get("backend_provenance")
    expected_key = f"condition.{expected_backend}.hidden"
    if not isinstance(provenance, dict) or provenance.get("tensor_key") != expected_key:
        raise ValueError(f"{expected_backend} execution manifest condition tensor provenance mismatch")
    if not _is_sha256(provenance.get("hidden_sha256")):
        raise ValueError(f"{expected_backend} execution manifest condition digest is invalid")

    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or set(artifact) != {"path", "size", "sha256"}:
        raise ValueError(f"{expected_backend} execution artifact declaration schema mismatch")
    if artifact["path"] != archive_path.name or type(artifact["size"]) is not int or artifact["size"] <= 0:
        raise ValueError(f"{expected_backend} execution artifact filename or size declaration mismatch")
    if not _is_sha256(artifact["sha256"]):
        raise ValueError(f"{expected_backend} execution artifact digest is invalid")

    prepared = manifest.get("prepared")
    prepared_fields = ("artifact_path", "artifact_sha256", "manifest_path", "manifest_sha256")
    if not isinstance(prepared, dict) or not all(field in prepared for field in prepared_fields):
        raise ValueError(f"{expected_backend} execution manifest prepared provenance is incomplete")
    if not _is_sha256(prepared["artifact_sha256"]) or not _is_sha256(prepared["manifest_sha256"]):
        raise ValueError(f"{expected_backend} execution manifest prepared provenance digest is invalid")
    if not all(isinstance(prepared[field], str) and prepared[field] for field in ("artifact_path", "manifest_path")):
        raise ValueError(f"{expected_backend} execution manifest prepared provenance path is invalid")

    _validate_canonical_section(manifest.get("source"), "source")
    _validate_canonical_section(manifest.get("checkpoint"), "checkpoint")
    if manifest["source"].get("digest") != PINNED_SOURCE_DIGEST:
        raise ValueError(f"{expected_backend} execution source is not the pinned C7 source")
    if manifest["checkpoint"].get("digest") != PINNED_CHECKPOINT_DIGEST:
        raise ValueError(f"{expected_backend} execution checkpoint is not the pinned C7 checkpoint")
    if manifest.get("geometry") != EXPECTED_GEOMETRY:
        raise ValueError(f"{expected_backend} execution manifest production geometry mismatch")
    _validate_tensor_declarations(manifest.get("tensors"))

    media = manifest.get("media")
    if not isinstance(media, dict) or media.get("requested") is not True or media.get("written") is not True:
        raise ValueError(f"{expected_backend} execution manifest does not attest requested and written media")
    expected_media_name = archive_path.with_suffix(".mp4").name
    if media.get("path") != expected_media_name or type(media.get("size")) is not int or media["size"] <= 0:
        raise ValueError(f"{expected_backend} execution media declaration mismatch")
    if not _is_sha256(media.get("sha256")):
        raise ValueError(f"{expected_backend} execution media digest is invalid")
    media_path = (manifest_path.parent / media["path"]).resolve()
    if not media_path.is_file():
        raise ValueError(f"{expected_backend} production media is absent: {media_path}")
    return media_path


def _validate_shared_provenance(official: Mapping[str, Any], native: Mapping[str, Any]) -> bool:
    official_prepared = official.get("prepared")
    native_prepared = native.get("prepared")
    if not isinstance(official_prepared, dict) or not isinstance(native_prepared, dict):
        raise ValueError("prepared provenance is absent")
    for field in ("artifact_sha256", "manifest_sha256"):
        if official_prepared.get(field) != native_prepared.get(field):
            raise ValueError(f"prepared {field.removesuffix('_sha256')} digest mismatch")
    for section in ("source", "checkpoint", "geometry"):
        if official.get(section) != native.get(section):
            raise ValueError(f"official and native {section} provenance mismatch")
    official_digest = official.get("backend_provenance", {}).get("hidden_sha256")
    native_digest = native.get("backend_provenance", {}).get("hidden_sha256")
    return official_digest != native_digest


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


def _attest_archive(path: Path, manifest: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    if not path.is_file():
        raise ValueError(f"production archive does not exist: {path}")
    actual_size = path.stat().st_size
    actual_sha256 = _sha256(path)
    artifact = manifest["artifact"]
    if actual_size != artifact["size"] or actual_sha256 != artifact["sha256"]:
        raise ValueError(f"production archive size or SHA256 mismatch: {path}")

    declarations = manifest["tensors"]
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(EXECUTION_TENSOR_SPECS):
            raise ValueError(f"production archive tensor keys mismatch: {path}")
        for name in sorted(EXECUTION_TENSOR_SPECS):
            value = handle.get_tensor(name).contiguous()
            if _tensor_summary(value) != declarations[name]:
                raise ValueError(f"production archive tensor {name!r} summary or SHA256 mismatch: {path}")
        if int(handle.get_tensor("decoded.audio_sample_rate").item()) != AUDIO_SAMPLE_RATE:
            raise ValueError(f"production archive decoded audio sample rate is not {AUDIO_SAMPLE_RATE}: {path}")

    return {
        "path": str(path),
        "size": actual_size,
        "sha256": actual_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "tensor_count": len(EXECUTION_TENSOR_SPECS),
        "tensor_summaries_sha256": _canonical_digest(declarations),
    }


def _resolve_declared_path(value: str, declaring_manifest_path: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = declaring_manifest_path.parent / path
    return path.resolve()


def _attest_prepared_source(
    official: Mapping[str, Any],
    official_manifest_path: Path,
    native: Mapping[str, Any],
    native_manifest_path: Path,
) -> dict[str, Any]:
    official_prepared = official["prepared"]
    native_prepared = native["prepared"]
    official_archive_path = _resolve_declared_path(official_prepared["artifact_path"], official_manifest_path)
    native_archive_path = _resolve_declared_path(native_prepared["artifact_path"], native_manifest_path)
    official_prepared_manifest_path = _resolve_declared_path(official_prepared["manifest_path"], official_manifest_path)
    native_prepared_manifest_path = _resolve_declared_path(native_prepared["manifest_path"], native_manifest_path)
    if official_archive_path != native_archive_path or official_prepared_manifest_path != native_prepared_manifest_path:
        raise ValueError("official and native executions do not reference the same prepared artifact pair")
    if not official_archive_path.is_file() or not official_prepared_manifest_path.is_file():
        raise ValueError("prepared artifact pair referenced by production executions is absent")

    archive_sha256 = _sha256(official_archive_path)
    manifest_sha256 = _sha256(official_prepared_manifest_path)
    if archive_sha256 != official_prepared["artifact_sha256"]:
        raise ValueError("prepared artifact SHA256 does not match production execution manifests")
    if manifest_sha256 != official_prepared["manifest_sha256"]:
        raise ValueError("prepared manifest SHA256 does not match production execution manifests")

    prepared_manifest = _load_json(official_prepared_manifest_path, "prepared manifest")
    if (
        prepared_manifest.get("schema") != PREPARED_SCHEMA
        or prepared_manifest.get("status") != "C7_PRODUCTION_PREPARED"
    ):
        raise ValueError("prepared manifest schema or status mismatch")
    artifact = prepared_manifest.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("path") != official_archive_path.name:
        raise ValueError("prepared manifest artifact declaration mismatch")
    if artifact.get("size") != official_archive_path.stat().st_size or artifact.get("sha256") != archive_sha256:
        raise ValueError("prepared artifact size or SHA256 does not match its prepared manifest")
    for section in ("source", "checkpoint", "geometry"):
        if prepared_manifest.get(section) != official.get(section):
            raise ValueError(f"prepared and execution {section} provenance mismatch")
    return {
        "archive_path": str(official_archive_path),
        "archive_size": official_archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "manifest_path": str(official_prepared_manifest_path),
        "manifest_size": official_prepared_manifest_path.stat().st_size,
        "manifest_sha256": manifest_sha256,
    }


class _MetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.reference_squared = 0.0
        self.candidate_squared = 0.0
        self.difference_squared = 0.0
        self.absolute_difference = 0.0
        self.dot = 0.0
        self.max_absolute = 0.0

    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        if reference.shape != candidate.shape:
            raise ValueError(f"metric tensor shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}")
        x = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        y = candidate.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if not bool(torch.isfinite(x).all().item()) or not bool(torch.isfinite(y).all().item()):
            raise ValueError("metric tensors must be finite")
        difference = y - x
        self.count += x.numel()
        self.reference_squared += float(torch.dot(x, x).item())
        self.candidate_squared += float(torch.dot(y, y).item())
        self.difference_squared += float(torch.dot(difference, difference).item())
        self.absolute_difference += float(difference.abs().sum().item())
        self.dot += float(torch.dot(x, y).item())
        if difference.numel():
            self.max_absolute = max(self.max_absolute, float(difference.abs().max().item()))

    def finish(self) -> dict[str, float | int]:
        if self.count == 0:
            raise ValueError("cannot compute metrics for empty tensors")
        if self.reference_squared == 0.0:
            relative_l2 = 0.0 if self.difference_squared == 0.0 else math.inf
        else:
            relative_l2 = math.sqrt(self.difference_squared / self.reference_squared)
        norm_product = math.sqrt(self.reference_squared * self.candidate_squared)
        if norm_product == 0.0:
            cosine = 1.0 if self.difference_squared == 0.0 else 0.0
        else:
            cosine = self.dot / norm_product
        return {
            "numel": self.count,
            "relative_l2": relative_l2,
            "cosine": max(-1.0, min(1.0, cosine)),
            "max_abs": self.max_absolute,
            "mae": self.absolute_difference / self.count,
            "rmse": math.sqrt(self.difference_squared / self.count),
        }


def _tensor_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    chunk_numel: int = 1_048_576,
) -> dict[str, float | int]:
    if reference.shape != candidate.shape:
        raise ValueError(f"metric tensor shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}")
    if chunk_numel <= 0:
        raise ValueError("metric chunk size must be positive")
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    accumulator = _MetricAccumulator()
    for offset in range(0, reference_flat.numel(), chunk_numel):
        accumulator.update(reference_flat[offset : offset + chunk_numel], candidate_flat[offset : offset + chunk_numel])
    return accumulator.finish()


def _archive_tensor_metrics(reference_handle: Any, candidate_handle: Any, name: str) -> dict[str, Any]:
    reference_slice = reference_handle.get_slice(name)
    candidate_slice = candidate_handle.get_slice(name)
    shape = tuple(reference_slice.get_shape())
    if shape != tuple(candidate_slice.get_shape()):
        raise ValueError(f"archive metric tensor {name!r} shape mismatch")
    accumulator = _MetricAccumulator()
    if not shape:
        accumulator.update(reference_handle.get_tensor(name), candidate_handle.get_tensor(name))
    else:
        row_numel = math.prod(shape[1:])
        rows_per_chunk = max(1, 1_048_576 // row_numel)
        for start in range(0, shape[0], rows_per_chunk):
            stop = min(shape[0], start + rows_per_chunk)
            accumulator.update(reference_slice[start:stop], candidate_slice[start:stop])
    return {"shape": list(shape), **accumulator.finish()}


def _gaussian_kernel() -> np.ndarray:
    radius = SSIM_WINDOW_SIZE // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(coordinates * coordinates) / (2.0 * SSIM_SIGMA * SSIM_SIGMA))
    return kernel / kernel.sum()


def _valid_gaussian_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = len(kernel) // 2
    horizontal = convolve1d(image, kernel, axis=1, mode="constant", cval=0.0)
    filtered = convolve1d(horizontal, kernel, axis=0, mode="constant", cval=0.0)
    return filtered[radius:-radius, radius:-radius]


def _channel_ssim(reference: torch.Tensor, candidate: torch.Tensor, kernel: np.ndarray) -> float:
    x = reference.detach().to("cpu").numpy().astype(np.float64, copy=False)
    y = candidate.detach().to("cpu").numpy().astype(np.float64, copy=False)
    mean_x = _valid_gaussian_filter(x, kernel)
    mean_y = _valid_gaussian_filter(y, kernel)
    variance_x = _valid_gaussian_filter(x * x, kernel) - mean_x * mean_x
    variance_y = _valid_gaussian_filter(y * y, kernel) - mean_y * mean_y
    covariance = _valid_gaussian_filter(x * y, kernel) - mean_x * mean_y
    variance_x = np.maximum(variance_x, 0.0)
    variance_y = np.maximum(variance_y, 0.0)
    numerator = (2.0 * mean_x * mean_y + SSIM_C1) * (2.0 * covariance + SSIM_C2)
    denominator = (mean_x * mean_x + mean_y * mean_y + SSIM_C1) * (variance_x + variance_y + SSIM_C2)
    return float(np.mean(numerator / denominator, dtype=np.float64))


def _psnr(mse: float) -> float:
    return math.inf if mse == 0.0 else 10.0 * math.log10((255.0 * 255.0) / mse)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty metric distribution")
    return {
        "min": min(values),
        "mean": math.fsum(values) / len(values),
        "max": max(values),
    }


class _VideoAccumulator:
    def __init__(self) -> None:
        self.squared_error = 0
        self.numel = 0
        self.per_frame_ssim: list[float] = []
        self.per_frame_psnr: list[float] = []
        self.kernel = _gaussian_kernel()

    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        if reference.shape != candidate.shape or reference.ndim != 5:
            raise ValueError("video metric chunks must have matching [1,C,F,H,W] shapes")
        if reference.shape[0] != 1 or reference.shape[1] != 3:
            raise ValueError("video metric chunks must contain one RGB batch")
        if reference.shape[-2] < SSIM_WINDOW_SIZE or reference.shape[-1] < SSIM_WINDOW_SIZE:
            raise ValueError("video frames are smaller than the 11x11 SSIM window")
        if reference.dtype != torch.uint8 or candidate.dtype != torch.uint8:
            raise ValueError("video metrics require raw uint8 tensors")
        for frame in range(reference.shape[2]):
            difference = reference[0, :, frame].to(torch.int32) - candidate[0, :, frame].to(torch.int32)
            frame_squared_error = int(difference.square().sum(dtype=torch.int64).item())
            frame_numel = difference.numel()
            self.squared_error += frame_squared_error
            self.numel += frame_numel
            self.per_frame_psnr.append(_psnr(frame_squared_error / frame_numel))
            channel_ssim = [
                _channel_ssim(reference[0, channel, frame], candidate[0, channel, frame], self.kernel)
                for channel in range(3)
            ]
            self.per_frame_ssim.append(math.fsum(channel_ssim) / 3.0)

    def finish(self, shape: Sequence[int]) -> dict[str, Any]:
        if self.numel == 0:
            raise ValueError("cannot compute video metrics for empty tensors")
        mse = self.squared_error / self.numel
        return {
            "shape": list(shape),
            "raw_uint8": {
                "all_frame_mse": mse,
                "all_frame_psnr_db": _psnr(mse),
                "per_frame_psnr_db": _distribution(self.per_frame_psnr),
            },
            "ssim": {
                "all_frame_mean": math.fsum(self.per_frame_ssim) / len(self.per_frame_ssim),
                "per_frame": _distribution(self.per_frame_ssim),
                "window": {
                    "size": [SSIM_WINDOW_SIZE, SSIM_WINDOW_SIZE],
                    "kernel": "Gaussian",
                    "sigma": SSIM_SIGMA,
                    "c1": SSIM_C1,
                    "c2": SSIM_C2,
                    "convolution": "valid",
                    "covariance": "Gaussian-weighted population covariance",
                    "aggregation": "mean over RGB channels, valid pixels, and frames",
                },
            },
        }


def _video_metrics(reference: torch.Tensor, candidate: torch.Tensor, *, frame_chunk: int = 1) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.ndim != 5:
        raise ValueError("video tensors must have matching [1,3,F,H,W] shapes")
    if frame_chunk <= 0:
        raise ValueError("video frame chunk must be positive")
    accumulator = _VideoAccumulator()
    for start in range(0, reference.shape[2], frame_chunk):
        stop = min(reference.shape[2], start + frame_chunk)
        accumulator.update(reference[:, :, start:stop], candidate[:, :, start:stop])
    return accumulator.finish(reference.shape)


def _archive_video_metrics(
    reference_handle: Any,
    candidate_handle: Any,
    *,
    frame_chunk: int,
) -> dict[str, Any]:
    if frame_chunk <= 0:
        raise ValueError("video frame chunk must be positive")
    name = "decoded.video_uint8"
    reference_slice = reference_handle.get_slice(name)
    candidate_slice = candidate_handle.get_slice(name)
    shape = tuple(reference_slice.get_shape())
    if shape != VIDEO_SHAPE or tuple(candidate_slice.get_shape()) != VIDEO_SHAPE:
        raise ValueError("decoded video archive shape mismatch")
    accumulator = _VideoAccumulator()
    for start in range(0, VIDEO_FRAMES, frame_chunk):
        stop = min(VIDEO_FRAMES, start + frame_chunk)
        accumulator.update(reference_slice[:, :, start:stop, :, :], candidate_slice[:, :, start:stop, :, :])
    return accumulator.finish(shape)


def _centered_pearson(reference: torch.Tensor, candidate: torch.Tensor, chunk_numel: int = 1_048_576) -> float:
    x = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    y = candidate.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if x.shape != y.shape or x.numel() == 0:
        raise ValueError("Pearson inputs must have matching nonempty shapes")
    mean_x = float(x.mean().item())
    mean_y = float(y.mean().item())
    covariance = 0.0
    variance_x = 0.0
    variance_y = 0.0
    for offset in range(0, x.numel(), chunk_numel):
        centered_x = x[offset : offset + chunk_numel] - mean_x
        centered_y = y[offset : offset + chunk_numel] - mean_y
        covariance += float(torch.dot(centered_x, centered_y).item())
        variance_x += float(torch.dot(centered_x, centered_x).item())
        variance_y += float(torch.dot(centered_y, centered_y).item())
    norm_product = math.sqrt(variance_x * variance_y)
    if norm_product == 0.0:
        if variance_x == 0.0 and variance_y == 0.0 and bool(torch.equal(x, y)):
            return 1.0
        return 0.0
    return max(-1.0, min(1.0, covariance / norm_product))


def _spectral_cosine(reference: torch.Tensor, candidate: torch.Tensor, n_fft: int) -> float:
    if reference.numel() < n_fft or candidate.numel() < n_fft:
        raise ValueError(f"audio is shorter than STFT FFT size {n_fft}")
    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
    hop_length = n_fft // 4
    reference_stft = torch.stft(
        reference.detach().to(device="cpu", dtype=torch.float64),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=False,
        return_complex=True,
    )
    candidate_stft = torch.stft(
        candidate.detach().to(device="cpu", dtype=torch.float64),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=False,
        return_complex=True,
    )
    return float(_tensor_metrics(reference_stft.abs(), candidate_stft.abs())["cosine"])


def _audio_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[0] != 1:
        raise ValueError("audio tensors must have matching [1,C,S] shapes")
    if reference.shape[1] != AUDIO_CHANNELS:
        raise ValueError("audio metrics require stereo tensors")
    if not reference.is_floating_point() or not candidate.is_floating_point():
        raise ValueError("audio metrics require floating-point waveform tensors")
    channel_names = ("left", "right")
    correlations = {
        channel_names[channel]: _centered_pearson(reference[0, channel], candidate[0, channel])
        for channel in range(AUDIO_CHANNELS)
    }
    rms_ratios = {}
    for channel, name in enumerate(channel_names):
        reference_rms = float(reference[0, channel].to(torch.float64).square().mean().sqrt().item())
        candidate_rms = float(candidate[0, channel].to(torch.float64).square().mean().sqrt().item())
        if reference_rms == 0.0:
            rms_ratios[name] = 1.0 if candidate_rms == 0.0 else math.inf
        else:
            rms_ratios[name] = candidate_rms / reference_rms
    spectral = {}
    for n_fft in STFT_FFT_SIZES:
        per_channel = {
            channel_names[channel]: _spectral_cosine(reference[0, channel], candidate[0, channel], n_fft)
            for channel in range(AUDIO_CHANNELS)
        }
        spectral[str(n_fft)] = {
            "hop_length": n_fft // 4,
            "per_channel": per_channel,
            "minimum": min(per_channel.values()),
        }
    return {
        "shape": list(reference.shape),
        "waveform": _tensor_metrics(reference, candidate),
        "centered_pearson": {
            "per_channel": correlations,
            "minimum": min(correlations.values()),
        },
        "rms_ratio_native_over_official": {
            "per_channel": rms_ratios,
            "min": min(rms_ratios.values()),
            "max": max(rms_ratios.values()),
        },
        "stft_magnitude_spectral_cosine": spectral,
    }


def _duration_seconds(stream: Any) -> float | None:
    if stream.duration is None or stream.time_base is None:
        return None
    return float(stream.duration * stream.time_base)


def _inspect_media(path: Path, declaration: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("MiniMax-H3 media comparison requires PyAV") from error

    actual_size = path.stat().st_size
    actual_sha256 = _sha256(path)
    if actual_size != declaration.get("size") or actual_sha256 != declaration.get("sha256"):
        raise ValueError(f"production media size or SHA256 mismatch: {path}")

    with av.open(str(path)) as container:
        video_streams = [stream for stream in container.streams if stream.type == "video"]
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        stream_count = len(container.streams)
        container_duration = None if container.duration is None else float(container.duration / av.time_base)
        if len(video_streams) != 1 or len(audio_streams) != 1:
            return {
                "path": str(path),
                "size": actual_size,
                "sha256": actual_sha256,
                "checks": {"exactly_one_video_and_audio_stream": False, "exactly_two_streams": stream_count == 2},
                "passed": False,
            }
        video_stream = video_streams[0]
        audio_stream = audio_streams[0]
        video_duration = _duration_seconds(video_stream)
        audio_duration = _duration_seconds(audio_stream)
        video_time_base = None if video_stream.time_base is None else float(video_stream.time_base)
        audio_time_base = None if audio_stream.time_base is None else float(audio_stream.time_base)
        video_codec = video_stream.codec_context.name
        audio_codec = audio_stream.codec_context.name
        audio_layout = getattr(getattr(audio_stream.codec_context, "layout", None), "name", None)
        video_rate = video_stream.average_rate
        declared_video_frames = video_stream.frames
        video_width = video_stream.codec_context.width
        video_height = video_stream.codec_context.height
        audio_channels = audio_stream.codec_context.channels
        audio_sample_rate = audio_stream.codec_context.sample_rate
        video_time_base_text = str(video_stream.time_base)
        audio_time_base_text = str(audio_stream.time_base)

    with av.open(str(path)) as container:
        decoded_video_frames = sum(1 for _ in container.decode(video=0))
    with av.open(str(path)) as container:
        decoded_audio_samples = sum(frame.samples for frame in container.decode(audio=0))

    expected_video_duration = VIDEO_FRAMES / VIDEO_FPS
    expected_audio_duration = AUDIO_SAMPLES / AUDIO_SAMPLE_RATE
    longest_source_duration = max(expected_video_duration, expected_audio_duration)
    decoded_audio_padding_samples = decoded_audio_samples - AUDIO_SAMPLES
    format_padding_seconds = None if container_duration is None else container_duration - longest_source_duration
    checks = {
        "exactly_one_video_and_audio_stream": True,
        "exactly_two_streams": stream_count == 2,
        "video_codec_h264": video_codec == "h264",
        "video_declared_frame_count_124": declared_video_frames == VIDEO_FRAMES,
        "video_decoded_frame_count_124": decoded_video_frames == VIDEO_FRAMES,
        "video_dimensions_1344x768": video_width == VIDEO_WIDTH and video_height == VIDEO_HEIGHT,
        "video_fps_24": video_rate is not None and Fraction(video_rate) == Fraction(VIDEO_FPS, 1),
        "audio_codec_aac": audio_codec == "aac",
        "audio_stereo": audio_channels == AUDIO_CHANNELS and audio_layout == "stereo",
        "audio_sample_rate_32000": audio_sample_rate == AUDIO_SAMPLE_RATE,
        "video_stream_duration": (
            video_duration is not None
            and video_time_base is not None
            and abs(video_duration - expected_video_duration) <= video_time_base
        ),
        "audio_stream_duration": (
            audio_duration is not None
            and audio_time_base is not None
            and abs(audio_duration - expected_audio_duration) <= audio_time_base
        ),
        "decoded_aac_padding_within_one_frame": 0 <= decoded_audio_padding_samples < 1024,
        "container_duration_matches_longest_source": (
            format_padding_seconds is not None
            and abs(format_padding_seconds) <= THRESHOLDS["media_structure"]["container_duration_tolerance_seconds"]
        ),
    }
    return {
        "path": str(path),
        "size": actual_size,
        "sha256": actual_sha256,
        "video": {
            "codec": video_codec,
            "declared_frames": declared_video_frames,
            "decoded_frames": decoded_video_frames,
            "width": video_width,
            "height": video_height,
            "fps": None if video_rate is None else float(video_rate),
            "time_base": video_time_base_text,
            "duration_tolerance_seconds": video_time_base,
            "stream_duration_seconds": video_duration,
            "expected_source_tensor_duration_seconds": expected_video_duration,
            "stream_duration_delta_seconds": (
                None if video_duration is None else video_duration - expected_video_duration
            ),
        },
        "audio": {
            "codec": audio_codec,
            "channels": audio_channels,
            "layout": audio_layout,
            "sample_rate": audio_sample_rate,
            "time_base": audio_time_base_text,
            "duration_tolerance_seconds": audio_time_base,
            "stream_duration_seconds": audio_duration,
            "expected_source_tensor_duration_seconds": expected_audio_duration,
            "stream_duration_delta_seconds": (
                None if audio_duration is None else audio_duration - expected_audio_duration
            ),
            "decoded_samples": decoded_audio_samples,
            "decoded_aac_padding_samples": decoded_audio_padding_samples,
            "decoded_aac_padding_seconds": decoded_audio_padding_samples / AUDIO_SAMPLE_RATE,
        },
        "container": {
            "duration_seconds": container_duration,
            "longest_source_tensor_duration_seconds": longest_source_duration,
            "duration_padding_seconds": format_padding_seconds,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _evaluate_gate(
    video: Mapping[str, Any],
    audio: Mapping[str, Any],
    media: Mapping[str, Mapping[str, Any]],
    condition_digest_differs: bool,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []

    def require(metric: str, actual: Any, passed: bool, requirement: str) -> None:
        if not passed:
            failures.append({"metric": metric, "actual": actual, "requirement": requirement})

    ssim = video["ssim"]["all_frame_mean"]
    psnr = video["raw_uint8"]["all_frame_psnr_db"]
    require("video.ssim", ssim, math.isfinite(ssim) and ssim >= THRESHOLDS["video"]["ssim_minimum"], ">= 0.97")
    require("video.psnr_db", psnr, psnr >= THRESHOLDS["video"]["psnr_db_minimum"], ">= 34.0")

    correlation = audio["centered_pearson"]["minimum"]
    require(
        "audio.minimum_centered_pearson",
        correlation,
        math.isfinite(correlation) and correlation >= THRESHOLDS["audio"]["minimum_centered_pearson"],
        ">= 0.98",
    )
    rms_min = audio["rms_ratio_native_over_official"]["min"]
    rms_max = audio["rms_ratio_native_over_official"]["max"]
    rms_minimum = THRESHOLDS["audio"]["rms_ratio_minimum"]
    rms_maximum = THRESHOLDS["audio"]["rms_ratio_maximum"]
    require("audio.rms_ratio_min", rms_min, math.isfinite(rms_min) and rms_min >= rms_minimum, f">= {rms_minimum}")
    require("audio.rms_ratio_max", rms_max, math.isfinite(rms_max) and rms_max <= rms_maximum, f"<= {rms_maximum}")
    spectral_minimum = THRESHOLDS["audio"]["minimum_stft_magnitude_spectral_cosine"]
    for n_fft in STFT_FFT_SIZES:
        spectral_cosine = audio["stft_magnitude_spectral_cosine"][str(n_fft)]["minimum"]
        require(
            f"audio.stft_{n_fft}_minimum_spectral_cosine",
            spectral_cosine,
            math.isfinite(spectral_cosine) and spectral_cosine >= spectral_minimum,
            f">= {spectral_minimum}",
        )
    expected_media_labels = {"official_hf", "xllm_native"}
    require(
        "media.required_inspections",
        sorted(media),
        set(media) == expected_media_labels,
        "exactly official_hf and xllm_native",
    )
    for label in sorted(expected_media_labels):
        inspection = media.get(label, {})
        for check, passed in inspection.get("checks", {}).items():
            require(f"media.{label}.{check}", passed, passed is True, "true")
        require(
            f"media.{label}.all_structure_checks", inspection.get("passed"), inspection.get("passed") is True, "true"
        )
    passed = not failures
    return {
        "state": (
            "NATIVE_MEDIA_PASS_NON_BIT_EXACT"
            if passed and condition_digest_differs
            else "NATIVE_MEDIA_PASS_BIT_EXACT"
            if passed
            else "NATIVE_MEDIA_FAIL"
        ),
        "passed": passed,
        "failed_metrics": failures,
    }


def _json_compatible(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("report contains NaN")
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _atomic_write_json(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temp_path.open("x", encoding="utf-8") as handle:
            json.dump(_json_compatible(report), handle, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    from safetensors import safe_open

    official_archive_path = args.official_archive.resolve()
    official_manifest_path = args.official_manifest.resolve()
    native_archive_path = args.native_archive.resolve()
    native_manifest_path = args.native_manifest.resolve()
    output_report_path = args.output_report.resolve()

    official_manifest = _load_json(official_manifest_path, "official execution manifest")
    native_manifest = _load_json(native_manifest_path, "native execution manifest")
    official_media_path = _validate_execution_manifest(
        official_manifest, official_manifest_path, official_archive_path, "official_hf"
    )
    native_media_path = _validate_execution_manifest(
        native_manifest, native_manifest_path, native_archive_path, "xllm_native"
    )
    condition_digest_differs = _validate_shared_provenance(official_manifest, native_manifest)
    official_attestation = _attest_archive(official_archive_path, official_manifest, official_manifest_path)
    native_attestation = _attest_archive(native_archive_path, native_manifest, native_manifest_path)
    prepared_attestation = _attest_prepared_source(
        official_manifest,
        official_manifest_path,
        native_manifest,
        native_manifest_path,
    )

    with ExitStack() as stack:
        official_handle = stack.enter_context(safe_open(str(official_archive_path), framework="pt", device="cpu"))
        native_handle = stack.enter_context(safe_open(str(native_archive_path), framework="pt", device="cpu"))
        trajectory = {}
        for forward in SNAPSHOT_FORWARDS:
            trajectory[f"forward_{forward:03d}"] = {
                media_type: _archive_tensor_metrics(
                    official_handle,
                    native_handle,
                    f"trajectory.forward_{forward:03d}.{media_type}_rows",
                )
                for media_type in ("video", "audio")
            }
        final_latents = {
            media_type: _archive_tensor_metrics(
                official_handle,
                native_handle,
                f"final.target_{media_type}_latent",
            )
            for media_type in ("video", "audio")
        }
        video = _archive_video_metrics(official_handle, native_handle, frame_chunk=args.video_frame_chunk)
        official_audio = official_handle.get_tensor("decoded.audio_float32").contiguous()
        native_audio = native_handle.get_tensor("decoded.audio_float32").contiguous()
        audio = _audio_metrics(official_audio, native_audio)

    media = {
        "official_hf": _inspect_media(official_media_path, official_manifest["media"]),
        "xllm_native": _inspect_media(native_media_path, native_manifest["media"]),
    }
    gate = _evaluate_gate(video, audio, media, condition_digest_differs)
    source_video_duration = Fraction(VIDEO_FRAMES, VIDEO_FPS)
    source_audio_duration = Fraction(AUDIO_SAMPLES, AUDIO_SAMPLE_RATE)
    source_offset = source_audio_duration - source_video_duration
    if source_offset != Fraction(1, 120):
        raise ValueError("source tensor A/V duration offset contract changed")

    official_attestation["media"] = {
        "path": media["official_hf"]["path"],
        "size": media["official_hf"]["size"],
        "sha256": media["official_hf"]["sha256"],
    }
    native_attestation["media"] = {
        "path": media["xllm_native"]["path"],
        "size": media["xllm_native"]["size"],
        "sha256": media["xllm_native"]["sha256"],
    }
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "focused MiniMax-H3 C7 official-HF versus xLLM-native production media comparison",
        "thresholds": THRESHOLDS,
        "sources": {
            "official_hf": official_attestation,
            "xllm_native": native_attestation,
            "prepared": prepared_attestation,
            "shared_source_digest": official_manifest["source"]["digest"],
            "shared_checkpoint_digest": official_manifest["checkpoint"]["digest"],
            "shared_geometry_sha256": _canonical_digest(official_manifest["geometry"]),
            "condition_digests": {
                "official_hf": official_manifest["backend_provenance"]["hidden_sha256"],
                "xllm_native": native_manifest["backend_provenance"]["hidden_sha256"],
                "differ": condition_digest_differs,
            },
        },
        "source_tensor_timing": {
            "video_duration_seconds": float(source_video_duration),
            "audio_duration_seconds": float(source_audio_duration),
            "audio_minus_video_seconds": float(source_offset),
            "audio_minus_video_milliseconds": float(source_offset * 1000),
            "exact_offset_fraction_seconds": "1/120",
        },
        "trajectory_tensor_metrics": trajectory,
        "final_target_latent_tensor_metrics": final_latents,
        "video_metrics": video,
        "audio_metrics": audio,
        "media_structure": media,
        "gate": gate,
    }
    _atomic_write_json(output_report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = compare(args)
    print(args.output_report.resolve())
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
