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
"""Build a deterministic MiniMax-H3 condition cache from local tensor archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

SCHEMA_VERSION = 1
CONDITION_ABI = {
    "name": "minimax_h3_qwen_layer50_condition",
    "version": 1,
    "decoder_layer_index": 49,
    "hidden_state_slot": 50,
    "hidden_size": 5120,
    "prompt_embeds_dtype": "bfloat16",
    "text_token_tags_dtype": "int64",
    "allowed_text_token_tags": [0, 1, 2],
}
BACKEND_APPROVAL_STATUS = {
    "official_hf": "approved correctness fallback",
    "xllm_native": "experimental/native-gate-pending",
}

_BACKENDS = tuple(BACKEND_APPROVAL_STATUS)
_HIDDEN_KEY = "layer_49_output_pre_final_norm"
_TAGS_KEY = "h3_token_tags"
_CONDITION_FILE = "condition.safetensors"
_MANIFEST_FILE = "manifest.json"
_HASH_CHUNK_SIZE = 16 * 1024 * 1024
_ALLOWED_TAGS = frozenset(CONDITION_ABI["allowed_text_token_tags"])


class CacheValidationError(ValueError):
    """Raised when an existing condition cache is incomplete or corrupted."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON identically across invocations and dictionary insertion order."""
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _validate_backend(backend: str) -> None:
    if backend not in _BACKENDS:
        raise ValueError(f"Unsupported backend {backend!r}; expected one of {_BACKENDS}")


def _validate_sha256(value: str | None, name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA256 digest") from error
    if value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def _normalization_parameters(fps: float, video_sample_fps: float, reference_short_edge: int) -> dict[str, Any]:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not math.isfinite(video_sample_fps) or video_sample_fps <= 0:
        raise ValueError("video_sample_fps must be finite and positive")
    if not isinstance(reference_short_edge, int) or isinstance(reference_short_edge, bool) or reference_short_edge <= 0:
        raise ValueError("reference_short_edge must be a positive integer")
    return {
        "fps": float(fps),
        "video_sample_fps": float(video_sample_fps),
        "reference_short_edge": int(reference_short_edge),
    }


def build_cache_key_inputs(
    *,
    backend: str,
    source_identity: Sequence[Mapping[str, str]],
    checkpoint_identity: Mapping[str, Any],
    prompt_bytes_sha256: str | None,
    ordered_references: Sequence[Mapping[str, str]],
    fps: float,
    video_sample_fps: float,
    reference_short_edge: int,
) -> dict[str, Any]:
    """Build the complete, path-independent identity hashed for a cache key."""
    _validate_backend(backend)
    _validate_sha256(prompt_bytes_sha256, "prompt_bytes_sha256", optional=True)

    normalized_sources = []
    for index, source in enumerate(source_identity):
        role = source.get("role")
        digest = source.get("sha256")
        if not isinstance(role, str) or not role:
            raise ValueError(f"source_identity[{index}].role must be non-empty")
        _validate_sha256(digest, f"source_identity[{index}].sha256")
        normalized_sources.append({"role": role, "sha256": digest})

    normalized_references = []
    for index, reference in enumerate(ordered_references):
        reference_type = reference.get("type")
        digest = reference.get("sha256")
        if not isinstance(reference_type, str) or not reference_type:
            raise ValueError(f"ordered_references[{index}].type must be non-empty")
        _validate_sha256(digest, f"ordered_references[{index}].sha256")
        normalized_references.append({"type": reference_type, "sha256": digest})

    inputs = {
        "backend": backend,
        "condition_abi": dict(CONDITION_ABI),
        "source_identity": normalized_sources,
        "checkpoint_identity": dict(checkpoint_identity),
        "prompt_bytes_sha256": prompt_bytes_sha256,
        "ordered_references": normalized_references,
        "normalization_parameters": _normalization_parameters(fps, video_sample_fps, reference_short_edge),
    }
    canonical_json_bytes(inputs)
    return inputs


def compute_cache_key(cache_key_inputs: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(cache_key_inputs)).hexdigest()


def build_cache_key(**kwargs: Any) -> str:
    return compute_cache_key(build_cache_key_inputs(**kwargs))


def _require_tensor(archive: Mapping[str, Any], key: str, archive_name: str) -> torch.Tensor:
    value = archive.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{archive_name} must contain tensor key {key!r}")
    return value


def _validate_normalized_condition(prompt_embeds: torch.Tensor, text_token_tags: torch.Tensor) -> None:
    if prompt_embeds.ndim != 2 or prompt_embeds.shape[1] != CONDITION_ABI["hidden_size"]:
        raise ValueError(
            f"prompt_embeds must have shape [N, {CONDITION_ABI['hidden_size']}], got {list(prompt_embeds.shape)}"
        )
    if prompt_embeds.shape[0] == 0:
        raise ValueError("prompt_embeds must contain at least one token")
    if prompt_embeds.dtype != torch.bfloat16:
        raise ValueError(f"prompt_embeds must be bfloat16, got {prompt_embeds.dtype}")
    if not prompt_embeds.is_contiguous():
        raise ValueError("prompt_embeds must be contiguous")
    if not bool(torch.isfinite(prompt_embeds).all().item()):
        raise ValueError("prompt_embeds contains non-finite values")

    if text_token_tags.ndim != 1:
        raise ValueError(f"text_token_tags must have shape [N], got {list(text_token_tags.shape)}")
    if text_token_tags.dtype != torch.int64:
        raise ValueError(f"text_token_tags must be int64, got {text_token_tags.dtype}")
    if not text_token_tags.is_contiguous():
        raise ValueError("text_token_tags must be contiguous")
    if not bool(torch.isfinite(text_token_tags).all().item()):
        raise ValueError("text_token_tags contains non-finite values")
    if text_token_tags.shape[0] != prompt_embeds.shape[0]:
        raise ValueError(
            "prompt_embeds and text_token_tags token counts differ: "
            f"{prompt_embeds.shape[0]} != {text_token_tags.shape[0]}"
        )
    actual_tags = set(text_token_tags.unique().tolist())
    if not actual_tags.issubset(_ALLOWED_TAGS):
        raise ValueError(
            f"text_token_tags values must be a subset of {sorted(_ALLOWED_TAGS)}, got {sorted(actual_tags)}"
        )


def extract_condition_tensors(
    backend: str,
    source_tensors: Mapping[str, Any],
    official_metadata_tensors: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Extract and normalize condition tensors without loading a model or device runtime."""
    _validate_backend(backend)
    if backend == "official_hf":
        if official_metadata_tensors is not None:
            raise ValueError("official metadata/golden tensors are only valid with backend xllm_native")
        tag_archive = source_tensors
        tag_archive_name = "official source archive"
    else:
        if official_metadata_tensors is None:
            raise ValueError("xllm_native requires an explicit official metadata/golden archive for tags; no fallback")
        tag_archive = official_metadata_tensors
        tag_archive_name = "official metadata/golden archive"

    hidden = _require_tensor(source_tensors, _HIDDEN_KEY, f"{backend} source archive")
    if hidden.ndim == 3:
        if hidden.shape[0] != 1:
            raise ValueError(f"{_HIDDEN_KEY} may only have a singleton batch dimension, got {list(hidden.shape)}")
        hidden = hidden.squeeze(0)
    if hidden.ndim != 2 or hidden.shape[1] != CONDITION_ABI["hidden_size"]:
        raise ValueError(
            f"{_HIDDEN_KEY} must normalize to [N, {CONDITION_ABI['hidden_size']}], got {list(hidden.shape)}"
        )
    if not hidden.is_floating_point():
        raise ValueError(f"{_HIDDEN_KEY} must be floating point, got {hidden.dtype}")
    if not bool(torch.isfinite(hidden).all().item()):
        raise ValueError(f"{_HIDDEN_KEY} contains non-finite values")

    tags = _require_tensor(tag_archive, _TAGS_KEY, tag_archive_name)
    if tags.ndim != 1:
        raise ValueError(f"{_TAGS_KEY} must have shape [N], got {list(tags.shape)}")
    if tags.dtype != torch.int64:
        raise ValueError(f"{_TAGS_KEY} must be int64, got {tags.dtype}")

    prompt_embeds = hidden.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    text_token_tags = tags.detach().to(device="cpu").contiguous()
    _validate_normalized_condition(prompt_embeds, text_token_tags)
    return {"prompt_embeds": prompt_embeds, "text_token_tags": text_token_tags}


def load_tensor_archive(path: Path) -> Mapping[str, Any]:
    try:
        archive = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"Failed to load local tensor archive {path}: {error}") from error
    if not isinstance(archive, Mapping):
        raise ValueError(f"Expected a tensor mapping in {path}, got {type(archive).__name__}")
    return archive


def _resolved_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{description} does not exist or is not a file: {resolved}")
    return resolved


def _source_archive_identity(path: Path, role: str) -> dict[str, str]:
    return {"role": role, "path": str(path), "sha256": file_sha256(path)}


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "sha256": tensor_sha256(tensor),
    }


def _manifest_checkpoint_identity(
    values: Mapping[str, str], checkpoint_manifest: Path | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_file = None
    if checkpoint_manifest is not None:
        path = _resolved_file(checkpoint_manifest, "checkpoint identity manifest")
        manifest_file = {"path": str(path), "sha256": file_sha256(path)}
    key_identity = {
        "values": dict(values),
        "manifest_sha256": None if manifest_file is None else manifest_file["sha256"],
    }
    manifest_identity = {"values": dict(values), "manifest": manifest_file}
    return key_identity, manifest_identity


def _prompt_identity(prompt_path: Path | None) -> dict[str, Any] | None:
    if prompt_path is None:
        return None
    path = _resolved_file(prompt_path, "prompt file")
    return {"path": str(path), "sha256": file_sha256(path), "byte_count": path.stat().st_size}


def _reference_identities(references: Sequence[tuple[str, Path]]) -> list[dict[str, str]]:
    identities = []
    for index, (reference_type, reference_path) in enumerate(references):
        if not reference_type:
            raise ValueError(f"Reference {index} has an empty type")
        path = _resolved_file(reference_path, f"reference {index}")
        identities.append({"type": reference_type, "path": str(path), "sha256": file_sha256(path)})
    return identities


def _write_cache_entry(
    cache_dir: Path,
    tensors: Mapping[str, torch.Tensor],
    manifest: dict[str, Any],
) -> None:
    suffix = f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    condition_temp = cache_dir / f".{_CONDITION_FILE}{suffix}"
    manifest_temp = cache_dir / f".{_MANIFEST_FILE}{suffix}"
    try:
        save_file(dict(tensors), str(condition_temp))
        manifest["artifacts"] = {
            _CONDITION_FILE: {
                "sha256": file_sha256(condition_temp),
            }
        }
        with manifest_temp.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(condition_temp, cache_dir / _CONDITION_FILE)
        os.replace(manifest_temp, cache_dir / _MANIFEST_FILE)
    finally:
        condition_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)


def validate_cache_entry(cache_dir: Path, expected_key: str | None = None) -> dict[str, Any]:
    """Validate cache structure, key identity, artifact bytes, and tensor ABI."""
    cache_dir = cache_dir.resolve()
    try:
        if not cache_dir.is_dir():
            raise CacheValidationError(f"Cache directory does not exist: {cache_dir}")
        manifest_path = cache_dir / _MANIFEST_FILE
        condition_path = cache_dir / _CONDITION_FILE
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise CacheValidationError("manifest.json must contain a JSON object")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise CacheValidationError("manifest schema version mismatch")
        if manifest.get("condition_abi") != CONDITION_ABI:
            raise CacheValidationError("manifest condition ABI mismatch")
        if manifest.get("hidden_state") != {"decoder_layer_index": 49, "hidden_state_slot": 50}:
            raise CacheValidationError("manifest hidden-state slot or layer index mismatch")

        backend = manifest.get("backend")
        _validate_backend(backend)
        if manifest.get("backend_approval_status") != BACKEND_APPROVAL_STATUS[backend]:
            raise CacheValidationError("manifest backend approval status mismatch")

        cache_key_inputs = manifest.get("cache_key_inputs")
        if not isinstance(cache_key_inputs, dict):
            raise CacheValidationError("manifest cache_key_inputs must be an object")
        actual_key = compute_cache_key(cache_key_inputs)
        if manifest.get("cache_key") != actual_key:
            raise CacheValidationError("manifest cache key does not match its canonical inputs")
        if expected_key is not None and actual_key != expected_key:
            raise CacheValidationError(f"cache key mismatch: expected {expected_key}, got {actual_key}")
        if cache_dir.name != actual_key:
            raise CacheValidationError(f"cache directory name does not match key {actual_key}")
        if cache_key_inputs.get("backend") != backend or cache_key_inputs.get("condition_abi") != CONDITION_ABI:
            raise CacheValidationError("cache key provenance does not match manifest")
        if manifest.get("normalization_parameters") != cache_key_inputs.get("normalization_parameters"):
            raise CacheValidationError("normalization parameters do not match cache key inputs")

        source_archives = manifest.get("source_archives")
        if not isinstance(source_archives, list):
            raise CacheValidationError("manifest source_archives must be a list")
        source_identity = []
        for source in source_archives:
            if not isinstance(source, dict) or not Path(source.get("path", "")).is_absolute():
                raise CacheValidationError("source archive paths must be absolute")
            _validate_sha256(source.get("sha256"), "source archive sha256")
            source_identity.append({"role": source.get("role"), "sha256": source.get("sha256")})
        if source_identity != cache_key_inputs.get("source_identity"):
            raise CacheValidationError("source archive provenance does not match cache key inputs")

        prompt = manifest.get("prompt")
        prompt_digest = None if prompt is None else prompt.get("sha256")
        if prompt_digest != cache_key_inputs.get("prompt_bytes_sha256"):
            raise CacheValidationError("prompt provenance does not match cache key inputs")
        references = manifest.get("ordered_references")
        if not isinstance(references, list):
            raise CacheValidationError("manifest ordered_references must be a list")
        reference_identity = [{"type": item.get("type"), "sha256": item.get("sha256")} for item in references]
        if reference_identity != cache_key_inputs.get("ordered_references"):
            raise CacheValidationError("reference provenance does not match cache key inputs")

        checkpoint = manifest.get("checkpoint_identity_inputs")
        if not isinstance(checkpoint, dict):
            raise CacheValidationError("manifest checkpoint_identity_inputs must be an object")
        checkpoint_manifest = checkpoint.get("manifest")
        checkpoint_key_identity = {
            "values": checkpoint.get("values"),
            "manifest_sha256": None if checkpoint_manifest is None else checkpoint_manifest.get("sha256"),
        }
        if checkpoint_key_identity != cache_key_inputs.get("checkpoint_identity"):
            raise CacheValidationError("checkpoint provenance does not match cache key inputs")

        artifact = manifest.get("artifacts", {}).get(_CONDITION_FILE, {})
        if file_sha256(condition_path) != artifact.get("sha256"):
            raise CacheValidationError("condition.safetensors file SHA256 mismatch")
        tensors = load_file(str(condition_path), device="cpu")
        if set(tensors) != {"prompt_embeds", "text_token_tags"}:
            raise CacheValidationError("condition.safetensors has unexpected tensor keys")
        _validate_normalized_condition(tensors["prompt_embeds"], tensors["text_token_tags"])
        expected_tensors = manifest.get("tensors")
        actual_tensors = {name: _tensor_metadata(tensor) for name, tensor in tensors.items()}
        if expected_tensors != actual_tensors:
            raise CacheValidationError("condition tensor metadata or SHA256 mismatch")
        return manifest
    except CacheValidationError:
        raise
    except Exception as error:
        raise CacheValidationError(f"Invalid cache entry {cache_dir}: {error}") from error


def create_condition_cache(
    *,
    backend: str,
    source_archive: Path,
    cache_root: Path,
    official_metadata_archive: Path | None = None,
    checkpoint_identity: Mapping[str, str] | None = None,
    checkpoint_manifest: Path | None = None,
    prompt_path: Path | None = None,
    references: Sequence[tuple[str, Path]] = (),
    fps: float = 24.0,
    video_sample_fps: float = 2.0,
    reference_short_edge: int = 2048,
    force: bool = False,
) -> Path:
    """Create or validate one content-addressed offline condition cache entry."""
    _validate_backend(backend)
    source_archive = _resolved_file(source_archive, f"{backend} source archive")
    if backend == "official_hf":
        if official_metadata_archive is not None:
            raise ValueError("--official-metadata-archive is only valid with --backend xllm_native")
        source_archives = [_source_archive_identity(source_archive, "official_reference")]
        metadata_archive = None
    else:
        if official_metadata_archive is None:
            raise ValueError("xllm_native requires --official-metadata-archive; no backend fallback is permitted")
        metadata_archive = _resolved_file(official_metadata_archive, "official metadata/golden archive")
        source_archives = [
            _source_archive_identity(source_archive, "xllm_native_condition"),
            _source_archive_identity(metadata_archive, "official_metadata_golden"),
        ]

    normalization = _normalization_parameters(fps, video_sample_fps, reference_short_edge)
    checkpoint_key_identity, checkpoint_manifest_identity = _manifest_checkpoint_identity(
        checkpoint_identity or {}, checkpoint_manifest
    )
    prompt = _prompt_identity(prompt_path)
    reference_identities = _reference_identities(references)
    cache_key_inputs = build_cache_key_inputs(
        backend=backend,
        source_identity=[{"role": item["role"], "sha256": item["sha256"]} for item in source_archives],
        checkpoint_identity=checkpoint_key_identity,
        prompt_bytes_sha256=None if prompt is None else prompt["sha256"],
        ordered_references=[{"type": item["type"], "sha256": item["sha256"]} for item in reference_identities],
        **normalization,
    )
    cache_key = compute_cache_key(cache_key_inputs)
    cache_root = cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root / cache_key

    if cache_dir.exists():
        try:
            validate_cache_entry(cache_dir, expected_key=cache_key)
        except CacheValidationError as error:
            if not force:
                raise CacheValidationError(
                    f"Existing cache entry is invalid; pass --force to replace it safely: {cache_dir}: {error}"
                ) from error
        else:
            return cache_dir
    cache_dir.mkdir(parents=False, exist_ok=True)

    source_tensors = load_tensor_archive(source_archive)
    metadata_tensors = None if metadata_archive is None else load_tensor_archive(metadata_archive)
    tensors = extract_condition_tensors(backend, source_tensors, metadata_tensors)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "condition_abi": dict(CONDITION_ABI),
        "backend": backend,
        "backend_approval_status": BACKEND_APPROVAL_STATUS[backend],
        "hidden_state": {"decoder_layer_index": 49, "hidden_state_slot": 50},
        "cache_key": cache_key,
        "cache_key_inputs": cache_key_inputs,
        "source_archives": source_archives,
        "checkpoint_identity_inputs": checkpoint_manifest_identity,
        "prompt": prompt,
        "ordered_references": reference_identities,
        "normalization_parameters": normalization,
        "tensors": {name: _tensor_metadata(tensor) for name, tensor in tensors.items()},
    }
    _write_cache_entry(cache_dir, tensors, manifest)
    validate_cache_entry(cache_dir, expected_key=cache_key)
    return cache_dir


def _parse_checkpoint_identity(values: Sequence[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, identity = value.partition("=")
        if not separator or not name or not identity:
            raise ValueError(f"Checkpoint identity must be NAME=VALUE, got {value!r}")
        if name in result:
            raise ValueError(f"Duplicate checkpoint identity name: {name}")
        result[name] = identity
    return result


def _parse_references(values: Sequence[str]) -> list[tuple[str, Path]]:
    references = []
    for value in values:
        reference_type, separator, path = value.partition("=")
        if not separator or not reference_type or not path:
            raise ValueError(f"Reference must be TYPE=PATH, got {value!r}")
        references.append((reference_type, Path(path)))
    return references


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=_BACKENDS, default="official_hf")
    parser.add_argument("--source-archive", type=Path, required=True, help="Existing local Qwen layer-50 archive.")
    parser.add_argument(
        "--official-metadata-archive",
        "--official-golden-archive",
        dest="official_metadata_archive",
        type=Path,
        help="Official Qwen reference archive supplying xllm_native token tags.",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-identity",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Repeatable checkpoint identity input, such as revision or checkpoint digest.",
    )
    parser.add_argument("--checkpoint-manifest", type=Path, help="Optional local checkpoint identity manifest.")
    parser.add_argument("--prompt-path", type=Path, help="Optional prompt file, hashed as raw bytes.")
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="TYPE=PATH",
        help="Repeatable ordered local reference and type.",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--video-sample-fps", type=float, default=2.0)
    parser.add_argument("--reference-short-edge", type=int, default=2048)
    parser.add_argument("--force", action="store_true", help="Replace an invalid existing cache entry.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cache_dir = create_condition_cache(
        backend=args.backend,
        source_archive=args.source_archive,
        official_metadata_archive=args.official_metadata_archive,
        cache_root=args.cache_root,
        checkpoint_identity=_parse_checkpoint_identity(args.checkpoint_identity),
        checkpoint_manifest=args.checkpoint_manifest,
        prompt_path=args.prompt_path,
        references=_parse_references(args.reference),
        fps=args.fps,
        video_sample_fps=args.video_sample_fps,
        reference_short_edge=args.reference_short_edge,
        force=args.force,
    )
    print(cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
