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
"""Audit a local MiniMax-H3 Ref2VA checkpoint without loading tensor data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger

_REQUIRED_COMPONENTS = (
    "processor",
    "tokenizer",
    "text_encoder",
    "transformer",
    "video_vae",
    "audio_vae",
)
_INDEX_NAMES = ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json")
_HASH_CHUNK_SIZE = 16 * 1024 * 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MiniMax-H3 Ref2VA checkpoint identity and shard coverage.")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Ref2VA checkpoint directory.")
    parser.add_argument("--output-path", type=Path, required=True, help="JSON manifest output path.")
    parser.add_argument(
        "--skip-weight-hashes",
        action="store_true",
        help="Hash metadata only. Full weight hashing is the default H1 identity gate.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_length_data = handle.read(8)
        if len(header_length_data) != 8:
            raise ValueError(f"Invalid safetensors header in {path}")
        header_length = struct.unpack("<Q", header_length_data)[0]
        header = json.loads(handle.read(header_length))
    if not isinstance(header, dict):
        raise ValueError(f"Expected a safetensors header object in {path}")
    header.pop("__metadata__", None)
    return header


def _discover_hf_revisions(checkpoint_path: Path) -> list[str]:
    model_root = checkpoint_path.parent
    metadata_root = model_root / ".cache" / "huggingface" / "download" / checkpoint_path.name
    if not metadata_root.is_dir():
        return []

    revisions: set[str] = set()
    for path in metadata_root.rglob("*.metadata"):
        lines = path.read_text(encoding="utf-8").splitlines()
        if lines:
            revisions.add(lines[0])
    return sorted(revisions)


def _audit_component(checkpoint_path: Path, component_name: str) -> dict[str, Any]:
    component_path = checkpoint_path / component_name
    files = sorted(path for path in component_path.rglob("*") if path.is_file())
    index_paths = [component_path / name for name in _INDEX_NAMES if (component_path / name).is_file()]

    indexed_keys = 0
    indexed_shards: set[str] = set()
    missing_shards: list[str] = []
    index_total_size = 0
    for index_path in index_paths:
        index = _load_json(index_path)
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise ValueError(f"Expected `weight_map` object in {index_path}")
        indexed_keys += len(weight_map)
        indexed_shards.update(str(value) for value in weight_map.values())
        metadata = index.get("metadata", {})
        if isinstance(metadata, dict):
            index_total_size += int(metadata.get("total_size", 0))
    for shard in sorted(indexed_shards):
        if not (component_path / shard).is_file():
            missing_shards.append(shard)

    standalone_safetensors = [
        path for path in files if path.suffix == ".safetensors" and path.name not in indexed_shards
    ]
    standalone_keys = sum(len(_read_safetensors_header(path)) for path in standalone_safetensors)
    return {
        "file_count": len(files),
        "total_file_bytes": sum(path.stat().st_size for path in files),
        "index_files": [path.name for path in index_paths],
        "indexed_key_count": indexed_keys,
        "indexed_shards": sorted(indexed_shards),
        "index_total_tensor_bytes": index_total_size,
        "standalone_safetensors": [str(path.relative_to(component_path)) for path in standalone_safetensors],
        "standalone_key_count": standalone_keys,
        "missing_shards": missing_shards,
    }


def _build_file_manifest(checkpoint_path: Path, skip_weight_hashes: bool) -> tuple[list[dict[str, Any]], str]:
    entries: list[dict[str, Any]] = []
    identity_digest = hashlib.sha256()
    files = sorted(path for path in checkpoint_path.rglob("*") if path.is_file())
    for path in files:
        relative_path = str(path.relative_to(checkpoint_path))
        is_weight = path.suffix == ".safetensors"
        digest = None if is_weight and skip_weight_hashes else _sha256(path)
        size = path.stat().st_size
        entries.append({"path": relative_path, "size": size, "sha256": digest})
        identity_digest.update(f"{relative_path}\0{size}\0{digest or 'SKIPPED'}\n".encode())
        logger.info(f"Audited {relative_path}: {size} bytes")
    return entries, identity_digest.hexdigest()


def _validate_model_index(model_index: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if model_index.get("_class_name") != "MiniMaxH3Pipeline":
        errors.append("model_index._class_name must be MiniMaxH3Pipeline")
    h3_metadata = model_index.get("_minimax_h3")
    if not isinstance(h3_metadata, dict) or h3_metadata.get("partition") != "ref2va":
        errors.append("model_index._minimax_h3.partition must be ref2va")
    if model_index.get("transformer") != ["diffusers", "MiniMaxH3DiTModel"]:
        errors.append("model_index.transformer must identify MiniMaxH3DiTModel")
    return errors


def main() -> None:
    args = _parse_args()
    checkpoint_path = args.checkpoint_path.resolve()
    output_path = args.output_path.resolve()
    if not checkpoint_path.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_path}")

    missing_components = [name for name in _REQUIRED_COMPONENTS if not (checkpoint_path / name).is_dir()]
    model_index_path = checkpoint_path / "model_index.json"
    if not model_index_path.is_file():
        raise ValueError(f"Missing model index: {model_index_path}")

    model_index = _load_json(model_index_path)
    validation_errors = _validate_model_index(model_index)
    if missing_components:
        validation_errors.append(f"Missing components: {missing_components}")

    components = {
        name: _audit_component(checkpoint_path, name)
        for name in _REQUIRED_COMPONENTS
        if (checkpoint_path / name).is_dir()
    }
    for name, component in components.items():
        if component["missing_shards"]:
            validation_errors.append(f"{name} has missing shards: {component['missing_shards']}")

    files, checkpoint_digest = _build_file_manifest(checkpoint_path, args.skip_weight_hashes)
    h3_metadata = model_index.get("_minimax_h3", {})
    if not isinstance(h3_metadata, dict):
        h3_metadata = {}
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_digest": checkpoint_digest,
        "weight_hashes_skipped": bool(args.skip_weight_hashes),
        "huggingface_source_revisions": _discover_hf_revisions(checkpoint_path),
        "model_identity": {
            "class_name": model_index.get("_class_name"),
            "partition": h3_metadata.get("partition"),
            "transformer": model_index.get("transformer"),
            "sigma_shift_scales": h3_metadata.get("sigma_shift_scales"),
        },
        "components": components,
        "files": files,
        "validation_errors": validation_errors,
        "status": "PASS" if not validation_errors and not args.skip_weight_hashes else "INCOMPLETE",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if validation_errors:
        raise ValueError(f"Checkpoint audit failed: {validation_errors}")
    logger.info(f"Wrote checkpoint manifest to {output_path} with status {manifest['status']}")


if __name__ == "__main__":
    main()
