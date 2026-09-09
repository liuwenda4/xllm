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
"""Build and validate the MiniMax-H3 source-to-xLLM weight contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger

_SOURCE_INDEX = "model.safetensors.index.json"
_TARGET_INDEX = "diffusion_pytorch_model.safetensors.index.json"
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "U8": 1, "BOOL": 1}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the MiniMax-H3 Ref2VA weight coverage and layout ledger.")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Original Ref2VA checkpoint directory.")
    parser.add_argument("--converted-path", type=Path, required=True, help="Converted Modular Diffusers directory.")
    parser.add_argument("--converter-path", type=Path, required=True, help="Pinned official conversion script.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output JSONL ledger path.")
    parser.add_argument("--summary-path", type=Path, required=True, help="Output JSON summary path.")
    parser.add_argument("--expected-variant", default="ref2va", choices=["ref2va", "fl2va"])
    parser.add_argument("--validate-payload", action="store_true", help="Bitwise-check converted tensor payloads.")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _load_converter(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("minimax_h3_official_converter", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load converter module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_size_data = handle.read(8)
        if len(header_size_data) != 8:
            raise ValueError(f"Invalid safetensors header in {path}")
        header_size = struct.unpack("<Q", header_size_data)[0]
        header = json.loads(handle.read(header_size))
    if not isinstance(header, dict):
        raise ValueError(f"Expected a safetensors header object in {path}")
    header.pop("__metadata__", None)
    return header


def _tensor_metadata(key: str, shard: str, header: dict[str, Any]) -> dict[str, Any]:
    value = header[key]
    shape = [int(dimension) for dimension in value["shape"]]
    dtype = str(value["dtype"])
    offsets = value["data_offsets"]
    return {
        "key": key,
        "shard": shard,
        "dtype": dtype,
        "shape": shape,
        "nbytes": int(offsets[1] - offsets[0]),
    }


def _load_indexed_metadata(directory: Path, index_name: str) -> dict[str, dict[str, Any]]:
    index_path = directory / index_name
    index = _load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Expected `weight_map` in {index_path}")

    headers: dict[str, dict[str, Any]] = {}
    shard_headers: dict[str, dict[str, Any]] = {}
    for shard in sorted(set(str(value) for value in weight_map.values())):
        shard_path = directory / shard
        if not shard_path.is_file():
            raise ValueError(f"Index references missing shard: {shard_path}")
        header = _read_safetensors_header(shard_path)
        duplicate_keys = sorted(set(headers).intersection(header))
        if duplicate_keys:
            raise ValueError(f"Duplicate tensor keys across {directory} shards: {duplicate_keys[:5]}")
        headers.update(header)
        shard_headers[shard] = header

    _validate_exact_keys(set(headers), set(weight_map), str(index_path))
    metadata: dict[str, dict[str, Any]] = {}
    for key, shard_value in weight_map.items():
        shard = str(shard_value)
        if key not in shard_headers[shard]:
            raise ValueError(f"Index routes {key} to {shard}, but the shard does not contain it")
        metadata[key] = _tensor_metadata(key, shard, headers)
    return metadata


def _load_single_file_metadata(path: Path) -> dict[str, dict[str, Any]]:
    header = _read_safetensors_header(path)
    return {key: _tensor_metadata(key, path.name, header) for key in sorted(header)}


def _validate_exact_keys(actual: set[str], expected: set[str], label: str) -> None:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(f"{label} key mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")


def _validate_tensor_metadata(metadata: dict[str, Any], shape: list[int], dtype: str, label: str) -> None:
    if metadata["shape"] != shape:
        raise ValueError(f"{label} shape mismatch: expected {shape}, got {metadata['shape']}")
    if metadata["dtype"] != dtype:
        raise ValueError(f"{label} dtype mismatch: expected {dtype}, got {metadata['dtype']}")
    expected_nbytes = _DTYPE_BYTES[dtype]
    for dimension in shape:
        expected_nbytes *= dimension
    if metadata["nbytes"] != expected_nbytes:
        raise ValueError(f"{label} byte-size mismatch: expected {expected_nbytes}, got {metadata['nbytes']}")


def _validate_variant(model_index: dict[str, Any], expected_variant: str) -> None:
    h3_metadata = model_index.get("_minimax_h3")
    if not isinstance(h3_metadata, dict):
        raise ValueError("model_index.json has no `_minimax_h3` object")
    actual_variant = h3_metadata.get("partition")
    if actual_variant != expected_variant:
        raise ValueError(f"Wrong MiniMax-H3 variant: expected {expected_variant}, got {actual_variant}")


def _reorder_interleaved_qkv(weight: Any, num_heads: int, head_dim: int) -> tuple[Any, Any, Any]:
    expected_rows = num_heads * 3 * head_dim
    if weight.ndim < 1 or weight.shape[0] != expected_rows:
        raise ValueError(f"QKV row mismatch: expected {expected_rows}, got {list(weight.shape)}")
    grouped = weight.reshape(num_heads, 3 * head_dim, *weight.shape[1:])
    query, key, value = grouped.split(head_dim, dim=1)
    return tuple(tensor.reshape(num_heads * head_dim, *weight.shape[1:]).contiguous() for tensor in (query, key, value))


def _swap_halves(weight: Any) -> Any:
    import torch

    if weight.shape[0] % 2 != 0:
        raise ValueError(f"Cannot swap odd first dimension {weight.shape[0]}")
    first, second = weight.chunk(2, dim=0)
    return torch.cat((second, first), dim=0).contiguous()


def _operation_for_transformer_key(source_key: str, targets: list[Any]) -> str:
    if not targets:
        return "drop_recompute"
    if source_key.endswith(".attn.qkv_proj.weight"):
        return "qkv_deinterleave_split"
    if source_key.endswith(".mlp.fc1.weight"):
        return "swap_halves"
    if len(targets) == 1 and targets[0][0] == source_key:
        return "identity"
    return "rename"


def _operation_for_video_key(source_key: str, targets: list[str]) -> str:
    if not targets:
        return "drop_unused_zero"
    if ".attn.to_qkv." in source_key:
        return "qkv_deinterleave_split"
    if ".ff.w1." in source_key:
        return "swap_halves"
    if len(targets) == 1 and targets[0] == source_key:
        return "identity"
    return "rename"


def _expected_transformer_dtype(source_key: str, converter: ModuleType) -> str:
    if source_key in converter.MINIMAX_H3_TRANSFORMER_DROPPED_KEYS:
        return "F32"
    return "F32" if source_key.startswith(converter.MINIMAX_H3_FP32_SOURCE_PREFIXES) else "BF16"


def _build_transformer_rows(
    checkpoint_path: Path, converted_path: Path, converter: ModuleType
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    source = _load_indexed_metadata(checkpoint_path / "transformer", _SOURCE_INDEX)
    target = _load_indexed_metadata(converted_path / "transformer_ref", _TARGET_INDEX)
    config = converter.MINIMAX_H3_TRANSFORMER_CONFIG
    plan = converter.get_transformer_key_plan(config)
    _validate_exact_keys(set(source), set(plan), "transformer source plan")

    expected_targets = {target_key for targets in plan.values() for target_key, _ in targets}
    _validate_exact_keys(set(target), expected_targets, "transformer target plan")
    rows: list[dict[str, Any]] = []
    for source_key, target_specs in plan.items():
        source_metadata = source[source_key]
        expected_dtype = _expected_transformer_dtype(source_key, converter)
        if not target_specs:
            expected_source_shape = [int(config["rope_freq_dim"])]
        elif source_key.endswith(".attn.qkv_proj.weight"):
            expected_source_shape = [sum(spec[1][0] for spec in target_specs), *target_specs[0][1][1:]]
        else:
            expected_source_shape = target_specs[0][1]
        _validate_tensor_metadata(source_metadata, expected_source_shape, expected_dtype, source_key)
        targets: list[dict[str, Any]] = []
        for target_key, expected_shape in target_specs:
            target_metadata = target[target_key]
            _validate_tensor_metadata(target_metadata, expected_shape, expected_dtype, target_key)
            targets.append(
                {
                    "component": "minimax_h3_transformer",
                    "parameter": target_key,
                    "shard": target_metadata["shard"],
                    "shape": target_metadata["shape"],
                    "dtype": target_metadata["dtype"],
                    "nbytes": target_metadata["nbytes"],
                }
            )
        rows.append(
            {
                "component": "transformer",
                "variant": "ref2va",
                "source_key": source_key,
                "source_shard": source_metadata["shard"],
                "source_shape": source_metadata["shape"],
                "source_dtype": source_metadata["dtype"],
                "source_nbytes": source_metadata["nbytes"],
                "official_module": "MiniMaxH3DiTModel",
                "xllm_module": "MiniMaxH3Transformer",
                "operation": _operation_for_transformer_key(source_key, target_specs),
                "transpose": False,
                "targets": targets,
                "load_status": "validated",
            }
        )
    return rows, source, target


def _build_video_rows(
    checkpoint_path: Path, converted_path: Path, converter: ModuleType
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    source = _load_single_file_metadata(checkpoint_path / "video_vae" / "source" / "model.safetensors")
    target = _load_indexed_metadata(converted_path / "vae", _TARGET_INDEX)
    plan = converter.get_video_vae_key_plan(converter.MINIMAX_H3_VIDEO_VAE_CONFIG)
    _validate_exact_keys(set(source), set(plan), "video VAE source plan")
    expected_targets = {target_key for targets in plan.values() for target_key in targets}
    _validate_exact_keys(set(target), expected_targets, "video VAE target plan")

    rows: list[dict[str, Any]] = []
    for source_key, target_keys in plan.items():
        source_metadata = source[source_key]
        if not target_keys:
            expected_source_shape = [
                1,
                1,
                int(converter.MINIMAX_H3_VIDEO_VAE_CONFIG["decoder_num_attention_heads"])
                * int(converter.MINIMAX_H3_VIDEO_VAE_CONFIG["decoder_attention_head_dim"]),
            ]
        elif ".attn.to_qkv." in source_key:
            first_target_shape = target[target_keys[0]]["shape"]
            expected_source_shape = [
                sum(target[target_key]["shape"][0] for target_key in target_keys),
                *first_target_shape[1:],
            ]
        else:
            expected_source_shape = target[target_keys[0]]["shape"]
        _validate_tensor_metadata(source_metadata, expected_source_shape, "F32", source_key)
        targets = []
        for target_key in target_keys:
            target_metadata = target[target_key]
            _validate_tensor_metadata(target_metadata, target_metadata["shape"], "F32", target_key)
            targets.append(
                {
                    "component": "minimax_h3_video_vae",
                    "parameter": target_key,
                    "shard": target_metadata["shard"],
                    "shape": target_metadata["shape"],
                    "dtype": target_metadata["dtype"],
                    "nbytes": target_metadata["nbytes"],
                }
            )
        rows.append(
            {
                "component": "video_vae",
                "variant": "shared",
                "source_key": source_key,
                "source_shard": source_metadata["shard"],
                "source_shape": source_metadata["shape"],
                "source_dtype": source_metadata["dtype"],
                "source_nbytes": source_metadata["nbytes"],
                "official_module": "AutoencoderKLLegacy",
                "xllm_module": "MiniMaxH3VideoVAE",
                "operation": _operation_for_video_key(source_key, target_keys),
                "transpose": False,
                "targets": targets,
                "load_status": "validated",
            }
        )
    return rows, source, target


def _build_audio_rows(
    checkpoint_path: Path, converted_path: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    source = _load_single_file_metadata(checkpoint_path / "audio_vae" / "model.safetensors")
    target = _load_single_file_metadata(converted_path / "audio_vae" / "diffusion_pytorch_model.safetensors")
    _validate_exact_keys(set(source), set(target), "audio VAE identity plan")
    rows: list[dict[str, Any]] = []
    for source_key, source_metadata in source.items():
        target_metadata = target[source_key]
        _validate_tensor_metadata(source_metadata, source_metadata["shape"], "F32", source_key)
        _validate_tensor_metadata(target_metadata, source_metadata["shape"], "F32", source_key)
        rows.append(
            {
                "component": "audio_vae",
                "variant": "shared",
                "source_key": source_key,
                "source_shard": source_metadata["shard"],
                "source_shape": source_metadata["shape"],
                "source_dtype": source_metadata["dtype"],
                "source_nbytes": source_metadata["nbytes"],
                "official_module": "MiniMaxH3AudioVAE",
                "xllm_module": "MiniMaxH3AudioVAE",
                "operation": "identity",
                "transpose": False,
                "targets": [
                    {
                        "component": "minimax_h3_audio_vae",
                        "parameter": source_key,
                        "shard": target_metadata["shard"],
                        "shape": target_metadata["shape"],
                        "dtype": target_metadata["dtype"],
                        "nbytes": target_metadata["nbytes"],
                    }
                ],
                "load_status": "validated",
            }
        )
    return rows, source, target


def _text_encoder_expected_shapes(config: dict[str, Any]) -> dict[str, list[int]]:
    text = config["text_config"]
    vision = config["vision_config"]
    hidden_size = int(text["hidden_size"])
    query_width = int(text["num_attention_heads"]) * int(text["head_dim"])
    kv_width = int(text["num_key_value_heads"]) * int(text["head_dim"])
    intermediate_size = int(text["intermediate_size"])
    vocab_size = int(text["vocab_size"])
    shapes = {
        "model.language_model.embed_tokens.weight": [vocab_size, hidden_size],
        "model.language_model.norm.weight": [hidden_size],
        "lm_head.weight": [vocab_size, hidden_size],
    }
    for layer_index in range(int(text["num_hidden_layers"])):
        prefix = f"model.language_model.layers.{layer_index}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": [hidden_size],
                f"{prefix}.post_attention_layernorm.weight": [hidden_size],
                f"{prefix}.self_attn.q_proj.weight": [query_width, hidden_size],
                f"{prefix}.self_attn.k_proj.weight": [kv_width, hidden_size],
                f"{prefix}.self_attn.v_proj.weight": [kv_width, hidden_size],
                f"{prefix}.self_attn.o_proj.weight": [hidden_size, query_width],
                f"{prefix}.self_attn.q_norm.weight": [int(text["head_dim"])],
                f"{prefix}.self_attn.k_norm.weight": [int(text["head_dim"])],
                f"{prefix}.mlp.gate_proj.weight": [intermediate_size, hidden_size],
                f"{prefix}.mlp.up_proj.weight": [intermediate_size, hidden_size],
                f"{prefix}.mlp.down_proj.weight": [hidden_size, intermediate_size],
            }
        )

    vision_hidden = int(vision["hidden_size"])
    vision_intermediate = int(vision["intermediate_size"])
    vision_qkv_width = 3 * vision_hidden
    shapes.update(
        {
            "model.visual.patch_embed.proj.weight": [
                vision_hidden,
                int(vision["in_channels"]),
                int(vision["temporal_patch_size"]),
                int(vision["patch_size"]),
                int(vision["patch_size"]),
            ],
            "model.visual.patch_embed.proj.bias": [vision_hidden],
            "model.visual.pos_embed.weight": [int(vision["num_position_embeddings"]), vision_hidden],
        }
    )
    for layer_index in range(int(vision["depth"])):
        prefix = f"model.visual.blocks.{layer_index}"
        shapes.update(
            {
                f"{prefix}.norm1.weight": [vision_hidden],
                f"{prefix}.norm1.bias": [vision_hidden],
                f"{prefix}.norm2.weight": [vision_hidden],
                f"{prefix}.norm2.bias": [vision_hidden],
                f"{prefix}.attn.qkv.weight": [vision_qkv_width, vision_hidden],
                f"{prefix}.attn.qkv.bias": [vision_qkv_width],
                f"{prefix}.attn.proj.weight": [vision_hidden, vision_hidden],
                f"{prefix}.attn.proj.bias": [vision_hidden],
                f"{prefix}.mlp.linear_fc1.weight": [vision_intermediate, vision_hidden],
                f"{prefix}.mlp.linear_fc1.bias": [vision_intermediate],
                f"{prefix}.mlp.linear_fc2.weight": [vision_hidden, vision_intermediate],
                f"{prefix}.mlp.linear_fc2.bias": [vision_hidden],
            }
        )

    merged_hidden = vision_hidden * int(vision["spatial_merge_size"]) ** 2
    merger_prefixes = ["model.visual.merger"] + [
        f"model.visual.deepstack_merger_list.{index}" for index in range(len(vision["deepstack_visual_indexes"]))
    ]
    for index, prefix in enumerate(merger_prefixes):
        norm_hidden = vision_hidden if index == 0 else merged_hidden
        shapes.update(
            {
                f"{prefix}.norm.weight": [norm_hidden],
                f"{prefix}.norm.bias": [norm_hidden],
                f"{prefix}.linear_fc1.weight": [merged_hidden, merged_hidden],
                f"{prefix}.linear_fc1.bias": [merged_hidden],
                f"{prefix}.linear_fc2.weight": [int(vision["out_hidden_size"]), merged_hidden],
                f"{prefix}.linear_fc2.bias": [int(vision["out_hidden_size"])],
            }
        )
    return shapes


def _qwen_shard_axis(key: str) -> int | None:
    if key == "model.language_model.embed_tokens.weight":
        return 1
    if key == "lm_head.weight":
        return 0
    if any(suffix in key for suffix in ("q_proj.weight", "k_proj.weight", "v_proj.weight")):
        return 0
    if any(suffix in key for suffix in ("gate_proj.weight", "up_proj.weight")):
        return 0
    if any(suffix in key for suffix in ("o_proj.weight", "down_proj.weight")):
        return 1
    return None


def _validate_configs(checkpoint_path: Path, converted_path: Path, converter: ModuleType) -> dict[str, Any]:
    transformer_source = _load_json(checkpoint_path / "transformer" / "config.json")
    transformer_target = _load_json(converted_path / "transformer_ref" / "config.json")
    transformer_renames = {
        "num_refiner_layers": "token_refiner_num_layers",
        "ffn_dim": "ffn_hidden_size",
        "in_channels": "latents_dim",
        "audio_in_channels": "audio_latents_dim",
        "freq_dim": "timestep_input_dim",
        "time_embed_hidden_dim": "time_embed_hidden_size",
        "rope_freq_dim": "rope_inv_freq_len",
    }
    for target_key, expected_value in converter.MINIMAX_H3_TRANSFORMER_CONFIG.items():
        source_key = transformer_renames.get(target_key, target_key)
        if source_key in transformer_source and transformer_source[source_key] != expected_value:
            raise ValueError(
                f"Transformer config mismatch for {source_key}: expected {expected_value}, "
                f"got {transformer_source[source_key]}"
            )
        if transformer_target.get(target_key) != expected_value:
            raise ValueError(
                f"Converted Transformer config mismatch for {target_key}: expected {expected_value}, "
                f"got {transformer_target.get(target_key)}"
            )
    hidden_size = int(transformer_source["hidden_size"])
    if transformer_source.get("adaln_out_features") != 6 * 3 * hidden_size:
        raise ValueError("Transformer adaln_out_features is inconsistent with six parameters and three modalities")
    if transformer_source.get("final_adaln_out_features") != 2 * hidden_size:
        raise ValueError("Transformer final_adaln_out_features is inconsistent with shift and scale")

    video_wrapper = _load_json(checkpoint_path / "video_vae" / "config.json")
    video_target = _load_json(converted_path / "vae" / "config.json")
    for key, expected_value in converter.MINIMAX_H3_VIDEO_VAE_CONFIG.items():
        if video_target.get(key) != expected_value:
            raise ValueError(
                f"Video VAE config mismatch for {key}: expected {expected_value}, got {video_target.get(key)}"
            )
    for key in ("latents_mean", "latents_std"):
        if video_target.get(key) != video_wrapper.get(key):
            raise ValueError(f"Video VAE {key} differs between source wrapper and converted config")

    expected_audio_config = converter.get_audio_vae_config(str(checkpoint_path))
    audio_target = _load_json(converted_path / "audio_vae" / "config.json")
    for key, expected_value in expected_audio_config.items():
        if audio_target.get(key) != expected_value:
            raise ValueError(
                f"Audio VAE config mismatch for {key}: expected {expected_value}, got {audio_target.get(key)}"
            )

    model_index = _load_json(checkpoint_path / "model_index.json")
    shifts = model_index["_minimax_h3"]["sigma_shift_scales"]
    video_scheduler = _load_json(converted_path / "scheduler" / "scheduler_config.json")
    audio_scheduler = _load_json(converted_path / "audio_scheduler" / "scheduler_config.json")
    if video_scheduler.get("shift") != shifts["video"] or audio_scheduler.get("shift") != shifts["audio"]:
        raise ValueError("Converted scheduler shifts do not match the Ref2VA model index")

    text_encoder = _load_json(checkpoint_path / "text_encoder" / "config.json")
    text = text_encoder["text_config"]
    vision = text_encoder["vision_config"]
    expected_text = {
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "intermediate_size": 25600,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    expected_vision = {
        "depth": 27,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_heads": 16,
        "out_hidden_size": 5120,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
    }
    for key, expected_value in expected_text.items():
        if text.get(key) != expected_value:
            raise ValueError(f"Qwen text config mismatch for {key}: expected {expected_value}, got {text.get(key)}")
    for key, expected_value in expected_vision.items():
        if vision.get(key) != expected_value:
            raise ValueError(f"Qwen vision config mismatch for {key}: expected {expected_value}, got {vision.get(key)}")

    modular_index = _load_json(converted_path / "modular_model_index.json")
    transformer_ref_spec = modular_index.get("transformer_ref")
    if not isinstance(transformer_ref_spec, list) or transformer_ref_spec[2].get("subfolder") != "transformer_ref":
        raise ValueError("Modular index does not route Ref2VA to transformer_ref")
    return {
        "transformer_layers": transformer_source["num_layers"],
        "transformer_hidden_size": hidden_size,
        "transformer_heads": transformer_source["num_attention_heads"],
        "transformer_head_dim": transformer_source["attention_head_dim"],
        "transformer_ffn_hidden_size": transformer_source["ffn_hidden_size"],
        "qwen_layers": text["num_hidden_layers"],
        "qwen_hidden_size": text["hidden_size"],
        "qwen_capture_layer_index": 49,
        "qwen_capture_hidden_states_slot": 50,
        "video_latent_channels": video_target["latent_channels"],
        "audio_latent_channels": audio_target["latent_channels"],
        "audio_sampling_rate": audio_target["sampling_rate"],
        "video_sigma_shift": video_scheduler["shift"],
        "audio_sigma_shift": audio_scheduler["shift"],
    }


def _build_text_encoder_rows(checkpoint_path: Path) -> list[dict[str, Any]]:
    component_path = checkpoint_path / "text_encoder"
    source = _load_indexed_metadata(component_path, _SOURCE_INDEX)
    config = _load_json(component_path / "config.json")
    expected_shapes = _text_encoder_expected_shapes(config)
    _validate_exact_keys(set(source), set(expected_shapes), "Qwen3-VL text encoder plan")

    rows: list[dict[str, Any]] = []
    for source_key, shape in expected_shapes.items():
        metadata = source[source_key]
        _validate_tensor_metadata(metadata, shape, "BF16", source_key)
        if source_key == "lm_head.weight":
            xllm_module = "QWen3ForCausalLM"
            target_parameter = "lm_head.weight"
            load_status = "accounted_unused_by_h3"
        elif source_key.startswith("model.visual."):
            xllm_module = "Qwen3_VisionTransformer"
            target_parameter = source_key.removeprefix("model.visual.")
            load_status = "validated_existing_qwen3_vl_loader"
        else:
            xllm_module = "QWen3Model"
            target_parameter = source_key.removeprefix("model.language_model.")
            load_status = "validated_existing_qwen3_vl_loader"
        rows.append(
            {
                "component": "text_encoder",
                "variant": "shared",
                "source_key": source_key,
                "source_shard": metadata["shard"],
                "source_shape": metadata["shape"],
                "source_dtype": metadata["dtype"],
                "source_nbytes": metadata["nbytes"],
                "official_module": "Qwen3VLForConditionalGeneration",
                "xllm_module": xllm_module,
                "operation": "existing_qwen3_vl_loader",
                "transpose": False,
                "targets": [
                    {
                        "component": "minimax_h3_text_encoder",
                        "parameter": target_parameter,
                        "shape": metadata["shape"],
                        "dtype": metadata["dtype"],
                        "tp_shard_axis": _qwen_shard_axis(source_key),
                    }
                ],
                "load_status": load_status,
            }
        )
    return rows


class _TensorReaders:
    def __init__(self, roots: list[Path]) -> None:
        self._roots = roots
        self._stack = ExitStack()
        self._readers: dict[Path, Any] = {}

    def __enter__(self) -> _TensorReaders:
        from safetensors import safe_open

        for root in self._roots:
            for path in root.glob("*.safetensors"):
                self._readers[path] = self._stack.enter_context(safe_open(path, framework="pt", device="cpu"))
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._stack.close()

    def get(self, root: Path, metadata: dict[str, Any]) -> Any:
        return self._readers[root / metadata["shard"]].get_tensor(metadata["key"])


def _validate_payloads(
    checkpoint_path: Path,
    converted_path: Path,
    converter: ModuleType,
    transformer_source: dict[str, dict[str, Any]],
    transformer_target: dict[str, dict[str, Any]],
    video_source: dict[str, dict[str, Any]],
    video_target: dict[str, dict[str, Any]],
    audio_source: dict[str, dict[str, Any]],
    audio_target: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    import torch

    transformer_source_root = checkpoint_path / "transformer"
    transformer_target_root = converted_path / "transformer_ref"
    video_source_root = checkpoint_path / "video_vae" / "source"
    video_target_root = converted_path / "vae"
    audio_source_root = checkpoint_path / "audio_vae"
    audio_target_root = converted_path / "audio_vae"
    roots = [
        transformer_source_root,
        transformer_target_root,
        video_source_root,
        video_target_root,
        audio_source_root,
        audio_target_root,
    ]
    validated = 0
    validated_drops = 0
    with _TensorReaders(roots) as readers:
        transformer_plan = converter.get_transformer_key_plan(converter.MINIMAX_H3_TRANSFORMER_CONFIG)
        for source_key, target_specs in transformer_plan.items():
            source_tensor = readers.get(transformer_source_root, transformer_source[source_key])
            if not target_specs:
                expected = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
                if not torch.equal(source_tensor, expected):
                    raise ValueError("rope.inv_freq does not match the recomputed reference")
                validated_drops += 1
                continue
            if source_key.endswith(".attn.qkv_proj.weight"):
                expected_tensors = _reorder_interleaved_qkv(source_tensor, 56, 128)
            elif source_key.endswith(".mlp.fc1.weight"):
                expected_tensors = (_swap_halves(source_tensor),)
            else:
                expected_tensors = (source_tensor,)
            for (target_key, _), expected_tensor in zip(target_specs, expected_tensors):
                actual_tensor = readers.get(transformer_target_root, transformer_target[target_key])
                if not torch.equal(actual_tensor, expected_tensor):
                    raise ValueError(f"Transformer payload mismatch: {source_key} -> {target_key}")
                validated += 1

        video_plan = converter.get_video_vae_key_plan(converter.MINIMAX_H3_VIDEO_VAE_CONFIG)
        for source_key, target_keys in video_plan.items():
            source_tensor = readers.get(video_source_root, video_source[source_key])
            if not target_keys:
                if torch.count_nonzero(source_tensor).item() != 0:
                    raise ValueError("Video VAE decoder.mask_token must be all zero before dropping")
                validated_drops += 1
                continue
            if ".attn.to_qkv." in source_key:
                expected_tensors = _reorder_interleaved_qkv(source_tensor, 32, 64)
            elif ".ff.w1." in source_key:
                expected_tensors = (_swap_halves(source_tensor),)
            else:
                expected_tensors = (source_tensor,)
            for target_key, expected_tensor in zip(target_keys, expected_tensors):
                actual_tensor = readers.get(video_target_root, video_target[target_key])
                if not torch.equal(actual_tensor, expected_tensor):
                    raise ValueError(f"Video VAE payload mismatch: {source_key} -> {target_key}")
                validated += 1

        for source_key, source_metadata in audio_source.items():
            source_tensor = readers.get(audio_source_root, source_metadata)
            actual_tensor = readers.get(audio_target_root, audio_target[source_key])
            if not torch.equal(actual_tensor, source_tensor):
                raise ValueError(f"Audio VAE payload mismatch: {source_key}")
            validated += 1
    return validated, validated_drops


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    checkpoint_path = args.checkpoint_path.resolve()
    converted_path = args.converted_path.resolve()
    converter_path = args.converter_path.resolve()
    for path in (checkpoint_path, converted_path):
        if not path.is_dir():
            raise ValueError(f"Required directory does not exist: {path}")
    if not converter_path.is_file():
        raise ValueError(f"Converter does not exist: {converter_path}")

    _validate_variant(_load_json(checkpoint_path / "model_index.json"), args.expected_variant)
    converter = _load_converter(converter_path)
    config_contract = _validate_configs(checkpoint_path, converted_path, converter)
    transformer_rows, transformer_source, transformer_target = _build_transformer_rows(
        checkpoint_path, converted_path, converter
    )
    video_rows, video_source, video_target = _build_video_rows(checkpoint_path, converted_path, converter)
    audio_rows, audio_source, audio_target = _build_audio_rows(checkpoint_path, converted_path)
    text_rows = _build_text_encoder_rows(checkpoint_path)
    rows = transformer_rows + text_rows + video_rows + audio_rows

    payload_validated_targets = 0
    dropped_source_invariants = 0
    if args.validate_payload:
        payload_validated_targets, dropped_source_invariants = _validate_payloads(
            checkpoint_path,
            converted_path,
            converter,
            transformer_source,
            transformer_target,
            video_source,
            video_target,
            audio_source,
            audio_target,
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    operation_counts = Counter(row["operation"] for row in rows)
    component_counts = Counter(row["component"] for row in rows)
    source_bytes = sum(row["source_nbytes"] for row in rows)
    target_count = sum(len(row["targets"]) for row in rows)
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "variant": args.expected_variant,
        "checkpoint_path": str(checkpoint_path),
        "converted_path": str(converted_path),
        "converter_path": str(converter_path),
        "converter_sha256": _file_sha256(converter_path),
        "source_tensor_count": len(rows),
        "target_tensor_count": target_count,
        "source_tensor_bytes": source_bytes,
        "config_contract": config_contract,
        "component_source_counts": dict(sorted(component_counts.items())),
        "operation_counts": dict(sorted(operation_counts.items())),
        "payload_validation_enabled": bool(args.validate_payload),
        "payload_validated_target_count": payload_validated_targets,
        "dropped_source_invariant_count": dropped_source_invariants,
        "validation_errors": [],
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logger.info(f"Wrote {len(rows)} source rows and {target_count} target mappings to {args.output_path}")


if __name__ == "__main__":
    main()
