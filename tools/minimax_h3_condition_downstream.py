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
"""Compare official and native H3 conditions through the H3 text prefix."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from tools.minimax_h3_condition_cache import validate_cache_entry
from tools.minimax_h3_qwen_xllm import _compare, _digest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-path", type=Path, required=True)
    parser.add_argument("--official-condition-cache", type=Path, required=True)
    parser.add_argument("--native-condition-archive", type=Path, required=True)
    parser.add_argument("--native-key", default="layer_49_output_pre_final_norm")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--skip-block0", action="store_true")
    return parser.parse_args()


def _normalize_native_hidden(archive: dict[str, Any], key: str, expected_shape: torch.Size) -> torch.Tensor:
    value = archive.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"native archive must contain tensor key {key!r}")
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError("native condition may only have a singleton batch dimension")
        value = value.squeeze(0)
    if value.shape != expected_shape:
        raise ValueError(f"native condition shape {list(value.shape)} does not match official {list(expected_shape)}")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all().item()):
        raise ValueError("native condition must be a finite floating-point tensor")
    return value.to(device="cpu", dtype=torch.bfloat16).contiguous()


def _build_text_prefix_layout(text_token_tags: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    if text_token_tags.ndim != 1 or text_token_tags.dtype != torch.int64:
        raise ValueError("text_token_tags must be int64 with shape [N]")
    sequence_length = text_token_tags.numel()
    position_ids = torch.zeros((sequence_length, 3), dtype=torch.float64, device=device)
    position_ids[:, 0] = torch.arange(sequence_length, dtype=torch.float64, device=device)
    timestep_indices = torch.zeros(sequence_length, dtype=torch.long, device=device)
    token_tags = text_token_tags.to(device)
    return {
        "position_ids": position_ids,
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "adaln_indices": timestep_indices * 3 + token_tags,
    }


def _run_prefix(
    model: Any,
    hidden: torch.Tensor,
    layout: dict[str, torch.Tensor],
    timestep: torch.Tensor,
    *,
    run_block0: bool,
) -> dict[str, torch.Tensor]:
    device = layout["position_ids"].device
    projected = model.context_embedder(hidden.unsqueeze(0).to(device))
    refined = model.token_refiner(projected)
    outputs = {
        "condition_proj": projected.squeeze(0).cpu(),
        "token_refiner": refined.squeeze(0).cpu(),
    }
    if run_block0:
        rotary = model.rope(layout["position_ids"])
        time_dtype = next(model.time_embedder.parameters()).dtype
        temb = model.time_embedder(model.time_proj(timestep).to(time_dtype))
        block0 = model.transformer_blocks[0](refined, temb, layout["adaln_indices"], rotary)
        outputs["block0"] = block0.squeeze(0).cpu()
    return outputs


def main() -> None:
    args = _parse_args()
    transformer_path = args.transformer_path.resolve()
    condition_cache = args.official_condition_cache.resolve()
    native_archive_path = args.native_condition_archive.resolve()
    output_dir = args.output_dir.resolve()
    if not transformer_path.is_dir() or not native_archive_path.is_file():
        raise ValueError("transformer path and native condition archive must exist")
    if not 0.0 <= args.timestep <= 1.0:
        raise ValueError("timestep must be in [0, 1]")

    manifest = validate_cache_entry(condition_cache)
    if manifest["backend"] != "official_hf":
        raise ValueError("downstream correctness reference requires an official_hf condition cache")
    official_tensors = load_file(str(condition_cache / "condition.safetensors"), device="cpu")
    official_hidden = official_tensors["prompt_embeds"]
    text_token_tags = official_tensors["text_token_tags"]
    native_archive = torch.load(native_archive_path, map_location="cpu", weights_only=True)
    if not isinstance(native_archive, dict):
        raise ValueError("native condition archive must contain a tensor mapping")
    native_hidden = _normalize_native_hidden(native_archive, args.native_key, official_hidden.shape)

    if not args.device.startswith("npu"):
        raise ValueError("the H3 downstream runner requires an NPU device")
    import torch_npu  # noqa: F401
    from diffusers.models.transformers import MiniMaxH3Transformer3DModel

    device = torch.device(args.device)
    torch.npu.set_device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    modules = [model.context_embedder, model.token_refiner]
    if not args.skip_block0:
        modules.extend([model.time_embedder, model.rope, model.transformer_blocks[0]])
    for module in modules:
        module.to(device)

    layout = _build_text_prefix_layout(text_token_tags, device)
    timestep = torch.tensor([args.timestep], dtype=torch.float32, device=device)
    torch.npu.reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    with torch.no_grad():
        official_outputs = _run_prefix(
            model,
            official_hidden,
            layout,
            timestep,
            run_block0=not args.skip_block0,
        )
        native_outputs = _run_prefix(
            model,
            native_hidden,
            layout,
            timestep,
            run_block0=not args.skip_block0,
        )
    torch.npu.synchronize(device)
    inference_seconds = time.perf_counter() - started_at

    modality_masks = {
        "h3_vision_segment": text_token_tags == 0,
        "h3_text": text_token_tags == 1,
        "h3_audio": text_token_tags == 2,
    }
    comparisons = [_compare("layer_49_condition", native_hidden, official_hidden, modality_masks)]
    comparisons.extend(
        _compare(name, native_outputs[name], official_outputs[name], modality_masks) for name in official_outputs
    )
    tensor_archive = {
        "official_layer_49_condition": official_hidden,
        "native_layer_49_condition": native_hidden,
        "text_token_tags": text_token_tags,
    }
    for name, value in official_outputs.items():
        tensor_archive[f"official_{name}"] = value.contiguous()
        tensor_archive[f"native_{name}"] = native_outputs[name].contiguous()
    tensor_path = output_dir / "condition_downstream.safetensors"
    save_file(tensor_archive, str(tensor_path))
    summary = {
        "status": "H3_NATIVE_DOWNSTREAM_REPORT",
        "official_fallback_status": "PASS",
        "native_status": "PENDING_PACKED_DENOISE_GATE",
        "scope": "text_prefix_only",
        "condition_cache_key": manifest["cache_key"],
        "condition_cache_backend": manifest["backend"],
        "native_condition_archive": str(native_archive_path),
        "native_condition_tensor_sha256": _digest(native_hidden),
        "transformer_path": str(transformer_path),
        "device": str(device),
        "timestep": args.timestep,
        "inference_seconds": inference_seconds,
        "comparisons": comparisons,
        "tensor_archive": str(tensor_path),
        "peak_device_memory_bytes": int(torch.npu.max_memory_allocated(device)),
        "peak_device_reserved_bytes": int(torch.npu.max_memory_reserved(device)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
