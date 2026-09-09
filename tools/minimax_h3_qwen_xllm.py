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
"""Run xLLM's Qwen3-VL NPU path and compare MiniMax-H3 layer-50 golden data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.logger import logger


class _SafetensorStateDict:
    def __init__(self, checkpoint_path: Path) -> None:
        from safetensors import safe_open

        index_path = checkpoint_path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self._weight_map: dict[str, str] = index["weight_map"]
        self._readers = {
            shard: safe_open(checkpoint_path / shard, framework="pt", device="cpu")
            for shard in sorted(set(self._weight_map.values()))
        }

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get_tensor(self, name: str) -> Any:
        return self._readers[self._weight_map[name]].get_tensor(name)


class _NpuPrefillAttention:
    def __init__(self, device: Any) -> None:
        self.device = device
        self._metadata = None
        self._kv_caches: list[Any] = []
        self._causal_masks: dict[int, Any] = {}

    @property
    def is_mla(self) -> bool:
        return False

    @property
    def num_kv_blocks(self) -> int:
        return 1

    @property
    def page_size(self) -> int:
        return 1

    def bind_kv_caches(self, kv_caches: list[Any]) -> None:
        self._kv_caches = kv_caches

    def prepare(self, metadata: Any, *, graph_mode: bool = False) -> None:
        del graph_mode
        self._metadata = metadata

    def execute(self, q: Any, k: Any, v: Any, layer: Any) -> Any:
        import torch
        import torch_npu

        num_tokens = q.shape[0]
        num_heads = layer.num_heads
        num_kv_heads = layer.num_kv_heads
        q = q.reshape(num_tokens, num_heads, layer.head_dim).contiguous()
        k = k.reshape(num_tokens, num_kv_heads, layer.head_dim).contiguous()
        v = v.reshape(num_tokens, num_kv_heads, layer.head_dim).contiguous()
        causal_mask = self._causal_masks.get(2048)
        if causal_mask is None:
            causal_mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool), diagonal=1).to(self.device)
            self._causal_masks[2048] = causal_mask
        output = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=num_heads,
            input_layout="TND",
            atten_mask=causal_mask,
            scale=layer.scale,
            actual_seq_qlen=[num_tokens],
            actual_seq_kvlen=[num_tokens],
            sparse_mode=2,
        )[0]
        return output.reshape(num_tokens, num_heads * layer.head_dim)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare xLLM Qwen3-VL NPU layer-50 output with H3 golden data.")
    parser.add_argument("--text-checkpoint", type=Path, required=True)
    parser.add_argument("--golden-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--max-abs-error", type=float, default=0.25)
    return parser.parse_args()


def _load_flat_config(path: Path, device: str) -> dict[str, Any]:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    text = config["text_config"]
    vision = config["vision_config"]
    flat = dict(text)
    flat.update(
        {
            "model_type": "qwen3_vl",
            "dtype": "bfloat16",
            "device": device,
            "world_size": 1,
            "tp_size": 1,
            "tp_rank": 0,
            "dp_size": 1,
            "dp_rank": 0,
            "cp_size": 1,
            "image_token_id": config["image_token_id"],
            "video_token_id": config["video_token_id"],
            "vision_start_token_id": config["vision_start_token_id"],
            "vision_end_token_id": config["vision_end_token_id"],
            "rope_scaling_mrope_section": text["rope_scaling"]["mrope_section"],
            "rope_scaling_rope_type": text["rope_scaling"]["rope_type"],
            "mm_deepstack_visual_indexes": vision["deepstack_visual_indexes"],
            "mm_num_hidden_layers": vision["depth"],
            "mm_hidden_act": vision["hidden_act"],
            "mm_hidden_size": vision["hidden_size"],
            "mm_intermediate_size": vision["intermediate_size"],
            "mm_num_attention_heads": vision["num_heads"],
            "mm_num_channels": vision["in_channels"],
            "mm_projection_dim": vision["out_hidden_size"],
            "mm_patch_size": vision["patch_size"],
            "mm_num_position_embeddings": vision["num_position_embeddings"],
            "mm_spatial_merge_size": vision["spatial_merge_size"],
            "mm_temporal_patch_size": vision["temporal_patch_size"],
            "layers_to_capture": [49],
        }
    )
    return flat


def _digest(tensor: Any) -> str:
    import torch

    return hashlib.sha256(tensor.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _compare(name: str, actual: Any, expected: Any) -> dict[str, Any]:
    import torch

    actual_cpu = actual.detach().to("cpu").float()
    expected_cpu = expected.detach().to("cpu").float()
    if actual_cpu.shape != expected_cpu.shape:
        return {
            "name": name,
            "shape_match": False,
            "actual_shape": list(actual_cpu.shape),
            "expected_shape": list(expected_cpu.shape),
        }
    difference = (actual_cpu - expected_cpu).abs()
    max_index = int(difference.reshape(-1).argmax().item())
    return {
        "name": name,
        "shape_match": True,
        "actual_shape": list(actual_cpu.shape),
        "expected_shape": list(expected_cpu.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "actual_sha256": _digest(actual),
        "expected_sha256": _digest(expected),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "nonzero_error_count": int(torch.count_nonzero(difference).item()),
        "max_error_flat_index": max_index,
    }


def _run_layerwise_qwen(
    model: Any, input_ids: Any, positions: Any, device: Any
) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    import torch

    hidden = model.model._inputs_embeds
    model.model._inputs_embeds = None
    positions = positions.to(torch.int64).contiguous()
    cos, sin = None, None
    mrope_section: list[int] | None = None
    cos_sin_cache = model.model.rotary.cos_sin_cache
    if positions.dim() == 2 and model.model.mrope_section:
        cos_sin_cache = model.model._mrope_cos_sin
        mrope_section = model.model.mrope_section
    else:
        cos, sin = model.model.rotary(positions)

    residual = None
    aux_hidden_buffer = model.model.aux_hidden_capture.create_buffer(hidden)
    layer_0_output = None
    layer_49_output = None
    selected_outputs: dict[str, Any] = {}
    layer_43_nodes: dict[str, Any] = {}
    for layer_index, layer in enumerate(model.model.layers):
        layer.to(device)
        hooks = []
        if layer_index == 43:

            def capture(name: str) -> Any:
                def callback(module: Any, args: tuple[Any, ...], output: Any) -> None:
                    del module, args
                    value = output[0] if isinstance(output, tuple) else output
                    layer_43_nodes[name] = value.detach().cpu()

                return callback

            hooks = [
                layer.input_layernorm.register_forward_hook(capture("input_norm")),
                layer.self_attn.register_forward_hook(capture("self_attention")),
                layer.post_attention_layernorm.register_forward_hook(capture("post_attention_norm")),
                layer.mlp.gate_up_proj.register_forward_hook(capture("gate_up_proj")),
                layer.mlp.register_forward_hook(capture("mlp")),
            ]
        try:
            hidden, residual = layer(
                hidden,
                residual,
                positions,
                cos_sin_cache,
                cos,
                sin,
                mrope_section,
            )
            if model.model.deepstack_input_embeds is not None and layer_index < len(model.model.deepstack_input_embeds):
                hidden = hidden + model.model.deepstack_input_embeds[layer_index]
            model.model.aux_hidden_capture.capture_layer(layer_index, hidden, residual, aux_hidden_buffer)
            if layer_index == 0:
                layer_0_output = hidden + residual
            if layer_index == 49:
                layer_49_output = hidden + residual
            if layer_index in (0, 1, 2, 3, 4, 8, 16) or 32 <= layer_index < 50:
                selected_outputs[f"layer_{layer_index}_output_pre_final_norm"] = hidden + residual
        finally:
            for hook in hooks:
                hook.remove()
            layer.to("cpu")

    model.model.deepstack_input_embeds = None
    final_hidden, _ = model.model.norm(hidden, residual)
    if layer_0_output is None or layer_49_output is None:
        raise RuntimeError("Qwen layerwise execution did not reach layers 0 and 49")
    finalized = model.model.aux_hidden_capture.finalize(final_hidden, aux_hidden_buffer)
    if not isinstance(finalized, tuple):
        raise RuntimeError("Qwen layerwise execution did not produce captured hidden states")
    return finalized[0], finalized[1], layer_0_output, selected_outputs, layer_43_nodes


def _run_isolated_layer_43(model: Any, hidden: Any, positions: Any, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    positions = positions.to(torch.int64).contiguous()
    cos_sin_cache = model.model._mrope_cos_sin
    nodes: dict[str, Any] = {}

    def capture(name: str) -> Any:
        def callback(module: Any, args: tuple[Any, ...], output: Any) -> None:
            del module, args
            value = output[0] if isinstance(output, tuple) else output
            nodes[name] = value.detach().cpu()

        return callback

    layer = model.model.layers[43]
    layer.to(device)
    hooks = [
        layer.input_layernorm.register_forward_hook(capture("input_norm")),
        layer.self_attn.register_forward_hook(capture("self_attention")),
        layer.post_attention_layernorm.register_forward_hook(capture("post_attention_norm")),
        layer.mlp.gate_up_proj.register_forward_hook(capture("gate_up_proj")),
        layer.mlp.register_forward_hook(capture("mlp")),
    ]
    try:
        branch, residual = layer(
            hidden.to(device),
            None,
            positions,
            cos_sin_cache,
            None,
            None,
            model.model.mrope_section,
        )
        output = branch + residual
    finally:
        for hook in hooks:
            hook.remove()
        layer.to("cpu")
    return output, nodes


def main() -> None:
    args = _parse_args()
    text_checkpoint = args.text_checkpoint.resolve()
    golden_path = args.golden_path.resolve()
    output_dir = args.output_dir.resolve()
    if not text_checkpoint.is_dir() or not golden_path.is_file():
        raise ValueError("text checkpoint and golden tensor file must exist")

    import torch

    import xllm

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    device = torch.device(args.device)
    if device.type != "npu":
        raise ValueError("The xLLM Qwen layer-50 runner requires an NPU device")
    torch.npu.set_device(device)
    xllm._load_public_api()
    from xllm.python import initialize_runtime

    initialize_runtime()
    from xllm.python.model_executor.forward_context import ForwardContext, forward_context
    from xllm.python.models.qwen3_vl import Qwen3VLForConditionalGeneration

    golden = torch.load(golden_path, map_location="cpu", weights_only=True)
    config = _load_flat_config(text_checkpoint, "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "device": str(device),
        "golden_path": str(golden_path),
        "text_checkpoint": str(text_checkpoint),
        "layers_to_capture": [49],
        "max_abs_error": args.max_abs_error,
        "torch_version": torch.__version__,
        "xllm_commit": os.popen("git -C /data/workspace/lwd/xllm rev-parse HEAD").read().strip(),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    logger.info("Constructing xLLM Qwen3-VL on CPU")
    model = Qwen3VLForConditionalGeneration(config)
    model.load_weights([_SafetensorStateDict(text_checkpoint)], tp_rank=0, tp_size=1)
    model.lm_head = None
    model.model.embed_tokens.to(device)
    model.model.rotary.to(device)
    model.model.norm.to(device)
    model.model._mrope_cos_sin = model.model._mrope_cos_sin.to(device)
    model.device = device
    logger.info("Loaded Qwen3-VL weights and moved the conditioner to NPU")

    input_ids = golden["input_ids"].squeeze(0)
    position_ids = golden["position_ids"]
    pixel_values = golden["pixel_values"]
    image_grid_thw = golden["image_grid_thw"]
    del pixel_values, image_grid_thw
    model.model._inputs_embeds = golden["input_embeddings"].squeeze(0).to(device)
    image_positions = golden["mm_token_type_ids"].squeeze(0) == 1
    model.model.deepstack_input_embeds = []
    for feature in golden["visual_deepstack_features"]:
        full_feature = torch.zeros_like(model.model._inputs_embeds)
        full_feature[image_positions.to(device)] = feature.to(device)
        model.model.deepstack_input_embeds.append(full_feature)
    merged_input_embeddings = model.model._inputs_embeds.detach().clone()
    metadata = SimpleNamespace(is_prefill=True, is_chunked_prefill=False)
    attention_backend = _NpuPrefillAttention(device)
    attention_backend.prepare(metadata)
    context = ForwardContext(attention_backend, device, metadata, [])

    if device.type == "npu":
        torch.npu.reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    with torch.no_grad(), forward_context(context):
        output = _run_layerwise_qwen(model, input_ids.to(device), position_ids.to(device), device)
        isolated_layer_43_output, isolated_layer_43_nodes = _run_isolated_layer_43(
            model,
            golden["selected_hidden_states"]["layer_42_output_pre_final_norm"].squeeze(0),
            position_ids.to(device),
            device,
        )
    torch.npu.synchronize(device)
    inference_seconds = time.perf_counter() - started_at
    if not isinstance(output, tuple) or len(output) != 5:
        raise RuntimeError("xLLM Qwen3-VL did not return the expected hidden states with capture enabled")
    final_hidden, layer_49_output, layer_0_output, selected_outputs, layer_43_nodes = output

    comparisons = [
        _compare("merged_input_embeddings", merged_input_embeddings, golden["input_embeddings"].squeeze(0)),
        _compare("final_norm_output", final_hidden, golden["final_norm_output"].squeeze(0)),
    ]
    for name, actual in selected_outputs.items():
        layer_index = int(name.split("_")[1])
        expected = golden["selected_hidden_states"][name].squeeze(0)
        if layer_index < 3:
            expected = expected.clone()
            expected[image_positions] += golden["visual_deepstack_features"][layer_index]
        comparisons.append(_compare(name, actual, expected))
    for name, actual in layer_43_nodes.items():
        if name == "gate_up_proj":
            expected = torch.cat(
                [golden["layer_43_nodes"]["gate_proj"], golden["layer_43_nodes"]["up_proj"]], dim=-1
            ).squeeze(0)
        else:
            expected = golden["layer_43_nodes"][name].squeeze(0)
        if name not in ("gate_proj", "up_proj"):
            comparisons.append(_compare(f"layer_43_{name}", actual, expected))
    comparisons.append(
        _compare(
            "isolated_layer_43_output",
            isolated_layer_43_output,
            golden["selected_hidden_states"]["layer_43_output_pre_final_norm"].squeeze(0),
        )
    )
    for name, actual in isolated_layer_43_nodes.items():
        if name == "gate_up_proj":
            expected = torch.cat(
                [golden["layer_43_nodes"]["gate_proj"], golden["layer_43_nodes"]["up_proj"]], dim=-1
            ).squeeze(0)
        else:
            expected = golden["layer_43_nodes"][name].squeeze(0)
        comparisons.append(_compare(f"isolated_layer_43_{name}", actual, expected))
    if "gate_up_proj" in layer_43_nodes:
        import torch.nn.functional as F

        gate_up = layer_43_nodes["gate_up_proj"]
        actual_activation = torch.ops.xllm_ops.silu_and_mul(gate_up.to(device)).cpu()
        expected_activation = F.silu(golden["layer_43_nodes"]["gate_proj"].squeeze(0)) * golden["layer_43_nodes"][
            "up_proj"
        ].squeeze(0)
        comparisons.append(_compare("layer_43_silu_and_mul", actual_activation, expected_activation))
    for comparison in comparisons:
        logger.info(
            f"{comparison['name']}: shape_match={comparison['shape_match']} "
            f"max_abs_error={comparison.get('max_abs_error', 'n/a')}"
        )
    model_state = {
        "merged_input_embeddings": merged_input_embeddings.cpu(),
        "final_norm_output": final_hidden.cpu(),
    }
    model_state.update({name: value.cpu() for name, value in selected_outputs.items()})
    model_state.update({f"layer_43_{name}": value for name, value in layer_43_nodes.items()})
    model_state["isolated_layer_43_output"] = isolated_layer_43_output.cpu()
    model_state.update({f"isolated_layer_43_{name}": value for name, value in isolated_layer_43_nodes.items()})
    torch.save(model_state, output_dir / "qwen_layer50_xllm.pt")
    summary = {
        "status": "H3_QWEN_LAYER50_NPU_PASS"
        if all(item["shape_match"] and item["max_abs_error"] <= args.max_abs_error for item in comparisons)
        else "H3_QWEN_LAYER50_NPU_DIVERGED",
        "device": str(device),
        "inference_seconds": inference_seconds,
        "comparisons": comparisons,
        "golden_input_ids_sha256": _digest(golden["input_ids"]),
        "golden_position_ids_sha256": _digest(golden["position_ids"]),
        "xllm_tensor_archive_sha256": _digest(torch.cat([value.flatten() for value in model_state.values()])),
        "peak_device_memory_bytes": int(torch.npu.max_memory_allocated(device)),
        "peak_device_reserved_bytes": int(torch.npu.max_memory_reserved(device)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if summary["status"] != "H3_QWEN_LAYER50_NPU_PASS":
        raise RuntimeError("xLLM Qwen3-VL layer-50 comparison exceeded the configured error threshold")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("xLLM MiniMax-H3 Qwen layer-50 run failed")
        raise
