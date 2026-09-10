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
import math
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
    def __init__(self, device: Any, backend: str) -> None:
        self.device = device
        self.backend = backend
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
        if self.backend == "torch_eager":
            return self._execute_torch_eager(q, k, v, layer)
        if self.backend == "npu_sdpa":
            import torch.nn.functional as F

            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=layer.scale,
                enable_gqa=True,
            )
            return output.squeeze(0).transpose(0, 1).reshape(num_tokens, num_heads * layer.head_dim)
        if self.backend == "npu_fia":
            output = torch_npu.npu_fused_infer_attention_score(
                query=q,
                key=k,
                value=v,
                atten_mask=causal_mask,
                block_table=None,
                input_layout="TND",
                block_size=128,
                actual_seq_lengths=[num_tokens],
                actual_seq_lengths_kv=[num_tokens],
                num_key_value_heads=num_kv_heads,
                num_heads=num_heads,
                scale=layer.scale,
                pre_tokens=2147483647,
                next_tokens=0,
                sparse_mode=3,
            )[0]
            return output.reshape(num_tokens, num_heads * layer.head_dim)
        output = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=num_heads,
            input_layout="TND",
            atten_mask=causal_mask,
            scale=layer.scale,
            pre_tockens=2147483647,
            next_tockens=0,
            actual_seq_qlen=[num_tokens],
            actual_seq_kvlen=[num_tokens],
            sparse_mode=3,
        )[0]
        return output.reshape(num_tokens, num_heads * layer.head_dim)

    def _execute_torch_eager(self, q: Any, k: Any, v: Any, layer: Any) -> Any:
        import torch

        num_tokens, num_heads, head_dim = q.shape
        num_kv_heads = k.shape[1]
        group_size = num_heads // num_kv_heads
        causal_mask = self._causal_masks.get(num_tokens)
        if causal_mask is None:
            causal_mask = torch.triu(torch.ones((num_tokens, num_tokens), dtype=torch.bool), diagonal=1).to(self.device)
            self._causal_masks[num_tokens] = causal_mask

        output = torch.empty_like(q)
        for head_start in range(0, num_heads, 8):
            head_end = min(head_start + 8, num_heads)
            q_group = q[:, head_start:head_end, :].transpose(0, 1)
            kv_index = head_start // group_size
            k_group = k[:, kv_index, :].unsqueeze(0).expand(head_end - head_start, -1, -1)
            v_group = v[:, kv_index, :].unsqueeze(0).expand(head_end - head_start, -1, -1)
            scores = torch.matmul(q_group, k_group.transpose(1, 2)) * layer.scale
            scores = scores.float().masked_fill(causal_mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            output[:, head_start:head_end, :] = torch.matmul(weights, v_group).transpose(0, 1)
        return output.reshape(num_tokens, num_heads * head_dim)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare xLLM Qwen3-VL NPU layer-50 output with H3 golden data.")
    parser.add_argument("--text-checkpoint", type=Path, required=True)
    parser.add_argument("--golden-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--max-abs-error", type=float, default=0.25)
    parser.add_argument(
        "--attention-backend", choices=["npu_fusion", "npu_sdpa", "npu_fia", "torch_eager"], default="npu_fusion"
    )
    parser.add_argument("--isolated-only", action="store_true")
    parser.add_argument("--isolated-layer", type=int)
    parser.add_argument("--residual-mode", choices=["exact", "fused", "hybrid"], default="hybrid")
    parser.add_argument("--activation-backend", choices=["xllm_swiglu", "torch_eager"], default="xllm_swiglu")
    parser.add_argument("--norm-backend", choices=["xllm", "torch_eager"], default="xllm")
    parser.add_argument("--split-projections", action="store_true")
    parser.add_argument("--reset-interval", type=int, default=0)
    parser.add_argument("--gate-mode", choices=["report_only", "legacy_max_abs"], default="report_only")
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


def _ordered_bf16(value: Any) -> Any:
    import torch

    bits = value.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    magnitude = bits & 0x7FFF
    return torch.where((bits & 0x8000) != 0, 0x8000 - magnitude, 0x8000 + magnitude)


def _metric_values(value: Any, mask: Any | None) -> Any:
    import torch

    if mask is None:
        return value.reshape(-1)
    if mask.ndim != 1:
        raise ValueError("metric mask must be a one-dimensional boolean tensor")
    if mask.dtype != torch.bool or value.ndim == 0 or value.shape[0] != mask.numel():
        raise ValueError("metric mask must be boolean and match the tensor's first dimension")
    return value[mask].reshape(-1)


def _metric_summary(actual: Any, expected: Any, mask: Any | None = None) -> dict[str, Any]:
    import torch

    actual_values = _metric_values(actual, mask)
    expected_values = _metric_values(expected, mask)
    count = actual_values.numel()
    if count == 0:
        return {"count": 0, "max_abs_error": 0.0, "mean_abs_error": 0.0}

    chunk_size = 1 << 20
    sample_limit = 1_000_000
    sample_step = max(1, math.ceil(count / sample_limit))
    sampled_errors: list[torch.Tensor] = []
    sum_abs = 0.0
    sum_error_squared = 0.0
    sum_actual_squared = 0.0
    sum_expected_squared = 0.0
    sum_dot = 0.0
    max_abs_error = 0.0
    max_error_flat_index = 0
    nonzero_error_count = 0
    envelope_outlier_count = 0
    nonfinite_mismatch_count = 0
    bf16_ulp_max = 0
    bf16_sign_crossing_count = 0
    bf16_same_sign_ulp_max = 0

    for start in range(0, count, chunk_size):
        actual_chunk = actual_values[start : start + chunk_size]
        expected_chunk = expected_values[start : start + chunk_size]
        finite = torch.isfinite(actual_chunk) & torch.isfinite(expected_chunk)
        both_nan = torch.isnan(actual_chunk) & torch.isnan(expected_chunk)
        same_nonfinite = both_nan | (actual_chunk == expected_chunk)
        nonfinite_mismatch_count += int((~finite & ~same_nonfinite).sum().item())

        difference = torch.zeros_like(actual_chunk, dtype=torch.float32)
        if bool(finite.any().item()):
            actual_float = actual_chunk[finite].float()
            expected_float = expected_chunk[finite].float()
            difference[finite] = (actual_float - expected_float).abs()
            delta = actual_float.double() - expected_float.double()
            sum_abs += float(difference[finite].double().sum().item())
            sum_error_squared += float((delta * delta).sum().item())
            sum_actual_squared += float((actual_float.double() * actual_float.double()).sum().item())
            sum_expected_squared += float((expected_float.double() * expected_float.double()).sum().item())
            sum_dot += float((actual_float.double() * expected_float.double()).sum().item())

        chunk_max, chunk_index = difference.max(dim=0)
        chunk_max_value = float(chunk_max.item())
        if chunk_max_value > max_abs_error:
            max_abs_error = chunk_max_value
            max_error_flat_index = start + int(chunk_index.item())
        nonzero_error_count += int(torch.count_nonzero(difference).item())
        envelope_outlier_count += int((difference > (0.25 + 0.0 * expected_chunk.float().abs())).sum().item())

        local_start = (-start) % sample_step
        sampled_errors.append(difference[local_start::sample_step].cpu())

        if actual_chunk.dtype == torch.bfloat16 and expected_chunk.dtype == torch.bfloat16:
            actual_bits = actual_chunk.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
            expected_bits = expected_chunk.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
            actual_magnitude = actual_bits & 0x7FFF
            expected_magnitude = expected_bits & 0x7FFF
            sign_crossing = (
                (((actual_bits ^ expected_bits) & 0x8000) != 0)
                & (actual_magnitude != 0)
                & (expected_magnitude != 0)
                & finite
            )
            bf16_sign_crossing_count += int(sign_crossing.sum().item())
            ordered_actual = _ordered_bf16(actual_chunk)
            ordered_expected = _ordered_bf16(expected_chunk)
            ulp = (ordered_actual - ordered_expected).abs()
            same_sign_ulp = ulp[finite & ~sign_crossing]
            if same_sign_ulp.numel():
                bf16_same_sign_ulp_max = max(bf16_same_sign_ulp_max, int(same_sign_ulp.max().item()))
                bf16_ulp_max = max(bf16_ulp_max, int(same_sign_ulp.max().item()))

    sampled = torch.cat(sampled_errors)
    quantiles = torch.quantile(sampled, torch.tensor([0.99, 0.9999], dtype=torch.float32))
    denominator = float(count)
    relative_l2 = math.sqrt(sum_error_squared / sum_expected_squared) if sum_expected_squared else None
    cosine = None
    if sum_actual_squared and sum_expected_squared:
        cosine = sum_dot / math.sqrt(sum_actual_squared * sum_expected_squared)
        cosine = max(-1.0, min(1.0, cosine))
    return {
        "count": count,
        "max_abs_error": max_abs_error,
        "mean_abs_error": sum_abs / denominator,
        "nonzero_error_count": nonzero_error_count,
        "max_error_flat_index": max_error_flat_index,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "abs_error_percentiles": {
            "p99": float(quantiles[0].item()),
            "p99_99": float(quantiles[1].item()),
        },
        "percentile_sample_count": int(sampled.numel()),
        "envelope": {
            "atol": 0.25,
            "rtol": 0.0,
            "outlier_count": envelope_outlier_count,
            "outlier_ratio": envelope_outlier_count / denominator,
        },
        "nonfinite_mismatch_count": nonfinite_mismatch_count,
        "bf16": {
            "same_sign_ulp_max": bf16_same_sign_ulp_max,
            "ulp_max_excluding_sign_crossings": bf16_ulp_max,
            "sign_crossing_count": bf16_sign_crossing_count,
        },
    }


def _compare(
    name: str,
    actual: Any,
    expected: Any,
    modality_masks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    actual_cpu = actual.detach().to("cpu")
    expected_cpu = expected.detach().to("cpu")
    if actual_cpu.shape != expected_cpu.shape:
        return {
            "name": name,
            "shape_match": False,
            "actual_shape": list(actual_cpu.shape),
            "expected_shape": list(expected_cpu.shape),
        }
    metrics = _metric_summary(actual_cpu, expected_cpu)
    result = {
        "name": name,
        "shape_match": True,
        "actual_shape": list(actual_cpu.shape),
        "expected_shape": list(expected_cpu.shape),
        "actual_dtype": str(actual.dtype),
        "expected_dtype": str(expected.dtype),
        "actual_sha256": _digest(actual),
        "expected_sha256": _digest(expected),
        **metrics,
    }
    if modality_masks is not None and actual_cpu.ndim > 0:
        result["modalities"] = {
            modality: _metric_summary(actual_cpu, expected_cpu, mask) for modality, mask in modality_masks.items()
        }
    return result


def _legacy_max_abs_pass(comparisons: list[dict[str, Any]], threshold: float) -> bool:
    return all(
        item["shape_match"] and item.get("nonfinite_mismatch_count", 0) == 0 and item["max_abs_error"] <= threshold
        for item in comparisons
    )


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
    deepstack_layers = len(model.model.deepstack_input_embeds or [])
    reset_interval = getattr(model.model, "teacher_force_reset_interval", 0)
    teacher_force_states = getattr(model.model, "teacher_force_states", None)
    for layer_index, layer in enumerate(model.model.layers):
        if reset_interval and layer_index > 0 and layer_index % reset_interval == 0:
            if teacher_force_states is None:
                raise RuntimeError("teacher-forcing states are not initialized")
            reset_key = f"layer_{layer_index - 1}_output_pre_final_norm"
            hidden = teacher_force_states[reset_key].squeeze(0).to(device)
            if layer_index - 1 < deepstack_layers:
                hidden = hidden + model.model.deepstack_input_embeds[layer_index - 1]
            residual = None
        use_exact_residual = model.model.exact_residual or (
            model.model.hybrid_residual and layer_index < deepstack_layers
        )
        if model.model.hybrid_residual and layer_index == deepstack_layers:
            residual = None
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
            if use_exact_residual:
                residual = hidden
                hidden = layer.input_layernorm(hidden)
                hidden = layer.self_attn(positions, hidden, cos_sin_cache, cos, sin, mrope_section)
                residual = residual + hidden
                hidden = layer.post_attention_layernorm(residual)
                hidden = residual + layer.mlp(hidden)
            else:
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
                if use_exact_residual:
                    hidden = hidden + model.model.deepstack_input_embeds[layer_index]
                else:
                    hidden = hidden + model.model.deepstack_input_embeds[layer_index]
            model.model.aux_hidden_capture.capture_layer(
                layer_index,
                hidden,
                None if use_exact_residual else residual,
                aux_hidden_buffer,
            )
            full_hidden = hidden if use_exact_residual else hidden + residual
            if layer_index == 0:
                layer_0_output = full_hidden
            if layer_index == 49:
                layer_49_output = full_hidden
            if reset_interval or layer_index in (0, 1, 2, 3, 4, 8, 16) or 32 <= layer_index < 50:
                selected_outputs[f"layer_{layer_index}_output_pre_final_norm"] = full_hidden
        finally:
            for hook in hooks:
                hook.remove()
            layer.to("cpu")

    model.model.deepstack_input_embeds = None
    if model.model.exact_residual:
        final_hidden = model.model.norm(hidden)
    else:
        final_hidden, _ = model.model.norm(hidden, residual)
    if layer_0_output is None or layer_49_output is None:
        raise RuntimeError("Qwen layerwise execution did not reach layers 0 and 49")
    finalized = model.model.aux_hidden_capture.finalize(final_hidden, aux_hidden_buffer)
    if not isinstance(finalized, tuple):
        raise RuntimeError("Qwen layerwise execution did not produce captured hidden states")
    return finalized[0], finalized[1], layer_0_output, selected_outputs, layer_43_nodes


def _official_layer_output(golden: dict[str, Any], layer_index: int, image_positions: Any) -> Any:
    if layer_index < 0:
        return golden["input_embeddings"].squeeze(0)
    name = f"layer_{layer_index}_output_pre_final_norm"
    states = golden.get("selected_hidden_states", {})
    if name not in states:
        raise ValueError(f"official golden does not contain {name}; use --capture-all-layers")
    hidden = states[name].squeeze(0)
    if layer_index < len(golden["visual_deepstack_features"]):
        hidden = hidden.clone()
        hidden[image_positions] += golden["visual_deepstack_features"][layer_index]
    return hidden


def _run_isolated_layer(
    model: Any,
    hidden: Any,
    positions: Any,
    device: Any,
    layer_index: int,
) -> tuple[Any, dict[str, Any]]:
    import torch

    positions = positions.to(torch.int64).contiguous()
    cos_sin_cache = model.model._mrope_cos_sin
    nodes: dict[str, Any] = {}

    def record(name: str, value: Any) -> None:
        nodes[name] = value.detach().cpu()

    def capture(name: str) -> Any:
        def callback(module: Any, args: tuple[Any, ...], output: Any) -> None:
            del module, args
            value = output[0] if isinstance(output, tuple) else output
            record(name, value)

        return callback

    layer = model.model.layers[layer_index]
    layer.to(device)
    layer.self_attn.diagnostic_callback = record
    layer.mlp.diagnostic_callback = record
    hooks = [
        layer.input_layernorm.register_forward_hook(capture("input_norm")),
        layer.self_attn.register_forward_hook(capture("self_attention")),
        layer.post_attention_layernorm.register_forward_hook(capture("post_attention_norm")),
        layer.mlp.register_forward_hook(capture("mlp")),
    ]
    try:
        hidden = hidden.to(device)
        residual = hidden
        hidden = layer.input_layernorm(hidden)
        hidden = layer.self_attn(
            positions,
            hidden,
            cos_sin_cache,
            None,
            None,
            model.model.mrope_section,
        )
        residual = residual + hidden
        record("first_residual", residual)
        hidden = layer.post_attention_layernorm(residual)
        output = residual + layer.mlp(hidden)
        record("second_residual", output)
        if model.model.deepstack_input_embeds is not None and layer_index < len(model.model.deepstack_input_embeds):
            output = output + model.model.deepstack_input_embeds[layer_index]
            record("output_after_deepstack", output)
    finally:
        layer.self_attn.diagnostic_callback = None
        layer.mlp.diagnostic_callback = None
        for hook in hooks:
            hook.remove()
        layer.to("cpu")
    return output, nodes


def main() -> None:
    args = _parse_args()
    text_checkpoint = args.text_checkpoint.resolve()
    golden_path = args.golden_path.resolve()
    output_dir = args.output_dir.resolve()
    isolated_layer = args.isolated_layer
    if args.isolated_only:
        if isolated_layer is not None and isolated_layer != 43:
            raise ValueError("--isolated-only is an alias for --isolated-layer 43")
        isolated_layer = 43
    if isolated_layer is not None and not 0 <= isolated_layer < 64:
        raise ValueError("isolated layer must be in [0, 63]")
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
    from xllm.python import kernels
    from xllm.python.model_executor.forward_context import ForwardContext, forward_context
    from xllm.python.models.qwen3_vl import Qwen3VLForConditionalGeneration

    if args.activation_backend == "torch_eager":

        def torch_silu_and_mul(value: Any) -> Any:
            import torch.nn.functional as F

            gate, up = value.chunk(2, dim=-1)
            return F.silu(gate) * up

        kernels.silu_and_mul = torch_silu_and_mul

    if args.norm_backend == "torch_eager":

        def torch_rms_norm(value: Any, weight: Any, eps: float) -> Any:
            input_dtype = value.dtype
            normalized = value.float()
            variance = normalized.square().mean(dim=-1, keepdim=True)
            normalized = normalized * torch.rsqrt(variance + eps)
            return weight * normalized.to(input_dtype)

        kernels.rms_norm = torch_rms_norm

    golden = torch.load(golden_path, map_location="cpu", weights_only=True)
    if args.reset_interval < 0:
        raise ValueError("reset interval must be non-negative")
    if args.reset_interval:
        states = golden.get("selected_hidden_states", {})
        missing = [
            f"layer_{index}_output_pre_final_norm"
            for index in range(63)
            if index % args.reset_interval == args.reset_interval - 1
        ]
        missing = [name for name in missing if name not in states]
        if missing:
            raise ValueError("teacher-forcing golden is missing reset states: " + ", ".join(missing[:3]))
    config = _load_flat_config(text_checkpoint, "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "device": str(device),
        "golden_path": str(golden_path),
        "text_checkpoint": str(text_checkpoint),
        "layers_to_capture": [49],
        "max_abs_error": args.max_abs_error,
        "attention_backend": args.attention_backend,
        "residual_mode": args.residual_mode,
        "activation_backend": args.activation_backend,
        "norm_backend": args.norm_backend,
        "split_projections": args.split_projections,
        "reset_interval": args.reset_interval,
        "gate_mode": args.gate_mode,
        "isolated_layer": isolated_layer,
        "torch_version": torch.__version__,
        "xllm_commit": os.popen("git -C /data/workspace/lwd/xllm rev-parse HEAD").read().strip(),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    logger.info("Constructing xLLM Qwen3-VL on CPU")
    model = Qwen3VLForConditionalGeneration(config)
    model.model.exact_residual = args.residual_mode == "exact"
    model.model.hybrid_residual = args.residual_mode == "hybrid"
    if args.split_projections:
        for layer in model.model.layers:
            layer.self_attn.split_qkv = True
            layer.mlp.split_gate_up = True
    if args.norm_backend == "torch_eager":
        for layer in model.model.layers:
            layer.self_attn.use_qk_norm_modules = True
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
    model.model.teacher_force_reset_interval = args.reset_interval
    model.model.teacher_force_states = golden.get("selected_hidden_states")
    merged_input_embeddings = model.model._inputs_embeds.detach().clone()
    h3_token_tags = golden["h3_token_tags"].to(torch.int64)
    modality_masks = {
        "qwen_image": image_positions,
        "qwen_text": ~image_positions,
        "h3_vision_segment": h3_token_tags == 0,
        "h3_text": h3_token_tags == 1,
    }

    def compare_tensors(name: str, actual: Any, expected: Any) -> dict[str, Any]:
        return _compare(name, actual, expected, modality_masks)

    metadata = SimpleNamespace(is_prefill=True, is_chunked_prefill=False)
    attention_backend = _NpuPrefillAttention(device, args.attention_backend)
    attention_backend.prepare(metadata)
    context = ForwardContext(attention_backend, device, metadata, [])

    if device.type == "npu":
        torch.npu.reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    with torch.no_grad(), forward_context(context):
        if isolated_layer is not None:
            output = None
            isolated_input = _official_layer_output(golden, isolated_layer - 1, image_positions)
            isolated_output, isolated_nodes = _run_isolated_layer(
                model,
                isolated_input,
                position_ids.to(device),
                device,
                isolated_layer,
            )
        else:
            output = _run_layerwise_qwen(model, input_ids.to(device), position_ids.to(device), device)
            isolated_input = _official_layer_output(golden, 42, image_positions)
            isolated_output, isolated_nodes = _run_isolated_layer(
                model,
                isolated_input,
                position_ids.to(device),
                device,
                43,
            )
    torch.npu.synchronize(device)
    inference_seconds = time.perf_counter() - started_at
    if isolated_layer is not None:
        expected_output = _official_layer_output(golden, isolated_layer, image_positions)
        comparisons = [compare_tensors(f"isolated_layer_{isolated_layer}_output", isolated_output, expected_output)]
        expected_nodes = golden.get(f"layer_{isolated_layer}_nodes")
        if expected_nodes is None and isolated_layer == 43:
            expected_nodes = golden.get("layer_43_nodes")
        expected_names = {
            "input_norm": "input_norm",
            "self_attention": "self_attention",
            "post_attention_norm": "post_attention_norm",
            "gate_projection": "gate_proj",
            "up_projection": "up_proj",
            "mlp": "mlp",
        }
        if expected_nodes is not None:
            for name, actual in isolated_nodes.items():
                expected_name = expected_names.get(name)
                if expected_name is None or expected_name not in expected_nodes:
                    continue
                expected = expected_nodes[expected_name].squeeze(0)
                comparisons.append(compare_tensors(f"isolated_layer_{isolated_layer}_{name}", actual, expected))
            if isolated_layer == 43 and "silu_and_mul" in isolated_nodes:
                import torch.nn.functional as F

                expected_activation = F.silu(expected_nodes["gate_proj"].squeeze(0)) * expected_nodes[
                    "up_proj"
                ].squeeze(0)
                comparisons.append(
                    compare_tensors(
                        f"isolated_layer_{isolated_layer}_silu_and_mul",
                        isolated_nodes["silu_and_mul"],
                        expected_activation,
                    )
                )
        legacy_pass = _legacy_max_abs_pass(comparisons, args.max_abs_error)
        summary = {
            "status": (
                f"H3_QWEN_LAYER{isolated_layer}_ISOLATED_PASS"
                if args.gate_mode == "legacy_max_abs" and legacy_pass
                else f"H3_QWEN_LAYER{isolated_layer}_ISOLATED_REPORT"
            ),
            "gate": {
                "mode": args.gate_mode,
                "legacy_max_abs_passed": legacy_pass,
                "native_status": "PENDING_DOWNSTREAM_GATE",
            },
            "mode": f"isolated_layer_{isolated_layer}",
            "layer_index": isolated_layer,
            "attention_backend": args.attention_backend,
            "inference_seconds": inference_seconds,
            "comparisons": comparisons,
            "captured_nodes": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": _digest(value),
                }
                for name, value in isolated_nodes.items()
            },
            "peak_device_memory_bytes": int(torch.npu.max_memory_allocated(device)),
            "peak_device_reserved_bytes": int(torch.npu.max_memory_reserved(device)),
        }
        torch.save(
            {
                f"isolated_layer_{isolated_layer}_output": isolated_output.cpu(),
                **{f"isolated_layer_{isolated_layer}_{name}": value for name, value in isolated_nodes.items()},
            },
            output_dir / f"qwen_layer{isolated_layer}_isolated_xllm.pt",
        )
        (output_dir / f"isolated_layer{isolated_layer}_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for comparison in comparisons:
            logger.info(
                f"{comparison['name']}: shape_match={comparison['shape_match']} "
                f"max_abs_error={comparison.get('max_abs_error', 'n/a')}"
            )
        if args.gate_mode == "legacy_max_abs" and not legacy_pass:
            raise RuntimeError("xLLM isolated layer-43 comparison exceeded the configured error threshold")
        return
    if not isinstance(output, tuple) or len(output) != 5:
        raise RuntimeError("xLLM Qwen3-VL did not return the expected hidden states with capture enabled")
    final_hidden, layer_49_output, layer_0_output, selected_outputs, layer_43_nodes = output

    comparisons = [
        compare_tensors("merged_input_embeddings", merged_input_embeddings, golden["input_embeddings"].squeeze(0)),
        compare_tensors("final_norm_output", final_hidden, golden["final_norm_output"].squeeze(0)),
    ]
    for name, actual in selected_outputs.items():
        layer_index = int(name.split("_")[1])
        expected = golden["selected_hidden_states"][name].squeeze(0)
        if layer_index < 3:
            expected = expected.clone()
            expected[image_positions] += golden["visual_deepstack_features"][layer_index]
        comparisons.append(compare_tensors(name, actual, expected))
    for name, actual in layer_43_nodes.items():
        if name == "gate_up_proj":
            expected = torch.cat(
                [golden["layer_43_nodes"]["gate_proj"], golden["layer_43_nodes"]["up_proj"]], dim=-1
            ).squeeze(0)
        else:
            expected = golden["layer_43_nodes"][name].squeeze(0)
        if name not in ("gate_proj", "up_proj"):
            comparisons.append(compare_tensors(f"layer_43_{name}", actual, expected))
    comparisons.append(
        compare_tensors(
            "isolated_layer_43_output",
            isolated_output,
            golden["selected_hidden_states"]["layer_43_output_pre_final_norm"].squeeze(0),
        )
    )
    expected_names = {
        "input_norm": "input_norm",
        "self_attention": "self_attention",
        "post_attention_norm": "post_attention_norm",
        "gate_projection": "gate_proj",
        "up_projection": "up_proj",
        "mlp": "mlp",
    }
    for name, actual in isolated_nodes.items():
        expected_name = expected_names.get(name)
        if expected_name is None or expected_name not in golden["layer_43_nodes"]:
            continue
        expected = golden["layer_43_nodes"][expected_name].squeeze(0)
        comparisons.append(compare_tensors(f"isolated_layer_43_{name}", actual, expected))
    if "silu_and_mul" in isolated_nodes:
        import torch.nn.functional as F

        expected_activation = F.silu(golden["layer_43_nodes"]["gate_proj"].squeeze(0)) * golden["layer_43_nodes"][
            "up_proj"
        ].squeeze(0)
        comparisons.append(
            compare_tensors("isolated_layer_43_silu_and_mul", isolated_nodes["silu_and_mul"], expected_activation)
        )
    if "gate_up_proj" in layer_43_nodes:
        import torch.nn.functional as F

        gate_up = layer_43_nodes["gate_up_proj"]
        actual_activation = torch.ops.xllm_ops.silu_and_mul(gate_up.to(device)).cpu()
        expected_activation = F.silu(golden["layer_43_nodes"]["gate_proj"].squeeze(0)) * golden["layer_43_nodes"][
            "up_proj"
        ].squeeze(0)
        comparisons.append(compare_tensors("layer_43_silu_and_mul", actual_activation, expected_activation))
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
    model_state["isolated_layer_43_output"] = isolated_output.cpu()
    model_state.update({f"isolated_layer_43_{name}": value for name, value in isolated_nodes.items()})
    torch.save(model_state, output_dir / "qwen_layer50_xllm.pt")
    legacy_pass = _legacy_max_abs_pass(comparisons, args.max_abs_error)
    summary = {
        "status": (
            "H3_QWEN_LAYER50_NPU_PASS"
            if args.gate_mode == "legacy_max_abs" and legacy_pass
            else (
                "H3_QWEN_LAYER50_NATIVE_PENDING_DOWNSTREAM_GATE"
                if args.gate_mode == "report_only"
                else "H3_QWEN_LAYER50_NPU_DIVERGED"
            )
        ),
        "gate": {
            "mode": args.gate_mode,
            "legacy_max_abs_threshold": args.max_abs_error,
            "legacy_max_abs_passed": legacy_pass,
            "native_status": "PENDING_DOWNSTREAM_CONDITION_GATE",
        },
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
    if args.gate_mode == "legacy_max_abs" and not legacy_pass:
        raise RuntimeError("xLLM Qwen3-VL layer-50 comparison exceeded the configured error threshold")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("xLLM MiniMax-H3 Qwen layer-50 run failed")
        raise
