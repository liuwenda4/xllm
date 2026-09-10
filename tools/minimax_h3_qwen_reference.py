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
"""Dump the official MiniMax-H3 Qwen3-VL layer-50 conditioning tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger
from tools.minimax_h3_reference import _file_digest, _tensor_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump MiniMax-H3's official Qwen3-VL hidden_states[50].")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Converted local modular checkpoint.")
    parser.add_argument("--image-path", type=Path, required=True, help="Local Ref2VA image reference.")
    parser.add_argument("--prompt-path", type=Path, required=True, help="UTF-8 Ref2VA prompt file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output tensor and summary directory.")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--reference-short-edge", type=int, default=2048)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--capture-all-layers", action="store_true")
    parser.add_argument("--trace-attention", action="store_true")
    return parser.parse_args()


def _normalize_reference(image_path: Path, short_edge: int) -> Any:
    from diffusers import VaeImageProcessor
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference

    image = MiniMaxH3ImageReference.from_file(str(image_path)).image
    width, height = image.size
    scale = short_edge / min(width, height)
    normalized_height = max(32, round(height * scale / 32) * 32)
    normalized_width = max(32, round(width * scale / 32) * 32)
    processor = VaeImageProcessor(vae_scale_factor=16)
    image = processor.resize(image, height=normalized_height, width=normalized_width)
    return MiniMaxH3ImageReference(image=image)


def _load_components(checkpoint_path: Path, device: Any) -> tuple[Any, Any, Any]:
    import torch
    from diffusers.hooks import apply_group_offloading
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        subfolder="text_encoder",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    tokenizer = Qwen2TokenizerFast.from_pretrained(checkpoint_path, subfolder="tokenizer", local_files_only=True)
    processor = Qwen3VLProcessor.from_pretrained(checkpoint_path, subfolder="processor", local_files_only=True)
    apply_group_offloading(
        text_encoder.model,
        onload_device=device,
        offload_device=torch.device("cpu"),
        offload_type="leaf_level",
        use_stream=False,
    )
    return text_encoder, tokenizer, processor


def main() -> None:
    args = _parse_args()
    checkpoint_path = args.checkpoint_path.resolve()
    image_path = args.image_path.resolve()
    prompt_path = args.prompt_path.resolve()
    output_dir = args.output_dir.resolve()
    for path in (checkpoint_path, image_path, prompt_path):
        if not path.exists():
            raise ValueError(f"Required input does not exist: {path}")

    import diffusers
    import torch
    import transformers
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
        torch.npu.reset_peak_memory_stats(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    reference = _normalize_reference(image_path, args.reference_short_edge)
    started_at = time.perf_counter()
    text_encoder, tokenizer, processor = _load_components(checkpoint_path, device)
    load_seconds = time.perf_counter() - started_at
    visual_capture: dict[str, Any] = {}

    def capture_visual_output(module: Any, args: tuple[Any, ...], output: Any) -> None:
        del module, args
        visual_capture["main"] = output.pooler_output.detach().cpu()
        visual_capture["deepstack"] = [value.detach().cpu() for value in (output.deepstack_features or [])]

    visual_hook = text_encoder.model.visual.register_forward_hook(capture_visual_output)
    layer_43_capture: dict[str, Any] = {}

    def capture_layer_43(name: str) -> Any:
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            del module, args
            value = output[0] if isinstance(output, tuple) else output
            layer_43_capture[name] = value.detach().cpu()

        return hook

    layer_43 = text_encoder.model.language_model.layers[43]
    layer_43_hooks = [
        layer_43.input_layernorm.register_forward_hook(capture_layer_43("input_norm")),
        layer_43.self_attn.register_forward_hook(capture_layer_43("self_attention")),
        layer_43.post_attention_layernorm.register_forward_hook(capture_layer_43("post_attention_norm")),
        layer_43.mlp.gate_proj.register_forward_hook(capture_layer_43("gate_proj")),
        layer_43.mlp.up_proj.register_forward_hook(capture_layer_43("up_proj")),
        layer_43.mlp.register_forward_hook(capture_layer_43("mlp")),
    ]

    encoder_step = MiniMaxH3Ref2VATextEncoderStep(video_sample_fps=2.0)
    vision_inputs, image_token_counts, video_token_counts, video_timestamps = encoder_step._gather_vision_features(
        processor,
        [reference],
        fps=args.fps,
    )
    token_ids, h3_token_tags = encoder_step._build_presentation(
        tokenizer,
        prompt,
        [reference],
        image_token_counts,
        video_token_counts,
        video_timestamps,
        text_tag=1,
        video_tag=0,
    )

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    mm_token_type_ids = torch.tensor(
        processor.create_mm_token_type_ids([token_ids]),
        dtype=torch.long,
        device=device,
    )
    vision_kwargs = {
        name: value.to(device, text_encoder.dtype) if name.startswith("pixel_") else value.to(device)
        for name, value in vision_inputs.items()
    }
    position_ids, rope_deltas = text_encoder.model.get_rope_index(
        input_ids,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=vision_kwargs.get("image_grid_thw"),
        video_grid_thw=vision_kwargs.get("video_grid_thw"),
        attention_mask=attention_mask,
    )

    logger.info(f"Running official Qwen3-VL conditioner with {len(token_ids)} tokens")
    inference_started_at = time.perf_counter()
    profiler = None
    step_hooks = []
    original_sdpa = None
    if args.trace_attention:
        import torch.nn.functional as F
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward

        attention = text_encoder.model.language_model.layers[43].self_attn
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            attention.config._attn_implementation, eager_attention_forward
        )
        logger.info(
            "Official attention backend: "
            f"interface={interface.__module__}.{interface.__qualname__} "
            f"implementation={attention.config._attn_implementation} "
            f"head_dim={attention.head_dim} heads={attention.config.num_attention_heads} "
            f"kv_heads={attention.config.num_key_value_heads} scale={attention.scaling}"
        )
        trace_path = output_dir / "h3_layer43_attention_trace.json"
        profiler = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=42, warmup=1, active=1, repeat=1),
            on_trace_ready=lambda profile: profile.export_chrome_trace(str(trace_path)),
            record_shapes=True,
            with_stack=False,
        )
        original_sdpa = F.scaled_dot_product_attention
        active = {"value": False}

        def logged_sdpa(*args: Any, **kwargs: Any) -> Any:
            if active["value"]:
                query, key, value = args[:3]
                logger.info(
                    "H3_SDPA "
                    + json.dumps(
                        {
                            "q_shape": list(query.shape),
                            "k_shape": list(key.shape),
                            "v_shape": list(value.shape),
                            "q_dtype": str(query.dtype),
                            "k_dtype": str(key.dtype),
                            "v_dtype": str(value.dtype),
                            "attn_mask": None if kwargs.get("attn_mask") is None else list(kwargs["attn_mask"].shape),
                            "dropout_p": kwargs.get("dropout_p", args[4] if len(args) > 4 else 0.0),
                            "is_causal": kwargs.get("is_causal", args[5] if len(args) > 5 else False),
                            "scale": kwargs.get("scale"),
                            "enable_gqa": kwargs.get("enable_gqa", False),
                        },
                        sort_keys=True,
                    )
                )
            return original_sdpa(*args, **kwargs)

        F.scaled_dot_product_attention = logged_sdpa

        def enable_layer43(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            del module, args, kwargs
            active["value"] = True

        def disable_layer43(module: Any, args: tuple[Any, ...], output: Any) -> None:
            del module, args, output
            active["value"] = False

        flag_hooks = [
            attention.register_forward_pre_hook(enable_layer43, with_kwargs=True),
            attention.register_forward_hook(disable_layer43),
        ]
        step_hooks = [
            layer.register_forward_hook(lambda module, args, output: profiler.step())
            for layer in text_encoder.model.language_model.layers
        ]
        profiler.__enter__()
    try:
        with torch.no_grad():
            outputs = text_encoder.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
                output_hidden_states=True,
                **vision_kwargs,
            )
    finally:
        if profiler is not None:
            profiler.__exit__(None, None, None)
        if original_sdpa is not None:
            F.scaled_dot_product_attention = original_sdpa
        for hook in step_hooks + (flag_hooks if args.trace_attention else []):
            hook.remove()
    visual_hook.remove()
    for hook in layer_43_hooks:
        hook.remove()
    if device.type == "npu":
        torch.npu.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started_at

    input_embeddings = outputs.hidden_states[0]
    layer_0_output = outputs.hidden_states[1]
    layer_49_output = outputs.hidden_states[50]
    final_norm_output = outputs.last_hidden_state
    selected_layer_indices = tuple(range(64)) if args.capture_all_layers else (0, 1, 2, 3, 4, 8, 16, *range(32, 50))
    selected_hidden_states = {
        f"layer_{index}_output_pre_final_norm": outputs.hidden_states[index + 1].cpu()
        for index in selected_layer_indices
    }
    tensors = {
        "input_ids": input_ids.cpu(),
        "attention_mask": attention_mask.cpu(),
        "mm_token_type_ids": mm_token_type_ids.cpu(),
        "h3_token_tags": torch.tensor(h3_token_tags, dtype=torch.long),
        "image_grid_thw": vision_kwargs["image_grid_thw"].cpu(),
        "pixel_values": vision_kwargs["pixel_values"].cpu(),
        "position_ids": position_ids[:, 0].cpu(),
        "rope_deltas": rope_deltas.cpu(),
        "input_embeddings": input_embeddings.cpu(),
        "layer_0_output": layer_0_output.cpu(),
        "layer_49_output_pre_final_norm": layer_49_output.cpu(),
        "final_norm_output": final_norm_output.cpu(),
        "visual_main_feature": visual_capture["main"],
        "visual_deepstack_features": visual_capture["deepstack"],
        "selected_hidden_states": selected_hidden_states,
        "layer_43_nodes": layer_43_capture,
    }
    tensor_path = output_dir / "qwen_layer50_reference.pt"
    torch.save(tensors, tensor_path)
    summary = {
        "status": "H3_QWEN_LAYER50_REFERENCE_PASS",
        "checkpoint_path": str(checkpoint_path),
        "image_path": str(image_path),
        "image_sha256": _file_digest(image_path),
        "prompt_path": str(prompt_path),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "device": str(device),
        "diffusers_version": diffusers.__version__,
        "transformers_version": transformers.__version__,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "tensor_archive_sha256": _file_digest(tensor_path),
        "input_embeddings": _tensor_summary(input_embeddings),
        "layer_0_output": _tensor_summary(layer_0_output),
        "layer_49_output_pre_final_norm": _tensor_summary(layer_49_output),
        "final_norm_output": _tensor_summary(final_norm_output),
        "visual_main_feature": _tensor_summary(visual_capture["main"]),
        "visual_deepstack_features": [_tensor_summary(value) for value in visual_capture["deepstack"]],
        "selected_hidden_states": {name: _tensor_summary(value) for name, value in selected_hidden_states.items()},
        "layer_43_nodes": {name: _tensor_summary(value) for name, value in layer_43_capture.items()},
        "input_ids": _tensor_summary(input_ids),
        "mm_token_type_ids": _tensor_summary(mm_token_type_ids),
        "h3_token_tags": _tensor_summary(tensors["h3_token_tags"]),
        "image_grid_thw": _tensor_summary(vision_kwargs["image_grid_thw"]),
        "pixel_values": _tensor_summary(vision_kwargs["pixel_values"]),
        "position_ids": _tensor_summary(position_ids[:, 0]),
        "rope_deltas": _tensor_summary(rope_deltas),
        "lm_head_executed": False,
    }
    if device.type == "npu":
        summary["peak_device_memory_bytes"] = int(torch.npu.max_memory_allocated(device))
        summary["peak_device_reserved_bytes"] = int(torch.npu.max_memory_reserved(device))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logger.info(f"Wrote Qwen layer-50 reference to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("MiniMax-H3 Qwen reference dump failed")
        raise
