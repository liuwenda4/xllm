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
"""Generate the official MiniMax-H3 C4 single-block Golden on one NPU."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger

PINNED_VLLM_REVISION = "e540dfb72f0439be08c26fe61912d8df572fdc7c"
PINNED_DIFFUSERS_REVISION = "d30c748f5f5d0925a5af14dc0e6a6de983025e63"
VLLM_TRANSFORMER_SHA256 = "41cdedd5bfd18298b24e28bb58d498d18a65513466715415675a89fbbf07cbef"
DIFFUSERS_TRANSFORMER_SHA256 = "1926b1bc15a5bebda05e3dc8cde1b3955d56641ba7f8f9d6e90c8f78c66ae30c"
DEFAULT_TRANSFORMER_PATH = Path("/data/workspace/lwd/minimax/checkpoints/Ref2VA-diffusers-d30c748f/transformer_ref")
DEFAULT_PACKING_GOLDEN = Path(
    "/data/workspace/lwd/minimax/artifacts/h3/20260910T223223Z-h3-c3/minimax_h3_packing_reference.safetensors"
)
DEFAULT_VLLM_SOURCE = Path("/data/workspace/lwd/vllm-omni")
TENSOR_ARCHIVE = "minimax_h3_block_reference.safetensors"
MANIFEST_NAME = "minimax_h3_block_reference.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-path", type=Path, default=DEFAULT_TRANSFORMER_PATH)
    parser.add_argument("--packing-golden", type=Path, default=DEFAULT_PACKING_GOLDEN)
    parser.add_argument("--vllm-source", type=Path, default=DEFAULT_VLLM_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_digest(tensor: Any) -> str:
    import torch

    value = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    value = tensor.detach().to("cpu").contiguous()
    finite = torch.isfinite(value) if value.is_floating_point() else torch.ones_like(value, dtype=torch.bool)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "numel": value.numel(),
        "finite": bool(finite.all().item()),
        "sha256": _tensor_digest(value),
    }


def _canonical_digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_manifest(vllm_source: Path, official_class: type[Any]) -> dict[str, Any]:
    vllm_path = vllm_source / "vllm_omni" / "diffusion" / "models" / "minimax_h3" / "minimax_h3_transformer.py"
    revision = _git_revision(vllm_source)
    if revision != PINNED_VLLM_REVISION:
        raise ValueError(f"vllm-omni revision mismatch: {revision}")
    vllm_digest = _sha256(vllm_path)
    if vllm_digest != VLLM_TRANSFORMER_SHA256:
        raise ValueError(f"vllm-omni transformer source digest mismatch: {vllm_digest}")

    diffusers_source_value = inspect.getsourcefile(official_class)
    if diffusers_source_value is None:
        raise ValueError("cannot locate the official Diffusers transformer source")
    diffusers_source = Path(diffusers_source_value).resolve()
    diffusers_digest = _sha256(diffusers_source)
    if diffusers_digest != DIFFUSERS_TRANSFORMER_SHA256:
        raise ValueError(f"Diffusers transformer source digest mismatch: {diffusers_digest}")
    source = {
        "vllm_omni": {
            "revision": revision,
            "path": str(vllm_path.resolve()),
            "sha256": vllm_digest,
        },
        "diffusers": {
            "revision": PINNED_DIFFUSERS_REVISION,
            "path": str(diffusers_source),
            "sha256": diffusers_digest,
        },
    }
    source["digest"] = _canonical_digest(source)
    return source


def _checkpoint_manifest(transformer_path: Path) -> dict[str, Any]:
    config_path = transformer_path / "config.json"
    index_path = transformer_path / "diffusion_pytorch_model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("converted transformer index has no weight_map")
    shard_names = sorted(set(weight_map.values()))
    files = [config_path, index_path, *(transformer_path / name for name in shard_names)]
    for path in files:
        if not path.is_file():
            raise ValueError(f"converted transformer file does not exist: {path}")
    logger.info(f"Hashing {len(files)} converted transformer files")
    inventory = {path.name: {"size": path.stat().st_size, "sha256": _sha256(path)} for path in files}
    return {
        "path": str(transformer_path),
        "total_size": int(index.get("metadata", {}).get("total_size", 0)),
        "tensor_count": len(weight_map),
        "files": inventory,
        "digest": _canonical_digest(inventory),
    }


def _packing_fixture(packing_golden: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors.torch import load_file

    archive = load_file(str(packing_golden), device="cpu")
    prefix = "one_image."
    required = (
        "condition_hidden",
        "input_ids",
        "img_pos",
        "audio_pos",
        "text_pos",
        "update_mask",
        "audio_update_mask",
        "position_ids",
        "token_tags",
        "cu_seqlens",
    )
    fixture = {}
    for name in required:
        key = prefix + name
        if key not in archive:
            raise ValueError(f"C3 packing Golden is missing {key!r}")
        fixture[name] = archive[key]
    used_length = int(fixture["cu_seqlens"][1].item())
    aligned_length = int(fixture["input_ids"].numel())
    if used_length != 23 or aligned_length != 64:
        raise ValueError(f"C4 requires the C3 one-image 23/64 fixture, got {used_length}/{aligned_length}")
    metadata = {
        "path": str(packing_golden),
        "sha256": _sha256(packing_golden),
        "used_length": used_length,
        "aligned_length": aligned_length,
    }
    packing_manifest = packing_golden.with_name("minimax_h3_packing_reference.json")
    if packing_manifest.is_file():
        metadata["manifest_path"] = str(packing_manifest)
        metadata["manifest_sha256"] = _sha256(packing_manifest)
    return fixture, metadata


def _deterministic_rows(row_count: int, width: int, offset: int) -> Any:
    import torch

    values = torch.arange(row_count * width, dtype=torch.float32)
    return values.add(offset).remainder(257).sub(128).div(128).reshape(row_count, width).contiguous()


def _move_c4_modules(model: Any, device: Any) -> list[str]:
    names = [
        "context_embedder",
        "token_refiner",
        "time_proj",
        "time_embedder",
        "rope",
        "proj_in",
        "audio_proj_in",
        "transformer_blocks.0",
        "norm_out",
        "proj_out",
        "audio_proj_out",
    ]
    for name in names:
        model.get_submodule(name).to(device)
    return names


def _run_token_refiner(model: Any, projected: Any, output: dict[str, Any]) -> Any:
    hidden = projected
    for index, block in enumerate(model.token_refiner.refiner_blocks):
        attention_delta = block.attn(block.norm1(hidden))
        after_attention = hidden + attention_delta
        mlp_delta = block.ff(block.norm2(after_attention))
        hidden = after_attention + mlp_delta
        output[f"token_refiner.block{index}.attention_delta"] = attention_delta.squeeze(0)
        output[f"token_refiner.block{index}.mlp_delta"] = mlp_delta.squeeze(0)
        output[f"token_refiner.block{index}.output"] = hidden.squeeze(0)
    return model.token_refiner.final_norm(hidden)


def _run_reference(model: Any, fixture: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    used = 23
    aligned = 64
    condition = fixture["condition_hidden"].to(device)
    image_positions = fixture["img_pos"].to(device)
    audio_positions = fixture["audio_pos"].to(device)
    text_positions = fixture["text_pos"].to(device)
    update_mask = fixture["update_mask"].to(device)
    audio_update_mask = fixture["audio_update_mask"].to(device)
    position_ids = fixture["position_ids"][:used].to(device)
    token_tags = fixture["token_tags"][:used].to(device)

    video_rows = _deterministic_rows(image_positions.numel(), 96, 0).to(device)
    audio_rows = _deterministic_rows(audio_positions.numel(), 32, 41).to(device)
    timesteps = torch.tensor([0.0, 0.35, 0.75], dtype=torch.float32, device=device)
    inverse_indices = torch.zeros(aligned, dtype=torch.int64, device=device)
    inverse_indices[7:11] = 1
    inverse_indices[11:23] = 2
    used_inverse = inverse_indices[:used]

    output: dict[str, Any] = {
        "input.condition_hidden": condition.squeeze(0),
        "input.video_rows": video_rows,
        "input.audio_rows": audio_rows,
        "input.timesteps": timesteps,
        "input.inverse_indices": inverse_indices,
        "layout.position_ids": fixture["position_ids"].to(device),
        "layout.token_tags": fixture["token_tags"].to(device),
        "layout.img_pos": image_positions,
        "layout.audio_pos": audio_positions,
        "layout.text_pos": text_positions,
        "layout.update_mask": update_mask,
        "layout.audio_update_mask": audio_update_mask,
        "layout.cu_seqlens": fixture["cu_seqlens"].to(device),
    }

    projected = model.context_embedder(condition)
    output["condition_projection"] = projected.squeeze(0)
    refined = _run_token_refiner(model, projected, output)
    output["refined_condition"] = refined.squeeze(0)

    video_embedding = model.proj_in(video_rows.unsqueeze(0))
    audio_embedding = model.audio_proj_in(audio_rows.unsqueeze(0))
    output["video_embedding"] = video_embedding.squeeze(0)
    output["audio_embedding"] = audio_embedding.squeeze(0)
    packed = torch.zeros((1, used, 5376), dtype=torch.bfloat16, device=device)
    packed = packed.index_copy(1, text_positions, refined)
    packed = packed.index_copy(1, image_positions, video_embedding.to(torch.bfloat16))
    packed = packed.index_copy(1, audio_positions, audio_embedding.to(torch.bfloat16))
    output["packed_hidden"] = packed.squeeze(0)

    time_frequency = model.time_proj(timesteps)
    time_linear1 = model.time_embedder.linear_1(time_frequency.to(model.time_embedder.linear_1.weight.dtype))
    time_silu = torch.nn.functional.silu(time_linear1)
    time_embedding = model.time_embedder.linear_2(time_silu)
    output["time_frequency"] = time_frequency
    output["time_linear1"] = time_linear1
    output["time_silu"] = time_silu
    output["time_embedding"] = time_embedding

    rope_cos, rope_sin = model.rope(position_ids)
    per_axis = position_ids.to(torch.float32).unsqueeze(-1) * model.rope.inv_freq.view(1, 1, -1)
    rope_half = per_axis.reshape(used, -1)
    rope_frequencies = torch.cat((rope_half, rope_half), dim=-1)
    output["rope_frequencies"] = rope_frequencies
    output["rope_cos"] = rope_cos
    output["rope_sin"] = rope_sin

    combined_indices = used_inverse * 3 + token_tags.clamp_min(0)
    output["combined_indices"] = combined_indices
    block = model.transformer_blocks[0]
    adaln_tuple = block.adaln_proj(time_embedding)
    adaln = torch.stack(adaln_tuple, dim=1).reshape(timesteps.numel(), 3, 6, 5376)
    output["block0.adaln"] = adaln
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_tuple

    attention_input = block.norm1(packed)
    attention_input = attention_input * (1.0 + scale_msa.index_select(0, combined_indices).unsqueeze(0))
    attention_input = attention_input + shift_msa.index_select(0, combined_indices).unsqueeze(0)
    attention_output = block.attn(attention_input, (rope_cos, rope_sin))
    attention_delta = gate_msa.index_select(0, combined_indices).unsqueeze(0) * attention_output
    after_attention = packed + attention_delta
    output["block0.attention_input"] = attention_input.squeeze(0)
    output["block0.attention_output"] = attention_output.squeeze(0)
    output["block0.attention_delta"] = attention_delta.squeeze(0)
    output["block0.after_attention"] = after_attention.squeeze(0)

    mlp_input = block.norm2(after_attention)
    mlp_input = mlp_input * (1.0 + scale_mlp.index_select(0, combined_indices).unsqueeze(0))
    mlp_input = mlp_input + shift_mlp.index_select(0, combined_indices).unsqueeze(0)
    mlp_output = block.ff(mlp_input)
    mlp_delta = gate_mlp.index_select(0, combined_indices).unsqueeze(0) * mlp_output
    block_output = after_attention + mlp_delta
    output["block0.mlp_input"] = mlp_input.squeeze(0)
    output["block0.mlp_output"] = mlp_output.squeeze(0)
    output["block0.mlp_delta"] = mlp_delta.squeeze(0)
    output["block0.output"] = block_output.squeeze(0)

    final_shift, final_scale = model.norm_out.linear(
        torch.nn.functional.silu(time_embedding).to(model.norm_out.linear.weight.dtype)
    ).chunk(2, dim=-1)
    final_activation = model.norm_out.norm(block_output)
    final_activation = final_activation * (1.0 + final_scale.index_select(0, used_inverse).unsqueeze(0))
    final_activation = final_activation + final_shift.index_select(0, used_inverse).unsqueeze(0)
    final_fp32 = final_activation.to(torch.float32)
    all_video_logits = model.proj_out(final_fp32).squeeze(0)
    all_audio_logits = model.audio_proj_out(final_fp32).squeeze(0)
    selected_video = all_video_logits.index_select(0, image_positions) * update_mask.unsqueeze(-1)
    selected_audio = all_audio_logits.index_select(0, audio_positions) * audio_update_mask.unsqueeze(-1)
    output["final_activation"] = final_activation.squeeze(0)
    output["all_video_logits"] = all_video_logits
    output["all_audio_logits"] = all_audio_logits
    output["selected_video_logits"] = selected_video
    output["selected_audio_logits"] = selected_audio

    return {name: tensor.detach().to("cpu").contiguous() for name, tensor in output.items()}


def _runtime_manifest(torch: Any, torch_npu: Any, diffusers: Any, device: Any) -> dict[str, Any]:
    runtime = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "diffusers": diffusers.__version__,
        "device": str(device),
        "device_name": torch.npu.get_device_name(device),
    }
    runtime["digest"] = _canonical_digest(runtime)
    return runtime


def _write_output(output_dir: Path, archive: dict[str, Any], manifest: dict[str, Any]) -> Path:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / TENSOR_ARCHIVE
    manifest_path = output_dir / MANIFEST_NAME
    suffix = f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    tensor_temp = output_dir / f".{TENSOR_ARCHIVE}{suffix}"
    manifest_temp = output_dir / f".{MANIFEST_NAME}{suffix}"
    try:
        save_file(archive, str(tensor_temp))
        manifest["artifact"] = {
            "path": TENSOR_ARCHIVE,
            "size": tensor_temp.stat().st_size,
            "sha256": _sha256(tensor_temp),
        }
        with manifest_temp.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tensor_temp, tensor_path)
        os.replace(manifest_temp, manifest_path)
    finally:
        tensor_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)
    return manifest_path


def main() -> None:
    args = _parse_args()
    transformer_path = args.transformer_path.resolve()
    packing_golden = args.packing_golden.resolve()
    vllm_source = args.vllm_source.resolve()
    output_dir = args.output_dir.resolve()
    if not transformer_path.is_dir() or not packing_golden.is_file() or not vllm_source.is_dir():
        raise ValueError("transformer, C3 packing Golden, and pinned vllm-omni paths must exist")
    if not args.device.startswith("npu"):
        raise ValueError("the official C4 Golden must run on one NPU")

    import diffusers
    import torch
    import torch_npu
    from diffusers.models.transformers import MiniMaxH3Transformer3DModel

    device = torch.device(args.device)
    torch.npu.set_device(device)
    source = _source_manifest(vllm_source, MiniMaxH3Transformer3DModel)
    fixture, packing = _packing_fixture(packing_golden)
    checkpoint = _checkpoint_manifest(transformer_path)

    logger.info(f"Loading official converted transformer from {transformer_path}")
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    moved_modules = _move_c4_modules(model, device)
    torch.npu.reset_peak_memory_stats(device)
    with torch.no_grad():
        archive = _run_reference(model, fixture, device)
    torch.npu.synchronize(device)
    for name, tensor in archive.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"official Golden tensor {name!r} contains NaN or Inf")

    manifest = {
        "schema": "xllm.minimax_h3.block_reference/v1",
        "status": "OFFICIAL_USED_ROWS_GOLDEN",
        "scope": "condition_proj+token_refiner+embeddings+time+rope+block0+final_heads",
        "source": source,
        "checkpoint": checkpoint,
        "runtime": _runtime_manifest(torch, torch_npu, diffusers, device),
        "packing": packing,
        "execution": {
            "moved_modules": moved_modules,
            "official_attention_rows": 23,
            "aligned_padding_rows": 41,
            "padding_policy": "not executed by official Diffusers; xLLM tests [0,23] and [23,64] isolation separately",
            "peak_npu_memory_allocated": int(torch.npu.max_memory_allocated(device)),
            "peak_npu_memory_reserved": int(torch.npu.max_memory_reserved(device)),
        },
        "tensors": {name: _tensor_summary(tensor) for name, tensor in sorted(archive.items())},
    }
    manifest_path = _write_output(output_dir, archive, manifest)
    print(manifest_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("MiniMax-H3 C4 Golden generation failed")
        raise
