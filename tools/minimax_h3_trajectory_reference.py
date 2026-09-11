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
"""Generate the official MiniMax-H3 Base 50-layer trajectory Golden."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger
from tools.minimax_h3_block_reference import (
    DEFAULT_PACKING_GOLDEN,
    DEFAULT_TRANSFORMER_PATH,
    DEFAULT_VLLM_SOURCE,
    PINNED_DIFFUSERS_REVISION,
    PINNED_VLLM_REVISION,
    _canonical_digest,
    _checkpoint_manifest,
    _deterministic_rows,
    _git_revision,
    _runtime_manifest,
    _sha256,
    _tensor_digest,
    _tensor_summary,
)

ARCHIVE_NAME = "minimax_h3_trajectory_reference.safetensors"
MANIFEST_NAME = "minimax_h3_trajectory_reference.json"
LAYER_DIGESTS_NAME = "layer_digests.jsonl"
DIFFUSERS_TRANSFORMER_SHA256 = "1926b1bc15a5bebda05e3dc8cde1b3955d56641ba7f8f9d6e90c8f78c66ae30c"
DIFFUSERS_SCHEDULER_SHA256 = "307d5bf755337ef00c47237f9ac8be116e627d26e1df3b5f0bd504a80f9de8dd"
VLLM_TRANSFORMER_SHA256 = "41cdedd5bfd18298b24e28bb58d498d18a65513466715415675a89fbbf07cbef"
VLLM_DENOISE_SHA256 = "4fa35d80713ff911a8bf8ed34eb6f67b6cddc5ee036dd08400680a5dcd48efa7"
VLLM_SCHEDULER_SHA256 = "e075d34a5415b0e72e91cbf25cf90cddeb037ec9397a2f560387661f2f3b192e"
BASE_POINT_COUNT = 50
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
IMAGE_CONDITION_TIMESTEP = 0.999
AUDIO_CONDITION_TIMESTEP = 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-path", type=Path, default=DEFAULT_TRANSFORMER_PATH)
    parser.add_argument("--packing-golden", type=Path, default=DEFAULT_PACKING_GOLDEN)
    parser.add_argument("--vllm-source", type=Path, default=DEFAULT_VLLM_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-devices", type=int, default=2)
    return parser.parse_args()


def _validate_source(path: Path, expected: str) -> dict[str, Any]:
    digest = _sha256(path)
    if digest != expected:
        raise ValueError(f"source digest mismatch for {path}: {digest}")
    return {"path": str(path.resolve()), "sha256": digest}


def _source_manifest(vllm_source: Path, official_class: type[Any], scheduler_class: type[Any]) -> dict[str, Any]:
    import inspect

    revision = _git_revision(vllm_source)
    if revision != PINNED_VLLM_REVISION:
        raise ValueError(f"vllm-omni revision mismatch: {revision}")
    diffusers_transformer = Path(inspect.getsourcefile(official_class) or "").resolve()
    diffusers_scheduler = Path(inspect.getsourcefile(scheduler_class) or "").resolve()
    vllm_root = vllm_source / "vllm_omni" / "diffusion" / "models" / "minimax_h3"
    source = {
        "diffusers": {
            "revision": PINNED_DIFFUSERS_REVISION,
            "transformer": _validate_source(diffusers_transformer, DIFFUSERS_TRANSFORMER_SHA256),
            "scheduler": _validate_source(diffusers_scheduler, DIFFUSERS_SCHEDULER_SHA256),
        },
        "vllm_omni": {
            "revision": revision,
            "transformer": _validate_source(vllm_root / "minimax_h3_transformer.py", VLLM_TRANSFORMER_SHA256),
            "denoise_loop": _validate_source(vllm_root / "denoise_loop.py", VLLM_DENOISE_SHA256),
            "scheduler": _validate_source(
                vllm_root / "scheduling_minimax_h3_euler_ancestral.py", VLLM_SCHEDULER_SHA256
            ),
        },
    }
    source["digest"] = _canonical_digest(source)
    return source


def _packing_fixture(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors.torch import load_file

    archive = load_file(str(path), device="cpu")
    prefix = "image_audio."
    names = (
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
    fixture: dict[str, Any] = {}
    for name in names:
        key = prefix + name
        if key not in archive:
            raise ValueError(f"C3 packing Golden is missing {key!r}")
        fixture[name] = archive[key]
    used = int(fixture["cu_seqlens"][1].item())
    aligned = int(fixture["input_ids"].numel())
    if used != 27 or aligned != 64:
        raise ValueError(f"C5 requires image_audio 27/64 fixture, got {used}/{aligned}")
    if bool(fixture["update_mask"].all()) or bool(fixture["audio_update_mask"].all()):
        raise ValueError("C5 fixture must contain both visual and audio anchors")
    return fixture, {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "case": "image_audio",
        "used_length": used,
        "aligned_length": aligned,
    }


def _row_timestep_plan(fixture: dict[str, Any], t_video: float, t_audio: float) -> tuple[Any, Any, Any]:
    import torch

    aligned = int(fixture["input_ids"].numel())
    row_timesteps = torch.full((aligned,), t_video, dtype=torch.float32)
    image_positions = fixture["img_pos"].view(-1).to(torch.long)
    audio_positions = fixture["audio_pos"].view(-1).to(torch.long)
    image_update = fixture["update_mask"].view(-1).to(torch.bool)
    audio_update = fixture["audio_update_mask"].view(-1).to(torch.bool)
    row_timesteps[image_positions[image_update]] = t_video
    row_timesteps[image_positions[~image_update]] = max(t_video, IMAGE_CONDITION_TIMESTEP)
    row_timesteps[audio_positions[audio_update]] = t_audio
    row_timesteps[audio_positions[~audio_update]] = max(t_audio, AUDIO_CONDITION_TIMESTEP)
    unique, inverse = torch.unique(row_timesteps, sorted=True, return_inverse=True)
    return row_timesteps, unique, inverse


def _place_model(model: Any, devices: list[Any]) -> dict[str, Any]:
    first = devices[0]
    last = devices[-1]
    for name in (
        "context_embedder",
        "token_refiner",
        "time_proj",
        "time_embedder",
        "rope",
        "proj_in",
        "audio_proj_in",
    ):
        model.get_submodule(name).to(first)
    for name in ("norm_out", "proj_out", "audio_proj_out"):
        model.get_submodule(name).to(last)

    block_devices = []
    block_count = len(model.transformer_blocks)
    for index, block in enumerate(model.transformer_blocks):
        device_index = min(index * len(devices) // block_count, len(devices) - 1)
        block.to(devices[device_index])
        block_devices.append(str(devices[device_index]))
    return {
        "devices": [str(device) for device in devices],
        "block_devices": block_devices,
        "placement": "contiguous_pipeline",
    }


def _forward(
    model: Any,
    fixture: dict[str, Any],
    video_rows: Any,
    audio_rows: Any,
    unique_timesteps: Any,
    inverse_indices: Any,
    step: int,
    archive: dict[str, Any],
    layer_digests: list[dict[str, Any]],
) -> tuple[Any, Any]:
    import torch

    used = int(fixture["cu_seqlens"][1].item())
    first = next(model.context_embedder.parameters()).device
    last = next(model.norm_out.parameters()).device
    condition = fixture["condition_hidden"].to(first)
    image_positions_first = fixture["img_pos"].view(-1).to(torch.long).to(first)
    audio_positions_first = fixture["audio_pos"].view(-1).to(torch.long).to(first)
    text_positions_first = fixture["text_pos"].view(-1).to(torch.long).to(first)

    video_embedding = model.proj_in(video_rows.to(first).unsqueeze(0))
    audio_embedding = model.audio_proj_in(audio_rows.to(first).unsqueeze(0))
    text_embedding = model.token_refiner(model.context_embedder(condition))
    hidden = text_embedding.new_zeros((1, used, text_embedding.shape[-1]))
    hidden = hidden.index_copy(1, text_positions_first, text_embedding)
    hidden = hidden.index_copy(1, image_positions_first, video_embedding.to(text_embedding.dtype))
    hidden = hidden.index_copy(1, audio_positions_first, audio_embedding.to(text_embedding.dtype))

    time_frequency = model.time_proj(unique_timesteps.to(first))
    time_embedding = model.time_embedder(time_frequency.to(model.time_embedder.linear_1.weight.dtype))
    rotary = model.rope(fixture["position_ids"][:used].to(first))
    adaln_indices = inverse_indices[:used].to(first) * 3 + fixture["token_tags"][:used].to(first)
    if step == 0:
        position_ids = fixture["position_ids"][:used].to(first)
        per_axis = position_ids.to(torch.float32).unsqueeze(-1) * model.rope.inv_freq.view(1, 1, -1)
        rope_half = per_axis.reshape(used, -1)
        archive["step_000.condition_projection"] = (
            model.context_embedder(condition).squeeze(0).detach().to("cpu").contiguous()
        )
        archive["step_000.refined_condition"] = text_embedding.squeeze(0).detach().to("cpu").contiguous()
        archive["step_000.video_embedding"] = video_embedding.squeeze(0).detach().to("cpu").contiguous()
        archive["step_000.audio_embedding"] = audio_embedding.squeeze(0).detach().to("cpu").contiguous()
        archive["step_000.packed_hidden"] = hidden.squeeze(0).detach().to("cpu").contiguous()
        archive["step_000.time_embedding"] = time_embedding.detach().to("cpu").contiguous()
        archive["step_000.rope_frequencies"] = torch.cat((rope_half, rope_half), dim=-1).detach().to("cpu").contiguous()
        archive["step_000.combined_indices"] = adaln_indices.detach().to("cpu").contiguous()
    for layer, block in enumerate(model.transformer_blocks):
        device = next(block.parameters()).device
        hidden = hidden.to(device)
        block_time = time_embedding.to(device)
        block_indices = adaln_indices.to(device)
        block_rotary = tuple(value.to(device) for value in rotary)
        if step == 0 and layer == 0:
            adaln_tuple = block.adaln_proj(block_time)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_tuple
            norm1_output = block.norm1(hidden)
            attention_input = norm1_output
            attention_input = attention_input * (1.0 + scale_msa.index_select(0, block_indices))
            attention_input = attention_input + shift_msa.index_select(0, block_indices)
            attention_output = block.attn(attention_input, block_rotary)
            attention_delta = gate_msa.index_select(0, block_indices) * attention_output
            after_attention = hidden + attention_delta
            norm2_output = block.norm2(after_attention)
            mlp_input = norm2_output
            mlp_input = mlp_input * (1.0 + scale_mlp.index_select(0, block_indices))
            mlp_input = mlp_input + shift_mlp.index_select(0, block_indices)
            mlp_output = block.ff(mlp_input)
            mlp_delta = gate_mlp.index_select(0, block_indices) * mlp_output
            hidden = after_attention + mlp_delta
            archive["step_000.block_00.adaln"] = (
                torch.stack(adaln_tuple, dim=1)
                .reshape(block_time.shape[0], 3, 6, hidden.shape[-1])
                .detach()
                .to("cpu")
                .contiguous()
            )
            for name, value in zip(
                ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"),
                adaln_tuple,
                strict=True,
            ):
                archive[f"step_000.block_00.{name}_selected"] = (
                    value.index_select(0, block_indices).detach().to("cpu").contiguous()
                )
            archive["step_000.block_00.norm1_output"] = norm1_output.squeeze(0).detach().to("cpu").contiguous()
            archive["step_000.block_00.attention_input"] = attention_input.squeeze(0).detach().to("cpu").contiguous()
            archive["step_000.block_00.attention_output"] = attention_output.squeeze(0).detach().to("cpu").contiguous()
            archive["step_000.block_00.attention_delta"] = attention_delta.squeeze(0).detach().to("cpu").contiguous()
            archive["step_000.block_00.norm2_output"] = norm2_output.squeeze(0).detach().to("cpu").contiguous()
            archive["step_000.block_00.mlp_input"] = mlp_input.squeeze(0).detach().to("cpu").contiguous()
            archive["step_000.block_00.mlp_output"] = mlp_output.squeeze(0).detach().to("cpu").contiguous()
            archive["step_000.block_00.mlp_delta"] = mlp_delta.squeeze(0).detach().to("cpu").contiguous()
        else:
            hidden = block(hidden, block_time, block_indices, block_rotary)
        layer_value = hidden.squeeze(0).detach().to("cpu").contiguous()
        layer_digests.append(
            {
                "step": step,
                "layer": layer,
                "dtype": str(layer_value.dtype).removeprefix("torch."),
                "shape": list(layer_value.shape),
                "sha256": _tensor_digest(layer_value),
            }
        )
        archive[f"step_{step:03d}.block_{layer:02d}.output"] = layer_value

    hidden = hidden.to(last)
    time_embedding_last = time_embedding.to(last)
    inverse_last = inverse_indices[:used].to(last)
    final_activation = model.norm_out(hidden, time_embedding_last, inverse_last)
    final_fp32 = final_activation.to(model.proj_out.weight.dtype)
    all_video = model.proj_out(final_fp32)
    all_audio = model.audio_proj_out(final_fp32)
    image_positions_last = fixture["img_pos"].view(-1).to(torch.long).to(last)
    audio_positions_last = fixture["audio_pos"].view(-1).to(torch.long).to(last)
    video_velocity = all_video.index_select(1, image_positions_last).squeeze(0)
    audio_velocity = all_audio.index_select(1, audio_positions_last).squeeze(0)
    if step == 0:
        archive["step_000.final_activation"] = final_activation.squeeze(0).detach().to("cpu").contiguous()
        archive["step_000.all_video_logits"] = all_video.squeeze(0).detach().to("cpu").contiguous()
        archive["step_000.all_audio_logits"] = all_audio.squeeze(0).detach().to("cpu").contiguous()
    return video_velocity.to(first), audio_velocity.to(first)


def _run_trajectory(model: Any, fixture: dict[str, Any], scheduler_class: type[Any]) -> tuple[dict[str, Any], list]:
    import torch

    first = next(model.context_embedder.parameters()).device
    video_scheduler = scheduler_class(shift=VIDEO_SHIFT)
    audio_scheduler = scheduler_class(shift=AUDIO_SHIFT)
    video_scheduler.set_timesteps(BASE_POINT_COUNT)
    audio_scheduler.set_timesteps(BASE_POINT_COUNT)
    if video_scheduler.sigmas.numel() != BASE_POINT_COUNT or audio_scheduler.sigmas.numel() != BASE_POINT_COUNT:
        raise ValueError("Base schedule must contain exactly 50 sigma points")

    image_update = fixture["update_mask"].view(-1).to(torch.bool).to(first)
    audio_update = fixture["audio_update_mask"].view(-1).to(torch.bool).to(first)
    video_rows = _deterministic_rows(image_update.numel(), 96, 73).to(first)
    audio_rows = _deterministic_rows(audio_update.numel(), 32, 131).to(first)
    video_anchor = video_rows[~image_update].clone()
    audio_anchor = audio_rows[~audio_update].clone()
    archive: dict[str, Any] = {
        "schedule.video_sigmas": video_scheduler.sigmas,
        "schedule.audio_sigmas": audio_scheduler.sigmas,
        "schedule.video_timesteps": video_scheduler.timesteps,
        "schedule.audio_timesteps": audio_scheduler.timesteps,
        "input.initial_video_rows": video_rows.detach().to("cpu"),
        "input.initial_audio_rows": audio_rows.detach().to("cpu"),
        "input.video_anchor": video_anchor.detach().to("cpu"),
        "input.audio_anchor": audio_anchor.detach().to("cpu"),
        "layout.update_mask": fixture["update_mask"],
        "layout.audio_update_mask": fixture["audio_update_mask"],
        "layout.img_pos": fixture["img_pos"],
        "layout.audio_pos": fixture["audio_pos"],
        "layout.text_pos": fixture["text_pos"],
        "layout.position_ids": fixture["position_ids"],
        "layout.token_tags": fixture["token_tags"],
        "layout.cu_seqlens": fixture["cu_seqlens"],
    }
    layer_digests: list[dict[str, Any]] = []
    for step in range(BASE_POINT_COUNT - 1):
        t_video = float(video_scheduler.timesteps[step].item())
        t_audio = float(audio_scheduler.timesteps[step].item())
        row_timesteps, unique, inverse = _row_timestep_plan(fixture, t_video, t_audio)
        video_velocity, audio_velocity = _forward(
            model,
            fixture,
            video_rows,
            audio_rows,
            unique,
            inverse,
            step,
            archive,
            layer_digests,
        )
        target_video_velocity = video_velocity[image_update]
        target_audio_velocity = audio_velocity[audio_update]
        video_target = video_rows[image_update]
        audio_target = audio_rows[audio_update]
        video_x0 = video_target + (1.0 - video_scheduler.timesteps[step].to(first)) * target_video_velocity
        audio_x0 = audio_target + (1.0 - audio_scheduler.timesteps[step].to(first)) * target_audio_velocity
        next_video = video_scheduler.step(
            target_video_velocity, video_scheduler.timesteps[step], video_target
        ).prev_sample
        next_audio = audio_scheduler.step(
            target_audio_velocity, audio_scheduler.timesteps[step], audio_target
        ).prev_sample
        video_rows = video_rows.clone()
        audio_rows = audio_rows.clone()
        video_rows[image_update] = next_video
        audio_rows[audio_update] = next_audio
        video_rows[~image_update] = video_anchor
        audio_rows[~audio_update] = audio_anchor
        if not torch.equal(video_rows[~image_update], video_anchor) or not torch.equal(
            audio_rows[~audio_update], audio_anchor
        ):
            raise ValueError(f"condition anchor changed after step {step}")

        prefix = f"step_{step:03d}."
        archive[prefix + "row_timesteps"] = row_timesteps
        archive[prefix + "unique_timesteps"] = unique
        archive[prefix + "inverse_indices"] = inverse
        archive[prefix + "video_velocity"] = video_velocity.detach().to("cpu")
        archive[prefix + "audio_velocity"] = audio_velocity.detach().to("cpu")
        archive[prefix + "video_x0"] = video_x0.detach().to("cpu")
        archive[prefix + "audio_x0"] = audio_x0.detach().to("cpu")
        archive[prefix + "video_rows_after"] = video_rows.detach().to("cpu")
        archive[prefix + "audio_rows_after"] = audio_rows.detach().to("cpu")
        logger.info(f"Completed official MiniMax-H3 Base forward {step + 1}/49")
    return archive, layer_digests


def _write_output(
    output_dir: Path, archive: dict[str, Any], layer_digests: list[dict[str, Any]], manifest: dict[str, Any]
) -> Path:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    archive_temp = output_dir / f".{ARCHIVE_NAME}{suffix}"
    digest_temp = output_dir / f".{LAYER_DIGESTS_NAME}{suffix}"
    manifest_temp = output_dir / f".{MANIFEST_NAME}{suffix}"
    try:
        save_file({name: value.contiguous() for name, value in archive.items()}, str(archive_temp))
        with digest_temp.open("x", encoding="utf-8") as handle:
            for row in layer_digests:
                handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        manifest["artifacts"] = {
            ARCHIVE_NAME: {"size": archive_temp.stat().st_size, "sha256": _sha256(archive_temp)},
            LAYER_DIGESTS_NAME: {"size": digest_temp.stat().st_size, "sha256": _sha256(digest_temp)},
        }
        with manifest_temp.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(archive_temp, output_dir / ARCHIVE_NAME)
        os.replace(digest_temp, output_dir / LAYER_DIGESTS_NAME)
        os.replace(manifest_temp, output_dir / MANIFEST_NAME)
    finally:
        archive_temp.unlink(missing_ok=True)
        digest_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)
    return output_dir / MANIFEST_NAME


def main() -> None:
    args = _parse_args()
    if args.num_devices < 2:
        raise ValueError("C5 official full-stack Golden requires at least two NPUs")

    import diffusers
    import torch
    import torch_npu
    from diffusers.models.transformers import MiniMaxH3Transformer3DModel
    from diffusers.schedulers import MiniMaxH3Scheduler

    if torch.npu.device_count() < args.num_devices:
        raise ValueError(f"requested {args.num_devices} NPUs, found {torch.npu.device_count()}")
    devices = [torch.device(f"npu:{index}") for index in range(args.num_devices)]
    for device in devices:
        torch.npu.set_device(device)
        torch.npu.empty_cache()

    transformer_path = args.transformer_path.resolve()
    packing_path = args.packing_golden.resolve()
    vllm_source = args.vllm_source.resolve()
    fixture, packing = _packing_fixture(packing_path)
    source = _source_manifest(vllm_source, MiniMaxH3Transformer3DModel, MiniMaxH3Scheduler)
    checkpoint = _checkpoint_manifest(transformer_path)
    logger.info(f"Loading official full transformer from {transformer_path}")
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    placement = _place_model(model, devices)
    for device in devices:
        torch.npu.reset_peak_memory_stats(device)
    with torch.inference_mode():
        archive, layer_digests = _run_trajectory(model, fixture, MiniMaxH3Scheduler)
    for device in devices:
        torch.npu.synchronize(device)
    for name, value in archive.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"official trajectory tensor {name!r} contains NaN or Inf")

    execution = {
        **placement,
        "base_sigma_points": BASE_POINT_COUNT,
        "transformer_forwards": BASE_POINT_COUNT - 1,
        "block_forwards": (BASE_POINT_COUNT - 1) * len(model.transformer_blocks),
        "layer_digest_count": len(layer_digests),
        "peak_npu_memory_allocated": {str(device): int(torch.npu.max_memory_allocated(device)) for device in devices},
        "peak_npu_memory_reserved": {str(device): int(torch.npu.max_memory_reserved(device)) for device in devices},
    }
    manifest = {
        "schema": "xllm.minimax_h3.trajectory_reference/v1",
        "status": "OFFICIAL_BASE_50_LAYER_TRAJECTORY",
        "source": source,
        "checkpoint": checkpoint,
        "packing": packing,
        "runtime": _runtime_manifest(torch, torch_npu, diffusers, devices[0]),
        "execution": execution,
        "schedule": {
            "point_count": BASE_POINT_COUNT,
            "forward_count": BASE_POINT_COUNT - 1,
            "video_shift": VIDEO_SHIFT,
            "audio_shift": AUDIO_SHIFT,
            "video_terminal_sigma": float(archive["schedule.video_sigmas"][-1]),
            "audio_terminal_sigma": float(archive["schedule.audio_sigmas"][-1]),
        },
        "anchors": {
            "visual_rows": int((~fixture["update_mask"]).sum()),
            "audio_rows": int((~fixture["audio_update_mask"]).sum()),
            "exact_after_every_step": True,
        },
        "tensors": {name: _tensor_summary(value) for name, value in sorted(archive.items())},
    }
    print(_write_output(args.output_dir.resolve(), archive, layer_digests, manifest))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("MiniMax-H3 C5 trajectory Golden generation failed")
        raise
