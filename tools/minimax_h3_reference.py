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
"""Run a pinned MiniMax-H3 diffusers Ref2VA reference case on one accelerator."""

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MiniMax-H3 Ref2VA text-plus-image reference case.")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Converted local modular checkpoint.")
    parser.add_argument("--image-path", type=Path, required=True, help="Local reference image.")
    parser.add_argument("--prompt-path", type=Path, required=True, help="UTF-8 text prompt file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for tensors, media, and summary.")
    parser.add_argument("--device", default="npu:0", help="Execution device used by group offloading.")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _tensor_digest(tensor: Any) -> str:
    import torch

    cpu_tensor = tensor.detach().to("cpu").contiguous()
    byte_view = cpu_tensor.view(torch.uint8)
    return hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    cpu_tensor = tensor.detach().to("cpu")
    finite = torch.isfinite(cpu_tensor)
    finite_values = cpu_tensor[finite].float()
    return {
        "shape": list(cpu_tensor.shape),
        "dtype": str(cpu_tensor.dtype),
        "device": str(tensor.device),
        "sha256": _tensor_digest(cpu_tensor),
        "finite_count": int(finite.sum().item()),
        "element_count": cpu_tensor.numel(),
        "min": float(finite_values.min().item()) if finite_values.numel() else None,
        "max": float(finite_values.max().item()) if finite_values.numel() else None,
        "mean": float(finite_values.mean().item()) if finite_values.numel() else None,
    }


def _configure_group_offload(pipe: Any, device: Any) -> None:
    import torch
    from diffusers.hooks import apply_group_offloading

    offload = {"onload_device": device, "offload_device": torch.device("cpu"), "use_stream": False}
    apply_group_offloading(
        pipe.transformer_ref,
        offload_type="block_level",
        num_blocks_per_group=1,
        **offload,
    )
    apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
    apply_group_offloading(pipe.vae, offload_type="leaf_level", **offload)
    pipe.audio_vae.to(device)


def _normalize_video(video: Any) -> Any:
    import torch

    if isinstance(video, list):
        video = video[0]
    if video.ndim == 5:
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected four-dimensional video output, got {list(video.shape)}")
    if video.shape[1] == 3:
        video = video.permute(0, 2, 3, 1)
    if video.shape[-1] != 3:
        raise ValueError(f"Expected RGB video output, got {list(video.shape)}")
    return (video.clamp(0, 1) * 255).round().to(torch.uint8)


def _probe_container(path: Path) -> dict[str, Any]:
    import av

    with av.open(str(path)) as container:
        video_stream = next(stream for stream in container.streams if stream.type == "video")
        audio_stream = next(stream for stream in container.streams if stream.type == "audio")
        video_duration = float(video_stream.duration * video_stream.time_base)
        audio_duration = float(audio_stream.duration * audio_stream.time_base)
        return {
            "sha256": _file_digest(path),
            "video_codec": video_stream.codec_context.name,
            "video_frame_count": video_stream.frames,
            "video_fps": float(video_stream.average_rate),
            "video_width": video_stream.codec_context.width,
            "video_height": video_stream.codec_context.height,
            "video_duration_seconds": video_duration,
            "audio_codec": audio_stream.codec_context.name,
            "audio_frame_count": audio_stream.frames,
            "audio_sample_rate": audio_stream.codec_context.sample_rate,
            "audio_channels": audio_stream.codec_context.channels,
            "audio_sample_count": audio_stream.duration,
            "audio_duration_seconds": audio_duration,
            "duration_difference_seconds": abs(video_duration - audio_duration),
        }


def _save_outputs(results: Any, output_dir: Path) -> dict[str, Any]:
    import torch
    from diffusers.utils.export_utils import encode_video

    video = _normalize_video(results["videos"])
    audio = results["audio"]
    if audio.ndim == 3:
        audio = audio[0]
    sampling_rate = int(results["sampling_rate"])

    tensor_path = output_dir / "outputs.pt"
    media_path = output_dir / "output.mp4"
    torch.save({"video": video.cpu(), "audio": audio.cpu(), "sampling_rate": sampling_rate}, tensor_path)
    encode_video(
        video,
        fps=24,
        output_path=str(media_path),
        audio=audio,
        audio_sample_rate=sampling_rate,
    )
    flattened_frames = video.to("cpu").reshape(video.shape[0], -1)
    return {
        "video": _tensor_summary(video),
        "audio": _tensor_summary(audio),
        "tensor_file_sha256": _file_digest(tensor_path),
        "container": _probe_container(media_path),
        "sampling_rate": sampling_rate,
        "frame_count": int(video.shape[0]),
        "fps": 24,
        "video_duration_seconds": float(video.shape[0] / 24),
        "audio_duration_seconds": float(audio.shape[-1] / sampling_rate),
        "audio_channels": int(audio.shape[0]),
        "black_frame_count": int((flattened_frames == 0).all(dim=1).sum().item()),
        "constant_frame_count": int(
            (flattened_frames.max(dim=1).values == flattened_frames.min(dim=1).values).sum().item()
        ),
    }


def main() -> None:
    args = _parse_args()
    checkpoint_path = args.checkpoint_path.resolve()
    image_path = args.image_path.resolve()
    prompt_path = args.prompt_path.resolve()
    output_dir = args.output_dir.resolve()
    for path in (checkpoint_path, image_path, prompt_path):
        if not path.exists():
            raise ValueError(f"Required input does not exist: {path}")
    if args.num_frames < 124:
        raise ValueError("The pinned diffusers reference requires at least 124 frames (about 5 seconds at 24 FPS).")

    import torch
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    run_config = {
        "checkpoint_path": str(checkpoint_path),
        "image_path": str(image_path),
        "image_sha256": _file_digest(image_path),
        "prompt_path": str(prompt_path),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "device": str(device),
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "torch_version": torch.__version__,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    logger.info(f"Loading MiniMax-H3 components from {checkpoint_path}")
    started_at = time.perf_counter()
    pipe = ModularPipeline.from_pretrained(checkpoint_path, workflow="ref2va", local_files_only=True)
    pipe.load_components(dtype=torch.bfloat16, local_files_only=True)
    pipe.transformer_ref.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.audio_vae.requires_grad_(False)
    _configure_group_offload(pipe, device)
    load_seconds = time.perf_counter() - started_at

    if device.type == "npu":
        torch.npu.reset_peak_memory_stats(device)
    reference = MiniMaxH3ImageReference.from_file(str(image_path))
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    logger.info(
        f"Running Ref2VA: {args.num_frames} frames, {args.width}x{args.height}, "
        f"{args.num_inference_steps} sigma points, seed {args.seed}"
    )
    inference_started_at = time.perf_counter()
    results = pipe(
        prompt=prompt,
        references=[reference],
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        output_type="pt",
        output=["videos", "audio", "sampling_rate"],
    )
    if device.type == "npu":
        torch.npu.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started_at

    output_summary = _save_outputs(results, output_dir)
    summary = {
        "status": "H3_REFERENCE_PASS",
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "outputs": output_summary,
    }
    if device.type == "npu":
        summary["peak_device_memory_bytes"] = int(torch.npu.max_memory_allocated(device))
        summary["peak_device_reserved_bytes"] = int(torch.npu.max_memory_reserved(device))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logger.info(f"Reference output written to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("MiniMax-H3 reference run failed")
        raise
