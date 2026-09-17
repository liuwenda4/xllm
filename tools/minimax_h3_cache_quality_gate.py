#!/usr/bin/env python3
"""Compare dense and CacheDiT MiniMax-H3 MP4 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__:
    from tools.minimax_h3_ref2va_media_compare import (
        STFT_FFT_SIZES,
        _atomic_write_json,
        _audio_metrics,
        _video_metrics,
    )
else:
    from minimax_h3_ref2va_media_compare import (
        STFT_FFT_SIZES,
        _atomic_write_json,
        _audio_metrics,
        _video_metrics,
    )

CACHE_PATTERN = re.compile(
    r"MINIMAX_H3_CACHE_RESULT rank=(?P<rank>\d+) enabled=(?P<enabled>[01]) "
    r"dense_forwards=(?P<dense>\d+) cache_hits=(?P<hits>\d+) "
    r"similarity_checks=(?P<checks>\d+) block_forwards=(?P<blocks>\d+) "
    r"hit_forwards=(?P<hit_forwards>[0-9,]*)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_success(summary_path: Path) -> tuple[dict[str, Any], Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or summary.get("mode") != "success-only":
        raise ValueError(f"runtime summary is not a successful success-only Gate: {summary_path}")
    outputs = summary.get("successful_outputs")
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
        raise ValueError(f"runtime summary must contain exactly one successful output: {summary_path}")
    output = outputs[0]
    path = Path(output.get("path", "")).resolve()
    if not path.is_file() or output.get("sha256") != _sha256(path):
        raise ValueError(f"runtime MP4 attestation failed: {summary_path}")
    return summary, path


def _parse_cache_records(log_dir: Path, expected_world_size: int = 16) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for log_path in sorted(log_dir.glob("node_*.log")):
        matches = list(CACHE_PATTERN.finditer(log_path.read_text(encoding="utf-8", errors="replace")))
        if len(matches) != 1:
            raise ValueError(f"expected exactly one CacheDiT record in {log_path}, got {len(matches)}")
        match = matches[0]
        hit_forwards = [int(value) for value in match.group("hit_forwards").split(",") if value]
        records.append(
            {
                "rank": int(match.group("rank")),
                "enabled": match.group("enabled") == "1",
                "dense_forwards": int(match.group("dense")),
                "cache_hits": int(match.group("hits")),
                "similarity_checks": int(match.group("checks")),
                "block_forwards": int(match.group("blocks")),
                "hit_forwards": hit_forwards,
            }
        )
    if {record["rank"] for record in records} != set(range(expected_world_size)):
        raise ValueError("CacheDiT rank inventory is incomplete")
    reference = {key: value for key, value in records[0].items() if key != "rank"}
    for record in records:
        if {key: value for key, value in record.items() if key != "rank"} != reference:
            raise ValueError("CacheDiT ranks disagree on cache counters or hit forwards")
    hits = reference["cache_hits"]
    if (
        not reference["enabled"]
        or hits <= 0
        or reference["dense_forwards"] + hits != 49
        or reference["similarity_checks"] < hits
        or reference["similarity_checks"] > 45
        or reference["block_forwards"] != 2450 - 49 * hits
        or len(reference["hit_forwards"]) != hits
    ):
        raise ValueError("CacheDiT execution counters violate the H3 quality contract")
    return {"consensus": reference, "records": sorted(records, key=lambda value: value["rank"])}


def _decode_mp4(path: Path, decoder_binary: Path, output_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    output_dir.mkdir()
    video_path = output_dir / "video.rgb"
    audio_path = output_dir / "audio.f32"
    metadata_path = output_dir / "metadata.json"
    environment = os.environ.copy()
    environment.update(
        {
            "MINIMAX_H3_QUALITY_MP4": str(path),
            "MINIMAX_H3_QUALITY_VIDEO_RAW": str(video_path),
            "MINIMAX_H3_QUALITY_AUDIO_RAW": str(audio_path),
            "MINIMAX_H3_QUALITY_METADATA": str(metadata_path),
        }
    )
    result = subprocess.run(
        [
            str(decoder_binary),
            "--gtest_filter=MMCodecTest.ExtractsExternalProductionMp4ForQualityGateWhenRequested",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        check=False,
    )
    (output_dir / "decoder.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0 or "[  PASSED  ] 1 test." not in result.stdout:
        raise RuntimeError(f"quality decoder failed for {path} with code {result.returncode}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    video_shape = tuple(metadata.get("video_shape", []))
    audio_shape = tuple(metadata.get("audio_shape", []))
    if (
        video_shape != (124, 3, 768, 1344)
        or metadata.get("video_dtype") != "uint8"
        or len(audio_shape) != 2
        or audio_shape[0] != 2
        or metadata.get("audio_dtype") != "float32"
        or metadata.get("audio_sample_rate") != 32000
        or metadata.get("audio_channels") != 2
    ):
        raise ValueError(f"quality decoder metadata mismatch for {path}")
    video_array = np.memmap(video_path, dtype=np.uint8, mode="c", shape=video_shape)
    audio_array = np.memmap(audio_path, dtype=np.float32, mode="c", shape=audio_shape)
    video = torch.from_numpy(video_array).permute(1, 0, 2, 3).unsqueeze(0)
    audio = torch.from_numpy(audio_array).unsqueeze(0)
    return video, audio


def _evaluate(video: dict[str, Any], audio: dict[str, Any]) -> dict[str, Any]:
    failures = []

    def require(metric: str, actual: float, passed: bool, requirement: str) -> None:
        if not passed:
            failures.append({"metric": metric, "actual": actual, "requirement": requirement})

    ssim = video["ssim"]["all_frame_mean"]
    psnr = video["raw_uint8"]["all_frame_psnr_db"]
    require("video.ssim", ssim, math.isfinite(ssim) and ssim >= 0.97, ">= 0.97")
    require("video.psnr_db", psnr, psnr >= 34.0, ">= 34.0")
    pearson = audio["centered_pearson"]["minimum"]
    require("audio.pearson", pearson, math.isfinite(pearson) and pearson >= 0.98, ">= 0.98")
    rms_min = audio["rms_ratio_native_over_official"]["min"]
    rms_max = audio["rms_ratio_native_over_official"]["max"]
    require("audio.rms_min", rms_min, math.isfinite(rms_min) and rms_min >= 0.95, ">= 0.95")
    require("audio.rms_max", rms_max, math.isfinite(rms_max) and rms_max <= 1.05, "<= 1.05")
    for n_fft in STFT_FFT_SIZES:
        cosine = audio["stft_magnitude_spectral_cosine"][str(n_fft)]["minimum"]
        require(f"audio.stft_{n_fft}", cosine, math.isfinite(cosine) and cosine >= 0.98, ">= 0.98")
    return {"passed": not failures, "failed_metrics": failures}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-summary", type=Path, required=True)
    parser.add_argument("--cache-summary", type=Path, required=True)
    parser.add_argument("--cache-log-dir", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--media-decoder-binary", type=Path, required=True)
    parser.add_argument("--video-frame-chunk", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dense_summary, dense_mp4 = _load_success(args.dense_summary.resolve())
    cache_summary, cache_mp4 = _load_success(args.cache_summary.resolve())
    cache = _parse_cache_records(args.cache_log_dir.resolve())
    decoder_binary = args.media_decoder_binary.resolve()
    if not decoder_binary.is_file():
        raise ValueError(f"quality decoder binary does not exist: {decoder_binary}")
    with tempfile.TemporaryDirectory(prefix="minimax-h3-quality-") as temporary:
        temporary_path = Path(temporary)
        dense_video, dense_audio = _decode_mp4(dense_mp4, decoder_binary, temporary_path / "dense")
        cache_video, cache_audio = _decode_mp4(cache_mp4, decoder_binary, temporary_path / "cache")
        video = _video_metrics(dense_video, cache_video, frame_chunk=args.video_frame_chunk)
        audio = _audio_metrics(dense_audio, cache_audio)
    gate = _evaluate(video, audio)
    report = {
        "schema": "xllm.minimax_h3.cachedit_quality/v1",
        "status": "PASS" if gate["passed"] else "FAIL",
        "dense": {"summary": str(args.dense_summary.resolve()), "mp4": str(dense_mp4), "sha256": _sha256(dense_mp4)},
        "cache": {
            "summary": str(args.cache_summary.resolve()),
            "mp4": str(cache_mp4),
            "sha256": _sha256(cache_mp4),
            "execution": cache,
        },
        "http_duration_seconds": {
            "dense": dense_summary["events"][2]["duration_seconds"],
            "cache": cache_summary["events"][2]["duration_seconds"],
        },
        "video": video,
        "audio": audio,
        "gate": gate,
    }
    _atomic_write_json(args.output_report.resolve(), report)
    print(args.output_report.resolve())
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
