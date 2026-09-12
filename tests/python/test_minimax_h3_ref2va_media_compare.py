# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import copy
import math

import pytest
import torch

from tools.minimax_h3_ref2va_media_compare import (
    _audio_metrics,
    _evaluate_gate,
    _tensor_metrics,
    _validate_shared_provenance,
    _video_metrics,
)


def _passing_video_metrics() -> dict:
    return {
        "ssim": {"all_frame_mean": 0.97},
        "raw_uint8": {"all_frame_psnr_db": 34.0},
    }


def _passing_audio_metrics() -> dict:
    return {
        "centered_pearson": {"minimum": 0.98},
        "rms_ratio_native_over_official": {"min": 0.95, "max": 1.05},
        "stft_magnitude_spectral_cosine": {str(n_fft): {"minimum": 0.98} for n_fft in (512, 1024, 2048)},
    }


def _passing_media_structure() -> dict:
    return {
        "official_hf": {"checks": {"structure": True}, "passed": True},
        "xllm_native": {"checks": {"structure": True}, "passed": True},
    }


def test_tensor_metrics_report_expected_values_in_chunks() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0, 4.0])
    candidate = torch.tensor([1.0, 2.0, 2.0, 6.0])

    metrics = _tensor_metrics(reference, candidate, chunk_numel=2)

    assert metrics["numel"] == 4
    assert metrics["relative_l2"] == pytest.approx(math.sqrt(5.0 / 30.0))
    assert metrics["cosine"] == pytest.approx(35.0 / math.sqrt(30.0 * 45.0))
    assert metrics["max_abs"] == 2.0
    assert metrics["mae"] == pytest.approx(0.75)
    assert metrics["rmse"] == pytest.approx(math.sqrt(5.0 / 4.0))


def test_identical_video_has_exact_ssim_and_infinite_psnr() -> None:
    generator = torch.Generator().manual_seed(91)
    video = torch.randint(0, 256, (1, 3, 2, 16, 17), dtype=torch.uint8, generator=generator)

    metrics = _video_metrics(video, video.clone(), frame_chunk=1)

    assert metrics["raw_uint8"]["all_frame_mse"] == 0.0
    assert math.isinf(metrics["raw_uint8"]["all_frame_psnr_db"])
    assert metrics["raw_uint8"]["per_frame_psnr_db"] == {
        "min": math.inf,
        "mean": math.inf,
        "max": math.inf,
    }
    assert metrics["ssim"]["all_frame_mean"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["ssim"]["per_frame"]["min"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["ssim"]["per_frame"]["max"] == pytest.approx(1.0, abs=1e-12)


def test_degraded_video_fails_predeclared_video_thresholds() -> None:
    generator = torch.Generator().manual_seed(27)
    reference = torch.randint(0, 256, (1, 3, 2, 20, 20), dtype=torch.uint8, generator=generator)
    degraded = torch.zeros_like(reference)

    video = _video_metrics(reference, degraded)
    gate = _evaluate_gate(video, _passing_audio_metrics(), _passing_media_structure(), True)

    assert gate["state"] == "NATIVE_MEDIA_FAIL"
    assert {failure["metric"] for failure in gate["failed_metrics"]} >= {"video.ssim", "video.psnr_db"}


def test_audio_correlation_rms_and_spectral_metrics() -> None:
    sample_count = 8192
    time = torch.arange(sample_count, dtype=torch.float64) / 32000.0
    left = 0.4 * torch.sin(2.0 * math.pi * 440.0 * time) + 0.1 * torch.sin(2.0 * math.pi * 880.0 * time)
    right = 0.3 * torch.cos(2.0 * math.pi * 330.0 * time) + 0.05 * torch.sin(2.0 * math.pi * 1200.0 * time)
    reference = torch.stack((left, right)).unsqueeze(0).to(torch.float32)
    candidate = reference * 0.99

    metrics = _audio_metrics(reference, candidate)

    assert metrics["waveform"]["cosine"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["centered_pearson"]["minimum"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["rms_ratio_native_over_official"]["min"] == pytest.approx(0.99, rel=1e-6)
    assert metrics["rms_ratio_native_over_official"]["max"] == pytest.approx(0.99, rel=1e-6)
    assert set(metrics["stft_magnitude_spectral_cosine"]) == {"512", "1024", "2048"}
    for spectral in metrics["stft_magnitude_spectral_cosine"].values():
        assert spectral["minimum"] == pytest.approx(1.0, abs=1e-12)


def test_threshold_decision_requires_metrics_structure_and_distinct_condition() -> None:
    passing = _evaluate_gate(
        _passing_video_metrics(),
        _passing_audio_metrics(),
        _passing_media_structure(),
        True,
    )
    assert passing == {
        "state": "NATIVE_MEDIA_PASS_NON_BIT_EXACT",
        "passed": True,
        "failed_metrics": [],
    }
    exact = _evaluate_gate(
        _passing_video_metrics(),
        _passing_audio_metrics(),
        _passing_media_structure(),
        False,
    )
    assert exact == {
        "state": "NATIVE_MEDIA_PASS_BIT_EXACT",
        "passed": True,
        "failed_metrics": [],
    }

    media_failure = _passing_media_structure()
    media_failure["xllm_native"] = {"checks": {"structure": False}, "passed": False}
    failing = _evaluate_gate(_passing_video_metrics(), _passing_audio_metrics(), media_failure, False)
    failed_names = {failure["metric"] for failure in failing["failed_metrics"]}
    assert failing["state"] == "NATIVE_MEDIA_FAIL"
    assert "media.xllm_native.structure" in failed_names


def test_shared_manifest_mismatch_is_rejected() -> None:
    official = {
        "prepared": {"artifact_sha256": "a" * 64, "manifest_sha256": "b" * 64},
        "source": {"digest": "c" * 64},
        "checkpoint": {"digest": "d" * 64},
        "geometry": {"target": "fixed"},
        "backend_provenance": {"hidden_sha256": "e" * 64},
    }
    native = copy.deepcopy(official)
    native["backend_provenance"]["hidden_sha256"] = "f" * 64
    assert _validate_shared_provenance(official, native) is True

    native["prepared"]["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prepared artifact digest mismatch"):
        _validate_shared_provenance(official, native)
