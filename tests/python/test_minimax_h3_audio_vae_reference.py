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

import math
import sys
from typing import Any

import pytest
import torch

from tools.minimax_h3_audio_vae_reference import (
    DECODE_LATENT_SHAPE,
    DEFAULT_DIFFUSERS_SOURCE,
    DIFFUSERS_SOURCE_SHA256,
    INPUT_SAMPLE_RATE,
    INPUT_SHAPE,
    LATENT_CHANNELS,
    PADDED_INPUT_SHAPE,
    POSTERIOR_SEED,
    PRODUCTION_LATENT_SHAPE,
    PRODUCTION_PIPELINE_SHAPE,
    RESAMPLED_SHAPE,
    VAE_SAMPLE_RATE,
    _activate_diffusers_source,
    _audio_geometry,
    _channel_major_rows,
    _denormalize_latents,
    _deterministic_epsilon,
    _deterministic_normalized_latent,
    _deterministic_stereo_waveform,
    _fixture_manifest,
    _normalize_latents,
    _official_audio_preprocess,
    _posterior_math,
    _reshape_mono_batch_and_pad,
    _source_manifest,
    _validate_source,
)


def test_official_preprocess_truncates_before_one_resample_and_pads_geometry() -> None:
    waveform = torch.arange(2 * 3203, dtype=torch.float32).reshape(2, 3203)
    calls: list[tuple[int, int, tuple[int, ...]]] = []

    def _factory(source_rate: int, target_rate: int) -> Any:
        def _resample(value: torch.Tensor) -> torch.Tensor:
            calls.append((source_rate, target_rate, tuple(value.shape)))
            return value.repeat_interleave(2, dim=-1)

        return _resample

    resampled = _official_audio_preprocess(
        waveform,
        sample_rate=INPUT_SAMPLE_RATE,
        target_sample_rate=VAE_SAMPLE_RATE,
        max_duration=3201 / INPUT_SAMPLE_RATE,
        resample_factory=_factory,
    )
    network_input, padded, right_pad = _reshape_mono_batch_and_pad(resampled)

    assert calls == [(INPUT_SAMPLE_RATE, VAE_SAMPLE_RATE, (2, 3201))]
    assert resampled.shape == RESAMPLED_SHAPE
    assert network_input.shape == (2, 1, 6402)
    assert padded.shape == PADDED_INPUT_SHAPE
    assert right_pad == 798
    assert torch.equal(padded[..., :6402], network_input)
    assert torch.count_nonzero(padded[..., 6402:]).item() == 0


def test_official_preprocess_duplicates_mono_before_resampling() -> None:
    mono = torch.tensor([[0.0, 0.25, -0.5, 0.75]], dtype=torch.float64)
    captured: list[torch.Tensor] = []

    def _factory(_source_rate: int, _target_rate: int) -> Any:
        def _identity(value: torch.Tensor) -> torch.Tensor:
            captured.append(value.clone())
            return value

        return _identity

    stereo = _official_audio_preprocess(mono, 16000, 32000, 1.0, resample_factory=_factory)

    assert stereo.shape == (2, 4)
    assert stereo.dtype == torch.float32
    assert stereo.is_contiguous()
    assert len(captured) == 1
    assert captured[0].shape == (2, 4)
    assert torch.equal(stereo[0], stereo[1])


def test_posterior_uses_log_standard_deviation_without_half_factor() -> None:
    mean = torch.tensor([1.5, -2.0], dtype=torch.float32).view(1, 2, 1)
    logs = torch.log(torch.tensor([4.0, 0.25], dtype=torch.float32)).view_as(mean)
    epsilon = torch.tensor([0.25, -0.5], dtype=torch.float32).view_as(mean)

    posterior = _posterior_math(mean, logs, epsilon)

    expected_std = torch.tensor([4.0, 0.25], dtype=torch.float32).view_as(mean)
    assert torch.equal(posterior["mean"], mean)
    assert torch.equal(posterior["logs"], logs)
    assert torch.equal(posterior["mode"], mean)
    assert torch.allclose(posterior["std"], expected_std, rtol=1e-6, atol=0.0)
    assert torch.allclose(posterior["sample"], mean + expected_std * epsilon, rtol=1e-6, atol=0.0)
    assert not torch.allclose(posterior["std"], torch.exp(0.5 * logs))


def test_per_channel_normalization_roundtrip_and_channel_major_rows() -> None:
    latents = torch.tensor(
        [
            [[1.0, 2.0], [10.0, 14.0]],
            [[3.0, 4.0], [18.0, 22.0]],
        ],
        dtype=torch.float32,
    )
    latents_mean = (1.0, 10.0)
    latents_std = (2.0, 4.0)

    normalized = _normalize_latents(latents, latents_mean, latents_std)
    restored = _denormalize_latents(normalized, latents_mean, latents_std)
    rows = _channel_major_rows(normalized)

    assert torch.equal(restored, latents)
    assert torch.equal(
        rows,
        torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 1.0],
                [1.0, 2.0],
                [1.5, 3.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_analytic_stereo_input_is_distinct_deterministic_fp32() -> None:
    waveform = _deterministic_stereo_waveform()

    assert waveform.shape == INPUT_SHAPE
    assert waveform.dtype == torch.float32
    assert waveform.is_contiguous()
    assert torch.equal(waveform, _deterministic_stereo_waveform())
    assert not torch.equal(waveform[0], waveform[1])
    assert waveform[0, 0].item() == pytest.approx(0.20 * math.cos(0.25), abs=1e-7)
    assert waveform[1, 0].item() == pytest.approx(0.45 + 0.25 * math.sin(0.50), abs=1e-7)


def test_independent_normalized_latent_and_epsilon_are_deterministic() -> None:
    latent = _deterministic_normalized_latent()
    epsilon = _deterministic_epsilon(DECODE_LATENT_SHAPE)

    assert latent.shape == DECODE_LATENT_SHAPE
    assert latent.dtype == torch.float32
    assert latent.is_contiguous()
    assert torch.equal(latent, _deterministic_normalized_latent())
    assert torch.equal(epsilon, _deterministic_epsilon(DECODE_LATENT_SHAPE, POSTERIOR_SEED))
    assert not torch.equal(latent[0], latent[1])
    assert latent[1, 31, 7].item() == pytest.approx(((97 + 53 * 31 + 29 * 7) % 257 - 128) / 128.0)


def test_audio_geometry_is_40hz_and_has_exact_sample_counts() -> None:
    primary = _audio_geometry(8)
    production = _audio_geometry(207)
    production_fixture = _fixture_manifest()["production_smoke"]

    assert primary["latent_rate_hz"] == 40.0
    assert primary["output_samples"] == 6400
    assert primary["duration_seconds"] == 0.2
    assert production["latent_rate_hz"] == 40.0
    assert production["output_samples"] == 165600
    assert production["duration_seconds"] == pytest.approx(5.175)
    assert production_fixture["normalized_latent_shape"] == list(PRODUCTION_LATENT_SHAPE)
    assert production_fixture["vae_output_shape"] == [2, 1, 165600]
    assert production_fixture["pipeline_stereo_shape"] == list(PRODUCTION_PIPELINE_SHAPE)
    assert production_fixture["expected_calls"] == {"model.decode": 1}
    assert production_fixture["archive_policy"] == "summaries_only"
    assert PRODUCTION_PIPELINE_SHAPE == (1, 2, 165600)
    assert LATENT_CHANNELS == 32


def test_source_digest_failure_is_fatal(tmp_path) -> None:
    source = tmp_path / "autoencoder.py"
    source.write_text("changed\n", encoding="ascii")

    with pytest.raises(ValueError, match="source digest mismatch"):
        _validate_source(source, "0" * 64)


@pytest.mark.skipif(not DEFAULT_DIFFUSERS_SOURCE.is_dir(), reason="pinned Diffusers source is unavailable")
def test_current_pinned_sources_match_recorded_digests() -> None:
    source = _source_manifest(DEFAULT_DIFFUSERS_SOURCE)

    assert source["revision_short"] == "d30c748f"
    assert set(source["files"]) == set(DIFFUSERS_SOURCE_SHA256)
    assert all(len(entry["sha256"]) == 64 for entry in source["files"].values())


@pytest.mark.skipif(not DEFAULT_DIFFUSERS_SOURCE.is_dir(), reason="pinned Diffusers source is unavailable")
def test_pinned_diffusers_source_is_moved_to_import_front() -> None:
    source = str((DEFAULT_DIFFUSERS_SOURCE / "src").resolve())
    original = list(sys.path)
    try:
        sys.path[:] = ["/tmp/unpinned", source, *[entry for entry in sys.path if entry != source]]
        _activate_diffusers_source(DEFAULT_DIFFUSERS_SOURCE)
        assert sys.path[0] == source
        assert sys.path.count(source) == 1
    finally:
        sys.path[:] = original
