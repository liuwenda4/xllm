# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import pytest
import torch

from tools.minimax_h3_video_vae_reference import (
    DEFAULT_DIFFUSERS_SOURCE,
    DIFFUSERS_SOURCE_SHA256,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SHAPE,
    LATENT_SHAPE,
    POSTERIOR_LOGVAR_MAX,
    POSTERIOR_LOGVAR_MIN,
    _denormalize_latents,
    _deterministic_imagenet_input,
    _deterministic_normalized_latent,
    _fixture_plan,
    _normalize_latents,
    _posterior_math,
    _source_manifest,
    _temporal_plan,
    _tile_plan,
    _validate_source,
)


def test_deterministic_imagenet_input_is_analytic_fp32() -> None:
    video = _deterministic_imagenet_input()
    repeated = _deterministic_imagenet_input()

    assert video.shape == INPUT_SHAPE
    assert video.dtype == torch.float32
    assert video.is_contiguous()
    assert torch.equal(video, repeated)
    assert bool(video.isfinite().all())
    assert video[0, 0, 0, 0, 0].item() == pytest.approx((0.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0])
    pixel = ((29 * 2 + 17 * 38 + 7 * 287 + 3 * 287) % 256) / 255.0
    assert video[0, 2, 38, 287, 287].item() == pytest.approx((pixel - IMAGENET_MEAN[2]) / IMAGENET_STD[2])


def test_independent_decode_latent_is_deterministic_normalized_fp32() -> None:
    latent = _deterministic_normalized_latent()

    assert latent.shape == LATENT_SHAPE
    assert latent.dtype == torch.float32
    assert latent.is_contiguous()
    assert torch.equal(latent, _deterministic_normalized_latent())
    assert float(latent.min()) == -1.0
    assert float(latent.max()) == 1.0
    assert latent[0, 23, 11, 17, 17].item() == pytest.approx(
        ((53 * 23 + 29 * 11 + 11 * 17 + 7 * 17) % 257 - 128) / 128.0
    )


def test_temporal_plan_maps_39_frames_to_12_and_back() -> None:
    plan = _temporal_plan()
    encoder = plan["encoder"]
    decoder = plan["decoder"]

    assert encoder["input_frames"] == 39
    assert encoder["padded_input_frames"] == 51
    assert encoder["repeated_tail_frames"] == 12
    assert encoder["chunk_count"] == 3
    assert encoder["concatenated_moments_frames"] == 15
    assert encoder["trailing_tokens_dropped"] == 3
    assert encoder["output_latent_frames"] == 12
    assert decoder["input_latent_frames"] == 12
    assert decoder["chunk_count"] == 2
    assert decoder["token_overlap"] == 2
    assert decoder["frame_pre_padding"] == 3
    assert decoder["frame_overlap"] == 5
    assert decoder["output_frames"] == 39
    assert [clip["latent_input_range"] for clip in decoder["clip_plan"]] == [[0, 7], [5, 12]]


def test_tile_plan_has_real_two_by_two_grid_and_exact_overlaps() -> None:
    axis = _tile_plan()
    fixture = _fixture_plan()

    assert axis["sample_starts"] == [0, 32]
    assert axis["sample_lengths"] == [256, 256]
    assert axis["sample_overlaps"] == [224]
    assert axis["latent_starts"] == [0, 2]
    assert axis["latent_lengths"] == [16, 16]
    assert axis["latent_overlaps"] == [14]
    assert fixture["spatial"]["tiles_per_clip"] == 4
    assert len(fixture["encoder_tile_call_order"]) == 12
    assert len(fixture["decoder_tile_call_order"]) == 8
    assert fixture["encoder_tile_call_order"][0]["sample_y_range"] == [0, 256]
    assert fixture["encoder_tile_call_order"][-1]["sample_x_range"] == [32, 288]
    assert fixture["decoder_tile_call_order"][-1]["temporal_chunk"] == 1
    assert fixture["expected_calls"] == {
        "encoder_quant_conv": 12,
        "decoder_post_quant_conv": 8,
        "decoder_rope": 8,
        "decoder_block_0": 8,
        "decoder_block_35": 8,
    }


def test_posterior_math_clamps_logvar_and_uses_explicit_epsilon() -> None:
    mean = torch.tensor([1.5, -2.0], dtype=torch.float32).view(1, 2, 1, 1, 1)
    raw_logvar = torch.tensor([-40.0, 30.0], dtype=torch.float32).view(1, 2, 1, 1, 1)
    moments = torch.cat([mean, raw_logvar], dim=1)
    epsilon = torch.tensor([0.25, -0.5], dtype=torch.float32).view_as(mean)

    posterior = _posterior_math(moments, epsilon)

    expected_logvar = torch.tensor([POSTERIOR_LOGVAR_MIN, POSTERIOR_LOGVAR_MAX]).view_as(mean)
    expected_std = torch.exp(0.5 * expected_logvar)
    assert torch.equal(posterior["mean"], mean)
    assert torch.equal(posterior["raw_logvar"], raw_logvar)
    assert torch.equal(posterior["logvar"], expected_logvar)
    assert torch.equal(posterior["std"], expected_std)
    assert torch.equal(posterior["sample"], mean + expected_std * epsilon)


def test_fp16_rounding_and_per_channel_normalize_roundtrip() -> None:
    value = torch.tensor(
        [
            [
                [[[0.123456, -1.23456]]],
                [[[2.34567, -3.45678]]],
            ]
        ],
        dtype=torch.float32,
    )
    rounded = value.to(torch.float16).to(torch.float32)
    latents_mean = (0.25, -0.75)
    latents_std = (1.5, 2.25)

    normalized = _normalize_latents(rounded, latents_mean, latents_std)
    restored = _denormalize_latents(normalized, latents_mean, latents_std)

    assert normalized.dtype == torch.float32
    assert restored.dtype == torch.float32
    assert torch.allclose(restored, rounded, atol=3e-7, rtol=0.0)


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
