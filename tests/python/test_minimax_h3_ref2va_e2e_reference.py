# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from tools.minimax_h3_ref2va_e2e_reference import (
    AUDIO_REFERENCE_SHAPE,
    C3_PACKING_SHA256,
    CONDITION_SHAPE,
    CONDITION_TAGS,
    DEFAULT_C3_PACKING_GOLDEN,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DIFFUSERS_SOURCE,
    DIFFUSERS_SOURCE_SHA256,
    PATCH_SIZE,
    TARGET_AUDIO_SHAPE,
    TARGET_VIDEO_SHAPE,
    VISUAL_ANCHOR_TIMESTEP,
    VISUAL_REFERENCE_SHAPE,
    ReferenceSpec,
    _activate_diffusers_source,
    _build_c7_layout,
    _build_row_plans,
    _c7_references,
    _extract_target_rows,
    _initial_compact_rows,
    _mix_visual_anchor,
    _pack_audio_latents,
    _pack_video_latents,
    _prepare_explicit_inputs,
    _replace_target_rows,
    _source_manifest,
    _target_update_mask,
    _unpack_audio_rows,
    _unpack_video_rows,
    _validate_c3_equivalence,
    _validate_references,
)


def _official_helpers():
    _activate_diffusers_source(DEFAULT_DIFFUSERS_SOURCE)
    try:
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3Ref2VAPrepareLayoutStep,
            MiniMaxH3SetTimestepsStep,
        )
    except ImportError as error:
        pytest.skip(f"pinned Diffusers dependencies are unavailable: {error}")

    return MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence, MiniMaxH3SetTimestepsStep.build_row_timesteps


@pytest.mark.skipif(not DEFAULT_DIFFUSERS_SOURCE.is_dir(), reason="pinned Diffusers source is unavailable")
def test_c7_layout_has_exact_c3_order_positions_tags_and_shapes() -> None:
    layout_builder, _ = _official_helpers()
    layout = _build_c7_layout(layout_builder, _prepare_explicit_inputs())

    assert layout.sequence_length == 539
    assert layout.position_ids.shape == (539, 3)
    assert layout.position_ids.dtype == torch.float64
    assert layout.text_indices.tolist() == [0, 1, 2]
    assert layout.video_indices.shape == (512,)
    assert layout.audio_indices.shape == (24,)
    assert layout.video_indices[:64].tolist() == list(range(3, 67))
    assert layout.audio_indices[:8].tolist() == list(range(67, 75))
    assert layout.audio_indices[8:].tolist() == list(range(75, 91))
    assert layout.video_indices[64:].tolist() == list(range(91, 539))
    assert layout.num_condition_video_rows == 64
    assert layout.num_condition_audio_rows == 8
    assert layout.token_tags[:3].tolist() == list(CONDITION_TAGS)
    assert torch.count_nonzero(layout.token_tags == 0).item() == 513
    assert torch.count_nonzero(layout.token_tags == 1).item() == 2
    assert torch.count_nonzero(layout.token_tags == 2).item() == 24

    assert torch.equal(layout.position_ids[:3, 0], torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64))
    assert torch.equal(layout.position_ids[3:67, 0], torch.full((64,), 3.0, dtype=torch.float64))
    assert torch.equal(layout.position_ids[67:75, 0], torch.tensor([4, 5, 6, 7, 4, 5, 6, 7], dtype=torch.float64))
    assert torch.equal(
        layout.position_ids[75:91, 0],
        torch.tensor([*range(8, 16), *range(8, 16)], dtype=torch.float64),
    )
    assert layout.position_ids[91, 0].item() == 8.0
    assert layout.position_ids[3, 1:].tolist() == [0.0, 0.0]
    assert layout.position_ids[66, 1:].tolist() == [28.0, 28.0]


def test_compact_row_order_shapes_and_roundtrips() -> None:
    inputs = _prepare_explicit_inputs()
    visual_anchor = _mix_visual_anchor(inputs.visual_reference, inputs.visual_noise)
    video_rows, audio_rows, video_mask, audio_mask = _initial_compact_rows(inputs, visual_anchor)

    assert video_rows.shape == (512, 96)
    assert audio_rows.shape == (24, 32)
    assert torch.equal(video_rows[:64], _pack_video_latents(visual_anchor, PATCH_SIZE))
    assert torch.equal(video_rows[64:], _pack_video_latents(inputs.target_video_initial, PATCH_SIZE))
    assert torch.equal(audio_rows[:8], _pack_audio_latents(inputs.audio_reference))
    assert torch.equal(audio_rows[8:], _pack_audio_latents(inputs.target_audio_initial))
    assert torch.equal(_unpack_video_rows(video_rows[:64], VISUAL_REFERENCE_SHAPE), visual_anchor)
    assert torch.equal(_unpack_video_rows(video_rows[64:], TARGET_VIDEO_SHAPE), inputs.target_video_initial)
    assert torch.equal(_unpack_audio_rows(audio_rows[:8], AUDIO_REFERENCE_SHAPE[2]), inputs.audio_reference)
    assert torch.equal(_unpack_audio_rows(audio_rows[8:], TARGET_AUDIO_SHAPE[2]), inputs.target_audio_initial)
    assert video_mask.tolist() == [False] * 64 + [True] * 448
    assert audio_mask.tolist() == [False] * 8 + [True] * 16


def test_visual_anchor_mixing_and_target_replacement_preserve_anchors() -> None:
    inputs = _prepare_explicit_inputs()
    anchor = _mix_visual_anchor(inputs.visual_reference, inputs.visual_noise)
    timestep = torch.tensor(VISUAL_ANCHOR_TIMESTEP, dtype=torch.float32)
    expected = timestep * inputs.visual_reference
    expected += (1.0 - timestep) * inputs.visual_noise
    assert torch.equal(anchor, expected)

    rows = torch.arange(30, dtype=torch.float32).reshape(6, 5)
    mask = _target_update_mask(6, 2)
    replacement = torch.full((4, 5), -7.0)
    updated = _replace_target_rows(rows, replacement, mask)
    assert torch.equal(updated[:2], rows[:2])
    assert torch.equal(updated[2:], replacement)
    assert torch.equal(rows, torch.arange(30, dtype=torch.float32).reshape(6, 5))


def test_mask_target_extraction_is_compact_and_validated() -> None:
    rows = torch.arange(35, dtype=torch.float32).reshape(7, 5)
    mask = torch.tensor([False, False, True, False, True, True, False])
    extracted = _extract_target_rows(rows, mask)

    assert extracted.is_contiguous()
    assert torch.equal(extracted, rows[[2, 4, 5]])
    with pytest.raises(ValueError, match="one-dimensional bool"):
        _extract_target_rows(rows, mask.to(torch.int64))
    with pytest.raises(ValueError, match="both anchors and target"):
        _extract_target_rows(rows, torch.ones(7, dtype=torch.bool))


def test_explicit_inputs_are_rng_independent_and_exactly_typed() -> None:
    torch.manual_seed(12345)
    state_before = torch.get_rng_state().clone()
    first = _prepare_explicit_inputs()
    state_after = torch.get_rng_state().clone()
    torch.rand(97)
    second = _prepare_explicit_inputs()

    assert torch.equal(state_before, state_after)
    for field in fields(first):
        assert torch.equal(getattr(first, field.name), getattr(second, field.name))
    assert first.condition_hidden.shape == CONDITION_SHAPE
    assert first.condition_hidden.dtype == torch.bfloat16
    assert first.condition_tags.tolist() == [list(CONDITION_TAGS)]
    assert first.visual_reference.shape == VISUAL_REFERENCE_SHAPE
    assert first.visual_noise.shape == VISUAL_REFERENCE_SHAPE
    assert first.audio_reference.shape == AUDIO_REFERENCE_SHAPE
    assert first.target_video_initial.shape == TARGET_VIDEO_SHAPE
    assert first.target_audio_initial.shape == TARGET_AUDIO_SHAPE
    assert all(getattr(first, field.name).is_contiguous() for field in fields(first))


@pytest.mark.parametrize(
    ("references", "visual", "audio", "message"),
    [
        ([ReferenceSpec("audio", True)], [], [torch.empty(8, 32)], "require at least one image or video"),
        ([ReferenceSpec("mesh")], [], [], "invalid reference kind"),
        ([ReferenceSpec("image", True)], [torch.empty(1, 24, 1, 16, 16)], [], "cannot carry audio"),
        (
            [ReferenceSpec("audio", False), ReferenceSpec("image")],
            [torch.empty(1, 24, 1, 16, 16)],
            [],
            "must carry audio",
        ),
        ([ReferenceSpec("image")], [], [], "latent count mismatch"),
        ([ReferenceSpec("image")], [torch.empty(24, 1, 16, 16)], [], "invalid visual"),
        (
            [ReferenceSpec("image"), ReferenceSpec("audio", True)],
            [torch.empty(1, 24, 1, 16, 16)],
            [torch.empty(7, 32)],
            "invalid audio",
        ),
    ],
)
def test_invalid_references_are_rejected(references, visual, audio, message) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_references(references, visual, audio)


@pytest.mark.skipif(not DEFAULT_C3_PACKING_GOLDEN.is_file(), reason="C3 packing Golden is unavailable")
@pytest.mark.skipif(not DEFAULT_DIFFUSERS_SOURCE.is_dir(), reason="pinned Diffusers source is unavailable")
def test_official_diffusers_layout_is_exactly_equivalent_to_c3_used_rows() -> None:
    layout_builder, _ = _official_helpers()
    metadata = _validate_c3_equivalence(layout_builder, DEFAULT_C3_PACKING_GOLDEN)

    assert metadata["case"] == "image_audio"
    assert metadata["used_length"] == 27
    assert metadata["aligned_length"] == 64
    assert metadata["diffusers_used_rows_exact"] is True
    assert metadata["sha256"] == C3_PACKING_SHA256


@pytest.mark.skipif(not DEFAULT_CHECKPOINT_ROOT.is_dir(), reason="pinned checkpoint is unavailable")
@pytest.mark.skipif(not DEFAULT_DIFFUSERS_SOURCE.is_dir(), reason="pinned Diffusers source is unavailable")
def test_native_checkpoint_schedulers_have_50_points_and_49_row_plans() -> None:
    layout_builder, row_builder = _official_helpers()
    from diffusers import MiniMaxH3Scheduler

    inputs = _prepare_explicit_inputs()
    layout = _build_c7_layout(layout_builder, inputs)
    video = MiniMaxH3Scheduler.from_pretrained(DEFAULT_CHECKPOINT_ROOT / "scheduler", local_files_only=True)
    audio = MiniMaxH3Scheduler.from_pretrained(DEFAULT_CHECKPOINT_ROOT / "audio_scheduler", local_files_only=True)
    video.set_timesteps(50)
    audio.set_timesteps(50)
    plans = _build_row_plans(row_builder, layout, video.timesteps, audio.timesteps)

    assert video.config.shift == 12.0
    assert audio.config.shift == 3.0
    assert video.sigmas.shape == (50,)
    assert audio.sigmas.shape == (50,)
    assert video.timesteps.shape == (49,)
    assert audio.timesteps.shape == (49,)
    assert len(plans) == 49
    first_unique, first_inverse = plans[0]
    assert torch.equal(first_unique, torch.tensor([0.0, VISUAL_ANCHOR_TIMESTEP, 1.0]))
    assert first_inverse.shape == (539,)
    assert torch.equal(
        first_unique[first_inverse],
        torch.cat(
            (
                torch.zeros(3),
                torch.full((64,), VISUAL_ANCHOR_TIMESTEP),
                torch.ones(8),
                torch.zeros(16 + 448),
            )
        ),
    )


@pytest.mark.skipif(not DEFAULT_DIFFUSERS_SOURCE.is_dir(), reason="pinned Diffusers source is unavailable")
def test_current_diffusers_sources_match_all_c7_pins() -> None:
    source = _source_manifest(DEFAULT_DIFFUSERS_SOURCE)

    assert source["revision"] == "d30c748f5f5d0925a5af14dc0e6a6de983025e63"
    assert set(source["files"]) == set(DIFFUSERS_SOURCE_SHA256)
    assert all(entry["sha256"] == DIFFUSERS_SOURCE_SHA256[name] for name, entry in source["files"].items())


def test_c7_reference_order_is_image_then_audio() -> None:
    references = _c7_references()

    assert references == (ReferenceSpec("image"), ReferenceSpec("audio", has_audio=True))
