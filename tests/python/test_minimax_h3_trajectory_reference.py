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

from tools.minimax_h3_block_reference import DEFAULT_PACKING_GOLDEN
from tools.minimax_h3_trajectory_reference import (
    AUDIO_CONDITION_TIMESTEP,
    IMAGE_CONDITION_TIMESTEP,
    _packing_fixture,
    _row_timestep_plan,
)


@pytest.mark.skipif(not DEFAULT_PACKING_GOLDEN.is_file(), reason="C3 packing Golden is unavailable")
def test_image_audio_fixture_contains_both_anchor_modalities() -> None:
    fixture, metadata = _packing_fixture(DEFAULT_PACKING_GOLDEN)

    assert metadata["case"] == "image_audio"
    assert metadata["used_length"] == 27
    assert metadata["aligned_length"] == 64
    assert int((~fixture["update_mask"]).sum()) == 4
    assert int((~fixture["audio_update_mask"]).sum()) == 4


@pytest.mark.skipif(not DEFAULT_PACKING_GOLDEN.is_file(), reason="C3 packing Golden is unavailable")
def test_row_timestep_plan_pins_anchors_and_padding() -> None:
    fixture, _ = _packing_fixture(DEFAULT_PACKING_GOLDEN)
    rows, unique, inverse = _row_timestep_plan(fixture, 0.25, 0.5)
    image_positions = fixture["img_pos"].view(-1)
    audio_positions = fixture["audio_pos"].view(-1)
    image_update = fixture["update_mask"].view(-1)
    audio_update = fixture["audio_update_mask"].view(-1)

    assert rows.shape == (64,)
    assert rows.dtype == torch.float32
    assert torch.equal(rows[image_positions[image_update]], torch.full((12,), 0.25))
    assert torch.equal(rows[image_positions[~image_update]], torch.full((4,), IMAGE_CONDITION_TIMESTEP))
    assert torch.equal(rows[audio_positions[audio_update]], torch.full((4,), 0.5))
    assert torch.equal(rows[audio_positions[~audio_update]], torch.full((4,), AUDIO_CONDITION_TIMESTEP))
    assert torch.equal(rows[27:], torch.full((37,), 0.25))
    assert torch.equal(
        unique,
        torch.tensor(
            [0.25, 0.5, IMAGE_CONDITION_TIMESTEP, AUDIO_CONDITION_TIMESTEP],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(unique.index_select(0, inverse), rows)
