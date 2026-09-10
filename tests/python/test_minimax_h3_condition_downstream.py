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

from tools.minimax_h3_condition_downstream import _build_text_prefix_layout, _normalize_native_hidden


def test_normalize_native_hidden_accepts_singleton_batch() -> None:
    hidden = torch.zeros((1, 3, 5120), dtype=torch.float32)

    normalized = _normalize_native_hidden({"hidden": hidden}, "hidden", torch.Size((3, 5120)))

    assert normalized.shape == (3, 5120)
    assert normalized.dtype == torch.bfloat16
    assert normalized.is_contiguous()


def test_normalize_native_hidden_rejects_shape_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="does not match official"):
        _normalize_native_hidden({"hidden": torch.zeros(2, 4)}, "hidden", torch.Size((3, 4)))

    hidden = torch.zeros((3, 4), dtype=torch.float32)
    hidden[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite floating-point"):
        _normalize_native_hidden({"hidden": hidden}, "hidden", torch.Size((3, 4)))


def test_text_prefix_layout_matches_h3_contract() -> None:
    tags = torch.tensor([0, 1, 2], dtype=torch.int64)

    layout = _build_text_prefix_layout(tags, torch.device("cpu"))

    torch.testing.assert_close(layout["position_ids"][:, 0], torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64))
    assert torch.count_nonzero(layout["position_ids"][:, 1:]) == 0
    assert layout["timestep_indices"].tolist() == [0, 0, 0]
    assert layout["token_tags"].tolist() == [0, 1, 2]
    assert layout["adaln_indices"].tolist() == [0, 1, 2]
