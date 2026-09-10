# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import math

import pytest
import torch

from tools.minimax_h3_qwen_xllm import _compare, _legacy_max_abs_pass, _metric_summary, _ordered_bf16


def test_metric_summary_reports_relative_l2_and_stable_cosine() -> None:
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    actual = expected.clone()
    actual[0, 0] = torch.tensor(1.5, dtype=torch.bfloat16)

    metrics = _metric_summary(actual, expected)

    assert metrics["max_abs_error"] == 0.5
    assert metrics["mean_abs_error"] == 0.125
    assert metrics["relative_l2"] == pytest.approx(math.sqrt(0.25 / 30.0))
    assert 0.0 < metrics["cosine"] <= 1.0
    assert metrics["abs_error_percentiles"]["p99"] == pytest.approx(0.485)


def test_compare_reports_independent_qwen_modalities() -> None:
    expected = torch.zeros((3, 2), dtype=torch.bfloat16)
    actual = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]], dtype=torch.bfloat16)
    masks = {
        "qwen_image": torch.tensor([True, False, True]),
        "qwen_text": torch.tensor([False, True, False]),
    }

    result = _compare("hidden", actual, expected, masks)

    assert result["modalities"]["qwen_image"]["count"] == 4
    assert result["modalities"]["qwen_image"]["max_abs_error"] == 3.0
    assert result["modalities"]["qwen_text"]["count"] == 2
    assert result["modalities"]["qwen_text"]["max_abs_error"] == 2.0


def test_metric_mask_must_match_leading_dimension() -> None:
    value = torch.zeros((3, 2), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="one-dimensional boolean"):
        _metric_summary(value, value, torch.ones((3, 1), dtype=torch.bool))
    with pytest.raises(ValueError, match="boolean and match"):
        _metric_summary(value, value, torch.ones(2, dtype=torch.int64))


def test_ordered_bf16_collapses_signed_zero_and_counts_adjacent_values() -> None:
    values = torch.tensor([-0.0, 0.0, 1.0, 1.0078125], dtype=torch.bfloat16)
    ordered = _ordered_bf16(values)

    assert int((ordered[0] - ordered[1]).abs()) == 0
    assert int((ordered[2] - ordered[3]).abs()) == 1


def test_metric_summary_reports_sign_crossings_separately() -> None:
    expected = torch.tensor([-1.0, 0.0], dtype=torch.bfloat16)
    actual = torch.tensor([1.0, -0.0], dtype=torch.bfloat16)

    metrics = _metric_summary(actual, expected)

    assert metrics["bf16"]["sign_crossing_count"] == 1
    assert metrics["bf16"]["same_sign_ulp_max"] == 0


def test_legacy_gate_rejects_nonfinite_mismatches() -> None:
    comparison = _compare(
        "nonfinite",
        torch.tensor([float("nan")]),
        torch.tensor([0.0]),
    )

    assert comparison["max_abs_error"] == 0.0
    assert comparison["nonfinite_mismatch_count"] == 1
    assert not _legacy_max_abs_pass([comparison], 0.25)
