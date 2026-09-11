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

from tools.minimax_h3_block_reference import (
    DEFAULT_PACKING_GOLDEN,
    _canonical_digest,
    _deterministic_rows,
    _packing_fixture,
    _tensor_summary,
)


def test_deterministic_rows_are_stable_and_bounded() -> None:
    rows = _deterministic_rows(3, 4, 41)

    assert rows.shape == (3, 4)
    assert rows.dtype == torch.float32
    assert rows.is_contiguous()
    assert torch.equal(rows, _deterministic_rows(3, 4, 41))
    assert float(rows.min()) >= -1.0
    assert float(rows.max()) < 1.0


def test_canonical_digest_ignores_mapping_insertion_order() -> None:
    assert _canonical_digest({"a": 1, "b": 2}) == _canonical_digest({"b": 2, "a": 1})


def test_tensor_summary_records_nonfinite_state() -> None:
    summary = _tensor_summary(torch.tensor([1.0, float("nan")]))

    assert summary["shape"] == [2]
    assert summary["dtype"] == "float32"
    assert summary["finite"] is False
    assert len(summary["sha256"]) == 64


@pytest.mark.skipif(not DEFAULT_PACKING_GOLDEN.is_file(), reason="C3 packing Golden is unavailable")
def test_packing_fixture_is_attested_one_image_contract() -> None:
    fixture, metadata = _packing_fixture(DEFAULT_PACKING_GOLDEN)

    assert metadata["used_length"] == 23
    assert metadata["aligned_length"] == 64
    assert len(metadata["sha256"]) == 64
    assert fixture["condition_hidden"].shape == (1, 3, 5120)
    assert fixture["cu_seqlens"].tolist() == [0, 23, 64]
