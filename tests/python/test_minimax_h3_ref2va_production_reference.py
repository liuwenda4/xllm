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
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tools.minimax_h3_ref2va_production_reference import (
    ALIGNED_ROWS,
    AUDIO_COMPACT_ROWS,
    CONDITION_SHAPE,
    CONDITION_TAGS_SHA256,
    DEFAULT_DIFFUSERS_SOURCE,
    DEFAULT_REFERENCE_IMAGE,
    IMAGE_ROWS,
    NATIVE_CONDITION_SHA256,
    NATIVE_HIDDEN_SHA256,
    NATIVE_SUMMARY_SHA256,
    OFFICIAL_CONDITION_CACHE_KEY,
    OFFICIAL_CONDITION_MANIFEST_SHA256,
    OFFICIAL_CONDITION_SHA256,
    OFFICIAL_HIDDEN_SHA256,
    PINNED_DIFFUSERS_REVISION,
    PREPARED_CONDITION_KEYS,
    PREPARED_SCHEMA,
    PREPARED_TENSOR_SPECS,
    PRODUCTION_CHECKPOINT_SHA256,
    PRODUCTION_DIFFUSERS_SOURCE_SHA256,
    REFERENCE_IMAGE_SHA256,
    REQUEST_DRAW_ORDER,
    REQUEST_SEED,
    RESIZED_IMAGE_SIZE,
    RESIZED_PIXELS_SHAPE,
    ROOT_CHECKPOINT_SHA256,
    SOURCE_IMAGE_SIZE,
    TARGET_AUDIO_ROWS,
    TARGET_VIDEO_ROWS,
    TEXT_ROWS,
    USED_ROWS,
    VIDEO_COMPACT_ROWS,
    VISUAL_LATENT_SHAPE,
    PackedLayout,
    TensorSpec,
    _activate_diffusers_source,
    _build_production_layout,
    _condition_tensor_key,
    _draw_request_noise,
    _geometry_manifest,
    _layout_archive,
    _prepare_reference_pixels,
    _prove_pinned_randn_equivalence,
    _reference_resize_geometry,
    _sha256,
    _tensor_summary,
    _torch_randn_tensor,
    _validate_artifact_pair,
    _validate_prepared_manifest,
    _validate_production_layout,
    _validate_tensor_contract,
    _with_digest,
    _write_artifact_pair,
)


def _small_noise(*, draw_order=REQUEST_DRAW_ORDER):
    shape = (2, 3, 4)
    return _draw_request_noise(
        seed=REQUEST_SEED,
        visual_shape=shape,
        video_shape=shape,
        audio_rows_shape=shape,
        draw_order=draw_order,
    )


def _prepared_manifest_fixture() -> dict:
    source = _with_digest(
        {
            "revision": PINNED_DIFFUSERS_REVISION,
            "files": {name: {"sha256": digest} for name, digest in PRODUCTION_DIFFUSERS_SOURCE_SHA256.items()},
        }
    )
    checkpoint = {"modular_model_index": {"sha256": ROOT_CHECKPOINT_SHA256["modular_model_index.json"]}}
    for component_name, expected_files in PRODUCTION_CHECKPOINT_SHA256.items():
        checkpoint[component_name] = {"files": {name: {"sha256": digest} for name, digest in expected_files.items()}}
    _with_digest(checkpoint)
    inputs = _with_digest(
        {
            "official_condition": {
                "sha256": OFFICIAL_CONDITION_SHA256,
                "manifest_sha256": OFFICIAL_CONDITION_MANIFEST_SHA256,
                "cache_key": OFFICIAL_CONDITION_CACHE_KEY,
                "backend": "official_hf",
                "tensor": {"sha256": OFFICIAL_HIDDEN_SHA256, "shape": list(CONDITION_SHAPE), "dtype": "bfloat16"},
            },
            "native_condition": {
                "sha256": NATIVE_CONDITION_SHA256,
                "summary_sha256": NATIVE_SUMMARY_SHA256,
                "summary_status": "H3_NATIVE_DOWNSTREAM_REPORT",
                "backend": "xllm_native",
                "tensor": {"sha256": NATIVE_HIDDEN_SHA256, "shape": list(CONDITION_SHAPE), "dtype": "bfloat16"},
            },
            "common_tags": {"sha256": CONDITION_TAGS_SHA256, "shape": [CONDITION_SHAPE[0]], "dtype": "int64"},
        }
    )
    tensors = {}
    for name, spec in PREPARED_TENSOR_SPECS.items():
        shape = spec.shape if spec.shape is not None else (3,)
        tensors[name] = {"shape": list(shape), "dtype": str(spec.dtype).removeprefix("torch.")}
    tensors["condition.official_hf.hidden"]["sha256"] = OFFICIAL_HIDDEN_SHA256
    tensors["condition.xllm_native.hidden"]["sha256"] = NATIVE_HIDDEN_SHA256
    tensors["condition.text_token_tags"]["sha256"] = CONDITION_TAGS_SHA256
    return {
        "schema": PREPARED_SCHEMA,
        "status": "C7_PRODUCTION_PREPARED",
        "gate_evaluation": {"gate": "G8", "status": "NOT_EVALUATED"},
        "source": source,
        "checkpoint": checkpoint,
        "inputs": inputs,
        "image": {"sha256": REFERENCE_IMAGE_SHA256, "source_size_wh": list(SOURCE_IMAGE_SIZE)},
        "geometry": _geometry_manifest(),
        "generators": {
            "posterior": {"device": "cpu", "seed": 42, "scope": "fresh"},
            "request": {"device": "cpu", "seed": 42, "draw_order": list(REQUEST_DRAW_ORDER)},
        },
        "schedule": {
            "video_shift": 12.0,
            "audio_shift": 3.0,
            "sigma_points": 50,
            "transformer_forwards": 49,
            "visual_anchor_timestep": 0.999,
        },
        "condition_backends": {
            name: {
                "tensor_key": key,
                "hidden_sha256": OFFICIAL_HIDDEN_SHA256 if name == "official_hf" else NATIVE_HIDDEN_SHA256,
            }
            for name, key in PREPARED_CONDITION_KEYS.items()
        },
        "tensors": tensors,
    }


def _pinned_diffusers_helpers():
    if not DEFAULT_DIFFUSERS_SOURCE.is_dir():
        pytest.skip("pinned Diffusers source is unavailable")
    _activate_diffusers_source(DEFAULT_DIFFUSERS_SOURCE)
    try:
        from diffusers.image_processor import VaeImageProcessor
        from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep
        from diffusers.utils.torch_utils import randn_tensor
    except ImportError as error:
        pytest.skip(f"pinned Diffusers dependencies are unavailable: {error}")
    return VaeImageProcessor, MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence, randn_tensor


def test_production_geometry_and_row_ranges_are_exact() -> None:
    assert _reference_resize_geometry(*SOURCE_IMAGE_SIZE) == (2048, 5536)
    assert RESIZED_IMAGE_SIZE == (5536, 2048)
    assert IMAGE_ROWS == 1 * 64 * 173
    assert TARGET_VIDEO_ROWS == 37 * 24 * 42
    assert TARGET_AUDIO_ROWS == 2 * 207
    assert USED_ROWS == TEXT_ROWS + IMAGE_ROWS + TARGET_AUDIO_ROWS + TARGET_VIDEO_ROWS == 60132
    assert ALIGNED_ROWS == 60160
    assert VIDEO_COMPACT_ROWS == IMAGE_ROWS + TARGET_VIDEO_ROWS
    assert AUDIO_COMPACT_ROWS == TARGET_AUDIO_ROWS

    geometry = _geometry_manifest()
    assert geometry["packed_order"] == ["text", "image", "target_audio", "target_video"]
    assert geometry["ranges"] == {
        "text": [0, 11350],
        "image": [11350, 22422],
        "target_audio": [22422, 22836],
        "target_video": [22836, 60132],
    }


def test_reference_resize_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _reference_resize_geometry(0, 100)
    with pytest.raises(ValueError, match="within 1:4"):
        _reference_resize_geometry(500, 100)


def test_used_row_layout_masks_counts_and_order_are_exact() -> None:
    tags = torch.arange(TEXT_ROWS, dtype=torch.int64).remainder(2)
    image_start = TEXT_ROWS
    audio_start = image_start + IMAGE_ROWS
    video_start = audio_start + TARGET_AUDIO_ROWS
    video_indices = torch.cat((torch.arange(image_start, audio_start), torch.arange(video_start, USED_ROWS)))
    audio_indices = torch.arange(audio_start, video_start)
    packed_tags = torch.cat(
        (
            tags,
            torch.zeros(IMAGE_ROWS, dtype=torch.int64),
            torch.full((TARGET_AUDIO_ROWS,), 2, dtype=torch.int64),
            torch.zeros(TARGET_VIDEO_ROWS, dtype=torch.int64),
        )
    )
    layout = PackedLayout(
        position_ids=torch.zeros((USED_ROWS, 3), dtype=torch.float64),
        token_tags=packed_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=torch.arange(TEXT_ROWS),
        num_condition_video_rows=IMAGE_ROWS,
        num_condition_audio_rows=0,
    )
    _validate_production_layout(layout, tags)
    archive = _layout_archive(layout)

    assert archive["layout.ranges"].tolist() == [
        [0, 11350],
        [11350, 22422],
        [22422, 22836],
        [22836, 60132],
    ]
    assert archive["layout.video_update_mask"].tolist() == [False] * IMAGE_ROWS + [True] * TARGET_VIDEO_ROWS
    assert bool(archive["layout.audio_update_mask"].all())
    assert int(archive["layout.aligned_valid_mask"].sum()) == USED_ROWS
    assert not bool(archive["layout.aligned_valid_mask"][USED_ROWS:].any())


def test_request_rng_matches_one_sequential_cpu_generator() -> None:
    actual = _small_noise()
    generator = torch.Generator(device="cpu").manual_seed(REQUEST_SEED)
    expected = [torch.randn((2, 3, 4), generator=generator) for _ in REQUEST_DRAW_ORDER]

    assert torch.equal(actual.visual_anchor_noise, expected[0])
    assert torch.equal(actual.target_video_initial, expected[1])
    assert torch.equal(actual.target_audio_rows, expected[2])


def test_request_rng_is_deterministic_and_does_not_touch_global_state() -> None:
    torch.manual_seed(9182)
    state = torch.get_rng_state().clone()
    first = _small_noise()
    second = _small_noise()

    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(first.visual_anchor_noise, second.visual_anchor_noise)
    assert torch.equal(first.target_video_initial, second.target_video_initial)
    assert torch.equal(first.target_audio_rows, second.target_audio_rows)


def test_swapping_or_restarting_request_draws_changes_assigned_tensors() -> None:
    ordered = _small_noise()
    swapped = _small_noise(draw_order=("target_video_initial", "visual_anchor_noise", "target_audio_rows"))
    restarted = _draw_request_noise(
        visual_shape=(2, 3, 4),
        video_shape=(2, 3, 4),
        audio_rows_shape=(2, 3, 4),
        randn_tensor=lambda shape, **kwargs: _torch_randn_tensor(
            shape,
            generator=torch.Generator(device="cpu").manual_seed(REQUEST_SEED),
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        ),
    )

    assert not torch.equal(ordered.visual_anchor_noise, swapped.visual_anchor_noise)
    assert not torch.equal(ordered.target_video_initial, swapped.target_video_initial)
    assert torch.equal(restarted.visual_anchor_noise, restarted.target_video_initial)
    assert not torch.equal(ordered.target_video_initial, restarted.target_video_initial)
    assert not torch.equal(ordered.target_audio_rows, restarted.target_audio_rows)


def test_backend_selection_is_explicit_and_has_no_fallback() -> None:
    assert _condition_tensor_key("official_hf") == "condition.official_hf.hidden"
    assert _condition_tensor_key("xllm_native") == "condition.xllm_native.hidden"
    with pytest.raises(ValueError, match="no fallback"):
        _condition_tensor_key("unknown")

    manifest = _prepared_manifest_fixture()
    manifest["condition_backends"]["xllm_native"]["tensor_key"] = "condition.official_hf.hidden"
    with pytest.raises(ValueError, match="backend provenance mismatch"):
        _condition_tensor_key("xllm_native", manifest)


def test_prepared_manifest_validation_fails_closed() -> None:
    manifest = _prepared_manifest_fixture()
    _validate_prepared_manifest(manifest)

    wrong_revision = copy.deepcopy(manifest)
    wrong_revision["source"]["revision"] = "not-the-pinned-revision"
    wrong_revision["source"].pop("digest")
    _with_digest(wrong_revision["source"])
    with pytest.raises(ValueError, match="revision mismatch"):
        _validate_prepared_manifest(wrong_revision)

    wrong_draw_order = copy.deepcopy(manifest)
    wrong_draw_order["generators"]["request"]["draw_order"] = list(reversed(REQUEST_DRAW_ORDER))
    with pytest.raises(ValueError, match="draw order mismatch"):
        _validate_prepared_manifest(wrong_draw_order)

    wrong_dtype = copy.deepcopy(manifest)
    wrong_dtype["tensors"]["input.visual_anchor_noise"]["dtype"] = "float16"
    with pytest.raises(ValueError, match="dtype mismatch"):
        _validate_prepared_manifest(wrong_dtype)


def test_tensor_contract_and_artifact_pair_validation(tmp_path: Path) -> None:
    tensors = {
        "float": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "index": torch.tensor([2, 1, 0], dtype=torch.int64),
    }
    specs = {
        "float": TensorSpec((2, 3), torch.float32),
        "index": TensorSpec((3,), torch.int64),
    }
    _validate_tensor_contract(tensors, specs)
    assert _tensor_summary(torch.tensor(7, dtype=torch.int64))["shape"] == []
    with pytest.raises(ValueError, match="shape"):
        _validate_tensor_contract({**tensors, "float": tensors["float"].reshape(3, 2)}, specs)

    archive_path = tmp_path / "prepared.safetensors"
    save_file(tensors, str(archive_path))
    manifest = {
        "artifact": {
            "path": archive_path.name,
            "size": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
        },
        "tensors": {name: _tensor_summary(value) for name, value in tensors.items()},
    }
    retained = _validate_artifact_pair(archive_path, manifest, retain_names={"index"})
    assert set(retained) == {"index"}
    assert torch.equal(retained["index"], tensors["index"])

    corrupted = copy.deepcopy(manifest)
    corrupted["artifact"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="size or SHA256"):
        _validate_artifact_pair(archive_path, corrupted)

    shared = torch.arange(4, dtype=torch.float32)
    aliased = {"first": shared, "second": shared}
    aliased_manifest = {"tensors": {name: _tensor_summary(value) for name, value in aliased.items()}}
    _write_artifact_pair(tmp_path, "aliased.safetensors", "aliased.json", aliased, aliased_manifest)
    retained = _validate_artifact_pair(
        tmp_path / "aliased.safetensors", aliased_manifest, retain_names={"first", "second"}
    )
    assert torch.equal(retained["first"], retained["second"])


def test_pinned_randn_tensor_is_exactly_cpu_torch_randn() -> None:
    _, _, randn_tensor = _pinned_diffusers_helpers()
    proof = _prove_pinned_randn_equivalence(randn_tensor)
    assert proof["proved"] is True


def test_pinned_diffusers_builds_exact_production_layout() -> None:
    _, layout_builder, _ = _pinned_diffusers_helpers()
    tags = torch.ones(CONDITION_SHAPE[0], dtype=torch.int64)
    visual_latent = torch.empty(VISUAL_LATENT_SHAPE, dtype=torch.float32)
    layout = _build_production_layout(layout_builder, tags, visual_latent)

    assert layout.sequence_length == USED_ROWS
    assert layout.text_indices.tolist() == list(range(TEXT_ROWS))
    assert layout.video_indices[:IMAGE_ROWS].tolist() == list(range(TEXT_ROWS, TEXT_ROWS + IMAGE_ROWS))
    assert layout.audio_indices.tolist() == list(
        range(TEXT_ROWS + IMAGE_ROWS, TEXT_ROWS + IMAGE_ROWS + TARGET_AUDIO_ROWS)
    )
    assert layout.video_indices[IMAGE_ROWS:].tolist() == list(
        range(TEXT_ROWS + IMAGE_ROWS + TARGET_AUDIO_ROWS, USED_ROWS)
    )
    assert layout.num_condition_video_rows == IMAGE_ROWS
    assert layout.num_condition_audio_rows == 0


@pytest.mark.skipif(
    not DEFAULT_REFERENCE_IMAGE.is_file(),
    reason="pinned real reference image is unavailable",
)
def test_pinned_diffusers_resizes_real_reference_to_exact_uint8_pixels() -> None:
    VaeImageProcessor, _, _ = _pinned_diffusers_helpers()
    processor = VaeImageProcessor(vae_scale_factor=16)
    pixels = _prepare_reference_pixels(DEFAULT_REFERENCE_IMAGE, processor)

    assert processor.config.resample == "lanczos"
    assert pixels.dtype == torch.uint8
    assert pixels.is_contiguous()
    assert pixels.shape == RESIZED_PIXELS_SHAPE
