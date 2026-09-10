# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

import tools.minimax_h3_condition_cache as condition_cache
from tools.minimax_h3_condition_cache import (
    CacheValidationError,
    build_cache_key,
    build_cache_key_inputs,
    canonical_json_bytes,
    create_condition_cache,
    extract_condition_tensors,
    validate_cache_entry,
)


def _hidden(token_count: int = 3, *, batched: bool = True) -> torch.Tensor:
    value = torch.arange(token_count * 5120, dtype=torch.float32).reshape(5120, token_count).T.to(torch.bfloat16)
    return value.unsqueeze(0) if batched else value


def _save_archive(path: Path, hidden: torch.Tensor, tags: torch.Tensor | None = None) -> None:
    tensors = {"layer_49_output_pre_final_norm": hidden}
    if tags is not None:
        tensors["h3_token_tags"] = tags
    torch.save(tensors, path)


def _official_cache_args(tmp_path: Path) -> dict[str, Any]:
    source_archive = tmp_path / "qwen_layer50_reference.pt"
    hidden = _hidden()
    _save_archive(source_archive, hidden, torch.tensor([0, 1, 0], dtype=torch.int64))
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_bytes(b"local prompt\n")
    reference_path = tmp_path / "reference.bin"
    reference_path.write_bytes(b"local reference")
    checkpoint_manifest = tmp_path / "checkpoint.json"
    checkpoint_manifest.write_text('{"checkpoint_digest":"abc"}\n', encoding="utf-8")
    official_summary = tmp_path / "summary.json"
    official_summary.write_text(
        json.dumps(
            {
                "status": "H3_QWEN_LAYER50_REFERENCE_PASS",
                "tensor_archive_sha256": condition_cache.file_sha256(source_archive),
                "prompt_sha256": hashlib.sha256(b"local prompt\n").hexdigest(),
                "image_sha256": hashlib.sha256(b"local reference").hexdigest(),
                "lm_head_executed": False,
                "layer_49_output_pre_final_norm": {"shape": list(hidden.shape)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "backend": "official_hf",
        "source_archive": source_archive,
        "cache_root": tmp_path / "cache",
        "checkpoint_identity": {"revision": "local-revision"},
        "checkpoint_manifest": checkpoint_manifest,
        "official_reference_summary": official_summary,
        "prompt_path": prompt_path,
        "references": [("image", reference_path)],
        "fps": 24.0,
        "video_sample_fps": 2.0,
        "reference_short_edge": 2048,
    }


def test_cache_key_is_canonical_and_preserves_reference_order() -> None:
    common = {
        "backend": "official_hf",
        "prompt_bytes_sha256": "b" * 64,
        "ordered_references": [
            {"type": "image", "sha256": "c" * 64},
            {"type": "video", "sha256": "d" * 64},
        ],
        "fps": 24.0,
        "video_sample_fps": 2.0,
        "reference_short_edge": 2048,
    }
    first = build_cache_key(checkpoint_identity={"revision": "r1", "digest": "e" * 64}, **common)
    second = build_cache_key(checkpoint_identity={"digest": "e" * 64, "revision": "r1"}, **common)
    inputs = build_cache_key_inputs(checkpoint_identity={"revision": "r1", "digest": "e" * 64}, **common)

    assert first == second
    assert first == hashlib.sha256(canonical_json_bytes(inputs)).hexdigest()

    reversed_references = dict(common)
    reversed_references["ordered_references"] = list(reversed(common["ordered_references"]))
    assert build_cache_key(checkpoint_identity={"revision": "r1", "digest": "e" * 64}, **reversed_references) != first


def test_official_extraction_squeezes_batch_and_makes_bfloat16_contiguous() -> None:
    source = {
        "layer_49_output_pre_final_norm": _hidden(),
        "h3_token_tags": torch.tensor([0, 1, 0], dtype=torch.int64),
    }

    tensors = extract_condition_tensors("official_hf", source)

    assert tensors["prompt_embeds"].shape == (3, 5120)
    assert tensors["prompt_embeds"].dtype == torch.bfloat16
    assert tensors["prompt_embeds"].is_contiguous()
    assert tensors["text_token_tags"].dtype == torch.int64
    assert tensors["text_token_tags"].tolist() == [0, 1, 0]


def test_native_extraction_uses_only_explicit_official_tags() -> None:
    native = {
        "layer_49_output_pre_final_norm": _hidden(batched=False),
        "h3_token_tags": torch.tensor([2, 2, 2], dtype=torch.int64),
    }
    official = {"h3_token_tags": torch.tensor([0, 1, 0], dtype=torch.int64)}

    tensors = extract_condition_tensors("xllm_native", native, official)

    assert tensors["text_token_tags"].tolist() == [0, 1, 0]


def test_native_backend_never_falls_back_without_official_metadata() -> None:
    native = {"layer_49_output_pre_final_norm": _hidden(batched=False)}

    with pytest.raises(ValueError, match="explicit official metadata/golden archive.*no fallback"):
        extract_condition_tensors("xllm_native", native)


@pytest.mark.parametrize(
    ("hidden", "tags", "message"),
    [
        (_hidden().expand(2, -1, -1), torch.tensor([0, 1, 2]), "singleton batch dimension"),
        (torch.zeros(3, 64), torch.tensor([0, 1, 2]), "must normalize to"),
        (_hidden(), torch.tensor([0, 1]), "token counts differ"),
        (_hidden(), torch.tensor([0, 1, 2]), "subset of"),
        (_hidden(), torch.tensor([0, 1, 0], dtype=torch.int32), "must be int64"),
    ],
)
def test_extraction_rejects_invalid_shapes_and_tags(hidden: torch.Tensor, tags: torch.Tensor, message: str) -> None:
    source = {"layer_49_output_pre_final_norm": hidden, "h3_token_tags": tags}

    with pytest.raises(ValueError, match=message):
        extract_condition_tensors("official_hf", source)


def test_extraction_rejects_non_finite_hidden_values() -> None:
    hidden = _hidden()
    hidden[0, 0, 0] = torch.nan
    source = {
        "layer_49_output_pre_final_norm": hidden,
        "h3_token_tags": torch.tensor([0, 1, 0], dtype=torch.int64),
    }

    with pytest.raises(ValueError, match="non-finite"):
        extract_condition_tensors("official_hf", source)


def test_official_cache_manifest_and_condition_provenance(tmp_path: Path) -> None:
    arguments = _official_cache_args(tmp_path)

    cache_dir = create_condition_cache(**arguments)
    manifest = validate_cache_entry(cache_dir)
    tensors = load_file(str(cache_dir / "condition.safetensors"))

    assert cache_dir == arguments["cache_root"].resolve() / manifest["cache_key"]
    assert manifest["backend"] == "official_hf"
    assert manifest["backend_approval_status"] == "attested-offline-fixture"
    assert manifest["hidden_state"] == {"decoder_layer_index": 49, "hidden_state_slot": 50}
    assert manifest["source_archives"][0]["path"] == str(arguments["source_archive"].resolve())
    assert manifest["checkpoint_identity_inputs"]["values"] == {"revision": "local-revision"}
    assert manifest["prompt"]["sha256"] == hashlib.sha256(b"local prompt\n").hexdigest()
    assert manifest["ordered_references"][0]["type"] == "image"
    assert manifest["normalization_parameters"] == {
        "fps": 24.0,
        "video_sample_fps": 2.0,
        "reference_short_edge": 2048,
    }
    assert manifest["tensors"]["prompt_embeds"]["shape"] == [3, 5120]
    assert manifest["tensors"]["prompt_embeds"]["dtype"] == "bfloat16"
    assert manifest["tensors"]["text_token_tags"]["dtype"] == "int64"
    assert tensors["text_token_tags"].tolist() == [0, 1, 0]
    assert manifest["runtime_bundle_manifest"] == {
        "schema": "xllm.minimax_h3.text_conditioning/v1",
        "source_backend": "official_hf",
        "decoder_layer_index": 49,
        "hidden_state_slot": 50,
        "token_count": 3,
        "hidden_digest": manifest["tensors"]["prompt_embeds"]["sha256"],
        "token_tags_digest": manifest["tensors"]["text_token_tags"]["sha256"],
        "condition_cache_key": manifest["cache_key"],
    }


def test_valid_cache_hit_does_not_rewrite_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arguments = _official_cache_args(tmp_path)
    expected = create_condition_cache(**arguments)

    def fail_write(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        pytest.fail("valid cache hit attempted a rewrite")

    monkeypatch.setattr(condition_cache, "_write_cache_entry", fail_write)

    assert create_condition_cache(**arguments) == expected


def test_valid_cache_hit_does_not_wait_for_a_producer_lock(tmp_path: Path) -> None:
    arguments = _official_cache_args(tmp_path)
    cache_dir = create_condition_cache(**arguments)
    lock_path = arguments["cache_root"].resolve() / f".{cache_dir.name}.lock"
    lock_path.write_text("stale producer", encoding="utf-8")

    try:
        assert create_condition_cache(**arguments) == cache_dir
    finally:
        lock_path.unlink()


def test_official_cache_requires_matching_reference_attestation(tmp_path: Path) -> None:
    arguments = _official_cache_args(tmp_path)
    summary_path = arguments["official_reference_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["tensor_archive_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="tensor archive digest"):
        create_condition_cache(**arguments)


def test_official_cache_requires_reference_summary(tmp_path: Path) -> None:
    arguments = _official_cache_args(tmp_path)
    arguments["official_reference_summary"] = None

    with pytest.raises(ValueError, match="official-reference-summary"):
        create_condition_cache(**arguments)


def test_corrupted_condition_is_rejected_and_force_replaces_it(tmp_path: Path) -> None:
    arguments = _official_cache_args(tmp_path)
    cache_dir = create_condition_cache(**arguments)
    (cache_dir / "condition.safetensors").write_bytes(b"corrupted")

    with pytest.raises(CacheValidationError, match="Existing cache entry is invalid"):
        create_condition_cache(**arguments)

    rebuilt = create_condition_cache(**arguments, force=True)
    assert rebuilt == cache_dir
    validate_cache_entry(rebuilt)


def test_manifest_key_mismatch_is_rejected(tmp_path: Path) -> None:
    cache_dir = create_condition_cache(**_official_cache_args(tmp_path))
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cache_key"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CacheValidationError, match="canonical inputs"):
        validate_cache_entry(cache_dir)


def test_native_cache_records_experimental_provenance(tmp_path: Path) -> None:
    native_archive = tmp_path / "qwen_layer50_xllm.pt"
    official_archive = tmp_path / "qwen_layer50_reference.pt"
    _save_archive(native_archive, _hidden(batched=False))
    _save_archive(official_archive, torch.zeros(1), torch.tensor([0, 1, 0], dtype=torch.int64))
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("native prompt", encoding="utf-8")
    checkpoint_manifest = tmp_path / "checkpoint.json"
    checkpoint_manifest.write_text("{}\n", encoding="utf-8")

    cache_dir = create_condition_cache(
        backend="xllm_native",
        source_archive=native_archive,
        official_metadata_archive=official_archive,
        cache_root=tmp_path / "cache",
        checkpoint_identity={"revision": "test"},
        checkpoint_manifest=checkpoint_manifest,
        prompt_path=prompt_path,
    )
    manifest = validate_cache_entry(cache_dir)

    assert manifest["backend_approval_status"] == "experimental/native-gate-pending"
    assert [item["role"] for item in manifest["source_archives"]] == [
        "xllm_native_condition",
        "official_metadata_golden",
    ]
    assert load_file(str(cache_dir / "condition.safetensors"))["text_token_tags"].tolist() == [0, 1, 0]


def test_cli_defaults_to_official_and_prints_cache_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_archive = tmp_path / "qwen_layer50_reference.pt"
    hidden = _hidden()
    _save_archive(source_archive, hidden, torch.tensor([0, 1, 0], dtype=torch.int64))
    cache_root = tmp_path / "cache"
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("prompt", encoding="utf-8")
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(b"image")
    checkpoint_manifest = tmp_path / "checkpoint.json"
    checkpoint_manifest.write_text("{}\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "H3_QWEN_LAYER50_REFERENCE_PASS",
                "tensor_archive_sha256": condition_cache.file_sha256(source_archive),
                "prompt_sha256": condition_cache.file_sha256(prompt_path),
                "image_sha256": condition_cache.file_sha256(reference_path),
                "lm_head_executed": False,
                "layer_49_output_pre_final_norm": {"shape": list(hidden.shape)},
            }
        ),
        encoding="utf-8",
    )

    result = condition_cache.main(
        [
            "--source-archive",
            str(source_archive),
            "--cache-root",
            str(cache_root),
            "--checkpoint-identity",
            "revision=test",
            "--checkpoint-manifest",
            str(checkpoint_manifest),
            "--prompt-path",
            str(prompt_path),
            "--reference",
            f"image={reference_path}",
            "--official-reference-summary",
            str(summary_path),
        ]
    )

    assert result == 0
    printed_path = Path(capsys.readouterr().out.strip())
    assert printed_path.parent == cache_root.resolve()
    validate_cache_entry(printed_path)
