import argparse
import base64
import json
import os
import signal
from pathlib import Path

import pytest

import tools.minimax_h3_public_runtime_gate as runtime_gate


def test_request_bytes_sets_request_id_without_changing_contract() -> None:
    template = {"model": "MiniMax-H3", "request_id": "", "input": {}, "parameters": {}}

    payload = json.loads(runtime_gate._request_bytes(template, "request-a"))

    assert payload["request_id"] == "request-a"
    assert payload["model"] == "MiniMax-H3"


def test_gate_interrupt_handler_reports_signal() -> None:
    with pytest.raises(runtime_gate.GateInterrupted, match="SIGTERM"):
        runtime_gate._raise_gate_interrupted(signal.SIGTERM, None)


def test_success_response_writes_and_hashes_mp4(tmp_path: Path, monkeypatch) -> None:
    mp4 = b"\x00\x00\x00\x18ftyp" + b"x" * 2048
    result = {
        "video": base64.b64encode(mp4).decode("ascii"),
        "width": 1344,
        "height": 768,
        "seed": 42,
        "num_frames": 124,
        "fps": 24.0,
        "mime_type": "video/mp4",
        "container": "mp4",
        "audio_sample_rate": 32000,
        "audio_channels": 2,
    }
    probed: list[Path] = []

    def record_probe(probe_binary: Path, mp4_path: Path, log_path: Path) -> None:
        del probe_binary, log_path
        probed.append(mp4_path)

    monkeypatch.setattr(runtime_gate, "_probe_mp4", record_probe)

    metadata = runtime_gate._consume_success(
        json.dumps(
            {
                "id": "request-a",
                "object": "list",
                "model": "MiniMax-H3",
                "output": {"results": [result]},
            }
        ).encode(),
        tmp_path,
        "success-a",
        "request-a",
        "MiniMax-H3",
        42,
        Path("probe"),
    )

    assert metadata["sha256"] == runtime_gate._sha256_bytes(mp4)
    assert (tmp_path / "success-a.mp4").read_bytes() == mp4
    assert probed == [tmp_path / "success-a.mp4"]


def test_success_response_rejects_wrong_audio_metadata(tmp_path: Path) -> None:
    result = {
        "video": base64.b64encode(b"\x00\x00\x00\x18ftyp" + b"x" * 2048).decode("ascii"),
        "width": 1344,
        "height": 768,
        "seed": 42,
        "num_frames": 124,
        "fps": 24.0,
        "mime_type": "video/mp4",
        "container": "mp4",
        "audio_sample_rate": 16000,
        "audio_channels": 2,
    }

    with pytest.raises(ValueError, match="audio_sample_rate mismatch"):
        runtime_gate._consume_success(
            json.dumps(
                {
                    "id": "request-a",
                    "object": "list",
                    "model": "MiniMax-H3",
                    "output": {"results": [result]},
                }
            ).encode(),
            tmp_path,
            "bad-audio",
            "request-a",
            "MiniMax-H3",
            42,
            Path("probe"),
        )


def test_success_response_rejects_wrong_request_id(tmp_path: Path) -> None:
    response = {
        "id": "stale-request",
        "object": "list",
        "model": "MiniMax-H3",
        "output": {"results": []},
    }

    with pytest.raises(ValueError, match="response id mismatch"):
        runtime_gate._consume_success(
            json.dumps(response).encode(),
            tmp_path,
            "stale",
            "request-a",
            "MiniMax-H3",
            42,
            Path("probe"),
        )


def test_cleanup_shm_namespace_removes_only_matching_objects(tmp_path: Path, monkeypatch) -> None:
    owned = tmp_path / "xllm_29842_rank_0_output"
    unrelated = tmp_path / "xllm_29843_rank_0_output"
    prefix_collision = tmp_path / "xllm_298420_rank_0_output"
    owned.write_bytes(b"owned")
    unrelated.write_bytes(b"unrelated")
    prefix_collision.write_bytes(b"prefix")
    original_path = runtime_gate.Path

    def mapped_path(value: str) -> Path:
        return tmp_path if value == "/dev/shm" else original_path(value)

    monkeypatch.setattr(runtime_gate, "Path", mapped_path)

    removed, remaining = runtime_gate._cleanup_shm_namespace(29842)

    assert removed == [str(owned)]
    assert remaining == []
    assert unrelated.read_bytes() == b"unrelated"
    assert prefix_collision.read_bytes() == b"prefix"


def test_server_command_must_use_declared_master_port() -> None:
    runtime_gate._validate_server_master_port(["xllm", "--master_node_addr=127.0.0.1:29842"], 29842)

    with pytest.raises(ValueError, match="does not match"):
        runtime_gate._validate_server_master_port(["xllm", "--master_node_addr=127.0.0.1:29843"], 29842)


def test_gate_modes_are_mutually_exclusive() -> None:
    runtime_gate._validate_gate_mode(False, False, 0)
    runtime_gate._validate_gate_mode(False, True, 0)
    runtime_gate._validate_gate_mode(True, False, 0)
    runtime_gate._validate_gate_mode(False, False, 4)

    with pytest.raises(ValueError, match="mutually exclusive"):
        runtime_gate._validate_gate_mode(True, True, 0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        runtime_gate._validate_gate_mode(False, True, 4)


def _gate_mode_arguments(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "shutdown_smoke_only": False,
        "success_only": False,
        "steady_success_count": 0,
        "failure_rank": None,
        "failure_batch_index": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_failure_injection_is_scoped_to_the_lifecycle_mode() -> None:
    lifecycle = _gate_mode_arguments()

    assert runtime_gate._resolve_failure_injection(lifecycle) == {
        "XLLM_TEST_DIT_FAILURE_INJECTION": "1",
        "XLLM_TEST_DIT_FAIL_RANK": "7",
        "XLLM_TEST_DIT_FAIL_BATCH_INDEX": "3",
    }
    assert (lifecycle.failure_rank, lifecycle.failure_batch_index) == (7, 3)
    assert runtime_gate._resolve_failure_injection(_gate_mode_arguments(success_only=True)) == {}
    assert runtime_gate._resolve_failure_injection(_gate_mode_arguments(steady_success_count=3)) == {}
    assert runtime_gate._resolve_failure_injection(_gate_mode_arguments(steady_success_count=4)) == {}
    assert runtime_gate._resolve_failure_injection(_gate_mode_arguments(shutdown_smoke_only=True)) == {}

    tuned = _gate_mode_arguments(failure_rank=5, failure_batch_index=2)
    assert runtime_gate._resolve_failure_injection(tuned) == {
        "XLLM_TEST_DIT_FAILURE_INJECTION": "1",
        "XLLM_TEST_DIT_FAIL_RANK": "5",
        "XLLM_TEST_DIT_FAIL_BATCH_INDEX": "2",
    }


def test_failure_injection_requests_outside_the_lifecycle_mode_fail_loudly() -> None:
    for mode in (
        {"success_only": True},
        {"steady_success_count": 4},
        {"shutdown_smoke_only": True},
    ):
        with pytest.raises(ValueError, match="honoured only by the default lifecycle mode"):
            runtime_gate._resolve_failure_injection(_gate_mode_arguments(failure_rank=7, **mode))
        with pytest.raises(ValueError, match="honoured only by the default lifecycle mode"):
            runtime_gate._resolve_failure_injection(_gate_mode_arguments(failure_batch_index=3, **mode))


def test_failure_injection_rejects_invalid_rank_and_batch_index() -> None:
    with pytest.raises(ValueError, match="failure rank must be nonnegative"):
        runtime_gate._resolve_failure_injection(_gate_mode_arguments(failure_rank=-1))
    with pytest.raises(ValueError, match="failure batch index must be positive"):
        runtime_gate._resolve_failure_injection(_gate_mode_arguments(failure_batch_index=0))


def test_export_failure_injection_clears_variables_the_mode_did_not_request(monkeypatch) -> None:
    for name in runtime_gate.FAILURE_INJECTION_ENVIRONMENT:
        monkeypatch.setenv(name, "stale")

    discarded = runtime_gate._export_failure_injection({})

    assert discarded == dict.fromkeys(runtime_gate.FAILURE_INJECTION_ENVIRONMENT, "stale")
    assert [name for name in runtime_gate.FAILURE_INJECTION_ENVIRONMENT if name in os.environ] == []


def test_export_failure_injection_publishes_the_lifecycle_contract(monkeypatch) -> None:
    injection = {
        "XLLM_TEST_DIT_FAILURE_INJECTION": "1",
        "XLLM_TEST_DIT_FAIL_RANK": "7",
        "XLLM_TEST_DIT_FAIL_BATCH_INDEX": "3",
    }
    for name in injection:
        monkeypatch.delenv(name, raising=False)

    assert runtime_gate._export_failure_injection(injection) == {}
    assert {name: os.environ[name] for name in injection} == injection


def test_mp4_probe_rejects_zero_selected_tests(tmp_path: Path, monkeypatch) -> None:
    result = runtime_gate.subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="[==========] Running 0 tests from 0 test suites.\n[  PASSED  ] 0 tests.\n",
    )
    monkeypatch.setattr(runtime_gate.subprocess, "run", lambda *args, **kwargs: result)

    with pytest.raises(RuntimeError, match="MP4 stream probe failed"):
        runtime_gate._probe_mp4(Path("probe"), Path("video.mp4"), tmp_path / "probe.log")


def test_collect_stage_timings_groups_requests_and_takes_stage_maxima(tmp_path: Path) -> None:
    (tmp_path / "node_0.log").write_text(
        "I runtime MINIMAX_H3_STAGE_TIMING rank=0 request=1 stage=denoise "
        "elapsed_ms=170001.25 cumulative_ms=180002.5\n"
        "I runtime MINIMAX_H3_STAGE_TIMING rank=0 request=1 stage=request_total "
        "elapsed_ms=281749 cumulative_ms=281749\n",
        encoding="utf-8",
    )
    (tmp_path / "node_1.log").write_text(
        "I runtime MINIMAX_H3_STAGE_TIMING rank=1 request=1 stage=denoise "
        "elapsed_ms=170101.5 cumulative_ms=180103\n"
        "unrelated log line\n",
        encoding="utf-8",
    )

    timings = runtime_gate._collect_stage_timings(tmp_path)

    request = timings["requests"]["1"]
    assert len(timings["records"]) == 3
    assert request["by_rank"]["0"]["request_total"]["elapsed_ms"] == 281749.0
    assert request["max_elapsed_ms_by_stage"]["denoise"] == 170101.5


def test_collect_stage_timings_rejects_duplicate_rank_stage(tmp_path: Path) -> None:
    line = "MINIMAX_H3_STAGE_TIMING rank=0 request=1 stage=denoise elapsed_ms=1 cumulative_ms=2\n"
    (tmp_path / "node_0.log").write_text(line + line, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate MiniMax-H3 stage timing"):
        runtime_gate._collect_stage_timings(tmp_path)


def _complete_stage_timings() -> dict[str, object]:
    by_rank = {}
    maximums = {}
    for rank in range(runtime_gate.EXPECTED_WORLD_SIZE):
        stage_names = list(runtime_gate.COMMON_TIMING_STAGES)
        if rank == 0:
            stage_names += list(runtime_gate.RANK_ZERO_TIMING_STAGES)
        stages = {}
        cumulative = 0.0
        for stage in stage_names:
            elapsed = 100.0 if stage == "request_total" else 1.0
            cumulative = elapsed if stage == "request_total" else cumulative + elapsed
            stages[stage] = {"elapsed_ms": elapsed, "cumulative_ms": cumulative}
            maximums[stage] = max(maximums.get(stage, 0.0), elapsed)
        by_rank[str(rank)] = stages
    return {"records": [], "requests": {"1": {"by_rank": by_rank, "max_elapsed_ms_by_stage": maximums}}}


def test_validate_complete_stage_timings_requires_all_rank_stages() -> None:
    timings = _complete_stage_timings()
    runtime_gate._validate_complete_stage_timings(timings, [1])

    del timings["requests"]["1"]["by_rank"]["7"]["denoise"]
    with pytest.raises(ValueError, match="rank=7.*denoise"):
        runtime_gate._validate_complete_stage_timings(timings, [1])
