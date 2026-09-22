#!/usr/bin/env python3
"""Run the MiniMax-H3 public service lifecycle gate."""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import http.client
import json
import math
import os
import re
import signal
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch
from safetensors.torch import load_file

if __package__:
    from tools.minimax_h3_condition_cache import image_pixels_sha256
else:
    from minimax_h3_condition_cache import image_pixels_sha256

EXPECTED_CONDITION_SHAPE = (11350, 5120)
EXPECTED_TAGS_SHAPE = (11350,)
EXPECTED_WIDTH = 1344
EXPECTED_HEIGHT = 768
EXPECTED_FRAMES = 124
EXPECTED_FPS = 24.0
EXPECTED_SAMPLE_RATE = 32000
EXPECTED_AUDIO_CHANNELS = 2
EXPECTED_WORLD_SIZE = 16
DEFAULT_FAILURE_RANK = 7
DEFAULT_FAILURE_BATCH_INDEX = 3
FAILURE_INJECTION_ENVIRONMENT = (
    "XLLM_TEST_DIT_FAILURE_INJECTION",
    "XLLM_TEST_DIT_FAIL_RANK",
    "XLLM_TEST_DIT_FAIL_BATCH_INDEX",
)
COMMON_TIMING_STAGES = (
    "input_preparation",
    "reference_vae_load_encode",
    "latent_packing",
    "denoiser_load",
    "denoise",
    "final_unpack",
    "request_total",
)
RANK_ZERO_TIMING_STAGES = (
    "video_vae_load",
    "video_vae_decode_transfer",
    "audio_vae_load",
    "audio_vae_decode_transfer",
    "media_encode",
)
STAGE_TIMING_PATTERN = re.compile(
    r"MINIMAX_H3_STAGE_TIMING "
    r"rank=(?P<rank>\d+) request=(?P<request>\d+) "
    r"stage=(?P<stage>[a-z0-9_]+) "
    r"elapsed_ms=(?P<elapsed>[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?) "
    r"cumulative_ms=(?P<cumulative>[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)"
)


class GateInterrupted(RuntimeError):
    pass


class ShutdownSmokeComplete(RuntimeError):
    pass


class SuccessOnlyComplete(RuntimeError):
    pass


def _raise_gate_interrupted(signum: int, _frame: object) -> None:
    raise GateInterrupted(f"public runtime gate received signal {signal.Signals(signum).name}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().to("cpu").contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _collect_stage_timings(server_log_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for log_path in sorted(server_log_dir.glob("node_*.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for match in STAGE_TIMING_PATTERN.finditer(text):
            rank = int(match.group("rank"))
            request = int(match.group("request"))
            stage = match.group("stage")
            elapsed_ms = float(match.group("elapsed"))
            cumulative_ms = float(match.group("cumulative"))
            if not math.isfinite(elapsed_ms) or not math.isfinite(cumulative_ms):
                raise ValueError(f"non-finite MiniMax-H3 stage timing in {log_path}")
            key = (request, rank, stage)
            if key in seen:
                raise ValueError(f"duplicate MiniMax-H3 stage timing for request={request}, rank={rank}, stage={stage}")
            seen.add(key)
            records.append(
                {
                    "request": request,
                    "rank": rank,
                    "stage": stage,
                    "elapsed_ms": elapsed_ms,
                    "cumulative_ms": cumulative_ms,
                    "log": str(log_path),
                }
            )

    requests: dict[str, Any] = {}
    for record in sorted(records, key=lambda value: (value["request"], value["rank"], value["cumulative_ms"])):
        request = requests.setdefault(
            str(record["request"]),
            {"by_rank": {}, "max_elapsed_ms_by_stage": {}},
        )
        rank = request["by_rank"].setdefault(str(record["rank"]), {})
        rank[record["stage"]] = {
            "elapsed_ms": record["elapsed_ms"],
            "cumulative_ms": record["cumulative_ms"],
        }
        maximums = request["max_elapsed_ms_by_stage"]
        maximums[record["stage"]] = max(maximums.get(record["stage"], 0.0), record["elapsed_ms"])
    return {"records": records, "requests": requests}


def _validate_complete_stage_timings(timings: dict[str, Any], expected_requests: list[int]) -> None:
    requests = timings.get("requests")
    if not isinstance(requests, dict):
        raise ValueError("MiniMax-H3 stage timing summary has no requests")
    expected_ranks = {str(rank) for rank in range(EXPECTED_WORLD_SIZE)}
    for request_sequence in expected_requests:
        request = requests.get(str(request_sequence))
        if not isinstance(request, dict) or not isinstance(request.get("by_rank"), dict):
            raise ValueError(f"missing MiniMax-H3 stage timings for request={request_sequence}")
        by_rank = request["by_rank"]
        if set(by_rank) != expected_ranks:
            missing = sorted(expected_ranks - set(by_rank), key=int)
            extra = sorted(set(by_rank) - expected_ranks)
            raise ValueError(
                f"MiniMax-H3 request={request_sequence} timing rank inventory mismatch: "
                f"missing={missing}, extra={extra}"
            )
        for rank in range(EXPECTED_WORLD_SIZE):
            stages = by_rank[str(rank)]
            required = set(COMMON_TIMING_STAGES)
            if rank == 0:
                required.update(RANK_ZERO_TIMING_STAGES)
            missing = sorted(required - set(stages))
            if missing:
                raise ValueError(
                    f"MiniMax-H3 request={request_sequence}, rank={rank} is missing stage timings: {missing}"
                )
            total = stages["request_total"]
            if total["elapsed_ms"] != total["cumulative_ms"]:
                raise ValueError(
                    f"MiniMax-H3 request={request_sequence}, rank={rank} has an inconsistent request_total"
                )


class Evidence:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.events: list[dict[str, Any]] = []
        self.event_path = output_dir / "events.jsonl"

    def record(self, phase: str, **values: Any) -> None:
        event = {"phase": phase, "monotonic_seconds": time.monotonic(), **values}
        self.events.append(event)
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, allow_nan=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _load_request_template(cache_dir: Path, image_path: Path, model: str, seed: int) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    condition_path = cache_dir / "condition.safetensors"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 3:
        raise ValueError("MiniMax-H3 public gate requires condition cache schema v3")
    runtime_manifest = manifest.get("runtime_bundle_manifest")
    if not isinstance(runtime_manifest, dict):
        raise ValueError("condition cache has no runtime bundle manifest")
    if runtime_manifest.get("schema") != "xllm.minimax_h3.text_conditioning/v1":
        raise ValueError("condition cache runtime schema mismatch")
    if runtime_manifest.get("source_backend") != "official_hf":
        raise ValueError("public gate requires the attested official_hf condition")
    if runtime_manifest.get("token_count") != EXPECTED_CONDITION_SHAPE[0]:
        raise ValueError("condition cache token count is not the production contract")
    if runtime_manifest.get("condition_cache_key") != manifest.get("cache_key"):
        raise ValueError("runtime condition cache key mismatch")
    if cache_dir.name != manifest.get("cache_key"):
        raise ValueError("condition cache directory does not match its key")
    expected_artifact_digest = manifest.get("artifacts", {}).get("condition.safetensors", {}).get("sha256")
    if _sha256_bytes(condition_path.read_bytes()) != expected_artifact_digest:
        raise ValueError("condition.safetensors artifact digest mismatch")

    tensors = load_file(str(condition_path), device="cpu")
    if set(tensors) != {"prompt_embeds", "text_token_tags"}:
        raise ValueError("condition cache tensor inventory mismatch")
    prompt_embeds = tensors["prompt_embeds"].contiguous()
    text_token_tags = tensors["text_token_tags"].contiguous()
    if tuple(prompt_embeds.shape) != EXPECTED_CONDITION_SHAPE or prompt_embeds.dtype != torch.bfloat16:
        raise ValueError("prompt_embeds does not match [11350,5120] BF16")
    if tuple(text_token_tags.shape) != EXPECTED_TAGS_SHAPE or text_token_tags.dtype != torch.int64:
        raise ValueError("text_token_tags does not match [11350] int64")
    if not bool(torch.logical_or(text_token_tags == 0, text_token_tags == 1).all().item()):
        raise ValueError("text_token_tags contains values outside {0,1}")

    prompt_bytes = _tensor_bytes(prompt_embeds)
    tag_bytes = _tensor_bytes(text_token_tags)
    if _sha256_bytes(prompt_bytes) != runtime_manifest.get("hidden_digest"):
        raise ValueError("prompt_embeds digest mismatch")
    if _sha256_bytes(tag_bytes) != runtime_manifest.get("token_tags_digest"):
        raise ValueError("text_token_tags digest mismatch")
    if image_pixels_sha256(image_path) != runtime_manifest.get("reference_pixels_digest"):
        raise ValueError("reference image pixels do not match the condition bundle")

    return {
        "model": model,
        "request_id": "",
        "input": {
            "prompt": "",
            "prompt_embed": {
                "name": "prompt_embed",
                "datatype": "BF16",
                "shape": list(EXPECTED_CONDITION_SHAPE),
                "contents": {"bytes_contents": base64.b64encode(prompt_bytes).decode("ascii")},
            },
            "image": base64.b64encode(image_path.read_bytes()).decode("ascii"),
            "text_token_tags": {
                "name": "text_token_tags",
                "datatype": "INT64",
                "shape": list(EXPECTED_TAGS_SHAPE),
                "contents": {"int64_contents": text_token_tags.tolist()},
            },
            "condition_schema": runtime_manifest["schema"],
            "condition_source_backend": runtime_manifest["source_backend"],
            "condition_manifest_json": json.dumps(
                runtime_manifest, allow_nan=False, separators=(",", ":"), sort_keys=True
            ),
        },
        "parameters": {
            "size": f"{EXPECTED_WIDTH}*{EXPECTED_HEIGHT}",
            "num_inference_steps": 50,
            "num_videos_per_prompt": 1,
            "seed": seed,
            "num_frames": EXPECTED_FRAMES,
            "fps": EXPECTED_FPS,
        },
    }


def _request_bytes(template: dict[str, Any], request_id: str) -> bytes:
    template["request_id"] = request_id
    return json.dumps(template, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _health_url(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    return f"{parsed.scheme}://{parsed.netloc}/health"


def _wait_for_health(endpoint: str, server: subprocess.Popen[bytes], timeout: float) -> None:
    health_url = _health_url(endpoint)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = server.poll()
        if return_code is not None:
            raise RuntimeError(f"server launcher exited before readiness with code {return_code}")
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy within {timeout} seconds")


def _post(endpoint: str, payload: bytes, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _disconnect_after_upload(endpoint: str, payload: bytes, delay: float) -> tuple[str, int]:
    parsed = urlsplit(endpoint)
    if parsed.scheme != "http" or not parsed.hostname:
        raise ValueError("disconnect gate currently requires an http:// endpoint")
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=60)
    try:
        connection.putrequest("POST", path)
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(len(payload)))
        connection.endheaders(payload)
        if connection.sock is None:
            raise RuntimeError("disconnect gate has no connected client socket")
        client_host, client_port = connection.sock.getsockname()
        time.sleep(delay)
        connection.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        return str(client_host), int(client_port)
    finally:
        connection.close()


def _wait_for_log(
    path: Path,
    needles: str | tuple[str, ...],
    server: subprocess.Popen[bytes],
    timeout: float,
    offset: int = 0,
) -> str:
    expected = (needles,) if isinstance(needles, str) else needles
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = server.poll()
        if return_code is not None:
            raise RuntimeError(f"server launcher exited while waiting for log evidence with code {return_code}")
        if path.is_file():
            text = path.read_bytes()[offset:].decode("utf-8", errors="replace")
            for needle in expected:
                if needle in text:
                    return needle
        time.sleep(2)
    raise TimeoutError(f"did not find any of {expected!r} in {path} within {timeout} seconds")


def _wait_for_log_line(
    path: Path,
    required_fields: tuple[str, ...],
    server: subprocess.Popen[bytes],
    timeout: float,
    offset: int = 0,
) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = server.poll()
        if return_code is not None:
            raise RuntimeError(f"server launcher exited while waiting for log evidence with code {return_code}")
        if path.is_file():
            text = path.read_bytes()[offset:].decode("utf-8", errors="replace")
            for line in text.splitlines():
                if all(field in line for field in required_fields):
                    return line
        time.sleep(2)
    raise TimeoutError(f"did not find one line containing {required_fields!r} in {path} within {timeout} seconds")


def _probe_mp4(probe_binary: Path, mp4_path: Path, log_path: Path) -> None:
    environment = os.environ.copy()
    environment["MINIMAX_H3_EXTERNAL_MP4"] = str(mp4_path)
    result = subprocess.run(
        [str(probe_binary), "--gtest_filter=MMCodecTest.ValidatesExternalProductionMp4WhenRequested"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        check=False,
        text=True,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    if (
        result.returncode != 0
        or "[==========] Running 1 test from 1 test suite." not in result.stdout
        or "[  PASSED  ] 1 test." not in result.stdout
    ):
        raise RuntimeError(f"MP4 stream probe failed for {mp4_path} with code {result.returncode}")


def _consume_success(
    body: bytes,
    output_dir: Path,
    phase: str,
    request_id: str,
    model: str,
    seed: int,
    probe_binary: Path,
) -> dict[str, Any]:
    response = json.loads(body)
    expected_envelope = {"id": request_id, "object": "list", "model": model}
    for key, value in expected_envelope.items():
        if response.get(key) != value:
            raise ValueError(f"{phase} response {key} mismatch: {response.get(key)!r} != {value!r}")
    results = response.get("output", {}).get("results", [])
    if len(results) != 1:
        raise ValueError(f"{phase} response must contain exactly one result")
    result = results[0]
    expected = {
        "width": EXPECTED_WIDTH,
        "height": EXPECTED_HEIGHT,
        "seed": seed,
        "num_frames": EXPECTED_FRAMES,
        "fps": EXPECTED_FPS,
        "mime_type": "video/mp4",
        "container": "mp4",
        "audio_sample_rate": EXPECTED_SAMPLE_RATE,
        "audio_channels": EXPECTED_AUDIO_CHANNELS,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"{phase} response {key} mismatch: {result.get(key)!r} != {value!r}")
    video = result.get("video")
    if not isinstance(video, str):
        raise ValueError(f"{phase} response has no base64 video")
    mp4 = base64.b64decode(video, validate=True)
    if len(mp4) <= 1024 or b"ftyp" not in mp4[:64]:
        raise ValueError(f"{phase} response is not a non-empty MP4")
    mp4_path = output_dir / f"{phase}.mp4"
    mp4_path.write_bytes(mp4)
    _probe_mp4(probe_binary, mp4_path, output_dir / f"{phase}-probe.log")
    metadata = {key: result[key] for key in expected}
    metadata.update({"bytes": len(mp4), "sha256": _sha256_bytes(mp4), "path": str(mp4_path)})
    _write_json(output_dir / f"{phase}.json", metadata)
    return metadata


def _run_success(
    endpoint: str,
    template: dict[str, Any],
    output_dir: Path,
    phase: str,
    seed: int,
    timeout: float,
    probe_binary: Path,
    evidence: Evidence,
) -> dict[str, Any]:
    request_id = f"h3-gate-{phase}"
    payload = _request_bytes(template, request_id)
    started = time.monotonic()
    status, body = _post(endpoint, payload, timeout)
    duration = time.monotonic() - started
    if status != 200:
        raise RuntimeError(f"{phase} returned HTTP {status}: {body[:2048]!r}")
    metadata = _consume_success(body, output_dir, phase, request_id, template["model"], seed, probe_binary)
    evidence.record(phase, status="PASS", duration_seconds=duration, **metadata)
    return metadata


def _become_child_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _descendant_processes(root_pid: int) -> dict[int, str]:
    process_table: dict[int, tuple[int, str]] = {}
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            value = (process_dir / "stat").read_text(encoding="utf-8")
            fields = value[value.rfind(")") + 2 :].split()
            process_table[int(process_dir.name)] = (int(fields[1]), fields[0])
        except (FileNotFoundError, IndexError, ValueError):
            continue
    descendants = {}
    parents = {root_pid}
    while parents:
        children = {
            pid: state
            for pid, (parent_pid, state) in process_table.items()
            if parent_pid in parents and pid not in descendants
        }
        descendants.update(children)
        parents = set(children)
    return descendants


def _reap_exited_children() -> list[int]:
    reaped = []
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid <= 0:
            break
        reaped.append(pid)
    return reaped


def _terminate_server(server: subprocess.Popen[bytes], timeout: float) -> tuple[int, bool, list[int]]:
    gate_pid = os.getpid()
    hard_deadline = time.monotonic() + timeout
    kill_reserve = min(5.0, timeout / 4.0)
    if server.poll() is None:
        server.send_signal(signal.SIGTERM)
    try:
        return_code = server.wait(timeout=max(0.0, timeout - kill_reserve))
    except subprocess.TimeoutExpired:
        server.kill()
        try:
            return_code = server.wait(timeout=max(0.0, hard_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            return_code = -signal.SIGKILL
    reaped = _reap_exited_children()
    remaining = lambda: _descendant_processes(gate_pid)
    descendant_kill_at = hard_deadline - min(2.0, timeout / 8.0)
    while remaining() and time.monotonic() < descendant_kill_at:
        reaped.extend(_reap_exited_children())
        time.sleep(0.1)
    for pid, state in remaining().items():
        if state == "Z":
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    while remaining() and time.monotonic() < hard_deadline:
        reaped.extend(_reap_exited_children())
        time.sleep(0.1)
    reaped.extend(_reap_exited_children())
    return return_code, not remaining(), sorted(set(reaped))


def _shm_namespace_paths(master_port: int) -> list[Path]:
    return sorted(Path("/dev/shm").glob(f"xllm_{master_port}_*"))


def _cleanup_shm_namespace(master_port: int) -> tuple[list[str], list[str]]:
    matches = _shm_namespace_paths(master_port)
    removed = []
    for path in matches:
        path.unlink()
        removed.append(str(path))
    remaining = [str(path) for path in _shm_namespace_paths(master_port)]
    return removed, remaining


def _validate_server_master_port(command: list[str], master_port: int) -> None:
    values = [argument.split("=", 1)[1] for argument in command if argument.startswith("--master_node_addr=")]
    if len(values) != 1 or not values[0].endswith(f":{master_port}"):
        raise ValueError("server command master_node_addr does not match --master-port")


def _validate_gate_mode(shutdown_smoke_only: bool, success_only: bool, steady_success_count: int) -> None:
    if sum((shutdown_smoke_only, success_only, steady_success_count > 0)) > 1:
        raise ValueError("shutdown smoke, success-only, and steady-success modes are mutually exclusive")


def _resolve_failure_injection(args: argparse.Namespace) -> dict[str, str]:
    """Return the DiT fault-injection environment the selected mode may export.

    Only the default lifecycle mode issues a request that must fail, so it is the only mode that may inject, and it
    fills in the lifecycle defaults on ``args``. Injection cannot be scoped to one phase of a longer run: the server
    inherits the environment at launch and keeps it for its whole lifetime, and the consumer
    (``WorkerService::should_inject_dit_failure``) matches ``XLLM_TEST_DIT_FAIL_BATCH_INDEX`` against a per-worker
    count of every DiT request served since startup, not against a position inside one batch. A mode that requires
    only successful requests therefore exports nothing, and an injection request it could never honour is an error.
    """
    lifecycle_mode = not (args.shutdown_smoke_only or args.success_only or args.steady_success_count > 0)
    if not lifecycle_mode:
        if args.failure_rank is not None or args.failure_batch_index is not None:
            raise ValueError(
                "--failure-rank and --failure-batch-index are honoured only by the default lifecycle mode, "
                "which is the only mode that issues a request expected to fail"
            )
        return {}
    args.failure_rank = DEFAULT_FAILURE_RANK if args.failure_rank is None else args.failure_rank
    args.failure_batch_index = (
        DEFAULT_FAILURE_BATCH_INDEX if args.failure_batch_index is None else args.failure_batch_index
    )
    if args.failure_rank < 0 or args.failure_batch_index <= 0:
        raise ValueError("failure rank must be nonnegative and failure batch index must be positive")
    return {
        "XLLM_TEST_DIT_FAILURE_INJECTION": "1",
        "XLLM_TEST_DIT_FAIL_RANK": str(args.failure_rank),
        "XLLM_TEST_DIT_FAIL_BATCH_INDEX": str(args.failure_batch_index),
    }


def _export_failure_injection(injection: dict[str, str]) -> dict[str, str]:
    """Publish the selected mode's injection contract, dropping variables it did not request."""
    unrequested = [name for name in FAILURE_INJECTION_ENVIRONMENT if name not in injection]
    discarded = {name: os.environ.pop(name) for name in unrequested if name in os.environ}
    os.environ.update(injection)
    return discarded


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-log-dir", type=Path, required=True)
    parser.add_argument("--media-probe-binary", type=Path, required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--health-timeout", type=float, default=7200.0)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
    parser.add_argument("--disconnect-delay", type=float, default=5.0)
    parser.add_argument("--disconnect-timeout", type=float, default=900.0)
    parser.add_argument("--post-disconnect-settle", type=float, default=5.0)
    parser.add_argument("--shutdown-timeout", type=float, default=180.0)
    parser.add_argument(
        "--failure-rank",
        type=int,
        default=None,
        help=(
            "rank whose DiT worker fails during the lifecycle failure phase "
            f"(default: {DEFAULT_FAILURE_RANK}; rejected outside the default lifecycle mode)"
        ),
    )
    parser.add_argument(
        "--failure-batch-index",
        type=int,
        default=None,
        help=(
            "1-based count of DiT requests the failing worker has served since the server started, not a position "
            f"within one batch (default: {DEFAULT_FAILURE_BATCH_INDEX}, the lifecycle failure-c request; rejected "
            "outside the default lifecycle mode)"
        ),
    )
    parser.add_argument("--shutdown-smoke-only", action="store_true")
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument(
        "--steady-success-count",
        type=int,
        default=0,
        help=(
            "TOTAL successful requests of the steady-state mode, the single warmup request included: 4 means "
            "1 warmup + 3 measured, which is the project's steady-state definition; 0 disables the mode, minimum 2"
        ),
    )
    parser.add_argument("--shutdown-smoke-hold-seconds", type=float, default=0.0)
    parser.add_argument("server_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.server_command and args.server_command[0] == "--":
        args.server_command = args.server_command[1:]
    if not args.server_command:
        parser.error("a server command is required after --")
    return args


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    server_log_dir = args.server_log_dir.resolve()
    if server_log_dir.exists() and any(server_log_dir.iterdir()):
        raise ValueError(f"server log directory must be absent or empty: {server_log_dir}")
    server_log_dir.mkdir(parents=True, exist_ok=True)
    args.server_log_dir = server_log_dir
    evidence = Evidence(output_dir)
    if os.getpgrp() != os.getpid():
        raise RuntimeError("public runtime gate must run as a process-group leader under npu_lease.py")
    _become_child_subreaper()
    signal.signal(signal.SIGINT, _raise_gate_interrupted)
    signal.signal(signal.SIGTERM, _raise_gate_interrupted)
    template = _load_request_template(args.cache_dir.resolve(), args.image.resolve(), args.model, args.seed)
    _validate_server_master_port(args.server_command, args.master_port)
    _validate_gate_mode(args.shutdown_smoke_only, args.success_only, args.steady_success_count)
    if args.steady_success_count not in (0,) and args.steady_success_count < 2:
        raise ValueError("steady-success mode requires at least one warmup and one measured request")
    if args.shutdown_smoke_hold_seconds < 0:
        raise ValueError("shutdown smoke hold seconds must be nonnegative")
    failure_injection = _resolve_failure_injection(args)
    discarded_failure_injection = _export_failure_injection(failure_injection)
    preexisting_shm = [str(path) for path in _shm_namespace_paths(args.master_port)]
    if preexisting_shm:
        raise ValueError(f"shared-memory namespace is not clean before launch: {preexisting_shm}")
    evidence.record(
        "fixture",
        status="PASS",
        cache_key=args.cache_dir.resolve().name,
        payload_bytes=len(_request_bytes(template, "h3-gate-size-check")),
        failure_rank=args.failure_rank,
        failure_batch_index=args.failure_batch_index,
        failure_injection=bool(failure_injection),
        discarded_ambient_failure_injection=discarded_failure_injection,
    )

    launcher_log_path = output_dir / "launcher.log"
    summary: dict[str, Any] = {"status": "FAIL", "events": evidence.events}
    server_returncode: int | None = None
    server_group_empty = False
    reaped_server_pids: list[int] = []
    expected_timing_requests: list[int] = []
    next_timing_request = 1
    with launcher_log_path.open("wb") as launcher_log:
        server = subprocess.Popen(
            args.server_command,
            stdout=launcher_log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_health(args.endpoint, server, args.health_timeout)
            evidence.record("server-ready", status="PASS", pid=server.pid)
            if args.shutdown_smoke_only:
                time.sleep(args.shutdown_smoke_hold_seconds)
                evidence.record(
                    "shutdown-smoke-ready",
                    status="PASS",
                    hold_seconds=args.shutdown_smoke_hold_seconds,
                )
                raise ShutdownSmokeComplete

            if args.steady_success_count > 0:
                outputs = []
                for index in range(args.steady_success_count):
                    phase = "warmup" if index == 0 else f"steady-{index:02d}"
                    outputs.append(
                        _run_success(
                            args.endpoint,
                            template,
                            output_dir,
                            phase,
                            args.seed,
                            args.request_timeout,
                            args.media_probe_binary,
                            evidence,
                        )
                    )
                    expected_timing_requests.append(next_timing_request)
                    next_timing_request += 1
                hashes = {output["sha256"] for output in outputs}
                if len(hashes) != 1:
                    raise RuntimeError(f"steady successful requests produced different MP4 hashes: {hashes}")
                evidence.record(
                    "determinism",
                    status="PASS",
                    sha256=next(iter(hashes)),
                    successful_requests=len(outputs),
                )
                summary = {
                    "status": "PASS",
                    "mode": "steady-success",
                    "warmup_requests": 1,
                    "measured_requests": args.steady_success_count - 1,
                    "events": evidence.events,
                    "successful_outputs": outputs,
                }
                raise SuccessOnlyComplete

            outputs = []
            outputs.append(
                _run_success(
                    args.endpoint,
                    template,
                    output_dir,
                    "success-a",
                    args.seed,
                    args.request_timeout,
                    args.media_probe_binary,
                    evidence,
                )
            )
            expected_timing_requests.append(next_timing_request)
            next_timing_request += 1
            if args.success_only:
                summary = {
                    "status": "PASS",
                    "mode": "success-only",
                    "events": evidence.events,
                    "successful_outputs": outputs,
                }
                raise SuccessOnlyComplete
            outputs.append(
                _run_success(
                    args.endpoint,
                    template,
                    output_dir,
                    "success-b",
                    args.seed,
                    args.request_timeout,
                    args.media_probe_binary,
                    evidence,
                )
            )
            expected_timing_requests.append(next_timing_request)
            next_timing_request += 1

            failure_payload = _request_bytes(template, "h3-gate-failure-c")
            failure_started = time.monotonic()
            failure_status, failure_body = _post(args.endpoint, failure_payload, args.request_timeout)
            failure_duration = time.monotonic() - failure_started
            if failure_status < 400:
                raise RuntimeError("failure-c unexpectedly returned a successful response")
            _wait_for_log(
                args.server_log_dir / "node_7.log",
                f"injected post-forward DiT worker failure on rank {args.failure_rank} "
                f"for batch {args.failure_batch_index}",
                server,
                args.request_timeout,
            )
            evidence.record(
                "failure-c",
                status="PASS",
                http_status=failure_status,
                duration_seconds=failure_duration,
                response=bytes(failure_body[:2048]).decode("utf-8", errors="replace"),
            )
            next_timing_request += 1

            outputs.append(
                _run_success(
                    args.endpoint,
                    template,
                    output_dir,
                    "recovery-d",
                    args.seed,
                    args.request_timeout,
                    args.media_probe_binary,
                    evidence,
                )
            )
            expected_timing_requests.append(next_timing_request)
            next_timing_request += 1

            disconnect_id = "h3-gate-disconnect-e"
            disconnect_payload = _request_bytes(template, disconnect_id)
            driver_log = args.server_log_dir / "node_0.log"
            disconnect_log_offset = driver_log.stat().st_size
            disconnect_started = time.monotonic()
            client_host, client_port = _disconnect_after_upload(
                args.endpoint, disconnect_payload, args.disconnect_delay
            )
            server_port = urlsplit(args.endpoint).port
            disconnect_evidence = _wait_for_log_line(
                driver_log,
                (
                    "Fail to write into Socket",
                    f"addr={client_host}:{client_port}:{server_port}",
                ),
                server,
                args.disconnect_timeout,
                disconnect_log_offset,
            )
            time.sleep(args.post_disconnect_settle)
            evidence.record(
                "disconnect-e",
                status="PASS",
                duration_seconds=time.monotonic() - disconnect_started,
                evidence=disconnect_evidence,
                client=f"{client_host}:{client_port}",
                semantics="client reset connection; completed response could not be written and was discarded",
            )
            expected_timing_requests.append(next_timing_request)
            next_timing_request += 1

            outputs.append(
                _run_success(
                    args.endpoint,
                    template,
                    output_dir,
                    "recovery-f",
                    args.seed,
                    args.request_timeout,
                    args.media_probe_binary,
                    evidence,
                )
            )
            expected_timing_requests.append(next_timing_request)
            next_timing_request += 1
            hashes = {output["sha256"] for output in outputs}
            if len(hashes) != 1:
                raise RuntimeError(f"deterministic successful requests produced different MP4 hashes: {hashes}")
            evidence.record("determinism", status="PASS", sha256=next(iter(hashes)), successful_requests=len(outputs))
            summary = {"status": "PASS", "events": evidence.events, "successful_outputs": outputs}
        except ShutdownSmokeComplete:
            summary = {"status": "PASS", "mode": "shutdown-smoke-only", "events": evidence.events}
        except SuccessOnlyComplete:
            pass
        except BaseException as error:
            evidence.record("gate-error", status="FAIL", error=repr(error))
            summary = {"status": "FAIL", "events": evidence.events, "error": repr(error)}
            raise
        finally:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            server_returncode, server_group_empty, reaped_server_pids = _terminate_server(server, args.shutdown_timeout)
            evidence.record(
                "server-shutdown",
                status="PASS" if server_group_empty else "FAIL",
                returncode=server_returncode,
                process_group_empty=server_group_empty,
                reaped_pids=reaped_server_pids,
            )
            summary["events"] = evidence.events
            summary["server_returncode"] = server_returncode
            stage_timings = _collect_stage_timings(args.server_log_dir)
            summary["stage_timings"] = stage_timings
            if expected_timing_requests:
                try:
                    _validate_complete_stage_timings(stage_timings, expected_timing_requests)
                except ValueError as error:
                    summary["status"] = "FAIL"
                    summary["stage_timing_error"] = str(error)
            leaked_shm = [str(path) for path in _shm_namespace_paths(args.master_port)]
            summary["leaked_shm_before_cleanup"] = leaked_shm
            removed_shm, remaining_shm = _cleanup_shm_namespace(args.master_port)
            summary["removed_shm"] = removed_shm
            summary["remaining_shm"] = remaining_shm
            if leaked_shm:
                summary["status"] = "FAIL"
                summary["cleanup_error"] = "server leaked shared-memory objects; gate cleanup restored the namespace"
            if remaining_shm:
                summary["status"] = "FAIL"
                summary["cleanup_error"] = "shared-memory objects remain after gate cleanup"
            if server_returncode not in (0, 130, -signal.SIGTERM):
                summary["status"] = "FAIL"
                summary["shutdown_error"] = f"server launcher exited with unexpected code {server_returncode}"
            if not server_group_empty:
                summary["status"] = "FAIL"
                summary["process_group_error"] = "server child process group remains after shutdown"
            _write_json(output_dir / "summary.json", summary)

    if summary["status"] != "PASS":
        return 1
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
