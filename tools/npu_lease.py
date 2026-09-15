#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run one command under cooperative, evidenced Ascend NPU leases."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "xllm.npu_lease/v1"
SNAPSHOT_SCHEMA = "xllm.npu_lease_snapshot/v1"
FINAL_SCHEMA = "xllm.npu_lease_final/v1"
ACTIVITY_KEYS = (
    "NPU Utilization(%)",
    "Aicore Usage Rate(%)",
    "Aivector Usage Rate(%)",
    "Aicpu Usage Rate(%)",
    "Aicube Usage Rate(%)",
    "DDR Bandwidth Usage Rate(%)",
    "HBM Bandwidth Usage Rate(%)",
)
PID_PATTERN = re.compile(r"Process id:\s*(\d+).*?Process memory\(MB\):\s*(\d+)")


@dataclass(frozen=True)
class NpuDevice:
    card_id: int
    chip_id: int
    logic_id: int
    physical_id: int
    name: str


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float


class LeaseBlocked(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def parse_device_spec(spec: str) -> list[int]:
    """Parse comma-separated logical IDs and inclusive ranges."""
    values: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            raise ValueError("device specification contains an empty item")
        if "-" in item:
            fields = item.split("-")
            if len(fields) != 2 or not all(field.isdigit() for field in fields):
                raise ValueError(f"invalid device range {item!r}")
            start, stop = (int(field) for field in fields)
            if start > stop:
                raise ValueError(f"descending device range {item!r}")
            values.extend(range(start, stop + 1))
        elif item.isdigit():
            values.append(int(item))
        else:
            raise ValueError(f"invalid device ID {item!r}")
    if not values:
        raise ValueError("at least one device is required")
    if len(set(values)) != len(values):
        raise ValueError("device specification contains duplicates")
    return values


def parse_mapping(output: str) -> dict[int, NpuDevice]:
    """Parse `npu-smi info -m` without treating MCU rows as devices."""
    devices: dict[int, NpuDevice] = {}
    for raw_line in output.splitlines():
        fields = raw_line.split()
        if len(fields) != 5 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        if fields[2] == "-" and fields[3] == "-" and fields[4].lower() == "mcu":
            continue
        if not fields[2].isdigit() or not fields[3].isdigit() or fields[4].lower() == "mcu":
            raise ValueError(f"malformed compute-chip mapping row: {raw_line!r}")
        device = NpuDevice(
            card_id=int(fields[0]),
            chip_id=int(fields[1]),
            logic_id=int(fields[2]),
            physical_id=int(fields[3]),
            name=fields[4],
        )
        if device.logic_id in devices:
            raise ValueError(f"duplicate logical NPU ID {device.logic_id}")
        devices[device.logic_id] = device
    if not devices:
        raise ValueError("npu-smi mapping contains no compute chips")
    physical_ids = [device.physical_id for device in devices.values()]
    if len(set(physical_ids)) != len(physical_ids):
        raise ValueError("npu-smi mapping contains duplicate physical IDs")
    return devices


def parse_key_values(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in values:
            raise ValueError(f"duplicate npu-smi field {key!r}")
        values[key] = value.strip()
    if not values:
        raise ValueError("npu-smi output contains no key/value fields")
    return values


def parse_usage(output: str) -> dict[str, int]:
    values = parse_key_values(output)
    usage: dict[str, int] = {}
    for key in ACTIVITY_KEYS:
        raw_value = values.get(key)
        if raw_value is None or not raw_value.isdigit():
            raise ValueError(f"npu-smi usages is missing integer field {key!r}")
        usage[key] = int(raw_value)
    return usage


def parse_health(output: str) -> dict[str, str]:
    values = parse_key_values(output)
    status = values.get("Health Status")
    if not status:
        raise ValueError("npu-smi health is missing Health Status")
    return {
        "status": status,
        "error_code": values.get("Error Code", ""),
        "error_information": values.get("Error Information", ""),
    }


def parse_processes(output: str) -> list[dict[str, int]]:
    if "No process in device." in output:
        if PID_PATTERN.search(output):
            raise ValueError("npu-smi proc-mem contains both no-process and process records")
        return []
    processes = [
        {"driver_pid": int(match.group(1)), "memory_mb": int(match.group(2))} for match in PID_PATTERN.finditer(output)
    ]
    if not processes:
        raise ValueError("npu-smi proc-mem output is not recognized")
    pids = [process["driver_pid"] for process in processes]
    if len(set(pids)) != len(pids):
        raise ValueError("npu-smi proc-mem contains duplicate process IDs")
    return processes


def _run_command(argv: Sequence[str], timeout: float) -> CommandResult:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            argv=list(argv),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=(time.monotonic() - started) * 1000.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return CommandResult(
            argv=list(argv),
            returncode=-1,
            stdout="",
            stderr=f"{type(error).__name__}: {error}",
            duration_ms=(time.monotonic() - started) * 1000.0,
        )


def _require_success(result: CommandResult, description: str) -> str:
    if result.returncode != 0:
        raise LeaseBlocked(f"{description} failed with return code {result.returncode}: {result.stderr.strip()}")
    return result.stdout


class DeviceLease:
    def __init__(self, lock_root: Path, devices: Sequence[NpuDevice], metadata: Mapping[str, Any]) -> None:
        self._lock_root = lock_root
        self._devices = sorted(devices, key=lambda device: device.physical_id)
        self._metadata = dict(metadata)
        self._handles: list[Any] = []

    @property
    def file_descriptors(self) -> tuple[int, ...]:
        return tuple(handle.fileno() for handle in self._handles)

    def acquire(self) -> list[dict[str, Any]]:
        if not self._lock_root.is_dir():
            raise LeaseBlocked(f"lock root must already exist: {self._lock_root}")
        identities = []
        try:
            for device in self._devices:
                path = self._lock_root / f"npu-physical-{device.physical_id}.lock"
                handle = path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    handle.seek(0)
                    owner = handle.read().strip()
                    handle.close()
                    raise LeaseBlocked(f"NPU {device.logic_id} lease is held: {owner}") from error
                handle.seek(0)
                handle.truncate()
                handle.write(json.dumps({**self._metadata, "device": asdict(device)}, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                self._handles.append(handle)
                stat = os.fstat(handle.fileno())
                identities.append(
                    {
                        "path": str(path),
                        "device": asdict(device),
                        "inode": stat.st_ino,
                        "device_id": stat.st_dev,
                    }
                )
        except BaseException:
            self.release()
            raise
        return identities

    def release(self) -> None:
        for handle in reversed(self._handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        self._handles.clear()


def _query(argv: Sequence[str], timeout: float, description: str) -> tuple[CommandResult, str]:
    result = _run_command(argv, timeout)
    return result, _require_success(result, description)


def collect_snapshot(
    devices: Sequence[NpuDevice],
    *,
    npu_smi: str,
    timeout: float,
    include_usage: bool,
    include_health: bool,
) -> dict[str, Any]:
    commands: list[dict[str, Any]] = []
    observed: dict[str, Any] = {}

    def capture(
        argv: Sequence[str],
        description: str,
        parser: Any,
        errors: list[str],
    ) -> Any:
        result = _run_command(argv, timeout)
        commands.append(asdict(result))
        if result.returncode != 0:
            errors.append(f"{description} failed with return code {result.returncode}: {result.stderr.strip()}")
            return None
        try:
            return parser(result.stdout)
        except (TypeError, ValueError) as error:
            errors.append(f"{description} parse failed: {error}")
            return None

    for device in devices:
        device_key = str(device.logic_id)
        errors: list[str] = []
        entry: dict[str, Any] = {"mapping": asdict(device), "errors": errors}
        processes = capture(
            [npu_smi, "info", "-t", "proc-mem", "-i", str(device.card_id), "-c", str(device.chip_id)],
            f"NPU {device.logic_id} process query",
            parse_processes,
            errors,
        )
        if processes is not None:
            entry["processes"] = processes
        if include_usage:
            usage = capture(
                [npu_smi, "info", "-t", "usages", "-i", str(device.card_id), "-c", str(device.chip_id)],
                f"NPU {device.logic_id} usage query",
                parse_usage,
                errors,
            )
            if usage is not None:
                entry["usage"] = usage
        if include_health:
            health = capture(
                [npu_smi, "info", "-t", "health", "-i", str(device.card_id), "-c", str(device.chip_id)],
                f"NPU {device.logic_id} health query",
                parse_health,
                errors,
            )
            if health is not None:
                entry["health"] = health
        observed[device_key] = entry
    return {
        "schema": SNAPSHOT_SCHEMA,
        "captured_at_utc": _utc_now(),
        "captured_at_monotonic": time.monotonic(),
        "devices": observed,
        "commands": commands,
    }


def preflight_blockers(snapshot: Mapping[str, Any]) -> list[str]:
    blockers = []
    devices = snapshot.get("devices")
    if not isinstance(devices, Mapping):
        return ["snapshot:missing_devices"]
    for logic_id, value in devices.items():
        if not isinstance(value, Mapping):
            blockers.append(f"npu_{logic_id}:malformed")
            continue
        processes = value.get("processes")
        if not isinstance(processes, list) or processes:
            blockers.append(f"npu_{logic_id}:process_present")
        health = value.get("health")
        if not isinstance(health, Mapping) or health.get("status") != "OK":
            blockers.append(f"npu_{logic_id}:unhealthy")
        usage = value.get("usage")
        if not isinstance(usage, Mapping):
            blockers.append(f"npu_{logic_id}:missing_usage")
        else:
            for key in ACTIVITY_KEYS:
                if usage.get(key) != 0:
                    blockers.append(f"npu_{logic_id}:active:{key}")
    return blockers


def monitor_blockers(
    snapshot: Mapping[str, Any],
    allowed_driver_pids: dict[int, set[int]],
    expected_processes_per_device: int = 1,
) -> list[str]:
    blockers = []
    devices = snapshot.get("devices")
    if not isinstance(devices, Mapping):
        return ["monitor:missing_devices"]
    for logic_id_text, value in devices.items():
        logic_id = int(logic_id_text)
        if not isinstance(value, Mapping):
            blockers.append(f"npu_{logic_id}:malformed")
            continue
        processes = value.get("processes")
        if not isinstance(processes, list):
            blockers.append(f"npu_{logic_id}:missing_processes")
            continue
        pids = [process.get("driver_pid") for process in processes if isinstance(process, Mapping)]
        if len(pids) != len(processes) or any(isinstance(pid, bool) or not isinstance(pid, int) for pid in pids):
            blockers.append(f"npu_{logic_id}:malformed_process")
            continue
        if len(pids) > expected_processes_per_device:
            blockers.append(f"npu_{logic_id}:multiple_processes:{pids}")
            continue
        if pids:
            observed_pids = set(pids)
            allowed_pids = allowed_driver_pids[logic_id]
            if not allowed_pids:
                allowed_pids.update(observed_pids)
            elif observed_pids != allowed_pids:
                blockers.append(f"npu_{logic_id}:driver_pids_changed:{sorted(allowed_pids)}->{sorted(observed_pids)}")
        health = value.get("health")
        if health is not None and (not isinstance(health, Mapping) or health.get("status") != "OK"):
            blockers.append(f"npu_{logic_id}:unhealthy")
    return blockers


def required_process_blockers(
    allowed_driver_pids: Mapping[int, set[int]], expected_processes_per_device: int
) -> list[str]:
    return [
        f"npu_{logic_id}:expected_{expected_processes_per_device}_job_processes_observed_{len(pids)}"
        for logic_id, pids in allowed_driver_pids.items()
        if len(pids) != expected_processes_per_device
    ]


def quiescence_blockers(snapshot: Mapping[str, Any]) -> list[str]:
    blockers = []
    devices = snapshot.get("devices")
    if not isinstance(devices, Mapping):
        return ["final:missing_devices"]
    for logic_id, value in devices.items():
        if not isinstance(value, Mapping) or value.get("errors"):
            blockers.append(f"npu_{logic_id}:final_snapshot_malformed")
            continue
        processes = value.get("processes")
        if not isinstance(processes, list) or processes:
            blockers.append(f"npu_{logic_id}:process_present_after_child")
        health = value.get("health")
        if not isinstance(health, Mapping) or health.get("status") != "OK":
            blockers.append(f"npu_{logic_id}:unhealthy_after_child")
    return blockers


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _wait_process_group_empty(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.1)
    return not _process_group_exists(pgid)


def _terminate_process_group(child: subprocess.Popen[Any], pgid: int, grace_seconds: float) -> list[dict[str, Any]]:
    signals = []
    if not _process_group_exists(pgid):
        child.poll()
        return signals
    os.killpg(pgid, signal.SIGTERM)
    signals.append({"signal": "SIGTERM", "at_utc": _utc_now(), "pgid": pgid})
    if not _wait_process_group_empty(pgid, grace_seconds):
        os.killpg(pgid, signal.SIGKILL)
        signals.append({"signal": "SIGKILL", "at_utc": _utc_now(), "pgid": pgid})
        _wait_process_group_empty(pgid, max(1.0, grace_seconds))
    child.poll()
    return signals


def _validate_number(value: float, name: str, *, positive: bool = False) -> float:
    if not math.isfinite(value) or value < 0 or (positive and value <= 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'} and finite")
    return value


def run(args: argparse.Namespace) -> int:
    device_ids = parse_device_spec(args.devices)
    if args.expected_processes_per_device <= 0:
        raise ValueError("expected processes per device must be positive")
    for value, name, positive in (
        (args.snapshot_delay, "snapshot delay", False),
        (args.poll_interval, "poll interval", True),
        (args.term_grace, "termination grace", False),
        (args.query_timeout, "query timeout", True),
    ):
        _validate_number(value, name, positive=positive)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")

    evidence_dir = args.evidence_dir.resolve()
    if evidence_dir.exists():
        raise ValueError(f"refusing to overwrite evidence directory: {evidence_dir}")
    if not evidence_dir.parent.is_dir():
        raise ValueError(f"evidence parent must already exist: {evidence_dir.parent}")
    evidence_dir.mkdir()
    snapshot_dir = evidence_dir / "snapshots"
    snapshot_dir.mkdir()

    manifest = {
        "schema": SCHEMA,
        "status": "INITIALIZING",
        "created_at_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip(),
        "uid": os.getuid(),
        "pid": os.getpid(),
        "requested_logic_ids": device_ids,
        "devices": [],
        "command": command,
        "npu_smi": args.npu_smi,
        "mapping_command": None,
        "lock_root": str(args.lock_root.resolve()),
        "lock_identities": [],
        "child": None,
        "blockers": [],
        "signals": [],
    }
    _write_json(evidence_dir / "manifest.json", manifest)

    interrupted: list[int] = []
    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        interrupted.append(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)

    try:
        mapping_result = _run_command([args.npu_smi, "info", "-m"], args.query_timeout)
        manifest["mapping_command"] = asdict(mapping_result)
        _write_json(evidence_dir / "manifest.json", manifest)
        mapping_output = _require_success(mapping_result, "NPU mapping")
        mapping = parse_mapping(mapping_output)
        missing = sorted(set(device_ids) - set(mapping))
        if missing:
            raise LeaseBlocked(f"requested logical NPUs are absent: {missing}")
        devices = [mapping[device_id] for device_id in device_ids]
        if len({device.physical_id for device in devices}) != len(devices):
            raise LeaseBlocked("requested logical NPUs do not map to unique physical IDs")
        manifest["devices"] = [asdict(device) for device in devices]
        _write_json(evidence_dir / "manifest.json", manifest)
    except BaseException as error:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        manifest["status"] = "BLOCKED_MAPPING"
        manifest["blockers"] = [f"{type(error).__name__}: {error}"]
        manifest["finished_at_utc"] = _utc_now()
        _write_json(evidence_dir / "manifest.json", manifest)
        _write_json(
            evidence_dir / "final.json",
            {
                "schema": FINAL_SCHEMA,
                "status": "BLOCKED_MAPPING",
                "exit_code": 3,
                "finished_at_utc": _utc_now(),
                "manifest": {"path": "manifest.json", "sha256": _sha256(evidence_dir / "manifest.json")},
                "snapshots": [],
                "blockers": manifest["blockers"],
            },
        )
        print(evidence_dir / "final.json")
        return 3

    lease = DeviceLease(
        args.lock_root.resolve(),
        devices,
        {"schema": SCHEMA, "pid": os.getpid(), "evidence_dir": str(evidence_dir), "created_at_utc": _utc_now()},
    )
    child: subprocess.Popen[Any] | None = None
    child_pgid: int | None = None
    snapshot_paths: list[Path] = []
    status = "FAILED"
    exit_code = 1
    blockers: list[str] = []
    signal_history: list[dict[str, Any]] = []
    try:
        manifest["lock_identities"] = lease.acquire()
        manifest["status"] = "PREFLIGHT"
        _write_json(evidence_dir / "manifest.json", manifest)
        for wave in range(2):
            snapshot = collect_snapshot(
                devices,
                npu_smi=args.npu_smi,
                timeout=args.query_timeout,
                include_usage=True,
                include_health=True,
            )
            path = snapshot_dir / f"preflight-{wave + 1:02d}.json"
            _write_json(path, snapshot)
            snapshot_paths.append(path)
            blockers.extend(preflight_blockers(snapshot))
            if interrupted:
                blockers.append(f"launcher_received_signal:{signal.Signals(interrupted[0]).name}")
                status = "INTERRUPTED"
                exit_code = 128 + interrupted[0]
                return exit_code
            if wave == 0 and args.snapshot_delay:
                time.sleep(args.snapshot_delay)
        if blockers:
            status = "BLOCKED_PREFLIGHT"
            exit_code = 3
            return exit_code

        remap_result = _run_command([args.npu_smi, "info", "-m"], args.query_timeout)
        manifest["remapping_command"] = asdict(remap_result)
        _write_json(evidence_dir / "manifest.json", manifest)
        remap_output = _require_success(remap_result, "NPU remapping")
        if parse_mapping(remap_output) != mapping:
            blockers.append("npu_mapping_changed_after_lock")
            status = "BLOCKED_MAPPING_CHANGED"
            exit_code = 3
            return exit_code
        stdout_path = evidence_dir / "child.stdout.log"
        stderr_path = evidence_dir / "child.stderr.log"
        environment = os.environ.copy()
        environment["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(device_id) for device_id in device_ids)
        environment["XLLM_NPU_LEASE_EVIDENCE_DIR"] = str(evidence_dir)
        allowed_driver_pids = {device.logic_id: set() for device in devices}
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            child = subprocess.Popen(
                command,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
                pass_fds=lease.file_descriptors,
            )
            child_pgid = os.getpgid(child.pid)
            manifest["child"] = {
                "pid": child.pid,
                "pgid": child_pgid,
                "started_at_utc": _utc_now(),
                "stdout": stdout_path.name,
                "stderr": stderr_path.name,
            }
            manifest["status"] = "RUNNING"
            status = "RUNNING"
            _write_json(evidence_dir / "manifest.json", manifest)
            poll_index = 0
            while child.poll() is None:
                if interrupted:
                    blockers.append(f"launcher_received_signal:{signal.Signals(interrupted[0]).name}")
                    signal_history.extend(_terminate_process_group(child, child_pgid, args.term_grace))
                    status = "INTERRUPTED"
                    exit_code = 128 + interrupted[0]
                    break
                poll_index += 1
                snapshot = collect_snapshot(
                    devices,
                    npu_smi=args.npu_smi,
                    timeout=args.query_timeout,
                    include_usage=False,
                    include_health=poll_index % args.health_every == 0,
                )
                path = snapshot_dir / f"monitor-{poll_index:06d}.json"
                _write_json(path, snapshot)
                snapshot_paths.append(path)
                observed_blockers = monitor_blockers(
                    snapshot,
                    allowed_driver_pids,
                    args.expected_processes_per_device,
                )
                if observed_blockers:
                    blockers.extend(observed_blockers)
                    signal_history.extend(_terminate_process_group(child, child_pgid, args.term_grace))
                    status = "CONTAMINATED_RESOURCE_CONTENTION"
                    exit_code = 4
                    break
                time.sleep(args.poll_interval)
            if status == "RUNNING":
                child_returncode = child.wait()
                status = "PASS" if child_returncode == 0 else "CHILD_FAILED"
                exit_code = child_returncode if child_returncode >= 0 else 128 - child_returncode
                blockers.extend(required_process_blockers(allowed_driver_pids, args.expected_processes_per_device))
                if child_pgid is not None and not _wait_process_group_empty(child_pgid, args.term_grace):
                    blockers.append("child_process_group_not_empty_after_leader_exit")
                    signal_history.extend(_terminate_process_group(child, child_pgid, args.term_grace))
                if blockers and status == "PASS":
                    status = "CONTAMINATED_RESOURCE_CONTENTION"
                    exit_code = 4

        final_snapshot = collect_snapshot(
            devices,
            npu_smi=args.npu_smi,
            timeout=args.query_timeout,
            include_usage=True,
            include_health=True,
        )
        final_snapshot_path = snapshot_dir / "final.json"
        _write_json(final_snapshot_path, final_snapshot)
        snapshot_paths.append(final_snapshot_path)
        final_blockers = quiescence_blockers(final_snapshot)
        if final_blockers:
            blockers.extend(final_blockers)
            if status == "PASS":
                status = "CONTAMINATED_RESOURCE_CONTENTION"
                exit_code = 4
        return exit_code
    except LeaseBlocked as error:
        blockers.append(str(error))
        status = "BLOCKED"
        exit_code = 3
        return exit_code
    except BaseException as error:
        blockers.append(f"{type(error).__name__}: {error}")
        if child is not None and child_pgid is not None:
            try:
                signal_history.extend(_terminate_process_group(child, child_pgid, args.term_grace))
            except (OSError, ProcessLookupError) as terminate_error:
                blockers.append(f"termination_failed:{type(terminate_error).__name__}:{terminate_error}")
        status = "FAILED"
        exit_code = 1
        return exit_code
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        manifest["status"] = status
        manifest["blockers"] = blockers
        manifest["signals"] = signal_history
        if child is not None:
            manifest["child_returncode"] = child.poll()
        manifest["finished_at_utc"] = _utc_now()
        try:
            _write_json(evidence_dir / "manifest.json", manifest)
            final = {
                "schema": FINAL_SCHEMA,
                "status": status,
                "exit_code": exit_code,
                "finished_at_utc": _utc_now(),
                "manifest": {
                    "path": "manifest.json",
                    "sha256": _sha256(evidence_dir / "manifest.json"),
                },
                "snapshots": [
                    {"path": str(path.relative_to(evidence_dir)), "sha256": _sha256(path)} for path in snapshot_paths
                ],
                "blockers": blockers,
            }
            _write_json(evidence_dir / "final.json", final)
            print(evidence_dir / "final.json")
        finally:
            lease.release()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", required=True, help="Logical NPU IDs, for example 0-1 or 0,2,4")
    parser.add_argument("--lock-root", type=Path, required=True, help="Existing host-shared lock directory")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--npu-smi", default="npu-smi")
    parser.add_argument("--snapshot-delay", type=float, default=2.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--health-every", type=int, default=12)
    parser.add_argument("--expected-processes-per-device", type=int, default=1)
    parser.add_argument("--term-grace", type=float, default=15.0)
    parser.add_argument("--query-timeout", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.health_every <= 0:
        parser.error("--health-every must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
