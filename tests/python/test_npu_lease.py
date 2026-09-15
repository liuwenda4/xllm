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

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tools.npu_lease import (
    ACTIVITY_KEYS,
    DeviceLease,
    LeaseBlocked,
    NpuDevice,
    monitor_blockers,
    parse_device_spec,
    parse_health,
    parse_mapping,
    parse_processes,
    parse_usage,
    preflight_blockers,
    quiescence_blockers,
    required_process_blockers,
    run,
)

MAPPING = """
 NPU ID Chip ID Chip Logic ID Chip Phy-ID Chip Name
 0 0 0 0 Ascend910
 0 1 1 1 Ascend910
 0 2 - - Mcu
 1 0 2 2 Ascend910
 1 1 3 3 Ascend910
 1 2 - - Mcu
"""


def _device(logic_id: int) -> NpuDevice:
    return NpuDevice(
        card_id=logic_id // 2,
        chip_id=logic_id % 2,
        logic_id=logic_id,
        physical_id=logic_id,
        name="Ascend910",
    )


def _idle_snapshot() -> dict:
    return {
        "devices": {
            "0": {
                "processes": [],
                "usage": {key: 0 for key in ACTIVITY_KEYS},
                "health": {"status": "OK"},
            }
        }
    }


def test_parse_device_spec_supports_ranges_and_rejects_duplicates() -> None:
    assert parse_device_spec("0-2,4,6-7") == [0, 1, 2, 4, 6, 7]
    with pytest.raises(ValueError, match="duplicates"):
        parse_device_spec("0-2,2")
    with pytest.raises(ValueError, match="descending"):
        parse_device_spec("3-1")


def test_parse_mapping_uses_all_compute_chips_and_excludes_mcu() -> None:
    mapping = parse_mapping(MAPPING)
    assert list(mapping) == [0, 1, 2, 3]
    assert mapping[3] == _device(3)
    with pytest.raises(ValueError, match="no compute chips"):
        parse_mapping("0 2 - - Mcu")


def test_parse_usage_requires_every_activity_metric() -> None:
    output = "\n".join(f"{key} : 0" for key in ACTIVITY_KEYS)
    assert parse_usage(output) == {key: 0 for key in ACTIVITY_KEYS}
    with pytest.raises(ValueError, match="missing integer field"):
        parse_usage(output.replace(f"{ACTIVITY_KEYS[0]} : 0\n", ""))


def test_parse_health_and_processes_fail_closed() -> None:
    assert parse_health("Health Status : OK\nError Code : NA\nError Information : NA") == {
        "status": "OK",
        "error_code": "NA",
        "error_information": "NA",
    }
    assert parse_processes("No process in device.") == []
    assert parse_processes("Get pid name failed.\nProcess id:123 Process name: Process memory(MB):456") == [
        {"driver_pid": 123, "memory_mb": 456}
    ]
    with pytest.raises(ValueError, match="not recognized"):
        parse_processes("unexpected")


def test_preflight_rejects_process_activity_and_health() -> None:
    assert preflight_blockers(_idle_snapshot()) == []
    snapshot = _idle_snapshot()
    snapshot["devices"]["0"]["processes"] = [{"driver_pid": 1, "memory_mb": 1}]
    snapshot["devices"]["0"]["usage"][ACTIVITY_KEYS[0]] = 5
    snapshot["devices"]["0"]["health"]["status"] = "Fault"
    blockers = preflight_blockers(snapshot)
    assert "npu_0:process_present" in blockers
    assert any("active" in blocker for blocker in blockers)
    assert "npu_0:unhealthy" in blockers


def test_monitor_learns_one_pid_then_rejects_change_or_second_pid() -> None:
    allowed = {0: set()}
    snapshot = {"devices": {"0": {"processes": [{"driver_pid": 10, "memory_mb": 1}]}}}
    assert monitor_blockers(snapshot, allowed) == []
    assert allowed == {0: {10}}
    assert monitor_blockers(snapshot, allowed) == []
    changed = {"devices": {"0": {"processes": [{"driver_pid": 11, "memory_mb": 1}]}}}
    assert monitor_blockers(changed, allowed) == ["npu_0:driver_pids_changed:[10]->[11]"]
    multiple = {
        "devices": {
            "0": {
                "processes": [
                    {"driver_pid": 10, "memory_mb": 1},
                    {"driver_pid": 11, "memory_mb": 1},
                ]
            }
        }
    }
    assert "multiple_processes" in monitor_blockers(multiple, allowed)[0]


def test_required_process_and_final_quiescence_fail_closed() -> None:
    assert required_process_blockers({0: {10}, 1: {11}}, 1) == []
    assert required_process_blockers({0: set(), 1: {11}}, 1) == ["npu_0:expected_1_job_processes_observed_0"]
    assert quiescence_blockers({"devices": {"0": {"errors": [], "processes": [], "health": {"status": "OK"}}}}) == []
    blockers = quiescence_blockers(
        {
            "devices": {
                "0": {
                    "errors": [],
                    "processes": [{"driver_pid": 10, "memory_mb": 1}],
                    "health": {"status": "OK"},
                }
            }
        }
    )
    assert blockers == ["npu_0:process_present_after_child"]


def test_overlapping_leases_fail_and_disjoint_leases_succeed(tmp_path: Path) -> None:
    first = DeviceLease(tmp_path, [_device(0)], {"owner": "first"})
    overlap = DeviceLease(tmp_path, [_device(0)], {"owner": "overlap"})
    disjoint = DeviceLease(tmp_path, [_device(1)], {"owner": "disjoint"})
    first.acquire()
    try:
        with pytest.raises(LeaseBlocked, match="lease is held"):
            overlap.acquire()
        disjoint.acquire()
        disjoint.release()
    finally:
        first.release()


def test_lease_metadata_is_durable_and_lock_file_is_not_unlinked(tmp_path: Path) -> None:
    lease = DeviceLease(tmp_path, [_device(0)], {"owner": "test"})
    identities = lease.acquire()
    path = Path(identities[0]["path"])
    assert json.loads(path.read_text(encoding="utf-8"))["owner"] == "test"
    lease.release()
    assert path.is_file()


def test_mapping_failure_still_writes_fail_closed_evidence(tmp_path: Path) -> None:
    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    evidence_dir = tmp_path / "evidence"
    args = argparse.Namespace(
        devices="0",
        snapshot_delay=0.0,
        poll_interval=1.0,
        term_grace=0.0,
        query_timeout=1.0,
        command=["--", "/bin/true"],
        evidence_dir=evidence_dir,
        npu_smi=str(tmp_path / "missing-npu-smi"),
        lock_root=lock_root,
        health_every=1,
        expected_processes_per_device=1,
    )
    assert run(args) == 3
    final = json.loads((evidence_dir / "final.json").read_text(encoding="utf-8"))
    assert final["status"] == "BLOCKED_MAPPING"
    assert final["exit_code"] == 3
    assert final["blockers"]
