#!/usr/bin/env bash

set -euo pipefail

WORLD_SIZE=16
BINARY="${ACLSHMEM_TEST_BINARY:-/data/workspace/lwd/xllm/build/lib.linux-aarch64-cpython-311/xllm/aclshmem_moe_w8a8_eager_test}"
ARTIFACT_BASE="${ACLSHMEM_TEST_ARTIFACT_BASE:-/data/workspace/lwd/tilelang-ascend/artifacts/moe}"
HCCL_PORT="${ACLSHMEM_TEST_HCCL_PORT:-29135}"
SHMEM_IP_PORT="${XLLM_ACLSHMEM_IP_PORT:-tcp://192.168.0.131:8911}"
SHMEM_HEAP_BYTES="${XLLM_ACLSHMEM_HEAP_BYTES:-67108864}"
TIMEOUT_SECONDS="${ACLSHMEM_TEST_TIMEOUT_SECONDS:-900}"
ATTEMPT_ID="${ACLSHMEM_TEST_ATTEMPT_ID:-$(date -u +%Y%m%dT%H%M%SZ)-xllm-fused-moe-w8a8-ep16}"
ARTIFACT_DIR="$ARTIFACT_BASE/$ATTEMPT_ID"

if [[ ! -x "$BINARY" ]]; then
  printf 'Test binary is missing or not executable: %s\n' "$BINARY" >&2
  exit 2
fi

mkdir -p "$ARTIFACT_DIR"
env | sort > "$ARTIFACT_DIR/environment.txt"

printf 'attempt_id=%s\n' "$ATTEMPT_ID"
printf 'artifact_dir=%s\n' "$ARTIFACT_DIR"
printf 'binary=%s\n' "$BINARY"
printf 'world_size=%d hccl_port=%s shmem_ip_port=%s timeout=%s\n' \
  "$WORLD_SIZE" "$HCCL_PORT" "$SHMEM_IP_PORT" "$TIMEOUT_SECONDS"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
  rank_log="$ARTIFACT_DIR/rank${rank}.log"
  timeout --signal=TERM --kill-after=30 "$TIMEOUT_SECONDS" \
    env \
      ACLSHMEM_TEST_RANK="$rank" \
      ACLSHMEM_TEST_HCCL_PORT="$HCCL_PORT" \
      XLLM_ACLSHMEM_IP_PORT="$SHMEM_IP_PORT" \
      XLLM_ACLSHMEM_HEAP_BYTES="$SHMEM_HEAP_BYTES" \
      HCCL_CONNECT_TIMEOUT="$TIMEOUT_SECONDS" \
      HCCL_EXEC_TIMEOUT="$TIMEOUT_SECONDS" \
      GLOG_logtostderr=1 \
      "$BINARY" \
        --gtest_filter=AclShmemMoeW8A8EagerTest.RunsRealGeometryThroughExactBucket \
        > "$rank_log" 2>&1 &
  pids+=("$!")
done

exit_codes=()
all_passed=true
for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
  set +e
  wait "${pids[$rank]}"
  exit_code=$?
  set -e
  exit_codes+=("$exit_code")
  printf 'rank=%d exit_code=%d log=%s\n' \
    "$rank" "$exit_code" "$ARTIFACT_DIR/rank${rank}.log"
  if ((exit_code != 0)); then
    all_passed=false
  fi
done

trap - EXIT INT TERM

python - "$ARTIFACT_DIR/summary.json" "$ATTEMPT_ID" "$BINARY" \
  "$HCCL_PORT" "$SHMEM_IP_PORT" "$SHMEM_HEAP_BYTES" "$TIMEOUT_SECONDS" \
  "$all_passed" "${exit_codes[@]}" <<'PY'
import json
import sys

(
    output,
    attempt_id,
    binary,
    hccl_port,
    shmem_ip_port,
    shmem_heap_bytes,
    timeout_seconds,
    all_passed,
    *exit_codes,
) = sys.argv[1:]

summary = {
    "attempt_id": attempt_id,
    "binary": binary,
    "geometry": {
        "local_tokens": 4,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "global_experts": 256,
        "local_experts": 16,
        "topk": 6,
        "ep_world_size": 16,
        "max_capacity": 1024,
        "expert_dtype": "int8",
        "activation_dtype": "bfloat16",
    },
    "generations": 4,
    "route_pattern": "3 local routes + 3 remote routes per token",
    "weights": "synthetic sparse diagonal W8A8",
    "runtime": {
        "hccl_port": int(hccl_port),
        "shmem_ip_port": shmem_ip_port,
        "shmem_heap_bytes": int(shmem_heap_bytes),
        "timeout_seconds": int(timeout_seconds),
    },
    "exit_codes": [int(value) for value in exit_codes],
    "ok": all_passed == "true",
}

with open(output, "w", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

if [[ "$all_passed" != "true" ]]; then
  printf 'EP16 W8A8 test failed; see %s\n' "$ARTIFACT_DIR" >&2
  exit 1
fi

printf 'EP16 W8A8 test passed; summary=%s\n' "$ARTIFACT_DIR/summary.json"
