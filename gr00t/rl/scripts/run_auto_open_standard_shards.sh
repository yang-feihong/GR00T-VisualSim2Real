#!/usr/bin/env bash
set -euo pipefail

# Standards-compliant RGB collection scales as independent per-seed shards.
# Each worker exposes one physical GPU to one Isaac App at a time.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENTRY="${REPO_ROOT}/gr00t/rl/scripts/doorman_auto_open.py"

DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to a new dataset directory}"
SEED_START="${SEED_START:-1700}"
NUM_EPISODES="${NUM_EPISODES:-4}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
MAX_STEPS="${MAX_STEPS:-3000}"

IFS=',' read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS_CSV}"
if (( NUM_EPISODES < 1 || ${#GPU_IDS_ARRAY[@]} < 1 )); then
    echo "NUM_EPISODES and GPU_IDS must be non-empty and positive." >&2
    exit 2
fi
if [[ -e "${DATASET_ROOT}" ]]; then
    echo "Refusing to overwrite existing DATASET_ROOT: ${DATASET_ROOT}" >&2
    exit 2
fi

mkdir -p "${DATASET_ROOT}/shards" "${DATASET_ROOT}/logs"
declare -a PIDS=()

cleanup() {
    local pid
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup INT TERM EXIT

gpu_count="${#GPU_IDS_ARRAY[@]}"
for index in "${!GPU_IDS_ARRAY[@]}"; do
    gpu_id="${GPU_IDS_ARRAY[index]}"
    (
        worker_failures=0
        for ((offset=index; offset<NUM_EPISODES; offset+=gpu_count)); do
            seed=$((SEED_START + offset))
            shard_name="episode-seed${seed}"
            shard_root="${DATASET_ROOT}/shards/${shard_name}"
            control_root="${DATASET_ROOT}/logs/${shard_name}-control"
            console_log="${DATASET_ROOT}/logs/${shard_name}.console.log"
            echo "Starting ${shard_name} on physical GPU ${gpu_id}"
            if ! CUDA_VISIBLE_DEVICES="${gpu_id}" python "${ENTRY}" \
                --headless \
                --device cuda:0 \
                --seed "${seed}" \
                --num_episodes 1 \
                --max_steps "${MAX_STEPS}" \
                --summary_only \
                --standard_dataset_dir "${shard_root}" \
                --output_dir "${control_root}" \
                >"${console_log}" 2>&1; then
                echo "${shard_name} failed; inspect ${console_log}." >&2
                worker_failures=$((worker_failures + 1))
            fi
        done
        exit "${worker_failures}"
    ) &
    PIDS+=("$!")
done

failures=0
for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[index]}"; then
        echo "GPU worker ${index} failed; inspect DATASET_ROOT/logs." >&2
        failures=$((failures + 1))
    fi
done
trap - INT TERM EXIT

DATASET_ROOT="${DATASET_ROOT}" SEED_START="${SEED_START}" \
NUM_EPISODES="${NUM_EPISODES}" \
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DATASET_ROOT"])
seed_start = int(os.environ["SEED_START"])
count = int(os.environ["NUM_EPISODES"])
names = [f"episode-seed{seed}" for seed in range(seed_start, seed_start + count)]
rows = []
for name in names:
    report_path = root / "shards" / name / "meta" / "acceptance_report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    acceptance = None
    if report is not None:
        reason_counts = {}
        for item in report.get("quarantine", []):
            for reason in item.get("reasons", []):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        acceptance = {
            "total_episodes": report.get("total_episodes", 0),
            "accepted_episodes": report.get("accepted_episodes", 0),
            "quarantined_episodes": report.get("quarantined_episodes", 0),
            "quarantine_reason_counts": reason_counts,
            "exceptions": report.get("exceptions", []),
        }
    rows.append(
        {
            "name": name,
            "path": f"shards/{name}",
            "acceptance_report": f"shards/{name}/meta/acceptance_report.json",
            "acceptance": acceptance,
        }
    )
(root / "manifest.json").write_text(
    json.dumps(
        {
            "seed_start": int(os.environ["SEED_START"]),
            "requested_episodes": int(os.environ["NUM_EPISODES"]),
            "shards": rows,
        },
        indent=2,
        ensure_ascii=False,
    )
)
PY

if (( failures > 0 )); then
    exit 1
fi
echo "Standard collection completed: ${DATASET_ROOT}/manifest.json"
