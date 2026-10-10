#!/usr/bin/env bash
set -euo pipefail

# Four-GPU AutoOpen video regression. Each Isaac Sim process sees exactly one
# physical GPU, so both physics and RTX rendering use that process's GPU 0.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENTRY="${REPO_ROOT}/gr00t/rl/scripts/doorman_auto_open.py"

RUN_NAME="${RUN_NAME:?Set RUN_NAME to a unique result name}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
SEEDS_CSV="${SEEDS:-1700,1701,1702,1703}"
MAX_STEPS="${MAX_STEPS:-3000}"

IFS=',' read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS_CSV}"
IFS=',' read -r -a SEEDS_ARRAY <<< "${SEEDS_CSV}"

if [[ "${#GPU_IDS_ARRAY[@]}" -ne "${#SEEDS_ARRAY[@]}" ]]; then
    echo "GPU_IDS and SEEDS must contain the same number of entries." >&2
    exit 2
fi

declare -a PIDS=()
declare -a OUTPUT_DIRS=()

cleanup() {
    local pid
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup INT TERM EXIT

cd "${REPO_ROOT}"

for index in "${!SEEDS_ARRAY[@]}"; do
    seed="${SEEDS_ARRAY[index]}"
    gpu_id="${GPU_IDS_ARRAY[index]}"
    output_dir="logs_rl/${RUN_NAME}_seed${seed}_6cam"
    console_log="${output_dir}.console.log"

    if [[ -e "${output_dir}" || -e "${console_log}" ]]; then
        echo "Refusing to overwrite existing output: ${output_dir}" >&2
        exit 2
    fi

    OUTPUT_DIRS+=("${output_dir}")
    echo "Starting seed ${seed} on physical GPU ${gpu_id}: ${output_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python "${ENTRY}" \
        --headless \
        --enable_cameras \
        --record_video \
        --device cuda:0 \
        --seed "${seed}" \
        --num_episodes 1 \
        --max_steps "${MAX_STEPS}" \
        --save_failures \
        --output_dir "${output_dir}" \
        "$@" \
        >"${console_log}" 2>&1 &
    PIDS+=("$!")
done

failures=0
for index in "${!PIDS[@]}"; do
    seed="${SEEDS_ARRAY[index]}"
    if wait "${PIDS[index]}"; then
        metadata="${OUTPUT_DIRS[index]}/episode_000001/metadata.json"
        if [[ ! -f "${metadata}" ]]; then
            echo "Seed ${seed} produced no metadata: ${metadata}" >&2
            failures=$((failures + 1))
        elif python -c \
            'import json, sys; raise SystemExit(not json.load(open(sys.argv[1]))["success"])' \
            "${metadata}"; then
            echo "Seed ${seed} passed: ${OUTPUT_DIRS[index]}"
        else
            reason="$(python -c \
                'import json, sys; print(json.load(open(sys.argv[1])).get("failure_reason"))' \
                "${metadata}")"
            echo "Seed ${seed} failed AutoOpen: ${reason}: ${OUTPUT_DIRS[index]}" >&2
            failures=$((failures + 1))
        fi
    else
        status=$?
        echo "Seed ${seed} failed with exit code ${status}: ${OUTPUT_DIRS[index]}.console.log" >&2
        failures=$((failures + 1))
    fi
done

trap - INT TERM EXIT

if (( failures > 0 )); then
    echo "AutoOpen regression failed for ${failures} seed(s)." >&2
    exit 1
fi

echo "AutoOpen regression completed for all ${#SEEDS_ARRAY[@]} seeds."
