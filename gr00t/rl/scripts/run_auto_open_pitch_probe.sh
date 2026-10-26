#!/usr/bin/env bash
set -euo pipefail

# Dedicated debug entry for reporting production MOVE_TO_PREGRASP pitch selection.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENTRY="${REPO_ROOT}/gr00t/rl/scripts/doorman_auto_open.py"
PITCH_PROBE_SEEDS="${PITCH_PROBE_SEEDS:-1700,1701,1702,1703}"
PITCH_PROBE_GPU="${PITCH_PROBE_GPU:-0}"
PITCH_PROBE_OUTPUT="${PITCH_PROBE_OUTPUT:-${REPO_ROOT}/pitch_probe_results.json}"
PITCH_PROBE_LOG="${PITCH_PROBE_LOG:-${REPO_ROOT}/logs_rl/pitch_probe.console.log}"
PITCH_PROBE_RUN_DIR="${PITCH_PROBE_RUN_DIR:-${REPO_ROOT}/logs_rl/pitch_probe_run}"

if (( $# != 0 )); then
    echo "Configure PITCH_PROBE_SEEDS instead of command-line arguments." >&2
    exit 2
fi
if [[ ! -d /workspace/isaaclab ]]; then
    echo "Run this debug entry inside the doorman_train container." >&2
    exit 2
fi
IFS=',' read -r -a seeds <<< "${PITCH_PROBE_SEEDS}"
if (( ${#seeds[@]} == 0 )); then
    echo "PITCH_PROBE_SEEDS must contain at least one seed." >&2
    exit 2
fi
for seed in "${seeds[@]}"; do
    [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "Invalid seed: ${seed}" >&2; exit 2; }
done

mkdir -p "$(dirname "${PITCH_PROBE_OUTPUT}")" "$(dirname "${PITCH_PROBE_LOG}")" "${PITCH_PROBE_RUN_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export AUTO_OPEN_FIXED_ENTRY=run_auto_open_pitch_probe.sh
CUDA_VISIBLE_DEVICES="${PITCH_PROBE_GPU}" python "${ENTRY}" \
    --headless \
    --device cuda:0 \
    --seed "${seeds[0]}" \
    --episode_seeds "${seeds[@]}" \
    --batched_envs "${#seeds[@]}" \
    --max_steps 1200 \
    --disable_rgb_cameras \
    --summary_only \
    --collection_camera_mode none \
    --doorman_door_open_io out \
    --terminal_stage STOW_ARM_AND_CLOSE_GRIPPER \
    --pitch_probe_output "${PITCH_PROBE_OUTPUT}" \
    --output_dir "${PITCH_PROBE_RUN_DIR}" \
    >"${PITCH_PROBE_LOG}" 2>&1

cat "${PITCH_PROBE_OUTPUT}"
