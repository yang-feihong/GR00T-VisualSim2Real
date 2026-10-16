#!/usr/bin/env bash
set -euo pipefail

# Sole AutoOpen entry: one vectorized data-collection process per physical GPU.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENTRY="${REPO_ROOT}/gr00t/rl/scripts/doorman_auto_open.py"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
MAX_STEPS="${MAX_STEPS:-4000}"
CAMERA_MODE="${CAMERA_MODE:-none}"
DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to a new dataset directory}"

if (( $# != 0 )); then
    echo "The fixed AutoOpen collection entry accepts no command-line arguments." >&2
    exit 2
fi
if ! command -v python >/dev/null 2>&1 || [[ ! -d /workspace/isaaclab ]]; then
    echo "Run this launcher inside the doorman_train runtime container." >&2
    exit 2
fi
if [[ ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_STEPS must be a positive integer." >&2
    exit 2
fi
if [[ "${CAMERA_MODE}" != none && "${CAMERA_MODE}" != dual && "${CAMERA_MODE}" != six ]]; then
    echo "CAMERA_MODE must be none, dual, or six." >&2
    exit 2
fi
if [[ -e "${DATASET_ROOT}" ]]; then
    echo "Refusing to overwrite existing DATASET_ROOT: ${DATASET_ROOT}" >&2
    exit 2
fi

IFS=',' read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS_CSV}"
if (( ${#GPU_IDS_ARRAY[@]} < 1 )); then
    echo "GPU_IDS must contain at least one physical GPU." >&2
    exit 2
fi

declare -a SEED_ARRAY=()
if [[ -n "${SEEDS:-}" ]]; then
    IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
else
    SEED_START="${SEED_START:-1700}"
    NUM_EPISODES="${NUM_EPISODES:-1000}"
    if [[ ! "${SEED_START}" =~ ^[0-9]+$ || ! "${NUM_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
        echo "SEED_START and NUM_EPISODES must be non-negative/positive integers." >&2
        exit 2
    fi
    for ((index=0; index<NUM_EPISODES; index++)); do
        SEED_ARRAY+=("$((SEED_START + index))")
    done
fi
declare -A SEEN_SEEDS=()
for seed in "${SEED_ARRAY[@]}"; do
    if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
        echo "SEEDS must contain non-negative integers." >&2
        exit 2
    fi
    if [[ -n "${SEEN_SEEDS[${seed}]:-}" ]]; then
        echo "SEEDS contains duplicate seed ${seed}." >&2
        exit 2
    fi
    SEEN_SEEDS[${seed}]=1
done
if (( ${#SEED_ARRAY[@]} < 1 )); then
    echo "At least one seed is required." >&2
    exit 2
fi

declare -a COMMON_RUNTIME_ARGS=(
    --headless
    --device cuda:0
    --max_steps "${MAX_STEPS}"
    --batch_contact_ik_backend exact_workers
    --batch_path_backend exact_workers
    --batch_panel_clearance_backend gpu
    --batch_geometry_fallback_backend numpy_batch
    --summary_only
    --collection_camera_mode "${CAMERA_MODE}"
)
declare -a COLLECTION_ARGS=()
if [[ "${CAMERA_MODE}" == none ]]; then
    COLLECTION_ARGS+=(--disable_rgb_cameras)
else
    COLLECTION_ARGS+=(
        --camera_mount_position_noise_m 0.02
        --camera_mount_rotation_noise_deg 15.0
        --camera_focal_scale_noise_fraction 0.08
        --camera_principal_point_noise_px 5.0
        --camera_distortion_noise_fraction 0.10
    )
fi
if [[ "${SEMANTIC_TARGET_NOISE:-0}" == 1 ]]; then
    COLLECTION_ARGS+=(
        --target_noise approach_normal_m=0.03
        --target_noise approach_tangent_m=0.03
        --target_noise panel_vertical_m=0.05
        --target_noise through_tangent_m=0.08
        --target_noise through_yaw_rad=0.06
    )
elif [[ "${SEMANTIC_TARGET_NOISE:-0}" != 0 ]]; then
    echo "SEMANTIC_TARGET_NOISE must be 0 or 1." >&2
    exit 2
fi

export AUTO_OPEN_FIXED_ENTRY=run_auto_open.sh
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

cd "${REPO_ROOT}"
start_epoch="$(date +%s)"
gpu_count="${#GPU_IDS_ARRAY[@]}"
rank_count="${gpu_count}"
if (( rank_count > ${#SEED_ARRAY[@]} )); then
    rank_count="${#SEED_ARRAY[@]}"
fi
for ((rank=0; rank<rank_count; rank++)); do
    gpu_id="${GPU_IDS_ARRAY[rank]}"
    rank_seeds=()
    for ((offset=rank; offset<${#SEED_ARRAY[@]}; offset+=rank_count)); do
        rank_seeds+=("${SEED_ARRAY[offset]}")
    done
    CUDA_VISIBLE_DEVICES="${gpu_id}" python "${ENTRY}" \
        "${COMMON_RUNTIME_ARGS[@]}" "${COLLECTION_ARGS[@]}" \
        --seed "${rank_seeds[0]}" --episode_seeds "${rank_seeds[@]}" \
        --batched_envs "${#rank_seeds[@]}" \
        --steady_path_workers "${STEADY_PATH_WORKERS:-4}" \
        --standard_dataset_dir "${DATASET_ROOT}/shards" \
        --output_dir "${DATASET_ROOT}/logs/rank${rank}-control" \
        >"${DATASET_ROOT}/logs/rank${rank}.console.log" 2>&1 &
    PIDS+=("$!")
done

failures=0
for rank in "${!PIDS[@]}"; do
    wait "${PIDS[rank]}" || failures=$((failures + 1))
done
if (( failures != 0 )); then
    echo "AutoOpen collection rank failed; aggregation was skipped." >&2
    exit 1
fi

seeds_csv="$(IFS=,; echo "${SEED_ARRAY[*]}")"
elapsed=$(( $(date +%s) - start_epoch ))
DATASET_ROOT="${DATASET_ROOT}" SEEDS_CSV="${seeds_csv}" \
CAMERA_MODE="${CAMERA_MODE}" COLLECTION_WALL_TIME_S="${elapsed}" python - <<'PY'
import os
from pathlib import Path
from gr00t.rl.data_collection.auto_open_writer import aggregate_standard_shards

mode = os.environ["CAMERA_MODE"]
enabled = mode != "none"
aggregate_standard_shards(
    Path(os.environ["DATASET_ROOT"]),
    [int(value) for value in os.environ["SEEDS_CSV"].split(",")],
    camera_mount_position_noise_m=0.02 if enabled else 0.0,
    camera_mount_rotation_noise_deg=15.0 if enabled else 0.0,
    camera_focal_scale_noise_fraction=0.08 if enabled else 0.0,
    camera_principal_point_noise_px=5.0 if enabled else 0.0,
    camera_distortion_noise_fraction=0.10 if enabled else 0.0,
    collection_wall_time_s=int(os.environ["COLLECTION_WALL_TIME_S"]),
)
PY
echo "AutoOpen collection completed: ${DATASET_ROOT}/manifest.json"
trap - INT TERM EXIT
