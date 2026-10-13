#!/usr/bin/env bash
set -euo pipefail

# Sole maintained AutoOpen experiment entry. Select one reviewed workflow with
# AUTO_OPEN_WORKFLOW; arbitrary Python arguments are intentionally unsupported.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENTRY="${REPO_ROOT}/gr00t/rl/scripts/doorman_auto_open.py"
WORKFLOW="${AUTO_OPEN_WORKFLOW:-capability}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
MAX_STEPS="${MAX_STEPS:-4000}"

if (( $# != 0 )); then
    echo "The fixed AutoOpen launcher does not accept command-line arguments." >&2
    exit 2
fi
if ! command -v python >/dev/null 2>&1 || [[ ! -d /workspace/isaaclab ]]; then
    echo "Run this launcher inside the doorman_train runtime container." >&2
    echo "Example: docker exec doorman_train bash -lc 'cd /workspace/GR00T-VisualSim2Real && AUTO_OPEN_WORKFLOW=${WORKFLOW} ... bash gr00t/rl/scripts/run_auto_open.sh'" >&2
    exit 2
fi
if [[ ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_STEPS must be a positive integer." >&2
    exit 2
fi

IFS=',' read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS_CSV}"
if (( ${#GPU_IDS_ARRAY[@]} < 1 )); then
    echo "GPU_IDS must contain at least one physical GPU." >&2
    exit 2
fi

declare -a PIDS=()
declare -a COMMON_RUNTIME_ARGS=(
    --headless
    --device cuda:0
    --max_steps "${MAX_STEPS}"
    --batch_contact_ik_backend exact_workers
    --batch_path_backend exact_workers
    --batch_panel_clearance_backend gpu
    --batch_geometry_fallback_backend numpy_batch
)
export AUTO_OPEN_FIXED_ENTRY=run_auto_open.sh

run_auto_open_python() {
    python "${ENTRY}" "${COMMON_RUNTIME_ARGS[@]}" "$@"
}

cleanup() {
    local pid
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup INT TERM EXIT

run_capability() {
    local preset="$1"
    local output_root="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new result directory}"
    local seed_start="${SEED_START:-1700}"
    local envs_per_gpu="${ENVS_PER_GPU:-250}"
    local steady_workers="${STEADY_PATH_WORKERS:-4}"
    local -a preset_args=()
    if [[ "${preset}" == semantic_collection ]]; then
        preset_args=(
            --target_noise approach_normal_m=0.03
            --target_noise approach_tangent_m=0.03
            --target_noise panel_vertical_m=0.05
            --target_noise through_tangent_m=0.08
            --target_noise through_yaw_rad=0.06
        )
    fi
    if [[ ! "${seed_start}" =~ ^[0-9]+$ || ! "${envs_per_gpu}" =~ ^[1-9][0-9]*$ ]]; then
        echo "SEED_START and ENVS_PER_GPU must be non-negative/positive integers." >&2
        exit 2
    fi
    if [[ -e "${output_root}" ]]; then
        echo "Refusing to overwrite existing OUTPUT_ROOT: ${output_root}" >&2
        exit 2
    fi
    mkdir -p "${output_root}"
    local start_epoch="$(date +%s)"
    local rank gpu_id rank_seed rank_dir console_log
    for rank in "${!GPU_IDS_ARRAY[@]}"; do
        gpu_id="${GPU_IDS_ARRAY[rank]}"
        rank_seed=$((seed_start + rank * envs_per_gpu))
        rank_dir="${output_root}/rank${rank}"
        console_log="${output_root}/rank${rank}.console.log"
        echo "Starting rank ${rank}: GPU ${gpu_id}, seeds ${rank_seed}-$((rank_seed + envs_per_gpu - 1))"
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
            run_auto_open_python --disable_rgb_cameras --summary_only \
            --seed "${rank_seed}" --batched_envs "${envs_per_gpu}" \
            --steady_path_workers "${steady_workers}" \
            --output_dir "${rank_dir}" "${preset_args[@]}" \
            >"${console_log}" 2>&1 &
        PIDS+=("$!")
    done

    local failures=0
    for rank in "${!PIDS[@]}"; do
        if ! wait "${PIDS[rank]}"; then
            echo "Rank ${rank} failed; inspect ${output_root}/rank${rank}.console.log" >&2
            failures=$((failures + 1))
        fi
    done
    for rank in "${!PIDS[@]}"; do
        rank_seed=$((seed_start + rank * envs_per_gpu))
        rank_dir="${output_root}/rank${rank}"
        console_log="${output_root}/rank${rank}.console.log"
        if grep -Eq 'Traceback \(most recent call last\)|^[A-Za-z]+Error:' "${console_log}"; then
            failures=$((failures + 1))
            continue
        fi
        if ! python - "${rank_dir}" "${rank_seed}" "${envs_per_gpu}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = set(range(int(sys.argv[2]), int(sys.argv[2]) + int(sys.argv[3])))
paths = list(root.glob("seed_*/metadata.json"))
seeds = [int(json.loads(path.read_text())["seed"]) for path in paths]
actual = set(seeds)
if len(paths) != len(expected) or len(seeds) != len(actual) or actual != expected:
    raise SystemExit(1)
PY
        then
            echo "Rank ${rank} metadata is incomplete or duplicated." >&2
            failures=$((failures + 1))
        fi
    done
    local elapsed=$(( $(date +%s) - start_epoch ))
    printf 'workflow=%s\nexit_failures=%d\nelapsed_s=%d\n' \
        "${preset}" "${failures}" "${elapsed}" >"${output_root}/launcher_status.txt"
    (( failures == 0 )) || exit 1
    echo "Completed $((envs_per_gpu * ${#GPU_IDS_ARRAY[@]})) environments in ${elapsed} s."
}

run_video_regression() {
    local run_name="${RUN_NAME:?Set RUN_NAME to a unique result name}"
    local seeds_csv="${SEEDS:-1700,1701,1702,1703}"
    local -a seeds output_dirs
    IFS=',' read -r -a seeds <<< "${seeds_csv}"
    if [[ "${#GPU_IDS_ARRAY[@]}" -ne "${#seeds[@]}" ]]; then
        echo "GPU_IDS and SEEDS must contain the same number of entries." >&2
        exit 2
    fi
    local index seed gpu_id output_dir console_log
    for index in "${!seeds[@]}"; do
        seed="${seeds[index]}"
        gpu_id="${GPU_IDS_ARRAY[index]}"
        output_dir="logs_rl/${run_name}_seed${seed}_video"
        console_log="${output_dir}.console.log"
        if [[ -e "${output_dir}" || -e "${console_log}" ]]; then
            echo "Refusing to overwrite existing output: ${output_dir}" >&2
            exit 2
        fi
        output_dirs+=("${output_dir}")
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
            run_auto_open_python --enable_cameras --record_video \
            --seed "${seed}" --batched_envs 1 \
            --output_dir "${output_dir}" --steady_path_workers 1 \
            >"${console_log}" 2>&1 &
        PIDS+=("$!")
    done
    local failures=0 metadata
    for index in "${!PIDS[@]}"; do
        if ! wait "${PIDS[index]}"; then
            failures=$((failures + 1))
            continue
        fi
        metadata="${output_dirs[index]}/seed_${seeds[index]}/metadata.json"
        if [[ ! -f "${metadata}" ]] || ! python -c \
            'import json,sys; raise SystemExit(not json.load(open(sys.argv[1]))["success"])' \
            "${metadata}"; then
            failures=$((failures + 1))
        fi
    done
    (( failures == 0 )) || exit 1
    echo "Video regression completed for ${#seeds[@]} seeds."
}

run_standard_collection() {
    local start_epoch="$(date +%s)"
    local dataset_root="${DATASET_ROOT:?Set DATASET_ROOT to a new dataset directory}"
    local seed_start="${SEED_START:-1700}"
    local requested_count="${NUM_EPISODES:-4}"
    local -a standard_seeds=()
    if [[ -n "${SEEDS:-}" ]]; then
        IFS=',' read -r -a standard_seeds <<< "${SEEDS}"
    else
        local offset
        for ((offset=0; offset<requested_count; offset++)); do
            standard_seeds+=("$((seed_start + offset))")
        done
    fi
    local count="${#standard_seeds[@]}"
    if (( count < 1 )); then
        echo "Standard collection requires at least one seed." >&2
        exit 2
    fi
    local camera_mount_position_noise_m=0.02
    local camera_mount_rotation_noise_deg=15.0
    local camera_focal_scale_noise_fraction=0.08
    local camera_principal_point_noise_px=5.0
    local camera_distortion_noise_fraction=0.10
    if [[ -e "${dataset_root}" ]]; then
        echo "Refusing to overwrite existing DATASET_ROOT: ${dataset_root}" >&2
        exit 2
    fi
    mkdir -p "${dataset_root}/shards" "${dataset_root}/logs"
    local gpu_count="${#GPU_IDS_ARRAY[@]}" index gpu_id
    for index in "${!GPU_IDS_ARRAY[@]}"; do
        gpu_id="${GPU_IDS_ARRAY[index]}"
        (
            local worker_failures=0 offset seed name shard control log
            for ((offset=index; offset<count; offset+=gpu_count)); do
                seed="${standard_seeds[offset]}"; name="episode-seed${seed}"
                shard="${dataset_root}/shards/${name}"
                control="${dataset_root}/logs/${name}-control"
                log="${dataset_root}/logs/${name}.console.log"
                if ! CUDA_VISIBLE_DEVICES="${gpu_id}" \
                    run_auto_open_python --seed "${seed}" \
                    --batched_envs 1 --summary_only --steady_path_workers 1 \
                    --standard_dataset_dir "${shard}" \
                    --camera_mount_position_noise_m "${camera_mount_position_noise_m}" \
                    --camera_mount_rotation_noise_deg "${camera_mount_rotation_noise_deg}" \
                    --camera_focal_scale_noise_fraction "${camera_focal_scale_noise_fraction}" \
                    --camera_principal_point_noise_px "${camera_principal_point_noise_px}" \
                    --camera_distortion_noise_fraction "${camera_distortion_noise_fraction}" \
                    --output_dir "${control}" >"${log}" 2>&1; then
                    worker_failures=$((worker_failures + 1))
                fi
            done
            exit "${worker_failures}"
        ) &
        PIDS+=("$!")
    done
    local failures=0
    for index in "${!PIDS[@]}"; do
        wait "${PIDS[index]}" || failures=$((failures + 1))
    done
    if (( failures != 0 )); then
        echo "Standard collection workers failed; shard aggregation was skipped." >&2
        exit 1
    fi
    local standard_seeds_csv elapsed
    standard_seeds_csv="$(IFS=,; echo "${standard_seeds[*]}")"
    elapsed=$(( $(date +%s) - start_epoch ))
    DATASET_ROOT="${dataset_root}" SEEDS_CSV="${standard_seeds_csv}" \
    CAMERA_MOUNT_POSITION_NOISE_M="${camera_mount_position_noise_m}" \
    CAMERA_MOUNT_ROTATION_NOISE_DEG="${camera_mount_rotation_noise_deg}" \
    CAMERA_FOCAL_SCALE_NOISE_FRACTION="${camera_focal_scale_noise_fraction}" \
    CAMERA_PRINCIPAL_POINT_NOISE_PX="${camera_principal_point_noise_px}" \
    CAMERA_DISTORTION_NOISE_FRACTION="${camera_distortion_noise_fraction}" \
    COLLECTION_WALL_TIME_S="${elapsed}" python - <<'PY'
import os
from pathlib import Path

from gr00t.rl.data_collection.auto_open_writer import aggregate_standard_shards

root = Path(os.environ["DATASET_ROOT"])
seeds = [int(value) for value in os.environ["SEEDS_CSV"].split(",")]
aggregate_standard_shards(
    root,
    seeds,
    camera_mount_position_noise_m=float(
        os.environ["CAMERA_MOUNT_POSITION_NOISE_M"]
    ),
    camera_mount_rotation_noise_deg=float(
        os.environ["CAMERA_MOUNT_ROTATION_NOISE_DEG"]
    ),
    camera_focal_scale_noise_fraction=float(
        os.environ["CAMERA_FOCAL_SCALE_NOISE_FRACTION"]
    ),
    camera_principal_point_noise_px=float(
        os.environ["CAMERA_PRINCIPAL_POINT_NOISE_PX"]
    ),
    camera_distortion_noise_fraction=float(
        os.environ["CAMERA_DISTORTION_NOISE_FRACTION"]
    ),
    collection_wall_time_s=int(os.environ["COLLECTION_WALL_TIME_S"]),
)
PY
    echo "Standard collection completed: ${dataset_root}/manifest.json"
}

cd "${REPO_ROOT}"
case "${WORKFLOW}" in
    capability) run_capability baseline ;;
    semantic_collection) run_capability semantic_collection ;;
    video_regression) run_video_regression ;;
    standard_collection) run_standard_collection ;;
    *)
        echo "Unknown AUTO_OPEN_WORKFLOW=${WORKFLOW}. Expected capability, semantic_collection, video_regression, or standard_collection." >&2
        exit 2
        ;;
esac

trap - INT TERM EXIT
