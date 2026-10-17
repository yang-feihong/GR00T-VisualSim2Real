#!/usr/bin/env bash
set -euo pipefail

# Sole AutoOpen entry: canonical no-camera collection, strict selection, then
# camera-only trajectory replay into an official dataset release.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENTRY="${REPO_ROOT}/gr00t/rl/scripts/doorman_auto_open.py"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
MAX_STEPS="${MAX_STEPS:-4000}"
REPLAY_RENDER_BATCH_SIZE="${REPLAY_RENDER_BATCH_SIZE:-8}"
AUTO_OPEN_MODE="${AUTO_OPEN_MODE:-official}"
FORMAL_DATA_ROOT="${REPO_ROOT}/datasets/auto_open"
RELEASE_VERSION="v1"
RELEASE_STAMP="$(date +%Y%m%dT%H%M%S)"

source_fingerprint() {
    find \
        "${REPO_ROOT}/gr00t/rl/auto_open" \
        "${REPO_ROOT}/gr00t/rl/data_collection" \
        "${REPO_ROOT}/gr00t/rl/isaac_utils/playground/env_rand/door.py" \
        "${REPO_ROOT}/gr00t/rl/scripts/doorman_auto_open.py" \
        "${REPO_ROOT}/gr00t/rl/scripts/run_auto_open.sh" \
        "${REPO_ROOT}/gr00t/rl/third_party/visual_wholebody_lowlevel/legged_gym/envs" \
        "${REPO_ROOT}/gr00t/rl/third_party/visual_wholebody_lowlevel/legged_gym/scripts" \
        "${REPO_ROOT}/gr00t/rl/third_party/visual_wholebody_lowlevel/legged_gym/utils" \
        -type f \( -name '*.py' -o -name '*.json' -o -name '*.sh' \) -print0 \
        | sort -z \
        | xargs -0 sha256sum \
        | sha256sum \
        | cut -d' ' -f1
}

if (( $# != 0 )); then
    echo "The fixed AutoOpen workflow accepts no command-line arguments." >&2
    exit 2
fi
if ! command -v python >/dev/null 2>&1 || [[ ! -d /workspace/isaaclab ]]; then
    echo "Run this launcher inside the doorman_train runtime container." >&2
    exit 2
fi
if [[ -z "${SEED_START:-}" || -z "${SEED_END:-}" ]]; then
    echo "Set the inclusive SEED_START and SEED_END range." >&2
    exit 2
fi
if [[ ! "${SEED_START}" =~ ^[0-9]+$ || ! "${SEED_END}" =~ ^[0-9]+$ ]]; then
    echo "SEED_START and SEED_END must be non-negative integers." >&2
    exit 2
fi
if (( SEED_END < SEED_START )); then
    echo "SEED_END must be greater than or equal to SEED_START." >&2
    exit 2
fi
if [[ ! "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_STEPS must be a positive integer." >&2
    exit 2
fi
if [[ ! "${REPLAY_RENDER_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPLAY_RENDER_BATCH_SIZE must be a positive integer." >&2
    exit 2
fi
if [[ "${AUTO_OPEN_MODE}" != official && "${AUTO_OPEN_MODE}" != debug ]]; then
    echo "AUTO_OPEN_MODE must be official or debug." >&2
    exit 2
fi

IFS=',' read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS_CSV}"
if (( ${#GPU_IDS_ARRAY[@]} < 1 )); then
    echo "GPU_IDS must contain at least one physical GPU." >&2
    exit 2
fi
for gpu_id in "${GPU_IDS_ARRAY[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "GPU_IDS must contain comma-separated non-negative integers." >&2
        exit 2
    fi
done

declare -a ALL_SEEDS=()
for ((seed=SEED_START; seed<=SEED_END; seed++)); do
    ALL_SEEDS+=("${seed}")
done

range_tag="seeds${SEED_START}-${SEED_END}"
SOURCE_FINGERPRINT="$(source_fingerprint)"
mkdir -p "${FORMAL_DATA_ROOT}/canonical" "${FORMAL_DATA_ROOT}/debug" \
    "${FORMAL_DATA_ROOT}/releases" "${FORMAL_DATA_ROOT}/work"

export AUTO_OPEN_FIXED_ENTRY=run_auto_open.sh
declare -a ACTIVE_PIDS=()
cleanup() {
    local pid
    for pid in "${ACTIVE_PIDS[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup INT TERM EXIT

run_phase() {
    local dataset_root="$1"
    local camera_mode="$2"
    local replay_source="$3"
    local seed_array_name="$4"
    local -n phase_seeds="${seed_array_name}"
    local phase_start rank_count rank gpu_id offset failures elapsed seeds_csv
    local -a common_args collection_args rank_seeds

    if [[ -e "${dataset_root}" ]]; then
        echo "Refusing to overwrite phase output: ${dataset_root}" >&2
        return 2
    fi
    mkdir -p "${dataset_root}/shards" "${dataset_root}/logs"
    common_args=(
        --headless
        --device cuda:0
        --max_steps "${MAX_STEPS}"
        --batch_contact_ik_backend exact_workers
        --batch_path_backend exact_workers
        --batch_panel_clearance_backend gpu
        --batch_geometry_fallback_backend numpy_batch
        --summary_only
        --collection_camera_mode "${camera_mode}"
    )
    if [[ "${camera_mode}" == none ]]; then
        collection_args=(--disable_rgb_cameras)
    else
        collection_args=(
            --camera_mount_position_noise_m 0.02
            --camera_mount_rotation_noise_deg 15.0
            --camera_focal_scale_noise_fraction 0.08
            --camera_principal_point_noise_px 5.0
            --camera_distortion_noise_fraction 0.10
        )
        if [[ "${camera_mode}" == dual ]]; then
            collection_args+=(
                --replay_source_dataset_dir "${replay_source}"
                --replay_render_batch_size "${REPLAY_RENDER_BATCH_SIZE}"
            )
        fi
    fi

    phase_start="$(date +%s)"
    rank_count="${#GPU_IDS_ARRAY[@]}"
    if (( rank_count > ${#phase_seeds[@]} )); then
        rank_count="${#phase_seeds[@]}"
    fi
    ACTIVE_PIDS=()
    cd "${REPO_ROOT}"
    for ((rank=0; rank<rank_count; rank++)); do
        gpu_id="${GPU_IDS_ARRAY[rank]}"
        rank_seeds=()
        for ((offset=rank; offset<${#phase_seeds[@]}; offset+=rank_count)); do
            rank_seeds+=("${phase_seeds[offset]}")
        done
        CUDA_VISIBLE_DEVICES="${gpu_id}" python "${ENTRY}" \
            "${common_args[@]}" "${collection_args[@]}" \
            --seed "${rank_seeds[0]}" --episode_seeds "${rank_seeds[@]}" \
            --batched_envs "${#rank_seeds[@]}" \
            --steady_path_workers "${STEADY_PATH_WORKERS:-4}" \
            --standard_dataset_dir "${dataset_root}/shards" \
            --output_dir "${dataset_root}/logs/rank${rank}-control" \
            >"${dataset_root}/logs/rank${rank}.console.log" 2>&1 &
        ACTIVE_PIDS+=("$!")
    done

    failures=0
    for rank in "${!ACTIVE_PIDS[@]}"; do
        wait "${ACTIVE_PIDS[rank]}" || failures=$((failures + 1))
    done
    ACTIVE_PIDS=()
    if (( failures != 0 )); then
        echo "AutoOpen ${camera_mode} phase failed; aggregation was skipped." >&2
        return 1
    fi

    seeds_csv="$(IFS=,; echo "${phase_seeds[*]}")"
    elapsed=$(( $(date +%s) - phase_start ))
    PHASE_OUTPUT_ROOT="${dataset_root}" PHASE_SEEDS_CSV="${seeds_csv}" \
    PHASE_CAMERAS="${camera_mode}" COLLECTION_WALL_TIME_S="${elapsed}" python - <<'PY'
import os
from pathlib import Path
from gr00t.rl.data_collection.auto_open_writer import aggregate_standard_shards

mode = os.environ["PHASE_CAMERAS"]
enabled = mode != "none"
aggregate_standard_shards(
    Path(os.environ["PHASE_OUTPUT_ROOT"]),
    [int(value) for value in os.environ["PHASE_SEEDS_CSV"].split(",")],
    camera_mount_position_noise_m=0.02 if enabled else 0.0,
    camera_mount_rotation_noise_deg=15.0 if enabled else 0.0,
    camera_focal_scale_noise_fraction=0.08 if enabled else 0.0,
    camera_principal_point_noise_px=5.0 if enabled else 0.0,
    camera_distortion_noise_fraction=0.10 if enabled else 0.0,
    collection_wall_time_s=int(os.environ["COLLECTION_WALL_TIME_S"]),
)
PY
    echo "AutoOpen ${camera_mode} phase completed: ${dataset_root}/manifest.json"
}

if [[ "${AUTO_OPEN_MODE}" == debug ]]; then
    debug_work_root="${FORMAL_DATA_ROOT}/work/debug_${range_tag}_${RELEASE_STAMP}_$$"
    debug_pending_root="${debug_work_root}/six_camera.pending"
    mkdir -p "${debug_work_root}"
    echo "[workflow] debug: six-camera collection (${#ALL_SEEDS[@]} seeds)"
    run_phase "${debug_pending_root}" six "" ALL_SEEDS
    debug_accepted_count="$(SOURCE_FINGERPRINT="${SOURCE_FINGERPRINT}" \
        SEED_START="${SEED_START}" SEED_END="${SEED_END}" \
        python - "${debug_pending_root}/acceptance_report.json" <<'PY'
import json
import os
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
report = json.loads(report_path.read_text(encoding="utf-8"))
accepted = sorted(
    int(row["seed"])
    for row in report["episodes"]
    if bool(row.get("accepted"))
)
if len(accepted) != int(report.get("accepted_episodes", -1)):
    raise SystemExit("debug accepted seed extraction disagrees with aggregate report")
(report_path.parent / "accepted_seeds.txt").write_text(
    "".join(f"{seed}\n" for seed in accepted), encoding="utf-8"
)
(report_path.parent / "workflow.json").write_text(
    json.dumps(
        {
            "workflow": "auto_open_debug_six_camera_v1",
            "seed_range": {
                "start": int(os.environ["SEED_START"]),
                "end": int(os.environ["SEED_END"]),
                "inclusive": True,
            },
            "source_fingerprint_sha256": os.environ["SOURCE_FINGERPRINT"],
            "requested_episodes": int(report["total_episodes"]),
            "accepted_episodes": len(accepted),
            "camera_mode": "six",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(len(accepted))
PY
)"
    debug_name="auto_open_debug_${RELEASE_VERSION}_${range_tag}_episodes${debug_accepted_count}_six_camera_${RELEASE_STAMP}"
    debug_root="${FORMAL_DATA_ROOT}/debug/${debug_name}"
    if [[ -e "${debug_root}" ]]; then
        echo "Refusing to overwrite debug dataset: ${debug_root}" >&2
        exit 2
    fi
    mv "${debug_pending_root}" "${debug_root}"
    rmdir "${debug_work_root}"
    echo "[workflow] debug strict acceptance: ${debug_accepted_count}/${#ALL_SEEDS[@]}"
    echo "[workflow] debug dataset completed: ${debug_root}"
    trap - INT TERM EXIT
    exit 0
fi

canonical_name="auto_open_${RELEASE_VERSION}_${range_tag}_canonical_no_camera_${RELEASE_STAMP}"
canonical_root="${FORMAL_DATA_ROOT}/canonical/${canonical_name}"
work_root="${FORMAL_DATA_ROOT}/work/${range_tag}_${RELEASE_STAMP}_attempt_$$"
accepted_seed_file="${work_root}/accepted_seeds.txt"
if [[ -e "${work_root}" ]]; then
    echo "Refusing to overwrite workflow work directory: ${work_root}" >&2
    exit 2
fi
mkdir -p "${work_root}"

echo "[workflow] phase 1/2: canonical no-camera collection (${#ALL_SEEDS[@]} seeds)"
run_phase "${canonical_root}" none "" ALL_SEEDS
SOURCE_FINGERPRINT="${SOURCE_FINGERPRINT}" SEED_START="${SEED_START}" \
SEED_END="${SEED_END}" python - "${canonical_root}/workflow_source.json" <<'PY'
import json
import os
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "workflow": "auto_open_official_two_phase_v1",
            "seed_range": {
                "start": int(os.environ["SEED_START"]),
                "end": int(os.environ["SEED_END"]),
                "inclusive": True,
            },
            "source_fingerprint_sha256": os.environ["SOURCE_FINGERPRINT"],
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

python - "${canonical_root}/acceptance_report.json" "${accepted_seed_file}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
accepted = sorted(
    int(row["seed"])
    for row in report["episodes"]
    if bool(row.get("accepted"))
)
if len(accepted) != int(report.get("accepted_episodes", -1)):
    raise SystemExit("accepted seed extraction disagrees with aggregate report")
if not accepted:
    raise SystemExit("no canonical episode passed strict acceptance")
Path(sys.argv[2]).write_text(
    "".join(f"{seed}\n" for seed in accepted), encoding="utf-8"
)
print(f"[workflow] strict canonical acceptance: {len(accepted)}/{report['total_episodes']}")
PY
mapfile -t ACCEPTED_SEEDS < "${accepted_seed_file}"

accepted_count="${#ACCEPTED_SEEDS[@]}"
release_name="auto_open_${RELEASE_VERSION}_${range_tag}_episodes${accepted_count}_dual_camera_${RELEASE_STAMP}"
pending_root="${work_root}/${release_name}.pending"
release_root="${FORMAL_DATA_ROOT}/releases/${release_name}"
echo "[workflow] phase 2/2: dual-camera replay (${accepted_count} accepted seeds)"
run_phase "${pending_root}" dual "${canonical_root}" ACCEPTED_SEEDS

python - "${pending_root}/acceptance_report.json" "${canonical_root}/acceptance_report.json" \
    "${accepted_count}" "${SEED_START}" "${SEED_END}" "${canonical_root}" \
    "${SOURCE_FINGERPRINT}" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
canonical_report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected = int(sys.argv[3])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("total_episodes") != expected:
    raise SystemExit("camera release episode count differs from selected canonical count")
if report.get("accepted_episodes") != expected or report.get("quarantined_episodes") != 0:
    raise SystemExit("camera release did not pass strict acceptance in full")
workflow = {
    "workflow": "auto_open_official_two_phase_v1",
    "seed_range": {"start": int(sys.argv[4]), "end": int(sys.argv[5]), "inclusive": True},
    "canonical_source": sys.argv[6],
    "source_fingerprint_sha256": sys.argv[7],
    "canonical_requested_episodes": int(canonical_report["total_episodes"]),
    "canonical_accepted_episodes": int(canonical_report["accepted_episodes"]),
    "released_episodes": expected,
    "camera_mode": "dual",
    "control_or_physics_rerun_for_images": False,
}
(report_path.parent / "workflow.json").write_text(
    json.dumps(workflow, indent=2) + "\n", encoding="utf-8"
)
PY

mv "${pending_root}" "${release_root}"
rm -f "${accepted_seed_file}"
rmdir "${work_root}"
echo "[workflow] official release completed: ${release_root}"
trap - INT TERM EXIT
