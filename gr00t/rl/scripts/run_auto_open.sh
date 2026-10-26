#!/usr/bin/env bash
set -euo pipefail

# Sole AutoOpen entry: canonical no-camera collection, strict selection, then
# camera-only trajectory replay into an official dataset release.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENTRY="${REPO_ROOT}/gr00t/rl/scripts/doorman_auto_open.py"

# On the host, supervise docker exec with a filesystem lease. If the caller is
# killed or disconnected, the heartbeat stops and the in-container watchdog
# terminates the complete workflow process tree.
if [[ ! -d /workspace/isaaclab ]]; then
    CONTAINER_NAME="${AUTO_OPEN_CONTAINER:-doorman_train}"
    LEASE_DIR="${REPO_ROOT}/datasets/auto_open/.launcher_leases"
    LEASE_ID="run-$$_$(date +%s%N)"
    HOST_LEASE="${LEASE_DIR}/${LEASE_ID}"
    CONTAINER_LEASE="/workspace/GR00T-VisualSim2Real/datasets/auto_open/.launcher_leases/${LEASE_ID}"
    mkdir -p "${LEASE_DIR}"
    : >"${HOST_LEASE}"
    HOST_OWNER_PID="$$"
    HOST_DOCKER_PID=""
    host_cleanup() {
        trap - HUP INT TERM EXIT
        rm -f "${HOST_LEASE}"
        if [[ -n "${HOST_DOCKER_PID}" ]]; then
            kill "${HOST_DOCKER_PID}" 2>/dev/null || true
            wait "${HOST_DOCKER_PID}" 2>/dev/null || true
        fi
    }
    trap host_cleanup HUP INT TERM EXIT
    (
        while kill -0 "${HOST_OWNER_PID}" 2>/dev/null; do
            touch "${HOST_LEASE}"
            sleep 2
        done
        rm -f "${HOST_LEASE}"
    ) &
    HEARTBEAT_PID="$!"
    docker_args=(exec -e "AUTO_OPEN_LAUNCH_LEASE=${CONTAINER_LEASE}")
    for name in SEED_START SEED_END AUTO_OPEN_MODE DOOR_MOTION GPU_IDS \
        STOP_AFTER_STAGE MAX_STEPS REPLAY_RENDER_BATCH_SIZE PULL_PANEL_CONTACT_BASELINE; do
        if [[ -v "${name}" ]]; then
            docker_args+=(-e "${name}=${!name}")
        fi
    done
    docker_args+=("${CONTAINER_NAME}" bash "/workspace/GR00T-VisualSim2Real/gr00t/rl/scripts/run_auto_open.sh")
    docker "${docker_args[@]}" &
    HOST_DOCKER_PID="$!"
    status=0
    wait "${HOST_DOCKER_PID}" || status="$?"
    HOST_DOCKER_PID=""
    kill "${HEARTBEAT_PID}" 2>/dev/null || true
    wait "${HEARTBEAT_PID}" 2>/dev/null || true
    rm -f "${HOST_LEASE}"
    trap - HUP INT TERM EXIT
    exit "${status}"
fi

# The runtime container exports the primary checkout on PYTHONPATH.  Pin this
# launcher's own checkout first so an isolated worktree cannot mix modules
# from two source trees.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
REPLAY_RENDER_BATCH_SIZE="${REPLAY_RENDER_BATCH_SIZE:-8}"
AUTO_OPEN_MODE="${AUTO_OPEN_MODE:-official}"
DOOR_MOTION="${DOOR_MOTION:-push}"
PULL_PANEL_CONTACT_BASELINE="${PULL_PANEL_CONTACT_BASELINE:-trunk_front_minus10}"
if [[ "${DOOR_MOTION}" == pull ]]; then
    MAX_STEPS="${MAX_STEPS:-16000}"
else
    MAX_STEPS="${MAX_STEPS:-4000}"
fi
if [[ -z "${STOP_AFTER_STAGE:-}" ]]; then
    STOP_AFTER_STAGE=PASS_THROUGH_DOORWAY
fi
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
if [[ "${AUTO_OPEN_MODE}" != official && "${AUTO_OPEN_MODE}" != canonical \
      && "${AUTO_OPEN_MODE}" != debug ]]; then
    echo "AUTO_OPEN_MODE must be official, canonical, or debug." >&2
    exit 2
fi
if [[ "${DOOR_MOTION}" != push && "${DOOR_MOTION}" != pull ]]; then
    echo "DOOR_MOTION must be push or pull." >&2
    exit 2
fi
if [[ "${STOP_AFTER_STAGE}" != PASS_THROUGH_DOORWAY && "${STOP_AFTER_STAGE}" != STOW_ARM_AND_CLOSE_GRIPPER \
      && "${STOP_AFTER_STAGE}" != MOVE_TO_GAP_STATION && "${STOP_AFTER_STAGE}" != INSERT_INTO_DOOR_GAP \
      && "${STOP_AFTER_STAGE}" != PRY_DOOR_OPEN && "${STOP_AFTER_STAGE}" != FOLLOW_WALL_TO_PANEL ]]; then
    echo "STOP_AFTER_STAGE must be PASS_THROUGH_DOORWAY, STOW_ARM_AND_CLOSE_GRIPPER, MOVE_TO_GAP_STATION, INSERT_INTO_DOOR_GAP, PRY_DOOR_OPEN, or FOLLOW_WALL_TO_PANEL." >&2
    exit 2
fi
if [[ "${DOOR_MOTION}" == push && ( "${STOP_AFTER_STAGE}" == MOVE_TO_GAP_STATION \
      || "${STOP_AFTER_STAGE}" == INSERT_INTO_DOOR_GAP \
      || "${STOP_AFTER_STAGE}" == PRY_DOOR_OPEN \
      || "${STOP_AFTER_STAGE}" == FOLLOW_WALL_TO_PANEL ) ]]; then
    echo "Gap-station, gap-insertion, prying, and wall-following terminal stages require DOOR_MOTION=pull." >&2
    exit 2
fi
if [[ "${AUTO_OPEN_MODE}" == official && "${STOP_AFTER_STAGE}" != PASS_THROUGH_DOORWAY ]]; then
    echo "Partial workflows cannot be published as official datasets; use canonical or debug." >&2
    exit 2
fi
if [[ "${DOOR_MOTION}" == push ]]; then
    DOOR_OPEN_IO=in
else
    DOOR_OPEN_IO=out
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
ACTIVE_MONITOR_PID=""
LEASE_WATCHDOG_PID=""
cleanup() {
    local pid deadline any_alive
    trap - HUP INT PIPE TERM EXIT
    if [[ -n "${LEASE_WATCHDOG_PID}" ]]; then
        kill "${LEASE_WATCHDOG_PID}" 2>/dev/null || true
        wait "${LEASE_WATCHDOG_PID}" 2>/dev/null || true
        LEASE_WATCHDOG_PID=""
    fi
    if [[ -n "${ACTIVE_MONITOR_PID}" ]]; then
        kill "${ACTIVE_MONITOR_PID}" 2>/dev/null || true
        wait "${ACTIVE_MONITOR_PID}" 2>/dev/null || true
        ACTIVE_MONITOR_PID=""
    fi
    for pid in "${ACTIVE_PIDS[@]:-}"; do
        kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    done
    deadline=$((SECONDS + 5))
    while (( SECONDS < deadline )); do
        any_alive=false
        for pid in "${ACTIVE_PIDS[@]:-}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                any_alive=true
                break
            fi
        done
        [[ "${any_alive}" == false ]] && break
        sleep 0.1
    done
    for pid in "${ACTIVE_PIDS[@]:-}"; do
        kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    done
    ACTIVE_PIDS=()
}
trap cleanup HUP INT PIPE TERM EXIT

if [[ -n "${AUTO_OPEN_LAUNCH_LEASE:-}" ]]; then
    LAUNCHER_PID="$$"
    (
        while kill -0 "${LAUNCHER_PID}" 2>/dev/null; do
            if [[ ! -e "${AUTO_OPEN_LAUNCH_LEASE}" ]]; then
                kill -TERM "${LAUNCHER_PID}" 2>/dev/null || true
                exit 0
            fi
            now="$(date +%s)"
            modified="$(stat -c %Y "${AUTO_OPEN_LAUNCH_LEASE}" 2>/dev/null || echo 0)"
            if (( now - modified > 8 )); then
                kill -TERM "${LAUNCHER_PID}" 2>/dev/null || true
                exit 0
            fi
            sleep 2
        done
    ) &
    LEASE_WATCHDOG_PID="$!"
fi

run_phase() {
    local dataset_root="$1"
    local camera_mode="$2"
    local replay_source="$3"
    local seed_array_name="$4"
    local -n phase_seeds="${seed_array_name}"
    local phase_start rank_count rank gpu_id offset failures elapsed seeds_csv
    local rank_status phase_status report_path
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
        --summary_only
        --collection_camera_mode "${camera_mode}"
        --doorman_door_open_io "${DOOR_OPEN_IO}"
        --terminal_stage "${STOP_AFTER_STAGE}"
        --pull_panel_contact_baseline "${PULL_PANEL_CONTACT_BASELINE}"
    )
    if [[ "${AUTO_OPEN_MODE}" == debug ]]; then
        common_args+=(--debug_trace)
    fi
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
        setsid --wait env CUDA_VISIBLE_DEVICES="${gpu_id}" python "${ENTRY}" \
            "${common_args[@]}" "${collection_args[@]}" \
            --seed "${rank_seeds[0]}" --episode_seeds "${rank_seeds[@]}" \
            --batched_envs "${#rank_seeds[@]}" \
            --standard_dataset_dir "${dataset_root}/shards" \
            --output_dir "${dataset_root}/logs/rank${rank}-control" \
            >"${dataset_root}/logs/rank${rank}.console.log" 2>&1 &
        ACTIVE_PIDS+=("$!")
    done

    (
        last_snapshot=""
        while true; do
            any_alive=false
            for pid in "${ACTIVE_PIDS[@]}"; do
                if kill -0 "${pid}" 2>/dev/null; then
                    any_alive=true
                    break
                fi
            done
            snapshot="$(
                for ((progress_rank=0; progress_rank<rank_count; progress_rank++)); do
                    progress_file="${dataset_root}/logs/rank${progress_rank}-control/progress.txt"
                    if [[ -s "${progress_file}" ]]; then
                        sed "s/^/[rank${progress_rank}] /" "${progress_file}"
                    fi
                done
            )"
            if [[ -n "${snapshot}" && "${snapshot}" != "${last_snapshot}" ]]; then
                echo "[workflow][progress] wall_s=$(( $(date +%s) - phase_start ))"
                printf '%s\n' "${snapshot}"
                last_snapshot="${snapshot}"
            fi
            [[ "${any_alive}" == false ]] && break
            sleep 2
        done
    ) &
    ACTIVE_MONITOR_PID="$!"

    failures=0
    phase_status=0
    for rank in "${!ACTIVE_PIDS[@]}"; do
        if wait "${ACTIVE_PIDS[rank]}"; then
            if [[ "${camera_mode}" != dual && ! -s "${dataset_root}/logs/rank${rank}-control/summary.json" ]]; then
                echo "AutoOpen rank=${rank} gpu=${GPU_IDS_ARRAY[rank]} exited 0 without completed summary: ${dataset_root}/logs/rank${rank}-control/summary.json; log=${dataset_root}/logs/rank${rank}.console.log" >&2
                failures=$((failures + 1))
                if (( phase_status == 0 )); then phase_status=1; fi
            fi
            # A clean process exit must still have finalized every assigned shard.
            for ((offset=rank; offset<${#phase_seeds[@]}; offset+=rank_count)); do
                report_path="${dataset_root}/shards/episode-seed${phase_seeds[offset]}/meta/acceptance_report.json"
                if [[ ! -s "${report_path}" ]]; then
                    echo "AutoOpen rank=${rank} gpu=${GPU_IDS_ARRAY[rank]} exited 0 without finalized shard: ${report_path}; log=${dataset_root}/logs/rank${rank}.console.log" >&2
                    failures=$((failures + 1))
                    if (( phase_status == 0 )); then phase_status=1; fi
                fi
            done
        else
            rank_status="$?"
            failures=$((failures + 1))
            if (( phase_status == 0 )); then phase_status="${rank_status}"; fi
            echo "AutoOpen rank=${rank} gpu=${GPU_IDS_ARRAY[rank]} pid=${ACTIVE_PIDS[rank]} failed with exit=${rank_status}; log=${dataset_root}/logs/rank${rank}.console.log" >&2
        fi
    done
    wait "${ACTIVE_MONITOR_PID}" 2>/dev/null || true
    ACTIVE_MONITOR_PID=""
    ACTIVE_PIDS=()
    if (( failures != 0 )); then
        echo "AutoOpen ${camera_mode} phase failed; aggregation was skipped." >&2
        return "${phase_status}"
    fi

    seeds_csv="$(IFS=,; echo "${phase_seeds[*]}")"
    elapsed=$(( $(date +%s) - phase_start ))
    PHASE_OUTPUT_ROOT="${dataset_root}" PHASE_SEEDS_CSV="${seeds_csv}" \
    PHASE_CAMERAS="${camera_mode}" COLLECTION_WALL_TIME_S="${elapsed}" python - <<'PY' || return "$?"
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
    debug_result_count="$(SOURCE_FINGERPRINT="${SOURCE_FINGERPRINT}" \
        SEED_START="${SEED_START}" SEED_END="${SEED_END}" \
        STOP_AFTER_STAGE="${STOP_AFTER_STAGE}" DOOR_MOTION="${DOOR_MOTION}" \
        PULL_PANEL_CONTACT_BASELINE="${PULL_PANEL_CONTACT_BASELINE}" \
        python - "${debug_pending_root}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
report_path = root / "acceptance_report.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
strict_accepted = sorted(
    int(row["seed"])
    for row in report["episodes"]
    if bool(row.get("accepted"))
)
if len(strict_accepted) != int(report.get("accepted_episodes", -1)):
    raise SystemExit("debug accepted seed extraction disagrees with aggregate report")
terminal_stage = os.environ["STOP_AFTER_STAGE"]
if terminal_stage == "PASS_THROUGH_DOORWAY":
    completed = strict_accepted
    result_label = "strict_acceptance"
    seed_file = "accepted_seeds.txt"
else:
    metadata_paths = sorted(root.glob("logs/rank*-control/seed_*/metadata.json"))
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    requested = int(os.environ["SEED_END"]) - int(os.environ["SEED_START"]) + 1
    if len(metadata) != requested:
        raise SystemExit(
            f"partial debug result count {len(metadata)} differs from requested {requested}"
        )
    completed = sorted(
        int(row["seed"])
        for row in metadata
        if bool(row.get("success"))
        and row.get("result", {}).get("requested_terminal_stage") == terminal_stage
    )
    result_label = "requested_terminal_completion"
    seed_file = "completed_terminal_seeds.txt"
(root / seed_file).write_text(
    "".join(f"{seed}\n" for seed in completed), encoding="utf-8"
)
(root / "workflow.json").write_text(
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
            "door_motion": os.environ["DOOR_MOTION"],
            "pull_panel_contact_baseline": os.environ["PULL_PANEL_CONTACT_BASELINE"],
            "requested_terminal_stage": terminal_stage,
            "result_label": result_label,
            "completed_requested_terminal": len(completed),
            "strict_full_workflow_accepted_episodes": len(strict_accepted),
            "camera_mode": "six",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(len(completed))
PY
)"
    if [[ "${STOP_AFTER_STAGE}" == PASS_THROUGH_DOORWAY ]]; then
        debug_result_tag="episodes${debug_result_count}"
    else
        debug_result_tag="terminal_${STOP_AFTER_STAGE,,}_success${debug_result_count}"
    fi
    debug_name="auto_open_debug_${DOOR_MOTION}_${RELEASE_VERSION}_${range_tag}_${debug_result_tag}_six_camera_${RELEASE_STAMP}"
    debug_root="${FORMAL_DATA_ROOT}/debug/${debug_name}"
    if [[ -e "${debug_root}" ]]; then
        echo "Refusing to overwrite debug dataset: ${debug_root}" >&2
        exit 2
    fi
    mv "${debug_pending_root}" "${debug_root}"
    rmdir "${debug_work_root}"
    python - "${debug_root}" <<'PY'
import sys
from pathlib import Path

from gr00t.rl.data_collection.auto_open_writer import write_six_camera_mosaics

manifest = write_six_camera_mosaics(Path(sys.argv[1]))
print(f"[workflow] six-camera mosaics completed: {len(manifest)} episodes")
PY
    if [[ "${STOP_AFTER_STAGE}" == PASS_THROUGH_DOORWAY ]]; then
        echo "[workflow] debug strict acceptance: ${debug_result_count}/${#ALL_SEEDS[@]}"
    else
        echo "[workflow] debug ${STOP_AFTER_STAGE} completion: ${debug_result_count}/${#ALL_SEEDS[@]}"
    fi
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

if [[ "${AUTO_OPEN_MODE}" == canonical ]]; then
    echo "[workflow] canonical no-camera collection (${#ALL_SEEDS[@]} seeds)"
else
    echo "[workflow] phase 1/2: canonical no-camera collection (${#ALL_SEEDS[@]} seeds)"
fi
run_phase "${canonical_root}" none "" ALL_SEEDS
if [[ "${STOP_AFTER_STAGE}" != PASS_THROUGH_DOORWAY ]]; then
    python - "${canonical_root}" "${STOP_AFTER_STAGE}" <<'PY'
import json
import sys
from pathlib import Path

root, terminal = Path(sys.argv[1]), sys.argv[2]
metadata = [json.loads(p.read_text()) for p in root.glob("logs/rank*-control/seed_*/metadata.json")]
completed = sorted(m["seed"] for m in metadata if m.get("success") and
                   m.get("result", {}).get("requested_terminal_stage") == terminal)
report = {"requested_terminal_stage": terminal, "episodes": len(metadata),
          "completed_terminal_seeds": completed, "not_full_workflow_acceptance": True}
(root / "partial_workflow_report.json").write_text(json.dumps(report, indent=2) + "\n")
(root / "completed_terminal_seeds.txt").write_text("".join(f"{s}\n" for s in completed))
print(f"[workflow] {terminal} prefix completion: {len(completed)}/{len(metadata)}; not full-task data acceptance")
PY
    rmdir "${work_root}"
    echo "[workflow] partial canonical validation completed: ${canonical_root}"
    trap - INT TERM EXIT
    exit 0
fi
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

if [[ "${AUTO_OPEN_MODE}" == canonical ]]; then
    rm -f "${accepted_seed_file}"
    rmdir "${work_root}"
    echo "[workflow] canonical validation completed: ${canonical_root}"
    trap - INT TERM EXIT
    exit 0
fi

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
