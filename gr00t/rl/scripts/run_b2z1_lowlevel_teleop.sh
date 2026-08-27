#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOWLEVEL_ROOT="${REPO_ROOT}/gr00t/rl/third_party/visual_wholebody_lowlevel"
ENTRY="${LOWLEVEL_ROOT}/legged_gym/scripts/manip_loco_interface.py"
RUN_DIR="${REPO_ROOT}/ckpt/b2z1-low/20260817_154058_is-dc6jrxz4suufnmlp-devmachine-0_gpu1_agent2_z8e0p66v"

GPU_ID="${GPU_ID:-0}"
HEADLESS="${HEADLESS:-false}"
ENABLE_DOORMAN_SCENE="${ENABLE_DOORMAN_SCENE:-true}"
CKPT_PATH="${CKPT_PATH:-${RUN_DIR}/model_40000.pt}"
RUN_METADATA_PATH="${RUN_METADATA_PATH:-${RUN_DIR}/run_metadata.json}"
USD_PATH="${USD_PATH:-${REPO_ROOT}/gr00t/rl/data/robots/b2z1_lab30/b2z1_mount_0_x0p22_y0_z0p17.usd}"
RTX_REFLECTIONS="${RTX_REFLECTIONS:-true}"
RTX_TRANSLUCENCY="${RTX_TRANSLUCENCY:-true}"
RTX_SUBSURFACE_SCATTERING="${RTX_SUBSURFACE_SCATTERING:-true}"
RTX_CAUSTICS="${RTX_CAUSTICS:-true}"
RTX_INDIRECT_DIFFUSE="${RTX_INDIRECT_DIFFUSE:-true}"
RTX_AMBIENT_OCCLUSION="${RTX_AMBIENT_OCCLUSION:-true}"
RTX_DIRECT_LIGHTING_SPP="${RTX_DIRECT_LIGHTING_SPP:-2}"
RTX_REFLECTIONS_SPP="${RTX_REFLECTIONS_SPP:-2}"

[[ -f "${CKPT_PATH}" ]] || { echo "Checkpoint not found: ${CKPT_PATH}" >&2; exit 1; }
[[ -f "${RUN_METADATA_PATH}" ]] || { echo "Run metadata not found: ${RUN_METADATA_PATH}" >&2; exit 1; }
[[ -f "${USD_PATH}" ]] || { echo "B2Z1 USD not found: ${USD_PATH}" >&2; exit 1; }

if [[ "${HEADLESS}" != "true" && -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is empty. Start/recreate the development container from a graphical session." >&2
    exit 1
fi

headless_args=(--no-headless)
if [[ "${HEADLESS}" == "true" ]]; then
    headless_args=(--headless)
fi

scene_args=()
if [[ "${ENABLE_DOORMAN_SCENE}" == "true" ]]; then
    scene_args+=(
        --enable_doorman_scene
        --doorman_randomize_robot_init true
        --doorman_randomize_door_init_state false
    )
fi

cd "${REPO_ROOT}"
exec python "${ENTRY}" \
    --task b2z1 \
    --num_envs 1 \
    --ckpt_path "${CKPT_PATH}" \
    --run_metadata_path "${RUN_METADATA_PATH}" \
    --robot_usd_path "${USD_PATH}" \
    --sim_device "cuda:${GPU_ID}" \
    --rl_device "cuda:${GPU_ID}" \
    --teleop_mode \
    --action_delay_mode auto \
    --rtx_reflections "${RTX_REFLECTIONS}" \
    --rtx_translucency "${RTX_TRANSLUCENCY}" \
    --rtx_subsurface_scattering "${RTX_SUBSURFACE_SCATTERING}" \
    --rtx_caustics "${RTX_CAUSTICS}" \
    --rtx_indirect_diffuse "${RTX_INDIRECT_DIFFUSE}" \
    --rtx_ambient_occlusion "${RTX_AMBIENT_OCCLUSION}" \
    --rtx_direct_lighting_spp "${RTX_DIRECT_LIGHTING_SPP}" \
    --rtx_reflections_spp "${RTX_REFLECTIONS_SPP}" \
    "${scene_args[@]}" \
    "${headless_args[@]}" \
    "$@"
