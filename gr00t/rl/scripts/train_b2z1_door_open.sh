#!/usr/bin/env bash
set -euo pipefail

# Official B2Z1 door-opening RL training entry.
# Defaults are intentionally small for launch validation; increase these for real runs.
GPU_IDS="${GPU_IDS:-${GPU_ID:-0}}"
IFS=',' read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ID_LIST[@]}}"
# NUM_ENVS is per GPU. Global parallel environments are NUM_ENVS * NUM_PROCESSES.
NUM_ENVS="${NUM_ENVS:-1}"
# Number of PPO updates to execute in this invocation.  The trainer itself
# expects an absolute global-step ceiling, so resumed runs translate this below.
NUM_TOTAL_BATCHES="${NUM_TOTAL_BATCHES:-2}"
NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-8}"
NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-1}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-500}"
DYNAMIC_MATERIAL_RANDOMIZATION="${DYNAMIC_MATERIAL_RANDOMIZATION:-false}"
DYNAMIC_MATERIAL_RANDOMIZATION_INTERVAL="${DYNAMIC_MATERIAL_RANDOMIZATION_INTERVAL:-1.0}"
LOG_FILE="${LOG_FILE:-/tmp/b2z1_door_train_validation.log}"
CHECKPOINT="${CHECKPOINT:-}"
TRAINING_STAGE="${TRAINING_STAGE:-}"
STAGE_INITIAL_STATE_SOURCE="${STAGE_INITIAL_STATE_SOURCE:-}"

export OMNI_KIT_ACCEPT_EULA="YES"
export ACCEPT_EULA="Y"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HYDRA_FULL_ERROR="1"

cd "$(dirname "$0")/../../.."

TARGET_TOTAL_BATCHES="${NUM_TOTAL_BATCHES}"
if [[ -n "${CHECKPOINT}" ]]; then
  checkpoint_step="$(python - "${CHECKPOINT}" <<'PY'
import sys

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
state = checkpoint.get("state")
print(int(getattr(state, "global_step", 0)))
PY
)"
  TARGET_TOTAL_BATCHES="$((checkpoint_step + NUM_TOTAL_BATCHES))"
  echo "Resuming at step ${checkpoint_step}; running ${NUM_TOTAL_BATCHES} updates through step ${TARGET_TOTAL_BATCHES}."
fi

launch_args=(
  --num_processes "${NUM_PROCESSES}"
  --num_machines 1
  --mixed_precision no
  --dynamo_backend no
)
if (( NUM_PROCESSES > 1 )); then
  launch_args+=(--multi_gpu --main_process_port "${MAIN_PROCESS_PORT:-29500}")
fi

checkpoint_args=()
if [[ -n "${CHECKPOINT}" ]]; then
  checkpoint_args+=("checkpoint=${CHECKPOINT}")
fi

stage_training_args=()
if [[ -n "${TRAINING_STAGE}" ]]; then
  stage_training_args+=("env.config.staged_reset_fixed_stage=${TRAINING_STAGE}")
fi

if [[ -n "${STAGE_INITIAL_STATE_SOURCE}" ]]; then
  stage_training_args+=(
    "env.config.staged_reset_initial_state_source=${STAGE_INITIAL_STATE_SOURCE}"
  )
fi

set +e
accelerate launch "${launch_args[@]}" gr00t/rl/train_agent_trl.py \
  +exp=wbmanip/door_open_b2z1_lstm \
  headless=true \
  num_envs="${NUM_ENVS}" \
  use_wandb=false \
  +force_exit_after_train=true \
  +simulator.config.enable_task_visual_materials="${DYNAMIC_MATERIAL_RANDOMIZATION}" \
  +simulator.config.task_randomize_material=true \
  +simulator.config.task_dynamic_material_randomization="${DYNAMIC_MATERIAL_RANDOMIZATION}" \
  +simulator.config.task_dynamic_material_randomization_interval="${DYNAMIC_MATERIAL_RANDOMIZATION_INTERVAL}" \
  algo.trl.num_total_batches="${TARGET_TOTAL_BATCHES}" \
  algo.trl.total_episodes=null \
  +algo.trl.logging_steps=1 \
  algo.trl.save_strategy=no \
  algo.config.num_steps_per_env="${NUM_STEPS_PER_ENV}" \
  algo.config.num_mini_batches="${NUM_MINI_BATCHES}" \
  callbacks.model_save.save_frequency="${SAVE_FREQUENCY}" \
  "${checkpoint_args[@]}" \
  "${stage_training_args[@]}" \
  "$@" \
  2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

if grep -Eq "Error executing job|Traceback|ValueError|RuntimeError|TypeError" "${LOG_FILE}"; then
  exit 1
fi

exit "${status}"
