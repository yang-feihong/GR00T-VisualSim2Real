#!/usr/bin/env bash
set -euo pipefail

# Official B2Z1 door-opening RL training entry.
# Defaults are intentionally small for launch validation; increase these for real runs.
GPU_ID="2"                 # Host GPU exposed as cuda:0 inside this process.
NUM_ENVS="1"
NUM_TOTAL_BATCHES="2"
NUM_STEPS_PER_ENV="8"
NUM_MINI_BATCHES="1"
DYNAMIC_MATERIAL_RANDOMIZATION="${DYNAMIC_MATERIAL_RANDOMIZATION:-false}"
DYNAMIC_MATERIAL_RANDOMIZATION_INTERVAL="${DYNAMIC_MATERIAL_RANDOMIZATION_INTERVAL:-1.0}"
LOG_FILE="${LOG_FILE:-/tmp/b2z1_door_train_validation.log}"

export OMNI_KIT_ACCEPT_EULA="YES"
export ACCEPT_EULA="Y"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HYDRA_FULL_ERROR="1"

cd "$(dirname "$0")/../../.."

set +e
accelerate launch --num_processes 1 gr00t/rl/train_agent_trl.py \
  +exp=wbmanip/door_open_b2z1_lstm \
  headless=true \
  num_envs="${NUM_ENVS}" \
  use_wandb=false \
  +force_exit_after_train=true \
  +simulator.config.enable_task_visual_materials="${DYNAMIC_MATERIAL_RANDOMIZATION}" \
  +simulator.config.task_randomize_material=true \
  +simulator.config.task_dynamic_material_randomization="${DYNAMIC_MATERIAL_RANDOMIZATION}" \
  +simulator.config.task_dynamic_material_randomization_interval="${DYNAMIC_MATERIAL_RANDOMIZATION_INTERVAL}" \
  algo.trl.num_total_batches="${NUM_TOTAL_BATCHES}" \
  algo.trl.total_episodes=null \
  +algo.trl.logging_steps=1 \
  algo.trl.save_strategy=no \
  algo.config.num_steps_per_env="${NUM_STEPS_PER_ENV}" \
  algo.config.num_mini_batches="${NUM_MINI_BATCHES}" \
  2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

if rg -q "Error executing job|Traceback|ValueError|RuntimeError|TypeError" "${LOG_FILE}"; then
  exit 1
fi

exit "${status}"
