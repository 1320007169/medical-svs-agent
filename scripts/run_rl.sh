#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:?set VERL_ROOT to a compatible VeRL checkout}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the SFT checkpoint}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT_DIR/data/processed/rl/train.parquet}"
VAL_FILE="${VAL_FILE:-$ROOT_DIR/data/processed/rl/val.parquet}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-4}"

[[ -f "$TRAIN_FILE" && -f "$VAL_FILE" ]] || {
  echo "error: run scripts/build_rl.py first" >&2
  exit 2
}
export PYTHONPATH="$ROOT_DIR/src:$VERL_ROOT:${PYTHONPATH:-}"
cd "$VERL_ROOT"
exec python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="[$TRAIN_FILE]" \
  data.val_files="[$VAL_FILE]" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.return_raw_chat=True \
  data.image_key=images \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_turns=6 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$ROOT_DIR/configs/openslide_tool.yaml" \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  custom_reward_function.path="$ROOT_DIR/src/medical_svs_agent/reward.py" \
  custom_reward_function.name=compute_score \
  reward_model.reward_manager=naive_async \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
  trainer.nnodes="$NNODES" \
  trainer.project_name=medical-svs-agent \
  trainer.default_local_dir="$ROOT_DIR/outputs/rl"
