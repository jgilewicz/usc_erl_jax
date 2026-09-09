#!/bin/bash
# Array job: one (algorithm, seed) pair per task, for a single TARGET_ENV.
# 5 algorithms x 5 seeds = 25 tasks.
#
#   TARGET_ENV=HalfCheetah-v5 sbatch --array=0-24 slurm_run_array.sh
#   TARGET_ENV=dog-stand      sbatch --array=0-24 slurm_run_array.sh
#   TARGET_ENV=myoLegWalk-v0  sbatch --array=0-24 slurm_run_array.sh
#
# Optional: TOTAL_STEPS (default 1_000_000)
#
# n_envs (SB3 baselines' vec-env count) is set from SLURM_CPUS_PER_TASK,
# so it always matches the actual allocation below. ERL ignores n_envs and
# always runs pop_size+1 workers (11 by default) regardless of -c: that's
# a mild, fine oversubscription at pop_size=10, but revisit (batch the
# population instead of 1 process per candidate) before pushing pop_size
# much higher. Override cores per-submission if a run needs more, e.g.:
#   TARGET_ENV=Ant-v5 sbatch --array=0-24 --cpus-per-task=16 slurm_run_array.sh

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=16gb
#SBATCH --time=0-12:00:00
#SBATCH --job-name=erl
#SBATCH -p lem-gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

set -euo pipefail

ENV="${TARGET_ENV:?set TARGET_ENV, e.g. TARGET_ENV=HalfCheetah-v5}"
TOTAL_STEPS="${TOTAL_STEPS:-1_000_000}"
N_ENVS="${SLURM_CPUS_PER_TASK:-8}"

ALGORITHMS=(sac ppo td3 crossq erl)
SEEDS=(0 1 2 3 4)
N_SEEDS=${#SEEDS[@]}

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ALGO="${ALGORITHMS[$((TASK_ID / N_SEEDS))]}"
SEED="${SEEDS[$((TASK_ID % N_SEEDS))]}"
ENV_SLUG="${ENV//\//_}"
RUN_NAME="${ALGO}_${ENV_SLUG}_seed${SEED}"

echo "Task ${TASK_ID} | ${ALGO} | ${ENV} | seed ${SEED}"

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# TODO: set this to the cluster checkout path for usc_erl_jax
PROJECT_DIR="/path/to/usc_erl_jax"
cd "${PROJECT_DIR}"

: "${WANDB_API_KEY:?export WANDB_API_KEY before submitting}"
export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_DIR}/wandb_logs"
mkdir -p logs "${WANDB_DIR}"

if uv run python -m train \
  algorithm="${ALGO}" \
  seed="${SEED}" \
  env.id="${ENV}" \
  eval_env.id="${ENV}" \
  total_steps="${TOTAL_STEPS}" \
  n_envs="${N_ENVS}" \
  wandb.enabled=true \
  "wandb.name=${RUN_NAME}" \
  "wandb.tags=[${ALGO},${ENV_SLUG}]" \
  hydra.run.dir="outputs/${RUN_NAME}"; then
  for d in "${WANDB_DIR}"/wandb/offline-run-*; do
    [[ -d "$d" ]] && wandb sync "$d"
  done
else
  echo "ERROR: ${RUN_NAME} failed" >&2
  exit 1
fi
