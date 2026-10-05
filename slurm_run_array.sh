#!/bin/bash -l
# One (condition, seed) pair per array task, for a single TARGET_ENV.
# 12 conditions x 5 seeds = 60 tasks: SBX baselines, ERL, SC-ERL x 4 modes,
# and the 3 uncertainty critics under a random gate (critic-vs-gate ablation).
#
#   cd /path/to/usc_erl_jax && TARGET_ENV=HalfCheetah-v5 \
#     sbatch --array=0-59 slurm_run_array.sh
#
# Task id = condition_index * 5 + seed, e.g. only SC-ERL ensemble
# (condition 7): sbatch --array=35-39 slurm_run_array.sh
#
# Optional: TOTAL_STEPS (default 1_000_000), TAG (extra wandb tag),
# EXTRA (space-separated Hydra overrides appended to every task, e.g.
# EXTRA="algorithm.pop_size=20" - only valid for conditions whose config
# has those keys).
#
# n_envs (SBX baselines' vec-env count) follows SLURM_CPUS_PER_TASK. ERL
# runs pop_size+1 env workers, SC-ERL pop_size+1 plus 2*eval.episodes for
# evaluation, regardless of -c. Raise -c if a run is CPU-bound, e.g.
#   sbatch --array=0-59 --cpus-per-task=16 slurm_run_array.sh

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
# default: the directory sbatch was run from (the checkout)
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?run sbatch from the checkout or set PROJECT_DIR}}"
TOTAL_STEPS="${TOTAL_STEPS:-1_000_000}"
TAG="${TAG:-sc-erl-v1}"

# name|overrides
CONDITIONS=(
  "sac|algorithm=sac"
  "ppo|algorithm=ppo"
  "td3|algorithm=td3"
  "crossq|algorithm=crossq"
  "erl|algorithm=erl"
  "sc_erl-random|algorithm=sc_erl algorithm.mode=random"
  "sc_erl-dropout|algorithm=sc_erl algorithm.mode=dropout"
  "sc_erl-ensemble|algorithm=sc_erl algorithm.mode=ensemble"
  "sc_erl-evidential|algorithm=sc_erl algorithm.mode=evidential"
  # omega=0.79 matches the ~21% real rate the uncertainty gate measured
  "sc_erl-dropout-rgate|algorithm=sc_erl algorithm.mode=dropout algorithm.random_gate=true algorithm.omega=0.79"
  "sc_erl-ensemble-rgate|algorithm=sc_erl algorithm.mode=ensemble algorithm.random_gate=true algorithm.omega=0.79"
  "sc_erl-evidential-rgate|algorithm=sc_erl algorithm.mode=evidential algorithm.random_gate=true algorithm.omega=0.79"
)
SEEDS=(0 1 2 3 4)
N_SEEDS=${#SEEDS[@]}

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CONDITION="${CONDITIONS[$((TASK_ID / N_SEEDS))]}"
SEED="${SEEDS[$((TASK_ID % N_SEEDS))]}"
NAME="${CONDITION%%|*}"
read -r -a OVERRIDES <<<"${CONDITION#*|} ${EXTRA:-}"
ENV_SLUG="${ENV//\//_}"
RUN_NAME="${NAME}_${ENV_SLUG}_s${SEED}"

echo "Task ${TASK_ID} | ${NAME} | ${ENV} | seed ${SEED}"

cd "${PROJECT_DIR}"
# no `module load`: the venv carries Python, jax[cuda12] wheels bundle CUDA.
# venv binaries, not uv's runner: parallel array tasks race on the shared uv cache
VENV="${PROJECT_DIR}/.venv/bin"
[[ -x "${VENV}/python" ]] || {
  echo "no venv at ${VENV} - run 'uv sync' in ${PROJECT_DIR} on the login node before submitting" >&2
  exit 1
}

: "${WANDB_API_KEY:?export WANDB_API_KEY before submitting}"
export WANDB_MODE=offline
# per-run dir: a shared one makes every task re-sync every other task's run
export WANDB_DIR="${PROJECT_DIR}/wandb_logs/${RUN_NAME}"
mkdir -p logs "${WANDB_DIR}"

if "${VENV}/python" -m train \
  "${OVERRIDES[@]}" \
  seed="${SEED}" \
  env.id="${ENV}" \
  eval_env.id="${ENV}" \
  total_steps="${TOTAL_STEPS}" \
  n_envs="${SLURM_CPUS_PER_TASK:-8}" \
  wandb.enabled=true \
  "wandb.name=${RUN_NAME}" \
  "wandb.tags=[${TAG},${NAME},${ENV_SLUG}]" \
  hydra.run.dir="outputs/${RUN_NAME}"; then
  for d in "${WANDB_DIR}"/wandb/offline-run-*; do
    [[ -d "$d" ]] && "${VENV}/wandb" sync "$d"
  done
else
  echo "ERROR: ${RUN_NAME} failed" >&2
  exit 1
fi
