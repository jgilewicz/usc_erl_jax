#!/bin/bash -l
# SEMARL p_surr sweep: one (condition, seed) pair per array task, one TARGET_ENV.
# 9 conditions x 3 seeds = 27 tasks.
#
# Population dump for scripts/surrogate_benchmark.py (condition 8) lands in
# outputs/semarl-dump_<env>_s<seed>/population.npz:
#   sbatch --array=24-26 slurm_semarl_sweep.sh
#
#   PROJECT_DIR=/path/to/usc_erl_jax TARGET_ENV=HalfCheetah-v5 \
#     sbatch --array=0-26 slurm_semarl_sweep.sh
#
# Rerun a single condition (e.g. adaptive = condition 6, seeds 0-2):
#   sbatch --array=18-20 slurm_semarl_sweep.sh
#
# Optional: TOTAL_STEPS (default 1_000_000)

#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=16gb
#SBATCH --time=0-12:00:00
#SBATCH --job-name=semarl-sweep
#SBATCH -p lem-gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

set -euo pipefail

ENV="${TARGET_ENV:?set TARGET_ENV, e.g. TARGET_ENV=HalfCheetah-v5}"
PROJECT_DIR="${PROJECT_DIR:?set PROJECT_DIR to the cluster checkout}"
TOTAL_STEPS="${TOTAL_STEPS:-1_000_000}"

# name|overrides - p_surr_min = p_surr_max = c reduces the adaptive gate to a constant c
CONDITIONS=(
  "erl|algorithm=erl"
  "semarl-p000|algorithm=semarl algorithm.p_surr_min=0.0 algorithm.p_surr_max=0.0"
  "semarl-p025|algorithm=semarl algorithm.p_surr_min=0.25 algorithm.p_surr_max=0.25"
  "semarl-p050|algorithm=semarl algorithm.p_surr_min=0.5 algorithm.p_surr_max=0.5"
  "semarl-p075|algorithm=semarl algorithm.p_surr_min=0.75 algorithm.p_surr_max=0.75"
  "semarl-p090|algorithm=semarl algorithm.p_surr_min=0.9 algorithm.p_surr_max=0.9"
  "semarl-adapt|algorithm=semarl"
  "td3|algorithm=td3"
  # p=0: every generation real, so every one is a benchmark sample
  "semarl-dump|algorithm=semarl algorithm.p_surr_min=0.0 algorithm.p_surr_max=0.0 algorithm.dump_path=population.npz algorithm.ensemble_size=5"
)
SEEDS=(0 1 2)
N_SEEDS=${#SEEDS[@]}

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CONDITION="${CONDITIONS[$((TASK_ID / N_SEEDS))]}"
SEED="${SEEDS[$((TASK_ID % N_SEEDS))]}"
NAME="${CONDITION%%|*}"
read -r -a OVERRIDES <<<"${CONDITION#*|}"
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
  "wandb.tags=[sweep-v2,${NAME},${ENV_SLUG}]" \
  hydra.run.dir="outputs/${RUN_NAME}"; then
  for d in "${WANDB_DIR}"/wandb/offline-run-*; do
    [[ -d "$d" ]] && "${VENV}/wandb" sync "$d"
  done
else
  echo "ERROR: ${RUN_NAME} failed" >&2
  exit 1
fi
