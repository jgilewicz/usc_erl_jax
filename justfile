set dotenv-load := false

# parallel env workers (SBX baselines): N_ENVS if set, else the Slurm CPU
# allocation, else the local core count. ERL/SC-ERL ignore it.
n_cpus := `echo "${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"`
n_envs := env_var_or_default("N_ENVS", n_cpus)

# list recipes
default:
    @just --list

# sync the venv from the lockfile
install:
    uv sync

# lint + autofix + format
lint:
    uv run ruff check --fix .
    uv run ruff format .

# lint without writing changes (CI)
lint-check:
    uv run ruff check .
    uv run ruff format --check .

# static type check
types:
    uv run ty check src

# everything: lint-check, types
check: lint-check types

# train one algorithm; extra tokens are Hydra overrides
#   just train sac Hopper-v5
#   just train sc_erl Swimmer-v5 algorithm.mode=ensemble
#   just train sc_erl Swimmer-v5 seed=3   # seed sweeps: seed=N is a Hydra override
#   N_ENVS=4 just train sac Hopper-v5   # override the detected core count
train algo="sac" env="Swimmer-v5" *overrides:
    uv run python -m train \
      algorithm={{algo}} env.id={{env}} eval_env.id={{env}} \
      n_envs={{n_envs}} {{overrides}}

# train every baseline + ERL/SC-ERL on one env, sequentially
train-all env="Swimmer-v5" *overrides:
    #!/usr/bin/env bash
    set -euo pipefail
    for algo in sac ppo td3 crossq erl sc_erl; do
      just train "$algo" "{{env}}" {{overrides}}
    done
