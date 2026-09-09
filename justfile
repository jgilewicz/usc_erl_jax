set dotenv-load := false

# parallel env workers: match the Slurm CPU allocation when present,
# else the local core count
n_cpus := `echo "${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"`

# list recipes
default:
    @just --list

# sync the venv from the lockfile
install:
    uv sync

# run the test suite (pass extra args: just test tests/baselines -k wandb)
test *args:
    uv run pytest -q {{args}}

# lint + autofix + format
lint:
    uv run ruff check --fix .
    uv run ruff format .

# lint without writing changes (CI)
lint-check:
    uv run ruff check .
    uv run ruff format --check .

# static type check on the training entrypoint + baselines package
types:
    uv run ty check src/train.py src/baselines

# everything: lint-check, types, tests
check: lint-check types test

# train one SBX baseline
#   just train sac Hopper-v5
#   just train crossq Ant-v5 seed=3 total_steps=2_000_000 wandb.enabled=false
#   just train sac Hopper-v5 n_envs=4   # override the detected core count
train algo="sac" env="Swimmer-v5" n_envs=n_cpus *overrides:
    uv run python -m train \
      algorithm={{algo}} env.id={{env}} eval_env.id={{env}} \
      n_envs={{n_envs}} {{overrides}}

# train every baseline on one env, sequentially
train-all env="Swimmer-v5" n_envs=n_cpus *overrides:
    #!/usr/bin/env bash
    set -euo pipefail
    for algo in sac ppo td3 crossq erl; do
      just train "$algo" "{{env}}" n_envs={{n_envs}} {{overrides}}
    done
