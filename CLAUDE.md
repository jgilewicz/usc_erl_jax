# CLAUDE.md — usc_erl_jax

Extends the global CLAUDE.md. Project-specific deltas only.

## Stack

- Python (per `pyproject.toml` `requires-python`), JAX + equinox + optax
  for ERL, [SBX](https://github.com/araffin/sbx) (stable-baselines3 on
  JAX) for baselines, Hydra for config, wandb for logging.
- `uv` / `ruff` / `ty` / `pytest` — same as global rules.

## Layout

```text
src/
  train.py           # single Hydra entrypoint, baselines + ERL
  algos/erl.py        # ERL (EvoRainbow) training loop
  algos/semarl.py     # SEMARL: ERL + h-step bootstrap horizon adapted to critic TD error
  baselines/          # SBX agent construction, vec-env wrapping, wandb callback
  common/             # replay buffer, rollout collection, TD3 core, EA/RL glue utils
  modules/             # equinox nn modules (Actor, Critic, SharedStateEmbedding, ActorHead) + CEM
  environments/        # env registry: mujoco, dm_control (dog-*), myosuite
  conf/                # Hydra configs (config.yaml + algorithm/*.yaml)
scripts/               # post-hoc analysis, not shipped in the wheel
```

## Conventions

- Functional JAX style throughout `algos/`, `common/`, `modules/` — no
  classes for logic (`CEM` is state-holding, not logic-holding). PyTorch
  rule from the global file doesn't apply here.
- New algorithm = new `src/conf/algorithm/<name>.yaml` + a `_run_<name>`
  in `src/train.py` added to the `dispatch` map, not a new entrypoint.
- Env registration goes through `environments.register_env` /
  `register_backend` — never `gym.register` directly in algo code.
- ERL and SEMARL are separate files but share params: `semarl.yaml`
  composes `erl.yaml` via `defaults`, `_run_semarl` reuses `_erl_kwargs`.
  `SEMARLConfig` subclasses `ERLConfig` (adds `h_min/h_max/h_beta/td_ema_decay`).

## Slurm

`slurm_run_array.sh` is array-job-per-`(algorithm, seed)` for one
`TARGET_ENV`. Set `PROJECT_DIR` before submitting (currently a
placeholder path). See README for the full sbatch sweep loop.
