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
  baselines/          # SBX agent construction, vec-env wrapping, wandb callback
  common/             # replay buffer, rollout collection, TD3 core, EA/RL glue utils
  modules/             # equinox nn modules (Actor, Critic, SharedStateEmbedding, ActorHead) + CEM
  environments/        # env registry: mujoco, dm_control (dog-*), myosuite
  conf/                # Hydra configs (config.yaml + algorithm/*.yaml)
```

## Conventions

- Functional JAX style throughout `algos/`, `common/`, `modules/` — no
  classes for logic (`CEM` is state-holding, not logic-holding). PyTorch
  rule from the global file doesn't apply here.
- New algorithm = new `src/conf/algorithm/<name>.yaml` + wiring in
  `src/train.py`'s `_run_erl`/`_run_sb3` dispatch, not a new entrypoint.
- Env registration goes through `environments.register_env` /
  `register_backend` — never `gym.register` directly in algo code.
- SEMARL is a placeholder (see README) — no `src/conf/algorithm/semarl.yaml`
  or implementation yet; don't scaffold it speculatively.

## Slurm

`slurm_run_array.sh` is array-job-per-`(algorithm, seed)` for one
`TARGET_ENV`. Set `PROJECT_DIR` before submitting (currently a
placeholder path). See README for the full sbatch sweep loop.
