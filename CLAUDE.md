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
  algos/semarl.py     # SEMARL: ERL + surrogate-evaluation frequency (p_surr) adapted to critic TD error
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
  `SEMARLConfig` subclasses `ERLConfig` (adds
  `p_surr_min/p_surr_max/p_beta/td_ema_decay`). SEMARL fixes the bootstrap
  horizon at the inherited `h_steps` and ignores the inherited `theta` —
  `p_surr` replaces it.
- Surrogate quality is judged by `surrogate_elite_overlap` (CEM keeps
  `argsort(-scores)[:parents]` and drops the rest of the ordering, so
  elite membership is all selection consumes; chance is 0.5). Rank
  correlation is the looser secondary view, `abs_err` only watches scale
  drift.
- Fitness estimators are scored against the true return, by metric-name
  suffix: `` = the `h_steps` bootstrap, `_noboot` = same with `γ^H·Q`
  dropped, `_critic` = `E_{s~D}[Q(s, π_i(s))]` over a replay batch,
  `_pevfa` = the same but policy-conditioned. **Arms are measured, not
  used to select** — adding one cannot change a baseline. Adding an arm
  means adding a suffix to the `arms` dict in
  `semarl.py` and to `ARMS` in `surrogate_diagnostics.py` — the metric
  names and the report table are generated from those.
- SEMARL runs **two vec envs**: `rl_env` (1 env, always a full `horizon`)
  and `pop_env` (`pop_size`, skipped entirely on surrogate generations).
  That skip is the env-step saving and it is why they cannot share a vec
  env. Consequences: arms are only measurable on real generations (no
  population rollout ⇒ no ground truth), and `num_updates` must scale with
  `gen_env_steps`, never a fixed count.
- `env_steps` is the x-axis for every performance claim — generations are
  not comparable across `p_surr` once the rollout is split.
- The buffer's `policy_id` column tags each transition with the policy
  that generated it (PeVFA's TD target needs that policy's action at
  `s'`). `collect_parallel_episode(..., policy_ids=...)`; callers that do
  not track policies store `-1` and PeVFA skips those rows. Raw `W` lives
  in a ring in `semarl.train`, sized `buffer_capacity // horizon`.
- `notes.md` holds the measured results and the list of refuted
  hypotheses. Check it before re-proposing a gating signal.

## Slurm

`slurm_run_array.sh` is array-job-per-`(algorithm, seed)` for one
`TARGET_ENV`. Set `PROJECT_DIR` before submitting (currently a
placeholder path). See README for the full sbatch sweep loop.
