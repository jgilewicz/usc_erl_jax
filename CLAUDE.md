# CLAUDE.md — usc_erl_jax

Extends the global CLAUDE.md. Project-specific deltas only.

## Stack

- Python (per `pyproject.toml` `requires-python`), JAX + equinox + optax
  for ERL, [SBX](https://github.com/araffin/sbx) (stable-baselines3 on
  JAX) for baselines, Hydra for config, wandb for logging.
- `uv` / `ruff` / `ty` — same as global rules (no pytest suite here).

## Layout

```text
src/
  train.py           # single Hydra entrypoint, baselines + ERL
  algos/erl.py        # ERL (EvoRainbow) training loop
  algos/sc_erl.py     # SC-ERL: per-individual surrogate gate (random | dropout | ensemble | evidential)
  baselines/          # SBX agent construction, vec-env wrapping, wandb callback
  common/             # replay buffer, rollout, fused TD3, critic heads per mode, surrogate gate/LCB, metrics
  modules/             # equinox nn modules (Critic, EvidentialCritic, SharedStateEmbedding, ActorHead) + CEM
  environments/        # env registry: mujoco, dm_control (dog-*), myosuite
  conf/                # Hydra configs (config.yaml + algorithm/*.yaml)
```

## Conventions

- Functional JAX style throughout `algos/`, `common/`, `modules/` — no
  classes for logic (`CEM` is state-holding, not logic-holding). PyTorch
  rule from the global file doesn't apply here.
- New algorithm = new `src/conf/algorithm/<name>.yaml` + a `_run_<name>`
  in `src/train.py` added to the `dispatch` map, not a new entrypoint.
- Env registration goes through `environments.register_env` /
  `register_backend` — never `gym.register` directly in algo code.
- ERL and SC-ERL share params and plumbing: `sc_erl.yaml` composes
  `erl.yaml`, `SCERLConfig` subclasses `ERLConfig`, `_run_sc_erl` reuses
  `_erl_kwargs`, and `algos/erl.py` exposes the shared `Run` /
  `build_run` / `train_and_merge` / `maybe_evaluate`. `theta`/`h_steps`
  are inert in SC-ERL.
- A surrogate mode = a `CriticHead` in `common/critic_heads.py`
  (`build`, `point`, `loss`, `stats`). critic1 carries the mode; critic2
  is always a plain `Critic` (usc_erl's asymmetric twin). Adding a mode =
  one `match` case + `MODES`.
- **Shadow rollout**: `pop_env` steps every individual; only gated-real
  ones are stored and counted. The algorithm must only ever see
  `observed = where(real, truth, 0)` — `truth` feeds metrics only.
  Never route `truth` into calibration, β, CEM or the anchor.
- Mixed fitness is calibrated by a per-generation offset measured on that
  generation's real individuals (critic scale drifts between gens). With
  no real individual the offset cannot change the ranking.
- CEM elite comes only from real-evaluated individuals
  (`CEM.tell(elite_candidates=...)`).
- TD3 updates run as one `eqx.filter_jit` `lax.fori_loop`
  (`make_td3_train`); pass `num_updates` as a `jnp` array or every new
  count recompiles. `num_updates` scales with collected env steps.
- Surrogate quality = `select/elite_overlap*`; gate quality = `gate/*`
  with "misranked" = wrong side of the elite cut under μ alone (never
  under the LCB: σ would sit in both label and score). Chance elite overlap is
  `parents/pop_size`.
- The mode changes TD3 (critic1 is the actor's critic), so gate claims
  need the same mode with `random_gate=true` at equal `real_frac`.
- `horizon_probe`: per-step shadow trajectories + own-state critic stats
  for offline analysis. Metric only, same rule as `truth`.
- `env_steps` is the x-axis for every performance claim.
- No `tests/` suite: verify with ruff, ty and a tiny-budget smoke run
  (`total_steps=3000 algorithm.horizon=100 algorithm.pop_size=4 ...`).
- `notes.md` (gitignored) holds measured results and refuted hypotheses.
  Critic uncertainty is closed as a negative result (§4.4–4.10: gating,
  own-state σ, σ vs Q error, MC racing, σ in the actor objective). Check
  it before re-proposing any uncertainty signal.

## Slurm

`slurm_run_array.sh` is the only slurm script: array-job-per-`(condition,
seed)` for one `TARGET_ENV` — sac, ppo, td3, crossq, erl, sc_erl × 4 modes,
3 σ-modes × `random_gate`, 12 × 5 = 60 tasks. `PROJECT_DIR` defaults to
`SLURM_SUBMIT_DIR`. Jobs run `.venv/bin/python` directly — `uv sync` on
the login node first; parallel `uv run` calls race on the shared uv cache.

- dog-* workers take ~0.55 GB each (pop + 1 + 2·eval processes): 32 GB at
  pop 10, 48 GB at pop 20.
- `dm_control.composer` forces `simplefilter("always",
  DeprecationWarning)` on import; warning filters for dm_control must be
  installed after `suite.load`, or slurm stderr grows by GBs.
- `$HOME` quota on WCSS is 50 GB — slurm logs and `.venv` (~10 GB) count.
