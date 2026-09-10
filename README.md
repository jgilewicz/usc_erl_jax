# USC-ERL (JAX)

Evolutionary-RL hybrids for continuous control, JAX/equinox rewrite of
[usc_erl](https://github.com/jgilewicz/usc_erl). Hydra-configured, one
entrypoint (`src/train.py`) for baselines and ERL alike.

## Baselines

SAC, TD3, PPO, CrossQ — via [SBX](https://github.com/araffin/sbx) (JAX port
of stable-baselines3), same `EvalCallback` + wandb logging as ERL.

```bash
just train sac Hopper-v5
just train-all Ant-v5           # sac, ppo, td3, crossq, sequentially
```

## ERL (EvoRainbow)

[Bai et al., EvoRainbow](https://openreview.net/pdf?id=75Hes6Zse4)-style
evolutionary-RL hybrid. Config: `src/conf/algorithm/erl.yaml`, impl:
`src/algos/erl.py`.

- **Backbone**: TD3 (`src/common/td3.py`).
- **Shared architecture**: `SharedStateEmbedding` computes `Z(s)` once;
  both the CEM population (`ActorHead`s) and the RL actor head read off
  the same embedding.
- **CEM evolution**: population of `ActorHead` params evolved by CEM
  (`src/modules/evo_module.py`) over `Z(s)`, with elitism.
- **Parallel integration**: population and RL actor act in the same
  vectorized rollout step (`collect_parallel_episode`) — one episode per
  generation feeds both the EA fitness and the replay buffer, no
  separate eval-then-train phases.
- **Genetic soft update**: each generation's champion head is
  soft-merged into the RL actor (`genetic_soft_update`, `ea_tau`); the RL
  actor is periodically injected back into the weakest population slot
  (`rl_to_ea_sync_period`).
- **Surrogate fitness**: per-generation coin flip (`theta`) between real
  full-episode return and a cheap H-step critic-bootstrap estimate
  (`h_step_bootstrap`, `h_steps`) — the base surrogate mechanism, no
  uncertainty gating yet.

```bash
just train erl HalfCheetah-v5
```

## SEMARL

[SEMARL](https://dl.acm.org/doi/10.1145/3795095.3805146) — ERL backbone
(shared embedding, CEM, parallel rollout, genetic soft update) where the
surrogate's bootstrap horizon `H` adapts to critic error instead of
ERL's fixed `h_steps`. Real vs surrogate fitness is still the fixed
`theta` coin flip (0.6, shared with ERL). Impl: `src/algos/semarl.py`,
config: `src/conf/algorithm/semarl.yaml` (inherits `erl.yaml`).

- **Absolute TD error**: each generation, on a fresh replay batch,
  `mean |r + γ(1−d)·min(Q1',Q2') − min(Q1,Q2)|` — clipped-double-Q
  Bellman residual of the current critic (`clipped_double_q`,
  `absolute_td_error`), smoothed into `td_error_ema` (`td_ema_decay`).
- **Adaptive H**: `H = round(h_min + (h_max−h_min)·(1−exp(−h_beta·|TD|_ema)))`
  (`adaptive_h_step`), taken from the previous generation's EMA. Accurate
  critic (low `|TD|`) → short `H`, lean on the critic bootstrap; noisy
  critic → `H` grows toward `h_max`, lean on real reward.
- **Metrics**: `td_error` / `td_error_ema` / `h_step`, plus the surrogate's
  per-generation agreement with the true full-episode return at three
  horizons (adaptive `H`, `h_min`, `h_max`) — `surrogate_rank_corr*`
  (Spearman over the population, scale-free, what CEM selection uses) and
  `surrogate_abs_err*` (kept only to watch surrogate/real scale drift). The
  dual-H pair tests whether a short horizon ranks worse when the critic is
  worse.
- `h_beta` must be tuned to the env's Bellman-residual scale (Q-values
  here sit on the undiscounted-return scale).
- The population rollout still runs the full `horizon` (feeds the buffer),
  so `H` currently trades surrogate bias/variance, not env steps — the
  two-call rollout split (RL actor full, population to `H`) is the next
  step for actual interaction savings.

```bash
just train semarl HalfCheetah-v5
# validate the adaptation from a finished run's metrics:
uv run python scripts/surrogate_diagnostics.py --wandb evo_rl/triage_erl/<run_id>
```

## Tooling

- `justfile`: `install`, `test`, `lint`/`lint-check`, `types`, `check`,
  `train`, `train-all`.
- `scripts/surrogate_diagnostics.py`: post-hoc — does `td_error` predict
  surrogate inaccuracy, and does a short `H` hurt more when the critic is
  worse (raw + trend-removed Spearman).
- `slurm_run_array.sh`: array job, one `(algorithm, seed)` task per
  index; 6 algos × 5 seeds = 30 tasks for one `TARGET_ENV`.

## Full experiment suite on slurm

```bash
export WANDB_API_KEY=...
for env in HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5 Swimmer-v5 \
           dog-stand dog-walk dog-trot dog-run \
           myoElbowPose1D6MRandom-v0 myoHandReachRandom-v0 \
           myoHandPenTwirlRandom-v0 myoHandObjHoldRandom-v0 myoLegWalk-v0; do
  TARGET_ENV="$env" sbatch --array=0-29 slurm_run_array.sh
done
```
