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
*frequency* of surrogate evaluation adapts to critic error. The bootstrap
horizon stays fixed at ERL's `h_steps`; what replaces ERL's fixed `theta`
coin flip is an adaptive `p_surr`. Impl: `src/algos/semarl.py`, config:
`src/conf/algorithm/semarl.yaml` (inherits `erl.yaml`).

Earlier versions adapted `H` instead. That was dropped: measured on
HalfCheetah, `rank_corr` rose monotonically with `H` (h_max +0.79 vs
h_min +0.27, h_max better in 99% of generations), so there was no critic
quality at which a short `H` paid — and since the rollout runs the full
`horizon` either way, a short `H` saved nothing to trade for it.

All tuning to date is HalfCheetah-only: `p_beta = 3.0` is set against
its `|TD|_rel ≈ 0.21`. The cross-env portability that the `mean|r|`
normalization is *for* has not been tested on a second env yet.

- **Relative TD error**: each generation, on a fresh replay batch,
  `mean |r + γ(1−d)·min(Q1',Q2') − min(Q1,Q2)|` (`clipped_double_q`,
  `absolute_td_error`) normalized by `mean|r|` from the same batch
  (`relative_td_error`). `|TD|` is a per-step residual, so the
  denominator has to be per-step too — `mean|Q|` is a discounted *return*,
  which buries a factor of `(1−γ)` in the ratio and shrinks it further as
  `Q` grows over training. Smoothed into `td_error_rel_ema`
  (`td_ema_decay`).
- **Adaptive p_surr**:
  `p_surr = p_surr_min + (p_surr_max−p_surr_min)·exp(−p_beta·|TD|_rel_ema)`
  (`adaptive_p_surr`), taken from the previous generation's EMA — the
  choice has to precede collection so it can gate the rollout once the
  population rollout is truncated. Accurate critic → more surrogate
  generations; noisy critic → fall back to real evaluation.
- **Metrics**: `td_error` (raw, informational) / `td_error_rel` /
  `td_error_rel_ema` / `p_surr`, plus the per-generation agreement with
  the true full-episode return for four fitness estimators, all scored
  for free every generation (suffix on each metric name):
  - `` (none) — the `h_steps` bootstrap actually driving selection.
  - `_noboot` — the same, with the `γ^H·Q` term dropped. Isolates how
    much of the surrogate is accumulated real reward rather than critic:
    the bootstrap's share of the value is exactly `γ^H` (8.1% at γ=0.99,
    H=250). If this matches the `H` arm, the critic contributes nothing.
  - `_critic` — `E_{s~D}[min(Q1,Q2)(s, π_i(s))]` over a replay batch, the
    standard critic-only fitness and the honest SEMARL-style baseline.
  - `_h0` — its degenerate one-state case (bootstrap at the episode's
    first state). Kept for contrast: every individual starts from
    near-identical states, so this arm must separate policies by their
    action at a single point, and lands near chance.

  For each arm:
  - `surrogate_elite_overlap*` — fraction of CEM's top-`parents` set the
    surrogate gets right. The headline number: `_cem_tell` keeps
    `argsort(-scores)[:parents]` and discards everything else, so this is
    all selection consumes. Chance is 0.5, not 0.
  - `surrogate_rank_corr*` — Spearman over the whole population; looser,
    also scores pairs selection never looks at.
  - `surrogate_abs_err*` — kept only to watch surrogate/real scale drift.
- `q_disagree_mean` / `q_disagree_rank_corr` — per-individual `|Q1−Q2|`
  at the surrogate's own bootstrap states, and its correlation with how
  far that individual is misranked. Logged but unused: it is the
  candidate gating signal for uncertainty-gated `p_surr`, and this says
  whether it carries anything before it is wired in.
- `p_beta` is a dimensionless sensitivity constant on the relative error,
  meant to be shared across envs (unlike a raw-`|TD|` threshold).
- `theta` and `h_steps` are inherited from `ERLConfig`; SEMARL uses
  `h_steps` as its fixed horizon and ignores `theta` (`p_surr` replaces
  it), so a SEMARL run's logged `theta` is inert.
- The population rollout still runs the full `horizon` (feeds the buffer),
  so `p_surr` currently trades surrogate bias/variance, not env steps —
  the two-call rollout split (RL actor full, population to `h_steps` on
  surrogate generations) is the next step for actual interaction savings.

```bash
just train semarl HalfCheetah-v5
# validate the adaptation from a finished run's metrics:
uv run python scripts/surrogate_diagnostics.py --wandb evo_rl/triage_erl/<run_id>
```

## Tooling

- `justfile`: `install`, `test`, `lint`/`lint-check`, `types`, `check`,
  `train`, `train-all`.
- `scripts/surrogate_diagnostics.py`: post-hoc — does relative `|TD|`
  predict a worse surrogate, does `p_surr` open when the surrogate is
  good, how the four fitness-estimator arms compare (including whether
  dropping the bootstrap changes anything), and whether `|Q1−Q2|` flags
  the misranked individuals (raw + trend-removed Spearman); flags a
  saturated `p_surr` — constant, *or* pinned against a rail, which is the
  same non-result and easy to miss in the range alone.
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
