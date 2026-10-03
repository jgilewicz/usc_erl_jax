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
(shared embedding, CEM, genetic soft update) where the *frequency* of
surrogate evaluation adapts to critic error: an adaptive `p_surr` replaces
ERL's fixed `theta` coin flip. Impl: `src/algos/semarl.py`, config:
`src/conf/algorithm/semarl.yaml` (inherits `erl.yaml`).

**Split rollout.** The RL actor and the population run in *separate* vec
envs. The actor always runs a full `horizon` (it is the gradient learner
and the only guaranteed source of fresh buffer data); the population
rollout is what a surrogate generation skips entirely. That skip is the
whole env-step saving — 11 000 → 1 000 steps per generation at
`pop_size=10` — and it is impossible while both step in lockstep. On a
surrogate generation selection runs on the batch-averaged critic value.

`num_updates` scales with steps actually collected, not a fixed count:
otherwise a surrogate generation silently inflates the update-to-data
ratio ~11× and reads as "the gate helped".

```text
CEM.ask() ──► 10 × ActorHead over Z(s)
                   │
      p_surr = p_min + (p_max−p_min)·exp(−p_beta·|TD|_rel_ema)
                   │   ← from the PREVIOUS generation's EMA: the
                   │     rollout itself depends on this decision
        ┌──────────┴──────────┐
1−p_surr│                     │p_surr
        ▼                     ▼
┌─────────────────┐  ┌─────────────────┐
│ REAL            │  │ SURROGATE       │
├─────────────────┤  ├─────────────────┤
│ rl_env   1×1000 │  │ rl_env   1×1000 │ ← always: gradient learner,
│ pop_env 10×1000 │  │ pop_env skipped │   only guaranteed fresh data
├─────────────────┤  ├─────────────────┤
│  11 000 steps   │  │   1 000 steps   │ ← the 11× saving
├─────────────────┤  ├─────────────────┤
│ f = true return │  │ f = E[Q(s,π_i)] │
│ arms scored ✓   │  │ no ground truth │
└────────┬────────┘  └────────┬────────┘
         └──────────┬─────────┘
                    ▼
           CEM.tell(f) ──► top-5 of 10 ──► new μ, Σ
                    │
                    ▼
     TD3 updates × (train_ratio · steps collected this gen)
                    │
                    ▼
     genetic_soft_update: champion head ──► RL actor
```

The surrogate is not free of the rollout it skips — it is trained on what
that rollout collects:

```text
  population rollouts ──► buffer ──► critic ──► surrogate fitness
         ▲                                              │
         │                                              ▼
         └────────────── CEM selection ◄────────────────┘
              (only on REAL generations)
```

Raising `p_surr` cuts the left edge of that loop: fewer population
rollouts ⇒ the buffer sees only the RL actor's distribution ⇒ the critic
loses the ability to rank *other* policies. So `p_surr` is bounded by the
critic's appetite for policy-diverse data, not by fitness accuracy — that
is the open question the `p_surr` sweep is meant to answer, not an
assumption baked into the code.

- **Relative TD error**: each generation, on a fresh replay batch,
  `mean |r + γ(1−d)·min(Q1',Q2') − min(Q1,Q2)|` (`clipped_double_q`,
  `absolute_td_error`) normalized by `mean|r|` from the same batch
  (`relative_td_error`). `|TD|` is a per-step residual, so the
  denominator has to be per-step too — `mean|Q|` is a discounted *return*,
  which buries a factor of `(1−γ)` in the ratio and shrinks it further as
  `Q` grows over training. Smoothed into `td_error_rel_ema`.
- **Adaptive p_surr**:
  `p_surr = p_surr_min + (p_surr_max−p_surr_min)·exp(−p_beta·|TD|_rel_ema)`
  (`adaptive_p_surr`), from the previous generation's EMA — the choice has
  to precede collection, since it decides whether the population rolls out
  at all.
- **Fitness estimators**, scored against the true return by metric-name
  suffix. Only measurable on real generations: a surrogate generation has
  no population rollout, so no ground truth and no `h_reward`.
  - `` (none) — the `h_steps` bootstrap.
  - `_noboot` — the same with `γ^H·Q` dropped. The bootstrap's share of
    the value is exactly `γ^H` (8.1% at γ=0.99, H=250), so if this matches
    the `H` arm the critic contributes nothing.
  - `_critic` — `E_{s~D}[min(Q1,Q2)(s, π_i(s))]` over a replay batch.
    What selects when the population rollout is skipped, at zero env
    steps. Averaging over the batch is what makes it work: scored at a
    single state it sits at chance.
  - metrics per arm: `surrogate_elite_overlap*` (headline — `_cem_tell`
    keeps `argsort(-scores)[:parents]` and discards the rest, so elite
    membership is all selection consumes; **chance is 0.5**),
    `surrogate_rank_corr*` (looser, scores pairs selection ignores),
    `surrogate_abs_err*` (scale drift only).
- `env_steps` / `env_steps_gen` — cumulative and per-generation
  interaction cost. The x-axis for every performance claim.
- `p_beta` is a dimensionless sensitivity constant on the relative error,
  meant to be shared across envs (unlike a raw-`|TD|` threshold).
- **Two inherited-but-inert params.** `theta` (`p_surr` replaces it) and
  `h_steps`, which in SEMARL sizes the h-bootstrap *diagnostic* arm only —
  nothing about training changes if you move it, because the bootstrap
  never selects here. Both are live in `erl.py`. `h_steps` would become
  live again only if a truncated-population generation were reintroduced
  as a third mode between "full real rollout" and "no rollout at all".

**Measured on HalfCheetah, see `notes.md` for the numbers.** Adapting `H`
was dropped (`rank_corr` rises monotonically with `H`). The adaptive gate
itself is so far indistinguishable from a fixed `p_surr`: no critic-derived
signal (`|TD|_rel`, `|Q1−Q2|`, H/2-vs-H rank stability, critic-ensemble
disagreement) predicts surrogate quality, and policy-conditioned surrogates
(PeVFA and offline alternatives) rank siblings at chance. All tuning is
HalfCheetah-only — `p_beta = 3.0` is set against its `|TD|_rel ≈ 0.21`,
and the cross-env portability the `mean|r|` normalization is *for* is
still untested.


```bash
just train semarl HalfCheetah-v5
```

## Tooling

- `justfile`: `install`, `test`, `lint`/`lint-check`, `types`, `check`,
  `train`, `train-all`.
- `slurm_run_array.sh`: array job, one `(algorithm, seed)` task per
  index; 6 algos × 5 seeds = 30 tasks for one `TARGET_ENV`.
- `slurm_semarl_sweep.sh`: array job, one `(condition, seed)` task per
  index — ERL, SEMARL at fixed `p_surr` ∈ {0, .25, .5, .75, .9}, adaptive
  SEMARL, TD3; 8 × 3 = 24 tasks. Runs `.venv/bin/python` directly, so
  `uv sync` on the login node first.
- Offline analysis tools (surrogate diagnostics, population dumps, surrogate
  and gate benchmarks, critic ensemble) were removed once their questions
  were settled; `notes.md` lists the commits to restore them from.

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
