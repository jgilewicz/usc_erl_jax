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
  (`h_step_bootstrap`, `h_steps`).

```bash
just train erl HalfCheetah-v5
```

## SC-ERL

Port of usc_erl's `SurrogateController` onto the ERL backbone (shared
embedding, CEM, genetic soft update). Each **individual** — not the whole
generation — is either rolled out or scored by the critic. Impl:
`src/algos/sc_erl.py`, config: `src/conf/algorithm/sc_erl.yaml` (inherits
`erl.yaml`; `theta`/`h_steps` inert).

```text
CEM.ask() ──► pop × ActorHead   (slot 0: RL actor, 1: best-ever real, -1: elite)
                   │
   μ_i, σ_i = E_{s~D}[critic1 stats(s, π_i(s))]   (shared replay batch, vmapped)
                   │
   gate ── random:      real_i ~ Bernoulli(1 − omega)   (omega 0.79)
        └─ uncertainty: real_i = cv_i > median(cv) + mad_k·MAD(cv)  or  ε-coin
                        cv_i = σ_i / (√|μ_i| + 1)
                   │
   rl_env  1×H  (always, stored)
   pop_env P×H  (shadow: all step, only real_i stored + counted)
                   │
   f_i = true return            if real_i
       = s·(μ_i − βσ_i) + offset  otherwise   (offset = mean bias on real_i)
                   │
   β ← Adam on residual variance over real_i   (modes with σ, ≥2 real)
   CEM.tell(f), elite only from real_i
   TD3 × train_ratio·(1 + n_real)·H   (one jit: lax.fori_loop)
```

| mode | critic1 | σ |
| --- | --- | --- |
| `random` | `Critic` | none (σ = 0) |
| `dropout` | `Critic(dropout=dropout_p)`, twin too | std over `mc_samples` MC passes |
| `ensemble` | `k_ensembles` vmapped `Critic`s, bootstrapped Huber | std over members |
| `evidential` | `EvidentialCritic`, NIG loss (`evidential_lam`) | √(β/(v(α−1))) |

critic2 is always a plain `Critic` (MSE); the TD3 target is
`min(point(critic1), critic2)`, the actor follows `point(critic1)`.

- **Critic-vs-gate ablation**: `algorithm.random_gate=true` keeps the
  mode's critic (TD3, σ, β fit, `gate/auc` logging) but gates by `omega`.
  The mode changes TD3 itself, so a mode beating `random` does not
  credit the gate until it also beats its own `random_gate` run.
- **Equal budget**: `omega` 0.79 matches the ~21% real rate the
  uncertainty gate measured on dog-stand.
- **Shadow rollout**: gymnasium vec envs can't step a subset, so every
  individual steps (≈ same wall-clock under async). Non-real returns go
  only to metrics; the algorithm sees `observed = where(real, truth, 0)`.
- **Anchor**: best-ever real-evaluated head re-enters after any
  generation with a real rollout, unless it already is the CEM elite.
- **RL injection**: every `rl_to_ea_sync_period` gens, only if `n_real > 0`.

```bash
just train sc_erl HalfCheetah-v5 algorithm.mode=ensemble
```

### Horizon probe (offline, equal budget)

`algorithm.horizon_probe=true` (σ-modes) writes
`<hydra run dir>/horizon_probe.npz` per generation:
- per-step shadow rewards, critic1 μ/σ on each individual's **own** visited
  states, replay-batch μ/σ, truth.
- metric only, never fed back to the algorithm.

Offline analysis of the dump lives outside the repo (results in `notes.md`).

### Metrics (wandb, x-axis `env_steps`)

Ground truth = the undiscounted return of one full episode per individual
(shadow rollout in SC-ERL, so it exists for every individual). Elite =
top `parents = pop_size // 2` by score; chance elite overlap = 0.5.

| metric | measures | function | truth |
| --- | --- | --- | --- |
| `perf/eval_rl` | policy quality, RL actor | mean deterministic return, `eval.episodes` episodes every `eval.interval` steps, off-budget | – |
| `perf/eval_elite` | policy quality, CEM elite | same, for the CEM elite head | – |
| `cost/real_frac` | evaluation cost | real rollouts / `pop_size` (ERL: 1 or 0 per gen) | – |
| `select/elite_overlap` | selection quality | share of the elite under the fitness CEM saw that is also truth-elite | per-individual true return |
| `select/elite_overlap_surr` | surrogate quality | same, for the calibrated surrogate on everyone (ERL: h-step bootstrap) | per-individual true return |
| `select/rank_corr_surr` | surrogate quality, whole ranking | Spearman(surrogate, truth), average ranks | per-individual true return |
| `gate/misranked_frac` | base rate | share of individuals μ puts on the wrong side of the elite cut | truth elite membership |
| `gate/precision` | gate quality | share of gated-real individuals that were misranked; vs `misranked_frac` = lift over random | misranked label |
| `gate/auc` | σ as a signal (σ-modes, under either gate) | P(cv of a misranked > cv of a correct one), ties ½ | misranked label |
| `train/critic_loss` | critic health (divergence) | mean TD loss over the generation's updates | – |

SC-ERL logs all; ERL logs `perf/`, `cost/`, `select/`, `train/`; SBX
baselines log `eval_reward`, `total_steps`, `critic_loss`. `select/` and
`gate/` start after warmup.

### Findings so far

Uncertainty of the critic did not help anywhere it was tried (dog-stand,
seed 0; details in the gitignored `notes.md`):
- gate AUC ≈ 0.5 for all three σ-modes, also on each individual's own
  states; uncertainty gate = `random_gate` in performance;
- σ does not track the critic's Q error even per state;
- gains of `ensemble` / `evidential` over `random` come from the critic
  in TD3, not from the gate;
- MC-dropout critics diverge around 640–700k steps.

## Tooling

- `justfile`: `install`, `lint`/`lint-check`, `types`, `check`, `train`,
  `train-all`.
- `slurm_run_array.sh`: the only slurm script; one `(condition, seed)`
  per task — sac, ppo, td3, crossq, erl, `sc_erl-random`, 3 σ-modes with
  the uncertainty gate, 3 with `random_gate`; 12 × 5 = 60 tasks. Run
  `sbatch` from the checkout (or set `PROJECT_DIR`); it runs
  `.venv/bin/python` directly, so `uv sync` on the login node first.
  `EXTRA="..."` appends Hydra overrides to every task in the array (keys
  must exist in that config, so not with the SBX baselines).
  wandb logs online by default (`WANDB_MODE=offline` syncs after the run).
- `experiments/<tag>/experiment.toml`: grid, metric, planned comparisons
  and job IDs for the global `wcss-experiment` Claude skill (supervision
  via `squeue`/`sacct` + wandb, HTML report). `cache/`, `report.html`,
  `record.md` are generated and gitignored.

## Full experiment suite on slurm

14 envs × 60 tasks = 840 runs (thesis matrix, tag `thesis-v1`):

```bash
export WANDB_API_KEY=...
submit() { TARGET_ENV="$1" TAG=thesis-v1 sbatch --array=0-59%20 \
  --cpus-per-task="$2" --mem="$3" slurm_run_array.sh; }
for e in HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5 Swimmer-v5; do
  submit "$e" 8 16gb; done
for e in dog-stand dog-walk dog-trot dog-run; do submit "$e" 16 32gb; done
for e in myoElbowPose1D6MRandom-v0 myoHandReachRandom-v0 \
         myoHandPenTwirlRandom-v0 myoHandObjHoldRandom-v0 myoLegWalk-v0; do
  submit "$e" 8 24gb; done
```

dog-* needs the memory: ~0.55 GB per env worker, pop + 1 + 2·eval workers.
