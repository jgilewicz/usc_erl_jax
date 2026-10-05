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
   gate ── random:      real_i ~ Bernoulli(1 − omega)
        └─ uncertainty: real_i = cv_i > median(cv) + mad_k·MAD(cv)  or  ε-coin
                        cv_i = σ_i / (√|μ_i| + 1)
                   │
   rl_env  1×H  (always, stored)
   pop_env P×H  (shadow: all step, only real_i stored + counted)
                   │
   f_i = true return            if real_i
       = s·(μ_i − βσ_i) + offset  otherwise   (offset = mean bias on real_i)
                   │
   β ← Adam on residual variance over real_i   (uncertainty modes, ≥2 real)
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

- **Shadow rollout**: gymnasium vec envs can't step a subset, so every
  individual steps (≈ same wall-clock under async). Non-real returns go
  only to metrics; the algorithm sees `observed = where(real, truth, 0)`.
- **Anchor**: best-ever real-evaluated head re-enters after any
  generation with a real rollout, unless it already is the CEM elite.
- **RL injection**: every `rl_to_ea_sync_period` gens, only if `n_real > 0`.

```bash
just train sc_erl HalfCheetah-v5 algorithm.mode=ensemble
```

### Metrics (wandb, x-axis `env_steps`)

- `perf/`: `rl_return`, `eval_rl`, `eval_elite` (deterministic, every
  `eval.interval` env steps, off-budget), `pop_true_best/mean`.
- `cost/`: `env_steps_gen`, `real_frac`.
- `train/`: `critic_loss`, `actor_loss`, `critic_updates`, `buffer_size`.
- `select/`: `elite_overlap`, `rank_corr` (fitness CEM saw vs truth),
  `*_surr` (calibrated surrogate on everyone vs truth), `regret`.
- `gate/`: misranked = wrong side of the elite cut under the critic's μ
  (σ left out so the label does not contain what `cv` scores);
  `misranked_frac`, `precision`, `recall`; uncertainty modes add `auc`
  (cv → misranked), `threshold`, `cv_mean/max`, `eps_frac`.

ERL logs the same `perf/`, `cost/`, `train/`, `select/` groups.

## Tooling

- `justfile`: `install`, `lint`/`lint-check`, `types`, `check`, `train`,
  `train-all`.
- `slurm_run_array.sh`: the only slurm script; one `(condition, seed)`
  per task — sac, ppo, td3, crossq, erl, sc_erl × 4 modes; 9 × 5 = 45
  tasks. Needs `PROJECT_DIR`; runs `.venv/bin/python` directly, so
  `uv sync` on the login node first. `EXTRA="..."` appends Hydra
  overrides to every task in the array (keys must exist in that config).

## Full experiment suite on slurm

```bash
export WANDB_API_KEY=... PROJECT_DIR=/path/to/usc_erl_jax
for env in HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5 Swimmer-v5 \
           dog-stand dog-walk dog-trot dog-run \
           myoElbowPose1D6MRandom-v0 myoHandReachRandom-v0 \
           myoHandPenTwirlRandom-v0 myoHandObjHoldRandom-v0 myoLegWalk-v0; do
  TARGET_ENV="$env" sbatch --array=0-44 slurm_run_array.sh
done
```
