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

## SEMARL (placeholder)

[SEMARL](https://dl.acm.org/doi/epdf/10.1145/3795095.3805146) — not yet
implemented.

## Tooling

- `justfile`: `install`, `test`, `lint`/`lint-check`, `types`, `check`,
  `train`, `train-all`.
- `slurm_run_array.sh`: array job, one `(algorithm, seed)` task per
  index; 5 algos × 5 seeds = 25 tasks for one `TARGET_ENV`.

## Full experiment suite on slurm

```bash
export WANDB_API_KEY=...
for env in HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5 Swimmer-v5 \
           dog-stand dog-walk dog-trot dog-run \
           myoElbowPose1D6MRandom-v0 myoHandReachRandom-v0 \
           myoHandPenTwirlRandom-v0 myoHandObjHoldRandom-v0 myoLegWalk-v0; do
  TARGET_ENV="$env" sbatch --array=0-24 slurm_run_array.sh
done
```
