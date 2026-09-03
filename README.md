# triage-erl

Unified gymnasium-style environment wrapper over three simulator backends, with
vectorized (parallel) execution and an optional JAX bridge.

## Backends & tasks

| backend    | registry names |
|------------|----------------|
| `dmc` (DeepMind Control Suite) | `dog-stand`, `dog-walk`, `dog-trot`, `dog-run` — obs 223, act 38 |
| `mujoco` (Gymnasium MuJoCo v5) | `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`, `Swimmer-v5` |
| `myosuite` | `myoElbowPose1D6MRandom-v0`, `myoHandReachRandom-v0`, `myoHandPenTwirlRandom-v0`, `myoHandObjHoldRandom-v0`, `myoLegWalk-v0` |

All observations are flattened to a 1-D `Box`; all actions are a 1-D `Box`.

## Usage

```python
from src.environments import make_env, make_vec_env, registered_envs

registered_envs()                      # every task name

env = make_env("dog-walk")             # single gymnasium.Env
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

venv = make_vec_env("HalfCheetah-v5", num_envs=8)   # AsyncVectorEnv (subprocesses)
venv = make_vec_env("Hopper-v5", 4, async_=False)   # SyncVectorEnv (one process)

# JAX bridge: reset/step return jnp arrays instead of numpy
venv = make_vec_env("dog-run", 8, to_jax=True)
obs, info = venv.reset(seed=0)          # obs is a jax.Array of shape (8, 223)
```

The async path uses the `spawn` start method (fork deadlocks with JAX's
threads), so scripts that call `make_vec_env(..., async_=True)` must guard entry
with `if __name__ == "__main__":`.

`make_env` validates that the built env matches the registered `EnvSpec`
(`dog-*` must be 223/38); a mismatch raises immediately.

## Baselines (SBX)

`src/baselines/` wraps [SBX](https://stable-baselines3.readthedocs.io/en/master/guide/sbx.html)
(Stable-Baselines3 for JAX) into Hydra-configured baselines over the registry
envs. Algorithms: `sac`, `ppo`, `td3`, `crossq` (all `MlpPolicy`).

```
just train sac Hopper-v5
just train crossq Ant-v5 seed=3 total_steps=2_000_000 wandb.enabled=false
just train-all Swimmer-v5            # sac, ppo, td3, crossq in sequence
```

`just train <algo> <env> [hydra overrides...]` runs
`python -m src.baselines.train`. Any config key is overridable on the CLI
(`algorithm.learning_rate=1e-4`, `n_envs=8`, `eval.interval=20000`).

### Config

Hydra config tree lives in `src/conf/`:

```
src/conf/
  config.yaml            seed, total_steps, n_envs, device, env/eval_env, eval, wandb, result_file
  algorithm/
    sac.yaml  ppo.yaml  td3.yaml  crossq.yaml   # @package algorithm — one file per algo
```

`config.yaml` selects `algorithm: sac` by default. `wandb` block: `project`
(`triage_erl`), `entity` (`evo_rl`), `name`, `tags`, `enabled` (false →
`mode=disabled`). `result_file: <path>` writes `{"eval_reward": float}` after
training, for hyperparameter sweeps.

`n_envs > 1` switches the training env to `SubprocVecEnv` (spawn); off-policy
algos default to 1. `EvalCallback` runs `eval.episodes` on a separate env every
`eval.interval` steps.

### wandb metrics

`WandbLoggingCallback` attaches a `KVWriter` to the SB3 logger that forwards a
fixed set of keys to `wandb.log` under canonical names, stepped by timesteps:

| wandb key      | source (first present wins)                    |
|----------------|------------------------------------------------|
| `total_steps`  | `time/total_timesteps`                         |
| `eval_reward`  | `eval/mean_reward`                             |
| `n_updates`    | `train/n_updates`                              |
| `actor_loss`   | `train/actor_loss` · `train/pg_loss` (PPO)     |
| `critic_loss`  | `train/critic_loss` · `train/value_loss` (PPO) |

Programmatic use:

```python
from src.baselines import build_vec_env, build_agent, WandbLoggingCallback

env = build_vec_env("Hopper-v5", num_envs=1, seed=0)
model = build_agent(algo_cfg, env, seed=0)   # algo_cfg: a src/conf/algorithm/*.yaml node
model.learn(200_000, callback=WandbLoggingCallback())   # after wandb.init()
```

## Layout

```
src/environments/
  base.py                  registry, EnvSpec, make_env, make_vec_env, JaxVectorEnv
  deepmind_control_env.py   dm_control + shimmy + FlattenObservation
  mujoco_env.py             gymnasium.make
  myo_suite_env.py          myosuite (import registers the ids)
src/baselines/
  agents.py                build_agent — SBX SAC/PPO/TD3/CrossQ from a config node
  envs.py                  build_vec_env — SB3 VecEnv over registry make_env
  wandb_logging.py         WandbOutputFormat (KVWriter) + WandbLoggingCallback
  train.py                 @hydra.main entrypoint + run_training(cfg)
src/conf/                  Hydra config tree (config.yaml + algorithm/*.yaml)
tests/environments/
tests/baselines/
```

## Environment

Python **3.12** (`uv sync`). Not 3.13: `myosuite` caps at `<3.14` and its
transitive `labmaze` ships no 3.13 wheel (source build needs Bazel). Pinned:
`gymnasium 1.2.3`, `mujoco 3.6.0`, `dm-control 1.0.38`, `shimmy 2.0.1`,
`myosuite 2.12.2` — the only mutually compatible set (`myosuite` needs
`mujoco<3.7`, which forces `dm-control<=1.0.38`).

```
uv sync
just check      # ruff (lint + format), ty check, pytest
```

`just` recipes: `install`, `test [args]`, `lint`, `lint-check`, `types`,
`check`, `train <algo> <env> [overrides]`, `train-all <env> [overrides]`.

## Known gaps

- DMC seeding: `dm_control` draws its own RNG from `task_kwargs["random"]`;
  `reset(seed=...)` does not thread through to it, so vectorized `dmc` envs
  share an initial-state distribution. Pass `env_kwargs={"random": <int>}` to
  `make_vec_env` if per-run determinism matters.
- `src/modules/deep_modules.py` is an unfinished stub (undefined `elu`, unused
  keys) — unrelated to the environments code.
