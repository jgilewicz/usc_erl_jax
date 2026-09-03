from __future__ import annotations

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
)

from src.environments import get_spec, make_env


def build_vec_env(name: str, num_envs: int, seed: int) -> VecEnv:
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    get_spec(name)  # fail fast on an unknown task name
    # fork deadlocks with JAX's threads, so spawn subprocesses
    vec_env_cls = DummyVecEnv if num_envs == 1 else SubprocVecEnv
    vec_env_kwargs = {} if num_envs == 1 else {"start_method": "spawn"}
    return make_vec_env(
        lambda: make_env(name),
        n_envs=num_envs,
        seed=seed,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )
