from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3.common.vec_env import VecEnv

from src.baselines.envs import build_vec_env


def test_build_vec_env_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="unknown env 'nope'"):
        build_vec_env("nope", 1, seed=0)


def test_build_vec_env_rejects_zero_envs() -> None:
    with pytest.raises(ValueError, match="num_envs must be >= 1"):
        build_vec_env("Swimmer-v5", 0, seed=0)


def test_build_vec_env_steps() -> None:
    env = build_vec_env("Swimmer-v5", 1, seed=0)
    try:
        assert isinstance(env, VecEnv)
        assert env.num_envs == 1
        env.reset()
        _, _, _, infos = env.step(np.array([env.action_space.sample()]))
        assert len(infos) == 1
    finally:
        env.close()
