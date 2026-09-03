from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from src.environments import (
    JaxVectorEnv,
    get_spec,
    make_env,
    make_vec_env,
    registered_envs,
)

ALL_ENVS = registered_envs()
DMC_ENVS = tuple(n for n in ALL_ENVS if get_spec(n).backend == "dmc")
MUJOCO_ENVS = tuple(n for n in ALL_ENVS if get_spec(n).backend == "mujoco")
MYO_ENVS = tuple(n for n in ALL_ENVS if get_spec(n).backend == "myosuite")


def test_registry_covers_every_backend_task():
    assert set(DMC_ENVS) == {"dog-stand", "dog-walk", "dog-trot", "dog-run"}
    assert set(MUJOCO_ENVS) == {
        "HalfCheetah-v5",
        "Hopper-v5",
        "Walker2d-v5",
        "Ant-v5",
        "Swimmer-v5",
    }
    assert set(MYO_ENVS) == {
        "myoElbowPose1D6MRandom-v0",
        "myoHandReachRandom-v0",
        "myoHandPenTwirlRandom-v0",
        "myoHandObjHoldRandom-v0",
        "myoLegWalk-v0",
    }


def test_make_env_unknown_name_raises():
    with pytest.raises(KeyError, match="unknown env 'nope'"):
        make_env("nope")


def test_make_vec_env_rejects_non_positive_num_envs():
    with pytest.raises(ValueError, match="num_envs must be >= 1"):
        make_vec_env("HalfCheetah-v5", 0)


@pytest.mark.parametrize("name", MUJOCO_ENVS)
def test_mujoco_env_reset_and_step(name):
    env = make_env(name)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        obs, reward, terminated, _truncated, _info = env.step(
            env.action_space.sample()
        )
        assert np.isfinite(reward)
        assert isinstance(terminated, bool) or terminated.dtype == np.bool_
    finally:
        env.close()


@pytest.mark.parametrize("name", DMC_ENVS)
def test_dmc_dog_env_has_expected_dims(name):
    env = make_env(name)
    try:
        assert env.observation_space.shape == (223,)
        assert env.action_space.shape == (38,)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (223,)
    finally:
        env.close()


def test_myo_env_reset_and_step():
    env = make_env("myoElbowPose1D6MRandom-v0")
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        env.step(env.action_space.sample())
    finally:
        env.close()


@pytest.mark.parametrize("async_", [False, True])
def test_sync_and_async_vec_env_step(async_):
    venv = make_vec_env("HalfCheetah-v5", 2, async_=async_)
    try:
        obs, _ = venv.reset(seed=0)
        assert obs.shape == (2, 17)
        actions = np.stack(
            [venv.single_action_space.sample() for _ in range(2)]
        )
        obs, reward, _terminated, _truncated, _ = venv.step(actions)
        assert obs.shape == (2, 17)
        assert reward.shape == (2,)
    finally:
        venv.close()


def test_jax_vec_env_returns_jax_arrays():
    venv = make_vec_env("HalfCheetah-v5", 2, async_=False, to_jax=True)
    assert isinstance(venv, JaxVectorEnv)
    try:
        obs, _ = venv.reset(seed=0)
        assert isinstance(obs, jnp.ndarray)
        actions = jnp.zeros((2, 6), dtype=jnp.float32)
        obs, reward, terminated, _truncated, _ = venv.step(actions)
        assert isinstance(obs, jnp.ndarray)
        assert isinstance(reward, jnp.ndarray)
        assert terminated.dtype == jnp.bool_
    finally:
        venv.close()
