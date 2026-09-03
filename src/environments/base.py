from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Protocol

import gymnasium as gym
import jax.numpy as jnp
import numpy as np

Backend = Literal["dmc", "mujoco", "myosuite"]


class EnvBuilder(Protocol):
    def __call__(self, task_id: str, /, **kwargs: Any) -> gym.Env: ...


@dataclass(frozen=True)
class EnvSpec:
    name: str
    backend: Backend
    task_id: str
    expected_obs_dim: int | None = None
    expected_act_dim: int | None = None


_REGISTRY: dict[str, EnvSpec] = {}
_BUILDERS: dict[Backend, EnvBuilder] = {}


def register_backend(backend: Backend, builder: EnvBuilder) -> None:
    _BUILDERS[backend] = builder


def register_env(spec: EnvSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"env {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec


def registered_envs() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def get_spec(name: str) -> EnvSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown env {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def make_env(name: str, /, **kwargs: Any) -> gym.Env:
    spec = get_spec(name)
    env = _BUILDERS[spec.backend](spec.task_id, **kwargs)
    _check_spec(env, spec)
    return env


def make_vec_env(
    name: str,
    num_envs: int,
    *,
    async_: bool = True,
    to_jax: bool = False,
    env_kwargs: dict[str, Any] | None = None,
) -> gym.vector.VectorEnv:
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    get_spec(name)
    env_fns = [
        partial(make_env, name, **(env_kwargs or {})) for _ in range(num_envs)
    ]
    if async_:
        venv: gym.vector.VectorEnv = gym.vector.AsyncVectorEnv(
            env_fns, context="spawn"
        )
    else:
        venv = gym.vector.SyncVectorEnv(env_fns)
    return JaxVectorEnv(venv) if to_jax else venv


def _check_spec(env: gym.Env, spec: EnvSpec) -> None:
    obs, act = env.observation_space, env.action_space
    if (
        not isinstance(obs, gym.spaces.Box)
        or obs.shape is None
        or len(obs.shape) != 1
    ):
        raise TypeError(
            f"{spec.name}: expected a 1-D Box observation, got {obs}"
        )
    if (
        not isinstance(act, gym.spaces.Box)
        or act.shape is None
        or len(act.shape) != 1
    ):
        raise TypeError(f"{spec.name}: expected a 1-D Box action, got {act}")
    if (
        spec.expected_obs_dim is not None
        and obs.shape[0] != spec.expected_obs_dim
    ):
        raise ValueError(
            f"{spec.name}: observation dim {obs.shape[0]} "
            f"!= expected {spec.expected_obs_dim}"
        )
    if (
        spec.expected_act_dim is not None
        and act.shape[0] != spec.expected_act_dim
    ):
        raise ValueError(
            f"{spec.name}: action dim {act.shape[0]} "
            f"!= expected {spec.expected_act_dim}"
        )


class JaxVectorEnv(gym.vector.VectorWrapper):
    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        # ty: ignore[invalid-argument-type]
        obs, info = self.env.reset(seed=seed, options=options)
        return jnp.asarray(obs), info

    def step(
        self, actions: Any
    ) -> tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, Any]
    ]:
        obs, reward, terminated, truncated, info = self.env.step(
            np.asarray(actions)
        )
        return (
            jnp.asarray(obs),
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(terminated, dtype=jnp.bool_),
            jnp.asarray(truncated, dtype=jnp.bool_),
            info,
        )
