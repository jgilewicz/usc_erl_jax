from __future__ import annotations

from typing import Any

import gymnasium as gym

from src.environments.base import EnvSpec, register_backend, register_env

_TASKS = (
    "HalfCheetah-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Ant-v5",
    "Swimmer-v5",
)


def _make(task_id: str, /, **kwargs: Any) -> gym.Env:
    return gym.make(task_id, **kwargs)


register_backend("mujoco", _make)
for _task in _TASKS:
    register_env(EnvSpec(name=_task, backend="mujoco", task_id=_task))
