from __future__ import annotations

from typing import Any

import gymnasium as gym
import myosuite  # noqa: F401  # import registers the myo* env ids with gymnasium

from src.environments.base import EnvSpec, register_backend, register_env

_TASKS = (
    "myoElbowPose1D6MRandom-v0",
    "myoHandReachRandom-v0",
    "myoHandPenTwirlRandom-v0",
    "myoHandObjHoldRandom-v0",
    "myoLegWalk-v0",
)


def _make(task_id: str, /, **kwargs: Any) -> gym.Env:
    return gym.make(task_id, **kwargs)


register_backend("myosuite", _make)
for _task in _TASKS:
    register_env(EnvSpec(name=_task, backend="myosuite", task_id=_task))
