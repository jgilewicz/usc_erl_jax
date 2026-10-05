from __future__ import annotations

from typing import Any

import gymnasium as gym

from environments.base import EnvSpec, register_backend, register_env

_TASKS = (
    "myoElbowPose1D6MRandom-v0",
    "myoHandReachRandom-v0",
    "myoHandPenTwirlRandom-v0",
    "myoHandObjHoldRandom-v0",
    "myoLegWalk-v0",
)


def _make(task_id: str, /, **kwargs: Any) -> gym.Env:
    # lazy: registers the myo* ids with gymnasium, but only in workers that
    # build a myosuite env
    import myosuite  # noqa: F401

    return gym.make(task_id, **kwargs)


register_backend("myosuite", _make)
for _task in _TASKS:
    register_env(EnvSpec(name=_task, backend="myosuite", task_id=_task))
