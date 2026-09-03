from __future__ import annotations

import warnings
from typing import Any

import gymnasium as gym
from dm_control import suite
from gymnasium.wrappers import FlattenObservation
from shimmy.dm_control_compatibility import DmControlCompatibilityV0

from src.environments.base import EnvSpec, register_backend, register_env

# dm_control's named-index code sets .shape on numpy arrays, deprecated in numpy 2.5.
warnings.filterwarnings(
    "ignore",
    message="Setting the shape on a NumPy array has been deprecated",
    category=DeprecationWarning,
    module="dm_control",
)

_DOG_TASKS = ("stand", "walk", "trot", "run")


def _make(
    task_id: str,
    /,
    *,
    render_mode: str | None = None,
    **task_kwargs: Any,
) -> gym.Env:
    domain, task = task_id.split("-", 1)
    dm_env = suite.load(
        domain_name=domain,
        task_name=task,
        task_kwargs=task_kwargs or None,
    )
    env = DmControlCompatibilityV0(dm_env, render_mode=render_mode)
    return FlattenObservation(env)


register_backend("dmc", _make)
for _task in _DOG_TASKS:
    register_env(
        EnvSpec(
            name=f"dog-{_task}",
            backend="dmc",
            task_id=f"dog-{_task}",
            expected_obs_dim=223,
            expected_act_dim=38,
        )
    )
