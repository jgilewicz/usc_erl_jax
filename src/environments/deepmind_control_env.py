from __future__ import annotations

import warnings
from typing import Any

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from environments.base import EnvSpec, register_backend, register_env

_DOG_TASKS = ("stand", "walk", "trot", "run")


def _make(
    task_id: str,
    /,
    *,
    render_mode: str | None = None,
    **task_kwargs: Any,
) -> gym.Env:
    # backend imports stay inside the builder: every env worker process
    # imports this module, and only dmc workers should pay for dm_control
    from dm_control import suite
    from shimmy.dm_control_compatibility import DmControlCompatibilityV0

    domain, task = task_id.split("-", 1)
    dm_env = suite.load(
        domain_name=domain,
        task_name=task,
        task_kwargs=task_kwargs or None,
    )
    # after the import: dm_control.composer.environment runs
    # simplefilter("always", DeprecationWarning) at import time, which
    # turned its numpy-2.5 `.shape =` warning into GBs of slurm stderr
    warnings.filterwarnings(
        "ignore",
        message="Setting the shape on a NumPy array has been deprecated",
        category=DeprecationWarning,
        module="dm_control",
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
