from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
from brax import envs
from brax import base
from brax.envs.base import PipelineEnv, State

from modules.mlp_policy import Unravel, act

# brax 0.14 warns on every env build that brax itself is unmaintained; the
# mjx backend it wraps is maintained and is what the QD benchmarks use
warnings.filterwarnings(
    "ignore", message="Brax System, piplines", category=UserWarning
)

StepFn = Callable[[State], jax.Array]


@dataclass(frozen=True)
class QDTask:
    env: PipelineEnv
    episode_length: int
    desc_low: jax.Array
    desc_high: jax.Array
    fitness: StepFn
    descriptor: StepFn
    # "mean": time-average over alive steps; "last": last alive step
    reduce: str


def _pipeline(state: State) -> base.State:
    assert state.pipeline_state is not None
    return state.pipeline_state


def _make_env(name: str) -> PipelineEnv:
    return cast(PipelineEnv, envs.get_environment(name, backend="mjx"))


def _feet_contact(env: PipelineEnv, feet: tuple[str, ...]) -> StepFn:
    names = [
        env.sys.mj_model.geom(i).name for i in range(env.sys.mj_model.ngeom)
    ]
    ids = jnp.array([names.index(f) for f in feet])
    floor = names.index("floor")

    def contact(state: State) -> jax.Array:
        c = _pipeline(state).contact
        assert c is not None
        touching = c.dist < 0
        pairs = c.geom

        def foot(fid: jax.Array) -> jax.Array:
            hit = ((pairs[:, 0] == fid) & (pairs[:, 1] == floor)) | (
                (pairs[:, 1] == fid) & (pairs[:, 0] == floor)
            )
            return jnp.any(hit & touching).astype(jnp.float32)

        return jax.vmap(foot)(ids)

    return contact


def _walker2d_uni() -> QDTask:
    env = _make_env("walker2d")
    return QDTask(
        env=env,
        episode_length=1000,
        desc_low=jnp.zeros(2),
        desc_high=jnp.ones(2),
        fitness=lambda s: s.reward,
        descriptor=_feet_contact(env, ("foot_geom", "foot_left_geom")),
        reduce="mean",
    )


def _ant_omni() -> QDTask:
    env = _make_env("ant")
    return QDTask(
        env=env,
        episode_length=250,
        desc_low=jnp.full(2, -30.0),
        desc_high=jnp.full(2, 30.0),
        # omni: no forward term, the descriptor is where the ant ends up
        fitness=lambda s: (
            s.metrics["reward_survive"] + s.metrics["reward_ctrl"]
        ),
        descriptor=lambda s: _pipeline(s).q[:2],
        reduce="last",
    )


TASKS: dict[str, Callable[[], QDTask]] = {
    "walker2d-uni": _walker2d_uni,
    "ant-omni": _ant_omni,
}


def make_task(name: str) -> QDTask:
    if name not in TASKS:
        raise ValueError(f"unknown QD task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]()


def rollout(
    task: QDTask, unravel: Unravel, genome: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    layers = unravel(genome)
    dim = task.desc_low.size
    init = task.env.reset(key)
    zero = jnp.zeros(dim)

    def step(
        carry: tuple[
            State, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
        ],
        _: None,
    ) -> tuple[
        tuple[State, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        None,
    ]:
        state, alive, fit, total, last, n = carry
        nxt = task.env.step(state, act(layers, cast(jax.Array, state.obs)))
        desc = task.descriptor(nxt)
        fit = fit + alive * task.fitness(nxt)
        total = total + alive * desc
        last = jnp.where(alive > 0, desc, last)
        n = n + alive
        alive = alive * (1.0 - nxt.done)
        return (nxt, alive, fit, total, last, n), None

    carry = (
        init,
        jnp.float32(1.0),
        jnp.float32(0.0),
        zero,
        zero,
        jnp.float32(0.0),
    )
    (_, _, fit, total, last, n), _ = jax.lax.scan(
        step, carry, None, length=task.episode_length
    )
    desc = total / jnp.maximum(n, 1.0) if task.reduce == "mean" else last
    return fit, jnp.clip(desc, task.desc_low, task.desc_high)


def evaluate(
    task: QDTask,
    unravel: Unravel,
    genomes: jax.Array,
    key: jax.Array,
    chunk: int,
) -> tuple[jax.Array, jax.Array]:
    # one fresh reset key per episode: reset noise is the uncertainty source
    keys = jax.random.split(key, genomes.shape[0])

    def one(args: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        return rollout(task, unravel, *args)

    return jax.lax.map(one, (genomes, keys), batch_size=chunk)
