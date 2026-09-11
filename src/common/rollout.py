from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from common.replay_buffer import Buffer, Transition

BatchPolicy = Callable[[jax.Array, jnp.ndarray], jnp.ndarray]
# (step, states, reward, terminated, truncated, next_states)
StepHook = Callable[
    [int, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    None,
]


@jax.jit
def _update_returns(
    returns: jnp.ndarray,
    alive: jnp.ndarray,
    reward: jnp.ndarray,
    terminated: jnp.ndarray,
    truncated: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    returns = returns + reward * alive
    alive = alive & ~(terminated | truncated)
    return returns, alive


def collect_parallel_episode(
    vec_env: gym.vector.VectorEnv,
    key: jax.Array,
    policy: BatchPolicy,
    buffer: Buffer,
    horizon: int,
    *,
    on_step: StepHook | None = None,
) -> tuple[jnp.ndarray, jax.Array]:
    states, _ = vec_env.reset(
        seed=int(jax.random.randint(key, (), 0, 2**31 - 1))
    )
    states = jnp.asarray(states)
    num_envs = states.shape[0]
    returns = jnp.zeros(num_envs)
    alive = jnp.ones(num_envs, dtype=bool)
    prev_done = np.zeros(num_envs, dtype=bool)

    for step in range(horizon):
        key, act_key = jax.random.split(key)
        actions = policy(act_key, states)
        next_states, reward, terminated, truncated, _ = vec_env.step(actions)
        returns, alive = _update_returns(
            returns, alive, reward, terminated, truncated
        )
        if on_step is not None:
            on_step(step, states, reward, terminated, truncated, next_states)

        valid = ~prev_done
        if valid.any():
            idx = jnp.asarray(np.nonzero(valid)[0])
            buffer.add(
                Transition(
                    state=states[idx],
                    action=jnp.asarray(actions)[idx],
                    reward=reward[idx],
                    next_state=next_states[idx],
                    done=terminated[idx],
                )
            )

        prev_done = np.asarray(terminated | truncated)
        states = next_states

    return returns, key
