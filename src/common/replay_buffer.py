from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Transition(NamedTuple):
    state: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_state: jnp.ndarray
    done: jnp.ndarray


class BufferState(NamedTuple):
    state: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_state: jnp.ndarray
    done: jnp.ndarray
    ptr: jnp.ndarray
    size: jnp.ndarray


def create(capacity: int, state_dim: int, action_dim: int) -> BufferState:
    return BufferState(
        state=jnp.zeros((capacity, state_dim), dtype=jnp.float32),
        action=jnp.zeros((capacity, action_dim), dtype=jnp.float32),
        reward=jnp.zeros((capacity, 1), dtype=jnp.float32),
        next_state=jnp.zeros((capacity, state_dim), dtype=jnp.float32),
        done=jnp.zeros((capacity, 1), dtype=jnp.float32),
        ptr=jnp.zeros((), dtype=jnp.int32),
        size=jnp.zeros((), dtype=jnp.int32),
    )


def add(buffer: BufferState, transition: Transition) -> BufferState:
    capacity = buffer.state.shape[0]
    num_envs = transition.state.shape[0]
    indices = (buffer.ptr + jnp.arange(num_envs)) % capacity

    reward = transition.reward.reshape(num_envs, 1).astype(jnp.float32)
    done = transition.done.reshape(num_envs, 1).astype(jnp.float32)

    return buffer._replace(
        state=buffer.state.at[indices].set(
            transition.state.astype(jnp.float32)
        ),
        action=buffer.action.at[indices].set(
            transition.action.astype(jnp.float32)
        ),
        reward=buffer.reward.at[indices].set(reward),
        next_state=buffer.next_state.at[indices].set(
            transition.next_state.astype(jnp.float32)
        ),
        done=buffer.done.at[indices].set(done),
        ptr=(buffer.ptr + num_envs) % capacity,
        size=jnp.minimum(buffer.size + num_envs, capacity),
    )


def sample(
    buffer: BufferState, rng: jax.Array, batch_size: int
) -> dict[str, jnp.ndarray]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    indices = jax.random.randint(rng, (batch_size,), 0, buffer.size)
    return {
        "state": buffer.state[indices],
        "action": buffer.action[indices],
        "reward": buffer.reward[indices],
        "next_state": buffer.next_state[indices],
        "done": buffer.done[indices],
    }
