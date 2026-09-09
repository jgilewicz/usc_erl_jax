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


@jax.jit
def _scatter(
    state: jnp.ndarray,
    action: jnp.ndarray,
    reward: jnp.ndarray,
    next_state: jnp.ndarray,
    done: jnp.ndarray,
    indices: jnp.ndarray,
    transition: Transition,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return (
        state.at[indices].set(transition.state),
        action.at[indices].set(transition.action),
        reward.at[indices].set(transition.reward),
        next_state.at[indices].set(transition.next_state),
        done.at[indices].set(transition.done),
    )


@jax.jit
def _gather(
    state: jnp.ndarray,
    action: jnp.ndarray,
    reward: jnp.ndarray,
    next_state: jnp.ndarray,
    done: jnp.ndarray,
    indices: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    return {
        "state": state[indices],
        "action": action[indices],
        "reward": reward[indices],
        "next_state": next_state[indices],
        "done": done[indices],
    }


class Buffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.state = jnp.zeros((capacity, state_dim), dtype=jnp.float32)
        self.action = jnp.zeros((capacity, action_dim), dtype=jnp.float32)
        self.reward = jnp.zeros((capacity, 1), dtype=jnp.float32)
        self.next_state = jnp.zeros((capacity, state_dim), dtype=jnp.float32)
        self.done = jnp.zeros((capacity, 1), dtype=jnp.float32)
        self.ptr = 0
        self.size = 0

    def add(self, transition: Transition) -> None:
        num_envs = transition.state.shape[0]
        indices = (self.ptr + jnp.arange(num_envs)) % self.capacity
        transition = Transition(
            state=transition.state.astype(jnp.float32),
            action=transition.action.astype(jnp.float32),
            reward=transition.reward.reshape(num_envs, 1).astype(jnp.float32),
            next_state=transition.next_state.astype(jnp.float32),
            done=transition.done.reshape(num_envs, 1).astype(jnp.float32),
        )
        self.state, self.action, self.reward, self.next_state, self.done = (
            _scatter(
                self.state,
                self.action,
                self.reward,
                self.next_state,
                self.done,
                indices,
                transition,
            )
        )
        self.ptr = (self.ptr + num_envs) % self.capacity
        self.size = min(self.size + num_envs, self.capacity)

    def sample(self, rng: jax.Array, batch_size: int) -> dict[str, jnp.ndarray]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        indices = jax.random.randint(rng, (batch_size,), 0, self.size)
        return self[indices]

    def __getitem__(self, indices: jnp.ndarray) -> dict[str, jnp.ndarray]:
        return _gather(
            self.state,
            self.action,
            self.reward,
            self.next_state,
            self.done,
            indices,
        )

    def __len__(self) -> int:
        return self.size
