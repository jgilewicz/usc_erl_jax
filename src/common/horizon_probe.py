from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from common.critic_heads import StatsFn
from modules.deep_modules import ActorHead, SharedStateEmbedding

# metric-only: everything recorded here comes from the shadow rollout and
# must never flow back into gate, calibration, beta, CEM or the anchor


@eqx.filter_jit
def own_state_stats(
    stats: StatsFn,
    embedding: SharedStateEmbedding,
    critic: Any,
    pop_heads: ActorHead,
    states: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    # states: (pop, H, obs), each individual's own visited states;
    # returns (pop, H) mu/sigma of Q(s_t, pi_i(s_t)), shared key = CRN
    def individual(head: ActorHead, own: jax.Array) -> tuple[Any, Any]:
        actions = jax.vmap(head)(jax.vmap(embedding)(own))
        return stats(critic, own, actions, key)

    return eqx.filter_vmap(individual)(pop_heads, states)


@dataclass
class StepRecorder:
    rewards: list[jax.Array] = field(default_factory=list)
    states: list[jax.Array] = field(default_factory=list)
    done: list[jax.Array] = field(default_factory=list)

    def __call__(
        self,
        step: int,
        reward: jnp.ndarray,
        terminated: jnp.ndarray,
        truncated: jnp.ndarray,
        next_states: jnp.ndarray,
    ) -> None:
        self.rewards.append(jnp.asarray(reward))
        self.states.append(jnp.asarray(next_states))
        self.done.append(jnp.asarray(terminated | truncated))

    def stacked(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        # (pop, H), (pop, H, obs), (pop, H); index t holds s_{t+1}
        return (
            jnp.stack(self.rewards, axis=1),
            jnp.stack(self.states, axis=1),
            jnp.stack(self.done, axis=1),
        )


@dataclass
class HorizonProbe:
    path: Path
    gamma: float
    save_every: int = 10
    rows: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def record(self, **arrays: Any) -> None:
        for name, value in arrays.items():
            self.rows.setdefault(name, []).append(np.asarray(value))
        if len(self.rows["generation"]) % self.save_every == 0:
            self.save()

    def save(self) -> None:
        if not self.rows:
            return
        arrays: dict[str, Any] = {k: np.stack(v) for k, v in self.rows.items()}
        np.savez_compressed(self.path, gamma=np.asarray(self.gamma), **arrays)
