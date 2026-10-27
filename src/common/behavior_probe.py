from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax

from common.horizon_probe import HorizonProbe
from common.replay_buffer import Buffer
from modules.deep_modules import ActorHead, SharedStateEmbedding

# metric-only: own key, own state set; nothing recorded here may flow back
# into gate, calibration, beta, CEM or the anchor

STATES = 512
# disjoint from the run's key stream (jax.random.key(seed))
_KEY_OFFSET = 7_919


@eqx.filter_jit
def behaviour(
    embedding: SharedStateEmbedding, heads: ActorHead, states: jax.Array
) -> jax.Array:
    # (n, S, act_dim): every head on the same states, current embedding
    z = jax.vmap(embedding)(states)
    return eqx.filter_vmap(lambda head: jax.vmap(head)(z))(heads)


@dataclass
class BehaviorProbe:
    writer: HorizonProbe
    key: jax.Array
    states: jax.Array | None = None

    @classmethod
    def create(cls, path: Path, gamma: float, seed: int) -> BehaviorProbe:
        key = jax.random.key(seed + _KEY_OFFSET)
        return cls(HorizonProbe(path, gamma), key)

    def record(
        self,
        buffer: Buffer,
        embedding: SharedStateEmbedding,
        pop_heads: ActorHead,
        mean_head: ActorHead,  # (1,)-stacked CEM mean as asked
        flat_pop: jax.Array,
        **scalars: jax.Array | int,
    ) -> None:
        # S is drawn once, on the first post-warmup generation, then fixed
        if self.states is None:
            self.states = buffer.sample(self.key, STATES)["state"]
        self.writer.record(
            actions=behaviour(embedding, pop_heads, self.states),
            mean_actions=behaviour(embedding, mean_head, self.states)[0],
            heads=flat_pop,
            **scalars,
        )
