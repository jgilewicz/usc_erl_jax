from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from common.replay_buffer import Buffer
from modules.deep_modules import ActorHead, SharedStateEmbedding

_N_PROBE = 64


@eqx.filter_jit
def _fingerprint(
    embedding: SharedStateEmbedding,
    heads: ActorHead,
    probe_states: jnp.ndarray,
) -> jnp.ndarray:
    # (pop, n_probe, action_dim): every individual's action on the same states
    z = jax.vmap(embedding)(probe_states)
    return jax.vmap(lambda head: jax.vmap(head)(z))(heads)


class PopulationDump:
    """Per-generation population record for offline surrogate benchmarks."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.fields: dict[str, list[np.ndarray]] = defaultdict(list)
        self.probe_states: np.ndarray | None = None

    def fingerprint(
        self,
        buffer: Buffer,
        embedding: SharedStateEmbedding,
        heads: ActorHead,
    ) -> jnp.ndarray:
        if self.probe_states is None:
            # drawn once: fingerprints compare across generations only
            # when taken on the same states
            batch = buffer.sample(jax.random.key(0), _N_PROBE)
            self.probe_states = np.asarray(batch["state"])
        return _fingerprint(embedding, heads, jnp.asarray(self.probe_states))

    def record(self, **fields: object) -> None:
        for name, value in fields.items():
            self.fields[name].append(np.asarray(value))

    def save(self) -> None:
        if not self.fields:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: np.stack(v) for name, v in self.fields.items()}
        if self.probe_states is not None:
            arrays["probe_states"] = self.probe_states
        # ty matches **arrays against `allow_pickle: bool` too; the keys
        # are field names, never allow_pickle
        np.savez_compressed(self.path, **arrays)  # ty: ignore[invalid-argument-type]
        n = len(self.fields["generation"])
        print(f"population dump: {n} generations -> {self.path}")
