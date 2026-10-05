from __future__ import annotations

from typing import Any

import gymnasium as gym
import jax.numpy as jnp
import numpy as np


class JaxVectorEnv(gym.vector.VectorWrapper):
    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        # ty: ignore[invalid-argument-type]
        obs, info = self.env.reset(seed=seed, options=options)
        return jnp.asarray(obs), info

    def step(
        self, actions: Any
    ) -> tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, Any]
    ]:
        obs, reward, terminated, truncated, info = self.env.step(
            np.asarray(actions)
        )
        return (
            jnp.asarray(obs),
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(terminated, dtype=jnp.bool_),
            jnp.asarray(truncated, dtype=jnp.bool_),
            info,
        )
