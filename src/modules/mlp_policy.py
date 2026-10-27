from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

Layers = list[tuple[jax.Array, jax.Array]]
Unravel = Callable[[jax.Array], Layers]


def init_layers(key: jax.Array, sizes: tuple[int, ...]) -> Layers:
    keys = jax.random.split(key, len(sizes) - 1)
    init = jax.nn.initializers.lecun_normal()
    return [
        (init(k, (fan_in, fan_out)), jnp.zeros(fan_out))
        for k, fan_in, fan_out in zip(keys, sizes[:-1], sizes[1:])
    ]


def flat_spec(sizes: tuple[int, ...]) -> tuple[int, Unravel]:
    flat, unravel = ravel_pytree(init_layers(jax.random.key(0), sizes))
    return flat.size, unravel


def init_population(
    key: jax.Array, sizes: tuple[int, ...], n: int
) -> jax.Array:
    def one(k: jax.Array) -> jax.Array:
        return ravel_pytree(init_layers(k, sizes))[0]

    return jax.vmap(one)(jax.random.split(key, n))


def act(layers: Layers, obs: jax.Array) -> jax.Array:
    x = obs
    for w, b in layers:
        x = jnp.tanh(x @ w + b)
    return x
