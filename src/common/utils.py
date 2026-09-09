from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

Params = TypeVar("Params")


@eqx.filter_jit
def genetic_soft_update(
    rl_params: Params, elite_params: Params, tau: float
) -> Params:
    rl_arrays, rl_static = eqx.partition(rl_params, eqx.is_array)
    elite_arrays, _ = eqx.partition(elite_params, eqx.is_array)
    merged = jax.tree.map(
        lambda rl, el: tau * el + (1 - tau) * rl, rl_arrays, elite_arrays
    )
    return eqx.combine(merged, rl_static)


@eqx.filter_jit
def h_step_bootstrap(
    h_steps: int,
    gamma: float,
    rewards: jax.Array,
    dones: jax.Array,
    q_value: jax.Array,
) -> jax.Array:
    survived = jnp.cumprod(1.0 - dones, axis=-1)
    alive_mask = jnp.concatenate(
        [jnp.ones_like(survived[..., :1]), survived[..., :-1]], axis=-1
    )
    discounted_rewards = jnp.sum(
        jnp.pow(gamma, jnp.arange(h_steps)) * rewards * alive_mask, axis=-1
    )
    bootstrap_mask = survived[..., -1]
    return (
        discounted_rewards + jnp.pow(gamma, h_steps) * q_value * bootstrap_mask
    )
