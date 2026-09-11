import math
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
    # h_steps=0 is the pure-critic surrogate: no real reward, and nothing has
    # terminated yet, so the bootstrap is unmasked. Guarded because the
    # general path indexes survived[..., -1], which has no last element here.
    if h_steps == 0:
        return q_value
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


@eqx.filter_jit
def absolute_td_error(
    reward: jax.Array,
    gamma: float,
    q_prim: jax.Array,
    q: jax.Array,
    dones: jax.Array,
):
    return jnp.abs(reward + gamma * (1 - dones) * q_prim - q).mean()


def relative_td_error(abs_td_error: float, reward_scale: float) -> float:
    return abs_td_error / (reward_scale + 1e-6)


def adaptive_p_surr(
    p_min: float, p_max: float, beta: float, rel_td_error: float
) -> float:
    return p_min + (p_max - p_min) * math.exp(-beta * rel_td_error)
