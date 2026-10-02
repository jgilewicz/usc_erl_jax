from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from modules.deep_modules import ActorHead, Critic, SharedStateEmbedding

# Port of usc_erl's EnsembleModule (PyTorch): independently initialised
# critics regress the TD3 target, each on its own Bernoulli(0.5) half of the
# batch, with 2% multiplicative target noise and a smooth-L1 loss. The masks
# and init are what make members disagree; the target is shared.
_BOOTSTRAP_P = 0.5
_TARGET_NOISE = 0.02


class EnsembleState(NamedTuple):
    # one Critic whose every array leaf has a leading (ensemble_size,) axis
    critics: Critic
    opt_state: optax.OptState


EnsembleStep = Callable[
    [EnsembleState, dict[str, jnp.ndarray], jnp.ndarray, jax.Array],
    tuple[EnsembleState, jnp.ndarray],
]


def init_ensemble(
    obs_dim: int,
    action_dim: int,
    size: int,
    *,
    key: jax.Array,
    hidden_dims: tuple[int, int],
    optimizer: optax.GradientTransformation,
) -> EnsembleState:
    critics = eqx.filter_vmap(
        lambda k: Critic(obs_dim, action_dim, key=k, hidden_dims=hidden_dims)
    )(jax.random.split(key, size))
    return EnsembleState(
        critics, optimizer.init(eqx.filter(critics, eqx.is_array))
    )


def make_ensemble_step(
    optimizer: optax.GradientTransformation, gamma: float, size: int
) -> EnsembleStep:
    @eqx.filter_jit
    def step(
        state: EnsembleState,
        batch: dict[str, jnp.ndarray],
        q_next: jnp.ndarray,
        key: jax.Array,
    ) -> tuple[EnsembleState, jnp.ndarray]:
        target = jax.lax.stop_gradient(
            batch["reward"] + gamma * (1.0 - batch["done"]) * q_next
        )
        mask_keys, noise_keys = jnp.split(jax.random.split(key, 2 * size), 2)

        def member_loss(
            critic: Critic, mask_key: jax.Array, noise_key: jax.Array
        ) -> jnp.ndarray:
            q = jax.vmap(critic)(batch["state"], batch["action"])
            mask = jax.random.bernoulli(mask_key, _BOOTSTRAP_P, q.shape)
            noisy = target * (
                1.0 + _TARGET_NOISE * jax.random.normal(noise_key, target.shape)
            )
            loss = optax.huber_loss(q, noisy, delta=1.0)
            return jnp.sum(loss * mask) / (jnp.sum(mask) + 1e-8)

        def total_loss(critics: Critic) -> jnp.ndarray:
            return jnp.sum(
                eqx.filter_vmap(member_loss)(critics, mask_keys, noise_keys)
            )

        loss, grads = eqx.filter_value_and_grad(total_loss)(state.critics)
        params = eqx.filter(state.critics, eqx.is_array)
        updates, opt_state = optimizer.update(grads, state.opt_state, params)
        critics = eqx.apply_updates(state.critics, updates)
        return EnsembleState(critics, opt_state), loss

    return step


@eqx.filter_jit
def ensemble_fitness(
    critics: Critic,
    embedding: SharedStateEmbedding,
    pop_heads: ActorHead,
    states: jnp.ndarray,
    undiscount_scale: float,
) -> jnp.ndarray:
    # (pop, ensemble_size): E_{s~D}[Q_m(s, pi_i(s))] per member, the same
    # batch average as the critic arm
    z = jax.vmap(embedding)(states)
    actions = jax.vmap(lambda head: jax.vmap(head)(z))(pop_heads)

    def member(critic: Critic) -> jnp.ndarray:
        return jax.vmap(lambda a: jnp.mean(jax.vmap(critic)(states, a)))(
            actions
        )

    return eqx.filter_vmap(member)(critics).T * undiscount_scale
