from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from common.utils import genetic_soft_update
from modules.deep_modules import ActorHead, PeVFA, SharedStateEmbedding

UnravelHead = Callable[[jax.Array], ActorHead]


class PeVFAState(NamedTuple):
    online: PeVFA
    target: PeVFA
    opt_state: optax.OptState


def init_pevfa(
    state_dim: int,
    action_dim: int,
    num_params: int,
    *,
    key: jax.Array,
    embed_dim: int,
    hidden_dims: tuple[int, int],
    optimizer: optax.GradientTransformation,
) -> PeVFAState:
    net = PeVFA(
        state_dim,
        action_dim,
        num_params,
        key=key,
        embed_dim=embed_dim,
        hidden_dims=hidden_dims,
    )
    return PeVFAState(
        online=net,
        target=net,
        opt_state=optimizer.init(eqx.filter(net, eqx.is_array)),
    )


def _batch_q(
    net: PeVFA,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    return jax.vmap(net)(states, actions, weights)[..., 0]


def make_pevfa_step(
    gamma: float,
    tau: float,
    optimizer: optax.GradientTransformation,
    unravel_head: UnravelHead,
) -> Callable[
    [PeVFAState, SharedStateEmbedding, dict[str, jnp.ndarray], jnp.ndarray],
    tuple[PeVFAState, jnp.ndarray],
]:
    def loss_fn(
        online: PeVFA,
        target: PeVFA,
        embedding: SharedStateEmbedding,
        batch: dict[str, jnp.ndarray],
        weights: jnp.ndarray,
    ) -> jnp.ndarray:
        q_pred = _batch_q(online, batch["state"], batch["action"], weights)
        z_next = jax.vmap(embedding)(batch["next_state"])
        heads = jax.vmap(unravel_head)(weights)
        next_action = jax.vmap(lambda h, z: h(z))(heads, z_next)
        q_next = _batch_q(target, batch["next_state"], next_action, weights)
        done = batch["done"][..., 0]
        reward = batch["reward"][..., 0]
        td_target = reward + (1.0 - done) * gamma * q_next
        return jnp.mean((q_pred - jax.lax.stop_gradient(td_target)) ** 2)

    @eqx.filter_jit
    def step(
        state: PeVFAState,
        embedding: SharedStateEmbedding,
        batch: dict[str, jnp.ndarray],
        weights: jnp.ndarray,
    ) -> tuple[PeVFAState, jnp.ndarray]:
        loss, grads = eqx.filter_value_and_grad(loss_fn)(
            state.online, state.target, embedding, batch, weights
        )
        updates, opt_state = optimizer.update(
            grads, state.opt_state, eqx.filter(state.online, eqx.is_array)
        )
        online = eqx.apply_updates(state.online, updates)
        target = genetic_soft_update(state.target, online, tau)
        return PeVFAState(online, target, opt_state), loss

    return step


@eqx.filter_jit
def population_fitness(
    net: PeVFA,
    embedding: SharedStateEmbedding,
    pop_heads: ActorHead,
    flat_pop: jnp.ndarray,
    states: jnp.ndarray,
    undiscount_scale: float,
) -> jnp.ndarray:
    z = jax.vmap(embedding)(states)

    def head_value(head: ActorHead, w: jnp.ndarray) -> jnp.ndarray:
        actions = jax.vmap(head)(z)
        tiled = jnp.broadcast_to(w, (states.shape[0], w.shape[0]))
        return jnp.mean(_batch_q(net, states, actions, tiled))

    return jax.vmap(head_value)(pop_heads, flat_pop) * undiscount_scale
