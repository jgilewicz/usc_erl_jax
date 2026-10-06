from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from common.critic_heads import CriticHead, plain_loss, plain_point
from common.replay_buffer import BufferData, sample_batch
from common.utils import genetic_soft_update
from modules.deep_modules import ActorHead, Critic, SharedStateEmbedding


class TD3Networks(NamedTuple):
    embedding: SharedStateEmbedding
    actor: ActorHead
    # Critic, a (K,)-stacked Critic ensemble or an EvidentialCritic
    critic1: Any
    critic2: Critic


class TD3State(NamedTuple):
    online: TD3Networks
    target: TD3Networks
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState


class TD3Config(NamedTuple):
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    action_limit: float = 1.0
    policy_freq: int = 2
    batch_size: int = 256


Batch = dict[str, jnp.ndarray]
# (state, buffer data, buffer size, num_updates, key)
#   -> (state, mean critic loss, mean actor loss)
TrainFn = Callable[
    [TD3State, BufferData, jax.Array, jax.Array, jax.Array],
    tuple[TD3State, jax.Array, jax.Array],
]


def init_td3(
    state_dim: int,
    action_dim: int,
    embedding_dim: int,
    *,
    key: jax.Array,
    head: CriticHead,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
) -> TD3State:
    ek, ak, c1k, c2k = jax.random.split(key, 4)
    embedding = SharedStateEmbedding(
        state_dim, embedding_dim=embedding_dim, key=ek
    )
    actor = ActorHead(embedding_dim, action_dim, key=ak)
    critic1 = head.build(c1k)
    critic2 = head.build_twin(c2k)

    online = TD3Networks(embedding, actor, critic1, critic2)
    return TD3State(
        online=online,
        target=online,
        actor_opt_state=actor_optimizer.init(
            eqx.filter((embedding, actor), eqx.is_array)
        ),
        critic_opt_state=critic_optimizer.init(
            eqx.filter((critic1, critic2), eqx.is_array)
        ),
    )


def _td_target(
    cfg: TD3Config,
    head: CriticHead,
    state: TD3State,
    batch: Batch,
    key: jax.Array,
) -> jax.Array:
    noise_key, q1_key, q2_key = jax.random.split(key, 3)
    noise = jnp.clip(
        jax.random.normal(noise_key, batch["action"].shape) * cfg.policy_noise,
        -cfg.noise_clip,
        cfg.noise_clip,
    )
    target = state.target
    z_next = jax.vmap(target.embedding)(batch["next_state"])
    next_action = jnp.clip(
        jax.vmap(target.actor)(z_next) + noise,
        -cfg.action_limit,
        cfg.action_limit,
    )
    # twin-min over point estimates: mean / mu for ensemble / evidential
    q_next = jnp.minimum(
        head.point(target.critic1, batch["next_state"], next_action, q1_key),
        plain_point(target.critic2, batch["next_state"], next_action, q2_key),
    )
    return jax.lax.stop_gradient(
        batch["reward"] + cfg.gamma * (1.0 - batch["done"]) * q_next
    )


def _critic_update(
    cfg: TD3Config,
    head: CriticHead,
    optimizer: optax.GradientTransformation,
    state: TD3State,
    batch: Batch,
    key: jax.Array,
) -> tuple[TD3State, jax.Array]:
    target_key, k1, k2 = jax.random.split(key, 3)
    target_q = _td_target(cfg, head, state, batch, target_key)

    def loss_fn(critics: tuple[Any, Critic]) -> jax.Array:
        critic1, critic2 = critics
        s, a = batch["state"], batch["action"]
        return head.loss(critic1, s, a, target_q, k1) + plain_loss(
            critic2, s, a, target_q, k2
        )

    critics = (state.online.critic1, state.online.critic2)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(critics)
    updates, opt_state = optimizer.update(
        grads, state.critic_opt_state, eqx.filter(critics, eqx.is_array)
    )
    critic1, critic2 = eqx.apply_updates(critics, updates)
    new_target = state.target._replace(
        critic1=genetic_soft_update(state.target.critic1, critic1, cfg.tau),
        critic2=genetic_soft_update(state.target.critic2, critic2, cfg.tau),
    )
    new_online = state.online._replace(critic1=critic1, critic2=critic2)
    new_state = state._replace(
        online=new_online, target=new_target, critic_opt_state=opt_state
    )
    return new_state, loss


def _actor_update(
    cfg: TD3Config,
    head: CriticHead,
    optimizer: optax.GradientTransformation,
    state: TD3State,
    batch: Batch,
    key: jax.Array,
) -> tuple[TD3State, jax.Array]:
    def loss_fn(
        embedding_actor: tuple[SharedStateEmbedding, ActorHead],
    ) -> jax.Array:
        embedding, actor = embedding_actor
        action = jax.vmap(actor)(jax.vmap(embedding)(batch["state"]))
        q = head.point(state.online.critic1, batch["state"], action, key)
        return -jnp.mean(q)

    embedding_actor = (state.online.embedding, state.online.actor)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(embedding_actor)
    updates, opt_state = optimizer.update(
        grads,
        state.actor_opt_state,
        eqx.filter(embedding_actor, eqx.is_array),
    )
    embedding, actor = eqx.apply_updates(embedding_actor, updates)
    new_target = state.target._replace(
        embedding=genetic_soft_update(
            state.target.embedding, embedding, cfg.tau
        ),
        actor=genetic_soft_update(state.target.actor, actor, cfg.tau),
    )
    new_online = state.online._replace(embedding=embedding, actor=actor)
    new_state = state._replace(
        online=new_online, target=new_target, actor_opt_state=opt_state
    )
    return new_state, loss


def make_td3_train(
    cfg: TD3Config,
    head: CriticHead,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
) -> TrainFn:
    @eqx.filter_jit
    def train(
        state: TD3State,
        data: BufferData,
        size: jax.Array,
        num_updates: jax.Array,
        key: jax.Array,
    ) -> tuple[TD3State, jax.Array, jax.Array]:
        dynamic, static = eqx.partition(state, eqx.is_array)

        def body(
            step: jax.Array, carry: tuple[Any, jax.Array, jax.Array, jax.Array]
        ) -> tuple[Any, jax.Array, jax.Array, jax.Array]:
            dynamic, key, critic_sum, actor_sum = carry
            key, sample_key, critic_key, actor_key = jax.random.split(key, 4)
            batch = sample_batch(data, size, sample_key, cfg.batch_size)
            state, critic_loss = _critic_update(
                cfg,
                head,
                critic_optimizer,
                eqx.combine(dynamic, static),
                batch,
                critic_key,
            )

            def with_actor(dyn: Any) -> tuple[Any, jax.Array]:
                new_state, loss = _actor_update(
                    cfg,
                    head,
                    actor_optimizer,
                    eqx.combine(dyn, static),
                    batch,
                    actor_key,
                )
                return eqx.filter(new_state, eqx.is_array), loss

            def skip(dyn: Any) -> tuple[Any, jax.Array]:
                return dyn, jnp.zeros(())

            dynamic, actor_loss = jax.lax.cond(
                step % cfg.policy_freq == 0,
                with_actor,
                skip,
                eqx.filter(state, eqx.is_array),
            )
            return (
                dynamic,
                key,
                critic_sum + critic_loss,
                actor_sum + actor_loss,
            )

        dynamic, _, critic_sum, actor_sum = jax.lax.fori_loop(
            0,
            num_updates,
            body,
            (dynamic, key, jnp.zeros(()), jnp.zeros(())),
        )
        actor_steps = (num_updates + cfg.policy_freq - 1) // cfg.policy_freq
        return (
            eqx.combine(dynamic, static),
            critic_sum / jnp.maximum(num_updates, 1),
            actor_sum / jnp.maximum(actor_steps, 1),
        )

    return train
