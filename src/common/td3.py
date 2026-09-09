from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from common.utils import genetic_soft_update
from modules.deep_modules import ActorHead, Critic, SharedStateEmbedding


class TD3Networks(NamedTuple):
    embedding: SharedStateEmbedding
    actor: ActorHead
    critic1: Critic
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


def init_td3(
    state_dim: int,
    action_dim: int,
    embedding_dim: int,
    *,
    key: jax.Array,
    hidden_dims: tuple[int, int] = (400, 300),
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
) -> TD3State:
    ek, ak, c1k, c2k = jax.random.split(key, 4)
    embedding = SharedStateEmbedding(
        state_dim, embedding_dim=embedding_dim, key=ek
    )
    actor = ActorHead(embedding_dim, action_dim, key=ak)
    critic1 = Critic(state_dim, action_dim, hidden_dims=hidden_dims, key=c1k)
    critic2 = Critic(state_dim, action_dim, hidden_dims=hidden_dims, key=c2k)

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


CriticStep = Callable[
    [TD3State, dict[str, jnp.ndarray], jax.Array], tuple[TD3State, jnp.ndarray]
]
ActorStep = Callable[
    [TD3State, dict[str, jnp.ndarray]], tuple[TD3State, jnp.ndarray]
]


def make_td3_steps(
    cfg: TD3Config,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
) -> tuple[CriticStep, ActorStep]:
    @eqx.filter_jit
    def critic_step(
        state: TD3State, batch: dict[str, jnp.ndarray], key: jax.Array
    ) -> tuple[TD3State, jnp.ndarray]:
        noise = jnp.clip(
            jax.random.normal(key, batch["action"].shape) * cfg.policy_noise,
            -cfg.noise_clip,
            cfg.noise_clip,
        )
        z_next = jax.vmap(state.target.embedding)(batch["next_state"])
        next_action = jnp.clip(
            jax.vmap(state.target.actor)(z_next) + noise,
            -cfg.action_limit,
            cfg.action_limit,
        )
        target_q1 = jax.vmap(state.target.critic1)(
            batch["next_state"], next_action
        )
        target_q2 = jax.vmap(state.target.critic2)(
            batch["next_state"], next_action
        )
        target_q = batch["reward"] + cfg.gamma * (1.0 - batch["done"]) * (
            jnp.minimum(target_q1, target_q2)
        )
        target_q = jax.lax.stop_gradient(target_q)

        def loss_fn(
            critics: tuple[Critic, Critic],
        ) -> jax.Array:
            critic1, critic2 = critics
            q1 = jax.vmap(critic1)(batch["state"], batch["action"])
            q2 = jax.vmap(critic2)(batch["state"], batch["action"])
            return jnp.mean((q1 - target_q) ** 2) + jnp.mean(
                (q2 - target_q) ** 2
            )

        critics = (state.online.critic1, state.online.critic2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(critics)
        updates, new_opt_state = critic_optimizer.update(
            grads, state.critic_opt_state, critics
        )
        new_critic1, new_critic2 = eqx.apply_updates(critics, updates)

        new_target = state.target._replace(
            critic1=genetic_soft_update(
                state.target.critic1, new_critic1, cfg.tau
            ),
            critic2=genetic_soft_update(
                state.target.critic2, new_critic2, cfg.tau
            ),
        )
        new_online = state.online._replace(
            critic1=new_critic1, critic2=new_critic2
        )
        return (
            state._replace(
                online=new_online,
                target=new_target,
                critic_opt_state=new_opt_state,
            ),
            loss,
        )

    @eqx.filter_jit
    def actor_step(
        state: TD3State, batch: dict[str, jnp.ndarray]
    ) -> tuple[TD3State, jnp.ndarray]:
        def loss_fn(
            embedding_actor: tuple[SharedStateEmbedding, ActorHead],
        ) -> jax.Array:
            embedding, actor = embedding_actor
            z = jax.vmap(embedding)(batch["state"])
            action = jax.vmap(actor)(z)
            q1 = jax.vmap(state.online.critic1)(batch["state"], action)
            return -jnp.mean(q1)

        embedding_actor = (state.online.embedding, state.online.actor)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(embedding_actor)
        updates, new_opt_state = actor_optimizer.update(
            grads, state.actor_opt_state, embedding_actor
        )
        new_embedding, new_actor = eqx.apply_updates(embedding_actor, updates)

        new_target = state.target._replace(
            embedding=genetic_soft_update(
                state.target.embedding, new_embedding, cfg.tau
            ),
            actor=genetic_soft_update(state.target.actor, new_actor, cfg.tau),
        )
        new_online = state.online._replace(
            embedding=new_embedding, actor=new_actor
        )
        return (
            state._replace(
                online=new_online,
                target=new_target,
                actor_opt_state=new_opt_state,
            ),
            loss,
        )

    return critic_step, actor_step
