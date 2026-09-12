from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

import environments
from common.replay_buffer import Buffer
from common.rollout import collect_parallel_episode
from common.td3 import TD3Config, TD3State, init_td3, make_td3_steps
from common.utils import genetic_soft_update, h_step_bootstrap
from modules.deep_modules import ActorHead, Critic, SharedStateEmbedding
from modules.evo_module import CEM


@dataclass(frozen=True)
class ERLConfig:
    env_name: str = "Swimmer-v5"
    seed: int = 0
    # env-step budget, not a generation count: once SEMARL skips population
    # rollouts a generation stops being a fixed amount of interaction, so
    # generations are not comparable across conditions and steps are.
    step_budget: int = 1_000_000
    horizon: int | None = None  # None: use the env's max_episode_steps
    async_env: bool = True

    # Population (CEM over ActorHead's flat params, evaluated on Z(s))
    pop_size: int = 10
    embedding_dim: int = 256
    sigma_init: float = 1e-3
    damp: float = 1e-3
    damp_limit: float = 1e-5
    elitism: bool = True

    # TD3 gradient agent
    critic_hidden_dims: tuple[int, int] = (400, 300)
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    exploration_noise: float = 0.1
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 128
    buffer_capacity: int = 1_000_000
    warmup_steps: int = 10_000
    train_ratio: float = 1.0  # gradient steps per env step collected

    # EA <-> RL gene flow, both gated until the buffer is past warmup
    rl_to_ea_sync_period: int = 5
    ea_tau: float = 0.005

    # P(real full-episode fitness) per generation, else surrogate; EvoRainbow's inverse P(surrogate)=0.2 default.
    theta: float = 0.6
    h_steps: int = 250


@eqx.filter_jit
def _step_policy(
    embedding: SharedStateEmbedding,
    pop_heads: ActorHead,
    actor: ActorHead,
    states: jnp.ndarray,
    key: jax.Array,
    action_limit: float,
    exploration_noise: float,
    warmup: bool,
) -> jnp.ndarray:
    z = jax.vmap(embedding)(states)
    pop_actions = jax.vmap(lambda head, zi: head(zi))(pop_heads, z[:-1])

    rl_key, warm_key = jax.random.split(key)
    rl_raw = actor(z[-1])
    if warmup:
        rl_action = jax.random.uniform(
            warm_key, rl_raw.shape, minval=-action_limit, maxval=action_limit
        )
    else:
        noise = jax.random.normal(rl_key, rl_raw.shape) * exploration_noise
        rl_action = jnp.clip(rl_raw + noise, -action_limit, action_limit)

    return jnp.concatenate([pop_actions, rl_action[None]], axis=0)


@eqx.filter_jit
def _deterministic_action(
    embedding: SharedStateEmbedding, actor: ActorHead, state: jnp.ndarray
) -> jnp.ndarray:
    return actor(embedding(state))


@eqx.filter_jit
def _surrogate_fitness(
    embedding: SharedStateEmbedding,
    critic1: Critic,
    critic2: Critic,
    pop_heads: ActorHead,
    h_state: jnp.ndarray,
    h_reward: jnp.ndarray,
    h_done: jnp.ndarray,
    h_steps: int,
    gamma: float,
    undiscount_scale: float,
) -> jnp.ndarray:
    z = jax.vmap(embedding)(h_state)
    own_actions = jax.vmap(lambda head, zi: head(zi))(pop_heads, z)
    q1 = jax.vmap(critic1)(h_state, own_actions)[..., 0]
    q2 = jax.vmap(critic2)(h_state, own_actions)[..., 0]
    q_bootstrap = jnp.minimum(q1, q2)
    discounted = h_step_bootstrap(h_steps, gamma, h_reward, h_done, q_bootstrap)
    # rescale the discounted bootstrap onto real_fitness's undiscounted scale; rank-preserving, so CEM selection is unaffected.
    return discounted * undiscount_scale


def _env_dims(
    env_name: str, horizon: int | None
) -> tuple[int, int, float, int]:
    env = environments.make_env(env_name)
    try:
        obs_space, action_space = env.observation_space, env.action_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(f"{env_name}: expected a Box observation space")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(f"{env_name}: expected a Box action space")
        obs_dim = int(obs_space.shape[0])
        action_dim = int(action_space.shape[0])
        low, high = action_space.low, action_space.high
        if not (np.allclose(high, -low) and np.allclose(high, high[0])):
            raise ValueError(
                f"{env_name}: expected a symmetric, uniform action bound "
                f"(all dims [-b, b]), got low={low}, high={high}"
            )
        action_limit = float(high[0])
        if horizon is None:
            if env.spec is None or env.spec.max_episode_steps is None:
                raise ValueError(
                    f"{env_name}: no max_episode_steps on env.spec; "
                    "pass ERLConfig(horizon=...) explicitly"
                )
            horizon = env.spec.max_episode_steps
    finally:
        env.close()
    return obs_dim, action_dim, action_limit, horizon


def train(
    cfg: ERLConfig,
    *,
    on_generation: Callable[[dict[str, float]], None] | None = None,
) -> TD3State:
    obs_dim, action_dim, action_limit, horizon = _env_dims(
        cfg.env_name, cfg.horizon
    )
    if cfg.h_steps > horizon:
        raise ValueError(
            f"h_steps ({cfg.h_steps}) must be <= horizon ({horizon})"
        )
    vec_env = environments.make_vec_env(
        cfg.env_name, cfg.pop_size + 1, async_=cfg.async_env, to_jax=True
    )
    buffer = Buffer(cfg.buffer_capacity, obs_dim, action_dim)

    key = jax.random.key(cfg.seed)
    key, cem_key, td3_key = jax.random.split(key, 3)

    template_head = ActorHead(
        cfg.embedding_dim, action_dim, key=cem_key, action_limit=action_limit
    )
    flat_template, unravel_head = ravel_pytree(template_head)
    num_params = flat_template.shape[0]

    cem = CEM(
        num_params,
        cem_key,
        sigma_init=cfg.sigma_init,
        pop_size=cfg.pop_size,
        damp=cfg.damp,
        damp_limit=cfg.damp_limit,
        elitisim=cfg.elitism,
    )

    actor_optimizer = optax.adam(cfg.actor_lr)
    critic_optimizer = optax.adam(cfg.critic_lr)
    td3_state = init_td3(
        obs_dim,
        action_dim,
        cfg.embedding_dim,
        key=td3_key,
        hidden_dims=cfg.critic_hidden_dims,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )
    td3_cfg = TD3Config(
        gamma=cfg.gamma,
        tau=cfg.tau,
        policy_noise=cfg.policy_noise,
        noise_clip=cfg.noise_clip,
        action_limit=action_limit,
    )
    critic_step, actor_step = make_td3_steps(
        td3_cfg, actor_optimizer, critic_optimizer
    )
    # projects the discounted bootstrap onto real_fitness's undiscounted scale (see _surrogate_fitness).
    undiscount_scale = horizon * (1.0 - cfg.gamma) / (1.0 - cfg.gamma**horizon)

    pending_injection: int | None = None
    env_steps = 0
    generation = 0
    # ERL never truncates: every generation is the same fixed cost.
    gen_env_steps = horizon * (cfg.pop_size + 1)
    try:
        while env_steps < cfg.step_budget:
            env_steps += gen_env_steps
            key, ask_key = jax.random.split(key)
            flat_pop = cem.ask(ask_key)
            if pending_injection is not None:
                rl_flat, _ = ravel_pytree(td3_state.online.actor)
                flat_pop = flat_pop.at[pending_injection].set(rl_flat)
            pop_heads = jax.vmap(unravel_head)(flat_pop)

            warmup = len(buffer) < cfg.warmup_steps

            def policy(
                act_key: jax.Array,
                states: jnp.ndarray,
                pop_heads: ActorHead = pop_heads,
                warmup: bool = warmup,
            ) -> jnp.ndarray:
                return _step_policy(
                    td3_state.online.embedding,
                    pop_heads,
                    td3_state.online.actor,
                    states,
                    act_key,
                    action_limit,
                    cfg.exploration_noise,
                    warmup,
                )

            h_reward = np.zeros((cfg.pop_size + 1, cfg.h_steps), np.float32)
            h_done = np.zeros((cfg.pop_size + 1, cfg.h_steps), np.float32)
            h_state_box: list[jnp.ndarray] = []

            def capture_h_step(
                step: int,
                reward: jnp.ndarray,
                terminated: jnp.ndarray,
                truncated: jnp.ndarray,
                next_states: jnp.ndarray,
                h_reward: np.ndarray = h_reward,
                h_done: np.ndarray = h_done,
                h_state_box: list[jnp.ndarray] = h_state_box,
            ) -> None:
                if step >= cfg.h_steps:
                    return
                h_reward[:, step] = np.asarray(reward)
                h_done[:, step] = np.asarray(terminated)
                if step == cfg.h_steps - 1:
                    h_state_box.append(next_states)

            returns, key = collect_parallel_episode(
                vec_env, key, policy, buffer, horizon, on_step=capture_h_step
            )
            real_fitness = returns[:-1]
            rl_return = float(returns[-1])

            surrogate_fitness = _surrogate_fitness(
                td3_state.online.embedding,
                td3_state.online.critic1,
                td3_state.online.critic2,
                pop_heads,
                h_state_box[0][:-1],
                jnp.asarray(h_reward[:-1]),
                jnp.asarray(h_done[:-1]),
                cfg.h_steps,
                cfg.gamma,
                undiscount_scale,
            )

            key, coin_key = jax.random.split(key)
            use_real_fitness = warmup or bool(
                jax.random.bernoulli(coin_key, cfg.theta)
            )
            fitness = real_fitness if use_real_fitness else surrogate_fitness

            cem.tell(fitness, flat_pop)
            pending_injection = None

            critic_losses = []
            actor_losses = []
            if not warmup:
                num_updates = int(
                    cfg.train_ratio * horizon * (cfg.pop_size + 1)
                )
                for step in range(num_updates):
                    key, sample_key, noise_key = jax.random.split(key, 3)
                    batch = buffer.sample(sample_key, cfg.batch_size)
                    td3_state, critic_loss = critic_step(
                        td3_state, batch, noise_key
                    )
                    critic_losses.append(critic_loss)
                    if step % cfg.policy_freq == 0:
                        td3_state, actor_loss = actor_step(td3_state, batch)
                        actor_losses.append(actor_loss)

                champion_idx = int(jnp.argmax(fitness))
                champion_head = unravel_head(flat_pop[champion_idx])
                new_actor = genetic_soft_update(
                    td3_state.online.actor, champion_head, cfg.ea_tau
                )
                td3_state = td3_state._replace(
                    online=td3_state.online._replace(actor=new_actor)
                )

                if generation % cfg.rl_to_ea_sync_period == 0:
                    pending_injection = int(jnp.argmin(fitness))

            metrics = {
                "generation": float(generation),
                "env_steps": float(env_steps),
                "env_steps_gen": float(gen_env_steps),
                "buffer_size": float(len(buffer)),
                "fitness_best": float(jnp.max(fitness)),
                "fitness_mean": float(jnp.mean(fitness)),
                "fitness_is_real": float(use_real_fitness),
                "fitness_real_mean": float(jnp.mean(real_fitness)),
                "fitness_surrogate_mean": float(jnp.mean(surrogate_fitness)),
                "rl_return": rl_return,
                "critic_loss": float(np.mean(critic_losses))
                if critic_losses
                else 0.0,
                "actor_loss": float(np.mean(actor_losses))
                if actor_losses
                else 0.0,
            }
            print(
                f"gen {generation:4d} | buffer {len(buffer):7d} | "
                f"fitness[{'real' if use_real_fitness else 'surrogate'}] "
                f"best={metrics['fitness_best']:8.1f} "
                f"mean={metrics['fitness_mean']:8.1f} | "
                f"rl_return={rl_return:8.1f} | "
                f"critic_loss={metrics['critic_loss']:8.4f} "
                f"actor_loss={metrics['actor_loss']:8.4f}"
            )
            if on_generation is not None:
                on_generation(metrics)
            generation += 1
    finally:
        vec_env.close()

    return td3_state


def evaluate_actor(
    env_name: str, td3_state: TD3State, *, episodes: int = 5
) -> float:
    env = environments.make_env(env_name)
    try:
        total_reward = 0.0
        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            while not done:
                action = np.asarray(
                    _deterministic_action(
                        td3_state.online.embedding,
                        td3_state.online.actor,
                        jnp.asarray(state),
                    )
                )
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
        return total_reward / episodes
    finally:
        env.close()


if __name__ == "__main__":
    train(ERLConfig())
