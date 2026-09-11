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
from common.td3 import (
    TD3Config,
    TD3State,
    clipped_double_q,
    init_td3,
    make_td3_steps,
)
from common.utils import (
    absolute_td_error,
    adaptive_p_surr,
    genetic_soft_update,
    h_step_bootstrap,
    relative_td_error,
)
from modules.deep_modules import ActorHead, Critic, SharedStateEmbedding
from modules.evo_module import CEM
from algos.erl import ERLConfig


@dataclass(frozen=True)
class SEMARLConfig(ERLConfig):
    # The bootstrap horizon is fixed at ERL's h_steps
    p_surr_min: float = 0.0
    p_surr_max: float = 0.9
    p_beta: float = 3.0
    td_ema_decay: float = 0.1


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


@eqx.filter_jit
def _critic_fitness(
    embedding: SharedStateEmbedding,
    critic1: Critic,
    critic2: Critic,
    pop_heads: ActorHead,
    states: jnp.ndarray,
    undiscount_scale: float,
) -> jnp.ndarray:
    # E_{s~D}[min(Q1,Q2)(s, pi_i(s))] over a replay batch - the standard
    # critic-only policy fitness in the ERL literature, and the honest
    # SEMARL-style baseline. The H=0 arm is its degenerate one-state case:
    # every individual starts from near-identical states, so that arm has to
    # separate policies by their action at a single point.
    z = jax.vmap(embedding)(states)

    def head_value(head: ActorHead) -> jnp.ndarray:
        actions = jax.vmap(head)(z)
        q1 = jax.vmap(critic1)(states, actions)[..., 0]
        q2 = jax.vmap(critic2)(states, actions)[..., 0]
        return jnp.mean(jnp.minimum(q1, q2))

    return jax.vmap(head_value)(pop_heads) * undiscount_scale


@eqx.filter_jit
def _reward_only_fitness(
    h_reward: jnp.ndarray,
    h_done: jnp.ndarray,
    h_steps: int,
    gamma: float,
    undiscount_scale: float,
) -> jnp.ndarray:
    # the h-step surrogate with the gamma^H * Q term zeroed out: isolates how
    # much of its accuracy is the accumulated real reward rather than the
    # critic. The bootstrap's share of the value is exactly gamma^H.
    no_bootstrap = jnp.zeros(h_reward.shape[0])
    discounted = h_step_bootstrap(
        h_steps, gamma, h_reward, h_done, no_bootstrap
    )
    return discounted * undiscount_scale


@eqx.filter_jit
def _critic_disagreement(
    embedding: SharedStateEmbedding,
    critic1: Critic,
    critic2: Critic,
    pop_heads: ActorHead,
    states: jnp.ndarray,
) -> jnp.ndarray:
    # |Q1 - Q2| at the surrogate's own bootstrap query points, per individual:
    # a free epistemic proxy that |TD| (global, on buffer actions) can't give.
    z = jax.vmap(embedding)(states)
    own_actions = jax.vmap(lambda head, zi: head(zi))(pop_heads, z)
    q1 = jax.vmap(critic1)(states, own_actions)[..., 0]
    q2 = jax.vmap(critic2)(states, own_actions)[..., 0]
    return jnp.abs(q1 - q2)


def _ranks(x: jnp.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(np.asarray(x)))


def _rank_corr(a: jnp.ndarray, b: jnp.ndarray) -> float:
    # Spearman between two population-length fitness vectors.
    if len(np.asarray(a)) < 3:
        return float("nan")
    return float(np.corrcoef(_ranks(a), _ranks(b))[0, 1])


def _elite_overlap(a: jnp.ndarray, b: jnp.ndarray, parents: int) -> float:
    # What CEM actually consume: the fraction of the top `parents` individuals that are shared between two population-length fitness vectors.
    x, y = np.asarray(a), np.asarray(b)
    if len(x) < parents:
        return float("nan")
    top_a = set(np.argsort(-x)[:parents].tolist())
    top_b = set(np.argsort(-y)[:parents].tolist())
    return len(top_a & top_b) / parents


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
    cfg: SEMARLConfig,
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
    if not 0.0 <= cfg.p_surr_min <= cfg.p_surr_max <= 1.0:
        raise ValueError(
            f"need 0 <= p_surr_min ({cfg.p_surr_min}) <= p_surr_max "
            f"({cfg.p_surr_max}) <= 1"
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
    td_rel_ema: float | None = None
    try:
        for generation in range(cfg.generations):
            key, ask_key = jax.random.split(key)
            flat_pop = cem.ask(ask_key)
            if pending_injection is not None:
                rl_flat, _ = ravel_pytree(td3_state.online.actor)
                flat_pop = flat_pop.at[pending_injection].set(rl_flat)
            pop_heads = jax.vmap(unravel_head)(flat_pop)

            warmup = len(buffer) < cfg.warmup_steps
            p_surr = (
                0.0
                if warmup or td_rel_ema is None
                else adaptive_p_surr(
                    cfg.p_surr_min, cfg.p_surr_max, cfg.p_beta, td_rel_ema
                )
            )
            key, coin_key = jax.random.split(key)
            use_real_fitness = not bool(jax.random.bernoulli(coin_key, p_surr))

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

            # snapshot the bootstrap state at h_steps
            h_reward = np.zeros((cfg.pop_size + 1, cfg.h_steps), np.float32)
            h_done = np.zeros((cfg.pop_size + 1, cfg.h_steps), np.float32)
            h_states: dict[int, jnp.ndarray] = {}

            def capture_h_step(
                step: int,
                states: jnp.ndarray,
                reward: jnp.ndarray,
                terminated: jnp.ndarray,
                truncated: jnp.ndarray,
                next_states: jnp.ndarray,
                h_reward: np.ndarray = h_reward,
                h_done: np.ndarray = h_done,
                h_states: dict[int, jnp.ndarray] = h_states,
            ) -> None:
                if step == 0:
                    h_states[0] = states
                if step >= cfg.h_steps:
                    return
                h_reward[:, step] = np.asarray(reward)
                h_done[:, step] = np.asarray(terminated)
                if step + 1 == cfg.h_steps:
                    h_states[cfg.h_steps] = next_states

            returns, key = collect_parallel_episode(
                vec_env, key, policy, buffer, horizon, on_step=capture_h_step
            )
            real_fitness = returns[:-1]
            rl_return = float(returns[-1])

            if warmup:
                td_error = 0.0
                td_error_rel = 0.0
            else:
                key, td_key, td_sample_key = jax.random.split(key, 3)
                td_batch = buffer.sample(td_sample_key, cfg.batch_size)
                q, q_prim = clipped_double_q(
                    td3_state, td_batch, td_key, td3_cfg
                )
                td_error = float(
                    absolute_td_error(
                        td_batch["reward"],
                        cfg.gamma,
                        q_prim,
                        q,
                        td_batch["done"],
                    )
                )
                td_error_rel = relative_td_error(
                    td_error, float(jnp.mean(jnp.abs(td_batch["reward"])))
                )
                td_rel_ema = (
                    td_error_rel
                    if td_rel_ema is None
                    else (1.0 - cfg.td_ema_decay) * td_rel_ema
                    + cfg.td_ema_decay * td_error_rel
                )

            def surrogate_at(
                h: int, pop_heads: ActorHead = pop_heads
            ) -> jnp.ndarray:
                return _surrogate_fitness(
                    td3_state.online.embedding,
                    td3_state.online.critic1,
                    td3_state.online.critic2,
                    pop_heads,
                    h_states[h][:-1],
                    jnp.asarray(h_reward[:-1, :h]),
                    jnp.asarray(h_done[:-1, :h]),
                    h,
                    cfg.gamma,
                    undiscount_scale,
                )

            surrogate_fitness = surrogate_at(cfg.h_steps)
            fitness = real_fitness if use_real_fitness else surrogate_fitness

            # real_fitness is the true full-episode return every generation,
            # so every arm below is scored for free against the truth:
            #   H       the h-step bootstrap actually driving selection
            #   h0      its degenerate one-state critic case
            #   critic  batch-averaged critic value (the fair SEMARL baseline)
            #   noboot  H with the gamma^H * Q term dropped (is the critic
            #           contributing anything, or is it all real reward?)
            key, crit_key = jax.random.split(key)
            arms = {
                "": surrogate_fitness,
                "_h0": surrogate_at(0),
                "_critic": _critic_fitness(
                    td3_state.online.embedding,
                    td3_state.online.critic1,
                    td3_state.online.critic2,
                    pop_heads,
                    buffer.sample(crit_key, cfg.batch_size)["state"],
                    undiscount_scale,
                ),
                "_noboot": _reward_only_fitness(
                    jnp.asarray(h_reward[:-1]),
                    jnp.asarray(h_done[:-1]),
                    cfg.h_steps,
                    cfg.gamma,
                    undiscount_scale,
                ),
            }
            arm_metrics: dict[str, float] = {}
            for suffix, arm in arms.items():
                arm_metrics[f"surrogate_abs_err{suffix}"] = float(
                    jnp.mean(jnp.abs(real_fitness - arm))
                )
                arm_metrics[f"surrogate_rank_corr{suffix}"] = _rank_corr(
                    real_fitness, arm
                )
                arm_metrics[f"surrogate_elite_overlap{suffix}"] = (
                    _elite_overlap(real_fitness, arm, cem.parents)
                )

            # does per-individual critic disagreement flag
            q_disagree = _critic_disagreement(
                td3_state.online.embedding,
                td3_state.online.critic1,
                td3_state.online.critic2,
                pop_heads,
                h_states[cfg.h_steps][:-1],
            )
            rank_shift = np.abs(
                _ranks(real_fitness) - _ranks(surrogate_fitness)
            ).astype(float)
            q_disagree_rank_corr = _rank_corr(
                q_disagree, jnp.asarray(rank_shift)
            )

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
                "buffer_size": float(len(buffer)),
                "fitness_best": float(jnp.max(fitness)),
                "fitness_mean": float(jnp.mean(fitness)),
                "fitness_is_real": float(use_real_fitness),
                "td_error": td_error,
                "td_error_rel": td_error_rel,
                "td_error_rel_ema": td_rel_ema
                if td_rel_ema is not None
                else 0.0,
                "p_surr": p_surr,
                "h_step": float(cfg.h_steps),
                **arm_metrics,
                "q_disagree_mean": float(jnp.mean(q_disagree)),
                "q_disagree_rank_corr": q_disagree_rank_corr,
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
                f"td={td_error:7.3f} tdrel={td_error_rel:6.3f} "
                f"p_surr={p_surr:.2f} | "
                f"rankcorr(H/0/crit/nobo)="
                f"{arm_metrics['surrogate_rank_corr']:+.2f}/"
                f"{arm_metrics['surrogate_rank_corr_h0']:+.2f}/"
                f"{arm_metrics['surrogate_rank_corr_critic']:+.2f}/"
                f"{arm_metrics['surrogate_rank_corr_noboot']:+.2f} | "
                f"rl_return={rl_return:8.1f} | "
                f"critic_loss={metrics['critic_loss']:8.4f} "
                f"actor_loss={metrics['actor_loss']:8.4f}"
            )
            if on_generation is not None:
                on_generation(metrics)
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
    train(SEMARLConfig())
