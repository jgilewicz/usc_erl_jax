from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

import environments
from common.critic_heads import CriticHead, make_head
from common.metrics import selection_metrics
from common.replay_buffer import Buffer
from common.rollout import collect_parallel_episode, evaluate_heads
from common.td3 import TD3Config, TD3State, TrainFn, init_td3, make_td3_train
from common.utils import genetic_soft_update, h_step_bootstrap
from modules.deep_modules import ActorHead, SharedStateEmbedding
from modules.evo_module import CEM

Metrics = dict[str, float]


@dataclass(frozen=True)
class ERLConfig:
    env_name: str = "Swimmer-v5"
    seed: int = 0
    # env-step budget, not generations: generation cost varies by algorithm
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

    # P(real full-episode fitness) per generation, else surrogate;
    # EvoRainbow's inverse P(surrogate)=0.2 default.
    theta: float = 0.6
    h_steps: int = 250

    # deterministic RL-actor + CEM-elite eval, off the env-step budget
    eval_env_name: str | None = None  # None: env_name
    eval_interval: int = 5_000
    eval_episodes: int = 5


@dataclass
class Run:
    # mutable per-run state shared by ERL and SC-ERL
    cfg: ERLConfig
    horizon: int
    action_limit: float
    buffer: Buffer
    cem: CEM
    unravel_head: Callable[[jax.Array], ActorHead]
    head: CriticHead
    td3_state: TD3State
    train_td3: TrainFn
    eval_env: gym.vector.VectorEnv
    key: jax.Array
    # projects discounted critic values onto the undiscounted return scale
    undiscount_scale: float
    env_steps: int = 0
    generation: int = 0
    next_eval: int = 0
    pending_injection: int | None = None
    closers: list[Callable[[], None]] = field(default_factory=list)

    def close(self) -> None:
        for close in self.closers:
            close()


def env_dims(env_name: str, horizon: int | None) -> tuple[int, int, float, int]:
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


def build_run(
    cfg: ERLConfig,
    make_critic_head: Callable[[int, int], CriticHead],
) -> Run:
    obs_dim, action_dim, action_limit, horizon = env_dims(
        cfg.env_name, cfg.horizon
    )
    key = jax.random.key(cfg.seed)
    key, cem_key, td3_key = jax.random.split(key, 3)

    template = ActorHead(
        cfg.embedding_dim, action_dim, key=cem_key, action_limit=action_limit
    )
    flat_template, unravel_head = ravel_pytree(template)
    cem = CEM(
        flat_template.shape[0],
        cem_key,
        sigma_init=cfg.sigma_init,
        pop_size=cfg.pop_size,
        damp=cfg.damp,
        damp_limit=cfg.damp_limit,
        elitisim=cfg.elitism,
    )

    head = make_critic_head(obs_dim, action_dim)
    actor_optimizer = optax.adam(cfg.actor_lr)
    critic_optimizer = optax.adam(cfg.critic_lr)
    td3_state = init_td3(
        obs_dim,
        action_dim,
        cfg.embedding_dim,
        key=td3_key,
        head=head,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )
    td3_cfg = TD3Config(
        gamma=cfg.gamma,
        tau=cfg.tau,
        policy_noise=cfg.policy_noise,
        noise_clip=cfg.noise_clip,
        action_limit=action_limit,
        policy_freq=cfg.policy_freq,
        batch_size=cfg.batch_size,
    )
    eval_env = environments.make_vec_env(
        cfg.eval_env_name or cfg.env_name,
        2 * cfg.eval_episodes,
        async_=cfg.async_env,
        to_jax=True,
    )
    return Run(
        cfg=cfg,
        horizon=horizon,
        action_limit=action_limit,
        buffer=Buffer(cfg.buffer_capacity, obs_dim, action_dim),
        cem=cem,
        unravel_head=unravel_head,
        head=head,
        td3_state=td3_state,
        train_td3=make_td3_train(
            td3_cfg, head, actor_optimizer, critic_optimizer
        ),
        eval_env=eval_env,
        key=key,
        undiscount_scale=horizon
        * (1.0 - cfg.gamma)
        / (1.0 - cfg.gamma**horizon),
        closers=[eval_env.close],
    )


def next_key(run: Run) -> jax.Array:
    run.key, sub = jax.random.split(run.key)
    return sub


def ask_population(run: Run) -> tuple[jax.Array, ActorHead]:
    flat_pop = run.cem.ask(next_key(run))
    if run.pending_injection is not None:
        rl_flat, _ = ravel_pytree(run.td3_state.online.actor)
        flat_pop = flat_pop.at[run.pending_injection].set(rl_flat)
        run.pending_injection = None
    return flat_pop, jax.vmap(run.unravel_head)(flat_pop)


def train_and_merge(run: Run, gen_env_steps: int) -> Metrics:
    # scales with steps actually collected: a fixed count inflates the
    # replay ratio on cheap generations
    num_updates = int(run.cfg.train_ratio * gen_env_steps)
    td3_state, critic_loss = run.train_td3(
        run.td3_state,
        run.buffer.data,
        jnp.asarray(len(run.buffer)),
        jnp.asarray(num_updates),
        next_key(run),
    )
    champion = run.unravel_head(run.cem.elite)
    actor = genetic_soft_update(
        td3_state.online.actor, champion, run.cfg.ea_tau
    )
    run.td3_state = td3_state._replace(
        online=td3_state.online._replace(actor=actor)
    )
    return {
        "train/critic_loss": float(critic_loss),
    }


def maybe_evaluate(run: Run) -> Metrics:
    if run.env_steps < run.next_eval:
        return {}
    run.next_eval = run.env_steps + run.cfg.eval_interval
    heads = jax.tree.map(
        lambda a, b: jnp.stack([a, b]) if eqx.is_array(a) else a,
        run.td3_state.online.actor,
        run.unravel_head(run.cem.elite),
    )
    # fixed key: every eval starts from the same initial states
    rl, elite = evaluate_heads(
        run.eval_env,
        jax.random.key(run.cfg.seed + 10_000),
        run.td3_state.online.embedding,
        heads,
        run.horizon,
    )
    return {"perf/eval_rl": float(rl), "perf/eval_elite": float(elite)}


def to_floats(metrics: dict[str, Any]) -> Metrics:
    return {k: float(v) for k, v in jax.device_get(metrics).items()}


def print_generation(run: Run, metrics: Metrics, tag: str) -> None:
    def show(name: str, fmt: str = "8.1f") -> str:
        value = metrics.get(name)
        return "-" if value is None else format(value, fmt)

    print(
        f"gen {run.generation:4d} | steps {run.env_steps:9d} | {tag} | "
        f"real={show('cost/real_frac', '.2f')} | "
        f"elite_ovl={show('select/elite_overlap', '.2f')} | "
        f"eval_rl={show('perf/eval_rl')} "
        f"eval_elite={show('perf/eval_elite')} | "
        f"critic_loss={show('train/critic_loss', '.4f')}"
    )


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
    critic1: Any,
    critic2: Any,
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
    # onto real_fitness's undiscounted scale; rank-preserving for CEM
    return discounted * undiscount_scale


def _rollout(
    run: Run,
    vec_env: gym.vector.VectorEnv,
    pop_heads: ActorHead,
    warmup: bool,
) -> tuple[jax.Array, jax.Array]:
    cfg = run.cfg
    embedding = run.td3_state.online.embedding
    actor = run.td3_state.online.actor

    def policy(act_key: jax.Array, states: jnp.ndarray) -> jnp.ndarray:
        return _step_policy(
            embedding,
            pop_heads,
            actor,
            states,
            act_key,
            run.action_limit,
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
    ) -> None:
        if step >= cfg.h_steps:
            return
        h_reward[:, step] = np.asarray(reward)
        h_done[:, step] = np.asarray(terminated)
        if step == cfg.h_steps - 1:
            h_state_box.append(next_states)

    returns, _ = collect_parallel_episode(
        vec_env,
        next_key(run),
        policy,
        run.buffer,
        run.horizon,
        on_step=capture_h_step,
    )
    surrogate = _surrogate_fitness(
        embedding,
        run.td3_state.online.critic1,
        run.td3_state.online.critic2,
        pop_heads,
        h_state_box[0][:-1],
        jnp.asarray(h_reward[:-1]),
        jnp.asarray(h_done[:-1]),
        cfg.h_steps,
        cfg.gamma,
        run.undiscount_scale,
    )
    return returns[:-1], surrogate


def _generation(run: Run, vec_env: gym.vector.VectorEnv) -> Metrics:
    cfg = run.cfg
    # ERL never truncates: every generation is the same fixed cost.
    gen_env_steps = run.horizon * (cfg.pop_size + 1)
    run.env_steps += gen_env_steps
    flat_pop, pop_heads = ask_population(run)
    warmup = len(run.buffer) < cfg.warmup_steps

    truth, surrogate = _rollout(run, vec_env, pop_heads, warmup)
    use_real = warmup or bool(jax.random.bernoulli(next_key(run), cfg.theta))
    fitness = truth if use_real else surrogate
    run.cem.tell(fitness, flat_pop)

    metrics: Metrics = {
        "cost/real_frac": float(use_real),
    }
    if not warmup:
        metrics |= to_floats(
            selection_metrics(truth, fitness, surrogate, run.cem.parents)
        )
        metrics |= train_and_merge(run, gen_env_steps)
        if run.generation % cfg.rl_to_ea_sync_period == 0:
            run.pending_injection = int(jnp.argmin(fitness))
    return metrics


def train(
    cfg: ERLConfig,
    *,
    on_generation: Callable[[Metrics], None] | None = None,
) -> TD3State:
    run = build_run(
        cfg,
        lambda obs_dim, action_dim: make_head(
            "random", obs_dim, action_dim, cfg.critic_hidden_dims
        ),
    )
    if cfg.h_steps > run.horizon:
        run.close()
        raise ValueError(
            f"h_steps ({cfg.h_steps}) must be <= horizon ({run.horizon})"
        )
    vec_env = environments.make_vec_env(
        cfg.env_name, cfg.pop_size + 1, async_=cfg.async_env, to_jax=True
    )
    run.closers.append(vec_env.close)
    try:
        while run.env_steps < cfg.step_budget:
            metrics = _generation(run, vec_env)
            metrics |= maybe_evaluate(run)
            metrics |= {
                "generation": float(run.generation),
                "env_steps": float(run.env_steps),
            }
            print_generation(run, metrics, "erl")
            if on_generation is not None:
                on_generation(metrics)
            run.generation += 1
    finally:
        run.close()
    return run.td3_state


def evaluate_actor(
    env_name: str,
    td3_state: TD3State,
    *,
    episodes: int = 5,
    seed: int | None = None,
) -> float:
    env = environments.make_env(env_name)
    try:
        total_reward = 0.0
        for i in range(episodes):
            state, _ = env.reset(seed=seed if i == 0 else None)
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
