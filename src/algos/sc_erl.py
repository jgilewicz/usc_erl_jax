from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

import environments
from algos.erl import (
    ERLConfig,
    Metrics,
    Run,
    ask_population,
    build_run,
    maybe_evaluate,
    next_key,
    print_generation,
    to_floats,
    train_and_merge,
)
from common.critic_heads import MODES, CriticHead, make_head
from common.metrics import (
    gate_metrics,
    selection_metrics,
    uncertainty_gate_metrics,
)
from common.rollout import collect_parallel_episode, heads_policy
from common.surrogate import (
    Calibrated,
    Gate,
    calibrate,
    gate,
    make_beta_step,
    population_stats,
)
from common.td3 import TD3State
from modules.deep_modules import ActorHead, SharedStateEmbedding

# CEM samples are iid, so slots are interchangeable; elitism owns slot -1
_RL_SLOT = 0
_ANCHOR_SLOT = 1


@dataclass(frozen=True)
class SCERLConfig(ERLConfig):
    # `theta`/`h_steps` inherited but inert: the per-individual gate
    # replaces ERL's per-generation coin
    mode: str = "random"
    # ablation: keep the mode's critic but gate at random (omega)
    random_gate: bool = False
    # P(surrogate) per individual under the random gate
    omega: float = 0.75
    # P(forced real rollout) per individual on top of the cv gate
    epsilon: float = 0.142
    # gate threshold = median(cv) + mad_k * MAD(cv)
    mad_k: float = 2.483
    # initial LCB weight, adapted online
    beta: float = 0.432
    beta_lr: float = 1e-3
    # shared replay states every individual is scored on
    surrogate_batch: int = 1024
    mc_samples: int = 25
    dropout_p: float = 0.1
    k_ensembles: int = 5
    evidential_lam: float = 0.1

    def __post_init__(self) -> None:
        checks = {
            f"mode must be one of {MODES}": self.mode in MODES,
            "omega must be in [0, 1]": 0.0 <= self.omega <= 1.0,
            "epsilon must be in [0, 1]": 0.0 <= self.epsilon <= 1.0,
            "beta must be > 0": self.beta > 0.0,
            "pop_size must be >= 3 (RL, anchor, elite slots)": self.pop_size
            >= 3,
            "warmup_steps must be >= 1": self.warmup_steps >= 1,
            "dropout_p must be in (0, 1)": self.mode != "dropout"
            or 0.0 < self.dropout_p < 1.0,
            "mc_samples must be >= 2": self.mode != "dropout"
            or self.mc_samples >= 2,
            "k_ensembles must be >= 2": self.mode != "ensemble"
            or self.k_ensembles >= 2,
        }
        failed = [msg for msg, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"invalid SCERLConfig: {'; '.join(failed)}")

    @property
    def has_sigma(self) -> bool:
        return self.mode != "random"

    @property
    def uses_uncertainty(self) -> bool:
        return self.has_sigma and not self.random_gate


@dataclass
class Surrogate:
    rl_env: gym.vector.VectorEnv
    pop_env: gym.vector.VectorEnv
    stats: Any
    log_beta: jax.Array
    beta_opt_state: optax.OptState
    beta_step: Any
    best_real_flat: jax.Array | None = None
    best_real_return: float = -np.inf
    pending_anchor: bool = False


@eqx.filter_jit
def _rl_policy(
    embedding: SharedStateEmbedding,
    actor: ActorHead,
    states: jnp.ndarray,
    key: jax.Array,
    action_limit: float,
    exploration_noise: float,
    warmup: bool,
) -> jnp.ndarray:
    raw = jax.vmap(lambda s: actor(embedding(s)))(states)
    rl_key, warm_key = jax.random.split(key)
    if warmup:
        return jax.random.uniform(
            warm_key, raw.shape, minval=-action_limit, maxval=action_limit
        )
    noise = jax.random.normal(rl_key, raw.shape) * exploration_noise
    return jnp.clip(raw + noise, -action_limit, action_limit)


def _head_factory(cfg: SCERLConfig) -> Callable[[int, int], CriticHead]:
    def factory(obs_dim: int, action_dim: int) -> CriticHead:
        return make_head(
            cfg.mode,
            obs_dim,
            action_dim,
            cfg.critic_hidden_dims,
            dropout_p=cfg.dropout_p,
            mc_samples=cfg.mc_samples,
            k_ensembles=cfg.k_ensembles,
            evidential_lam=cfg.evidential_lam,
        )

    return factory


def _build_surrogate(run: Run, cfg: SCERLConfig) -> Surrogate:
    # split vec envs: the RL actor always rolls out, the population only
    # stores the rollouts the gate sends to the env
    rl_env = environments.make_vec_env(
        cfg.env_name, 1, async_=cfg.async_env, to_jax=True
    )
    pop_env = environments.make_vec_env(
        cfg.env_name, cfg.pop_size, async_=cfg.async_env, to_jax=True
    )
    run.closers += [rl_env.close, pop_env.close]
    optimizer = optax.adam(cfg.beta_lr)
    log_beta = jnp.log(jnp.asarray(cfg.beta, dtype=jnp.float32))
    return Surrogate(
        rl_env=rl_env,
        pop_env=pop_env,
        stats=run.head.stats,
        log_beta=log_beta,
        beta_opt_state=optimizer.init(log_beta),
        beta_step=make_beta_step(optimizer),
    )


def _inject_anchor(run: Run, sc: Surrogate, flat_pop: jax.Array) -> jax.Array:
    # best-ever real-evaluated actor re-enters after every generation that
    # rolled anyone out, unless it already is the CEM elite
    anchor = sc.best_real_flat
    if not sc.pending_anchor or anchor is None:
        return flat_pop
    sc.pending_anchor = False
    if bool(jnp.array_equal(anchor, run.cem.elite)):
        return flat_pop
    return flat_pop.at[_ANCHOR_SLOT].set(anchor)


def _score(
    run: Run,
    sc: Surrogate,
    cfg: SCERLConfig,
    pop_heads: ActorHead,
    warmup: bool,
) -> tuple[jax.Array, jax.Array, Gate]:
    zeros = jnp.zeros(cfg.pop_size)
    if warmup:
        everyone = jnp.ones(cfg.pop_size, dtype=bool)
        return (
            zeros,
            zeros,
            Gate(everyone, everyone, zeros, jnp.asarray(jnp.nan)),
        )
    states = run.buffer.sample(next_key(run), cfg.surrogate_batch)["state"]
    mu, sigma = population_stats(
        sc.stats,
        run.td3_state.online.embedding,
        run.td3_state.online.critic1,
        pop_heads,
        states,
        next_key(run),
    )
    decision = gate(
        mu,
        sigma,
        next_key(run),
        cfg.omega,
        cfg.epsilon,
        cfg.mad_k,
        cfg.uses_uncertainty,
    )
    return mu, sigma, decision


def _rollouts(
    run: Run,
    sc: Surrogate,
    pop_heads: ActorHead,
    real: jax.Array,
    warmup: bool,
) -> tuple[float, jax.Array]:
    online = run.td3_state.online

    def rl_policy(act_key: jax.Array, states: jnp.ndarray) -> jnp.ndarray:
        return _rl_policy(
            online.embedding,
            online.actor,
            states,
            act_key,
            run.action_limit,
            run.cfg.exploration_noise,
            warmup,
        )

    def pop_policy(act_key: jax.Array, states: jnp.ndarray) -> jnp.ndarray:
        return heads_policy(online.embedding, pop_heads, states)

    rl_returns, _ = collect_parallel_episode(
        sc.rl_env, next_key(run), rl_policy, run.buffer, run.horizon
    )
    # shadow rollout: every individual steps (same wall-clock under async),
    # only `real` ones reach the buffer; the rest is ground truth for metrics
    truth, _ = collect_parallel_episode(
        sc.pop_env,
        next_key(run),
        pop_policy,
        run.buffer,
        run.horizon,
        store_mask=np.asarray(real),
    )
    return float(rl_returns[0]), truth


def _update_anchor(
    sc: Surrogate, flat_pop: jax.Array, observed: jax.Array, real: jax.Array
) -> None:
    best = int(jnp.argmax(jnp.where(real, observed, -jnp.inf)))
    if bool(real[best]) and float(observed[best]) > sc.best_real_return:
        sc.best_real_return = float(observed[best])
        sc.best_real_flat = flat_pop[best]


def _fit_beta(
    run: Run,
    sc: Surrogate,
    observed: jax.Array,
    decision: Gate,
    mu: jax.Array,
    sigma: jax.Array,
) -> None:
    sc.log_beta, sc.beta_opt_state = sc.beta_step(
        sc.log_beta,
        sc.beta_opt_state,
        observed,
        decision.real,
        mu,
        sigma,
        run.undiscount_scale,
    )


def _surrogate_metrics(
    cfg: SCERLConfig,
    parents: int,
    truth: jax.Array,
    mu: jax.Array,
    calibrated: Calibrated,
    decision: Gate,
) -> Metrics:
    metrics = selection_metrics(
        truth, calibrated.fitness, calibrated.surrogate, parents
    ) | gate_metrics(truth, mu, decision.real, parents)
    if cfg.has_sigma:
        metrics |= uncertainty_gate_metrics(
            truth,
            mu,
            decision.cv,
            decision.real,
            decision.deterministic,
            decision.threshold,
            parents,
        )
    return to_floats(metrics)


def _generation(run: Run, sc: Surrogate, cfg: SCERLConfig) -> Metrics:
    flat_pop, _ = ask_population(run)
    flat_pop = _inject_anchor(run, sc, flat_pop)
    pop_heads = jax.vmap(run.unravel_head)(flat_pop)
    warmup = len(run.buffer) < cfg.warmup_steps

    mu, sigma, decision = _score(run, sc, cfg, pop_heads, warmup)
    rl_return, truth = _rollouts(run, sc, pop_heads, decision.real, warmup)
    n_real = int(jnp.sum(decision.real))
    gen_env_steps = run.horizon * (1 + n_real)
    run.env_steps += gen_env_steps

    # the only path from the shadow rollout into the algorithm
    observed = jnp.where(decision.real, truth, 0.0)
    calibrated = calibrate(
        observed, decision.real, mu, sigma, sc.log_beta, run.undiscount_scale
    )
    _update_anchor(sc, flat_pop, observed, decision.real)
    run.cem.tell(calibrated.fitness, flat_pop, elite_candidates=decision.real)
    sc.pending_anchor = n_real > 0

    metrics: Metrics = {
        "perf/rl_return": rl_return,
        "perf/pop_true_best": float(jnp.max(truth)),
        "perf/pop_true_mean": float(jnp.mean(truth)),
        "cost/env_steps_gen": float(gen_env_steps),
        "cost/real_frac": n_real / cfg.pop_size,
        "train/buffer_size": float(len(run.buffer)),
    }
    if warmup:
        return metrics
    if cfg.has_sigma and n_real >= 2:
        _fit_beta(run, sc, observed, decision, mu, sigma)
    metrics |= _surrogate_metrics(
        cfg, run.cem.parents, truth, mu, calibrated, decision
    )
    metrics |= train_and_merge(run, gen_env_steps)
    if run.generation % cfg.rl_to_ea_sync_period == 0 and n_real > 0:
        run.pending_injection = _RL_SLOT
    return metrics


def train(
    cfg: SCERLConfig,
    *,
    on_generation: Callable[[Metrics], None] | None = None,
) -> TD3State:
    run = build_run(cfg, _head_factory(cfg))
    try:
        sc = _build_surrogate(run, cfg)
        while run.env_steps < cfg.step_budget:
            metrics = _generation(run, sc, cfg)
            metrics |= maybe_evaluate(run)
            metrics |= {
                "generation": float(run.generation),
                "env_steps": float(run.env_steps),
            }
            print_generation(
                run,
                metrics,
                f"{cfg.mode} real={metrics['cost/real_frac']:.1f}",
            )
            if on_generation is not None:
                on_generation(metrics)
            run.generation += 1
    finally:
        run.close()
    return run.td3_state


if __name__ == "__main__":
    train(SCERLConfig())
