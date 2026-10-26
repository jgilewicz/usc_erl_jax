from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from modules.deep_modules import Critic, EvidentialCritic

# usc_erl's SurrogateMode. "random" has no uncertainty: plain twin critics.
Mode = Literal["random", "dropout", "ensemble", "evidential", "bootstrapped"]
MODES: tuple[str, ...] = get_args(Mode)

# EnsembleModule.compute_loss: each member regresses its own Bernoulli(0.5)
# half of the batch against a 2%-noised target; masks + init make members
# disagree.
_BOOTSTRAP_P = 0.5
_TARGET_NOISE = 0.02

PointFn = Callable[[Any, jax.Array, jax.Array, jax.Array], jax.Array]
LossFn = Callable[[Any, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
StatsFn = Callable[
    [Any, jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]
]
# (target critic1, next states, next actions, shared next value, key)
#   -> next-state value critic1 regresses on; critic2 always gets `shared`
NextValueFn = Callable[
    [Any, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array
]


def _shared_next_value(
    critic: Any,
    next_states: jax.Array,
    next_actions: jax.Array,
    shared: jax.Array,
    key: jax.Array,
) -> jax.Array:
    return shared


# (critic1, states, actions) -> (K, B, 1) per-member Q; None without members
MemberFn = Callable[[Any, jax.Array, jax.Array], jax.Array]


class CriticHead(NamedTuple):
    # critic1 carries the mode's uncertainty; critic2 is always a plain
    # Critic (usc_erl's asymmetric twin), so only critic1 goes through here.
    build: Callable[[jax.Array], Any]
    build_twin: Callable[[jax.Array], Critic]
    # (B, 1) point estimate: TD3 target, actor loss
    point: PointFn
    loss: LossFn
    # per-state (mu, sigma), each (B,)
    stats: StatsFn
    next_value: NextValueFn = _shared_next_value
    members: MemberFn | None = None


def plain_point(
    critic: Critic, states: jax.Array, actions: jax.Array, key: jax.Array
) -> jax.Array:
    # per-row keys drive dropout; with dropout=0 they are ignored
    keys = jax.random.split(key, states.shape[0])
    return jax.vmap(lambda s, a, k: critic(s, a, key=k))(states, actions, keys)


def plain_loss(
    critic: Critic,
    states: jax.Array,
    actions: jax.Array,
    target: jax.Array,
    key: jax.Array,
) -> jax.Array:
    return jnp.mean((plain_point(critic, states, actions, key) - target) ** 2)


def _no_sigma(
    critic: Critic, states: jax.Array, actions: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    mu = plain_point(critic, states, actions, key)[:, 0]
    return mu, jnp.zeros_like(mu)


def _mc_dropout_stats(samples: int) -> StatsFn:
    def stats(
        critic: Critic, states: jax.Array, actions: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        q = jax.vmap(lambda k: plain_point(critic, states, actions, k)[:, 0])(
            jax.random.split(key, samples)
        )
        return q.mean(axis=0), q.std(axis=0, ddof=1)

    return stats


def _member_q(
    critics: Critic, states: jax.Array, actions: jax.Array
) -> jax.Array:
    # critics: one Critic whose array leaves carry a leading (K,) axis
    return eqx.filter_vmap(lambda c: jax.vmap(c)(states, actions))(critics)


def _ensemble_point(
    critics: Critic, states: jax.Array, actions: jax.Array, key: jax.Array
) -> jax.Array:
    return _member_q(critics, states, actions).mean(axis=0)


def _ensemble_stats(
    critics: Critic, states: jax.Array, actions: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    q = _member_q(critics, states, actions)[..., 0]
    return q.mean(axis=0), q.std(axis=0, ddof=1)


def _ensemble_loss(
    critics: Critic,
    states: jax.Array,
    actions: jax.Array,
    target: jax.Array,
    key: jax.Array,
) -> jax.Array:
    q = _member_q(critics, states, actions)
    mask_key, noise_key = jax.random.split(key)
    mask = jax.random.bernoulli(mask_key, _BOOTSTRAP_P, q.shape)
    noise = jax.random.normal(noise_key, q.shape)
    noisy_target = target[None] * (1.0 + _TARGET_NOISE * noise)
    loss = optax.huber_loss(q, noisy_target, delta=1.0)
    per_member = jnp.sum(loss * mask, axis=(1, 2)) / (
        jnp.sum(mask, axis=(1, 2)) + 1e-8
    )
    return jnp.sum(per_member)


class Bootstrapped(NamedTuple):
    # (K,)-stacked trainable members and frozen random priors (Osband et
    # al. 2018): Q_k = member_k + prior_scale * prior_k
    members: Critic
    priors: Critic


def _bootstrapped_q(prior_scale: float) -> Callable[..., jax.Array]:
    def member_q(
        critic: Bootstrapped, states: jax.Array, actions: jax.Array
    ) -> jax.Array:
        prior = jax.lax.stop_gradient(_member_q(critic.priors, states, actions))
        return _member_q(critic.members, states, actions) + prior_scale * prior

    return member_q


def _bootstrapped_head(
    build: Callable[[jax.Array], Bootstrapped],
    twin: Callable[[jax.Array], Critic],
    prior_scale: float,
) -> CriticHead:
    member_q = _bootstrapped_q(prior_scale)

    def point(
        critic: Bootstrapped, states: jax.Array, actions: jax.Array, key: Any
    ) -> jax.Array:
        return member_q(critic, states, actions).mean(axis=0)

    def stats(
        critic: Bootstrapped, states: jax.Array, actions: jax.Array, key: Any
    ) -> tuple[jax.Array, jax.Array]:
        q = member_q(critic, states, actions)[..., 0]
        return q.mean(axis=0), q.std(axis=0, ddof=1)

    def next_value(
        critic: Bootstrapped,
        next_states: jax.Array,
        next_actions: jax.Array,
        shared: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        # (K, B, 1) per-member values: each member bootstraps on itself, so
        # disagreement propagates through Bellman; the shift keeps TD3's
        # twin-min pessimism on the mean (mean_k target = shared)
        q = member_q(critic, next_states, next_actions)
        return q - q.mean(axis=0) + shared

    def loss(
        critic: Bootstrapped,
        states: jax.Array,
        actions: jax.Array,
        target: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        q = member_q(critic, states, actions)
        mask = jax.random.bernoulli(key, _BOOTSTRAP_P, q.shape)
        per_member = jnp.sum(
            optax.huber_loss(q, target, delta=1.0) * mask, axis=(1, 2)
        ) / (jnp.sum(mask, axis=(1, 2)) + 1e-8)
        return jnp.sum(per_member)

    return CriticHead(build, twin, point, loss, stats, next_value, member_q)


def _evidential_point(
    critic: EvidentialCritic,
    states: jax.Array,
    actions: jax.Array,
    key: jax.Array,
) -> jax.Array:
    return jax.vmap(critic)(states, actions)[0]


def _evidential_stats(
    critic: EvidentialCritic,
    states: jax.Array,
    actions: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    mu, v, alpha, beta = jax.vmap(critic)(states, actions)
    epistemic = jnp.nan_to_num(
        beta / (v * (alpha - 1.0) + 1e-6), nan=0.0, posinf=1e3, neginf=0.0
    )
    return mu[:, 0], jnp.sqrt(jnp.maximum(epistemic, 0.0))[:, 0]


def evidential_nll(
    target: jax.Array,
    mu: jax.Array,
    v: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    lam: float,
) -> jax.Array:
    # alpha clamp avoids lgamma overflow in float32
    alpha_safe = jnp.minimum(alpha, 1e4)
    two_b = jnp.maximum(2.0 * beta * (1.0 + v), 1e-8)
    error = target - mu
    nll = (
        0.5 * math.log(math.pi)
        - 0.5 * jnp.log(v)
        - alpha_safe * jnp.log(two_b)
        + (alpha_safe + 0.5) * jnp.log(jnp.maximum(two_b + v * error**2, 1e-8))
        + jax.lax.lgamma(alpha_safe)
        - jax.lax.lgamma(alpha_safe + 0.5)
    )
    # evidence regulariser: large errors are pushed toward low evidence
    reg = jnp.abs(error) * (2.0 * v + alpha)
    return jnp.mean(nll + lam * reg)


def _evidential_loss(lam: float) -> LossFn:
    def loss(
        critic: EvidentialCritic,
        states: jax.Array,
        actions: jax.Array,
        target: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        mu, v, alpha, beta = jax.vmap(critic)(states, actions)
        return evidential_nll(target, mu, v, alpha, beta, lam)

    return loss


def make_head(
    mode: str,
    obs_dim: int,
    action_dim: int,
    hidden_dims: tuple[int, int],
    *,
    dropout_p: float = 0.0,
    mc_samples: int = 25,
    k_ensembles: int = 5,
    evidential_lam: float = 0.1,
    prior_scale: float = 3.0,
) -> CriticHead:
    # usc_erl bakes dropout into every critic in dropout mode, twin included
    dropout = dropout_p if mode == "dropout" else 0.0

    def plain(key: jax.Array) -> Critic:
        return Critic(
            obs_dim,
            action_dim,
            key=key,
            dropout=dropout,
            hidden_dims=hidden_dims,
        )

    def ensemble(key: jax.Array) -> Critic:
        return eqx.filter_vmap(plain)(jax.random.split(key, k_ensembles))

    match mode:
        case "random":
            return CriticHead(plain, plain, plain_point, plain_loss, _no_sigma)
        case "dropout":
            stats = _mc_dropout_stats(mc_samples)
            return CriticHead(plain, plain, plain_point, plain_loss, stats)
        case "ensemble":
            return CriticHead(
                ensemble,
                plain,
                _ensemble_point,
                _ensemble_loss,
                _ensemble_stats,
                members=_member_q,
            )
        case "evidential":

            def evidential(key: jax.Array) -> EvidentialCritic:
                return EvidentialCritic(
                    obs_dim, action_dim, key=key, hidden_dims=hidden_dims
                )

            return CriticHead(
                evidential,
                plain,
                _evidential_point,
                _evidential_loss(evidential_lam),
                _evidential_stats,
            )
        case "bootstrapped":

            def bootstrapped(key: jax.Array) -> Bootstrapped:
                member_key, prior_key = jax.random.split(key)
                return Bootstrapped(ensemble(member_key), ensemble(prior_key))

            return _bootstrapped_head(bootstrapped, plain, prior_scale)
        case other:
            raise ValueError(f"unknown surrogate mode {other!r}; use {MODES}")
