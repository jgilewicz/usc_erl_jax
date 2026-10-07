from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from common.critic_heads import StatsFn
from modules.deep_modules import ActorHead, SharedStateEmbedding


class Gate(NamedTuple):
    real: jax.Array
    cv: jax.Array


class Calibrated(NamedTuple):
    # what CEM is told: true return where real, calibrated LCB elsewhere
    fitness: jax.Array
    # calibrated LCB on every individual
    surrogate: jax.Array


@eqx.filter_jit
def population_stats(
    stats: StatsFn,
    embedding: SharedStateEmbedding,
    critic: Any,
    pop_heads: ActorHead,
    states: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    # (pop,) mu_Q and sigma_Q averaged over one shared replay batch; one key
    # for every individual so MC-dropout masks are common random numbers
    z = jax.vmap(embedding)(states)

    def individual(head: ActorHead) -> tuple[jax.Array, jax.Array]:
        mu, sigma = stats(critic, states, jax.vmap(head)(z), key)
        return mu.mean(), sigma.mean()

    return eqx.filter_vmap(individual)(pop_heads)


@partial(jax.jit, static_argnames=("uses_uncertainty",))
def gate(
    mu: jax.Array,
    sigma: jax.Array,
    key: jax.Array,
    omega: float,
    epsilon: float,
    mad_k: float,
    uses_uncertainty: bool,
) -> Gate:
    coin = jax.random.uniform(key, mu.shape)
    # computed under the random gate too: its AUC is then measured without
    # cv having steered which individuals were selected
    cv = jnp.nan_to_num(
        sigma / (jnp.sqrt(jnp.abs(mu)) + 1.0), nan=0.0, posinf=1e3, neginf=0.0
    )
    if not uses_uncertainty:
        # omega = P(surrogate) per individual
        return Gate(coin > omega, cv)
    median = jnp.median(cv)
    threshold = median + mad_k * jnp.median(jnp.abs(cv - median))
    return Gate((cv > threshold) | (coin < epsilon), cv)


def _lcb(
    mu: jax.Array, sigma: jax.Array, beta: jax.Array, scale: float
) -> jax.Array:
    # scale maps the discounted Q onto the undiscounted return's units
    return scale * (mu - beta * sigma)


def _offset(observed: jax.Array, real: jax.Array, lcb: jax.Array) -> jax.Array:
    # critic bias measured on this generation's real rollouts; with none, no
    # real return is mixed in and a shared offset cannot change the ranking
    n_real = jnp.sum(real)
    residual = jnp.where(real, observed - lcb, 0.0)
    return jnp.sum(residual) / jnp.maximum(n_real, 1)


@jax.jit
def calibrate(
    observed: jax.Array,
    real: jax.Array,
    mu: jax.Array,
    sigma: jax.Array,
    log_beta: jax.Array,
    scale: float,
) -> Calibrated:
    # observed must be zero outside `real`: shadow returns never enter here
    lcb = _lcb(mu, sigma, jnp.exp(log_beta), scale)
    surrogate = lcb + _offset(observed, real, lcb)
    return Calibrated(jnp.where(real, observed, surrogate), surrogate)


def _beta_loss(
    log_beta: jax.Array,
    observed: jax.Array,
    real: jax.Array,
    mu: jax.Array,
    sigma: jax.Array,
    scale: float,
) -> jax.Array:
    # residual variance after the offset: beta explains what the shared
    # offset cannot, i.e. how much sigma tracks per-individual critic error
    lcb = _lcb(mu, sigma, jnp.exp(log_beta), scale)
    residual = observed - lcb - _offset(observed, real, lcb)
    n_real = jnp.maximum(jnp.sum(real), 1)
    return jnp.sum(jnp.where(real, residual**2, 0.0)) / n_real


def make_beta_step(optimizer: optax.GradientTransformation) -> Any:
    @jax.jit
    def step(
        log_beta: jax.Array,
        opt_state: optax.OptState,
        observed: jax.Array,
        real: jax.Array,
        mu: jax.Array,
        sigma: jax.Array,
        scale: float,
    ) -> tuple[jax.Array, optax.OptState]:
        grad = jax.grad(_beta_loss)(log_beta, observed, real, mu, sigma, scale)
        updates, opt_state = optimizer.update(grad, opt_state, log_beta)
        return jnp.asarray(optax.apply_updates(log_beta, updates)), opt_state

    return step
