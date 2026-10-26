from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

import environments
from common.critic_heads import MemberFn
from common.horizon_probe import HorizonProbe
from common.replay_buffer import Buffer
from common.rollout import collect_parallel_episode, heads_policy
from modules.deep_modules import ActorHead, SharedStateEmbedding

# metric-only: own envs, own key stream, own CEM samples and own copies of
# the heads; nothing recorded here may flow back into gate, calibration,
# beta, CEM, anchor or the buffer
# per step count T: mean = x_j + T ascent steps on mean_k Q_k, member = the
# same on Q_{j mod K}, rand = mean + a random direction of norm
# |member - mean| (is the member gain uncertainty or just spread?)
STEPS = (50, 200)
SETS = ("cem",) + tuple(
    f"{kind}{t}" for t in STEPS for kind in ("mean", "member", "rand")
)
CANDIDATES = 20
GRAD_BATCH = 256
_KEY_OFFSET = 40_000


@dataclass
class VariationProbe:
    vec_env: gym.vector.VectorEnv
    store: HorizonProbe
    key: jax.Array
    optimizer: optax.GradientTransformation


def make_variation_probe(
    env_name: str, seed: int, path: Path, *, async_env: bool, lr: float
) -> VariationProbe:
    vec_env = environments.make_vec_env(
        env_name, CANDIDATES, async_=async_env, to_jax=True
    )
    return VariationProbe(
        vec_env=vec_env,
        # gamma is unused by the variation analysis; HorizonProbe is the
        # npz writer
        store=HorizonProbe(path, gamma=0.0),
        key=jax.random.key(seed + _KEY_OFFSET),
        optimizer=optax.adam(lr),
    )


@dataclass(frozen=True)
class Population:
    # CEM search distribution as it stood when this generation was asked
    mean: jax.Array
    cov: jax.Array
    scale: float
    unravel: Callable[[jax.Array], ActorHead]


class Posterior(eqx.Module):
    # critic1 as it stood when the gate decided (before this gen's training)
    members: MemberFn = eqx.field(static=True)
    embedding: SharedStateEmbedding
    critic: Any
    k: int = eqx.field(static=True)


def _member_values(
    post: Posterior,
    unravel: Callable[[jax.Array], ActorHead],
    flat: jax.Array,
    states: jax.Array,
    z: jax.Array,
) -> jax.Array:
    # (K,) batch-mean Q_k(s, pi_flat(s)); embedding frozen, only the head
    # moves (CEM evolves ActorHead params)
    actions = jax.vmap(unravel(flat))(z)
    return post.members(post.critic, states, actions)[..., 0].mean(axis=1)


@eqx.filter_jit
def _ascend(
    post: Posterior,
    unravel: Callable[[jax.Array], ActorHead],
    optimizer: optax.GradientTransformation,
    flat: jax.Array,
    weights: jax.Array,
    states: jax.Array,
) -> jax.Array:
    # flat (N, D), weights (N, K) over members, states (T, B, obs):
    # T Adam steps per candidate on -weights . Q, one batch per step
    z = jax.vmap(jax.vmap(post.embedding))(states)

    def candidate(start: jax.Array, w: jax.Array) -> jax.Array:
        def loss(x: jax.Array, s: jax.Array, zs: jax.Array) -> jax.Array:
            return -jnp.dot(w, _member_values(post, unravel, x, s, zs))

        def step(
            carry: tuple[jax.Array, optax.OptState],
            batch: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, optax.OptState], None]:
            x, opt_state = carry
            grad = jax.grad(loss)(x, *batch)
            updates, opt_state = optimizer.update(grad, opt_state, x)
            return (
                jnp.asarray(optax.apply_updates(x, updates)),
                opt_state,
            ), None

        (end, _), _ = jax.lax.scan(
            step, (start, optimizer.init(start)), (states, z)
        )
        return end

    return jax.vmap(candidate)(flat, weights)


def _cosine(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12)


@eqx.filter_jit
def _diagnostics(
    post: Posterior,
    unravel: Callable[[jax.Array], ActorHead],
    sets: jax.Array,
    states: jax.Array,
) -> dict[str, jax.Array]:
    # sets (S, N, D) in SETS order; states (B, obs), one replay batch
    z = jax.vmap(post.embedding)(states)
    cem = sets[0]

    def pairwise_cos(flat: jax.Array) -> jax.Array:
        # mean cos between member gradients at the CEM sample: if ~1, the
        # posterior proposes the mean-critic direction
        jac = jax.jacrev(lambda f: _member_values(post, unravel, f, states, z))(
            flat
        )
        unit = jac / (jnp.linalg.norm(jac, axis=1, keepdims=True) + 1e-12)
        gram = unit @ unit.T
        return (gram.sum() - jnp.trace(gram)) / (post.k * (post.k - 1))

    q_mean = jax.vmap(
        jax.vmap(lambda f: _member_values(post, unravel, f, states, z).mean())
    )(sets)
    # per T: sets[1 + 3i] = mean, sets[2 + 3i] = member
    delta_cos = jnp.stack(
        [
            jax.vmap(_cosine)(sets[2 + 3 * i] - cem, sets[1 + 3 * i] - cem)
            for i in range(len(STEPS))
        ]
    )
    return {
        "q_mean": q_mean,
        "grad_cos": jax.vmap(pairwise_cos)(cem),
        "delta_cos": delta_cos,
        "step_norm": jnp.linalg.norm(sets[1:] - cem, axis=-1),
    }


def _episode(
    probe: VariationProbe,
    key: jax.Array,
    embedding: SharedStateEmbedding,
    heads: ActorHead,
    horizon: int,
) -> np.ndarray:
    def policy(act_key: jax.Array, states: jnp.ndarray) -> jnp.ndarray:
        return heads_policy(embedding, heads, states)

    returns, _ = collect_parallel_episode(
        probe.vec_env, key, policy, None, horizon
    )
    return np.asarray(returns)


def _weights(k: int) -> jax.Array:
    # (2, N, K): mean critic, then one member per candidate (j mod K)
    uniform = jnp.full((CANDIDATES, k), 1.0 / k)
    one_hot = jax.nn.one_hot(jnp.arange(CANDIDATES) % k, k)
    return jnp.stack([uniform, one_hot])


def _offspring(
    probe: VariationProbe,
    post: Posterior,
    unravel: Callable[[jax.Array], ActorHead],
    cem: jax.Array,
    states: jax.Array,
    direction: jax.Array,
) -> list[jax.Array]:
    # mean, member, rand per T; the T-step sets reuse the first T batches
    mean_w, member_w = _weights(post.k)
    sets = []
    for t in STEPS:
        mean = _ascend(post, unravel, probe.optimizer, cem, mean_w, states[:t])
        member = _ascend(
            post, unravel, probe.optimizer, cem, member_w, states[:t]
        )
        radius = jnp.linalg.norm(member - mean, axis=1, keepdims=True)
        sets += [mean, member, mean + radius * direction]
    return sets


def record_variation(
    probe: VariationProbe,
    pop: Population,
    post: Posterior,
    buffer: Buffer,
    horizon: int,
    *,
    generation: int,
    env_steps: int,
) -> None:
    # paired design: candidate j of every set starts from the same CEM
    # sample x_j and plays both episodes from the same start states, so
    # differences between sets are differences in variation only
    keys = jax.random.split(jax.random.fold_in(probe.key, generation), 6)
    noise = jax.random.normal(keys[0], (CANDIDATES, pop.mean.shape[0]))
    cem = pop.mean + pop.scale * noise * jnp.sqrt(pop.cov)
    states = buffer.sample(keys[1], max(STEPS) * GRAD_BATCH)["state"]
    states = states.reshape(max(STEPS), GRAD_BATCH, -1)
    direction = jax.random.normal(keys[5], cem.shape)
    direction /= jnp.linalg.norm(direction, axis=1, keepdims=True)
    sets = jnp.stack(
        [cem, *_offspring(probe, post, pop.unravel, cem, states, direction)]
    )
    held_out = buffer.sample(keys[2], GRAD_BATCH)["state"]
    diag = _diagnostics(post, pop.unravel, sets, held_out)
    first, second = [], []
    for flat in sets:
        heads = jax.vmap(pop.unravel)(flat)
        first.append(_episode(probe, keys[3], post.embedding, heads, horizon))
        second.append(_episode(probe, keys[4], post.embedding, heads, horizon))
    probe.store.record(
        first=np.stack(first),
        second=np.stack(second),
        **{name: np.asarray(value) for name, value in diag.items()},
        cem_norm=float(jnp.linalg.norm(pop.scale * jnp.sqrt(pop.cov))),
        generation=generation,
        env_steps=env_steps,
    )
