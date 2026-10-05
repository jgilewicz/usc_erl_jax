from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.stats import rankdata


def elite_mask(scores: jax.Array, parents: int) -> jax.Array:
    # the membership CEM consumes: argsort(-scores)[:parents]
    top = jnp.argsort(-scores)[:parents]
    return jnp.zeros(scores.shape[0], dtype=bool).at[top].set(True)


def elite_overlap(a: jax.Array, b: jax.Array, parents: int) -> jax.Array:
    shared = elite_mask(a, parents) & elite_mask(b, parents)
    return jnp.sum(shared) / parents


def spearman(a: jax.Array, b: jax.Array) -> jax.Array:
    # average ranks: ties must not get an arbitrary order
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = jnp.sqrt(jnp.sum(ra**2) * jnp.sum(rb**2))
    return jnp.where(denom > 0, jnp.sum(ra * rb) / denom, jnp.nan)


def auc(score: jax.Array, label: jax.Array) -> jax.Array:
    # P(score_pos > score_neg), ties count half; NaN without both classes
    pos, neg = label, ~label
    greater = (score[:, None] > score[None, :]).astype(jnp.float32)
    ties = (score[:, None] == score[None, :]).astype(jnp.float32)
    pairs = pos[:, None] & neg[None, :]
    n_pairs = jnp.sum(pairs)
    wins = jnp.sum((greater + 0.5 * ties) * pairs)
    return jnp.where(n_pairs > 0, wins / jnp.maximum(n_pairs, 1), jnp.nan)


def _ratio(num: jax.Array, den: jax.Array) -> jax.Array:
    return jnp.where(den > 0, num / jnp.maximum(den, 1), jnp.nan)


@partial(jax.jit, static_argnames=("parents",))
def selection_metrics(
    truth: jax.Array,
    used: jax.Array,
    surrogate: jax.Array,
    parents: int,
) -> dict[str, jax.Array]:
    # used: the fitness CEM was told; surrogate: the surrogate scored on
    # every individual (counterfactual for the ones that rolled out)
    return {
        "select/elite_overlap": elite_overlap(used, truth, parents),
        "select/elite_overlap_surr": elite_overlap(surrogate, truth, parents),
        "select/rank_corr": spearman(used, truth),
        "select/rank_corr_surr": spearman(surrogate, truth),
        "select/regret": jnp.max(truth) - truth[jnp.argmax(used)],
    }


@partial(jax.jit, static_argnames=("parents",))
def gate_metrics(
    truth: jax.Array,
    predicted: jax.Array,
    real: jax.Array,
    parents: int,
) -> dict[str, jax.Array]:
    # misranked = the critic's predicted value mu (no sigma: the label must
    # not contain the signal it scores) puts it on the wrong side of the
    # elite cut; the gate's job is to send exactly those to a real rollout
    misranked = elite_mask(predicted, parents) != elite_mask(truth, parents)
    hits = jnp.sum(real & misranked)
    return {
        "gate/misranked_frac": jnp.mean(misranked),
        "gate/precision": _ratio(hits, jnp.sum(real)),
        "gate/recall": _ratio(hits, jnp.sum(misranked)),
    }


@partial(jax.jit, static_argnames=("parents",))
def uncertainty_gate_metrics(
    truth: jax.Array,
    predicted: jax.Array,
    cv: jax.Array,
    real: jax.Array,
    deterministic: jax.Array,
    threshold: jax.Array,
    parents: int,
) -> dict[str, jax.Array]:
    misranked = elite_mask(predicted, parents) != elite_mask(truth, parents)
    return {
        "gate/auc": auc(cv, misranked),
        "gate/threshold": threshold,
        "gate/cv_mean": jnp.mean(cv),
        "gate/cv_max": jnp.max(cv),
        "gate/eps_frac": jnp.mean(real & ~deterministic),
    }
