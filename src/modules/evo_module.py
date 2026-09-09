from functools import partial

import jax.numpy as jnp
import jax


@partial(jax.jit, static_argnames=("pop_size",))
def _cem_ask(
    key: jax.Array,
    mu: jax.Array,
    cov: jax.Array,
    scale: float,
    pop_size: int,
) -> jax.Array:
    epsilon = jax.random.normal(key, (pop_size, mu.shape[0]))
    return mu + scale * epsilon * jnp.sqrt(cov)


@partial(jax.jit, static_argnames=("parents", "num_params"))
def _cem_tell(
    scores: jax.Array,
    solutions: jax.Array,
    mu: jax.Array,
    weights: jax.Array,
    parents: int,
    damp: float,
    damp_limit: float,
    tau: float,
    num_params: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    neg_scores = -scores
    idx_sorted = jnp.argsort(neg_scores)[:parents]
    parent_solutions = solutions[idx_sorted]

    new_damp = jnp.asarray(damp * tau + (1 - tau) * damp_limit)
    new_mu = jnp.dot(weights, parent_solutions)
    z = parent_solutions - mu
    new_cov = 1 / parents * jnp.dot(weights, z * z) + new_damp * jnp.ones(
        num_params
    )
    return new_mu, new_cov, new_damp, idx_sorted[0]


class CEM:
    def __init__(
        self,
        num_params: int,
        key: jax.Array,
        scale: float = 1.0,
        sigma_init: float = 1e-3,
        pop_size: int = 256,
        damp: float = 1e-3,
        damp_limit: float = 1e-5,
        tau: float = 0.95,
        parents: int | None = None,
        elitisim: bool = False,
    ) -> None:
        self.num_params = num_params
        self.scale = scale
        self.sigma_init = sigma_init
        self.pop_size = pop_size
        self.damp = damp
        self.damp_limit = damp_limit
        self.parents = parents if parents else pop_size // 2
        self.elitisim = elitisim
        self.tau = tau

        self.mu = jnp.zeros(num_params)
        self.cov = jnp.ones(num_params) * sigma_init

        self.elite = jnp.sqrt(sigma_init) * jax.random.normal(
            key, (num_params,)
        )
        self.elite_score = -jnp.inf

        weights = jnp.log((self.parents + 1) / jnp.arange(1, self.parents + 1))

        self.weights = weights / weights.sum()

    def ask(self, key: jax.Array, pop_size: int | None = None) -> jax.Array:
        if pop_size is None:
            pop_size = self.pop_size

        inds = _cem_ask(key, self.mu, self.cov, self.scale, pop_size)

        if self.elitisim:
            inds = inds.at[-1].set(self.elite)

        return inds

    def tell(
        self, scores: jax.Array, solutions: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        self.mu, self.cov, self.damp, elite_idx = _cem_tell(
            scores,
            solutions,
            self.mu,
            self.weights,
            self.parents,
            self.damp,
            self.damp_limit,
            self.tau,
            self.num_params,
        )
        self.elite = solutions[elite_idx]
        self.elite_score = -scores[elite_idx]

        return (self.mu, self.cov)
