from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from common.qd_archive import (
    Archive,
    Offspring,
    cell_index,
    empty_archive,
    insert,
)
from common.uqd_probe import Evaluator, UQDProbe
from environments.brax_qd import QDTask, evaluate, make_task
from modules.mlp_policy import Unravel, flat_spec, init_population


@dataclass(frozen=True)
class MAPElitesConfig:
    task: str
    seed: int
    iterations: int
    batch_size: int = 256
    grid_shape: tuple[int, ...] = (32, 32)
    hidden: tuple[int, ...] = (64, 64)
    iso_sigma: float = 0.005
    line_sigma: float = 0.05
    eval_chunk: int = 4096
    log_interval: int = 10
    probe_path: str | None = None
    probe_interval: int = 100
    probe_repeats: int = 16


def iso_line_dd(
    key: jax.Array, archive: Archive, n: int, iso: float, line: float
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # Vassiliades & Mouret 2018: x1 + iso*N(0, I) + line*N(0, 1)*(x2 - x1)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    occupied = jnp.isfinite(archive.fitness)
    p = occupied / occupied.sum()
    cells = archive.fitness.size
    first = jax.random.choice(k1, cells, (n,), p=p)
    second = jax.random.choice(k2, cells, (n,), p=p)
    x1, x2 = archive.genomes[first], archive.genomes[second]
    noise = iso * jax.random.normal(k3, x1.shape)
    pull = line * jax.random.normal(k4, (n, 1)) * (x2 - x1)
    return x1 + noise + pull, first, second


def _metrics(archive: Archive) -> dict[str, float]:
    fitness = np.asarray(archive.fitness)
    occupied = np.isfinite(fitness)
    return {
        "qd/coverage": float(occupied.mean()),
        "qd/max_fitness": float(fitness[occupied].max()),
        "qd/mean_fitness": float(fitness[occupied].mean()),
    }


def _evaluator(task: QDTask, unravel: Unravel, chunk: int) -> Evaluator:
    @jax.jit
    def run(genomes: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        return evaluate(task, unravel, genomes, key, chunk)

    return run


def train(
    cfg: MAPElitesConfig,
    on_iteration: Callable[[dict[str, Any]], None] = lambda _: None,
) -> Archive:
    task = make_task(cfg.task)
    obs_size = cast(int, task.env.observation_size)
    sizes = (obs_size, *cfg.hidden, task.env.action_size)
    genome_dim, unravel = flat_spec(sizes)
    cells = int(np.prod(cfg.grid_shape))
    run_eval = _evaluator(task, unravel, cfg.eval_chunk)

    def place(desc: jax.Array) -> jax.Array:
        return cell_index(task, cfg.grid_shape, desc)

    def make_batch(
        genomes: jax.Array, key: jax.Array, first: jax.Array, second: jax.Array
    ) -> Offspring:
        fit, desc = run_eval(genomes, key)
        return Offspring(genomes, fit, desc, place(desc), first, second)

    key = jax.random.key(cfg.seed)
    key, k_init, k_eval = jax.random.split(key, 3)
    none = jnp.full(cfg.batch_size, -1)
    insert_jit = jax.jit(insert)
    archive, _ = insert_jit(
        empty_archive(cells, genome_dim, task.desc_low.size),
        make_batch(
            init_population(k_init, sizes, cfg.batch_size), k_eval, none, none
        ),
    )
    probe = (
        None
        if cfg.probe_path is None
        else UQDProbe.create(cfg, run_eval, place, cells)
    )
    propose = jax.jit(iso_line_dd, static_argnums=(2,))
    for it in range(1, cfg.iterations + 1):
        key, k_var, k_eval = jax.random.split(key, 3)
        genomes, first, second = propose(
            k_var, archive, cfg.batch_size, cfg.iso_sigma, cfg.line_sigma
        )
        batch = make_batch(genomes, k_eval, first, second)
        before = archive
        archive, accepted = insert_jit(archive, batch)
        evaluations = (it + 1) * cfg.batch_size
        row: dict[str, Any] = {}
        if probe is not None:
            if it % cfg.probe_interval == 0:
                row = probe.record(
                    before, batch, accepted, it=it, evaluations=evaluations
                )
            probe.track(before, batch, accepted, it)
        if it % cfg.log_interval == 0 or row:
            on_iteration(
                _metrics(archive)
                | row
                | {
                    "evaluations": evaluations,
                    "env_steps": evaluations * task.episode_length,
                }
            )
    if probe is not None:
        probe.writer.save()
    return archive
