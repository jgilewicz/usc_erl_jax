from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from environments.brax_qd import QDTask


class Archive(NamedTuple):
    genomes: jax.Array
    fitness: jax.Array
    descriptors: jax.Array


class Offspring(NamedTuple):
    genomes: jax.Array
    fitness: jax.Array
    descriptors: jax.Array
    cells: jax.Array
    first: jax.Array
    second: jax.Array


def empty_archive(cells: int, genome_dim: int, desc_dim: int) -> Archive:
    return Archive(
        genomes=jnp.zeros((cells, genome_dim)),
        fitness=jnp.full(cells, -jnp.inf),
        descriptors=jnp.zeros((cells, desc_dim)),
    )


def cell_index(
    task: QDTask, grid_shape: tuple[int, ...], desc: jax.Array
) -> jax.Array:
    unit = (desc - task.desc_low) / (task.desc_high - task.desc_low)
    shape = jnp.array(grid_shape)
    bins = jnp.clip(jnp.floor(unit * shape).astype(jnp.int32), 0, shape - 1)
    return jnp.ravel_multi_index(tuple(bins.T), grid_shape, mode="clip")


def insert(archive: Archive, batch: Offspring) -> tuple[Archive, jax.Array]:
    n = batch.fitness.size
    num = archive.fitness.size
    cells = batch.cells
    best = jax.ops.segment_max(batch.fitness, cells, num_segments=num)
    idx = jnp.arange(n)
    # ties within a batch go to the lowest index
    winner = jax.ops.segment_min(
        jnp.where(batch.fitness == best[cells], idx, n),
        cells,
        num_segments=num,
    )
    accepted = (idx == winner[cells]) & (batch.fitness > archive.fitness[cells])
    target = jnp.where(accepted, cells, num)
    return (
        Archive(
            genomes=archive.genomes.at[target].set(batch.genomes, mode="drop"),
            fitness=archive.fitness.at[target].set(batch.fitness, mode="drop"),
            descriptors=archive.descriptors.at[target].set(
                batch.descriptors, mode="drop"
            ),
        ),
        accepted,
    )
