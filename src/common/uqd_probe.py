from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from common.horizon_probe import HorizonProbe
from common.qd_archive import Archive, Offspring

if TYPE_CHECKING:
    from algos.map_elites import MAPElitesConfig

# metric-only: repeated evaluations and lineage recorded here must never
# flow back into MAP-Elites insertion or variation (shadow rule)

_KEY_OFFSET = 7_919
Evaluator = Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]
Placer = Callable[[jax.Array], jax.Array]


class Lineage(NamedTuple):
    born: jax.Array
    parent: jax.Array
    parent_fitness: jax.Array
    # challenges the current elite has survived since it was inserted
    survived: jax.Array


@jax.jit
def _track(
    lineage: Lineage,
    fitness_before: jax.Array,
    batch: Offspring,
    accepted: jax.Array,
    it: jax.Array,
) -> Lineage:
    num = lineage.born.size
    cells = batch.cells
    occupied = jnp.isfinite(fitness_before)
    replaced = (
        jax.ops.segment_max(accepted.astype(jnp.int32), cells, num_segments=num)
        > 0
    )
    lost = (occupied[cells] & ~accepted).astype(jnp.int32)
    challengers = jax.ops.segment_sum(lost, cells, num_segments=num)
    target = jnp.where(accepted, cells, num)
    return Lineage(
        born=lineage.born.at[target].set(it, mode="drop"),
        parent=lineage.parent.at[target].set(batch.first, mode="drop"),
        parent_fitness=lineage.parent_fitness.at[target].set(
            fitness_before[batch.first], mode="drop"
        ),
        survived=jnp.where(replaced, 0, lineage.survived + challengers),
    )


def batch_winners(fit: np.ndarray, cells: np.ndarray) -> np.ndarray:
    # mirrors insert(): best per cell, ties to the lowest index
    order = np.lexsort((np.arange(fit.size), -fit, cells))
    head = np.ones(fit.size, dtype=bool)
    head[1:] = cells[order][1:] != cells[order][:-1]
    winners = np.zeros(fit.size, dtype=bool)
    winners[order[head]] = True
    return winners


@dataclass
class UQDProbe:
    writer: HorizonProbe
    key: jax.Array
    evaluate: Evaluator
    place: Placer
    repeats: int
    lineage: Lineage

    @classmethod
    def create(
        cls,
        cfg: MAPElitesConfig,
        evaluate: Evaluator,
        place: Placer,
        cells: int,
    ) -> UQDProbe:
        assert cfg.probe_path is not None
        minus = jnp.full(cells, -1, dtype=jnp.int32)
        return cls(
            # gamma is meaningless for QD; HorizonProbe stores it regardless
            writer=HorizonProbe(Path(cfg.probe_path), gamma=1.0, save_every=1),
            key=jax.random.key(cfg.seed + _KEY_OFFSET),
            evaluate=evaluate,
            place=place,
            repeats=cfg.probe_repeats,
            lineage=Lineage(
                born=minus,
                parent=minus,
                parent_fitness=jnp.full(cells, -jnp.inf),
                survived=jnp.zeros(cells, dtype=jnp.int32),
            ),
        )

    def track(
        self, before: Archive, batch: Offspring, accepted: jax.Array, it: int
    ) -> None:
        self.lineage = _track(
            self.lineage, before.fitness, batch, accepted, jnp.int32(it)
        )

    def _repeat(self, genomes: jax.Array) -> tuple[np.ndarray, np.ndarray]:
        self.key, sub = jax.random.split(self.key)
        n = genomes.shape[0]
        fit, desc = self.evaluate(jnp.repeat(genomes, self.repeats, 0), sub)
        return (
            np.asarray(fit).reshape(n, self.repeats),
            np.asarray(desc).reshape(n, self.repeats, -1),
        )

    def record(
        self,
        before: Archive,
        batch: Offspring,
        accepted: jax.Array,
        *,
        it: int,
        evaluations: int,
    ) -> dict[str, float]:
        elite_f, elite_d = self._repeat(before.genomes)
        off_f, off_d = self._repeat(batch.genomes)
        stored = np.asarray(before.fitness)
        cells = np.asarray(batch.cells)
        winners = batch_winners(np.asarray(batch.fitness), cells)
        # placement under the descriptor expectation, not the lucky draw
        own = np.asarray(self.place(jnp.asarray(elite_d.mean(axis=1))))
        placed = own == np.arange(stored.size)
        lineage = {
            f"elite_{name}": np.asarray(value)
            for name, value in self.lineage._asdict().items()
        }
        self.writer.record(
            generation=np.asarray(it),
            evaluations=np.asarray(evaluations),
            elite_fitness=stored,
            elite_desc=np.asarray(before.descriptors),
            elite_reps_f=elite_f,
            elite_reps_d=elite_d,
            elite_placed=placed,
            off_fitness=np.asarray(batch.fitness),
            off_desc=np.asarray(batch.descriptors),
            off_cell=cells,
            off_parent1=np.asarray(batch.first),
            off_parent2=np.asarray(batch.second),
            off_reps_f=off_f,
            off_reps_d=off_d,
            off_accepted=np.asarray(accepted),
            off_winner=winners,
            **lineage,
        )
        return _online_metrics(
            stored=stored,
            elite_f=elite_f,
            placed=placed,
            off_f=off_f,
            cells=cells,
            contested=winners & np.isfinite(stored)[cells],
            accepted=np.asarray(accepted),
        )


def _online_metrics(
    *,
    stored: np.ndarray,
    elite_f: np.ndarray,
    placed: np.ndarray,
    off_f: np.ndarray,
    cells: np.ndarray,
    contested: np.ndarray,
    accepted: np.ndarray,
) -> dict[str, float]:
    occupied = np.isfinite(stored)
    true_elite = elite_f.mean(axis=1)
    truth = off_f.mean(axis=1)[contested] > true_elite[cells[contested]]
    decision = (
        float((accepted[contested] == truth).mean())
        if contested.any()
        else float("nan")
    )
    return {
        "uqd/lucky_gap": float((stored - true_elite)[occupied].mean()),
        "uqd/placement_acc": float(placed[occupied].mean()),
        "uqd/decision_acc": decision,
    }
