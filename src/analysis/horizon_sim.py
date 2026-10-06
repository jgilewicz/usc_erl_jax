from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Offline equal-budget comparison of evaluation-allocation policies on a
# horizon_probe.npz: every individual's full shadow episode is known, so any
# policy "rolls out h_i steps of individual i" can be replayed for free and
# scored against the full-episode truth.

BUDGETS = (0.1, 0.2, 0.3, 0.5)
H0S = (25, 50, 100)
DRAWS = 20

Selector = Callable[["Gen", np.ndarray, int, np.random.Generator], np.ndarray]


class Gen(NamedTuple):
    cum: np.ndarray  # (P, H) undiscounted prefix return after h=t+1 steps
    own_mu: np.ndarray  # (P, H) Q(s_{t+1}, pi_i(s_{t+1}))
    own_sigma: np.ndarray
    replay_mu: np.ndarray  # (P,)
    truth: np.ndarray  # (P,)
    gamma: float
    scale: float


def elite_mask(scores: np.ndarray, parents: int) -> np.ndarray:
    mask = np.zeros(scores.shape[0], dtype=bool)
    mask[np.argsort(-scores, kind="stable")[:parents]] = True
    return mask


def overlap(est: np.ndarray, truth: np.ndarray, parents: int) -> float:
    shared = elite_mask(est, parents) & elite_mask(truth, parents)
    return float(shared.sum() / parents)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def estimate(g: Gen, h: int, tail: str) -> np.ndarray:
    horizon = g.cum.shape[1]
    prefix = g.cum[:, h - 1]
    if h == horizon or tail == "none":
        return prefix
    if tail == "reward":
        return prefix + (horizon - h) * prefix / h
    # Q ~ r_bar / (1 - gamma) on a non-terminating task
    return prefix + (horizon - h) * (1.0 - g.gamma) * g.own_mu[:, h - 1]


def mix(est: np.ndarray, full: np.ndarray, extended: np.ndarray) -> np.ndarray:
    # shared offset measured on the extended ones, as SC-ERL calibrates
    if not extended.any():
        return est
    offset = np.mean(full[extended] - est[extended])
    return np.where(extended, full, est + offset)


def _cut_distance(est: np.ndarray, parents: int) -> np.ndarray:
    ordered = np.sort(est)[::-1]
    cut = 0.5 * (ordered[parents - 1] + ordered[parents])
    return np.abs(est - cut)


def _cv(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return sigma / (np.sqrt(np.abs(mu)) + 1.0)


def _top(score: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    # random tie-break so constant scores degrade to random, not to slot 0
    order = np.lexsort((rng.random(score.shape[0]), -score))
    mask = np.zeros(score.shape[0], dtype=bool)
    mask[order[:k]] = True
    return mask


def _selectors(h0: int, parents: int) -> dict[str, Selector]:
    def misranked(g: Gen, est: np.ndarray) -> np.ndarray:
        return elite_mask(est, parents) != elite_mask(g.truth, parents)

    return {
        "random": lambda g, est, k, rng: _top(np.zeros(len(est)), k, rng),
        "cut": lambda g, est, k, rng: _top(
            -_cut_distance(est, parents), k, rng
        ),
        "sigma_prefix": lambda g, est, k, rng: _top(
            g.own_sigma[:, :h0].mean(axis=1), k, rng
        ),
        "cv_prefix": lambda g, est, k, rng: _top(
            _cv(g.own_mu[:, :h0], g.own_sigma[:, :h0]).mean(axis=1), k, rng
        ),
        "sigma_at_h0": lambda g, est, k, rng: _top(
            g.own_sigma[:, h0 - 1], k, rng
        ),
        "oracle": lambda g, est, k, rng: _top(
            misranked(g, est).astype(float), k, rng
        ),
    }


def _replay_full(
    g: Gen, b: float, parents: int, rng: np.random.Generator
) -> float:
    # what SC-ERL does now: k random full episodes, the rest replay critic
    k = round(b * len(g.truth))
    real = _top(np.zeros(len(g.truth)), k, rng)
    est = mix(g.scale * g.replay_mu, g.truth, real)
    return overlap(est, g.truth, parents)


def _two_stage(
    g: Gen,
    b: float,
    h0: int,
    pick: Selector,
    tail: str,
    rng: np.random.Generator,
) -> float:
    n, horizon = g.cum.shape
    parents = n // 2
    k = int((b * n * horizon - n * h0) // (horizon - h0))
    if k < 0:
        return float("nan")
    est = estimate(g, h0, tail)
    extended = pick(g, est, min(k, n), rng)
    return overlap(mix(est, g.truth, extended), g.truth, parents)


def load(path: Path) -> list[Gen]:
    d = np.load(path)
    gamma = float(d["gamma"])
    alive = np.cumsum(d["done"], axis=2) - d["done"] == 0
    cum = np.cumsum(d["rewards"] * alive, axis=2)
    gap = np.abs(cum[..., -1] - d["truth"]).max()
    if gap > 1e-2 * (1 + np.abs(d["truth"]).max()):
        raise ValueError(
            f"{path}: summed probe rewards differ from truth by {gap:.3g}; "
            "the probe and the shadow rollout are out of sync"
        )
    return [
        Gen(
            cum[i],
            d["own_mu"][i],
            d["own_sigma"][i],
            d["replay_mu"][i],
            d["truth"][i],
            gamma,
            float(d["undiscount_scale"][i]),
        )
        for i in range(cum.shape[0])
    ]


def _mean(fn: Callable[[np.random.Generator], float], seed: int) -> float:
    rng = np.random.default_rng(seed)
    draws = np.array([fn(rng) for _ in range(DRAWS)])
    # nan = budget below this policy's h0 floor
    return float("nan") if np.isnan(draws).all() else float(np.nanmean(draws))


def policy_table(gens: list[Gen]) -> dict[str, list[float]]:
    parents = len(gens[0].truth) // 2
    horizon = gens[0].cum.shape[1]
    rows: dict[str, list[float]] = {}
    rows["replay_critic+random_full"] = [
        np.mean(
            [
                _mean(lambda r: _replay_full(g, b, parents, r), i)
                for i, g in enumerate(gens)
            ]
        )
        for b in BUDGETS
    ]
    for tail in ("critic", "reward", "none"):
        rows[f"uniform_h[{tail}]"] = [
            np.mean(
                [
                    overlap(
                        estimate(g, max(1, int(b * horizon)), tail),
                        g.truth,
                        parents,
                    )
                    for g in gens
                ]
            )
            for b in BUDGETS
        ]
    for h0 in H0S:
        for name, pick in _selectors(h0, parents).items():
            for tail in ("critic", "reward"):
                rows[f"h0={h0} {name}[{tail}]"] = [
                    np.mean(
                        [
                            _mean(
                                lambda r: _two_stage(g, b, h0, pick, tail, r), i
                            )
                            for i, g in enumerate(gens)
                        ]
                    )
                    for b in BUDGETS
                ]
    return rows


def auc_table(gens: list[Gen]) -> dict[str, float]:
    # does the signal flag individuals the h0-estimate puts on the wrong
    # side of the elite cut?
    parents = len(gens[0].truth) // 2
    out: dict[str, float] = {}
    for h0 in H0S:
        for tail in ("critic", "reward"):
            scores: dict[str, list[float]] = {}
            for g in gens:
                est = estimate(g, h0, tail)
                label = elite_mask(est, parents) != elite_mask(g.truth, parents)
                signals = {
                    "sigma_prefix": g.own_sigma[:, :h0].mean(axis=1),
                    "cv_prefix": _cv(
                        g.own_mu[:, :h0], g.own_sigma[:, :h0]
                    ).mean(axis=1),
                    "sigma_at_h0": g.own_sigma[:, h0 - 1],
                    "-cut_distance": -_cut_distance(est, parents),
                }
                for name, s in signals.items():
                    scores.setdefault(name, []).append(auc(s, label))
            for name, v in scores.items():
                out[f"h0={h0} [{tail}] {name}"] = float(np.nanmean(v))
    return out


def _print(gens: list[Gen], title: str) -> None:
    print(f"\n## {title}: {len(gens)} gens, elite_overlap (chance 0.5)")
    print(f"{'policy':40}" + "".join(f"  b={b:<5}" for b in BUDGETS))
    for name, vals in policy_table(gens).items():
        print(f"{name:40}" + "".join(f"  {v:7.3f}" for v in vals))
    print(f"\n## {title}: AUC signal -> misranked under h0 estimate")
    for name, v in auc_table(gens).items():
        print(f"{name:40}  {v:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path, nargs="+")
    parser.add_argument("--thirds", action="store_true")
    args = parser.parse_args()
    for path in args.probe:
        gens = load(path)
        _print(gens, f"{path} all")
        if args.thirds:
            n = len(gens)
            for i, part in enumerate(
                (gens[: n // 3], gens[n // 3 : 2 * n // 3], gens[2 * n // 3 :])
            ):
                _print(part, f"{path} third {i + 1}")


if __name__ == "__main__":
    main()
