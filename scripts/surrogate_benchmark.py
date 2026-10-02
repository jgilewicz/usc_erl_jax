"""Offline benchmark of population surrogates on SEMARL population dumps.

Reads the .npz written by `algorithm.dump_path` (one per seed) and replays
selection without any RL: every candidate surrogate is trained on the
generations before t and scored on generation t's siblings. That is the
discrimination selection needs and the one PeVFA failed.

Part 1 - can a model rank siblings at all?  elite_overlap vs the true return
(chance 0.5). `critic` and `hstep` come straight from the dump: the batch-
averaged critic that selects on surrogate generations, and the h-step
bootstrap that costs H env steps per individual. The learned models use the
(policy, true return) pairs the run already paid for:
  ridge          MSE on raw returns - collapses to the mean like PeVFA did
  ridge-within   features and returns centred per generation, so only the
                 differences between siblings are fitted (= pairwise least
                 squares / RankRLS)
  rank           pairwise logistic loss on within-generation pairs
                 (Ranking-SVM style, cf. s*ACM-ES)
  gp             GP regression on within-generation-centred data; the only
                 one with a predictive std (cf. DTS-CMA-ES)
each on `w` (raw head weights) and `fp` (actions on fixed probe states).

Part 2 - per-individual evaluation, on the critic (the learned models sit
at chance, so their selection numbers say nothing). Spend k real rollouts
on chosen individuals, keep the critic for the rest (offset-calibrated on
the k true values), select top-`parents`. `oracle` is the best k-subset in
hindsight - the ceiling any selection criterion can reach. Compared at equal cost against SEMARL's
per-generation coin: k/pop * 1.0 + (1 - k/pop) * elite_overlap(surrogate).
`unc - boundary` is what uncertainty adds over picking the individuals the
surrogate ranks closest to the elite cut - the number a paper about
uncertainty has to show is positive.

    uv run python scripts/surrogate_benchmark.py outputs/*/population.npz
"""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

Prediction = tuple[np.ndarray, np.ndarray | None]
Model = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Prediction]

RIDGE_ALPHA = 1.0
GP_NOISE = 0.1
RANK_STEPS = 300
RANK_LR = 0.1
RANK_L2 = 1e-3
RANDOM_DRAWS = 20
MIN_REL_SD = 1e-3


def _elite(scores: np.ndarray, parents: int) -> set[int]:
    return set(np.argsort(-scores)[:parents].tolist())


def _elite_overlap(true: np.ndarray, est: np.ndarray, parents: int) -> float:
    return len(_elite(true, parents) & _elite(est, parents)) / parents


def _rank_corr(true: np.ndarray, est: np.ndarray) -> float:
    rt, re = np.argsort(np.argsort(true)), np.argsort(np.argsort(est))
    return float(np.corrcoef(rt, re)[0, 1])


def _center_groups(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    xc, yc = x.copy(), y.copy()
    for g in np.unique(groups):
        m = groups == g
        xc[m] -= xc[m].mean(0)
        yc[m] -= yc[m].mean()
    return xc, yc


def _standardize(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = train.mean(0), train.std(0)
    # a feature (near-)constant over the window cannot be learned, and a
    # tiny sd turns any test deviation into a z-score of ~1e3, which zeroes
    # the GP kernel and dominates ridge (seen on saturated tanh actions)
    keep = sd > MIN_REL_SD * np.median(sd)
    return (
        (train[:, keep] - mu[keep]) / sd[keep],
        (test[:, keep] - mu[keep]) / sd[keep],
    )


def _ridge_dual(x: np.ndarray, y: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    # dual form: d (up to 1542) exceeds n (window * pop) here
    gram = x @ x.T + RIDGE_ALPHA * np.eye(len(x))
    return x_test @ x.T @ np.linalg.solve(gram, y - y.mean()) + y.mean()


def ridge(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, x_test: np.ndarray
) -> Prediction:
    xs, xt = _standardize(x, x_test)
    return _ridge_dual(xs, y, xt), None


def ridge_within(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, x_test: np.ndarray
) -> Prediction:
    xc, yc = _center_groups(x, y, groups)
    xs, xt = _standardize(xc, x_test - x_test.mean(0))
    return _ridge_dual(xs, yc, xt), None


def rank(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, x_test: np.ndarray
) -> Prediction:
    xc, _ = _center_groups(x, y, groups)
    xs, xt = _standardize(xc, x_test - x_test.mean(0))
    diffs = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        for a in idx:
            for b in idx:
                if y[a] > y[b]:
                    diffs.append(xs[a] - xs[b])
    d = np.asarray(diffs)
    w = np.zeros(d.shape[1])
    for _ in range(RANK_STEPS):
        # logistic loss on "winner - loser" margins, all labels +1
        p = 1.0 / (1.0 + np.exp(np.clip(d @ w, -30, 30)))
        w -= RANK_LR * (-(d * p[:, None]).mean(0) + RANK_L2 * w)
    return xt @ w, None


def gp(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, x_test: np.ndarray
) -> Prediction:
    xc, yc = _center_groups(x, y, groups)
    xs, xt = _standardize(xc, x_test - x_test.mean(0))
    scale = yc.std() + 1e-8
    sq = ((xs[:, None] - xs[None]) ** 2).sum(-1)
    # median heuristic: no hyperparameter search on a window this small
    length2 = np.median(sq[sq > 0]) if (sq > 0).any() else 1.0
    k = np.exp(-sq / length2) + GP_NOISE * np.eye(len(xs))
    k_star = np.exp(-((xt[:, None] - xs[None]) ** 2).sum(-1) / length2)
    alpha = np.linalg.solve(k, yc / scale)
    v = np.linalg.solve(k, k_star.T)
    var = np.clip(1.0 - (k_star * v.T).sum(1), 1e-12, None)
    return k_star @ alpha * scale, np.sqrt(var) * scale


MODELS: dict[str, Model] = {
    "ridge": ridge,
    "ridge-within": ridge_within,
    "rank": rank,
    "gp": gp,
}


def _features(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n_gen, pop = d["real"].shape
    return {
        "w": d["flat_pop"].astype(np.float64),
        "fp": d["fingerprint"].reshape(n_gen, pop, -1).astype(np.float64),
    }


def _calibrated_mix(
    true: np.ndarray, est: np.ndarray, chosen: np.ndarray
) -> np.ndarray:
    # shift the surrogate onto the true scale by its mean error on the k
    # paid-for pairs. Measured on 3 seeds x 80 generations: an affine fit on
    # those k pairs (0.782 at k=5, boundary) and one on the previous 5
    # generations' pairs (0.752) both lose to this (0.818) - k points give a
    # wild slope, and the critic's scale drifts between generations.
    mixed = est.astype(np.float64) + (true[chosen] - est[chosen]).mean()
    mixed[chosen] = true[chosen]
    return mixed


def _boundary(est: np.ndarray, k: int, parents: int) -> np.ndarray:
    ranks = np.argsort(np.argsort(-est))
    return np.argsort(np.abs(ranks - (parents - 0.5)))[:k]


def _selection(
    true: np.ndarray,
    est: np.ndarray,
    std: np.ndarray | None,
    ks: list[int],
    rng: np.random.Generator,
) -> dict[str, float]:
    pop, parents = len(true), len(true) // 2
    base = _elite_overlap(true, est, parents)
    out = {}
    for k in ks:
        out[f"semarl k={k}"] = k / pop + (1 - k / pop) * base
        draws = [rng.choice(pop, k, replace=False) for _ in range(RANDOM_DRAWS)]
        out[f"random k={k}"] = float(
            np.mean(
                [
                    _elite_overlap(true, _calibrated_mix(true, est, c), parents)
                    for c in draws
                ]
            )
        )
        out[f"boundary k={k}"] = _elite_overlap(
            true,
            _calibrated_mix(true, est, _boundary(est, k, parents)),
            parents,
        )
        if std is not None:
            unc = np.argsort(-std)[:k]
            out[f"unc k={k}"] = _elite_overlap(
                true, _calibrated_mix(true, est, unc), parents
            )
        # best subset in hindsight: the ceiling for any selection criterion
        out[f"oracle k={k}"] = max(
            _elite_overlap(
                true, _calibrated_mix(true, est, np.array(c)), parents
            )
            for c in itertools.combinations(range(pop), k)
        )
    return out


def _test_generation(
    d: dict[str, np.ndarray],
    feats: dict[str, np.ndarray],
    t: int,
    window: int,
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    lo = max(0, t - window)
    true_train = d["real"][lo:t]
    groups = np.repeat(np.arange(lo, t), true_train.shape[1])
    preds: dict[str, tuple[np.ndarray, np.ndarray | None]] = {
        "critic": (d["critic"][t], None),
        "hstep": (d["hstep"][t], None),
    }
    for fname, f in feats.items():
        x = f[lo:t].reshape(-1, f.shape[-1])
        for mname, model in MODELS.items():
            preds[f"{mname}/{fname}"] = model(
                x, true_train.ravel(), groups, f[t]
            )
    return preds


def _benchmark(
    path: Path, window: int, min_history: int, ks: list[int]
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    d = dict(np.load(path))
    feats = _features(d)
    n_gen, pop = d["real"].shape
    if n_gen <= min_history:
        raise ValueError(
            f"{path} holds {n_gen} generations, need > --min-history "
            f"({min_history}); lower it or dump a longer run"
        )
    if max(ks) >= pop:
        raise ValueError(
            f"--k {max(ks)} must be below pop_size ({pop}) in {path}: "
            "k = pop is a fully real generation"
        )
    rng = np.random.default_rng(0)
    rank_scores: dict[str, list[float]] = defaultdict(list)
    sel_scores: dict[str, list[float]] = defaultdict(list)
    for t in range(min_history, n_gen):
        true = d["real"][t]
        for name, (est, std) in _test_generation(d, feats, t, window).items():
            rank_scores[f"{name} ovl"].append(
                _elite_overlap(true, est, pop // 2)
            )
            rank_scores[f"{name} rc"].append(_rank_corr(true, est))
            if name == "critic":
                for crit, v in _selection(true, est, std, ks, rng).items():
                    sel_scores[f"{name} | {crit}"].append(v)
    return rank_scores, sel_scores


def _print_ranking(per_seed: list[dict[str, list[float]]]) -> None:
    names = sorted({k.rsplit(" ", 1)[0] for s in per_seed for k in s})
    print(
        f"\n{'surrogate':<20}{'elite_ovl':>11}{'± seeds':>9}{'rank_corr':>11}"
    )
    for name in names:
        ovl = [np.mean(s[f"{name} ovl"]) for s in per_seed]
        rc = [np.mean(s[f"{name} rc"]) for s in per_seed]
        print(
            f"{name:<20}{np.mean(ovl):>11.3f}{np.std(ovl):>9.3f}"
            f"{np.mean(rc):>+11.3f}"
        )


def _print_selection(
    per_seed: list[dict[str, list[float]]], ks: list[int]
) -> None:
    print("\nPer-individual evaluation, elite_ovl at equal cost (k of pop)")
    crits = ["semarl", "random", "boundary", "unc", "oracle"]
    print(
        f"{'surrogate':<10}{'k':>3}"
        + "".join(f"{c:>10}" for c in crits)
        + f"{'unc-bnd':>10}"
    )
    for sur in ("critic",):
        for k in ks:
            cells, vals = "", {}
            for c in crits:
                key = f"{sur} | {c} k={k}"
                if any(key in s for s in per_seed):
                    vals[c] = float(
                        np.mean([np.mean(s[key]) for s in per_seed])
                    )
                    cells += f"{vals[c]:>10.3f}"
                else:
                    cells += f"{'-':>10}"
            gain = (
                f"{vals['unc'] - vals['boundary']:>+10.3f}"
                if "unc" in vals
                else f"{'-':>10}"
            )
            print(f"{sur:<10}{k:>3}{cells}{gain}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dumps", nargs="+", type=Path, help="population .npz")
    ap.add_argument("--window", type=int, default=20, help="training gens")
    ap.add_argument("--min-history", type=int, default=10)
    ap.add_argument("--k", type=int, nargs="+", default=[2, 3, 5])
    args = ap.parse_args()

    rank_runs, sel_runs = [], []
    for path in args.dumps:
        r, s = _benchmark(path, args.window, args.min_history, args.k)
        print(f"{path}: {len(r['critic ovl'])} test generations")
        rank_runs.append(r)
        sel_runs.append(s)
    _print_ranking(rank_runs)
    _print_selection(sel_runs, args.k)


if __name__ == "__main__":
    main()
