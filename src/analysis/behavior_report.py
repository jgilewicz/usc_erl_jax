from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# pre-registered in experiments/behavior-probe-v1/experiment.toml; change
# there first. Rows are post-warmup generations in recording order.
ARCHIVE = 20
FIRST = 40
SELECT_END = 80
PARENTS = 5
NEIGHBOURS = 10
# within a generation k = N - 1 predicts ~ -z_i / (N - 1): anti-correlated
LOO_NEIGHBOURS = 3
MIN_GAIN = 0.03
# one-sided 97.5 %
CI = (2.5, 97.5)
BLOCK = 10
RESAMPLES = 10_000
NOISE_GRID = np.logspace(-3, 1, 25)
EPS = 1e-8

Model = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Probe:
    actions: np.ndarray  # (G, N, S, A) pi_i on the fixed state set
    mean_actions: np.ndarray  # (G, S, A) CEM mean head as asked
    heads: np.ndarray  # (G, N, D) flat ActorHead params
    truth: np.ndarray  # (G, N) shadow return, every individual
    mu: np.ndarray  # (G, N) critic score the gate saw


def load(path: Path) -> Probe:
    if not path.is_file():
        raise FileNotFoundError(
            f"no behaviour probe at {path}: run sc_erl with "
            "algorithm.behavior_probe=true"
        )
    d = np.load(path)
    probe = Probe(**{name: d[name] for name in Probe.__dataclass_fields__})
    n_gen, pop = probe.truth.shape
    if pop < 2 * PARENTS or n_gen < SELECT_END + BLOCK:
        raise ValueError(
            f"{path}: {n_gen} generations x {pop} individuals; the "
            f"pre-registration needs >= {SELECT_END + BLOCK} generations "
            f"and >= {2 * PARENTS} individuals (top {PARENTS} elite)"
        )
    return probe


def zscore(x: np.ndarray) -> np.ndarray:
    # within generation: CEM only uses the ranking, critic scale drifts
    centred = x - x.mean(axis=-1, keepdims=True)
    return centred / (x.std(axis=-1, keepdims=True) + EPS)


def elite_overlap(score: np.ndarray, truth: np.ndarray) -> float:
    picked = np.argsort(-score, kind="stable")[:PARENTS]
    best = np.argsort(-truth, kind="stable")[:PARENTS]
    return len(np.intersect1d(picked, best)) / PARENTS


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def _sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a**2).sum(1)[:, None] + (b**2).sum(1)[None] - 2 * a @ b.T
    return np.maximum(d, 0.0)


def standardise(
    archive: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = archive.mean(axis=0)
    std = archive.std(axis=0)
    std = np.where(std > EPS, std, 1.0)
    return (archive - mean) / std, (query - mean) / std


def knn(
    xa: np.ndarray, ya: np.ndarray, xq: np.ndarray, k: int = NEIGHBOURS
) -> np.ndarray:
    dist = np.sqrt(_sq_dist(xq, xa))
    idx = np.argsort(dist, axis=1)[:, :k]
    near = np.take_along_axis(dist, idx, axis=1)
    weights = 1.0 / (near + EPS)
    return (weights * ya[idx]).sum(1) / weights.sum(1)


def gp(xa: np.ndarray, ya: np.ndarray, xq: np.ndarray) -> np.ndarray:
    # RBF, median-heuristic length scale, signal variance = var(y), noise
    # ratio by marginal likelihood on the archive
    d2 = _sq_dist(xa, xa)
    ell2 = np.median(d2[np.triu_indices(len(xa), k=1)]) + EPS
    lam, vecs = np.linalg.eigh(np.exp(-d2 / (2 * ell2)))
    lam = np.maximum(lam, 0.0)
    offset = ya.mean()
    yc = ya - offset
    var = yc.var() + EPS
    proj = vecs.T @ yc

    def neg_lml(noise: float) -> float:
        c = var * (lam + noise)
        return float(0.5 * (proj**2 / c).sum() + 0.5 * np.log(c).sum())

    noise = min(NOISE_GRID, key=neg_lml)
    alpha = vecs @ (proj / (var * (lam + noise)))
    k_star = np.exp(-_sq_dist(xq, xa) / (2 * ell2))
    return offset + var * k_star @ alpha


MODELS: dict[str, Model] = {"knn": knn, "gp": gp}


def _descriptors(probe: Probe) -> dict[str, np.ndarray]:
    n_gen, pop = probe.truth.shape
    return {
        "W": probe.heads.reshape(n_gen, pop, -1),
        "B": probe.actions.reshape(n_gen, pop, -1),
    }


def _predict(x: np.ndarray, y: np.ndarray, t: int, model: Model) -> np.ndarray:
    pop = x.shape[1]
    xa = x[t - ARCHIVE : t].reshape(ARCHIVE * pop, -1)
    ya = y[t - ARCHIVE : t].reshape(-1)
    xa, xq = standardise(xa, x[t])
    return model(xa, ya, xq)


def overlaps(probe: Probe) -> dict[str, np.ndarray]:
    # per predicted row t >= FIRST: elite overlap vs truth of mu, of each
    # model alone (target z(truth)) and of z(mu) + residual model
    rows = range(FIRST, len(probe.truth))
    z_truth, z_mu = zscore(probe.truth), zscore(probe.mu)
    resid = z_truth - z_mu
    out: dict[str, list[float]] = {"mu": [], "rho_mu": []}
    for t in rows:
        out["mu"].append(elite_overlap(probe.mu[t], probe.truth[t]))
        out["rho_mu"].append(spearman(probe.mu[t], probe.truth[t]))
    for desc, x in _descriptors(probe).items():
        for name, model in MODELS.items():
            tag = f"{desc}_{name}"
            for key in ("alone", "corr", "rho"):
                out[f"{key}_{tag}"] = []
            for t in rows:
                alone = _predict(x, z_truth, t, model)
                corr = z_mu[t] + _predict(x, resid, t, model)
                out[f"alone_{tag}"].append(elite_overlap(alone, probe.truth[t]))
                out[f"corr_{tag}"].append(elite_overlap(corr, probe.truth[t]))
                out[f"rho_{tag}"].append(spearman(alone, probe.truth[t]))
    return {k: np.asarray(v) for k, v in out.items()}


def block_ci(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    # moving-block bootstrap: consecutive generations share the CEM state
    n = len(values)
    block = min(BLOCK, n)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, (RESAMPLES, -(-n // block)))
    idx = (starts[..., None] + np.arange(block)).reshape(RESAMPLES, -1)
    means = values[idx[:, :n]].mean(axis=1)
    lo, hi = np.percentile(means, CI)
    return float(lo), float(hi)


def _with_ci(name: str, values: np.ndarray) -> dict[str, float]:
    lo, hi = block_ci(values)
    return {name: float(values.mean()), f"{name}_lo": lo, f"{name}_hi": hi}


def _best(ovl: dict[str, np.ndarray], prefix: str, base: np.ndarray) -> str:
    # model choice on the selection rows only; the test rows judge it
    select = slice(0, SELECT_END - FIRST)
    return max(
        MODELS,
        key=lambda m: float(
            (ovl[f"{prefix}_{m}"][select] - base[select]).mean()
        ),
    )


def _loo_within(probe: Probe) -> np.ndarray:
    # upper bound without transfer: siblings predict each other
    scores = []
    for t in range(FIRST, len(probe.truth)):
        x = probe.actions[t].reshape(len(probe.truth[t]), -1)
        x, _ = standardise(x, x)
        z = zscore(probe.truth[t])
        pred = np.array(
            [
                knn(
                    np.delete(x, i, 0),
                    np.delete(z, i),
                    x[i : i + 1],
                    LOO_NEIGHBOURS,
                )[0]
                for i in range(len(z))
            ]
        )
        scores.append(elite_overlap(pred, probe.truth[t]))
    return np.asarray(scores)


def _participation(x: np.ndarray) -> float:
    s2 = np.linalg.svd(x - x.mean(axis=0), compute_uv=False) ** 2
    return float(s2.sum() ** 2 / ((s2**2).sum() + EPS))


def geometry(probe: Probe) -> dict[str, float]:
    n_gen, pop = probe.truth.shape
    flat = probe.actions.reshape(n_gen, pop, -1)
    rows = range(FIRST, n_gen)
    siblings = [_participation(flat[t]) for t in rows]
    archive = [
        _participation(flat[t - ARCHIVE : t].reshape(ARCHIVE * pop, -1))
        for t in rows
    ]
    drift = np.abs(np.diff(probe.mean_actions, axis=0)).mean(axis=(1, 2))
    spread = np.abs(
        probe.actions - probe.actions.mean(axis=1, keepdims=True)
    ).mean(axis=(1, 2, 3))
    return {
        "pr_siblings": float(np.median(siblings)),
        "pr_archive": float(np.median(archive)),
        "drift_mean_head": float(drift.mean()),
        "sibling_spread": float(spread.mean()),
    }


def summarize(probe: Probe) -> dict[str, float]:
    ovl = overlaps(probe)
    test = slice(SELECT_END - FIRST, None)
    base = ovl["mu"]
    p2_model = _best(ovl, "corr_B", base)
    b_model = _best(ovl, "alone_B", np.zeros_like(base))
    w_model = _best(ovl, "alone_W", np.zeros_like(base))
    p2 = ovl[f"corr_B_{p2_model}"] - base
    p1 = ovl[f"alone_B_{b_model}"] - ovl[f"alone_W_{w_model}"]
    stats = {
        "generations": float(len(probe.truth)),
        "chance_overlap": PARENTS / probe.truth.shape[1],
        "test_rows": float(len(base[test])),
        **_with_ci("P2", p2[test]),
        **_with_ci("P1", p1[test]),
        # kNN / GP shrink r_hat towards 0, so P2 is conservative
        **_with_ci("B_vs_mu", (ovl[f"alone_B_{b_model}"] - base)[test]),
        "P2_model_is_gp": float(p2_model == "gp"),
        "mu_overlap": float(base[test].mean()),
        "mu_spearman": float(ovl["rho_mu"][test].mean()),
        "loo_within_B": float(_loo_within(probe)[test].mean()),
    }
    for key, values in ovl.items():
        if key.startswith(("alone_", "corr_", "rho_")):
            stats[key] = float(values[test].mean())
    stats["corr_W_vs_mu"] = float(
        (ovl[f"corr_W_{_best(ovl, 'corr_W', base)}"] - base)[test].mean()
    )
    return stats | geometry(probe)


def verdict(stats: dict[str, float]) -> str:
    if stats["P2_lo"] > 0.0 and stats["P2"] >= MIN_GAIN:
        return "GO: behaviour-space residual adds to the critic's elite"
    if stats["P1_lo"] > 0.0:
        return "NO-GO (P1 > 0: behaviour transfers better than weights)"
    return "NO-GO"


def report(stats: dict[str, float]) -> str:
    lines = ["| metric | value |", "| --- | --- |"]
    lines += [f"| {key} | {value:.4g} |" for key, value in stats.items()]
    lines.append(
        f"\nverdict (P2 >= {MIN_GAIN} and one-sided 97.5 % CI low > 0): "
        + verdict(stats)
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="behaviour probe: does a residual ranker on actions "
        "over a fixed state set, trained on past generations, improve the "
        "critic's elite?"
    )
    parser.add_argument("probe", type=Path, help="behavior_probe.npz")
    args = parser.parse_args()
    print(report(summarize(load(args.probe))))


if __name__ == "__main__":
    main()
