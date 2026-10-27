from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# pre-registered in experiments/uqd-probe-v1/experiment.toml; change there
# first
RULES = ("raw", "eb_hetero", "eb_lineage", "elite_truth")
TESTED = ("eb_hetero", "eb_lineage")
# 2 tested rules x 2 tasks: Bonferroni one-sided 97.5 % -> 0.625 % percentile
CI = (0.625, 99.375)
MIN_LUCKY_Z = 0.5
MIN_REGRET = 0.2
MIN_RECOVERED = 0.1
BLOCK = 3
RESAMPLES = 10_000
_EPS = 1e-6


@dataclass(frozen=True)
class Pairs:
    # one contested decision per row: the batch winner vs the incumbent
    y_o: np.ndarray
    y_e: np.ndarray
    mu_o: np.ndarray
    mu_e: np.ndarray
    var_o: np.ndarray
    var_e: np.ndarray
    mu_p: np.ndarray
    accepted: np.ndarray


@dataclass(frozen=True)
class Prior:
    m: float
    tau2: float
    tau2_lineage: float


def load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"no UQD probe at {path}: run map_elites with "
            "algorithm.uqd_probe=true"
        )
    d = np.load(path)
    return {k: d[k] for k in d.files}


def contested(d: dict[str, np.ndarray], g: int) -> Pairs:
    stored = d["elite_fitness"][g]
    cells = d["off_cell"][g]
    rows = d["off_winner"][g] & np.isfinite(stored[cells])
    target = cells[rows]
    elite_reps = d["elite_reps_f"][g]
    off_reps = d["off_reps_f"][g][rows]
    parents = d["off_parent1"][g][rows]
    return Pairs(
        y_o=d["off_fitness"][g][rows],
        y_e=stored[target],
        mu_o=off_reps.mean(axis=1),
        mu_e=elite_reps[target].mean(axis=1),
        var_o=off_reps.var(axis=1, ddof=1),
        var_e=elite_reps[target].var(axis=1, ddof=1),
        mu_p=elite_reps[parents].mean(axis=1),
        accepted=d["off_accepted"][g][rows],
    )


def prior(d: dict[str, np.ndarray], g: int) -> Prior:
    # method of moments over all offspring of the generation: the
    # population every elite was once drawn from
    y = d["off_fitness"][g]
    reps = d["off_reps_f"][g]
    noise = float(reps.var(axis=1, ddof=1).mean())
    parent = d["elite_reps_f"][g][d["off_parent1"][g]].mean(axis=1)
    return Prior(
        m=float(y.mean()),
        tau2=max(float(y.var(ddof=1)) - noise, _EPS),
        tau2_lineage=max(float((y - parent).var(ddof=1)) - noise, _EPS),
    )


def shrink(
    y: np.ndarray, var: np.ndarray, mean: np.ndarray | float, tau2: float
) -> np.ndarray:
    return mean + tau2 / (tau2 + var) * (y - mean)


def decisions(p: Pairs, pr: Prior) -> dict[str, np.ndarray]:
    # oracle sigma from the repeats; the elite has no lineage prior that
    # is still observable, so it keeps the population prior
    elite_post = shrink(p.y_e, p.var_e, pr.m, pr.tau2)
    return {
        "raw": p.y_o > p.y_e,
        "eb_hetero": shrink(p.y_o, p.var_o, pr.m, pr.tau2) > elite_post,
        "eb_lineage": shrink(p.y_o, p.var_o, p.mu_p, pr.tau2_lineage)
        > elite_post,
        "elite_truth": p.y_o > p.mu_e,
    }


def archive_stats(d: dict[str, np.ndarray], g: int) -> dict[str, float]:
    stored = d["elite_fitness"][g]
    occupied = np.isfinite(stored)
    reps = d["elite_reps_f"][g][occupied]
    mu = reps.mean(axis=1)
    sd = reps.std(axis=1, ddof=1)
    resid = (stored[occupied] - mu) / np.maximum(sd, _EPS)
    survived = d["elite_survived"][g][occupied]
    placed = d["elite_placed"][g][occupied]
    return {
        "lucky_z": float(
            (stored[occupied] - mu).mean() / np.sqrt((sd**2).mean() + _EPS)
        ),
        "placement_acc": float(placed.mean()),
        "corr_survived_luck": _corr(survived.astype(float), resid),
        "occupied": float(occupied.sum()),
    }


def heritability(d: dict[str, np.ndarray], g: int) -> dict[str, float]:
    off = d["off_reps_f"][g]
    par = d["elite_reps_f"][g][d["off_parent1"][g]]
    return {
        "herit_mu": _corr(off.mean(axis=1), par.mean(axis=1)),
        "herit_log_sd": _corr(
            np.log(off.std(axis=1, ddof=1) + _EPS),
            np.log(par.std(axis=1, ddof=1) + _EPS),
        ),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def generation(d: dict[str, np.ndarray], g: int) -> dict[str, float]:
    p = contested(d, g)
    if not np.array_equal(p.y_o > p.y_e, p.accepted):
        raise ValueError(
            f"generation {g}: raw rule disagrees with recorded insertion; "
            "uqd_probe.batch_winners and qd_archive.insert have diverged"
        )
    delta = p.mu_o - p.mu_e
    truth = delta > 0
    row = {"contested": float(delta.size)}
    row["gain_oracle"] = float(np.maximum(delta, 0).mean()) if delta.size else 0
    for name, accept in decisions(p, prior(d, g)).items():
        row[f"gain_{name}"] = (
            float((accept * delta).mean()) if delta.size else 0
        )
        row[f"acc_{name}"] = (
            float((accept == truth).mean()) if delta.size else np.nan
        )
    return row | archive_stats(d, g) | heritability(d, g)


def block_ci(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    # moving-block bootstrap: consecutive probes share the archive
    n = len(values)
    block = min(BLOCK, n)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, (RESAMPLES, -(-n // block)))
    idx = (starts[..., None] + np.arange(block)).reshape(RESAMPLES, -1)
    lo, hi = np.percentile(values[idx[:, :n]].mean(axis=1), CI)
    return float(lo), float(hi)


def qd_ratio(d: dict[str, np.ndarray]) -> float:
    # QD-score convention: offset by the run's lowest fitness so every cell
    # counts >= 0; a misplaced elite contributes nothing to its cell
    stored = d["elite_fitness"][-1]
    occupied = np.isfinite(stored)
    mu = d["elite_reps_f"][-1].mean(axis=1)
    every = d["elite_fitness"][np.isfinite(d["elite_fitness"])]
    offset = min(
        float(every.min()), float(d["elite_reps_f"].mean(axis=2).min())
    )
    illusory = (stored[occupied] - offset).sum()
    corrected = (mu - offset)[occupied & d["elite_placed"][-1]].sum()
    return float(corrected / max(illusory, _EPS))


def summarize(d: dict[str, np.ndarray]) -> dict[str, float]:
    rows = [generation(d, g) for g in range(len(d["generation"]))]
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out = {k: float(np.nanmean(v)) for k, v in cols.items()}
    headroom = cols["gain_oracle"] - cols["gain_raw"]
    out["regret_raw"] = float(
        headroom.sum() / max(cols["gain_oracle"].sum(), _EPS)
    )
    out["lucky_z_lo"], out["lucky_z_hi"] = block_ci(cols["lucky_z"])
    for name in RULES[1:]:
        diff = cols[f"gain_{name}"] - cols["gain_raw"]
        out[f"delta_{name}"] = float(diff.mean())
        out[f"delta_{name}_lo"], out[f"delta_{name}_hi"] = block_ci(diff)
        out[f"recovered_{name}"] = float(diff.sum() / max(headroom.sum(), _EPS))
    out["qd_corrected_over_stored"] = qd_ratio(d)
    return out


def verdict(stats: dict[str, float]) -> str:
    gap = (
        stats["lucky_z_lo"] > MIN_LUCKY_Z and stats["regret_raw"] >= MIN_REGRET
    )
    if not gap:
        return "NO-GO (P1: no gap)"
    passed = [
        name
        for name in TESTED
        if stats[f"delta_{name}_lo"] > 0
        and stats[f"recovered_{name}"] >= MIN_RECOVERED
    ]
    return f"GO ({', '.join(passed)})" if passed else "NO-GO (P2: no headroom)"


def report(results: dict[str, dict[str, float]]) -> str:
    tasks = list(results)
    lines = [
        "| metric | " + " | ".join(tasks) + " |",
        "| --- |" + " --- |" * len(tasks),
    ]
    for key in results[tasks[0]]:
        cells = " | ".join(f"{results[t][key]:.3f}" for t in tasks)
        lines.append(f"| {key} | {cells} |")
    lines.append(
        f"\nP1: lucky_z CI low > {MIN_LUCKY_Z} and regret_raw >= "
        f"{MIN_REGRET:.0%}; P2: some rule in {TESTED} with delta CI low > 0 "
        f"and >= {MIN_RECOVERED:.0%} of headroom recovered"
    )
    lines += [f"verdict {t}: {verdict(results[t])}" for t in tasks]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UQD probe: are MAP-Elites elites lucky, and do "
        "shrinkage replacement rules recover the lost decisions?"
    )
    parser.add_argument(
        "probes", nargs="+", help="TASK=PATH to uqd_probe.npz, one per task"
    )
    args = parser.parse_args()
    results = {}
    for spec in args.probes:
        task, sep, path = spec.partition("=")
        if not sep:
            raise ValueError(f"expected TASK=PATH, got {spec!r}")
        results[task] = summarize(load(Path(path)))
    print(report(results))


if __name__ == "__main__":
    main()
