"""Post-hoc validation of SEMARL's adaptive-p_surr mechanism.

Reads a run's per-generation metrics (wandb run or exported CSV). The
headline metric is *elite overlap* - the fraction of CEM's top-`parents`
set that the surrogate gets right - because that is all `_cem_tell`
consumes; rank correlation also scores pairs selection never looks at.

  1. Does |TD|_rel predict a worse surrogate?
     corr(td_error_rel, elite_overlap)          -- expect negative
  2. Does the gate move, and does it move the right way?
     p_surr spread, and corr(p_surr, elite_overlap) -- expect positive
  3. Baseline arm: h-step bootstrap (H) vs pure critic value (H=0).
     If H=0 overlap is near chance, a SEMARL-style critic surrogate
     cannot drive CEM here whatever gates it.
  4. Does per-individual critic disagreement flag the misranked ones?
     mean corr(|Q1-Q2|, rank displacement)      -- expect positive

|TD|_rel is |TD| / mean|r|: both per-step reward-scale quantities, so the
ratio is portable across envs and doesn't drift as returns grow.

Reported raw and detrended: the TD signal and surrogate quality both move
over training, and that shared trend alone inflates a raw correlation.
Trust the detrended number (Spearman on rolling-median residuals).

    uv run python scripts/surrogate_diagnostics.py --csv run.csv
    uv run python scripts/surrogate_diagnostics.py --wandb evo_rl/triage_erl/abc123
"""

from __future__ import annotations

import argparse
import csv as csvmod
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

# metric-name suffix -> label. "" is the h-step bootstrap driving selection.
ARMS = {
    "": "H (h-step)",
    "_noboot": "H, no bootstrap",
    "_half": "H/2",
    "_critic": "critic, batch-avg",
    "_h0": "critic, 1 state",
}

COLUMNS = [
    "generation",
    "td_error",
    "td_error_rel",
    "td_error_rel_ema",
    "p_surr",
    "h_step",
    "fitness_is_real",
    "q_disagree_mean",
    "q_disagree_rank_corr",
    "surrogate_rank_stability",
    "surrogate_elite_stability",
    *(
        f"surrogate_{metric}{suffix}"
        for metric in ("abs_err", "rank_corr", "elite_overlap")
        for suffix in ARMS
    ),
]


def _as_float(v: str | float | None) -> float:
    try:
        return float(v) if v not in (None, "", "NaN") else np.nan
    except (TypeError, ValueError):
        return np.nan


def _from_csv(path: Path) -> dict[str, np.ndarray]:
    rows = list(csvmod.DictReader(path.read_text().splitlines()))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    if "generation" not in rows[0]:
        raise ValueError(f"{path} has no 'generation' column")
    return {
        col: np.array([_as_float(r.get(col, "")) for r in rows])
        for col in COLUMNS
    }


def _from_wandb(spec: str) -> dict[str, np.ndarray]:
    import wandb

    api = wandb.Api()
    try:
        run = api.run(spec)
    except wandb.errors.CommError:
        # spec's last segment may be a display name, not the run id
        project, name = spec.rsplit("/", 1)
        matches = list(api.runs(project, filters={"display_name": name}))
        if not matches:
            raise ValueError(
                f"no run with id or name {name!r} in {project}"
            ) from None
        if len(matches) > 1:
            ids = ", ".join(r.id for r in matches)
            raise ValueError(
                f"{name!r} matches several runs in {project}: {ids} — "
                "pass one of those ids"
            ) from None
        run = matches[0]
    # no keys= filter: scan_history drops every row when any requested key
    # was never logged (e.g. elite_overlap on an older run).
    hist = [row for row in run.scan_history() if "generation" in row]
    if not hist:
        raise ValueError(f"run {run.id} has no per-generation history")
    return {
        col: np.array([_as_float(row.get(col)) for row in hist])
        for col in COLUMNS
    }


def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    window = min(window, len(x))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return np.zeros_like(x)
    pad = window // 2
    padded = np.pad(x, pad, mode="edge")
    return np.array(
        [np.nanmedian(padded[i : i + window]) for i in range(len(x))]
    )


def _detrend(x: np.ndarray, window: int) -> np.ndarray:
    return x - _rolling_median(x, window)


def _report(label: str, x: np.ndarray, y: np.ndarray, window: int) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        print(f"{label:<44} n={len(x):4d}  (too few points)")
        return
    r_raw, p_raw = spearmanr(x, y)
    xd, yd = _detrend(x, window), _detrend(y, window)
    if np.ptp(xd) == 0 or np.ptp(yd) == 0:
        det = "detrended n/a (constant within window)"
    else:
        r_det, p_det = spearmanr(xd, yd)
        det = f"detrended rho={r_det:+.3f} (p={p_det:.1e})"
    print(
        f"{label:<44} n={len(x):4d}  "
        f"raw rho={r_raw:+.3f} (p={p_raw:.1e})   {det}"
    )


def _rolling_spearman(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(len(x) - window + 1):
        xi, yi = x[i : i + window], y[i : i + window]
        m = np.isfinite(xi) & np.isfinite(yi)
        if m.sum() >= 5:
            out[i + window // 2] = spearmanr(xi[m], yi[m])[0]
    return out


def _describe_gate(p: np.ndarray) -> None:
    valid = p[np.isfinite(p)]
    if not len(valid):
        print("!! no p_surr logged - run predates the adaptive gate")
        return
    span = f"p_surr: [{valid.min():.3f}, {valid.max():.3f}] mean={valid.mean():.3f}"
    if np.ptp(valid) < 1e-3:
        print(
            f"!! p_surr is constant ({valid[0]:.3f}) for every generation "
            "-- the gate never adapted, so Q1/Q2 below measure nothing. "
            "Usually p_beta doesn't match this run's |TD|_rel scale."
        )
    elif valid.mean() < 0.05 or valid.mean() > 0.95:
        # a gate that technically moves but sits against a rail is the same
        # non-result as a constant one, and is easy to miss in the range.
        print(
            f"!! {span}\n   the gate moves but is pinned against a rail -- "
            "retune p_beta against this run's |TD|_rel before reading Q1/Q2."
        )
    else:
        print(span)


def _compare_arms(d: dict[str, np.ndarray]) -> None:
    # CEM keeps parents = pop_size // 2, so two unrelated rankings already
    # share half their elite set: 0.5 is chance, not 0.
    print("\nQ3  fitness estimators   [elite_overlap chance = 0.5]")
    print(f"  {'arm':<20}{'elite_ovl':>11}{'rank_corr':>11}{'abs_err':>12}")
    for suffix, label in ARMS.items():
        cells = []
        for metric in ("elite_overlap", "rank_corr", "abs_err"):
            v = d[f"surrogate_{metric}{suffix}"]
            cells.append(
                "     n/a" if not np.isfinite(v).any() else np.nanmean(v)
            )
        fmt = "".join(
            f"{c:>11}" if isinstance(c, str) else f"{c:>+11.3f}"
            for c in cells[:2]
        )
        err = cells[2]
        tail = f"{err:>12}" if isinstance(err, str) else f"{err:>12.1f}"
        print(f"  {label:<20}{fmt}{tail}")

    # if dropping gamma^H * Q changes nothing, the critic is not contributing
    # to the surrogate that actually drives selection.
    boot = d["surrogate_elite_overlap"]
    nobo = d["surrogate_elite_overlap_noboot"]
    if np.isfinite(nobo).any():
        delta = np.nanmean(boot) - np.nanmean(nobo)
        print(
            f"\n  bootstrap contribution: elite_overlap {delta:+.3f} "
            f"vs reward-only  ({'critic adds nothing' if abs(delta) < 0.02 else 'critic contributes'})"
        )


def _report_disagreement(rc: np.ndarray) -> None:
    print("\nQ4  does |Q1-Q2| flag the misranked individuals? (expect > 0)")
    valid = rc[np.isfinite(rc)]
    if len(valid) < 5:
        print("  (no q_disagree_rank_corr logged)")
        return
    # per-generation rho over only pop_size individuals is very noisy; the
    # signed-rank test over generations is what carries the evidence.
    stat = wilcoxon(valid, alternative="greater")
    print(
        f"  mean per-gen rho={valid.mean():+.3f}  "
        f"(>0 in {np.mean(valid > 0):.0%} of {len(valid)} gens, "
        f"wilcoxon p={stat.pvalue:.1e})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="exported per-generation CSV")
    src.add_argument(
        "--wandb", metavar="ENTITY/PROJECT/RUN_ID", help="wandb run path"
    )
    ap.add_argument("--window", type=int, default=21, help="detrend window")
    ap.add_argument(
        "--min-generation",
        type=int,
        default=0,
        help="drop generations before this (warmup / early noise)",
    )
    ap.add_argument(
        "--real-only",
        action="store_true",
        help="keep only generations with a true full-episode return",
    )
    ap.add_argument(
        "--out", type=Path, help="write rolling-Spearman series to this CSV"
    )
    args = ap.parse_args()

    d = _from_csv(args.csv) if args.csv else _from_wandb(args.wandb)
    keep = d["generation"] >= args.min_generation
    if args.real_only:
        keep &= d["fitness_is_real"] > 0.5
    d = {k: v[keep] for k, v in d.items()}
    print(f"{int(keep.sum())} generations after filtering")
    _describe_gate(d["p_surr"])

    td = d["td_error_rel"]
    overlap = d["surrogate_elite_overlap"]
    if not np.isfinite(overlap).any():
        raise ValueError(
            "run logged no surrogate_elite_overlap - it predates the "
            "adaptive-p_surr rewrite; re-run before analysing"
        )

    print("\nQ1  does |TD|_rel predict a worse surrogate? (expect < 0)")
    _report("  |TD|_rel vs elite_overlap", td, overlap, args.window)
    _report(
        "  |TD|_rel vs rank_corr", td, d["surrogate_rank_corr"], args.window
    )

    print("\nQ2  does the gate open when the surrogate is good? (expect > 0)")
    _report("  p_surr vs elite_overlap", d["p_surr"], overlap, args.window)

    _compare_arms(d)
    _report_disagreement(d["q_disagree_rank_corr"])

    # the rival gate signal: truncation error is what gamma^H says the critic
    # is not responsible for, and rank stability sees it without a critic.
    print("\nQ5  does H/2-vs-H rank stability predict the surrogate? (> 0)")
    for label, key in (
        ("  rank_stability vs elite_overlap", "surrogate_rank_stability"),
        ("  elite_stability vs elite_overlap", "surrogate_elite_stability"),
    ):
        if np.isfinite(d[key]).any():
            _report(label, d[key], overlap, args.window)
        else:
            print(f"{label:<44} (not logged - run predates it)")

    if args.out:
        series = {
            "generation": d["generation"],
            "roll_sp_td_overlap": _rolling_spearman(td, overlap, args.window),
            "roll_sp_p_overlap": _rolling_spearman(
                d["p_surr"], overlap, args.window
            ),
        }
        with args.out.open("w", newline="") as fh:
            w = csvmod.writer(fh)
            w.writerow(series.keys())
            w.writerows(zip(*series.values(), strict=True))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
