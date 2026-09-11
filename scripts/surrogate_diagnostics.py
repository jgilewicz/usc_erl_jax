"""Post-hoc validation of SEMARL's adaptive-H mechanism.

Reads a run's per-generation metrics (wandb run or exported CSV) and asks,
using the surrogate's *rank* correlation with the true return (scale-free -
what CEM selection depends on, and it survives the surrogate/real
return-scale mismatch):

  1. Does |TD|_rel predict a worse-ranking surrogate?
     corr(td_error_rel, rank_corr at a fixed horizon)      -- expect negative
  2. Premise - does a short horizon rank worse when the critic is worse?
     corr(td_error_rel, rank_corr_hmin - rank_corr_hmax)   -- expect negative

Uses td_error_rel (|TD| / mean|Q|) when the run logged it, else falls back
to the raw td_error (older runs) with a warning - the raw residual scales
with |Q|, which grows both across envs and within a run as returns grow, so
it is not the quantity adaptive_h_step's h_beta was calibrated against.

Also flags a saturated h_step (no variance = the run never actually
adapted, usually because h_beta didn't match this run's TD-error scale) -
that alone invalidates the correlations below, since there is nothing for
them to be measured against.

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
from scipy.stats import spearmanr

COLUMNS = [
    "generation",
    "td_error",
    "td_error_rel",
    "td_error_rel_ema",
    "h_step",
    "fitness_is_real",
    "surrogate_abs_err",
    "surrogate_abs_err_hmin",
    "surrogate_abs_err_hmax",
    "surrogate_rank_corr",
    "surrogate_rank_corr_hmin",
    "surrogate_rank_corr_hmax",
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
    # was never logged (e.g. rank_corr on an older run).
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
        print(f"{label:<40} n={len(x):4d}  (too few points)")
        return
    r_raw, p_raw = spearmanr(x, y)
    r_det, p_det = spearmanr(_detrend(x, window), _detrend(y, window))
    print(
        f"{label:<40} n={len(x):4d}  "
        f"raw rho={r_raw:+.3f} (p={p_raw:.1e})   "
        f"detrended rho={r_det:+.3f} (p={p_det:.1e})"
    )


def _rolling_spearman(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(len(x) - window + 1):
        xi, yi = x[i : i + window], y[i : i + window]
        m = np.isfinite(xi) & np.isfinite(yi)
        if m.sum() >= 5:
            out[i + window // 2] = spearmanr(xi[m], yi[m])[0]
    return out


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

    h = d["h_step"]
    h_valid = h[np.isfinite(h)]
    if len(h_valid) and h_valid.max() - h_valid.min() < 1:
        print(
            f"\n!! h_step is constant ({h_valid[0]:.0f}) for every generation "
            "-- H never adapted, so nothing below is measuring what it "
            "claims to. Usually h_beta doesn't match this run's TD-error "
            "scale (all-real-eval or a saturated sigmoid); fix that and "
            "re-run before trusting the correlations."
        )
    else:
        print(f"h_step range: [{h_valid.min():.0f}, {h_valid.max():.0f}]")

    if np.isfinite(d["td_error_rel"]).any():
        td = d["td_error_rel"]
        td_label = "|TD|_rel"
    else:
        print(
            "\n(no td_error_rel - older run; falling back to raw td_error, "
            "which is not comparable across envs/training progress)"
        )
        td = d["td_error"]
        td_label = "|TD|"
    print()

    rc_min = d["surrogate_rank_corr_hmin"]
    rc_max = d["surrogate_rank_corr_hmax"]
    rc_adapt = d["surrogate_rank_corr"]
    rc_gap = rc_min - rc_max
    has_rc = np.isfinite(rc_max).any()

    if has_rc:
        print(
            f"Q1  does {td_label} predict a worse-ranking surrogate? (expect < 0)"
        )
        _report(f"  {td_label} vs rank_corr(H=h_min)", td, rc_min, args.window)
        _report(f"  {td_label} vs rank_corr(H=h_max)", td, rc_max, args.window)
        print(
            "\nQ2  premise: short H ranks worse when critic worse (expect < 0)"
        )
        _report(
            f"  {td_label} vs (rank_corr_hmin - rank_corr_hmax)",
            td,
            rc_gap,
            args.window,
        )
        print("\nsecondary (confounded by compensation / lag)")
        _report(
            "  h_step vs rank_corr(adaptive H)",
            d["h_step"],
            rc_adapt,
            args.window,
        )
        print(
            f"\nmean rank_corr: h_min={np.nanmean(rc_min):+.3f}  "
            f"adaptive={np.nanmean(rc_adapt):+.3f}  "
            f"h_max={np.nanmean(rc_max):+.3f}"
        )
    else:
        print("(no rank_corr columns - run predates them; abs_err only)\n")

    err_min = d["surrogate_abs_err_hmin"]
    err_max = d["surrogate_abs_err_hmax"]
    print("\nabs_err (scale-sensitive, watch surrogate/real scale drift)")
    _report(
        f"  {td_label} vs (err_hmin - err_hmax)",
        td,
        err_min - err_max,
        args.window,
    )
    frac = float(np.nanmean((err_min < err_max).astype(float)))
    print(
        f"  err_hmin < err_hmax in {frac:.0%} of gens  "
        f"(mean err_hmin={np.nanmean(err_min):.2f}, "
        f"err_hmax={np.nanmean(err_max):.2f})"
    )

    if args.out:
        series = {
            "generation": d["generation"],
            "roll_sp_td_rc_max": _rolling_spearman(td, rc_max, args.window),
            "roll_sp_td_rc_gap": _rolling_spearman(td, rc_gap, args.window),
        }
        with args.out.open("w", newline="") as fh:
            w = csvmod.writer(fh)
            w.writerow(series.keys())
            w.writerows(zip(*series.values(), strict=True))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
