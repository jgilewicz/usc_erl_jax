"""Cost/accuracy frontier of SEMARL's fitness estimators.

Reads a run's per-generation metrics (wandb run or exported CSV) and reports
what each estimator buys per env step. The headline metric is *elite
overlap* - the fraction of CEM's top-`parents` set the estimator gets right
- because that is all `_cem_tell` consumes; rank correlation also scores
pairs selection never looks at. Chance is 0.5, not 0.

Arms (metric-name suffix):
  ""        the h-step bootstrap
  _noboot   the same with the gamma^H * Q term dropped - the bootstrap's
            share of the value is exactly gamma^H (8.1% at gamma=.99, H=250),
            so if this matches "" the critic contributes nothing
  _critic   E_{s~D}[Q(s, pi_i(s))] over a replay batch - what selects when
            the population rollout is skipped, at zero env steps

Arms are only scored on real generations: a surrogate generation skips the
population rollout, so there is no ground truth to score against.

Settled and no longer reported (see notes.md): |TD|_rel does not predict
surrogate quality on either arm, |Q1-Q2| does not flag misranked
individuals, and H/2-vs-H rank stability came back as noise. p_surr is
still summarised because an adaptive gate indistinguishable from a fixed
one is the SEMARL baseline result.

    uv run python scripts/surrogate_diagnostics.py --csv run.csv
    uv run python scripts/surrogate_diagnostics.py --wandb evo_rl/triage_erl/abc123
"""

from __future__ import annotations

import argparse
import csv as csvmod
from pathlib import Path

import numpy as np

ARMS = {
    "": "H (h-step)",
    "_noboot": "H, no bootstrap",
    "_critic": "critic, batch-avg",
    "_pevfa": "PeVFA Q(s,a,χ(W))",
}

COLUMNS = [
    "generation",
    "td_error_rel",
    "p_surr",
    "h_step",
    "fitness_is_real",
    "env_steps",
    "env_steps_gen",
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
    # no keys= filter: scan_history drops every row missing any requested key
    hist = [row for row in run.scan_history() if "generation" in row]
    if not hist:
        raise ValueError(f"run {run.id} has no per-generation history")
    return {
        col: np.array([_as_float(row.get(col)) for row in hist])
        for col in COLUMNS
    }


def _describe_gate(p: np.ndarray, is_real: np.ndarray) -> None:
    valid = p[np.isfinite(p)]
    if not len(valid):
        print("!! no p_surr logged - run predates the adaptive gate")
        return
    real_frac = np.nanmean(is_real > 0.5)
    span = (
        f"p_surr: [{valid.min():.3f}, {valid.max():.3f}] "
        f"mean={valid.mean():.3f}   real generations: {real_frac:.0%}"
    )
    if np.ptp(valid) < 1e-3:
        print(f"!! p_surr constant ({valid[0]:.3f}) - the gate never moved")
    elif valid.mean() < 0.05 or valid.mean() > 0.95:
        # pinned against a rail is as much a non-result as constant, just easy to miss
        print(f"!! {span}\n   gate pinned against a rail - retune p_beta")
    else:
        print(span)


def _compare_arms(d: dict[str, np.ndarray]) -> None:
    print("\nFitness estimators   [elite_overlap chance = 0.5]")
    header = f"  {'arm':<20}{'elite_ovl':>11}{'rank_corr':>11}{'abs_err':>12}"
    print(f"{header}{'steps/gen':>11}{'signal/1k':>11}")
    # env_steps_gen taken from the run: full population cost on real gens, RL actor alone on surrogate gens
    per_gen = d["env_steps_gen"]
    per_gen = per_gen[np.isfinite(per_gen)]
    if not len(per_gen):
        print("  (no env_steps_gen logged - cannot cost the arms)")
        return
    full, actor_only = float(per_gen.max()), float(per_gen.min())
    costs = {
        "": full,
        "_noboot": full,
        "_critic": actor_only,
        "_pevfa": actor_only,
    }
    for suffix, label in ARMS.items():
        vals = [
            np.nanmean(d[f"surrogate_{m}{suffix}"])
            for m in ("elite_overlap", "rank_corr", "abs_err")
        ]
        if not np.isfinite(vals[0]):
            print(f"  {label:<20}{'n/a':>11}")
            continue
        cost = costs[suffix]
        per_1k = (vals[0] - 0.5) / (cost / 1000.0)
        print(
            f"  {label:<20}{vals[0]:>+11.3f}{vals[1]:>+11.3f}"
            f"{vals[2]:>12.1f}{cost:>11.0f}{per_1k:>11.3f}"
        )

    boot = d["surrogate_elite_overlap"]
    nobo = d["surrogate_elite_overlap_noboot"]
    if np.isfinite(nobo).any() and np.isfinite(boot).any():
        delta = np.nanmean(boot) - np.nanmean(nobo)
        verdict = "critic adds nothing" if abs(delta) < 0.02 else "contributes"
        print(f"\n  bootstrap contribution: {delta:+.3f} elite_ovl ({verdict})")


def _report_convergence(d: dict[str, np.ndarray]) -> None:
    # arms decay together as CEM narrows; what matters is whether the gap between them narrows too
    g = d["generation"]
    finite = g[np.isfinite(g)]
    if len(finite) < 40:
        return
    print("\nOver training (all arms decay together = CEM convergence)")
    print(f"  {'gen':>12}" + "".join(f"{lab:>19}" for lab in ARMS.values()))
    edges = np.linspace(finite.min(), finite.max() + 1, 5)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (g >= lo) & (g < hi)
        if not sel.any():
            continue
        cells = "".join(
            f"{np.nanmean(d[f'surrogate_elite_overlap{s}'][sel]):>19.3f}"
            for s in ARMS
        )
        print(f"  {int(lo):5d}-{int(hi):5d}{cells}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="exported per-generation CSV")
    src.add_argument(
        "--wandb", metavar="ENTITY/PROJECT/RUN_ID", help="wandb run path"
    )
    ap.add_argument(
        "--min-generation",
        type=int,
        default=0,
        help="drop generations before this (warmup / early noise)",
    )
    args = ap.parse_args()

    d = _from_csv(args.csv) if args.csv else _from_wandb(args.wandb)
    keep = d["generation"] >= args.min_generation
    d = {k: v[keep] for k, v in d.items()}
    print(f"{int(keep.sum())} generations after filtering")
    _describe_gate(d["p_surr"], d["fitness_is_real"])

    if not np.isfinite(d["surrogate_elite_overlap_critic"]).any():
        raise ValueError(
            "run logged no surrogate_elite_overlap_critic - it predates the "
            "arm rewrite; re-run before analysing"
        )

    steps = d["env_steps"]
    if np.isfinite(steps).any():
        print(f"cumulative env steps: {np.nanmax(steps):,.0f}")

    _compare_arms(d)
    _report_convergence(d)


if __name__ == "__main__":
    main()
