"""Go/no-go on uncertainty estimation over PeVFA, read across a seeded sweep.

Two gates, fixed before the sweep ran. Both are per seed, and every seed
must pass - with 3 seeds a mean can be carried by one lucky run.

  A  chi(W) is alive:   crossgen_elite_overlap  PeVFA - critic > +0.02
     The critic has no policy input, so it is the control: anything PeVFA
     ranks beyond it on the run-wide archive is what chi(W) carries.
  B  PeVFA can select:  surrogate_elite_overlap PeVFA >= 0.62
     U-PeVFA would gate *selection*, which ranks CEM siblings - so the
     within-generation number is the one that has to clear the bar. 0.62 is
     the threshold set before semarl-pevfa-r1-hc-s0 (critic sits at ~0.66);
     "above 0.5" alone is too weak, chance is 0.5.

  A and B -> uncertainty estimation is worth building
  A only  -> chi(W) carries policy information but lacks the resolution to
             separate siblings; uncertainty over it would not fix selection
  neither -> PeVFA is closed

    uv run python scripts/pevfa_verdict.py --tag sweep-v2
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import numpy as np

GATE_A_MARGIN = 0.02
GATE_B_FLOOR = 0.62
METRICS = {
    "within_pevfa": "surrogate_elite_overlap_pevfa",
    "within_critic": "surrogate_elite_overlap_critic",
    "xgen_pevfa": "crossgen_elite_overlap_pevfa",
    "xgen_critic": "crossgen_elite_overlap_critic",
}


def _run_means(run: Any) -> dict[str, float]:
    # no keys= filter: scan_history drops every row missing any requested key
    hist = [row for row in run.scan_history() if "generation" in row]
    out = {}
    for short, key in METRICS.items():
        vals = [row[key] for row in hist if row.get(key) is not None]
        out[short] = float(np.mean(vals)) if vals else float("nan")
    return out


def _condition(run_name: str) -> str:
    # slurm_semarl_sweep.sh names runs <condition>_<env>_s<seed>
    return run_name.split("_")[0]


def _report(by_cond: dict[str, list[dict[str, float]]]) -> None:
    head = f"{'condition':<14}{'seeds':>6}{'within P/C':>16}{'xgen P/C':>16}"
    print(f"{head}{'A: P-C':>9}{'B: P>=.62':>11}")
    pass_a, pass_b = [], []
    for cond, seeds in sorted(by_cond.items()):
        if not np.isfinite([s["within_pevfa"] for s in seeds]).any():
            continue
        a = [s["xgen_pevfa"] - s["xgen_critic"] for s in seeds]
        b = [s["within_pevfa"] for s in seeds]
        pass_a.append(all(x > GATE_A_MARGIN for x in a))
        pass_b.append(all(x >= GATE_B_FLOOR for x in b))
        mean = {k: np.nanmean([s[k] for s in seeds]) for k in METRICS}
        print(
            f"{cond:<14}{len(seeds):>6}"
            f"{mean['within_pevfa']:>9.3f}/{mean['within_critic']:.3f}"
            f"{mean['xgen_pevfa']:>9.3f}/{mean['xgen_critic']:.3f}"
            f"{np.mean(a):>+9.3f}{'all' if pass_b[-1] else 'no':>11}"
        )
        per_seed = "  ".join(f"{x:+.3f}/{y:.3f}" for x, y in zip(a, b))
        print(f"{'':<14}per seed (A/B): {per_seed}")
    if not pass_a:
        raise ValueError("no run in the sweep logged PeVFA metrics")
    a_ok = any(pass_a)
    b_ok = any(a and b for a, b in zip(pass_a, pass_b))
    verdict = (
        "build uncertainty estimation"
        if a_ok and b_ok
        else "chi(W) alive, no resolution - uncertainty would not fix selection"
        if a_ok
        else "PeVFA closed"
    )
    print(
        f"\nGate A in some condition: {a_ok}   A+B in the same one: {b_ok}   -> {verdict}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="sweep-v2")
    ap.add_argument("--project", default="evo_rl/triage_erl")
    args = ap.parse_args()

    import wandb

    runs = wandb.Api().runs(args.project, filters={"tags": args.tag})
    by_cond: dict[str, list[dict[str, float]]] = defaultdict(list)
    for run in runs:
        if run.state != "finished":
            print(f"skipping {run.name}: {run.state}")
            continue
        by_cond[_condition(run.name)].append(_run_means(run))
    if not by_cond:
        raise ValueError(
            f"no finished runs tagged {args.tag!r} in {args.project}"
        )
    _report(by_cond)


if __name__ == "__main__":
    main()
