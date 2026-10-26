from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# pre-registered in experiments/variation-v1b/experiment.toml; change there
# first
ENVS = ("HalfCheetah-v5", "Swimmer-v5")
STEPS = (50, 200)
SETS = ("cem",) + tuple(
    f"{kind}{t}" for t in STEPS for kind in ("mean", "member", "rand")
)
PARENTS = 5
MAX_GRAD_COS = 0.9
MIN_GAIN = 0.01
# one-sided 97.5 % per (env, T), Bonferroni over envs x step counts
CI = (2.5 / (len(ENVS) * len(STEPS)), 100 - 2.5 / (len(ENVS) * len(STEPS)))
BLOCK = 10
RESAMPLES = 10_000


@dataclass(frozen=True)
class Probe:
    first: np.ndarray  # (G, S, N) episode 1: selects the elite
    second: np.ndarray  # (G, S, N) episode 2: judges it
    grad_cos: np.ndarray  # (G, N) mean pairwise member-gradient cos
    delta_cos: np.ndarray  # (G, T, N) cos(member step, mean step)
    step_norm: np.ndarray  # (G, S - 1, N) |set - cem|
    q_mean: np.ndarray  # (G, S, N) mean-critic Q per candidate
    cem_norm: np.ndarray  # (G,) |scale * sqrt(cov)|


def load(path: Path) -> Probe:
    if not path.is_file():
        raise FileNotFoundError(
            f"no variation probe at {path}: run sc_erl with "
            "algorithm.variation_probe=true"
        )
    d = np.load(path)
    probe = Probe(**{name: d[name] for name in Probe.__dataclass_fields__})
    if probe.first.shape[1] != len(SETS):
        raise ValueError(
            f"{path} has {probe.first.shape[1]} sets, expected {SETS}; "
            "the probe and this report disagree on SETS (a variation-v1 "
            "npz needs the report at commit c0421f9)"
        )
    return probe


def elite_quality(first: np.ndarray, second: np.ndarray) -> float:
    # CEM sees one episode per individual; its parents are the top by that
    # episode, judged on an independent one
    parents = np.argsort(-first, kind="stable")[:PARENTS]
    return float(second[parents].mean())


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


def _per_steps(
    probe: Probe, q: np.ndarray, truth: np.ndarray, t: int
) -> dict[str, float]:
    cem, mean, member, rand = (
        SETS.index(name)
        for name in ("cem", f"mean{t}", f"member{t}", f"rand{t}")
    )
    paired = (truth[:, member] - truth[:, mean]).mean(axis=1)
    cem_norm = probe.cem_norm[:, None]
    return {
        **_with_ci(f"member_vs_mean{t}", q[:, member] - q[:, mean]),
        **_with_ci(f"member_vs_rand{t}", q[:, member] - q[:, rand]),
        **_with_ci(f"rand_vs_mean{t}", q[:, rand] - q[:, mean]),
        **_with_ci(f"mean_vs_cem{t}", q[:, mean] - q[:, cem]),
        **_with_ci(f"paired_member_vs_mean{t}", paired),
        f"step_mean{t}_over_cem": float(
            np.median(probe.step_norm[:, mean - 1] / cem_norm)
        ),
        f"step_member{t}_over_cem": float(
            np.median(probe.step_norm[:, member - 1] / cem_norm)
        ),
        f"delta_cos{t}": float(probe.delta_cos[:, STEPS.index(t)].mean()),
    }


def summarize(probe: Probe) -> dict[str, float]:
    n_gen, n_sets = probe.first.shape[:2]
    q = np.array(
        [
            [
                elite_quality(probe.first[g, s], probe.second[g, s])
                for s in range(n_sets)
            ]
            for g in range(n_gen)
        ]
    )
    truth = (probe.first + probe.second) / 2
    stats = {
        "generations": float(n_gen),
        "grad_cos_median": float(np.median(probe.grad_cos.mean(axis=1))),
    }
    for t in STEPS:
        stats |= _per_steps(probe, q, truth, t)
    for s, name in enumerate(SETS):
        stats[f"q_{name}"] = float(q[:, s].mean())
        stats[f"truth_{name}"] = float(truth[:, s].mean())
        stats[f"spread_{name}"] = float(truth[:, s].std(axis=1).mean())
        stats[f"critic_q_{name}"] = float(probe.q_mean[:, s].mean())
    return stats


def _beats(stats: dict[str, float], name: str) -> bool:
    return stats[f"{name}_lo"] > 0.0 and stats[name] >= MIN_GAIN * abs(
        stats["q_cem"]
    )


def passes(stats: dict[str, float], t: int) -> bool:
    return (
        stats["grad_cos_median"] < MAX_GRAD_COS
        and _beats(stats, f"member_vs_mean{t}")
        and stats[f"member_vs_rand{t}_lo"] > 0.0
    )


def verdict(results: dict[str, dict[str, float]]) -> str:
    missing = [env for env in ENVS if env not in results]
    if missing:
        return f"no verdict: {', '.join(missing)} missing"
    cells = [(env, t) for env in ENVS for t in STEPS]
    go = [f"{env}@{t}" for env, t in cells if passes(results[env], t)]
    if go:
        return f"GO: uncertainty as variation ({', '.join(go)})"
    noise = [
        f"{env}@{t}"
        for env, t in cells
        if _beats(results[env], f"rand_vs_mean{t}")
    ]
    if noise:
        return f"NOISE: gradient + random spread helps ({', '.join(noise)})"
    return "NO-GO"


def report(results: dict[str, dict[str, float]]) -> str:
    envs = list(results)
    keys = list(next(iter(results.values())))
    lines = [
        "| metric | " + " | ".join(envs) + " |",
        "| --- |" + " --- |" * len(envs),
    ]
    for key in keys:
        cells = " | ".join(f"{results[env][key]:.4g}" for env in envs)
        lines.append(f"| {key} | {cells} |")
    lines.append(
        f"\nverdict (per env and T in {STEPS}: grad_cos_median < "
        f"{MAX_GRAD_COS}, member_vs_mean CI low > 0 and >= {MIN_GAIN:.0%} "
        "of |q_cem|, member_vs_rand CI low > 0): " + verdict(results)
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="variation probe: do per-member critic-gradient "
        "offspring beat mean-critic offspring, with and without a random "
        "spread of the same size?"
    )
    parser.add_argument(
        "probes",
        nargs="+",
        help="ENV=PATH to variation_probe.npz, one per env",
    )
    args = parser.parse_args()
    results = {}
    for spec in args.probes:
        env, sep, path = spec.partition("=")
        if not sep:
            raise ValueError(f"expected ENV=PATH, got {spec!r}")
        results[env] = summarize(load(Path(path)))
    print(report(results))


if __name__ == "__main__":
    main()
