from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# pre-registered in experiments/variation-v1/experiment.toml; change there
# first
ENVS = ("HalfCheetah-v5", "dog-stand", "Swimmer-v5")
SETS = ("cem", "mean", "member")
PARENTS = 5
MAX_GRAD_COS = 0.9
MIN_GAIN = 0.01
# one-sided 97.5 % per env, Bonferroni over the 3 envs
CI = (2.5 / len(ENVS), 100 - 2.5 / len(ENVS))
BLOCK = 10
RESAMPLES = 10_000


@dataclass(frozen=True)
class Probe:
    first: np.ndarray  # (G, 3, N) episode 1: selects the elite
    second: np.ndarray  # (G, 3, N) episode 2: judges it
    grad_cos: np.ndarray  # (G, N) mean pairwise member-gradient cos
    delta_cos: np.ndarray  # (G, N) cos(member step, mean step)
    step_norm: np.ndarray  # (G, 2, N) |mean - cem|, |member - cem|
    action_dist: np.ndarray  # (G, 2, N) member<->mean, mean<->cem
    q_mean: np.ndarray  # (G, 3, N) mean-critic Q per candidate
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
            "the probe and this report disagree on SETS"
        )
    return probe


def elite_quality(first: np.ndarray, second: np.ndarray) -> float:
    # CEM sees one episode per individual; its parents are the top by that
    # episode, judged on an independent one
    parents = np.argsort(-first, kind="stable")[:PARENTS]
    return float(second[parents].mean())


def icc(first: np.ndarray, second: np.ndarray) -> float:
    # one-way ICC(1,1) over candidates with two episodes each
    pair = np.stack([first, second], axis=1)
    n = len(first)
    msb = 2 * pair.mean(axis=1).var(ddof=1)
    msw = ((pair - pair.mean(axis=1, keepdims=True)) ** 2).sum() / n
    return float((msb - msw) / (msb + msw))


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


def summarize(probe: Probe) -> dict[str, float]:
    n_gen = len(probe.first)
    q = np.array(
        [
            [
                elite_quality(probe.first[g, s], probe.second[g, s])
                for s in range(3)
            ]
            for g in range(n_gen)
        ]
    )
    iccs = np.array(
        [
            [icc(probe.first[g, s], probe.second[g, s]) for s in range(3)]
            for g in range(n_gen)
        ]
    )
    truth = (probe.first + probe.second) / 2
    cem_norm = probe.cem_norm[:, None]
    stats = {
        "generations": float(n_gen),
        "grad_cos_median": float(np.median(probe.grad_cos.mean(axis=1))),
        **_with_ci("member_vs_mean", q[:, 2] - q[:, 1]),
        **_with_ci("mean_vs_cem", q[:, 1] - q[:, 0]),
        **_with_ci("member_vs_cem", q[:, 2] - q[:, 0]),
        "delta_cos": float(probe.delta_cos.mean()),
        "step_mean_over_cem": float(
            np.median(probe.step_norm[:, 0] / cem_norm)
        ),
        "step_member_over_cem": float(
            np.median(probe.step_norm[:, 1] / cem_norm)
        ),
        "action_dist_member_mean": float(probe.action_dist[:, 0].mean()),
        "action_dist_mean_cem": float(probe.action_dist[:, 1].mean()),
    }
    for s, name in enumerate(SETS):
        stats[f"q_{name}"] = float(q[:, s].mean())
        stats[f"truth_{name}"] = float(truth[:, s].mean())
        stats[f"icc_{name}"] = float(np.nanmean(iccs[:, s]))
        stats[f"critic_q_{name}"] = float(probe.q_mean[:, s].mean())
    return stats


def env_passes(stats: dict[str, float]) -> bool:
    return (
        stats["grad_cos_median"] < MAX_GRAD_COS
        and stats["member_vs_mean_lo"] > 0.0
        and stats["member_vs_mean"] >= MIN_GAIN * abs(stats["q_cem"])
    )


def verdict(results: dict[str, dict[str, float]]) -> str:
    missing = [env for env in ENVS if env not in results]
    if missing:
        return f"no verdict: {', '.join(missing)} missing"
    harmed = [env for env in ENVS if results[env]["member_vs_mean_hi"] < 0.0]
    passed = [env for env in ENVS if env_passes(results[env])]
    if harmed or not passed:
        why = f"harmed on {', '.join(harmed)}" if harmed else "no env passes"
        return f"NO-GO ({why})"
    return f"GO ({', '.join(passed)})"


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
        f"\nverdict (per env: grad_cos_median < {MAX_GRAD_COS}, "
        f"member_vs_mean CI low > 0 and >= {MIN_GAIN:.0%} of |q_cem|; "
        "no env with member_vs_mean CI high < 0): " + verdict(results)
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="variation probe: do per-member critic-gradient "
        "offspring give a better elite than mean-critic gradient offspring?"
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
