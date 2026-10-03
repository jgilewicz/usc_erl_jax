"""Offline test of a measured-error surrogate gate (s*ACM-ES style).

s*ACM-ES (Loshchilov, Schoenauer, Sebag, GECCO 2012) does not predict
surrogate quality: after every real generation it measures the surrogate's
ranking error on the points it just evaluated and trusts the surrogate for
    n = floor(L_max * max(0, (thr - err) / thr))
generations before the next real one. Every gate this project tried
predicted quality and failed; this one measures it.

Replayed on SEMARL population dumps from p_surr=0 runs, where every
generation has true returns, so any schedule can be scored: a real
generation selects perfectly (elite_overlap 1.0), a surrogate generation
gets the critic's elite_overlap on that generation. Compared at the same
fraction of real generations - the same env-step cost - against
  coin    SEMARL's fixed p_surr: f * 1 + (1 - f) * mean critic overlap
  oracle  the f*N generations where the critic is worst made real, in
          hindsight: the ceiling for any per-generation gate
`err` is the share of the 45 individual pairs the critic orders wrongly
(finer than elite_overlap, which moves in steps of 0.2 at pop_size 10).

The gate can only work if the critic's error at t predicts its error at
t+1..t+L; the detrended autocorrelation is reported first. Critic quality
also decays with training, and a gate that merely makes late generations
real would beat the coin too - so the final line also scores the gate with
err shuffled within blocks of 10 generations (trend kept, persistence
destroyed) and reports what is left beyond it. Parameters are
tuned on --tune dumps and scored on --test dumps, so the reported gain is
not fitted to the data it is reported on.

Optimistic in one way: the critic in a p_surr=0 dump was trained on every
generation's population rollout, so the replay ignores that surrogate
generations would starve its buffer.

    uv run python scripts/gate_benchmark.py --tune dumps/s*/population.npz \\
        --test dumps/ens/s*/population.npz
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from surrogate_benchmark import _elite_overlap

L_MAX = (1, 2, 4, 8, 16)
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)
LAGS = (1, 2, 4)
TREND_BLOCK = 10
N_PERMUTATIONS = 300


def _discordance(true: np.ndarray, est: np.ndarray) -> float:
    pairs = list(itertools.combinations(range(len(true)), 2))
    wrong = sum((true[i] - true[j]) * (est[i] - est[j]) < 0 for i, j in pairs)
    return wrong / len(pairs)


def _per_generation(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    real, critic = d["real"], d["critic"]
    parents = real.shape[1] // 2
    overlap = np.array(
        [_elite_overlap(r, c, parents) for r, c in zip(real, critic)]
    )
    err = np.array([_discordance(r, c) for r, c in zip(real, critic)])
    return overlap, err


def _detrended_autocorr(x: np.ndarray, lag: int) -> float:
    # critic quality decays as CEM converges; without the trend removed any
    # series looks predictable from its own past
    t = np.arange(len(x))
    resid = x - np.polyval(np.polyfit(t, x, 1), t)
    return float(np.corrcoef(resid[:-lag], resid[lag:])[0, 1])


def _replay(
    overlap: np.ndarray, err: np.ndarray, l_max: int, thr: float
) -> tuple[float, float]:
    # returns (fraction of real generations, mean elite_overlap)
    scores, n_real, t = [], 0, 0
    while t < len(overlap):
        scores.append(1.0)
        n_real += 1
        trust = int(l_max * max(0.0, (thr - err[t]) / thr))
        surrogate = overlap[t + 1 : t + 1 + trust]
        scores.extend(surrogate.tolist())
        t += 1 + len(surrogate)
    return n_real / len(overlap), float(np.mean(scores))


def _block_shuffle(
    x: np.ndarray, block: int, rng: np.random.Generator
) -> np.ndarray:
    y = x.copy()
    for start in range(0, len(y), block):
        rng.shuffle(y[start : start + block])
    return y


def _trend_null(
    runs: list[tuple[np.ndarray, np.ndarray]], l_max: int, thr: float
) -> np.ndarray:
    # err shuffled within blocks: the slow decay of critic quality survives,
    # the generation-to-generation persistence does not. Whatever the gate
    # still wins here, an increasing-p_real time schedule would win too.
    rng = np.random.default_rng(0)
    null = []
    for _ in range(N_PERMUTATIONS):
        gains = []
        for overlap, err in runs:
            shuffled = _block_shuffle(err, TREND_BLOCK, rng)
            frac, gate = _replay(overlap, shuffled, l_max, thr)
            gains.append(gate - _coin(overlap, frac))
        null.append(np.mean(gains))
    return np.asarray(null)


def _coin(overlap: np.ndarray, frac: float) -> float:
    return frac + (1.0 - frac) * float(overlap.mean())


def _oracle(overlap: np.ndarray, frac: float) -> float:
    n_real = round(frac * len(overlap))
    scores = np.sort(overlap)
    scores[:n_real] = 1.0
    return float(scores.mean())


def _grid(
    runs: list[tuple[np.ndarray, np.ndarray]],
) -> dict[tuple[int, float], list[tuple[float, float, float]]]:
    # per setting, per run: (real fraction, gate - coin, oracle - coin)
    out: dict[tuple[int, float], list[tuple[float, float, float]]] = {}
    for l_max, thr in itertools.product(L_MAX, THRESHOLDS):
        out[(l_max, thr)] = []
        for overlap, err in runs:
            frac, gate = _replay(overlap, err, l_max, thr)
            coin = _coin(overlap, frac)
            out[(l_max, thr)].append(
                (frac, gate - coin, _oracle(overlap, frac) - coin)
            )
    return out


def _report_autocorr(
    name: str, runs: list[tuple[np.ndarray, np.ndarray]]
) -> None:
    print(
        f"\n[{name}] detrended autocorrelation of the critic's per-generation quality"
    )
    print(
        f"  {'series':<14}"
        + "".join(f"{'lag ' + str(lag):>10}" for lag in LAGS)
    )
    for label, idx in (("elite_overlap", 0), ("pair error", 1)):
        cells = "".join(
            f"{np.mean([_detrended_autocorr(r[idx], lag) for r in runs]):>+10.3f}"
            for lag in LAGS
        )
        print(f"  {label:<14}{cells}")


def _report_grid(
    name: str, grid: dict[tuple[int, float], list[tuple[float, float, float]]]
) -> None:
    print(f"\n[{name}] gate vs coin at equal real fraction (mean over runs)")
    print(
        f"  {'L_max':>5}{'thr':>6}{'real frac':>11}{'gate-coin':>11}{'oracle-coin':>13}{'runs >0':>9}"
    )
    for (l_max, thr), rows in grid.items():
        frac, gain, ceil = (np.mean([r[i] for r in rows]) for i in range(3))
        wins = sum(r[1] > 0 for r in rows)
        print(
            f"  {l_max:>5}{thr:>6.1f}{frac:>11.2f}{gain:>+11.3f}"
            f"{ceil:>+13.3f}{wins:>6}/{len(rows)}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tune", nargs="+", type=Path, required=True)
    ap.add_argument("--test", nargs="+", type=Path, required=True)
    args = ap.parse_args()

    tune = [_per_generation(p) for p in args.tune]
    test = [_per_generation(p) for p in args.test]
    _report_autocorr("tune", tune)
    _report_autocorr("test", test)

    tune_grid = _grid(tune)
    _report_grid("tune", tune_grid)
    # only settings that actually skip generations; real frac 1.0 is a tie
    candidates = {
        k: v for k, v in tune_grid.items() if np.mean([r[0] for r in v]) < 0.95
    }
    best = max(candidates, key=lambda k: np.mean([r[1] for r in candidates[k]]))
    rows = _grid(test)[best]
    frac, gain, ceil = (np.mean([r[i] for r in rows]) for i in range(3))
    wins = sum(r[1] > 0 for r in rows)
    print(
        f"\nBest on tune: L_max={best[0]}, thr={best[1]}. On test: real frac "
        f"{frac:.2f}, gate-coin {gain:+.3f} ({wins}/{len(rows)} runs > 0), "
        f"oracle-coin {ceil:+.3f}"
    )
    null = _trend_null(test, *best)
    p_val = (np.sum(null >= gain) + 1) / (len(null) + 1)
    print(
        f"Trend-only null (err shuffled in blocks of {TREND_BLOCK}): "
        f"{null.mean():+.3f} ± {null.std():.3f}. Beyond trend: "
        f"{gain - null.mean():+.3f}, p = {p_val:.3f}"
    )


if __name__ == "__main__":
    main()
