from __future__ import annotations

import numpy as np

N_BOOT = 10_000
# seed -> values pooled over checkpoints or pairs of that seed
SeedValues = dict[int, list[float]]


def seed_bootstrap(
    by_seed: SeedValues, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Median of pooled values and its 95% percentile CI over seeds."""
    seeds = [s for s, v in by_seed.items() if np.isfinite(v).any()]
    if not seeds:
        return np.nan, np.nan, np.nan
    pooled = np.concatenate([by_seed[s] for s in seeds])
    point = float(np.nanmedian(pooled))
    stats = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.choice(seeds, len(seeds), replace=True)
        stats[b] = np.nanmedian(np.concatenate([by_seed[s] for s in pick]))
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation with average ranks for ties."""

    def ranks(v: np.ndarray) -> np.ndarray:
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        ends = np.cumsum(counts)
        return ((2 * ends - counts - 1) / 2.0)[inv.reshape(-1)]

    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    return float(xc @ (y - y.mean()) / (xc @ xc))


def iqm(x: np.ndarray) -> float:
    """Interquartile mean: mean of the middle 50% of sorted values."""
    x = np.sort(np.asarray(x, dtype=float))
    cut = int(0.25 * len(x))
    return float(x[cut : len(x) - cut].mean())


def resample(strata: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """N_BOOT stratified resamples, seeds drawn independently per stratum."""
    idx = [rng.integers(0, len(x), (N_BOOT, len(x))) for x in strata]
    return np.concatenate([x[i] for x, i in zip(strata, idx)], axis=1)


def boot_iqm(strata: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    draws = np.sort(resample(strata, rng), axis=1)
    cut = int(0.25 * draws.shape[1])
    return draws[:, cut : draws.shape[1] - cut].mean(axis=1)


def ci(point: float, draws: np.ndarray) -> dict[str, float]:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi)}


def iqm_ci(strata: list[np.ndarray], rng: np.random.Generator) -> dict:
    out: dict = ci(iqm(np.concatenate(strata)), boot_iqm(strata, rng))
    out["n"] = int(sum(len(x) for x in strata))
    return out


def iqm_diff(
    a: list[np.ndarray],
    b: list[np.ndarray],
    rng: np.random.Generator,
    family_alpha: float,
) -> dict[str, float]:
    """IQM(a) - IQM(b), bootstrap CI, two-sided p, Bonferroni CI.

    family_alpha is alpha / number of tests, so lo_bonf/hi_bonf bound the
    Holm-adjusted interval from outside.
    """
    point = iqm(np.concatenate(a)) - iqm(np.concatenate(b))
    d = boot_iqm(a, rng) - boot_iqm(b, rng)
    below = (np.sum(d <= 0) + 1) / (N_BOOT + 1)
    above = (np.sum(d >= 0) + 1) / (N_BOOT + 1)
    out = ci(point, d)
    out["p"] = float(min(1.0, 2 * min(below, above)))
    tail = 100 * family_alpha / 2
    lo, hi = np.percentile(d, [tail, 100 - tail])
    out["lo_bonf"], out["hi_bonf"] = float(lo), float(hi)
    return out


def holm(ps: list[float], alpha: float) -> list[bool]:
    order = np.argsort(ps)
    reject = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] > alpha / (len(ps) - rank):
            break
        reject[i] = True
    return reject


def prob_improvement(
    a: list[np.ndarray], b: list[np.ndarray], rng: np.random.Generator
) -> dict[str, float]:
    """P(a run of A beats a run of B), averaged over strata, with CI."""

    def p_of(xa: list[np.ndarray], xb: list[np.ndarray]) -> np.ndarray:
        per_env = []
        for x, y in zip(xa, xb):
            gt = (x[..., :, None] > y[..., None, :]).mean(axis=(-1, -2))
            eq = (x[..., :, None] == y[..., None, :]).mean(axis=(-1, -2))
            per_env.append(gt + 0.5 * eq)
        return np.mean(per_env, axis=0)

    point = float(p_of(a, b))
    da = [x[rng.integers(0, len(x), (N_BOOT, len(x)))] for x in a]
    db = [y[rng.integers(0, len(y), (N_BOOT, len(y)))] for y in b]
    return ci(point, p_of(da, db))


def mean_ci(x: np.ndarray, rng: np.random.Generator) -> dict:
    draws = x[rng.integers(0, len(x), (N_BOOT, len(x)))]
    out: dict = ci(float(x.mean()), draws.mean(axis=1))
    out["n"] = int(len(x))
    return out
