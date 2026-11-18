import numpy as np

N_BOOT = 10_000
# env -> condition -> per-seed scores; seeds are not paired across conditions
Cells = dict[str, dict[str, np.ndarray]]


def iqm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    lo, hi = np.percentile(x, [25, 75], axis=axis, keepdims=True)
    inside = (x >= lo) & (x <= hi)
    return np.sum(np.where(inside, x, 0), axis=axis) / inside.sum(axis=axis)


def normalize(cells: Cells) -> Cells:
    # per env: 0 = worst condition mean, 1 = best condition mean
    out: Cells = {}
    for env, by_cond in cells.items():
        means = [v.mean() for v in by_cond.values()]
        lo, hi = min(means), max(means)
        if hi == lo:
            raise ValueError(
                f"{env}: every condition has mean {lo}, cannot normalize. "
                "drop the env or check the scores"
            )
        out[env] = {c: (v - lo) / (hi - lo) for c, v in by_cond.items()}
    return out


def pooled(cells: Cells, cond: str) -> np.ndarray:
    return np.concatenate([by_cond[cond] for by_cond in cells.values()])


def boot_iqm(
    cells: Cells, conds: list[str], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    # stratified by env, seeds resampled independently per (env, condition)
    out = {}
    for c in conds:
        draws = []
        for by_cond in cells.values():
            v = by_cond[c]
            idx = rng.integers(0, len(v), (N_BOOT, len(v)))
            draws.append(v[idx])
        out[c] = iqm(np.concatenate(draws, axis=1))
    return out


def ci(draws: np.ndarray, level: float = 0.95) -> list[float]:
    tail = (1 - level) / 2 * 100
    return np.percentile(draws, [tail, 100 - tail]).tolist()


def prob_greater(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    gt = (a[..., :, None] > b[..., None, :]).mean(axis=(-1, -2))
    eq = (a[..., :, None] == b[..., None, :]).mean(axis=(-1, -2))
    return gt + 0.5 * eq


def prob_greater_ci(
    cells: Cells, a: str, b: str, rng: np.random.Generator
) -> dict[str, float | list[float]]:
    # P(A > B) for a random seed pair within an env, averaged over envs
    point = np.mean([prob_greater(d[a], d[b]) for d in cells.values()])
    draws = []
    for d in cells.values():
        va, vb = d[a], d[b]
        ia = rng.integers(0, len(va), (N_BOOT, len(va)))
        ib = rng.integers(0, len(vb), (N_BOOT, len(vb)))
        draws.append(prob_greater(va[ia], vb[ib]))
    return {"p": float(point), "ci": ci(np.mean(draws, axis=0))}


def p_superiority(diff_draws: np.ndarray) -> float:
    # two-sided percentile-bootstrap p for H0: diff = 0
    below = (1 + np.sum(diff_draws <= 0)) / (1 + len(diff_draws))
    above = (1 + np.sum(diff_draws >= 0)) / (1 + len(diff_draws))
    return float(min(1.0, 2 * min(below, above)))


def p_equivalence(diff_draws: np.ndarray, margin: float) -> float:
    # TOST: both one-sided tests must reject |diff| >= margin
    n = len(diff_draws)
    upper = (1 + np.sum(diff_draws >= margin)) / (1 + n)
    lower = (1 + np.sum(diff_draws <= -margin)) / (1 + n)
    return float(max(upper, lower))


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()
