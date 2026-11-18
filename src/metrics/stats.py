from __future__ import annotations

import numpy as np


def _ranks(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(
        values, return_inverse=True, return_counts=True
    )
    ends = np.cumsum(counts)
    starts = ends - counts
    return ((starts + ends - 1) / 2.0)[inverse.reshape(-1)]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation with average ranks, NaN if an input is flat."""
    rank_x, rank_y = _ranks(x), _ranks(y)
    if rank_x.std() == 0.0 or rank_y.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(rank_x, rank_y)[0, 1])
