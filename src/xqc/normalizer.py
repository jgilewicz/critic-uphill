from __future__ import annotations

import numpy as np

_SCALE_EPS = 1e-8


class RewardNormalizer:
    """Running scale of the discounted return, as in the original XQC loop.

    Rewards are divided by sqrt(var) + 1e-8 of the discounted return
    G = gamma * (1 - done) * G + r, tracked in float64 with parallel moments.
    """

    def __init__(self, gamma: float) -> None:
        self._gamma = gamma
        self._ret = 0.0
        self._mean = 0.0
        self._var = 1.0
        self._count = 0.0

    @property
    def count(self) -> int:
        return int(self._count)

    @property
    def var(self) -> float:
        return self._var

    @property
    def scale(self) -> float:
        return float(np.sqrt(self._var) + _SCALE_EPS)

    def update(self, reward: float, done: bool) -> None:
        """Fold one environment step into the return statistics."""
        self._ret = self._gamma * (1.0 - float(done)) * self._ret + reward
        self._merge(self._ret)

    def _merge(self, value: float) -> None:
        total = self._count + 1.0
        delta = value - self._mean
        self._mean += delta / total
        # batch of one: its own variance is zero, so only the mean shift stays
        m2 = self._var * self._count + delta**2 * self._count / total
        self._var = m2 / total
        self._count = total
