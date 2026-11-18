from __future__ import annotations

import jax

_METRIC_STREAM = 7919


class Schedule:
    """Fires once each time env_steps crosses a new multiple of interval."""

    def __init__(self, interval: int) -> None:
        if interval < 1:
            raise ValueError(f"interval must be >= 1, got {interval}")
        self.interval = interval
        self._last_bucket = 0

    def due(self, env_steps: int) -> bool:
        """Return True once per newly crossed interval boundary."""
        bucket = env_steps // self.interval
        if bucket <= self._last_bucket:
            return False
        self._last_bucket = bucket
        return True


def metric_key(seed: int, env_steps: int) -> jax.Array:
    """PRNG key for metrics, disjoint from the training key stream."""
    base = jax.random.fold_in(jax.random.key(seed), _METRIC_STREAM)
    return jax.random.fold_in(base, env_steps)
