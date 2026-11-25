from __future__ import annotations

from typing import Protocol

import jax


class StateEmbedding(Protocol):
    """Per-sample state encoder Z(s), shared by every actor head."""

    def __call__(self, state: jax.Array) -> jax.Array: ...


class QCritic(Protocol):
    """Per-sample critic: (state, action) -> Q of shape (1,)."""

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array: ...
