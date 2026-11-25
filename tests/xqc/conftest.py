from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from config import XQCConfig
from xqc.agent import XQCAgent

OBS_DIM = 11
ACTION_DIM = 3
BUFFER_ROWS = 512


def random_buffer(key: jax.Array) -> tuple[jax.Array, ...]:
    """Synthetic transitions with policy-scale actions in [-1, 1]."""
    k_s, k_a, k_r, k_n, k_d = jax.random.split(key, 5)
    return (
        jax.random.normal(k_s, (BUFFER_ROWS, OBS_DIM)),
        jax.random.uniform(k_a, (BUFFER_ROWS, ACTION_DIM), minval=-1, maxval=1),
        jax.random.normal(k_r, (BUFFER_ROWS, 1)),
        jax.random.normal(k_n, (BUFFER_ROWS, OBS_DIM)),
        (jax.random.uniform(k_d, (BUFFER_ROWS, 1)) < 0.1).astype(jnp.float32),
    )


def make_trained_agent(
    action_limit: float = 1.0, cfg: XQCConfig | None = None, updates: int = 3
) -> XQCAgent:
    """Agent after a few updates, so BatchNorm statistics are non-trivial."""
    agent = XQCAgent(
        OBS_DIM,
        ACTION_DIM,
        action_limit,
        cfg=cfg or XQCConfig(),
        total_steps=1000,
        seed=0,
    )
    data = random_buffer(jax.random.key(1))
    for _ in range(updates):
        agent.update(data, BUFFER_ROWS, 2.0)
    return agent


@pytest.fixture(scope="session")
def trained_agent() -> XQCAgent:
    return make_trained_agent()
