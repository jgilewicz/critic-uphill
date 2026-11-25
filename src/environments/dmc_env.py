from __future__ import annotations

import math
import os
import warnings
from typing import Any

import gymnasium as gym
import numpy as np

DMC_ENVS: dict[str, tuple[str, str]] = {
    "dmc-cheetah-run": ("cheetah", "run"),
    "dmc-quadruped-run": ("quadruped", "run"),
    "dmc-humanoid-run": ("humanoid", "run"),
    "dmc-dog-run": ("dog", "run"),
}


class DmcStateEnv(gym.Env):
    """Gymnasium view of a DeepMind Control suite task, state only.

    Observations are the sorted-key concatenation of the observation dict as
    one float32 vector. Actions are in [-1, 1] and map affinely onto the task's
    action bounds. Episodes end by truncation at the task's time limit.
    """

    def __init__(self, inner: gym.Env) -> None:
        self._inner = inner
        obs_space = inner.observation_space
        action_space = inner.action_space
        assert isinstance(obs_space, gym.spaces.Dict)
        assert isinstance(action_space, gym.spaces.Box)
        self._keys = tuple(sorted(obs_space.spaces))
        obs_dim = sum(
            math.prod(obs_space.spaces[key].shape or ()) for key in self._keys
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=action_space.shape, dtype=np.float32
        )
        self._low = action_space.low.astype(np.float64)
        self._high = action_space.high.astype(np.float64)

    def _flatten(self, obs: dict[str, Any]) -> np.ndarray:
        parts = [np.ravel(obs[key]) for key in self._keys]
        return np.concatenate(parts).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self._inner.reset(seed=seed, options=options)
        return self._flatten(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        scaled = self._low + (np.asarray(action) + 1.0) * 0.5 * (
            self._high - self._low
        )
        obs, reward, terminated, truncated, info = self._inner.step(scaled)
        if terminated:
            raise RuntimeError(
                "DMC task terminated early, expected truncation only"
            )
        return self._flatten(obs), float(reward), False, truncated, info

    def close(self) -> None:
        self._inner.close()


def make_dmc_env(env_id: str) -> gym.Env:
    """Instantiate a DMC suite task through shimmy, without a GL backend."""
    # dm_control imports a GL backend at import time unless MUJOCO_GL is set,
    # which fails or hangs on headless nodes. Observations are state only.
    os.environ.setdefault("MUJOCO_GL", "disable")
    from dm_control import suite
    from shimmy import DmControlCompatibilityV0

    # dm_control 1.0.38 sets ndarray.shape on every reset, deprecated in NumPy
    # 2.5. Installed after the imports, which reset the warning filters.
    warnings.filterwarnings(
        "ignore",
        message="Setting the shape on a NumPy array",
        category=DeprecationWarning,
    )
    domain, task = DMC_ENVS[env_id]
    return DmcStateEnv(DmControlCompatibilityV0(suite.load(domain, task)))
