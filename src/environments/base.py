from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from environments.mujoco_env import ALLOWED_ENVS, make_mujoco_env


@dataclass(frozen=True)
class EnvDims:
    """Observation, action, and horizon metadata for an environment."""

    obs_dim: int
    action_dim: int
    action_limit: float
    horizon: int


def validate_env_id(env_id: str) -> None:
    """Raise ValueError unless env_id is one of the supported MuJoCo v5 tasks."""
    if env_id not in ALLOWED_ENVS:
        allowed = ", ".join(ALLOWED_ENVS)
        raise ValueError(f"Unsupported env {env_id!r}; choose from: {allowed}")


def make_env(env_id: str) -> gym.Env:
    """Create a single validated MuJoCo v5 environment."""
    validate_env_id(env_id)
    env = make_mujoco_env(env_id)
    _check_spaces(env, env_id)
    return env


def env_dims(env_id: str, *, horizon: int | None = None) -> EnvDims:
    """Inspect observation, action, and episode horizon dimensions."""
    env = make_env(env_id)
    try:
        obs_space = env.observation_space
        act_space = env.action_space
        assert isinstance(obs_space, gym.spaces.Box)
        assert isinstance(act_space, gym.spaces.Box)
        assert obs_space.shape is not None and act_space.shape is not None
        max_steps = env.spec.max_episode_steps if env.spec is not None else None
        resolved = (
            horizon
            if horizon is not None
            else (int(max_steps) if max_steps is not None else 1000)
        )
        return EnvDims(
            obs_dim=int(obs_space.shape[0]),
            action_dim=int(act_space.shape[0]),
            action_limit=float(act_space.high[0]),
            horizon=resolved,
        )
    finally:
        env.close()


class EnvPool:
    """Fixed-size pool of environments stepped only while active."""

    def __init__(self, env_id: str, num_envs: int) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        validate_env_id(env_id)
        self.env_id = env_id
        self.num_envs = num_envs
        self._envs = [make_env(env_id) for _ in range(num_envs)]

    def reset(self, seed: int) -> np.ndarray:
        """Reset all environments with deterministic per-slot seeds."""
        states = [
            np.asarray(env.reset(seed=seed + i)[0], dtype=np.float32)
            for i, env in enumerate(self._envs)
        ]
        return np.stack(states, axis=0)

    def step(
        self, indices: Sequence[int] | np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step only the environments at indices; never auto-resets."""
        next_states: list[np.ndarray] = []
        rewards: list[float] = []
        terminateds: list[bool] = []
        truncateds: list[bool] = []
        for idx, action in zip(indices, actions, strict=True):
            obs, rew, term, trunc, _ = self._envs[int(idx)].step(action)
            next_states.append(np.asarray(obs, dtype=np.float32))
            rewards.append(float(rew))
            terminateds.append(bool(term))
            truncateds.append(bool(trunc))
        return (
            np.stack(next_states, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminateds, dtype=bool),
            np.asarray(truncateds, dtype=bool),
        )

    def close(self) -> None:
        """Close every environment in the pool."""
        for env in self._envs:
            env.close()


def make_vec_env(env_id: str, num_envs: int) -> EnvPool:
    """Build an EnvPool of num_envs copies of env_id."""
    return EnvPool(env_id, num_envs)


def _check_spaces(env: gym.Env, env_id: str) -> None:
    obs, act = env.observation_space, env.action_space
    if (
        not isinstance(obs, gym.spaces.Box)
        or obs.shape is None
        or len(obs.shape) != 1
    ):
        raise TypeError(f"{env_id}: expected 1-D Box obs, got {obs}")
    if (
        not isinstance(act, gym.spaces.Box)
        or act.shape is None
        or len(act.shape) != 1
    ):
        raise TypeError(f"{env_id}: expected 1-D Box action, got {act}")
