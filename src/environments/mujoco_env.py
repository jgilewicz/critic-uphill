from __future__ import annotations

import gymnasium as gym

ALLOWED_ENVS: tuple[str, ...] = (
    "HalfCheetah-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Ant-v5",
    "Swimmer-v5",
)


def make_mujoco_env(task_id: str) -> gym.Env:
    """Instantiate a MuJoCo v5 gymnasium environment."""
    return gym.make(task_id)
