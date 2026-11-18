from __future__ import annotations

from typing import Any

import gymnasium as gym
import jax
import numpy as np

from agents.base import RLAgent
from common.rollout import evaluation_episode
from config import ProbeConfig
from metrics.degeneration import DegenerationSample


class TruthProbe:
    """Block 4: real paired episodes for the critic's pick vs random picks.

    Episodes run on the evaluation env and never touch the step budget.
    """

    def __init__(self, env: gym.Env, cfg: ProbeConfig) -> None:
        self._env = env
        self._cfg = cfg

    def _returns(
        self, agent: RLAgent, head: Any, seeds: np.ndarray
    ) -> np.ndarray:
        return np.array(
            [
                evaluation_episode(
                    self._env,
                    lambda s: agent.act_with_head(head, s),
                    seed=int(seed),
                    gamma=1.0,
                ).ret
                for seed in seeds
            ]
        )

    def measure(
        self, agent: RLAgent, sample: DegenerationSample, key: jax.Array
    ) -> dict[str, float]:
        """Top-1 by dS plus uniform others, paired with the actor."""
        cfg = self._cfg
        rng = np.random.default_rng(
            int(jax.random.randint(key, (), 0, 2**31 - 1))
        )
        top = int(np.argmax(sample.delta_s))
        others = np.delete(np.arange(len(sample.delta_s)), top)
        picks = [
            top,
            *rng.choice(others, cfg.n_perturbations - 1, replace=False),
        ]
        seeds = rng.integers(0, 2**31 - 1, size=cfg.episodes)
        actor_returns = self._returns(agent, agent.actor_head, seeds)
        gains = np.array(
            [
                np.mean(
                    self._returns(
                        agent,
                        jax.tree.map(lambda x, i=i: x[i], sample.heads),
                        seeds,
                    )
                    - actor_returns
                )
                for i in picks
            ]
        )
        return {
            "probe/top1_true_gain": float(gains[0]),
            "probe/frac_true_better": float(np.mean(gains > 0.0)),
        }
