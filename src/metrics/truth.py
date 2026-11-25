from __future__ import annotations

from typing import Any

import gymnasium as gym
import jax
import numpy as np

from agents.base import OfflineAgent
from common.rollout import evaluation_episode
from config import ProbeConfig
from metrics.degeneration import DegenerationSample


class TruthProbe:
    """Block 4: real paired episodes for the critic's pick vs random picks.

    With shadow terms in the sample it pairs the top-1 picks of Q1, Q2, the
    shadow critic and the nearest head against random picks instead.

    Episodes run on the evaluation env and never touch the step budget.
    """

    def __init__(self, env: gym.Env, cfg: ProbeConfig) -> None:
        self._env = env
        self._cfg = cfg

    def _returns(
        self, agent: OfflineAgent, head: Any, seeds: np.ndarray
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

    def _gains(
        self,
        agent: OfflineAgent,
        sample: DegenerationSample,
        picks: list[int],
        seeds: np.ndarray,
    ) -> np.ndarray:
        actor_returns = self._returns(agent, agent.actor_head, seeds)
        return np.array(
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

    def measure(
        self, agent: OfflineAgent, sample: DegenerationSample, key: jax.Array
    ) -> dict[str, float]:
        """Top-1 by dS plus uniform others, paired with the actor."""
        rng = np.random.default_rng(
            int(jax.random.randint(key, (), 0, 2**31 - 1))
        )
        if sample.delta_q2 is not None and sample.delta_shadow is not None:
            return self._measure_shadow(
                agent, sample, sample.delta_q2, sample.delta_shadow, rng
            )
        cfg = self._cfg
        top = int(np.argmax(sample.delta_s))
        others = np.delete(np.arange(len(sample.delta_s)), top)
        picks = [
            top,
            *rng.choice(others, cfg.n_perturbations - 1, replace=False),
        ]
        seeds = rng.integers(0, 2**31 - 1, size=cfg.episodes)
        gains = self._gains(agent, sample, picks, seeds)
        return {
            "probe/top1_true_gain": float(gains[0]),
            "probe/frac_true_better": float(np.mean(gains > 0.0)),
        }

    def _measure_shadow(
        self,
        agent: OfflineAgent,
        sample: DegenerationSample,
        delta_q2: np.ndarray,
        delta_shadow: np.ndarray,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """Top-1 by Q1, Q2, shadow and nearest (max Q1 t2), random fill."""
        cfg = self._cfg
        named = {
            "q1": int(np.argmax(sample.delta_s)),
            "q2": int(np.argmax(delta_q2)),
            "shadow": int(np.argmax(delta_shadow)),
            "nearest": int(np.argmax(sample.t2)),
        }
        chosen = list(dict.fromkeys(named.values()))
        others = np.setdiff1d(np.arange(len(sample.delta_s)), chosen)
        n_random = min(cfg.n_perturbations - len(chosen), len(others))
        randoms = [int(i) for i in rng.choice(others, n_random, replace=False)]
        seeds = rng.integers(0, 2**31 - 1, size=cfg.episodes)
        picks = chosen + randoms
        gains = dict(
            zip(picks, self._gains(agent, sample, picks, seeds), strict=True)
        )
        metrics = {
            f"probe/gain_{name}": float(gains[idx])
            for name, idx in named.items()
        }
        metrics |= {
            f"probe/pick_{name}": float(idx) for name, idx in named.items()
        }
        metrics["probe/gain_random"] = float(
            np.mean([gains[i] for i in randoms])
        )
        metrics["probe/top1_true_gain"] = metrics["probe/gain_q1"]
        metrics["probe/frac_true_better"] = float(
            np.mean([g > 0.0 for g in gains.values()])
        )
        return metrics
