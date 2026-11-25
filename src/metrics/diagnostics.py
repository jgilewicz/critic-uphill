from __future__ import annotations

import gymnasium as gym
import jax

from agents.base import RLAgent
from common.replay_buffer import ReplayBuffer
from config import EvoRainbowAlgoConfig, RunConfig
from metrics.degeneration import DegenerationProbe
from metrics.schedule import Schedule, metric_key
from metrics.truth import TruthProbe


class Diagnostics:
    """Schedules the degeneration probe (block 3) and truth probe (block 4)."""

    def __init__(self, run_cfg: RunConfig, env: gym.Env) -> None:
        self._seed = run_cfg.seed
        algo = run_cfg.algorithm
        shadow = isinstance(algo, EvoRainbowAlgoConfig) and algo.shadow.enabled
        self._deg = DegenerationProbe(run_cfg.deg, shadow=shadow)
        self._deg_schedule = Schedule(run_cfg.deg.interval)
        self._truth = (
            TruthProbe(env, run_cfg.probe) if run_cfg.probe.enabled else None
        )
        self._truth_schedule = Schedule(run_cfg.probe.interval)

    def measure_deg(
        self, agent: RLAgent, buffer: ReplayBuffer, env_steps: int
    ) -> dict[str, float]:
        """Measure deg/* now, regardless of the schedule."""
        deg_key, _ = jax.random.split(metric_key(self._seed, env_steps))
        return self._deg.measure(agent, buffer, deg_key)

    def maybe_measure(
        self, agent: RLAgent, buffer: ReplayBuffer, env_steps: int
    ) -> dict[str, float]:
        """Measure whichever probes crossed a new interval boundary."""
        deg_due = self._deg_schedule.due(env_steps)
        truth_due = self._truth_schedule.due(env_steps)
        if not (deg_due or (truth_due and self._truth is not None)):
            return {}
        deg_key, truth_key = jax.random.split(metric_key(self._seed, env_steps))
        metrics = self._deg.measure(agent, buffer, deg_key)
        sample = self._deg.last
        if truth_due and self._truth is not None and sample is not None:
            metrics |= self._truth.measure(agent, sample, truth_key)
        return metrics
