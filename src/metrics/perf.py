from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from agents.base import RLAgent
from common.rollout import Budget, evaluation_episode
from environments import make_env
from metrics.schedule import Schedule
from modules.deep_modules import SharedStateEmbedding


class Champion(NamedTuple):
    """Best individual with the embedding it was scored on."""

    embedding: SharedStateEmbedding
    head: Any


@dataclass(frozen=True)
class EvalResult:
    """Deterministic evaluation of the RL actor and optional champion."""

    rl_return: float
    rl_std: float
    overestimation: float
    champion_return: float | None = None

    def as_metrics(self) -> dict[str, float]:
        """Flatten into logged perf/* and critic/* keys."""
        metrics = {
            "perf/eval_rl": self.rl_return,
            "perf/eval_rl_std": self.rl_std,
            "critic/overestimation": self.overestimation,
        }
        if self.champion_return is not None:
            metrics["perf/eval_champion"] = self.champion_return
            metrics["perf/champion_beats_actor"] = float(
                self.champion_return > self.rl_return
            )
        return metrics


def budget_metrics(budget: Budget, real_frac: float) -> dict[str, float]:
    """Step accounting keys logged at every iteration."""
    return {
        "env_steps": float(budget.env_steps),
        "budget/steps_total": float(budget.env_steps),
        "budget/steps_rl": float(budget.rl_steps),
        "budget/steps_pop": float(budget.pop_steps),
        "budget/real_frac": real_frac,
    }


class PerfMetrics:
    """Periodic deterministic evaluation outside the training step budget."""

    def __init__(
        self,
        env_id: str,
        *,
        interval: int,
        episodes: int,
        seed: int,
        gamma: float,
    ) -> None:
        if episodes < 1:
            raise ValueError(f"eval.episodes must be >= 1, got {episodes}")
        self.env = make_env(env_id)
        self.episodes = episodes
        self.seed = seed
        self._gamma = gamma
        self._schedule = Schedule(interval)
        self._last_eval: tuple[int, EvalResult] | None = None
        self._train_returns: list[float] = []

    def record_train_return(self, value: float) -> None:
        """Store the return of one finished RL training episode."""
        self._train_returns.append(value)

    def maybe_evaluate(
        self,
        agent: RLAgent,
        env_steps: int,
        *,
        champion: Champion | None = None,
    ) -> dict[str, float] | None:
        """Evaluate if env_steps crossed a new eval.interval boundary."""
        if not self._schedule.due(env_steps):
            return None
        result = self.evaluate(agent, champion=champion, env_steps=env_steps)
        metrics = result.as_metrics()
        if self._train_returns:
            metrics["perf/train_return"] = float(np.mean(self._train_returns))
            self._train_returns = []
        return metrics

    def evaluate(
        self,
        agent: RLAgent,
        *,
        champion: Champion | None = None,
        env_steps: int | None = None,
    ) -> EvalResult:
        """Run deterministic episodes on shared reset seeds."""
        if (
            env_steps is not None
            and self._last_eval is not None
            and self._last_eval[0] == env_steps
        ):
            return self._last_eval[1]
        seeds = [self.seed + i for i in range(self.episodes)]
        runs = [
            evaluation_episode(
                self.env, agent.act_deterministic, seed=s, gamma=self._gamma
            )
            for s in seeds
        ]
        returns = np.array([r.ret for r in runs])
        overestimation = np.mean(
            [agent.q1_at_actor(r.first_state) - r.discounted_ret for r in runs]
        )
        result = EvalResult(
            rl_return=float(returns.mean()),
            rl_std=float(returns.std()),
            overestimation=float(overestimation),
            champion_return=(
                None
                if champion is None
                else self._mean_return(agent, champion, seeds)
            ),
        )
        if env_steps is not None:
            self._last_eval = (env_steps, result)
        return result

    def _mean_return(
        self, agent: RLAgent, champion: Champion, seeds: list[int]
    ) -> float:
        returns = [
            evaluation_episode(
                self.env,
                lambda s: agent.act_with_head(
                    champion.head, s, embedding=champion.embedding
                ),
                seed=seed,
                gamma=self._gamma,
            ).ret
            for seed in seeds
        ]
        return float(np.mean(returns))

    def close(self) -> None:
        """Close the evaluation environment."""
        self.env.close()
