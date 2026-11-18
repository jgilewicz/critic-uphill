from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np

from agents.base import RLAgent
from common.replay_buffer import ReplayBuffer
from common.rollout import Budget, rollout_single_episode
from metrics.perf import PerfMetrics, budget_metrics

LogFn = Callable[[dict[str, float]], None]


def run_warmup(
    env: gym.Env,
    agent: RLAgent,
    buffer: ReplayBuffer,
    budget: Budget,
    *,
    warmup_steps: int,
    horizon: int,
    seed_rng: np.random.Generator,
    perf: PerfMetrics,
    on_log: LogFn | None,
) -> None:
    """Collect uniform-random RL episodes until warmup_steps or budget end."""
    limit = agent.action_limit
    shape = (agent.action_dim,)

    def uniform_act(_state: np.ndarray) -> np.ndarray:
        return seed_rng.uniform(-limit, limit, size=shape).astype(np.float32)

    while len(buffer) < warmup_steps and not budget.exhausted:
        max_steps = min(horizon, warmup_steps - len(buffer), budget.remaining)
        ep_seed = int(seed_rng.integers(0, 2**31 - 1))
        ep_return, _ = rollout_single_episode(
            env, uniform_act, buffer, budget, max_steps=max_steps, seed=ep_seed
        )
        perf.record_train_return(ep_return)
        eval_metrics = perf.maybe_evaluate(agent, budget.env_steps)
        if eval_metrics is not None and on_log is not None:
            on_log({**budget_metrics(budget, 0.0), **eval_metrics})


def print_summary(
    tag: str, budget: Budget, eval_ret: float | None = None
) -> None:
    """Print final budget breakdown to stdout."""
    eval_str = f"{eval_ret:.1f}" if eval_ret is not None else "n/a"
    print(
        f"done | {tag} | env_steps={budget.env_steps} | "
        f"rl_steps={budget.rl_steps} | pop_steps={budget.pop_steps} | "
        f"eval={eval_str}"
    )
