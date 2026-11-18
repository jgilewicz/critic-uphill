from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np

from agents.td3 import TD3
from common.replay_buffer import ReplayBuffer
from common.rollout import Budget, rollout_single_episode
from environments import make_env

LogFn = Callable[[dict[str, float]], None]


class PerfEvaluator:
    """Periodic deterministic evaluation outside the training step budget."""

    def __init__(
        self,
        env_id: str,
        *,
        interval: int,
        episodes: int,
        seed: int,
    ) -> None:
        if interval < 1:
            raise ValueError(f"eval.interval must be >= 1, got {interval}")
        if episodes < 1:
            raise ValueError(f"eval.episodes must be >= 1, got {episodes}")
        self.env = make_env(env_id)
        self.interval = interval
        self.episodes = episodes
        self.seed = seed
        self._last_bucket = 0

    def maybe_evaluate(self, agent: TD3, env_steps: int) -> float | None:
        """Evaluate if env_steps crossed a new eval.interval boundary."""
        bucket = env_steps // self.interval
        if bucket <= self._last_bucket:
            return None
        self._last_bucket = bucket
        return self.evaluate(agent)

    def evaluate(self, agent: TD3) -> float:
        """Run deterministic episodes and return mean episode reward."""
        total_reward = 0.0
        for i in range(self.episodes):
            obs_raw, _ = self.env.reset(seed=self.seed if i == 0 else None)
            state = np.asarray(obs_raw, dtype=np.float32)
            done = False
            while not done:
                action = agent.act_deterministic(state)
                obs_raw, reward, terminated, truncated, _ = self.env.step(
                    action
                )
                state = np.asarray(obs_raw, dtype=np.float32)
                total_reward += float(reward)
                done = bool(terminated or truncated)
        return total_reward / self.episodes

    def close(self) -> None:
        """Close the evaluation environment."""
        self.env.close()


def run_warmup(
    env: gym.Env,
    agent: TD3,
    buffer: ReplayBuffer,
    budget: Budget,
    *,
    warmup_steps: int,
    horizon: int,
    seed_rng: np.random.Generator,
    evaluator: PerfEvaluator,
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
        rollout_single_episode(
            env, uniform_act, buffer, budget, max_steps=max_steps, seed=ep_seed
        )
        eval_ret = evaluator.maybe_evaluate(agent, budget.env_steps)
        if eval_ret is not None and on_log is not None:
            on_log(
                {
                    "env_steps": float(budget.env_steps),
                    "cost/pop_steps": float(budget.pop_steps),
                    "perf/eval_rl": eval_ret,
                }
            )


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
