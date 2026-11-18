from __future__ import annotations

import jax
import numpy as np

from agents.td3 import TD3
from common.replay_buffer import ReplayBuffer
from common.rollout import Budget, rollout_single_episode
from common.training import LogFn, PerfEvaluator, print_summary, run_warmup
from config import RunConfig, TD3AlgoConfig
from environments import env_dims, make_env

_EVAL_SEED_OFFSET = 10_000


class TD3Trainer:
    """Training loop for standalone TD3 matching EvoRainbow's RL side."""

    def __init__(
        self,
        run_cfg: RunConfig,
        algo_cfg: TD3AlgoConfig,
        *,
        on_log: LogFn | None = None,
    ) -> None:
        self._run_cfg = run_cfg
        self._algo = algo_cfg
        self._on_log = on_log

        dims = env_dims(run_cfg.env.id)
        self._key = jax.random.key(run_cfg.seed)
        self._seed_rng = np.random.default_rng(run_cfg.seed)
        self._budget = Budget(run_cfg.total_steps)
        self._env = make_env(run_cfg.env.id)
        self._evaluator = PerfEvaluator(
            run_cfg.eval_env.id,
            interval=run_cfg.eval.interval,
            episodes=run_cfg.eval.episodes,
            seed=run_cfg.seed + _EVAL_SEED_OFFSET,
        )
        self._buffer = ReplayBuffer(
            algo_cfg.rl.buffer_capacity, dims.obs_dim, dims.action_dim
        )
        self._agent = TD3(
            dims.obs_dim,
            dims.action_dim,
            dims.action_limit,
            algo_cfg.embedding_dim,
            rl_cfg=algo_cfg.rl,
            key=self._next_key(),
        )

    def _next_key(self) -> jax.Array:
        self._key, sub = jax.random.split(self._key)
        return sub

    def _step_episode(self, episode: int) -> None:
        ep_seed = int(self._seed_rng.integers(0, 2**31 - 1))
        _, ep_steps = rollout_single_episode(
            self._env,
            lambda s: self._agent.act(s, self._next_key()),
            self._buffer,
            self._budget,
            max_steps=self._algo.horizon,
            seed=ep_seed,
        )
        num_updates = int(self._algo.rl.train_ratio * ep_steps)
        critic_loss = self._agent.update(
            self._buffer, num_updates, self._next_key()
        )
        eval_ret = self._evaluator.maybe_evaluate(
            self._agent, self._budget.env_steps
        )
        metrics: dict[str, float] = {
            "env_steps": float(self._budget.env_steps),
            "cost/pop_steps": float(self._budget.pop_steps),
        }
        if num_updates > 0:
            metrics["train/critic_loss"] = critic_loss
        if eval_ret is not None:
            metrics["perf/eval_rl"] = eval_ret
            print(
                f"ep {episode:4d} | steps {self._budget.env_steps:7d} | "
                f"eval={eval_ret:.1f} | critic_loss={critic_loss:.4f}"
            )
        if self._on_log is not None:
            self._on_log(metrics)

    def train(self) -> float:
        """Run warmup and episode-by-episode TD3 until the budget ends."""
        try:
            run_warmup(
                self._env,
                self._agent,
                self._buffer,
                self._budget,
                warmup_steps=self._algo.rl.warmup_steps,
                horizon=self._algo.horizon,
                seed_rng=self._seed_rng,
                evaluator=self._evaluator,
                on_log=self._on_log,
            )
            episode = 0
            while not self._budget.exhausted:
                self._step_episode(episode)
                episode += 1
            final_eval = self._evaluator.evaluate(
                self._agent, env_steps=self._budget.env_steps
            )
            print_summary("td3", self._budget, final_eval)
            return final_eval
        finally:
            self._env.close()
            self._evaluator.close()
