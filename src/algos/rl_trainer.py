from __future__ import annotations

import jax
import numpy as np

from agents import TD3
from common.checkpoint import make_checkpointer
from common.replay_buffer import ReplayBuffer
from common.reset import ResetController
from common.rollout import Budget, rollout_single_episode
from common.training import LogFn, print_summary, run_warmup
from config import RLAlgoConfig, RunConfig
from environments import env_dims, make_env
from metrics.diagnostics import Diagnostics
from metrics.perf import PerfMetrics, budget_metrics

_EVAL_SEED_OFFSET = 10_000


class RLTrainer:
    """Episode-by-episode training loop for any standalone RLAgent."""

    def __init__(
        self,
        run_cfg: RunConfig,
        algo_cfg: RLAlgoConfig,
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
        self._perf = PerfMetrics(
            run_cfg.eval_env.id,
            interval=run_cfg.eval.interval,
            episodes=run_cfg.eval.episodes,
            seed=run_cfg.seed + _EVAL_SEED_OFFSET,
            gamma=algo_cfg.rl.gamma,
        )
        self._diagnostics = Diagnostics(run_cfg, self._perf.env)
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
            shadow=algo_cfg.shadow.enabled,
        )
        self._resetter = self._make_resetter(algo_cfg)
        self._checkpointer = make_checkpointer(
            run_cfg, self._agent, self._buffer
        )

    def _make_resetter(self, algo_cfg: RLAlgoConfig) -> ResetController | None:
        if not algo_cfg.reset.enabled:
            return None
        return ResetController(
            algo_cfg.reset,
            self._agent,
            self._buffer,
            seed=self._run_cfg.seed,
            perf=self._perf,
            diagnostics=self._diagnostics,
            on_log=self._on_log,
        )

    def _next_key(self) -> jax.Array:
        self._key, sub = jax.random.split(self._key)
        return sub

    def _step_episode(self, episode: int) -> None:
        ep_seed = int(self._seed_rng.integers(0, 2**31 - 1))
        ep_return, ep_steps = rollout_single_episode(
            self._env,
            lambda s: self._agent.act(s, self._next_key()),
            self._buffer,
            self._budget,
            max_steps=self._algo.horizon,
            seed=ep_seed,
        )
        self._perf.record_train_return(ep_return)
        num_updates = int(self._algo.rl.train_ratio * ep_steps)
        losses = self._agent.update(self._buffer, num_updates, self._next_key())
        steps = self._budget.env_steps
        if self._resetter is not None:
            self._resetter.maybe_reset(steps)
        metrics = budget_metrics(self._budget, 0.0)
        metrics.update(losses)
        eval_metrics = self._perf.maybe_evaluate(self._agent, steps)
        if eval_metrics is not None:
            metrics.update(eval_metrics)
            print(
                f"ep {episode:4d} | steps {steps:7d} | "
                f"eval={eval_metrics['perf/eval_rl']:.1f} | "
                f"critic_loss={losses.get('critic/td_loss', 0.0):.4f}"
            )
        metrics.update(
            self._diagnostics.maybe_measure(self._agent, self._buffer, steps)
        )
        if self._on_log is not None:
            self._on_log(metrics)
        if self._resetter is not None:
            self._resetter.after_iteration(steps, eval_metrics)
        if self._checkpointer is not None:
            self._checkpointer.maybe_save(steps)

    def train(self) -> float:
        """Run warmup and episode-by-episode RL until the budget ends."""
        try:
            run_warmup(
                self._env,
                self._agent,
                self._buffer,
                self._budget,
                warmup_steps=self._algo.rl.warmup_steps,
                horizon=self._algo.horizon,
                seed_rng=self._seed_rng,
                perf=self._perf,
                on_log=self._on_log,
            )
            episode = 0
            while not self._budget.exhausted:
                self._step_episode(episode)
                episode += 1
            if self._resetter is not None:
                self._resetter.finish(self._budget.env_steps)
            if self._checkpointer is not None:
                self._checkpointer.finish(self._budget.env_steps)
            final_eval = self._perf.evaluate(
                self._agent, env_steps=self._budget.env_steps
            ).rl_return
            print_summary(self._algo.name, self._budget, final_eval)
            return final_eval
        finally:
            self._env.close()
            self._perf.close()
