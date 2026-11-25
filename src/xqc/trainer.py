from __future__ import annotations

import jax
import numpy as np

from common.replay_buffer import ReplayBuffer, Transition
from common.rollout import Budget
from common.training import LogFn, print_summary
from config import RunConfig, XQCConfig
from environments import env_dims, make_env
from metrics.perf import PerfMetrics, budget_metrics
from xqc.agent import XQCAgent
from xqc.checkpoint import make_xqc_checkpointer
from xqc.normalizer import RewardNormalizer

_EVAL_SEED_OFFSET = 10_000
_ACT_STREAM = 7


class XQCTrainer:
    """Step-by-step XQC loop: one env step, then updates_per_step updates."""

    def __init__(
        self,
        run_cfg: RunConfig,
        algo_cfg: XQCConfig,
        *,
        on_log: LogFn | None = None,
    ) -> None:
        self._run_cfg = run_cfg
        self._algo = algo_cfg
        self._on_log = on_log
        dims = env_dims(run_cfg.env.id)
        self._rng = np.random.default_rng(run_cfg.seed)
        self._act_key = jax.random.fold_in(
            jax.random.key(run_cfg.seed), _ACT_STREAM
        )
        self._budget = Budget(run_cfg.total_steps)
        self._env = make_env(run_cfg.env.id)
        self._buffer = ReplayBuffer(
            algo_cfg.buffer_capacity, dims.obs_dim, dims.action_dim
        )
        self._agent = XQCAgent(
            dims.obs_dim,
            dims.action_dim,
            dims.action_limit,
            cfg=algo_cfg,
            total_steps=run_cfg.total_steps,
            seed=run_cfg.seed,
        )
        self._normalizer = RewardNormalizer(self._agent.gamma)
        self._perf = PerfMetrics(
            run_cfg.eval_env.id,
            interval=run_cfg.eval.interval,
            episodes=run_cfg.eval.episodes,
            seed=run_cfg.seed + _EVAL_SEED_OFFSET,
            gamma=self._agent.gamma,
        )
        self._checkpointer = make_xqc_checkpointer(
            run_cfg, self._agent, self._buffer, self._normalizer
        )
        self._stats: dict[str, float] = {}

    def _policy_action(self, state: np.ndarray, step: int) -> np.ndarray:
        """Policy-scale action: uniform random before start_training."""
        # the original counts steps from 1: random while i < start_training
        if step + 1 < self._algo.start_training:
            shape = (self._agent.action_dim,)
            return self._rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        key = jax.random.fold_in(self._act_key, step)
        return self._agent.act(state, key)

    def _env_step(
        self, state: np.ndarray, step: int
    ) -> tuple[np.ndarray, float, bool]:
        """Take one real step and store it, return (state, reward, over)."""
        action = self._policy_action(state, step)
        next_raw, reward, terminated, truncated, _ = self._env.step(
            action * np.float32(self._agent.action_limit)
        )
        next_state = np.asarray(next_raw, dtype=np.float32)
        self._budget.consume_rl(1)
        self._normalizer.update(float(reward), terminated or truncated)
        self._buffer.add(
            Transition(
                state=state[None],
                action=action[None],
                reward=np.asarray([reward], dtype=np.float32),
                next_state=next_state[None],
                done=np.asarray(
                    [float(terminated and not truncated)], dtype=np.float32
                ),
            )
        )
        return next_state, float(reward), bool(terminated or truncated)

    def _reset_env(self) -> np.ndarray:
        seed = int(self._rng.integers(0, 2**31 - 1))
        state, _ = self._env.reset(seed=seed)
        return np.asarray(state, dtype=np.float32)

    def _learn(self, step: int) -> None:
        if step <= self._algo.start_training:
            return
        self._stats = self._agent.update(
            self._buffer.data, len(self._buffer), self._normalizer.scale
        )

    def _log(self, steps: int) -> None:
        self._agent.reward_scale = self._normalizer.scale
        eval_metrics = self._perf.maybe_evaluate(self._agent, steps)
        if eval_metrics is not None:
            print(
                f"steps {steps:7d} | eval={eval_metrics['perf/eval_rl']:.1f} | "
                f"critic_loss={self._stats.get('critic_loss', 0.0):.4f}"
            )
            if self._on_log is not None:
                metrics = budget_metrics(self._budget, 0.0)
                metrics.update(eval_metrics)
                metrics.update({f"xqc/{k}": v for k, v in self._stats.items()})
                metrics["xqc/reward_scale"] = self._normalizer.scale
                self._on_log(metrics)
        if self._checkpointer is not None:
            self._checkpointer.maybe_save(steps)

    def train(self) -> float:
        """Run until the budget ends and return the final evaluation."""
        try:
            state = self._reset_env()
            episode_return = 0.0
            while not self._budget.exhausted:
                step = self._budget.env_steps
                state, reward, over = self._env_step(state, step)
                episode_return += reward
                steps = self._budget.env_steps
                self._learn(step + 1)
                self._log(steps)
                if over:
                    self._perf.record_train_return(episode_return)
                    episode_return = 0.0
                    state = self._reset_env()
            if self._checkpointer is not None:
                self._checkpointer.finish(self._budget.env_steps)
            self._agent.reward_scale = self._normalizer.scale
            final_eval = self._perf.evaluate(
                self._agent, env_steps=self._budget.env_steps
            ).rl_return
            print_summary("xqc", self._budget, final_eval)
            return final_eval
        finally:
            self._env.close()
            self._perf.close()
