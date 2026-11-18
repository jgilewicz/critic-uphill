from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from agents.td3 import TD3
from common.replay_buffer import ReplayBuffer
from common.rollout import Budget, rollout_single_episode
from common.training import LogFn, PerfEvaluator, print_summary, run_warmup
from config import EvoRainbowAlgoConfig, RunConfig
from environments import env_dims, make_env, make_vec_env
from fitness import build_fitness_estimator
from modules.deep_modules import ActorHead
from modules.evo_module import CEM

_EVAL_SEED_OFFSET = 10_000


class EvoRainbowTrainer:
    """Training loop for EvoRainbow (shared embedding, CEM, TD3, surrogate)."""

    def __init__(
        self,
        run_cfg: RunConfig,
        algo_cfg: EvoRainbowAlgoConfig,
        *,
        on_log: LogFn | None = None,
    ) -> None:
        self._run_cfg = run_cfg
        self._algo = algo_cfg
        self._on_log = on_log

        dims = env_dims(run_cfg.env_id, horizon=algo_cfg.horizon)
        self._key = jax.random.key(run_cfg.seed)
        self._seed_rng = np.random.default_rng(run_cfg.seed)
        self._budget = Budget(run_cfg.total_steps)
        self._rl_env = make_env(run_cfg.env_id)
        self._pop_pool = make_vec_env(run_cfg.env_id, algo_cfg.pop_size)
        self._evaluator = PerfEvaluator(
            run_cfg.eval_env_id,
            interval=run_cfg.eval_interval,
            episodes=run_cfg.eval_episodes,
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
        self._cem = CEM(
            jnp.zeros_like(self._agent.actor_head_flat()),
            self._next_key(),
            pop_size=algo_cfg.pop_size,
            sigma_init=algo_cfg.cem.sigma_init,
            damp=algo_cfg.cem.damp,
            damp_limit=algo_cfg.cem.damp_limit,
            elitism=algo_cfg.cem.elitism,
        )
        self._estimator = build_fitness_estimator(
            algo_cfg.eval,
            self._pop_pool,
            self._agent,
            self._buffer,
            self._budget,
            horizon=algo_cfg.horizon,
            gamma=algo_cfg.rl.gamma,
        )

    def _next_key(self) -> jax.Array:
        self._key, sub = jax.random.split(self._key)
        return sub

    def _step_generation(
        self, generation: int, pending_injection: int | None
    ) -> int | None:
        flat_pop = self._cem.ask(self._next_key())
        if pending_injection is not None:
            flat_pop = flat_pop.at[pending_injection].set(
                self._agent.actor_head_flat()
            )
        heads = jax.vmap(self._agent.unravel_head)(flat_pop)

        rl_seed = int(self._seed_rng.integers(0, 2**31 - 1))
        _, rl_steps = rollout_single_episode(
            self._rl_env,
            lambda s: self._agent.act(s, self._next_key()),
            self._buffer,
            self._budget,
            max_steps=self._algo.horizon,
            seed=rl_seed,
        )
        pop_steps, next_injection = self._evaluate_and_evolve(
            generation, flat_pop, heads
        )
        gen_steps = rl_steps + pop_steps
        num_updates = int(self._algo.rl.train_ratio * gen_steps)
        critic_loss = self._agent.update(
            self._buffer, num_updates, self._next_key()
        )
        self._log_generation(generation, num_updates > 0, critic_loss)
        return next_injection

    def _evaluate_and_evolve(
        self, generation: int, flat_pop: jax.Array, heads: ActorHead
    ) -> tuple[int, int | None]:
        if self._budget.exhausted and self._estimator.uses_real_steps:
            return 0, None
        pop_seed = int(self._seed_rng.integers(0, 2**31 - 1))
        result = self._estimator.evaluate(
            heads,
            self._next_key(),
            seed=pop_seed,
        )
        self._cem.tell(result.fitness, flat_pop)
        elite_head = self._agent.unravel_head(self._cem.elite)
        self._agent.soft_update_actor_from(elite_head, self._algo.ea_to_rl.tau)
        period = self._algo.rl_to_ea.period
        next_injection = (
            int(jnp.argmin(result.fitness))
            if period > 0 and generation % period == 0
            else None
        )
        return result.steps, next_injection

    def _log_generation(
        self, generation: int, updated: bool, critic_loss: float
    ) -> None:
        eval_ret = self._evaluator.maybe_evaluate(
            self._agent, self._budget.env_steps
        )
        metrics: dict[str, float] = {
            "env_steps": float(self._budget.env_steps),
            "cost/pop_steps": float(self._budget.pop_steps),
        }
        if updated:
            metrics["train/critic_loss"] = critic_loss
        if eval_ret is not None:
            metrics["perf/eval_rl"] = eval_ret
        if eval_ret is not None or generation == 0:
            eval_str = f"{eval_ret:.1f}" if eval_ret is not None else "n/a"
            print(
                f"gen {generation:4d} | steps {self._budget.env_steps:7d} | "
                f"pop_steps={self._budget.pop_steps:7d} | "
                f"eval={eval_str} | critic_loss={critic_loss:.4f}"
            )
        if self._on_log is not None:
            self._on_log(metrics)

    def train(self) -> float:
        """Run warmup and EvoRainbow generations until the budget ends."""
        try:
            run_warmup(
                self._rl_env,
                self._agent,
                self._buffer,
                self._budget,
                warmup_steps=self._algo.rl.warmup_steps,
                horizon=self._algo.horizon,
                seed_rng=self._seed_rng,
                evaluator=self._evaluator,
                on_log=self._on_log,
            )
            generation = 0
            pending_injection: int | None = None
            while not self._budget.exhausted:
                pending_injection = self._step_generation(
                    generation, pending_injection
                )
                generation += 1
            final_eval = self._evaluator.evaluate(self._agent)
            tag = f"evorainbow-{self._algo.eval.mode}"
            print_summary(tag, self._budget, final_eval)
            return final_eval
        finally:
            self._rl_env.close()
            self._pop_pool.close()
            self._evaluator.close()
