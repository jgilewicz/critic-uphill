from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from agents.td3 import TD3
from common.replay_buffer import ReplayBuffer
from common.rollout import Budget, rollout_single_episode
from common.training import LogFn, PerfEvaluator, print_summary
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

        dims = env_dims(run_cfg.env.id)
        self._key = jax.random.key(run_cfg.seed)
        self._seed_rng = np.random.default_rng(run_cfg.seed)
        self._budget = Budget(run_cfg.total_steps)
        self._rl_env = make_env(run_cfg.env.id)
        self._pop_pool = make_vec_env(run_cfg.env.id, algo_cfg.pop_size)
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
        init_head = ActorHead(
            algo_cfg.embedding_dim,
            dims.action_dim,
            key=self._next_key(),
            action_limit=dims.action_limit,
        )
        self._cem = CEM(
            ravel_pytree(init_head)[0],
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

    def _random_action(self, _state: np.ndarray) -> np.ndarray:
        limit = self._agent.action_limit
        shape = (self._agent.action_dim,)
        return self._seed_rng.uniform(-limit, limit, size=shape).astype(
            np.float32
        )

    def _evaluate_population(
        self, flat_pop: jax.Array, *, in_warmup: bool
    ) -> tuple[jax.Array, int, int]:
        if in_warmup or (
            self._budget.exhausted and self._estimator.uses_real_steps
        ):
            zeros = jnp.zeros(self._algo.pop_size, dtype=jnp.float32)
            return zeros, 0, 0
        heads = jax.vmap(self._agent.unravel_head)(flat_pop)
        pop_seed = int(self._seed_rng.integers(0, 2**31 - 1))
        result = self._estimator.evaluate(
            heads, self._next_key(), seed=pop_seed
        )
        return result.fitness, result.steps, result.real_evals

    def _rollout_rl(self, *, in_warmup: bool) -> int:
        rl_seed = int(self._seed_rng.integers(0, 2**31 - 1))
        act_fn = (
            self._random_action
            if in_warmup
            else lambda s: self._agent.act(s, self._next_key())
        )
        _, rl_steps = rollout_single_episode(
            self._rl_env,
            act_fn,
            self._buffer,
            self._budget,
            max_steps=self._algo.horizon,
            seed=rl_seed,
        )
        return rl_steps

    def _next_injection(
        self, generation: int, fitness: jax.Array, *, post_warmup: bool
    ) -> int | None:
        period = self._algo.rl_to_ea.period
        if post_warmup and period > 0 and (generation + 1) % period == 0:
            return int(jnp.argmin(fitness))
        return None

    def _step_generation(
        self, generation: int, pending_injection: int | None
    ) -> int | None:
        in_warmup = self._budget.rl_steps < self._algo.rl.warmup_steps
        flat_pop = self._cem.ask(self._next_key())
        if pending_injection is not None:
            flat_pop = flat_pop.at[pending_injection].set(
                self._agent.actor_head_flat()
            )

        fitness, pop_steps, real_evals = self._evaluate_population(
            flat_pop, in_warmup=in_warmup
        )
        if not self._budget.exhausted:
            self._cem.tell(fitness, flat_pop)
            best_idx = int(jnp.argmax(fitness))
            if not in_warmup and best_idx != pending_injection:
                best_head = self._agent.unravel_head(flat_pop[best_idx])
                self._agent.soft_update_actor_from(
                    best_head, self._algo.ea_to_rl.tau
                )

        rl_steps = self._rollout_rl(in_warmup=in_warmup)
        post_warmup = self._budget.rl_steps >= self._algo.rl.warmup_steps
        gen_steps = pop_steps + rl_steps
        num_updates = (
            int(self._algo.rl.train_ratio * gen_steps) if post_warmup else 0
        )
        critic_loss = self._agent.update(
            self._buffer, num_updates, self._next_key()
        )
        self._log_generation(
            generation,
            num_updates > 0,
            critic_loss,
            real_evals=real_evals,
            in_warmup=in_warmup,
        )
        return self._next_injection(
            generation, fitness, post_warmup=post_warmup
        )

    def _log_generation(
        self,
        generation: int,
        updated: bool,
        critic_loss: float,
        *,
        real_evals: int,
        in_warmup: bool,
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
        if eval_ret is not None or generation == 0 or not in_warmup:
            eval_str = f"{eval_ret:.1f}" if eval_ret is not None else "n/a"
            print(
                f"gen {generation:4d} | steps {self._budget.env_steps:7d} | "
                f"pop_steps={self._budget.pop_steps:7d} | "
                f"real_evals={real_evals}/{self._algo.pop_size} | "
                f"eval={eval_str} | critic_loss={critic_loss:.4f}"
            )
        if self._on_log is not None:
            self._on_log(metrics)

    def train(self) -> float:
        """Run warmup and EvoRainbow generations until the budget ends."""
        try:
            generation = 0
            pending_injection: int | None = None
            while not self._budget.exhausted:
                pending_injection = self._step_generation(
                    generation, pending_injection
                )
                generation += 1
            final_eval = self._evaluator.evaluate(
                self._agent, env_steps=self._budget.env_steps
            )
            tag = f"evorainbow-{self._algo.eval.mode}"
            print_summary(tag, self._budget, final_eval)
            return final_eval
        finally:
            self._rl_env.close()
            self._pop_pool.close()
            self._evaluator.close()
