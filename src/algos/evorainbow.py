from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np

from agents import make_agent
from common.checkpoint import GenerationRecord, make_checkpointer
from common.replay_buffer import ReplayBuffer
from common.rollout import Budget, rollout_single_episode
from common.training import LogFn, print_summary
from config import EvoRainbowAlgoConfig, RunConfig
from environments import env_dims, make_env, make_vec_env
from fitness import FitnessResult, build_fitness_estimator
from metrics.diagnostics import Diagnostics
from metrics.perf import Champion, PerfMetrics, budget_metrics
from metrics.surrogate import SurrogateMetrics
from modules.evo_module import CEM

_EVAL_SEED_OFFSET = 10_000


class EvoRainbowTrainer:
    """EvoRainbow training loop: shared embedding, CEM, RL agent, surrogate."""

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
        self._perf = PerfMetrics(
            run_cfg.eval_env.id,
            interval=run_cfg.eval.interval,
            episodes=run_cfg.eval.episodes,
            seed=run_cfg.seed + _EVAL_SEED_OFFSET,
            gamma=algo_cfg.rl.gamma,
        )
        self._diagnostics = Diagnostics(run_cfg, self._perf.env)
        self._surrogate_metrics = SurrogateMetrics()
        self._champion: Champion | None = None
        self._individual_evals = 0
        self._real_evals = 0
        self._rl_updates = 0
        self._buffer = ReplayBuffer(
            algo_cfg.rl.buffer_capacity, dims.obs_dim, dims.action_dim
        )
        self._agent = make_agent(
            algo_cfg.agent,
            dims.obs_dim,
            dims.action_dim,
            dims.action_limit,
            algo_cfg.embedding_dim,
            rl_cfg=algo_cfg.rl,
            key=self._next_key(),
            shadow=algo_cfg.shadow.enabled,
        )
        self._cem = CEM(
            self._agent.init_cem_head(self._next_key()),
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
            shadow=algo_cfg.shadow.surrogate,
        )
        self._checkpointer = make_checkpointer(
            run_cfg, self._agent, self._buffer
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

    def _should_evaluate(self, *, in_warmup: bool) -> bool:
        exhausted = self._budget.exhausted and self._estimator.uses_real_steps
        return not (in_warmup or exhausted)

    def _evaluate_population(self, flat_pop: jax.Array) -> FitnessResult:
        heads = jax.vmap(self._agent.unravel_head)(flat_pop)
        pop_seed = int(self._seed_rng.integers(0, 2**31 - 1))
        result = self._estimator.evaluate(
            heads, self._next_key(), seed=pop_seed
        )
        self._individual_evals += self._algo.pop_size
        self._real_evals += int(result.real_mask.sum())
        return result

    @property
    def _real_frac(self) -> float:
        if self._individual_evals == 0:
            return 0.0
        return self._real_evals / self._individual_evals

    def _rollout_rl(self, *, in_warmup: bool) -> int:
        rl_seed = int(self._seed_rng.integers(0, 2**31 - 1))
        act_fn = (
            self._random_action
            if in_warmup
            else lambda s: self._agent.act(s, self._next_key())
        )
        ep_return, rl_steps = rollout_single_episode(
            self._rl_env,
            act_fn,
            self._buffer,
            self._budget,
            max_steps=self._algo.horizon,
            seed=rl_seed,
        )
        self._perf.record_train_return(ep_return)
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
        actor_flat = self._agent.actor_head_flat()
        flat_pop = self._cem.ask(self._next_key())
        if pending_injection is not None:
            flat_pop = flat_pop.at[pending_injection].set(actor_flat)

        evaluated = self._should_evaluate(in_warmup=in_warmup)
        result = (
            self._evaluate_population(flat_pop)
            if evaluated
            else FitnessResult.skipped(self._algo.pop_size)
        )
        fitness = result.fitness
        best_idx = int(jnp.argmax(fitness))
        gen_metrics: dict[str, float] = {}
        if evaluated:
            self._champion = Champion(
                self._agent.embedding,
                self._agent.unravel_head(flat_pop[best_idx]),
            )
            gen_metrics = self._surrogate_metrics.measure(
                np.asarray(flat_pop),
                np.asarray(actor_flat),
                result,
                pending_injection,
            )
            self._record_generation(
                generation, flat_pop, actor_flat, result, pending_injection
            )
        if not self._budget.exhausted:
            self._cem.tell(fitness, flat_pop)
            if not in_warmup and best_idx != pending_injection:
                best_head = self._agent.unravel_head(flat_pop[best_idx])
                self._agent.soft_update_actor_from(
                    best_head, self._algo.ea_to_rl.tau
                )

        rl_steps = self._rollout_rl(in_warmup=in_warmup)
        post_warmup = self._budget.rl_steps >= self._algo.rl.warmup_steps
        gen_steps = result.steps + rl_steps
        num_updates = (
            self._num_updates(generation, gen_steps) if post_warmup else 0
        )
        losses = self._agent.update(self._buffer, num_updates, self._next_key())
        self._rl_updates += num_updates
        self._log_generation(
            generation, {**losses, **gen_metrics}, result, in_warmup=in_warmup
        )
        if self._checkpointer is not None:
            self._checkpointer.maybe_save(self._budget.env_steps)
        return self._next_injection(
            generation, fitness, post_warmup=post_warmup
        )

    def _num_updates(self, generation: int, gen_steps: int) -> int:
        matched = self._algo.matched
        if not matched.enabled:
            return int(self._algo.rl.train_ratio * gen_steps)
        # spread the rest evenly, so the last generation hits rl_updates
        left = matched.rl_updates - self._rl_updates
        return left // (matched.generations - generation)

    def _done(self, generation: int) -> bool:
        matched = self._algo.matched
        if matched.enabled and generation >= matched.generations:
            return True
        return self._budget.exhausted

    def _record_generation(
        self,
        generation: int,
        flat_pop: jax.Array,
        actor_flat: jax.Array,
        result: FitnessResult,
        injected: int | None,
    ) -> None:
        if self._checkpointer is None:
            return
        self._checkpointer.record_generation(
            GenerationRecord(
                generation=generation,
                env_steps=self._budget.env_steps,
                flat_pop=np.asarray(flat_pop),
                actor_flat=np.asarray(actor_flat),
                injected=-1 if injected is None else injected,
                fitness=np.asarray(result.fitness),
                real_mask=np.asarray(result.real_mask),
            )
        )

    def _log_generation(
        self,
        generation: int,
        metrics_in: dict[str, float],
        result: FitnessResult,
        *,
        in_warmup: bool,
    ) -> None:
        steps = self._budget.env_steps
        metrics = budget_metrics(self._budget, self._real_frac)
        metrics["budget/generation"] = float(generation + 1)
        metrics["budget/rl_updates"] = float(self._rl_updates)
        metrics.update(metrics_in)
        eval_metrics = self._perf.maybe_evaluate(
            self._agent, steps, champion=self._champion
        )
        if eval_metrics is not None:
            metrics.update(eval_metrics)
        metrics.update(
            self._diagnostics.maybe_measure(self._agent, self._buffer, steps)
        )
        if eval_metrics is not None or generation == 0 or not in_warmup:
            eval_str = (
                f"{eval_metrics['perf/eval_rl']:.1f}"
                if eval_metrics is not None
                else "n/a"
            )
            print(
                f"gen {generation:4d} | steps {steps:7d} | "
                f"pop_steps={self._budget.pop_steps:7d} | "
                f"real_evals={int(result.real_mask.sum())}"
                f"/{self._algo.pop_size} | eval={eval_str} | "
                f"critic_loss={metrics.get('critic/td_loss', 0.0):.4f}"
            )
        if self._on_log is not None:
            self._on_log(metrics)

    def train(self) -> float:
        """Run warmup and EvoRainbow generations until the budget ends."""
        try:
            generation = 0
            pending_injection: int | None = None
            while not self._done(generation):
                pending_injection = self._step_generation(
                    generation, pending_injection
                )
                generation += 1
            if self._checkpointer is not None:
                self._checkpointer.finish(self._budget.env_steps)
            final_eval = self._perf.evaluate(
                self._agent, env_steps=self._budget.env_steps
            ).rl_return
            mode = "shadow" if self._algo.shadow.surrogate else None
            tag = (
                f"evorainbow-{self._algo.agent}-{mode or self._algo.eval.mode}"
            )
            print_summary(tag, self._budget, final_eval)
            return final_eval
        finally:
            self._rl_env.close()
            self._pop_pool.close()
            self._perf.close()
