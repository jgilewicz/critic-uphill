from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from agents.base import RLAgent
from common.replay_buffer import ReplayBuffer
from common.rollout import Budget, rollout_population
from config import EvalConfig
from environments import EnvPool


@dataclass(frozen=True)
class FitnessResult:
    """Population fitness scores, real env steps, and full eval count."""

    fitness: jax.Array
    steps: int
    real_evals: int = 0


class FitnessEstimator(Protocol):
    """Protocol for population fitness scoring strategies."""

    @property
    def uses_real_steps(self) -> bool:
        """Whether this estimator requires environment interactions."""
        ...

    def evaluate(
        self,
        heads: Any,
        key: jax.Array,
        *,
        seed: int,
        active_mask: np.ndarray | None = None,
    ) -> FitnessResult:
        """Score a stacked population of actor head instances."""
        ...


class FullEpisodeFitness:
    """Score each individual by one full real episode (up to horizon)."""

    uses_real_steps: bool = True

    def __init__(
        self,
        pool: EnvPool,
        agent: RLAgent,
        buffer: ReplayBuffer,
        budget: Budget,
        *,
        horizon: int,
    ) -> None:
        self._pool = pool
        self._agent = agent
        self._buffer = buffer
        self._budget = budget
        self._horizon = horizon

    def evaluate(
        self,
        heads: Any,
        key: jax.Array,
        *,
        seed: int,
        active_mask: np.ndarray | None = None,
    ) -> FitnessResult:
        """Roll out active individuals up to horizon and return returns."""
        del key
        rollout = rollout_population(
            self._pool,
            lambda states: self._agent.act_population(heads, states),
            self._buffer,
            self._budget,
            max_steps=self._horizon,
            discount=1.0,
            seed=seed,
            active_mask=active_mask,
        )
        evaluated = int(np.sum(rollout.steps_taken > 0))
        return FitnessResult(
            fitness=jnp.asarray(rollout.returns, dtype=jnp.float32),
            steps=rollout.total_steps,
            real_evals=evaluated,
        )


class CriticBufferFitness:
    """Score each individual by mean Q1(s, pi_i(s)) on replay buffer states."""

    uses_real_steps: bool = False

    def __init__(
        self,
        agent: RLAgent,
        buffer: ReplayBuffer,
        *,
        critic_states: int,
    ) -> None:
        self._agent = agent
        self._buffer = buffer
        self._critic_states = critic_states

    def evaluate(
        self,
        heads: Any,
        key: jax.Array,
        *,
        seed: int,
        active_mask: np.ndarray | None = None,
    ) -> FitnessResult:
        """Sample buffer states and evaluate mean online Q1."""
        del seed, active_mask
        states = self._buffer.sample_states(key, self._critic_states)
        fitness = self._agent.q1_population_mean(heads, states)
        return FitnessResult(fitness=fitness, steps=0, real_evals=0)


class HStepBootstrapFitness:
    """Score by H-step discounted return plus gamma^H Q1(s_H, pi_i(s_H))."""

    uses_real_steps: bool = True

    def __init__(
        self,
        pool: EnvPool,
        agent: RLAgent,
        buffer: ReplayBuffer,
        budget: Budget,
        *,
        h: int,
        gamma: float,
    ) -> None:
        self._pool = pool
        self._agent = agent
        self._buffer = buffer
        self._budget = budget
        self._h = h
        self._gamma = gamma

    def evaluate(
        self,
        heads: Any,
        key: jax.Array,
        *,
        seed: int,
        active_mask: np.ndarray | None = None,
    ) -> FitnessResult:
        """Run up to H real steps and bootstrap non-terminal endpoints."""
        del key
        rollout = rollout_population(
            self._pool,
            lambda states: self._agent.act_population(heads, states),
            self._buffer,
            self._budget,
            max_steps=self._h,
            discount=self._gamma,
            seed=seed,
            active_mask=active_mask,
        )
        if rollout.last_states.shape[1] == 0:
            return FitnessResult(
                fitness=jnp.zeros(self._pool.num_envs, dtype=jnp.float32),
                steps=0,
                real_evals=0,
            )
        q_end = np.asarray(
            self._agent.q1_population_states(heads, rollout.last_states),
            dtype=np.float64,
        )
        discounts = self._gamma ** rollout.steps_taken.astype(np.float64)
        bootstrap = np.where(rollout.terminated, 0.0, discounts * q_end)
        total = rollout.returns + bootstrap
        return FitnessResult(
            fitness=jnp.asarray(total, dtype=jnp.float32),
            steps=rollout.total_steps,
            real_evals=0,
        )


class IndividualControlFitness:
    """Per-individual gate: full episode w.p. p_real, else surrogate."""

    def __init__(
        self,
        surrogate: FitnessEstimator,
        full: FullEpisodeFitness,
        *,
        pop_size: int,
        p_real: float,
    ) -> None:
        self._surrogate = surrogate
        self._full = full
        self._pop_size = pop_size
        self._p_real = p_real

    @property
    def uses_real_steps(self) -> bool:
        """True if surrogate uses real steps or p_real can trigger full eval."""
        return self._surrogate.uses_real_steps or self._p_real > 0.0

    def evaluate(
        self,
        heads: Any,
        key: jax.Array,
        *,
        seed: int,
        active_mask: np.ndarray | None = None,
    ) -> FitnessResult:
        """Draw independent per-individual coins and combine fitness scores."""
        del active_mask
        coin_key, real_key, surr_key = jax.random.split(key, 3)
        real_mask = np.asarray(
            jax.random.uniform(coin_key, (self._pop_size,)) < self._p_real,
            dtype=bool,
        )
        fitness = np.zeros(self._pop_size, dtype=np.float32)
        total_steps = 0
        real_evals = 0

        if real_mask.any():
            real_res = self._full.evaluate(
                heads, real_key, seed=seed, active_mask=real_mask
            )
            fitness[real_mask] = np.asarray(real_res.fitness)[real_mask]
            total_steps += real_res.steps
            real_evals = real_res.real_evals

        surr_mask = ~real_mask
        if surr_mask.any():
            surr_res = self._surrogate.evaluate(
                heads, surr_key, seed=seed, active_mask=surr_mask
            )
            fitness[surr_mask] = np.asarray(surr_res.fitness)[surr_mask]
            total_steps += surr_res.steps

        return FitnessResult(
            fitness=jnp.asarray(fitness, dtype=jnp.float32),
            steps=total_steps,
            real_evals=real_evals,
        )


def build_fitness_estimator(
    eval_cfg: EvalConfig,
    pool: EnvPool,
    agent: RLAgent,
    buffer: ReplayBuffer,
    budget: Budget,
    *,
    horizon: int,
    gamma: float,
) -> FitnessEstimator:
    """Construct the configured fitness estimator with per-individual gating."""
    full = FullEpisodeFitness(pool, agent, buffer, budget, horizon=horizon)
    if eval_cfg.mode == "full":
        return full
    if eval_cfg.mode == "critic":
        surrogate: FitnessEstimator = CriticBufferFitness(
            agent, buffer, critic_states=eval_cfg.critic_states
        )
    else:
        surrogate = HStepBootstrapFitness(
            pool, agent, buffer, budget, h=eval_cfg.h, gamma=gamma
        )
    if eval_cfg.p_real == 0.0:
        return surrogate
    return IndividualControlFitness(
        surrogate, full, pop_size=pool.num_envs, p_real=eval_cfg.p_real
    )
