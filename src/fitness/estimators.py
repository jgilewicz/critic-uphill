from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from agents.td3 import TD3
from common.replay_buffer import ReplayBuffer
from common.rollout import Budget, rollout_population
from config import EvalConfig
from environments import EnvPool
from modules.deep_modules import ActorHead


@dataclass(frozen=True)
class FitnessResult:
    """Population fitness scores and real env steps consumed."""

    fitness: jax.Array
    steps: int


class FitnessEstimator(Protocol):
    """Protocol for population fitness scoring strategies."""

    @property
    def uses_real_steps(self) -> bool:
        """Whether this estimator requires environment interactions."""
        ...

    def evaluate(
        self, heads: ActorHead, key: jax.Array, *, seed: int
    ) -> FitnessResult:
        """Score a stacked population of ActorHead instances."""
        ...


class FullEpisodeFitness:
    """Score each individual by one full real episode (up to horizon)."""

    uses_real_steps: bool = True

    def __init__(
        self,
        pool: EnvPool,
        agent: TD3,
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
        self, heads: ActorHead, key: jax.Array, *, seed: int
    ) -> FitnessResult:
        """Roll out each individual up to horizon and return episode returns."""
        del key
        rollout = rollout_population(
            self._pool,
            lambda states: self._agent.act_population(heads, states),
            self._buffer,
            self._budget,
            max_steps=self._horizon,
            discount=1.0,
            seed=seed,
        )
        return FitnessResult(
            fitness=jnp.asarray(rollout.returns, dtype=jnp.float32),
            steps=rollout.total_steps,
        )


class CriticBufferFitness:
    """Score each individual by mean Q1(s, pi_i(s)) on replay buffer states."""

    uses_real_steps: bool = False

    def __init__(
        self,
        agent: TD3,
        buffer: ReplayBuffer,
        *,
        critic_states: int,
    ) -> None:
        self._agent = agent
        self._buffer = buffer
        self._critic_states = critic_states

    def evaluate(
        self, heads: ActorHead, key: jax.Array, *, seed: int
    ) -> FitnessResult:
        """Sample buffer states and evaluate mean online Q1."""
        del seed
        states = self._buffer.sample_states(key, self._critic_states)
        fitness = self._agent.q1_population_mean(heads, states)
        return FitnessResult(fitness=fitness, steps=0)


class HStepBootstrapFitness:
    """Score by H-step discounted return plus gamma^H Q1(s_H, pi_i(s_H))."""

    uses_real_steps: bool = True

    def __init__(
        self,
        pool: EnvPool,
        agent: TD3,
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
        self, heads: ActorHead, key: jax.Array, *, seed: int
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
        )


class GenerationControlFitness:
    """With probability p_real per generation, score with full episodes."""

    def __init__(
        self,
        surrogate: FitnessEstimator,
        full: FullEpisodeFitness,
        *,
        p_real: float,
    ) -> None:
        self._surrogate = surrogate
        self._full = full
        self._p_real = p_real

    @property
    def uses_real_steps(self) -> bool:
        """True if surrogate uses real steps or p_real can trigger full eval."""
        return self._surrogate.uses_real_steps or self._p_real > 0.0

    def evaluate(
        self, heads: ActorHead, key: jax.Array, *, seed: int
    ) -> FitnessResult:
        """Draw generation coin and delegate to full or surrogate estimator."""
        coin_key, eval_key = jax.random.split(key)
        use_full = (
            self._p_real > 0.0
            and float(jax.random.uniform(coin_key)) < self._p_real
        )
        estimator: FitnessEstimator = (
            self._full if use_full else self._surrogate
        )
        return estimator.evaluate(heads, eval_key, seed=seed)


def build_fitness_estimator(
    eval_cfg: EvalConfig,
    pool: EnvPool,
    agent: TD3,
    buffer: ReplayBuffer,
    budget: Budget,
    *,
    horizon: int,
    gamma: float,
) -> FitnessEstimator:
    """Construct the configured fitness estimator with generation control."""
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
    return GenerationControlFitness(surrogate, full, p_real=eval_cfg.p_real)
