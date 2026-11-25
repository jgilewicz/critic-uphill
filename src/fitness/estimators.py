from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from agents.base import RLAgent
from common.replay_buffer import ReplayBuffer
from common.rollout import (
    Budget,
    PrefixSnapshot,
    rollout_population,
)
from config import EvalConfig
from environments import EnvPool


_EPS = 1e-8


@dataclass(frozen=True)
class FitnessResult:
    """Population fitness, real env steps, and surrogate diagnostics.

    real_mask marks individuals scored by a full real episode. surrogate
    holds surrogate scores of those individuals (valid at real_mask).
    critic_share is mean |gamma^H Q| / mean |score| over the
    surrogate-scored individuals.
    """

    fitness: jax.Array
    steps: int
    real_mask: np.ndarray
    surrogate: np.ndarray | None = None
    critic_share: float | None = None
    prefix: PrefixSnapshot | None = None

    @classmethod
    def skipped(cls, pop_size: int) -> FitnessResult:
        """Zero fitness for generations that evaluate nothing."""
        return cls(
            fitness=jnp.zeros(pop_size, dtype=jnp.float32),
            steps=0,
            real_mask=np.zeros(pop_size, dtype=bool),
        )


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


class SurrogateEstimator(FitnessEstimator, Protocol):
    """Fitness estimator that can also rescore individuals run for real."""

    def score_real(
        self, heads: Any, key: jax.Array, real: FitnessResult
    ) -> np.ndarray:
        """Surrogate scores of real-evaluated individuals, no env steps."""
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
        prefix_h: int | None = None,
        gamma: float = 1.0,
        critic_states: int | None = None,
    ) -> None:
        self._pool = pool
        self._agent = agent
        self._buffer = buffer
        self._budget = budget
        self._horizon = horizon
        self._prefix_h = prefix_h
        self._gamma = gamma
        self._critic_states = critic_states

    def evaluate(
        self,
        heads: Any,
        key: jax.Array,
        *,
        seed: int,
        active_mask: np.ndarray | None = None,
    ) -> FitnessResult:
        """Roll out active individuals up to horizon and return returns."""
        rollout = rollout_population(
            self._pool,
            lambda states: self._agent.act_population(heads, states),
            self._buffer,
            self._budget,
            max_steps=self._horizon,
            discount=1.0,
            seed=seed,
            active_mask=active_mask,
            prefix_h=self._prefix_h,
            prefix_discount=self._gamma,
        )
        surrogate = None
        if self._critic_states is not None:
            # metrics only: CEM ranks the real returns, not this score
            states = self._buffer.sample_states(key, self._critic_states)
            surrogate = np.asarray(
                self._agent.q1_population_mean(heads, states)
            )
        return FitnessResult(
            fitness=jnp.asarray(rollout.returns, dtype=jnp.float32),
            steps=rollout.total_steps,
            real_mask=rollout.steps_taken > 0,
            surrogate=surrogate,
            prefix=rollout.prefix,
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

    def _scores(self, heads: Any, key: jax.Array) -> jax.Array:
        states = self._buffer.sample_states(key, self._critic_states)
        return self._agent.q1_population_mean(heads, states)

    def evaluate(
        self,
        heads: Any,
        key: jax.Array,
        *,
        seed: int,
        active_mask: np.ndarray | None = None,
    ) -> FitnessResult:
        """Sample buffer states and evaluate the mean surrogate critic."""
        del seed, active_mask
        fitness = self._scores(heads, key)
        return FitnessResult(
            fitness=fitness,
            steps=0,
            real_mask=np.zeros(fitness.shape[0], dtype=bool),
            surrogate=np.asarray(fitness),
        )

    def score_real(
        self, heads: Any, key: jax.Array, real: FitnessResult
    ) -> np.ndarray:
        """Surrogate score of every head on the surrogate fitness states."""
        del real
        return np.asarray(self._scores(heads, key))


def _hboot_scores(
    agent: RLAgent, heads: Any, prefix: PrefixSnapshot, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    q_end = np.asarray(
        agent.q1_population_states(heads, prefix.states), dtype=np.float64
    )
    discounts = gamma ** prefix.steps_taken.astype(np.float64)
    bootstrap = np.where(prefix.terminated, 0.0, discounts * q_end)
    return prefix.returns + bootstrap, bootstrap


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
        no_real = np.zeros(self._pool.num_envs, dtype=bool)
        if rollout.last_states.shape[1] == 0:
            return FitnessResult(
                fitness=jnp.zeros(self._pool.num_envs, dtype=jnp.float32),
                steps=0,
                real_mask=no_real,
            )
        prefix = PrefixSnapshot(
            returns=rollout.returns,
            states=rollout.last_states,
            terminated=rollout.terminated,
            steps_taken=rollout.steps_taken,
        )
        total, bootstrap = _hboot_scores(
            self._agent, heads, prefix, self._gamma
        )
        scored = (
            np.ones_like(no_real)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        share = np.mean(np.abs(bootstrap[scored])) / max(
            float(np.mean(np.abs(total[scored]))), _EPS
        )
        return FitnessResult(
            fitness=jnp.asarray(total, dtype=jnp.float32),
            steps=rollout.total_steps,
            real_mask=no_real,
            critic_share=float(share),
        )

    def score_real(
        self, heads: Any, key: jax.Array, real: FitnessResult
    ) -> np.ndarray:
        """H-step score from the first H steps of the real episodes."""
        del key
        if real.prefix is None:
            raise ValueError("hboot score_real needs a full-episode prefix")
        return _hboot_scores(self._agent, heads, real.prefix, self._gamma)[0]


class PrefixFitness:
    """Score by the undiscounted return of the first h real steps."""

    uses_real_steps: bool = True

    def __init__(
        self,
        pool: EnvPool,
        agent: RLAgent,
        buffer: ReplayBuffer,
        budget: Budget,
        *,
        h: int,
    ) -> None:
        self._pool = pool
        self._agent = agent
        self._buffer = buffer
        self._budget = budget
        self._h = h

    def evaluate(
        self,
        heads: Any,
        key: jax.Array,
        *,
        seed: int,
        active_mask: np.ndarray | None = None,
    ) -> FitnessResult:
        """Run up to h real steps per individual and sum the rewards."""
        del key
        rollout = rollout_population(
            self._pool,
            lambda states: self._agent.act_population(heads, states),
            self._buffer,
            self._budget,
            max_steps=self._h,
            discount=1.0,
            seed=seed,
            active_mask=active_mask,
        )
        return FitnessResult(
            fitness=jnp.asarray(rollout.returns, dtype=jnp.float32),
            steps=rollout.total_steps,
            real_mask=np.zeros(self._pool.num_envs, dtype=bool),
        )


class IndividualControlFitness:
    """Per-individual gate: full episode w.p. p_real, else surrogate."""

    def __init__(
        self,
        surrogate: SurrogateEstimator,
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
        evaluated = np.zeros(self._pop_size, dtype=bool)
        surrogate: np.ndarray | None = None

        if real_mask.any():
            real_res = self._full.evaluate(
                heads, real_key, seed=seed, active_mask=real_mask
            )
            fitness[real_mask] = np.asarray(real_res.fitness)[real_mask]
            total_steps += real_res.steps
            evaluated = real_res.real_mask

        surr_mask = ~real_mask
        surr_res: FitnessResult | None = None
        critic_share = None
        if surr_mask.any():
            surr_res = self._surrogate.evaluate(
                heads, surr_key, seed=seed, active_mask=surr_mask
            )
            fitness[surr_mask] = np.asarray(surr_res.fitness)[surr_mask]
            total_steps += surr_res.steps
            critic_share = surr_res.critic_share

        if evaluated.any():
            reusable = None if surr_res is None else surr_res.surrogate
            surrogate = (
                reusable
                if reusable is not None
                else self._surrogate.score_real(heads, surr_key, real_res)
            )

        return FitnessResult(
            fitness=jnp.asarray(fitness, dtype=jnp.float32),
            steps=total_steps,
            real_mask=evaluated,
            surrogate=surrogate,
            critic_share=critic_share,
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
    full = FullEpisodeFitness(
        pool,
        agent,
        buffer,
        budget,
        horizon=horizon,
        prefix_h=eval_cfg.h if eval_cfg.mode == "hboot" else None,
        gamma=gamma,
        critic_states=(
            eval_cfg.critic_states if eval_cfg.mode == "full" else None
        ),
    )
    if eval_cfg.mode == "full":
        return full
    if eval_cfg.mode == "prefix":
        return PrefixFitness(pool, agent, buffer, budget, h=eval_cfg.h)
    if eval_cfg.mode == "critic":
        surrogate: SurrogateEstimator = CriticBufferFitness(
            agent,
            buffer,
            critic_states=eval_cfg.critic_states,
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
