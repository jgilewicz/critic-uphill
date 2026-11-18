from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from common.replay_buffer import ReplayBuffer, Transition
from environments import EnvPool


class Budget:
    """Shared hard cap on real environment interactions."""

    def __init__(self, total_steps: int) -> None:
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        self.total_steps = total_steps
        self.rl_steps = 0
        self.pop_steps = 0

    @property
    def env_steps(self) -> int:
        """Total real environment steps consumed so far."""
        return self.rl_steps + self.pop_steps

    @property
    def remaining(self) -> int:
        """Remaining step budget (>= 0)."""
        return max(0, self.total_steps - self.env_steps)

    @property
    def exhausted(self) -> bool:
        """Whether the step budget has been reached."""
        return self.env_steps >= self.total_steps

    def consume_rl(self, steps: int) -> None:
        """Record real steps taken by the RL actor."""
        self._consume(steps)
        self.rl_steps += steps

    def consume_pop(self, steps: int) -> None:
        """Record real steps taken by the population."""
        self._consume(steps)
        self.pop_steps += steps

    def _consume(self, steps: int) -> None:
        if steps < 0:
            raise ValueError(f"steps must be >= 0, got {steps}")
        if self.env_steps + steps > self.total_steps:
            raise RuntimeError(
                f"budget exceeded: {self.env_steps} + {steps} > "
                f"{self.total_steps}"
            )


@dataclass(frozen=True)
class PrefixSnapshot:
    """Population rollout state after its first h steps (or at the end)."""

    returns: np.ndarray
    states: np.ndarray
    terminated: np.ndarray
    steps_taken: np.ndarray


@dataclass(frozen=True)
class PopRolloutResult:
    """Outcome of a population rollout over an EnvPool."""

    returns: np.ndarray
    last_states: np.ndarray
    terminated: np.ndarray
    steps_taken: np.ndarray
    total_steps: int
    prefix: PrefixSnapshot | None = None


@dataclass(frozen=True)
class EpisodeResult:
    """Return, discounted return and first state of one evaluation episode."""

    ret: float
    discounted_ret: float
    first_state: np.ndarray


def evaluation_episode(
    env: gym.Env,
    act_fn: Callable[[np.ndarray], np.ndarray],
    *,
    seed: int,
    gamma: float,
) -> EpisodeResult:
    """Run one full episode outside the budget and store nothing."""
    obs_raw, _ = env.reset(seed=seed)
    state = first_state = np.asarray(obs_raw, dtype=np.float32)
    ret = discounted_ret = 0.0
    factor = 1.0
    done = False
    while not done:
        obs_raw, reward, terminated, truncated, _ = env.step(act_fn(state))
        state = np.asarray(obs_raw, dtype=np.float32)
        ret += float(reward)
        discounted_ret += factor * float(reward)
        factor *= gamma
        done = bool(terminated or truncated)
    return EpisodeResult(ret, discounted_ret, first_state)


def rollout_single_episode(
    env: gym.Env,
    act_fn: Callable[[np.ndarray], np.ndarray],
    buffer: ReplayBuffer,
    budget: Budget,
    *,
    max_steps: int,
    seed: int,
) -> tuple[float, int]:
    """Run one episode up to max_steps and remaining budget."""
    if budget.exhausted or max_steps <= 0:
        return 0.0, 0
    state_raw, _ = env.reset(seed=seed)
    state = np.asarray(state_raw, dtype=np.float32)
    ep_return = 0.0
    steps = 0
    limit = min(max_steps, budget.remaining)
    for _ in range(limit):
        action = act_fn(state)
        next_raw, reward, terminated, truncated, _ = env.step(action)
        next_state = np.asarray(next_raw, dtype=np.float32)
        budget.consume_rl(1)
        buffer.add(
            Transition(
                state=state[None],
                action=action[None],
                reward=np.asarray([reward], dtype=np.float32),
                next_state=next_state[None],
                done=np.asarray([float(terminated)], dtype=np.float32),
            )
        )
        ep_return += float(reward)
        steps += 1
        state = next_state
        if terminated or truncated:
            break
    return ep_return, steps


def _initial_alive_mask(
    pop_size: int, active_mask: np.ndarray | None
) -> np.ndarray:
    if active_mask is None:
        return np.ones(pop_size, dtype=bool)
    mask = np.asarray(active_mask, dtype=bool)
    if mask.shape != (pop_size,):
        raise ValueError(
            f"active_mask must have shape ({pop_size},), got {mask.shape}"
        )
    return mask.copy()


class _PrefixTracker:
    """Accumulates discounted returns over the first h steps of a rollout."""

    def __init__(self, h: int, discount: float, pop_size: int) -> None:
        self._h = h
        self._discount = discount
        self._returns = np.zeros(pop_size, dtype=np.float64)
        self._factor = 1.0
        self.snapshot: PrefixSnapshot | None = None

    def observe(
        self,
        step: int,
        active: np.ndarray,
        rewards: np.ndarray,
        states: np.ndarray,
        terminated: np.ndarray,
        steps_taken: np.ndarray,
    ) -> None:
        if step >= self._h:
            return
        self._returns[active] += self._factor * rewards
        self._factor *= self._discount
        if step == self._h - 1:
            self.finish(states, terminated, steps_taken)

    def finish(
        self,
        states: np.ndarray,
        terminated: np.ndarray,
        steps_taken: np.ndarray,
    ) -> PrefixSnapshot:
        if self.snapshot is None:
            self.snapshot = PrefixSnapshot(
                returns=self._returns.copy(),
                states=states.copy(),
                terminated=terminated.copy(),
                steps_taken=steps_taken.copy(),
            )
        return self.snapshot


def rollout_population(
    pool: EnvPool,
    pop_act_fn: Callable[[np.ndarray], np.ndarray],
    buffer: ReplayBuffer,
    budget: Budget,
    *,
    max_steps: int,
    discount: float,
    seed: int,
    active_mask: np.ndarray | None = None,
    prefix_h: int | None = None,
    prefix_discount: float = 1.0,
) -> PopRolloutResult:
    """Roll out active population slots in parallel, stopping when done.

    With prefix_h, also snapshot the state after the first prefix_h steps.
    """
    pop_size = pool.num_envs
    alive = _initial_alive_mask(pop_size, active_mask)
    terminated_all = np.zeros(pop_size, dtype=bool)
    returns = np.zeros(pop_size, dtype=np.float64)
    steps_taken = np.zeros(pop_size, dtype=np.int32)
    if not alive.any() or budget.exhausted or max_steps <= 0:
        return PopRolloutResult(
            returns=returns,
            last_states=np.zeros((pop_size, 0), dtype=np.float32),
            terminated=terminated_all,
            steps_taken=steps_taken,
            total_steps=0,
        )

    states = pool.reset(seed)
    total_steps = 0
    factor = 1.0
    tracker = (
        None
        if prefix_h is None
        else _PrefixTracker(prefix_h, prefix_discount, pop_size)
    )

    for step in range(max_steps):
        active = np.flatnonzero(alive)
        if len(active) == 0 or budget.remaining == 0:
            break
        if len(active) > budget.remaining:
            active = active[: budget.remaining]
        actions = pop_act_fn(states)
        active_actions = actions[active]
        next_s, rewards, terms, truncs = pool.step(active, active_actions)
        step_count = len(active)
        budget.consume_pop(step_count)
        total_steps += step_count
        buffer.add(
            Transition(
                state=states[active],
                action=active_actions,
                reward=rewards,
                next_state=next_s,
                done=terms.astype(np.float32),
            )
        )
        states[active] = next_s
        steps_taken[active] += 1
        returns[active] += factor * rewards
        terminated_all[active] |= terms
        alive[active] &= ~(terms | truncs)
        factor *= discount
        if tracker is not None:
            tracker.observe(
                step, active, rewards, states, terminated_all, steps_taken
            )

    prefix = (
        None
        if tracker is None
        else tracker.finish(states, terminated_all, steps_taken)
    )
    return PopRolloutResult(
        returns=returns,
        last_states=states,
        terminated=terminated_all,
        steps_taken=steps_taken,
        total_steps=total_steps,
        prefix=prefix,
    )
