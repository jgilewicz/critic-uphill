from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from common.replay_buffer import ReplayBuffer
from common.td3 import FQESettings
from common.utils import critic_point
from config import TD3RLConfig
from modules.protocols import QCritic, StateEmbedding


@eqx.filter_jit
def _act_deterministic(
    embedding: StateEmbedding,
    actor: Any,
    state: jax.Array,
) -> jax.Array:
    return actor(embedding(state))


@eqx.filter_jit
def _q1_at_actor(
    critic: QCritic,
    embedding: StateEmbedding,
    actor: Any,
    state: jax.Array,
) -> jax.Array:
    return critic(state, actor(embedding(state)))[0]


@eqx.filter_jit
def _act_population(
    embedding: StateEmbedding,
    heads: Any,
    states: jax.Array,
) -> jax.Array:
    z = jax.vmap(embedding)(states)
    return eqx.filter_vmap(lambda head, zi: head(zi))(heads, z)


@eqx.filter_jit
def _q1_population_sum(
    critic: QCritic,
    embedding: StateEmbedding,
    heads: Any,
    states: jax.Array,
) -> jax.Array:
    z = jax.vmap(embedding)(states)

    def sum_q(head: Any) -> jax.Array:
        actions = jax.vmap(head)(z)
        return jnp.sum(critic_point(critic, states, actions)[:, 0])

    return eqx.filter_vmap(sum_q)(heads)


@eqx.filter_jit
def _q1_population_states(
    critic: QCritic,
    embedding: StateEmbedding,
    heads: Any,
    states: jax.Array,
) -> jax.Array:
    z = jax.vmap(embedding)(states)
    actions = eqx.filter_vmap(lambda head, zi: head(zi))(heads, z)
    return critic_point(critic, states, actions)[:, 0]


class OfflineAgent(abc.ABC):
    """Read-only agent view used by offline measurement and diagnostics.

    Exposes the actor as embedding Z(s) plus a head and the critic the actor
    follows as critic1, with batched deterministic actions and Q helpers.
    """

    _unravel_head: Callable[[jax.Array], Any]

    def __init__(
        self, obs_dim: int, action_dim: int, action_limit: float
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_limit = action_limit

    @property
    @abc.abstractmethod
    def embedding(self) -> StateEmbedding:
        """Shared online state encoder Z(s)."""

    @property
    @abc.abstractmethod
    def actor_head(self) -> Any:
        """Online actor head PyTree."""

    @property
    @abc.abstractmethod
    def critic1(self) -> QCritic:
        """Critic the actor follows, used by surrogate fitness and signals."""

    @property
    @abc.abstractmethod
    def critic2(self) -> QCritic:
        """Second critic scored as signal q2."""

    @property
    def signal_mapping(self) -> dict[str, str]:
        """Which network each critic signal of the measurement refers to."""
        return {
            "q1": "online critic1",
            "q2": "online critic2",
            "shadow": "first shadow critic head (None without a shadow)",
        }

    @property
    def shadow_critic(self) -> QCritic | None:
        """First head of the shadow critic, None without one."""
        return None

    @property
    @abc.abstractmethod
    def fqe(self) -> FQESettings:
        """FQE hyperparameters of the offline measurement."""

    @property
    def unravel_head(self) -> Callable[[jax.Array], Any]:
        """Map a flat vector back into the agent's actor head PyTree."""
        return self._unravel_head

    def actor_head_flat(self) -> jax.Array:
        """Flatten the current online actor head parameters."""
        return ravel_pytree(self.actor_head)[0]

    def act_with_head(
        self,
        head: Any,
        state: np.ndarray,
        *,
        embedding: StateEmbedding | None = None,
    ) -> np.ndarray:
        """Compute the deterministic action of any head over Z(s).

        Uses the online embedding unless a snapshot embedding is given.
        """
        encoder = self.embedding if embedding is None else embedding
        action = _act_deterministic(encoder, head, jnp.asarray(state))
        return np.asarray(action, dtype=np.float32)

    def act_deterministic(self, state: np.ndarray) -> np.ndarray:
        """Compute deterministic RL actor action for evaluation."""
        return self.act_with_head(self.actor_head, state)

    def q1_at_actor(self, state: np.ndarray) -> float:
        """Online Q1(s, pi(s)) for the deterministic RL actor."""
        q = _q1_at_actor(
            self.critic1, self.embedding, self.actor_head, jnp.asarray(state)
        )
        return float(q)

    def act_population(self, heads: Any, states: np.ndarray) -> np.ndarray:
        """Compute deterministic actions for stacked population heads."""
        actions = _act_population(self.embedding, heads, jnp.asarray(states))
        return np.asarray(actions, dtype=np.float32)

    def q1_population_mean(
        self,
        heads: Any,
        states: jax.Array,
        *,
        chunk_size: int = 4096,
        critic: QCritic | None = None,
    ) -> jax.Array:
        """Mean Q(s, head(Z(s))) over states for each head.

        Uses online Q1 unless another critic is given.
        """
        scorer = self.critic1 if critic is None else critic
        total = int(states.shape[0])
        pop_size = int(jax.tree.leaves(heads)[0].shape[0])
        acc = jnp.zeros(pop_size, dtype=jnp.float32)
        for start in range(0, total, chunk_size):
            batch_states = states[start : start + chunk_size]
            acc = acc + _q1_population_sum(
                scorer, self.embedding, heads, batch_states
            )
        return acc / float(total)

    def q1_population_states(self, heads: Any, states: np.ndarray) -> jax.Array:
        """Online Q1(s_i, head_i(Z(s_i))) for one state per individual."""
        return _q1_population_states(
            self.critic1, self.embedding, heads, jnp.asarray(states)
        )


class RLAgent(OfflineAgent):
    """Abstract off-policy RL agent compatible with RLTrainer and EvoRainbow."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_limit: float,
        embedding_dim: int,
        *,
        rl_cfg: TD3RLConfig,
        key: jax.Array,
    ) -> None:
        del rl_cfg, key
        super().__init__(obs_dim, action_dim, action_limit)
        self.embedding_dim = embedding_dim

    @abc.abstractmethod
    def set_actor_head(self, head: Any) -> None:
        """Replace the online RL actor head PyTree."""

    @abc.abstractmethod
    def init_cem_head(self, key: jax.Array) -> jax.Array:
        """Return a flattened unscaled head to initialize CEM's mean."""

    @abc.abstractmethod
    def act(self, state: np.ndarray, key: jax.Array) -> np.ndarray:
        """Sample an exploration action."""

    @abc.abstractmethod
    def update(
        self, buffer: ReplayBuffer, num_updates: int, key: jax.Array
    ) -> dict[str, float]:
        """Run num_updates RL gradient steps and return logged loss metrics."""

    @abc.abstractmethod
    def soft_update_actor_from(self, head: Any, tau: float) -> None:
        """Blend elite head into RL actor (and target actor if present)."""
