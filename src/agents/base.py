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
from common.utils import critic_point
from config import RLConfig
from modules.deep_modules import Critic, SharedStateEmbedding


@eqx.filter_jit
def _act_deterministic(
    embedding: SharedStateEmbedding,
    actor: Any,
    state: jax.Array,
) -> jax.Array:
    return actor(embedding(state))


@eqx.filter_jit
def _act_population(
    embedding: SharedStateEmbedding,
    heads: Any,
    states: jax.Array,
) -> jax.Array:
    z = jax.vmap(embedding)(states)
    return eqx.filter_vmap(lambda head, zi: head(zi))(heads, z)


@eqx.filter_jit
def _q1_population_sum(
    critic: Critic,
    embedding: SharedStateEmbedding,
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
    critic: Critic,
    embedding: SharedStateEmbedding,
    heads: Any,
    states: jax.Array,
) -> jax.Array:
    z = jax.vmap(embedding)(states)
    actions = eqx.filter_vmap(lambda head, zi: head(zi))(heads, z)
    return critic_point(critic, states, actions)[:, 0]


class RLAgent(abc.ABC):
    """Abstract off-policy RL agent compatible with RLTrainer and EvoRainbow."""

    _unravel_head: Callable[[jax.Array], Any]

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_limit: float,
        embedding_dim: int,
        *,
        rl_cfg: RLConfig,
        key: jax.Array,
    ) -> None:
        del rl_cfg, key
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.embedding_dim = embedding_dim

    @property
    @abc.abstractmethod
    def embedding(self) -> SharedStateEmbedding:
        """Shared online state encoder Z(s)."""

    @property
    @abc.abstractmethod
    def actor_head(self) -> Any:
        """Online RL actor head PyTree evolved by CEM."""

    @abc.abstractmethod
    def set_actor_head(self, head: Any) -> None:
        """Replace the online RL actor head PyTree."""

    @property
    @abc.abstractmethod
    def critic1(self) -> Critic:
        """Online Q1 critic used by surrogate fitness estimators."""

    @property
    def unravel_head(self) -> Callable[[jax.Array], Any]:
        """Map a flat vector back into the agent's actor head PyTree."""
        return self._unravel_head

    def actor_head_flat(self) -> jax.Array:
        """Flatten the current online RL actor head parameters."""
        return ravel_pytree(self.actor_head)[0]

    @abc.abstractmethod
    def init_cem_head(self, key: jax.Array) -> jax.Array:
        """Return a flattened unscaled head to initialize CEM's mean."""

    @abc.abstractmethod
    def act(self, state: np.ndarray, key: jax.Array) -> np.ndarray:
        """Sample an exploration action."""

    def act_deterministic(self, state: np.ndarray) -> np.ndarray:
        """Compute deterministic RL actor action for evaluation."""
        action = _act_deterministic(
            self.embedding, self.actor_head, jnp.asarray(state)
        )
        return np.asarray(action, dtype=np.float32)

    def act_population(self, heads: Any, states: np.ndarray) -> np.ndarray:
        """Compute deterministic actions for stacked population heads."""
        actions = _act_population(self.embedding, heads, jnp.asarray(states))
        return np.asarray(actions, dtype=np.float32)

    def q1_population_mean(
        self, heads: Any, states: jax.Array, *, chunk_size: int = 4096
    ) -> jax.Array:
        """Mean online Q1(s, head(Z(s))) over states for each head."""
        total = int(states.shape[0])
        pop_size = int(jax.tree.leaves(heads)[0].shape[0])
        acc = jnp.zeros(pop_size, dtype=jnp.float32)
        for start in range(0, total, chunk_size):
            batch_states = states[start : start + chunk_size]
            acc = acc + _q1_population_sum(
                self.critic1, self.embedding, heads, batch_states
            )
        return acc / float(total)

    def q1_population_states(self, heads: Any, states: np.ndarray) -> jax.Array:
        """Online Q1(s_i, head_i(Z(s_i))) for one state per individual."""
        return _q1_population_states(
            self.critic1, self.embedding, heads, jnp.asarray(states)
        )

    @abc.abstractmethod
    def update(
        self, buffer: ReplayBuffer, num_updates: int, key: jax.Array
    ) -> dict[str, float]:
        """Run num_updates RL gradient steps and return logged loss metrics."""

    @abc.abstractmethod
    def soft_update_actor_from(self, head: Any, tau: float) -> None:
        """Blend elite head into RL actor (and target actor if present)."""
