from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

from common.replay_buffer import ReplayBuffer
from common.td3 import (
    TD3Hyperparams,
    TD3State,
    critic_point,
    init_td3,
    make_td3_train,
)
from common.utils import genetic_soft_update
from config import RLConfig
from modules.deep_modules import ActorHead, Critic, SharedStateEmbedding


@eqx.filter_jit
def _act_deterministic(
    embedding: SharedStateEmbedding,
    actor: ActorHead,
    state: jax.Array,
) -> jax.Array:
    return actor(embedding(state))


@eqx.filter_jit
def _act_noisy(
    embedding: SharedStateEmbedding,
    actor: ActorHead,
    state: jax.Array,
    key: jax.Array,
    *,
    exploration_noise: float,
    action_limit: float,
) -> jax.Array:
    action = actor(embedding(state))
    noise = jax.random.normal(key, action.shape) * exploration_noise
    return jnp.clip(action + noise, -action_limit, action_limit)


@eqx.filter_jit
def _act_population(
    embedding: SharedStateEmbedding,
    heads: ActorHead,
    states: jax.Array,
) -> jax.Array:
    z = jax.vmap(embedding)(states)
    return eqx.filter_vmap(lambda head, zi: head(zi))(heads, z)


@eqx.filter_jit
def _q1_population_sum(
    critic: Critic,
    embedding: SharedStateEmbedding,
    heads: ActorHead,
    states: jax.Array,
) -> jax.Array:
    z = jax.vmap(embedding)(states)

    def sum_q(head: ActorHead) -> jax.Array:
        actions = jax.vmap(head)(z)
        return jnp.sum(critic_point(critic, states, actions)[:, 0])

    return eqx.filter_vmap(sum_q)(heads)


@eqx.filter_jit
def _q1_population_states(
    critic: Critic,
    embedding: SharedStateEmbedding,
    heads: ActorHead,
    states: jax.Array,
) -> jax.Array:
    z = jax.vmap(embedding)(states)
    actions = eqx.filter_vmap(lambda head, zi: head(zi))(heads, z)
    return critic_point(critic, states, actions)[:, 0]


_GRAD_CLIP_NORM = 10.0


def _make_adam(lr: float) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(_GRAD_CLIP_NORM),
        optax.adam(lr),
    )


class TD3:
    """TD3 agent owning actor embedding/head, twin critics, and optimizers."""

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
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.embedding_dim = embedding_dim
        self.rl_cfg = rl_cfg

        embedding_opt = _make_adam(1e-3)
        actor_opt = _make_adam(rl_cfg.actor_lr)
        critic_opt = _make_adam(rl_cfg.critic_lr)
        self.state: TD3State = init_td3(
            obs_dim,
            action_dim,
            embedding_dim,
            action_limit=action_limit,
            key=key,
            hidden_dims=rl_cfg.critic_hidden_dims,
            embedding_optimizer=embedding_opt,
            actor_optimizer=actor_opt,
            critic_optimizer=critic_opt,
        )
        hyperparams = TD3Hyperparams(
            gamma=rl_cfg.gamma,
            tau=rl_cfg.tau,
            policy_noise=rl_cfg.policy_noise,
            noise_clip=rl_cfg.noise_clip,
            action_limit=action_limit,
            policy_freq=rl_cfg.policy_freq,
            batch_size=rl_cfg.batch_size,
        )
        self._train_fn = make_td3_train(
            hyperparams,
            embedding_optimizer=embedding_opt,
            actor_optimizer=actor_opt,
            critic_optimizer=critic_opt,
        )
        _, self._unravel_head = ravel_pytree(self.state.online.actor)

    @property
    def unravel_head(self) -> Callable[[jax.Array], ActorHead]:
        """Map a flat vector back into an ActorHead PyTree."""
        return self._unravel_head

    def actor_head_flat(self) -> jax.Array:
        """Flatten the current online RL actor head parameters."""
        return ravel_pytree(self.state.online.actor)[0]

    def act(self, state: np.ndarray, key: jax.Array) -> np.ndarray:
        """Sample an exploration action with clipped Gaussian noise."""
        online = self.state.online
        action = _act_noisy(
            online.embedding,
            online.actor,
            jnp.asarray(state),
            key,
            exploration_noise=self.rl_cfg.exploration_noise,
            action_limit=self.action_limit,
        )
        return np.asarray(action, dtype=np.float32)

    def act_deterministic(self, state: np.ndarray) -> np.ndarray:
        """Compute deterministic RL actor action for evaluation."""
        online = self.state.online
        action = _act_deterministic(
            online.embedding, online.actor, jnp.asarray(state)
        )
        return np.asarray(action, dtype=np.float32)

    def act_population(
        self, heads: ActorHead, states: np.ndarray
    ) -> np.ndarray:
        """Compute actions for stacked population heads on per-env states."""
        actions = _act_population(
            self.state.online.embedding, heads, jnp.asarray(states)
        )
        return np.asarray(actions, dtype=np.float32)

    def q1_population_mean(
        self, heads: ActorHead, states: jax.Array, *, chunk_size: int = 4096
    ) -> jax.Array:
        """Mean online Q1(s, head(Z(s))) over states for each head."""
        total = int(states.shape[0])
        online = self.state.online
        acc = jnp.zeros(heads.out_layer.weight.shape[0], dtype=jnp.float32)
        for start in range(0, total, chunk_size):
            batch_states = states[start : start + chunk_size]
            acc = acc + _q1_population_sum(
                online.critic1, online.embedding, heads, batch_states
            )
        return acc / float(total)

    def q1_population_states(
        self, heads: ActorHead, states: np.ndarray
    ) -> jax.Array:
        """Online Q1(s_i, head_i(Z(s_i))) for one state per individual."""
        online = self.state.online
        return _q1_population_states(
            online.critic1, online.embedding, heads, jnp.asarray(states)
        )

    def update(
        self, buffer: ReplayBuffer, num_updates: int, key: jax.Array
    ) -> float:
        """Run num_updates TD3 gradient steps and return mean critic loss."""
        if num_updates <= 0 or len(buffer) == 0:
            return 0.0
        self.state, critic_loss = self._train_fn(
            self.state,
            buffer.data,
            jnp.asarray(len(buffer)),
            jnp.asarray(num_updates),
            key,
        )
        return float(critic_loss)

    def soft_update_actor_from(self, head: ActorHead, tau: float) -> None:
        """Genetic soft update: blend elite head into online & target actor."""
        if tau == 0.0:
            return
        online = self.state.online._replace(
            actor=genetic_soft_update(self.state.online.actor, head, tau)
        )
        target = self.state.target._replace(
            actor=genetic_soft_update(self.state.target.actor, head, tau)
        )
        self.state = self.state._replace(online=online, target=target)
