from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from agents.base import RLAgent
from common.replay_buffer import ReplayBuffer
from common.td3 import (
    TD3Hyperparams,
    TD3State,
    init_td3,
    make_td3_train,
)
from common.utils import genetic_soft_update, make_adam
from config import RLConfig, TD3RLConfig
from modules.deep_modules import ActorHead, Critic, SharedStateEmbedding


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


class TD3(RLAgent):
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
        if not isinstance(rl_cfg, TD3RLConfig):
            raise TypeError(
                f"TD3 requires TD3RLConfig, got {type(rl_cfg).__name__}"
            )
        super().__init__(
            obs_dim,
            action_dim,
            action_limit,
            embedding_dim,
            rl_cfg=rl_cfg,
            key=key,
        )
        self.rl_cfg = rl_cfg

        embedding_opt = make_adam(1e-3)
        actor_opt = make_adam(rl_cfg.actor_lr)
        critic_opt = make_adam(rl_cfg.critic_lr)
        self._optimizers = (embedding_opt, actor_opt, critic_opt)
        self.state: TD3State = self._init_state(key)
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

    def _init_state(self, key: jax.Array) -> TD3State:
        embedding_opt, actor_opt, critic_opt = self._optimizers
        return init_td3(
            self.obs_dim,
            self.action_dim,
            self.embedding_dim,
            action_limit=self.action_limit,
            key=key,
            hidden_dims=self.rl_cfg.critic_hidden_dims,
            embedding_optimizer=embedding_opt,
            actor_optimizer=actor_opt,
            critic_optimizer=critic_opt,
        )

    def reset(self, key: jax.Array) -> None:
        """Re-initialize all networks, targets and optimizer states."""
        self.state = self._init_state(key)

    @property
    def embedding(self) -> SharedStateEmbedding:
        """Shared online state encoder Z(s)."""
        return self.state.online.embedding

    @property
    def actor_head(self) -> ActorHead:
        """Online TD3 ActorHead PyTree."""
        return self.state.online.actor

    def set_actor_head(self, head: ActorHead) -> None:
        """Replace the online TD3 ActorHead PyTree."""
        online = self.state.online._replace(actor=head)
        self.state = self.state._replace(online=online)

    @property
    def critic1(self) -> Critic:
        """Online Q1 critic."""
        return self.state.online.critic1

    def init_cem_head(self, key: jax.Array) -> jax.Array:
        """Sample an unscaled ActorHead vector to initialize CEM's mean."""
        head = ActorHead(
            self.embedding_dim,
            self.action_dim,
            key=key,
            action_limit=self.action_limit,
        )
        return ravel_pytree(head)[0]

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

    def update(
        self, buffer: ReplayBuffer, num_updates: int, key: jax.Array
    ) -> dict[str, float]:
        """Run num_updates TD3 gradient steps and return mean loss stats."""
        if num_updates <= 0 or len(buffer) == 0:
            return {}
        self.state, stats = self._train_fn(
            self.state,
            buffer.data,
            jnp.asarray(len(buffer)),
            jnp.asarray(num_updates),
            key,
        )
        return {
            "critic/td_loss": float(stats.critic_loss),
            "critic/q_mean": float(stats.q_mean),
            "actor/loss": float(stats.actor_loss),
        }

    def soft_update_actor_from(self, head: ActorHead, tau: float) -> None:
        """Genetic soft update: blend elite head into online & target actor."""
        if tau == 0.0:
            return
        self.set_actor_head(genetic_soft_update(self.actor_head, head, tau))
        target = self.state.target._replace(
            actor=genetic_soft_update(self.state.target.actor, head, tau)
        )
        self.state = self.state._replace(target=target)
