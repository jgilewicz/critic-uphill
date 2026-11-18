from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from agents.base import RLAgent
from common.replay_buffer import ReplayBuffer
from common.sac import (
    SACHyperparams,
    SACOptimizers,
    SACState,
    init_sac,
    make_sac_train,
)
from common.utils import genetic_soft_update, make_adam
from config import RLConfig, SACRLConfig
from modules.deep_modules import (
    Critic,
    GaussianActorHead,
    SharedStateEmbedding,
)


@eqx.filter_jit
def _act_stochastic(
    embedding: SharedStateEmbedding,
    actor: GaussianActorHead,
    state: jax.Array,
    key: jax.Array,
) -> jax.Array:
    action, _ = actor.sample_and_log_prob(embedding(state), key)
    return action


class SAC(RLAgent):
    """SAC agent with twin critics, Gaussian actor, and learned alpha."""

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
        if not isinstance(rl_cfg, SACRLConfig):
            raise TypeError(
                f"SAC requires SACRLConfig, got {type(rl_cfg).__name__}"
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

        optimizers = SACOptimizers(
            embedding=make_adam(rl_cfg.actor_lr),
            actor=make_adam(rl_cfg.actor_lr),
            critic=make_adam(rl_cfg.critic_lr),
            alpha=make_adam(rl_cfg.alpha_lr),
        )
        self.state: SACState = init_sac(
            obs_dim,
            action_dim,
            embedding_dim,
            action_limit=action_limit,
            key=key,
            hidden_dims=rl_cfg.critic_hidden_dims,
            optimizers=optimizers,
        )
        hyperparams = SACHyperparams(
            gamma=rl_cfg.gamma,
            tau=rl_cfg.tau,
            target_entropy=-float(action_dim),
            batch_size=rl_cfg.batch_size,
        )
        self._train_fn = make_sac_train(hyperparams, optimizers=optimizers)
        _, self._unravel_head = ravel_pytree(self.state.online.actor)

    @property
    def embedding(self) -> SharedStateEmbedding:
        """Shared online state encoder Z(s)."""
        return self.state.online.embedding

    @property
    def actor_head(self) -> GaussianActorHead:
        """Online SAC GaussianActorHead PyTree."""
        return self.state.online.actor

    def set_actor_head(self, head: GaussianActorHead) -> None:
        """Replace the online SAC GaussianActorHead PyTree."""
        online = self.state.online._replace(actor=head)
        self.state = self.state._replace(online=online)

    @property
    def critic1(self) -> Critic:
        """Online Q1 critic."""
        return self.state.online.critic1

    def init_cem_head(self, key: jax.Array) -> jax.Array:
        """Sample an unscaled GaussianActorHead vector for CEM's mean."""
        head = GaussianActorHead(
            self.embedding_dim,
            self.action_dim,
            key=key,
            action_limit=self.action_limit,
        )
        return ravel_pytree(head)[0]

    def act(self, state: np.ndarray, key: jax.Array) -> np.ndarray:
        """Sample an exploration action from the squashed Gaussian policy."""
        online = self.state.online
        action = _act_stochastic(
            online.embedding, online.actor, jnp.asarray(state), key
        )
        return np.asarray(action, dtype=np.float32)

    def update(
        self, buffer: ReplayBuffer, num_updates: int, key: jax.Array
    ) -> dict[str, float]:
        """Run num_updates SAC steps and return mean critic/actor stats."""
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
            "actor/alpha": float(stats.alpha),
        }

    def soft_update_actor_from(
        self, head: GaussianActorHead, tau: float
    ) -> None:
        """Genetic soft update into online Gaussian actor (no actor target)."""
        if tau == 0.0:
            return
        self.set_actor_head(genetic_soft_update(self.actor_head, head, tau))
