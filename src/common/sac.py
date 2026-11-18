from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from common.replay_buffer import BufferData, sample_batch
from common.utils import critic_loss, critic_point, genetic_soft_update
from modules.deep_modules import (
    Critic,
    GaussianActorHead,
    SharedStateEmbedding,
)


class SACNetworks(NamedTuple):
    """Online network tuple for SAC."""

    embedding: SharedStateEmbedding
    actor: GaussianActorHead
    critic1: Critic
    critic2: Critic


class SACState(NamedTuple):
    """Full trainable and target state of SAC."""

    online: SACNetworks
    target_critic1: Critic
    target_critic2: Critic
    log_alpha: jax.Array
    embedding_opt_state: optax.OptState
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    alpha_opt_state: optax.OptState


class SACHyperparams(NamedTuple):
    """Static SAC hyperparameters passed into jitted updates."""

    gamma: float = 0.99
    tau: float = 0.005
    target_entropy: float = -1.0
    batch_size: int = 128


class SACOptimizers(NamedTuple):
    """Optimizers for SAC components."""

    embedding: optax.GradientTransformation
    actor: optax.GradientTransformation
    critic: optax.GradientTransformation
    alpha: optax.GradientTransformation


Batch = dict[str, jnp.ndarray]
SACTrainFn = Callable[
    [SACState, BufferData, jax.Array, jax.Array, jax.Array],
    tuple[SACState, jax.Array, jax.Array, jax.Array],
]


def init_sac(
    state_dim: int,
    action_dim: int,
    embedding_dim: int,
    *,
    action_limit: float,
    key: jax.Array,
    hidden_dims: tuple[int, int],
    optimizers: SACOptimizers,
) -> SACState:
    """Initialize online/target SAC nets, log_alpha, and optimizer states."""
    ek, ak, c1k, c2k = jax.random.split(key, 4)
    embedding = SharedStateEmbedding(state_dim, embedding_dim, key=ek)
    actor = GaussianActorHead(
        embedding_dim,
        action_dim,
        key=ak,
        action_limit=action_limit,
        init_scale=0.1,
    )
    critic1 = Critic(state_dim, action_dim, key=c1k, hidden_dims=hidden_dims)
    critic2 = Critic(state_dim, action_dim, key=c2k, hidden_dims=hidden_dims)
    online = SACNetworks(embedding, actor, critic1, critic2)
    log_alpha = jnp.zeros((), dtype=jnp.float32)
    return SACState(
        online=online,
        target_critic1=critic1,
        target_critic2=critic2,
        log_alpha=log_alpha,
        embedding_opt_state=optimizers.embedding.init(
            eqx.filter(embedding, eqx.is_array)
        ),
        actor_opt_state=optimizers.actor.init(eqx.filter(actor, eqx.is_array)),
        critic_opt_state=optimizers.critic.init(
            eqx.filter((critic1, critic2), eqx.is_array)
        ),
        alpha_opt_state=optimizers.alpha.init(log_alpha),
    )


def _sac_td_target(
    cfg: SACHyperparams,
    state: SACState,
    batch: Batch,
    key: jax.Array,
) -> jax.Array:
    z_next = jax.vmap(state.online.embedding)(batch["next_state"])
    keys = jax.random.split(key, cfg.batch_size)
    next_actions, next_log_probs = jax.vmap(
        state.online.actor.sample_and_log_prob
    )(z_next, keys)
    q_next = jnp.minimum(
        critic_point(state.target_critic1, batch["next_state"], next_actions),
        critic_point(state.target_critic2, batch["next_state"], next_actions),
    )
    alpha = jax.lax.stop_gradient(jnp.exp(state.log_alpha))
    v_next = q_next - alpha * next_log_probs[:, None]
    return jax.lax.stop_gradient(
        batch["reward"] + cfg.gamma * (1.0 - batch["done"]) * v_next
    )


def _critic_update(
    cfg: SACHyperparams,
    optimizer: optax.GradientTransformation,
    state: SACState,
    batch: Batch,
    key: jax.Array,
) -> tuple[SACState, jax.Array]:
    target_q = _sac_td_target(cfg, state, batch, key)
    s, a = batch["state"], batch["action"]

    def loss_fn(critics: tuple[Critic, Critic]) -> jax.Array:
        c1, c2 = critics
        return critic_loss(c1, s, a, target_q) + critic_loss(c2, s, a, target_q)

    critics = (state.online.critic1, state.online.critic2)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(critics)
    updates, opt_state = optimizer.update(
        grads, state.critic_opt_state, eqx.filter(critics, eqx.is_array)
    )
    critic1, critic2 = eqx.apply_updates(critics, updates)
    new_online = state.online._replace(critic1=critic1, critic2=critic2)
    return state._replace(online=new_online, critic_opt_state=opt_state), loss


def _actor_and_alpha_update(
    cfg: SACHyperparams,
    optimizers: SACOptimizers,
    state: SACState,
    batch: Batch,
    key: jax.Array,
) -> tuple[SACState, jax.Array, jax.Array]:
    alpha_val = jax.lax.stop_gradient(jnp.exp(state.log_alpha))
    sample_keys = jax.random.split(key, cfg.batch_size)

    def actor_loss_fn(
        embedding_actor: tuple[SharedStateEmbedding, GaussianActorHead],
    ) -> tuple[jax.Array, jax.Array]:
        embedding, actor = embedding_actor
        z = jax.vmap(embedding)(batch["state"])
        actions, log_probs = jax.vmap(actor.sample_and_log_prob)(z, sample_keys)
        q_pi = jnp.minimum(
            critic_point(state.online.critic1, batch["state"], actions),
            critic_point(state.online.critic2, batch["state"], actions),
        )
        loss = jnp.mean(alpha_val * log_probs[:, None] - q_pi)
        return loss, log_probs

    embedding_actor = (state.online.embedding, state.online.actor)
    (actor_loss, log_probs), grads = eqx.filter_value_and_grad(
        actor_loss_fn, has_aux=True
    )(embedding_actor)
    emb_grads, act_grads = grads
    emb_updates, emb_opt_state = optimizers.embedding.update(
        emb_grads,
        state.embedding_opt_state,
        eqx.filter(state.online.embedding, eqx.is_array),
    )
    act_updates, act_opt_state = optimizers.actor.update(
        act_grads,
        state.actor_opt_state,
        eqx.filter(state.online.actor, eqx.is_array),
    )
    embedding = eqx.apply_updates(state.online.embedding, emb_updates)
    actor = eqx.apply_updates(state.online.actor, act_updates)

    stopped_log_probs = jax.lax.stop_gradient(log_probs)

    def alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
        return -jnp.mean(
            jnp.exp(log_alpha) * (stopped_log_probs + cfg.target_entropy)
        )

    alpha_grad = jax.grad(alpha_loss_fn)(state.log_alpha)
    alpha_updates, alpha_opt_state = optimizers.alpha.update(
        alpha_grad, state.alpha_opt_state, state.log_alpha
    )
    new_log_alpha = jnp.asarray(
        optax.apply_updates(state.log_alpha, alpha_updates)
    )

    new_online = state.online._replace(embedding=embedding, actor=actor)
    new_state = state._replace(
        online=new_online,
        target_critic1=genetic_soft_update(
            state.target_critic1, state.online.critic1, cfg.tau
        ),
        target_critic2=genetic_soft_update(
            state.target_critic2, state.online.critic2, cfg.tau
        ),
        log_alpha=new_log_alpha,
        embedding_opt_state=emb_opt_state,
        actor_opt_state=act_opt_state,
        alpha_opt_state=alpha_opt_state,
    )
    return new_state, actor_loss, jnp.exp(new_log_alpha)


def make_sac_train(
    cfg: SACHyperparams,
    *,
    optimizers: SACOptimizers,
) -> SACTrainFn:
    """Compile a fused multi-step SAC update loop."""

    @eqx.filter_jit
    def train(
        state: SACState,
        data: BufferData,
        size: jax.Array,
        num_updates: jax.Array,
        key: jax.Array,
    ) -> tuple[SACState, jax.Array, jax.Array, jax.Array]:
        dynamic, static = eqx.partition(state, eqx.is_array)

        def body(
            _step: jax.Array,
            carry: tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array]:
            dyn, rng, critic_sum, actor_sum, alpha_sum = carry
            rng, sample_key, critic_key, actor_key = jax.random.split(rng, 4)
            batch = sample_batch(data, size, sample_key, cfg.batch_size)
            cur_state, c_loss = _critic_update(
                cfg,
                optimizers.critic,
                eqx.combine(dyn, static),
                batch,
                critic_key,
            )
            cur_state, a_loss, alpha = _actor_and_alpha_update(
                cfg, optimizers, cur_state, batch, actor_key
            )
            return (
                eqx.filter(cur_state, eqx.is_array),
                rng,
                critic_sum + c_loss,
                actor_sum + a_loss,
                alpha_sum + alpha,
            )

        zeros = jnp.zeros(())
        dynamic, _, critic_sum, actor_sum, alpha_sum = jax.lax.fori_loop(
            0, num_updates, body, (dynamic, key, zeros, zeros, zeros)
        )
        denom = jnp.maximum(num_updates, 1)
        return (
            eqx.combine(dynamic, static),
            critic_sum / denom,
            actor_sum / denom,
            alpha_sum / denom,
        )

    return train
