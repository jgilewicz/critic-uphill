from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from common.replay_buffer import BufferData, sample_batch
from common.utils import critic_loss, critic_point, genetic_soft_update
from modules.deep_modules import ActorHead, Critic, SharedStateEmbedding


class TD3Networks(NamedTuple):
    """Online or target network tuple for TD3."""

    embedding: SharedStateEmbedding
    actor: ActorHead
    critic1: Critic
    critic2: Critic


class ShadowState(NamedTuple):
    """Shadow twin critic: online and target heads plus optimizer state."""

    online: tuple[Critic, Critic]
    target: tuple[Critic, Critic]
    opt_state: optax.OptState


class TD3State(NamedTuple):
    """Full trainable and target state of TD3."""

    online: TD3Networks
    target: TD3Networks
    embedding_opt_state: optax.OptState
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    shadow: ShadowState | None = None


class TD3Hyperparams(NamedTuple):
    """Static TD3 hyperparameters passed into jitted updates."""

    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    action_limit: float = 1.0
    policy_freq: int = 2
    batch_size: int = 128


class TD3Stats(NamedTuple):
    """Means over one fused update loop."""

    critic_loss: jax.Array
    q_mean: jax.Array
    actor_loss: jax.Array


Batch = dict[str, jnp.ndarray]
TrainFn = Callable[
    [TD3State, BufferData, jax.Array, jax.Array, jax.Array],
    tuple[TD3State, TD3Stats],
]


def init_td3(
    state_dim: int,
    action_dim: int,
    embedding_dim: int,
    *,
    action_limit: float,
    key: jax.Array,
    hidden_dims: tuple[int, int],
    embedding_optimizer: optax.GradientTransformation,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
) -> TD3State:
    """Initialize online/target TD3 networks and optimizer states."""
    ek, ak, c1k, c2k = jax.random.split(key, 4)
    embedding = SharedStateEmbedding(state_dim, embedding_dim, key=ek)
    actor = ActorHead(
        embedding_dim,
        action_dim,
        key=ak,
        action_limit=action_limit,
        init_scale=0.1,
    )
    critic1 = Critic(state_dim, action_dim, key=c1k, hidden_dims=hidden_dims)
    critic2 = Critic(state_dim, action_dim, key=c2k, hidden_dims=hidden_dims)
    online = TD3Networks(embedding, actor, critic1, critic2)
    return TD3State(
        online=online,
        target=online,
        embedding_opt_state=embedding_optimizer.init(
            eqx.filter(embedding, eqx.is_array)
        ),
        actor_opt_state=actor_optimizer.init(eqx.filter(actor, eqx.is_array)),
        critic_opt_state=critic_optimizer.init(
            eqx.filter((critic1, critic2), eqx.is_array)
        ),
    )


def init_shadow(
    state_dim: int,
    action_dim: int,
    *,
    key: jax.Array,
    hidden_dims: tuple[int, int],
    optimizer: optax.GradientTransformation,
) -> ShadowState:
    """Initialize a shadow twin critic with its own targets."""
    k1, k2 = jax.random.split(key)
    critics = (
        Critic(state_dim, action_dim, key=k1, hidden_dims=hidden_dims),
        Critic(state_dim, action_dim, key=k2, hidden_dims=hidden_dims),
    )
    return ShadowState(
        online=critics,
        target=critics,
        opt_state=optimizer.init(eqx.filter(critics, eqx.is_array)),
    )


def _next_action(
    cfg: TD3Hyperparams,
    target: TD3Networks,
    batch: Batch,
    key: jax.Array,
) -> jax.Array:
    noise = jnp.clip(
        jax.random.normal(key, batch["action"].shape) * cfg.policy_noise,
        -cfg.noise_clip,
        cfg.noise_clip,
    )
    z_next = jax.vmap(target.embedding)(batch["next_state"])
    return jnp.clip(
        jax.vmap(target.actor)(z_next) + noise,
        -cfg.action_limit,
        cfg.action_limit,
    )


def _td_target(
    cfg: TD3Hyperparams,
    critics: tuple[Critic, Critic],
    batch: Batch,
    next_action: jax.Array,
) -> jax.Array:
    q_next = jnp.minimum(
        critic_point(critics[0], batch["next_state"], next_action),
        critic_point(critics[1], batch["next_state"], next_action),
    )
    return jax.lax.stop_gradient(
        batch["reward"] + cfg.gamma * (1.0 - batch["done"]) * q_next
    )


def _twin_loss(
    critics: tuple[Critic, Critic], batch: Batch, target_q: jax.Array
) -> tuple[jax.Array, jax.Array]:
    c1, c2 = critics
    s, a = batch["state"], batch["action"]
    loss = critic_loss(c1, s, a, target_q) + critic_loss(c2, s, a, target_q)
    return loss, jnp.mean(critic_point(c1, s, a))


def _shadow_update(
    cfg: TD3Hyperparams,
    optimizer: optax.GradientTransformation,
    shadow: ShadowState,
    batch: Batch,
    next_action: jax.Array,
) -> ShadowState:
    target_q = _td_target(cfg, shadow.target, batch, next_action)
    grads = eqx.filter_grad(_twin_loss, has_aux=True)(
        shadow.online, batch, target_q
    )[0]
    updates, opt_state = optimizer.update(
        grads, shadow.opt_state, eqx.filter(shadow.online, eqx.is_array)
    )
    online = eqx.apply_updates(shadow.online, updates)
    return shadow._replace(online=online, opt_state=opt_state)


def _critic_update(
    cfg: TD3Hyperparams,
    optimizer: optax.GradientTransformation,
    state: TD3State,
    batch: Batch,
    key: jax.Array,
) -> tuple[TD3State, jax.Array, jax.Array]:
    next_action = _next_action(cfg, state.target, batch, key)
    target_q = _td_target(
        cfg, (state.target.critic1, state.target.critic2), batch, next_action
    )
    critics = (state.online.critic1, state.online.critic2)
    (loss, q_mean), grads = eqx.filter_value_and_grad(_twin_loss, has_aux=True)(
        critics, batch, target_q
    )
    updates, opt_state = optimizer.update(
        grads, state.critic_opt_state, eqx.filter(critics, eqx.is_array)
    )
    critic1, critic2 = eqx.apply_updates(critics, updates)
    new_online = state.online._replace(critic1=critic1, critic2=critic2)
    new_state = state._replace(online=new_online, critic_opt_state=opt_state)
    if state.shadow is not None:
        new_state = new_state._replace(
            shadow=_shadow_update(
                cfg, optimizer, state.shadow, batch, next_action
            )
        )
    return new_state, loss, q_mean


def _soft_update_shadow(
    shadow: ShadowState | None, tau: float
) -> ShadowState | None:
    if shadow is None:
        return None
    return shadow._replace(
        target=genetic_soft_update(shadow.target, shadow.online, tau)
    )


def _actor_update(
    cfg: TD3Hyperparams,
    embedding_optimizer: optax.GradientTransformation,
    actor_optimizer: optax.GradientTransformation,
    state: TD3State,
    batch: Batch,
) -> tuple[TD3State, jax.Array]:
    def loss_fn(
        embedding_actor: tuple[SharedStateEmbedding, ActorHead],
    ) -> jax.Array:
        embedding, actor = embedding_actor
        action = jax.vmap(actor)(jax.vmap(embedding)(batch["state"]))
        q = critic_point(state.online.critic1, batch["state"], action)
        return -jnp.mean(q)

    embedding_actor = (state.online.embedding, state.online.actor)
    actor_loss, (emb_grads, act_grads) = eqx.filter_value_and_grad(loss_fn)(
        embedding_actor
    )
    emb_updates, emb_opt_state = embedding_optimizer.update(
        emb_grads,
        state.embedding_opt_state,
        eqx.filter(state.online.embedding, eqx.is_array),
    )
    act_updates, act_opt_state = actor_optimizer.update(
        act_grads,
        state.actor_opt_state,
        eqx.filter(state.online.actor, eqx.is_array),
    )
    embedding = eqx.apply_updates(state.online.embedding, emb_updates)
    actor = eqx.apply_updates(state.online.actor, act_updates)
    new_target = state.target._replace(
        embedding=genetic_soft_update(
            state.target.embedding, embedding, cfg.tau
        ),
        actor=genetic_soft_update(state.target.actor, actor, cfg.tau),
        critic1=genetic_soft_update(
            state.target.critic1, state.online.critic1, cfg.tau
        ),
        critic2=genetic_soft_update(
            state.target.critic2, state.online.critic2, cfg.tau
        ),
    )
    new_online = state.online._replace(embedding=embedding, actor=actor)
    new_state = state._replace(
        online=new_online,
        target=new_target,
        embedding_opt_state=emb_opt_state,
        actor_opt_state=act_opt_state,
        shadow=_soft_update_shadow(state.shadow, cfg.tau),
    )
    return new_state, actor_loss


def make_td3_train(
    cfg: TD3Hyperparams,
    *,
    embedding_optimizer: optax.GradientTransformation,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
) -> TrainFn:
    """Compile a fused multi-step TD3 update loop."""

    @eqx.filter_jit
    def train(
        state: TD3State,
        data: BufferData,
        size: jax.Array,
        num_updates: jax.Array,
        key: jax.Array,
    ) -> tuple[TD3State, TD3Stats]:
        dynamic, static = eqx.partition(state, eqx.is_array)

        def body(
            step: jax.Array, carry: tuple[Any, jax.Array, jax.Array]
        ) -> tuple[Any, jax.Array, jax.Array]:
            dynamic, key, sums = carry
            key, sample_key, critic_key = jax.random.split(key, 3)
            batch = sample_batch(data, size, sample_key, cfg.batch_size)
            cur_state, loss, q_mean = _critic_update(
                cfg,
                critic_optimizer,
                eqx.combine(dynamic, static),
                batch,
                critic_key,
            )

            def with_actor(dyn: Any) -> tuple[Any, jax.Array]:
                updated, actor_loss = _actor_update(
                    cfg,
                    embedding_optimizer,
                    actor_optimizer,
                    eqx.combine(dyn, static),
                    batch,
                )
                return eqx.filter(updated, eqx.is_array), actor_loss

            next_dyn, actor_loss = jax.lax.cond(
                step % cfg.policy_freq == 0,
                with_actor,
                lambda dyn: (dyn, jnp.zeros(())),
                eqx.filter(cur_state, eqx.is_array),
            )
            did_actor = (step % cfg.policy_freq == 0).astype(jnp.float32)
            new_sums = sums + jnp.stack([loss, q_mean, actor_loss, did_actor])
            return next_dyn, key, new_sums

        dynamic, _, sums = jax.lax.fori_loop(
            0, num_updates, body, (dynamic, key, jnp.zeros(4))
        )
        n = jnp.maximum(num_updates, 1)
        stats = TD3Stats(
            critic_loss=sums[0] / n,
            q_mean=sums[1] / n,
            actor_loss=sums[2] / jnp.maximum(sums[3], 1.0),
        )
        return eqx.combine(dynamic, static), stats

    return train
