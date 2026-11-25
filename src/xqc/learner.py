from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from common.replay_buffer import BufferData, sample_batch
from config import XQCConfig
from xqc.layers import project_unit_norm, trainable
from xqc.networks import (
    Actor,
    CriticEnsemble,
    sample_tanh_gaussian,
)

N_CRITICS = 2
Batch = dict[str, jax.Array]


class XQCNets(eqx.Module):
    """Everything a checkpoint stores: actor, critics, target, temperature."""

    actor: Actor
    critic: CriticEnsemble
    target_critic: CriticEnsemble
    log_temp: jax.Array


class XQCOpt(NamedTuple):
    actor: Any
    critic: Any
    temperature: Any


class XQCState(NamedTuple):
    nets: XQCNets
    opt: XQCOpt
    rng: jax.Array


class XQCOptimizers(NamedTuple):
    actor: optax.GradientTransformation
    critic: optax.GradientTransformation
    temperature: optax.GradientTransformation


class XQCHyper(NamedTuple):
    """Static scalars of one update step."""

    gamma: float
    tau: float
    target_entropy: float


class XQCSpec(NamedTuple):
    hyper: XQCHyper
    tx: XQCOptimizers


class Noise(NamedTuple):
    """Standard-normal draws of the critic target and actor loss samples."""

    critic: jax.Array
    actor: jax.Array


def discount_for_horizon(horizon: int) -> float:
    """XQC heuristic: clip((T/5 - 1) / (T/5), 0.95, 0.995)."""
    scale = horizon / 5.0
    return min(max((scale - 1.0) / scale, 0.95), 0.995)


def make_optimizers(cfg: XQCConfig, total_steps: int) -> XQCOptimizers:
    steps = total_steps * cfg.updates_per_step

    def adam(lr: float) -> optax.GradientTransformation:
        return optax.adam(optax.linear_schedule(lr, cfg.lr_end, steps))

    def every_delay(step: jax.typing.ArrayLike, **_: Any) -> jax.Array:
        return jnp.mod(step, cfg.policy_delay) == 0

    def delayed(tx: optax.GradientTransformation) -> Any:
        return optax.conditionally_mask(tx, every_delay)

    return XQCOptimizers(
        actor=delayed(adam(cfg.actor_lr)),
        critic=adam(cfg.critic_lr),
        temperature=delayed(adam(cfg.actor_lr)),
    )


def make_spec(cfg: XQCConfig, action_dim: int, total_steps: int) -> XQCSpec:
    hyper = XQCHyper(
        gamma=discount_for_horizon(cfg.horizon),
        tau=cfg.tau,
        target_entropy=-action_dim / 2,
    )
    return XQCSpec(hyper, make_optimizers(cfg, total_steps))


def init_nets(
    cfg: XQCConfig, obs_dim: int, action_dim: int, key: jax.Array
) -> XQCNets:
    actor_key, critic_key = jax.random.split(key)
    actor = Actor(
        obs_dim, action_dim, tuple(cfg.actor_hidden_dims), key=actor_key
    )
    critic = CriticEnsemble(
        obs_dim,
        action_dim,
        tuple(cfg.critic_hidden_dims),
        num_bins=cfg.num_bins,
        n_critics=N_CRITICS,
        min_v=cfg.min_v,
        max_v=cfg.max_v,
        key=critic_key,
    )
    critic = project_unit_norm(critic)
    return XQCNets(
        actor=project_unit_norm(actor),
        critic=critic,
        target_critic=critic,
        log_temp=jnp.log(jnp.asarray(cfg.init_temperature, jnp.float32)),
    )


def init_state(nets: XQCNets, tx: XQCOptimizers, rng: jax.Array) -> XQCState:
    opt = XQCOpt(
        actor=tx.actor.init(eqx.filter(nets.actor, trainable(nets.actor))),
        critic=tx.critic.init(eqx.filter(nets.critic, trainable(nets.critic))),
        temperature=tx.temperature.init(nets.log_temp),
    )
    return XQCState(nets, opt, rng)


def _apply_gradient(
    tx: optax.GradientTransformation,
    opt_state: Any,
    model: Any,
    loss_fn: Callable[[Any], tuple[jax.Array, tuple[Any, dict[str, Any]]]],
) -> tuple[Any, Any, dict[str, Any]]:
    """One optimizer step on the trainable leaves of model.

    loss_fn returns the model with refreshed BatchNorm statistics, which
    replace the old ones whatever the optimizer decided.
    """
    params, rest = eqx.partition(model, trainable(model))

    def wrapped(p: Any) -> tuple[jax.Array, tuple[Any, dict[str, Any]]]:
        return loss_fn(eqx.combine(p, rest))

    grads, (stat_model, info) = jax.grad(wrapped, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    _, stats = eqx.partition(stat_model, trainable(stat_model))
    return eqx.combine(new_params, stats), opt_state, info


def categorical_td_loss(
    pred_log_probs: jax.Array,
    target_log_probs: jax.Array,
    *,
    reward: jax.Array,
    mask: jax.Array,
    entropy_term: jax.Array,
    gamma: float,
    support: tuple[float, float],
) -> jax.Array:
    """Two-hot cross entropy of one critic against the projected target."""
    min_v, max_v = support
    num_bins = pred_log_probs.shape[-1]
    bin_values = jnp.linspace(min_v, max_v, num_bins).reshape(1, -1)
    target_values = reward + gamma * (bin_values - entropy_term) * mask
    target_values = jnp.clip(target_values, min_v, max_v)
    b = (target_values - min_v) / ((max_v - min_v) / (num_bins - 1))
    lower = jnp.floor(b)
    upper = jnp.ceil(b)
    mask_shape = (-1, num_bins, num_bins)

    def one_hot(index: jax.Array) -> jax.Array:
        flat = index.astype(jnp.int32).reshape(-1)
        return jax.nn.one_hot(flat, num_bins).reshape(mask_shape)

    lower_mask = one_hot(lower)
    upper_mask = one_hot(upper)
    probs = jnp.exp(target_log_probs)
    m_lower = (
        probs * (upper + (lower == upper).astype(jnp.float32) - b)
    ).reshape(-1, num_bins, 1)
    m_upper = (probs * (b - lower)).reshape((-1, num_bins, 1))
    target = jax.lax.stop_gradient(
        jnp.sum(m_lower * lower_mask + m_upper * upper_mask, axis=1)
    )
    return -jnp.mean(jnp.sum(target * pred_log_probs, axis=1))


class _CriticTargets(NamedTuple):
    obs: jax.Array
    action: jax.Array
    log_probs: jax.Array


def _critic_targets(
    nets: XQCNets, batch: Batch, eps: jax.Array
) -> tuple[_CriticTargets, jax.Array]:
    """Joint (s, s') batch and the target log-probs of the lower critic."""
    batch_size = batch["state"].shape[0]
    mean, log_std, _ = nets.actor.forward(batch["next_state"], False)
    next_action, next_log_prob = sample_tanh_gaussian(mean, log_std, eps)
    obs = jnp.concatenate([batch["state"], batch["next_state"]])
    action = jnp.concatenate([batch["action"], next_action])
    logits, _ = nets.target_critic.forward(obs, action, True)
    logits = logits[:, batch_size:]
    min_index = jnp.argmin(nets.target_critic.expected_values(logits), axis=0)
    log_probs = jax.nn.log_softmax(logits, axis=-1)[
        min_index, jnp.arange(batch_size)
    ]
    repeated = jnp.repeat(log_probs[None], N_CRITICS, axis=0)
    return _CriticTargets(obs, action, repeated), next_log_prob


def _critic_update(
    nets: XQCNets, batch: Batch, eps: jax.Array, spec: XQCSpec, opt: Any
) -> tuple[CriticEnsemble, Any, dict[str, Any]]:
    targets, next_log_prob = _critic_targets(nets, batch, eps)
    entropy_term = jnp.exp(nets.log_temp) * next_log_prob[:, None]
    batch_size = batch["state"].shape[0]
    support = (nets.critic.min_v, nets.critic.max_v)

    def loss_fn(critic: CriticEnsemble) -> tuple[jax.Array, tuple[Any, Any]]:
        logits, new_critic = critic.forward(targets.obs, targets.action, True)
        pred = jax.nn.log_softmax(logits[:, :batch_size], axis=-1)
        losses = jax.vmap(
            lambda p, t: categorical_td_loss(
                p,
                t,
                reward=batch["reward"],
                mask=1.0 - batch["done"],
                entropy_term=entropy_term,
                gamma=spec.hyper.gamma,
                support=support,
            )
        )(pred, targets.log_probs)
        loss = jnp.sum(losses)
        q_mean = jnp.mean(critic.expected_values(logits[:, :batch_size]))
        return loss, (new_critic, {"critic_loss": loss, "q_mean": q_mean})

    return _apply_gradient(spec.tx.critic, opt, nets.critic, loss_fn)


def _actor_update(
    nets: XQCNets,
    critic: CriticEnsemble,
    batch: Batch,
    eps: jax.Array,
    spec: XQCSpec,
    opt: Any,
) -> tuple[Actor, Any, dict[str, Any]]:
    temperature = jnp.exp(nets.log_temp)

    def loss_fn(actor: Actor) -> tuple[jax.Array, tuple[Any, Any]]:
        mean, log_std, new_actor = actor.forward(batch["state"], True)
        action, log_prob = sample_tanh_gaussian(mean, log_std, eps)
        q = jnp.min(critic.q_eval(batch["state"], action), axis=0)
        loss = jnp.mean(log_prob * temperature - q)
        info = {"actor_loss": loss, "entropy": -jnp.mean(log_prob)}
        return loss, (new_actor, info)

    return _apply_gradient(spec.tx.actor, opt, nets.actor, loss_fn)


def _temperature_update(
    log_temp: jax.Array,
    entropy: jax.Array,
    spec: XQCSpec,
    opt: Any,
) -> tuple[jax.Array, Any]:
    def loss_fn(value: jax.Array) -> jax.Array:
        return jnp.exp(value) * (entropy - spec.hyper.target_entropy)

    updates, opt = spec.tx.temperature.update(
        jax.grad(loss_fn)(log_temp), opt, log_temp
    )
    return jnp.asarray(optax.apply_updates(log_temp, updates)), opt


def soft_update(
    online: CriticEnsemble, target: CriticEnsemble, tau: float
) -> CriticEnsemble:
    """Blend online params into the target, keeping the target's BN stats."""
    mask = trainable(online)
    online_params, _ = eqx.partition(online, mask)
    target_params, target_stats = eqx.partition(target, mask)
    mixed = jax.tree.map(
        lambda p, tp: p * tau + tp * (1 - tau), online_params, target_params
    )
    return eqx.combine(mixed, target_stats)


def update_step(
    state: XQCState, batch: Batch, noise: Noise, spec: XQCSpec
) -> tuple[XQCState, dict[str, jax.Array]]:
    """One XQC update: critic, target, delayed actor, temperature."""
    nets, opt = state.nets, state.opt
    critic, critic_opt, critic_info = _critic_update(
        nets, batch, noise.critic, spec, opt.critic
    )
    critic = project_unit_norm(critic)
    target = soft_update(critic, nets.target_critic, spec.hyper.tau)
    actor, actor_opt, actor_info = _actor_update(
        nets, critic, batch, noise.actor, spec, opt.actor
    )
    actor = project_unit_norm(actor)
    log_temp, temp_opt = _temperature_update(
        nets.log_temp, actor_info["entropy"], spec, opt.temperature
    )
    info = {
        **critic_info,
        **actor_info,
        "temperature": jnp.exp(nets.log_temp),
    }
    new_nets = XQCNets(actor, critic, target, log_temp)
    new_opt = XQCOpt(actor_opt, critic_opt, temp_opt)
    return XQCState(new_nets, new_opt, state.rng), info


def make_xqc_train(
    spec: XQCSpec, *, batch_size: int, updates: int
) -> Callable[..., tuple[XQCState, dict[str, jax.Array]]]:
    """Jitted `updates` fused steps on uniform batches of the buffer."""

    def step(
        carry: tuple[XQCState, BufferData, jax.Array, jax.Array], _: None
    ) -> tuple[Any, dict[str, jax.Array]]:
        state, data, size, scale = carry
        rng, batch_key, critic_key, actor_key = jax.random.split(state.rng, 4)
        batch = sample_batch(data, size, batch_key, batch_size)
        batch["reward"] = batch["reward"] / scale
        action_shape = batch["action"].shape
        noise = Noise(
            jax.random.normal(critic_key, action_shape),
            jax.random.normal(actor_key, action_shape),
        )
        state, info = update_step(state._replace(rng=rng), batch, noise, spec)
        return (state, data, size, scale), info

    @jax.jit
    def train(
        state: XQCState,
        data: BufferData,
        size: jax.Array,
        reward_scale: jax.Array,
    ) -> tuple[XQCState, dict[str, jax.Array]]:
        (state, *_), infos = jax.lax.scan(
            step, (state, data, size, reward_scale), None, length=updates
        )
        return state, jax.tree.map(jnp.mean, infos)

    return train
