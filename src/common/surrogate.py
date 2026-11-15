from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from common.td3 import critic_point
from modules.deep_modules import ActorHead, Critic, SharedStateEmbedding


@eqx.filter_jit
def population_q(
    critic: Critic,
    embedding: SharedStateEmbedding,
    heads: ActorHead,
    states: jax.Array,
    key: jax.Array,
    *,
    stacked: bool,
) -> jax.Array:
    # (pop,) mean Q over one shared replay batch; one key for every
    # individual so dropout masks are common random numbers.
    # stacked: each individual carries its own (pop,)-stacked embedding
    def mean_q(z: jax.Array, head: ActorHead) -> jax.Array:
        actions = jax.vmap(head)(z)
        return critic_point(critic, states, actions, key)[:, 0].mean()

    if stacked:
        return eqx.filter_vmap(lambda e, h: mean_q(jax.vmap(e)(states), h))(
            embedding, heads
        )
    z = jax.vmap(embedding)(states)
    return eqx.filter_vmap(lambda h: mean_q(z, h))(heads)


@eqx.filter_jit
def policy_q(
    critic: Critic,
    embedding: SharedStateEmbedding,
    actor: ActorHead,
    states: jax.Array,
    key: jax.Array,
) -> jax.Array:
    actions = jax.vmap(actor)(jax.vmap(embedding)(states))
    return critic_point(critic, states, actions, key)[:, 0].mean()


@jax.jit
def critic_fitness(
    observed: jax.Array, real: jax.Array, mu: jax.Array, scale: float
) -> tuple[jax.Array, jax.Array]:
    # scale maps the discounted Q onto the undiscounted return's units. The
    # offset is the critic bias on this generation's real rollouts; with
    # none it is 0, and a shared offset cannot change the ranking anyway.
    # observed must be zero outside `real`
    q = scale * mu
    residual = jnp.where(real, observed - q, 0.0)
    offset = jnp.sum(residual) / jnp.maximum(jnp.sum(real), 1)
    return jnp.where(real, observed, q + offset), offset
