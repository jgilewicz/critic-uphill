from __future__ import annotations

from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from modules.deep_modules import Critic

Params = TypeVar("Params")

_DEFAULT_GRAD_CLIP_NORM = 10.0


def make_adam(
    lr: float, *, grad_clip_norm: float = _DEFAULT_GRAD_CLIP_NORM
) -> optax.GradientTransformation:
    """Create Adam optimizer chained with global grad-norm clipping."""
    return optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(lr),
    )


def critic_point(
    critic: Critic, states: jax.Array, actions: jax.Array
) -> jax.Array:
    """Evaluate Q(s, a) over a batch of states and actions."""
    return jax.vmap(critic)(states, actions)


def critic_loss(
    critic: Critic,
    states: jax.Array,
    actions: jax.Array,
    target: jax.Array,
) -> jax.Array:
    """Compute MSE Bellman loss for one critic."""
    return jnp.mean((critic_point(critic, states, actions) - target) ** 2)


@eqx.filter_jit
def genetic_soft_update(
    rl_params: Params, elite_params: Params, tau: float
) -> Params:
    """Blend elite parameters into rl_params: tau * elite + (1 - tau) * rl."""
    rl_arrays, rl_static = eqx.partition(rl_params, eqx.is_array)
    elite_arrays, _ = eqx.partition(elite_params, eqx.is_array)
    merged = jax.tree.map(
        lambda rl, el: tau * el + (1.0 - tau) * rl, rl_arrays, elite_arrays
    )
    return eqx.combine(merged, rl_static)
