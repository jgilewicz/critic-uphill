from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def _scaled_linear(
    in_dim: int,
    out_dim: int,
    *,
    key: jax.Array,
    init_scale: float,
) -> eqx.nn.Linear:
    layer = eqx.nn.Linear(in_dim, out_dim, key=key)
    if init_scale == 1.0:
        return layer
    assert layer.bias is not None
    return eqx.tree_at(
        lambda m: (m.weight, m.bias),
        layer,
        (layer.weight * init_scale, layer.bias * init_scale),
    )


class Critic(eqx.Module):
    """Q(s, a) network with LayerNorm + LeakyReLU MLP."""

    net: eqx.nn.Sequential

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        key: jax.Array,
        hidden_dims: tuple[int, int] = (400, 300),
        init_scale: float = 0.1,
    ) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        dim1, dim2 = hidden_dims
        out = _scaled_linear(dim2, 1, key=k3, init_scale=init_scale)
        self.net = eqx.nn.Sequential(
            [
                eqx.nn.Linear(state_dim + action_dim, dim1, key=k1),
                eqx.nn.LayerNorm(dim1),
                eqx.nn.Lambda(jax.nn.leaky_relu),
                eqx.nn.Linear(dim1, dim2, key=k2),
                eqx.nn.LayerNorm(dim2),
                eqx.nn.Lambda(jax.nn.leaky_relu),
                out,
            ]
        )

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        return self.net(jnp.concatenate([state, action], axis=-1))


class ActorHead(eqx.Module):
    """Linear + tanh policy head over the shared state embedding."""

    out_layer: eqx.nn.Linear
    action_limit: float = eqx.field(static=True)

    def __init__(
        self,
        embedding_dim: int,
        action_dim: int,
        *,
        key: jax.Array,
        action_limit: float = 1.0,
        init_scale: float = 1.0,
    ) -> None:
        self.action_limit = action_limit
        self.out_layer = _scaled_linear(
            embedding_dim, action_dim, key=key, init_scale=init_scale
        )

    def __call__(self, z: jax.Array) -> jax.Array:
        return jnp.tanh(self.out_layer(z)) * self.action_limit


class SharedStateEmbedding(eqx.Module):
    """Shared state encoder Z(s) trained by RL agent, read by ES heads."""

    embedding: eqx.nn.Sequential

    def __init__(
        self,
        state_dim: int,
        embedding_dim: int,
        *,
        key: jax.Array,
        hidden_dim: int = 400,
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.embedding = eqx.nn.Sequential(
            [
                eqx.nn.Linear(state_dim, hidden_dim, key=k1),
                eqx.nn.LayerNorm(hidden_dim),
                eqx.nn.Lambda(jnp.tanh),
                eqx.nn.Linear(hidden_dim, embedding_dim, key=k2),
                eqx.nn.LayerNorm(embedding_dim),
            ]
        )

    def __call__(self, state: jax.Array) -> jax.Array:
        return jnp.tanh(self.embedding(state))
