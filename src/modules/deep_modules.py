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


_LOG_STD_MIN = -20.0
_LOG_STD_MAX = 2.0


class GaussianActorHead(eqx.Module):
    """Tanh-squashed Gaussian policy head over the shared state embedding."""

    mean_layer: eqx.nn.Linear
    log_std_layer: eqx.nn.Linear
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
        k_mean, k_std = jax.random.split(key)
        self.action_limit = action_limit
        self.mean_layer = _scaled_linear(
            embedding_dim, action_dim, key=k_mean, init_scale=init_scale
        )
        self.log_std_layer = _scaled_linear(
            embedding_dim, action_dim, key=k_std, init_scale=init_scale
        )

    def __call__(self, z: jax.Array) -> jax.Array:
        return jnp.tanh(self.mean_layer(z)) * self.action_limit

    def sample_and_log_prob(
        self, z: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Sample a squashed action and its Jacobian-corrected log_prob."""
        mean = self.mean_layer(z)
        log_std = jnp.clip(self.log_std_layer(z), _LOG_STD_MIN, _LOG_STD_MAX)
        std = jnp.exp(log_std)
        eps = jax.random.normal(key, mean.shape)
        u = mean + std * eps
        action = jnp.tanh(u) * self.action_limit
        log_prob_u = -0.5 * (eps**2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
        log_det = 2.0 * (
            jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u)
        ) + jnp.log(self.action_limit)
        log_prob = jnp.sum(log_prob_u - log_det, axis=-1)
        return action, log_prob


class SharedStateEmbedding(eqx.Module):
    """Shared state encoder Z(s) trained by TD3 and read by ES heads."""

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
