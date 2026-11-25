from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from xqc.layers import BatchNorm, Dense

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0
_HIDDEN_INIT_SCALE = math.sqrt(2.0)


class Block(eqx.Module):
    """Dense (no bias) -> BatchNorm -> ReLU."""

    dense: Dense
    norm: BatchNorm

    def __init__(self, in_dim: int, out_dim: int, *, key: jax.Array) -> None:
        self.dense = Dense(
            in_dim,
            out_dim,
            key=key,
            use_bias=False,
            init_scale=_HIDDEN_INIT_SCALE,
        )
        self.norm = BatchNorm(out_dim)

    def forward(self, x: jax.Array, training: bool) -> tuple[jax.Array, Block]:
        y, norm = self.norm.forward(self.dense(x), training)
        return jax.nn.relu(y), eqx.tree_at(lambda m: m.norm, self, norm)


class Trunk(eqx.Module):
    """BatchNorm embedder followed by Blocks, shared by actor and critic."""

    embed_norm: BatchNorm
    blocks: tuple[Block, ...]

    def __init__(
        self, in_dim: int, hidden_dims: tuple[int, ...], *, key: jax.Array
    ) -> None:
        dims = (in_dim, *hidden_dims)
        keys = jax.random.split(key, len(hidden_dims))
        self.embed_norm = BatchNorm(in_dim)
        self.blocks = tuple(
            Block(d_in, d_out, key=k)
            for d_in, d_out, k in zip(dims[:-1], dims[1:], keys, strict=True)
        )

    def forward(self, x: jax.Array, training: bool) -> tuple[jax.Array, Trunk]:
        x, embed_norm = self.embed_norm.forward(x, training)
        blocks = []
        for block in self.blocks:
            x, block = block.forward(x, training)
            blocks.append(block)
        return x, eqx.tree_at(
            lambda m: (m.embed_norm, m.blocks),
            self,
            (embed_norm, tuple(blocks)),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Eval-mode features (running statistics)."""
        return self.forward(x, False)[0]


class Actor(eqx.Module):
    """Tanh-Gaussian policy: trunk, mean layer and log-std layer."""

    trunk: Trunk
    mean: Dense
    log_std: Dense

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        *,
        key: jax.Array,
    ) -> None:
        k_trunk, k_mean, k_std = jax.random.split(key, 3)
        self.trunk = Trunk(obs_dim, hidden_dims, key=k_trunk)
        width = hidden_dims[-1]
        self.mean = Dense(
            width,
            action_dim,
            key=k_mean,
            use_bias=True,
            init_scale=_HIDDEN_INIT_SCALE,
        )
        self.log_std = Dense(
            width, action_dim, key=k_std, use_bias=True, init_scale=1.0
        )

    def forward(
        self, obs: jax.Array, training: bool
    ) -> tuple[jax.Array, jax.Array, Actor]:
        """Return mean, clipped log-std and the actor with updated stats."""
        z, trunk = self.trunk.forward(obs, training)
        log_std = jnp.clip(self.log_std(z), LOG_STD_MIN, LOG_STD_MAX)
        return (
            self.mean(z),
            log_std,
            eqx.tree_at(lambda m: m.trunk, self, trunk),
        )


def sample_tanh_gaussian(
    mean: jax.Array, log_std: jax.Array, eps: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Sample tanh(mean + std * eps) with its log-probability.

    The log-probability reuses the pre-tanh sample, so it is the stable
    form -2 * (log 2 - x - softplus(-2x)) instead of an atanh round trip.
    """
    std = jnp.exp(log_std)
    x = mean + std * eps
    base = -0.5 * jnp.sum(
        jnp.square((x - mean) / std) + jnp.log(2 * jnp.pi) + 2 * jnp.log(std),
        axis=-1,
    )
    # intentional deviation from xqc-ref tfp atanh round trip: exact pre-tanh
    log_det = 2.0 * (math.log(2.0) - x - jax.nn.softplus(-2.0 * x))
    return jnp.tanh(x), base - jnp.sum(log_det, axis=-1)


class Critic(eqx.Module):
    """Categorical Q network over concat(obs, action)."""

    trunk: Trunk
    value: Dense

    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple[int, ...],
        num_bins: int,
        *,
        key: jax.Array,
    ) -> None:
        k_trunk, k_value = jax.random.split(key)
        self.trunk = Trunk(in_dim, hidden_dims, key=k_trunk)
        self.value = Dense(
            hidden_dims[-1],
            num_bins,
            key=k_value,
            use_bias=True,
            init_scale=_HIDDEN_INIT_SCALE,
        )

    def forward(self, x: jax.Array, training: bool) -> tuple[jax.Array, Critic]:
        z, trunk = self.trunk.forward(x, training)
        return self.value(z), eqx.tree_at(lambda m: m.trunk, self, trunk)


class CriticEnsemble(eqx.Module):
    """Independent categorical critics stacked on a leading member axis."""

    members: Critic
    min_v: float = eqx.field(static=True)
    max_v: float = eqx.field(static=True)

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        *,
        num_bins: int,
        n_critics: int,
        min_v: float,
        max_v: float,
        key: jax.Array,
    ) -> None:
        keys = jax.random.split(key, n_critics)
        self.members = eqx.filter_vmap(
            lambda k: Critic(obs_dim + action_dim, hidden_dims, num_bins, key=k)
        )(keys)
        self.min_v = min_v
        self.max_v = max_v

    @property
    def bin_values(self) -> jax.Array:
        num_bins = self.members.value.kernel.shape[-1]
        return jnp.linspace(self.min_v, self.max_v, num_bins, dtype=jnp.float32)

    def forward(
        self, obs: jax.Array, action: jax.Array, training: bool
    ) -> tuple[jax.Array, CriticEnsemble]:
        """Logits (members, batch, bins) and the ensemble with new stats."""
        x = jnp.concatenate([obs, action], axis=-1)
        logits, members = eqx.filter_vmap(lambda m: m.forward(x, training))(
            self.members
        )
        return logits, eqx.tree_at(lambda m: m.members, self, members)

    def expected_values(self, logits: jax.Array) -> jax.Array:
        """Mean of the categorical distribution over the bin support."""
        probs = jnp.exp(jax.nn.log_softmax(logits, axis=-1))
        return jnp.sum(probs * self.bin_values, axis=-1)

    def q_eval(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Eval-mode expected Q of every member, shape (members, ...)."""
        logits, _ = self.forward(obs, action, False)
        return self.expected_values(logits)
