from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

BN_MOMENTUM = 0.99
BN_EPS = 1e-3


class Dense(eqx.Module):
    """Affine layer with an orthogonal kernel, feature axis last."""

    kernel: jax.Array
    bias: jax.Array | None

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        key: jax.Array,
        use_bias: bool,
        init_scale: float,
    ) -> None:
        init = jax.nn.initializers.orthogonal(init_scale)
        self.kernel = init(key, (in_dim, out_dim), jnp.float32)
        self.bias = jnp.zeros(out_dim, jnp.float32) if use_bias else None

    def __call__(self, x: jax.Array) -> jax.Array:
        y = x @ self.kernel
        return y if self.bias is None else y + self.bias


class BatchNorm(eqx.Module):
    """BatchNorm over all axes but the last, with explicit running stats."""

    scale: jax.Array
    bias: jax.Array
    running_mean: jax.Array
    running_var: jax.Array

    def __init__(self, dim: int) -> None:
        self.scale = jnp.ones(dim, jnp.float32)
        self.bias = jnp.zeros(dim, jnp.float32)
        self.running_mean = jnp.zeros(dim, jnp.float32)
        self.running_var = jnp.ones(dim, jnp.float32)

    def forward(
        self, x: jax.Array, training: bool
    ) -> tuple[jax.Array, BatchNorm]:
        """Normalize x, returning the module with updated running stats."""
        new = self
        if training:
            axes = tuple(range(x.ndim - 1))
            mean = jnp.mean(x, axes)
            mean2 = jnp.mean(jnp.square(x), axes)
            var = jnp.maximum(0.0, mean2 - jnp.square(mean))
            new = eqx.tree_at(
                lambda m: (m.running_mean, m.running_var),
                self,
                (
                    BN_MOMENTUM * self.running_mean + (1 - BN_MOMENTUM) * mean,
                    BN_MOMENTUM * self.running_var + (1 - BN_MOMENTUM) * var,
                ),
            )
        else:
            mean, var = self.running_mean, self.running_var
        mul = jax.lax.rsqrt(var + BN_EPS) * self.scale
        return (x - mean) * mul + self.bias, new


def trainable(tree: Any) -> Any:
    """Bool mask: True on array leaves except BatchNorm running stats."""

    def keep(path: tuple[Any, ...], leaf: Any) -> bool:
        last = path[-1]
        is_stat = isinstance(
            last, jax.tree_util.GetAttrKey
        ) and last.name.startswith("running_")
        return eqx.is_array(leaf) and not is_stat

    return jax.tree_util.tree_map_with_path(keep, tree)


def project_unit_norm(tree: Any) -> Any:
    """Divide every Dense kernel by its column norm (weight-norm projection)."""

    def project(layer: Dense) -> Dense:
        norm = jnp.linalg.norm(layer.kernel, axis=-2, keepdims=True)
        return eqx.tree_at(lambda m: m.kernel, layer, layer.kernel / norm)

    return jax.tree.map(
        lambda x: project(x) if isinstance(x, Dense) else x,
        tree,
        is_leaf=lambda x: isinstance(x, Dense),
    )
