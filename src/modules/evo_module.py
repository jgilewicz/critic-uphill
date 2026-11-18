from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("pop_size",))
def _cem_ask(
    key: jax.Array,
    mu: jax.Array,
    cov: jax.Array,
    *,
    scale: float,
    pop_size: int,
) -> jax.Array:
    if pop_size % 2 == 0:
        half = jax.random.normal(key, (pop_size // 2, mu.shape[0]))
        epsilon = jnp.concatenate([half, -half], axis=0)
    else:
        epsilon = jax.random.normal(key, (pop_size, mu.shape[0]))
    return mu + scale * epsilon * jnp.sqrt(cov)


@partial(jax.jit, static_argnames=("parents", "num_params"))
def _cem_tell(
    scores: jax.Array,
    solutions: jax.Array,
    mu: jax.Array,
    weights: jax.Array,
    *,
    parents: int,
    damp: float,
    damp_limit: float,
    tau: float,
    num_params: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    idx_sorted = jnp.argsort(-scores)[:parents]
    parent_solutions = solutions[idx_sorted]

    new_damp = jnp.asarray(damp * tau + (1 - tau) * damp_limit)
    new_mu = jnp.dot(weights, parent_solutions)
    z = parent_solutions - mu
    new_cov = 1.0 / parents * jnp.dot(weights, z * z) + new_damp * jnp.ones(
        num_params
    )
    return new_mu, new_cov, new_damp, idx_sorted[0]


class CEM:
    """Cross-Entropy Method optimizer over flattened policy heads."""

    def __init__(
        self,
        mean: jax.Array,
        key: jax.Array,
        *,
        pop_size: int,
        sigma_init: float = 1e-3,
        damp: float = 1e-3,
        damp_limit: float = 1e-5,
        tau: float = 0.95,
        elitism: bool = False,
    ) -> None:
        num_params = int(mean.shape[0])
        self.num_params = num_params
        self.scale = 1.0
        self.sigma_init = sigma_init
        self.pop_size = pop_size
        self.damp = damp
        self.damp_limit = damp_limit
        self.parents = max(1, pop_size // 2)
        self.elitism = elitism
        self.tau = tau

        self.mu = mean
        self.cov = jnp.ones(num_params) * sigma_init
        self.elite = mean + jnp.sqrt(sigma_init) * jax.random.normal(
            key, (num_params,)
        )
        weights = jnp.log((self.parents + 1) / jnp.arange(1, self.parents + 1))
        self.weights = weights / weights.sum()

    def ask(self, key: jax.Array) -> jax.Array:
        """Sample a population of parameter vectors, keeping elite if set."""
        inds = _cem_ask(
            key, self.mu, self.cov, scale=self.scale, pop_size=self.pop_size
        )
        if self.elitism:
            inds = inds.at[-1].set(self.elite)
        return inds

    def tell(
        self, scores: jax.Array, solutions: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Update mean, diagonal covariance, and elite from fitness scores."""
        self.mu, self.cov, new_damp, elite_idx = _cem_tell(
            scores,
            solutions,
            self.mu,
            self.weights,
            parents=self.parents,
            damp=self.damp,
            damp_limit=self.damp_limit,
            tau=self.tau,
            num_params=self.num_params,
        )
        self.damp = float(new_damp)
        self.elite = solutions[elite_idx]
        return self.mu, self.cov
