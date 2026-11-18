from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from agents.base import RLAgent
from common.replay_buffer import ReplayBuffer
from config import DegConfig
from metrics.stats import spearman
from modules.deep_modules import Critic, SharedStateEmbedding

_TINY = 1e-30


class DegenerationTerms(NamedTuple):
    """Raw outputs of one jitted degeneration measurement."""

    heads: Any
    delta_s: jax.Array
    t1: jax.Array
    t2: jax.Array
    grad_norm: jax.Array
    action_grad_norm: jax.Array


@dataclass(frozen=True)
class DegenerationSample:
    """Perturbed heads of one measurement with their critic gain dS."""

    heads: Any
    delta_s: np.ndarray


def _perturb(head: Any, sigma: float, n: int, key: jax.Array) -> Any:
    """Stack n heads with per-leaf noise dW ~ N(0, (sigma * std(W))^2)."""
    params, static = eqx.partition(head, eqx.is_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(key, len(leaves))
    noisy = [
        leaf + sigma * jnp.std(leaf) * jax.random.normal(k, (n, *leaf.shape))
        for leaf, k in zip(leaves, keys, strict=True)
    ]
    return eqx.combine(jax.tree.unflatten(treedef, noisy), static)


def _stack_levels(
    head: Any, sigmas: tuple[float, ...], n: int, key: jax.Array
) -> Any:
    keys = jax.random.split(key, len(sigmas))
    levels = [
        _perturb(head, s, n, k) for s, k in zip(sigmas, keys, strict=True)
    ]
    arrays = [eqx.partition(h, eqx.is_array)[0] for h in levels]
    static = eqx.partition(levels[0], eqx.is_array)[1]
    joined = jax.tree.map(lambda *xs: jnp.concatenate(xs), *arrays)
    return eqx.combine(joined, static)


@eqx.filter_jit
def _measure(
    embedding: SharedStateEmbedding,
    critic: Critic,
    head: Any,
    states: jax.Array,
    key: jax.Array,
    *,
    sigmas: tuple[float, ...],
    n: int,
) -> DegenerationTerms:
    """t1, t2 and dS of perturbed heads, plus grad norms at the actor."""

    def q(s: jax.Array, a: jax.Array) -> jax.Array:
        return critic(s, a)[0]

    z = jax.vmap(embedding)(states)
    a0 = jax.vmap(head)(z)
    grad_a = jax.vmap(jax.grad(q, argnums=1))(states, a0)
    hess_a = jax.vmap(jax.hessian(q, argnums=1))(states, a0)

    def score(h: Any) -> jax.Array:
        return jnp.mean(jax.vmap(q)(states, jax.vmap(h)(z)))

    grads = jax.tree.leaves(eqx.filter_grad(score)(head))
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grads))
    base = score(head)
    heads = _stack_levels(head, sigmas, n, key)
    dyn, static = eqx.partition(heads, eqx.is_array)

    def per_head(dyn_head: Any) -> jax.Array:
        h = eqx.combine(dyn_head, static)
        acts = jax.vmap(h)(z)
        d = acts - a0
        gain = jnp.mean(jax.vmap(q)(states, acts)) - base
        t1 = jnp.mean(jnp.einsum("sa,sa->s", d, grad_a))
        t2 = 0.5 * jnp.mean(jnp.einsum("sa,sab,sb->s", d, hess_a, d))
        return jnp.stack([gain, t1, t2])

    out = jax.lax.map(per_head, dyn)
    return DegenerationTerms(
        heads=heads,
        delta_s=out[:, 0],
        t1=out[:, 1],
        t2=out[:, 2],
        grad_norm=grad_norm,
        action_grad_norm=jnp.mean(jnp.linalg.norm(grad_a, axis=-1)),
    )


def _perturbation_stats(
    delta_s: np.ndarray, t1: np.ndarray, t2: np.ndarray, suffix: str
) -> dict[str, float]:
    ratio = np.median(np.abs(t1)) / max(float(np.median(np.abs(t2))), _TINY)
    return {
        f"deg/ratio{suffix}": float(ratio),
        f"deg/rho_H{suffix}": spearman(delta_s, t2),
        f"deg/frac_critic_better{suffix}": float(np.mean(delta_s > 0.0)),
    }


class DegenerationProbe:
    """Block 3: how well Q1 explains the value of actor-head perturbations.

    measure() may be called at any step. The latest perturbations stay in
    `last` for the truth probe.
    """

    def __init__(self, cfg: DegConfig) -> None:
        self._cfg = cfg
        self.last: DegenerationSample | None = None

    def measure(
        self, agent: RLAgent, buffer: ReplayBuffer, key: jax.Array
    ) -> dict[str, float]:
        """Measure deg/* on buffer states, using no environment steps."""
        cfg = self._cfg
        states_key, perturb_key = jax.random.split(key)
        states = buffer.sample_states(states_key, cfg.n_states)
        terms = _measure(
            agent.embedding,
            agent.critic1,
            agent.actor_head,
            states,
            perturb_key,
            sigmas=cfg.sigmas,
            n=cfg.n_perturbations,
        )
        delta_s = np.asarray(terms.delta_s)
        self.last = DegenerationSample(terms.heads, delta_s)
        t1, t2 = np.asarray(terms.t1), np.asarray(terms.t2)
        metrics = {
            "deg/grad_norm": float(terms.grad_norm),
            "deg/action_grad_norm": float(terms.action_grad_norm),
        }
        metrics |= _perturbation_stats(delta_s, t1, t2, "")
        levels = np.repeat(np.arange(len(cfg.sigmas)), cfg.n_perturbations)
        for i in range(len(cfg.sigmas)):
            m = levels == i
            metrics |= _perturbation_stats(delta_s[m], t1[m], t2[m], f"_s{i}")
        return metrics
