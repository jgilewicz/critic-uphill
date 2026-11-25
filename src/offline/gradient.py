from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from agents.base import OfflineAgent
from common.utils import critic_point
from modules.protocols import QCritic, StateEmbedding
from offline.rollouts import HeadRunner


@eqx.filter_jit
def _q1_head_grad(
    embedding: StateEmbedding,
    critic: QCritic,
    head: Any,
    states: jax.Array,
) -> jax.Array:
    """E[J^T grad_a Q1] over states: d mean Q1(s, head(s)) / d head."""
    z = jax.vmap(embedding)(states)

    def score(h: Any) -> jax.Array:
        return jnp.mean(critic_point(critic, states, jax.vmap(h)(z))[:, 0])

    return ravel_pytree(eqx.filter_grad(score)(head))[0]


def perturbation_norm(head: Any, sigma: float) -> float:
    """Expected norm of a deg perturbation: sigma * std(leaf) per entry."""
    leaves = jax.tree.leaves(eqx.filter(head, eqx.is_array))
    var = sum(float(jnp.std(leaf)) ** 2 * leaf.size for leaf in leaves)
    return sigma * float(np.sqrt(var))


class GradientProbe:
    """Theorem 2 scope: Q1 actor gradient on buffer vs fresh actor states.

    Steps the actor head along the fresh-state gradient by the expected
    perturbation norm at each step sigma and pairs real episodes with the
    actor on the truth seeds.
    """

    def __init__(
        self,
        agent: OfflineAgent,
        runner: HeadRunner,
        *,
        seeds: np.ndarray,
        step_sigmas: tuple[float, ...],
    ) -> None:
        self._agent = agent
        self._runner = runner
        self._seeds = seeds
        self._step_sigmas = step_sigmas

    def measure(
        self,
        buffer_states: jax.Array,
        fresh_states: jax.Array,
        actor_returns: np.ndarray,
    ) -> tuple[dict[str, Any], int]:
        """Gradient stats and paired step gains, plus env steps used."""
        agent = self._agent
        g_buffer, g_fresh = (
            _q1_head_grad(agent.embedding, agent.critic1, agent.actor_head, s)
            for s in (buffer_states, fresh_states)
        )
        norm_buffer = float(jnp.linalg.norm(g_buffer))
        norm_fresh = float(jnp.linalg.norm(g_fresh))
        cosine = float(g_buffer @ g_fresh) / (norm_buffer * norm_fresh)
        lengths = [
            perturbation_norm(agent.actor_head, s) for s in self._step_sigmas
        ]
        theta = agent.actor_head_flat()
        direction = g_fresh / norm_fresh
        flat = jnp.stack([theta + eta * direction for eta in lengths])
        heads = jax.vmap(agent.unravel_head)(flat)
        truth = self._runner.run(
            agent.embedding, agent.critic1, heads, self._seeds
        )
        gains = (truth.returns - actor_returns).mean(1)
        stats = {
            "n_fresh_states": int(fresh_states.shape[0]),
            "grad_norm_buffer": norm_buffer,
            "grad_norm_fresh": norm_fresh,
            "cosine": cosine,
            "step_sigmas": list(self._step_sigmas),
            "step_lengths": lengths,
            "step_gains": gains.tolist(),
        }
        return stats, truth.steps
