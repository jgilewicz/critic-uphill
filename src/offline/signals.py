from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from agents.td3 import TD3
from modules.deep_modules import Critic, SharedStateEmbedding
from offline.rollouts import HeadRunner

Signals = dict[str, np.ndarray]


@eqx.filter_jit
def _action_sq_dist(
    embedding: SharedStateEmbedding,
    actor: Any,
    heads: Any,
    states: jax.Array,
) -> jax.Array:
    z = jax.vmap(embedding)(states)
    base = jax.vmap(actor)(z)

    def per_head(head: Any) -> jax.Array:
        diff = jax.vmap(head)(z) - base
        return jnp.mean(jnp.sum(diff**2, axis=-1))

    return eqx.filter_vmap(per_head)(heads)


@eqx.filter_jit
def _weight_dist(actor: Any, heads: Any) -> jax.Array:
    base = ravel_pytree(actor)[0]
    flat = eqx.filter_vmap(lambda head: ravel_pytree(head)[0])(heads)
    return jnp.linalg.norm(flat - base, axis=-1)


class SignalScorer:
    """Scores stacked actor heads by every offline signal, higher is better.

    Critic signals average Q(s, head(s)) over fixed buffer states. own_q1 and
    prefix_return come from h-step rollouts on seed set A. dist_action and
    dist_weight are minus the distance from the checkpoint actor.
    """

    def __init__(
        self,
        agent: TD3,
        critics: dict[str, Critic],
        states: jax.Array,
        runner: HeadRunner,
        *,
        seeds: np.ndarray,
        h: int,
    ) -> None:
        self._agent = agent
        self._critics = critics
        self._states = states
        self._runner = runner
        self._seeds = seeds
        self._h = h

    def score(self, heads: Any) -> tuple[Signals, int]:
        """All signals for heads and the env steps spent on prefixes."""
        agent = self._agent
        signals = {
            # small chunks: one critic pass per head and state at once
            name: np.asarray(
                agent.q1_population_mean(
                    heads, self._states, chunk_size=256, critic=critic
                )
            )
            for name, critic in self._critics.items()
        }
        prefix = self._runner.run(
            agent.embedding,
            agent.critic1,
            heads,
            self._seeds,
            max_steps=self._h,
        )
        signals["own_q1"] = prefix.q_sum.sum(1) / prefix.lengths.sum(1)
        signals["prefix_return"] = prefix.returns.mean(1)
        signals["dist_action"] = -np.asarray(
            _action_sq_dist(
                agent.embedding, agent.actor_head, heads, self._states
            )
        )
        signals["dist_weight"] = -np.asarray(
            _weight_dist(agent.actor_head, heads)
        )
        return signals, prefix.steps
