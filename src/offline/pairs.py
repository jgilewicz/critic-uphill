from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from offline.setup import OfflineSetup

# folded into the perturbation key, so pairs never reuse the signals draws
PAIR_STREAM = 104_729


def _pair_heads(
    head: Any, key: jax.Array, sigmas: tuple[float, ...], n_pairs: int
) -> Any:
    """Stack [actor, +eps x n, -eps x n] per sigma, eps ~ sigma * std(leaf)."""
    params, static = eqx.partition(head, eqx.is_array)
    leaves, treedef = jax.tree.flatten(params)
    stacked: list[list[jax.Array]] = [[leaf[None]] for leaf in leaves]
    for sigma, level_key in zip(
        sigmas, jax.random.split(key, len(sigmas)), strict=True
    ):
        keys = jax.random.split(level_key, len(leaves))
        for i, (leaf, k) in enumerate(zip(leaves, keys, strict=True)):
            eps = (
                sigma
                * jnp.std(leaf)
                * jax.random.normal(k, (n_pairs, *leaf.shape))
            )
            stacked[i] += [leaf + eps, leaf - eps]
    joined = [jnp.concatenate(parts) for parts in stacked]
    return eqx.combine(jax.tree.unflatten(treedef, joined), static)


class PairMeasurement:
    """Antithetic +-eps perturbation pairs on one checkpoint.

    Same buffer states and seed set B as SignalMeasurement. Per sigma in
    pair_sigmas, n_pairs noise draws eps, each scored as +eps and -eps by
    mean Q1 over the buffer states and by truth (paired real episodes). The
    odd part (f(+eps) - f(-eps)) / 2 is the linear response, the even part
    the curvature.
    """

    def __init__(self, setup: OfflineSetup) -> None:
        self._setup = setup

    def run(self) -> dict[str, Any]:
        setup, cfg, agent = self._setup, self._setup.cfg, self._setup.agent
        states = setup.buffer_states(setup.buffer_data())
        heads = _pair_heads(
            agent.actor_head,
            jax.random.fold_in(setup.perturb_key, PAIR_STREAM),
            cfg.pair_sigmas,
            cfg.n_pairs,
        )
        q1 = np.asarray(agent.q1_population_mean(heads, states, chunk_size=256))
        truth = setup.runner.run(
            agent.embedding, agent.critic1, heads, setup.seeds_b
        )
        gains = (truth.returns[1:] - truth.returns[:1]).mean(1)
        shape = (len(cfg.pair_sigmas), 2, cfg.n_pairs)
        return setup.header() | {
            "sigmas": list(cfg.pair_sigmas),
            "pair_sigma": np.repeat(cfg.pair_sigmas, cfg.n_pairs).tolist(),
            "actor_returns": truth.returns[0].tolist(),
            # [sigma][+eps, -eps][pair]
            "critic": (q1[1:] - q1[0]).reshape(shape).tolist(),
            "truth": gains.reshape(shape).tolist(),
            "truth_steps": truth.steps,
        }
