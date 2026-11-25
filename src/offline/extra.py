from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from common.utils import critic_point
from metrics.degeneration import measure_terms
from offline.setup import OfflineSetup

# own rng for the gradient state samples, offset by env_steps
NOISE_STREAM = 7_919


@eqx.filter_jit
def _per_state_grads(
    embedding: Any, critic: Any, head: Any, states: jax.Array
) -> jax.Array:
    """Flat d Q1(s, head(z_s)) / d head for every state."""
    flat, unravel = ravel_pytree(head)

    def one(s: jax.Array) -> jax.Array:
        def q(theta: jax.Array) -> jax.Array:
            a = unravel(theta)(embedding(s))
            return critic_point(critic, s[None], a[None])[0, 0]

        return jax.grad(q)(flat)

    return jax.lax.map(one, states, batch_size=256)


def _grad_stats(per_state: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Mean gradient, its norm and coherence (norm / mean per-state norm)."""
    mean = per_state.mean(0)
    norm = float(np.linalg.norm(mean))
    expected = float(np.linalg.norm(per_state, axis=1).mean())
    return mean, norm, norm / expected


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class ExtraMeasurement:
    """Q1 terms and gradient noise floor on one checkpoint.

    Terms rebuild the perturbed heads and buffer states of SignalMeasurement
    (same key and first rng draw) and store dS, t1 and t2 of every head. The
    noise floor compares mean Q1 head gradients between two disjoint halves
    of buffer states and of fresh states visited by the actor on seed set B.
    """

    def __init__(self, setup: OfflineSetup) -> None:
        self._setup = setup
        self._agent = setup.agent

    def _terms(self, states: jax.Array) -> dict[str, Any]:
        agent, cfg = self._agent, self._setup.cfg
        t = measure_terms(
            agent.embedding,
            agent.critic1,
            agent.actor_head,
            states,
            self._setup.perturb_key,
            sigmas=cfg.sigmas,
            n=cfg.n_perturbations,
        )
        q1 = agent.q1_population_mean(t.heads, states, chunk_size=256)
        return {
            "delta_s": np.asarray(t.delta_s).tolist(),
            "t1": np.asarray(t.t1).tolist(),
            "t2": np.asarray(t.t2).tolist(),
            # equals the q1 signal of SignalMeasurement, a pairing check
            "q1_signal": np.asarray(q1).tolist(),
        }

    def _halves(
        self, rng: np.random.Generator, rows: np.ndarray
    ) -> list[np.ndarray]:
        n = min(self._setup.cfg.n_states, len(rows) // 2)
        idx = rng.choice(len(rows), 2 * n, replace=False)
        return [rows[idx[:n]], rows[idx[n:]]]

    def _stats(self, states: np.ndarray) -> tuple[np.ndarray, float, float]:
        agent = self._agent
        grads = _per_state_grads(
            agent.embedding,
            agent.critic1,
            agent.actor_head,
            jnp.asarray(states),
        )
        return _grad_stats(np.asarray(grads))

    def _noise(self) -> dict[str, Any]:
        setup = self._setup
        actor = setup.actor_episodes()
        assert actor.states is not None
        rng = np.random.default_rng(
            NOISE_STREAM + int(setup.ckpt.meta["env_steps"])
        )
        buffer = np.asarray(setup.ckpt.arrays["buffer_state"])
        samples = {
            "buffer": self._halves(rng, buffer),
            "fresh": self._halves(rng, actor.states),
        }
        means: dict[str, list[np.ndarray]] = {}
        out: dict[str, Any] = {"n_fresh_total": int(len(actor.states))}
        for name, halves in samples.items():
            stats = [self._stats(s) for s in halves]
            means[name] = [st[0] for st in stats]
            out[name] = {
                "n": int(len(halves[0])),
                "cosine": _cos(stats[0][0], stats[1][0]),
                "norm_ratio": stats[0][1] / stats[1][1],
                "norms": [st[1] for st in stats],
                "coherence": [st[2] for st in stats],
            }
        b, f = means["buffer"], means["fresh"]
        out["cross"] = {
            "cosine": [_cos(b[i], f[i]) for i in range(2)],
            "norm_ratio": [
                float(np.linalg.norm(f[i]) / np.linalg.norm(b[i]))
                for i in range(2)
            ],
        }
        return out

    def run(self) -> dict[str, Any]:
        setup = self._setup
        states = setup.buffer_states(setup.buffer_data())
        return setup.header() | {
            "terms": self._terms(states),
            "noise": self._noise(),
        }
