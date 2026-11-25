from __future__ import annotations

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from common.utils import critic_point
from environments import make_env
from modules.protocols import QCritic, StateEmbedding


@eqx.filter_jit
def _act_and_q(
    embedding: StateEmbedding,
    critic: QCritic,
    heads: Any,
    states: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    z = jax.vmap(embedding)(states)
    actions = eqx.filter_vmap(lambda head, zi: head(zi))(heads, z)
    return actions, critic_point(critic, states, actions)[:, 0]


class RolloutBatch(NamedTuple):
    """Per (head, seed) rollout outcome, shaped (heads, seeds)."""

    returns: np.ndarray
    q_sum: np.ndarray
    lengths: np.ndarray
    states: np.ndarray | None

    @property
    def steps(self) -> int:
        return int(self.lengths.sum())


class HeadRunner:
    """Deterministic rollouts of stacked heads, one env per (head, seed).

    Every head runs on every seed. Q(s, head(s)) of the given critic is
    summed over the visited states. Nothing is stored or budgeted.
    """

    def __init__(self, env_id: str, max_envs: int) -> None:
        self._envs = [make_env(env_id) for _ in range(max_envs)]
        self._max_envs = max_envs

    def run(
        self,
        embedding: StateEmbedding,
        critic: QCritic,
        heads: Any,
        seeds: np.ndarray,
        *,
        max_steps: int | None = None,
        keep_states: bool = False,
    ) -> RolloutBatch:
        """Roll out heads x seeds until done or max_steps env steps."""
        n_heads = int(jax.tree.leaves(heads)[0].shape[0])
        head_idx = np.repeat(np.arange(n_heads), len(seeds))
        slot_seeds = np.tile(seeds, n_heads)
        returns = np.zeros(len(head_idx))
        q_sum = np.zeros(len(head_idx))
        lengths = np.zeros(len(head_idx), dtype=np.int64)
        kept: list[np.ndarray] | None = [] if keep_states else None
        for start in range(0, len(head_idx), self._max_envs):
            chunk = slice(start, start + self._max_envs)
            idx = head_idx[chunk]
            returns[chunk], q_sum[chunk], lengths[chunk] = self._run_chunk(
                embedding,
                critic,
                jax.tree.map(lambda x, idx=idx: x[idx], heads),
                slot_seeds[chunk],
                max_steps=max_steps,
                kept=kept,
            )
        shape = (n_heads, len(seeds))
        return RolloutBatch(
            returns=returns.reshape(shape),
            q_sum=q_sum.reshape(shape),
            lengths=lengths.reshape(shape),
            states=None if kept is None else np.concatenate(kept),
        )

    def _run_chunk(
        self,
        embedding: StateEmbedding,
        critic: QCritic,
        heads: Any,
        seeds: np.ndarray,
        *,
        max_steps: int | None,
        kept: list[np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        envs = self._envs[: len(seeds)]
        states = np.stack(
            [
                np.asarray(env.reset(seed=int(seed))[0], dtype=np.float32)
                for env, seed in zip(envs, seeds, strict=True)
            ]
        )
        alive = np.ones(len(seeds), dtype=bool)
        returns = np.zeros(len(seeds))
        q_sum = np.zeros(len(seeds))
        lengths = np.zeros(len(seeds), dtype=np.int64)
        step = 0
        while alive.any() and (max_steps is None or step < max_steps):
            actions, q = _act_and_q(
                embedding, critic, heads, jnp.asarray(states)
            )
            actions, q = np.asarray(actions), np.asarray(q)
            active = np.flatnonzero(alive)
            if kept is not None:
                kept.append(states[active].copy())
            q_sum[active] += q[active]
            lengths[active] += 1
            for i in active:
                obs, reward, terminated, truncated, _ = envs[i].step(actions[i])
                states[i] = obs
                returns[i] += float(reward)
                alive[i] = not (terminated or truncated)
            step += 1
        return returns, q_sum, lengths
