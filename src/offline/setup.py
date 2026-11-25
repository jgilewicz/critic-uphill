from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from common.checkpoint import BUFFER_COLUMNS, LoadedCheckpoint
from common.replay_buffer import BufferData
from config import OfflineConfig
from offline.rollouts import HeadRunner, RolloutBatch

# own stream: offline draws never overlap training or metric keys
OFFLINE_STREAM = 15_485_863


class OfflineSetup:
    """Keys, seed sets, buffer and runner shared by every offline measurement.

    Everything derives from fold_in(key(seed), OFFLINE_STREAM) and env_steps,
    so every measurement kind on one checkpoint sees the same buffer states,
    perturbation key and seed set B. Truth uses seed set B, prefix signals
    use the disjoint seed set A.
    """

    def __init__(self, cfg: OfflineConfig, ckpt: LoadedCheckpoint) -> None:
        self.cfg = cfg
        self.ckpt = ckpt
        self.agent = ckpt.agent
        run = ckpt.meta["run_config"]
        base = jax.random.fold_in(jax.random.key(run["seed"]), OFFLINE_STREAM)
        key = jax.random.fold_in(base, ckpt.meta["env_steps"])
        np_key, self.perturb_key, self.fqe_key = jax.random.split(key, 3)
        self.rng = np.random.default_rng(
            int(jax.random.randint(np_key, (), 0, 2**31 - 1))
        )
        n_seeds = cfg.truth_episodes + cfg.prefix_seeds
        seeds = int(self.rng.integers(0, 2**31 - 1 - n_seeds)) + np.arange(
            n_seeds
        )
        self.seeds_b = seeds[: cfg.truth_episodes]
        self.seeds_a = seeds[cfg.truth_episodes :]
        self.runner = HeadRunner(run["env"]["id"], cfg.max_envs)

    def buffer_data(self) -> BufferData:
        arrays = self.ckpt.arrays
        s, a, r, ns, d = (
            jnp.asarray(arrays[f"buffer_{name}"]) for name in BUFFER_COLUMNS
        )
        return s, a, r, ns, d

    def buffer_states(self, data: BufferData) -> jax.Array:
        """n_states buffer-tail states, the first draw of self.rng."""
        n_rows = int(data[0].shape[0])
        idx = self.rng.choice(n_rows, min(self.cfg.n_states, n_rows), False)
        return data[0][idx]

    def actor_episodes(self) -> RolloutBatch:
        """Checkpoint actor on seed set B, visited states kept."""
        agent = self.agent
        actor = self.runner.run(
            agent.embedding,
            agent.critic1,
            jax.tree.map(lambda x: x[None], agent.actor_head),
            self.seeds_b,
            keep_states=True,
        )
        assert actor.states is not None
        return actor

    def header(self) -> dict[str, Any]:
        meta = self.ckpt.meta
        return {
            "checkpoint": self.cfg.checkpoint,
            "env_id": meta["run_config"]["env"]["id"],
            "seed": meta["run_config"]["seed"],
            "fraction": meta["fraction"],
            "env_steps": meta["env_steps"],
            "seeds_b": self.seeds_b.tolist(),
        }
