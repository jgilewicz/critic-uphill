from __future__ import annotations

import dataclasses
import json
from collections import deque
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import equinox as eqx
import jax
import numpy as np

from agents.base import OfflineAgent
from agents.td3 import TD3
from common.replay_buffer import ReplayBuffer
from config import CheckpointConfig, RunConfig, TD3RLConfig, XQCConfig
from environments import env_dims

STATE_FILE = "state.eqx"
DATA_FILE = "data.npz"
META_FILE = "meta.json"
BUFFER_COLUMNS = ("state", "action", "reward", "next_state", "done")


class TrainedAgent(Protocol):
    """An agent whose whole training state can be serialized."""

    state: Any


class LoadedCheckpoint(NamedTuple):
    """A saved checkpoint: restored agent, arrays and metadata."""

    agent: OfflineAgent
    arrays: dict[str, np.ndarray]
    meta: dict[str, Any]


class GenerationRecord(NamedTuple):
    """One evaluated EvoRainbow generation, as seen at ask time."""

    generation: int
    env_steps: int
    flat_pop: np.ndarray
    actor_flat: np.ndarray
    injected: int
    fitness: np.ndarray
    real_mask: np.ndarray


def checkpoint_name(fraction: float) -> str:
    return f"p{round(fraction * 100):03d}"


def buffer_tail(buffer: ReplayBuffer, count: int) -> dict[str, np.ndarray]:
    """Last `count` transitions in insertion order (oldest first)."""
    n = min(count, buffer.size)
    indices = (buffer.ptr - n + np.arange(n)) % buffer.capacity
    return {
        f"buffer_{name}": np.asarray(column)[indices]
        for name, column in zip(BUFFER_COLUMNS, buffer.data, strict=True)
    }


def _population_arrays(
    records: list[GenerationRecord],
) -> dict[str, np.ndarray]:
    if not records:
        return {}
    return {
        "pop_generation": np.array([r.generation for r in records]),
        "pop_env_steps": np.array([r.env_steps for r in records]),
        "pop_flat": np.stack([r.flat_pop for r in records]),
        "pop_actor_flat": np.stack([r.actor_flat for r in records]),
        "pop_injected": np.array([r.injected for r in records]),
        "pop_fitness": np.stack([r.fitness for r in records]),
        "pop_real_mask": np.stack([r.real_mask for r in records]),
    }


class Checkpointer:
    """Saves agent state, buffer tail and recent generations for offline use.

    Saves at the first iteration boundary with env_steps >= fraction *
    total_steps. Reads only, so the training run is unchanged.
    """

    def __init__(
        self,
        cfg: CheckpointConfig,
        run_cfg: RunConfig,
        agent: TrainedAgent,
        buffer: ReplayBuffer,
    ) -> None:
        if cfg.dir is None:
            raise ValueError("Checkpointer needs checkpoint.dir")
        self._cfg = cfg
        self._run_cfg = run_cfg
        self._agent = agent
        self._buffer = buffer
        self._root = Path(cfg.dir)
        self._pending = list(cfg.fractions)
        self._records: deque[GenerationRecord] = deque(maxlen=cfg.generations)

    def record_generation(self, record: GenerationRecord) -> None:
        self._records.append(record)

    def maybe_save(self, env_steps: int) -> None:
        total = self._run_cfg.total_steps
        while self._pending and env_steps >= self._pending[0] * total:
            self._save(self._pending.pop(0), env_steps)

    def finish(self, env_steps: int) -> None:
        if self._pending:
            raise RuntimeError(
                f"run ended at env_steps={env_steps} before checkpoint "
                f"fractions {self._pending} of "
                f"total_steps={self._run_cfg.total_steps}"
            )

    def _arrays(self) -> dict[str, np.ndarray]:
        return {
            **buffer_tail(self._buffer, self._cfg.buffer_tail),
            **_population_arrays(list(self._records)),
        }

    def _extra_meta(self) -> dict[str, Any]:
        return {"algo": "td3", "generations": len(self._records)}

    def _save(self, fraction: float, env_steps: int) -> None:
        out = self._root / checkpoint_name(fraction)
        out.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(
            out / STATE_FILE, jax.device_get(self._agent.state)
        )
        with (out / DATA_FILE).open("wb") as f:
            np.savez(f, allow_pickle=False, **self._arrays())
        meta: dict[str, Any] = {
            "fraction": fraction,
            "env_steps": env_steps,
            "buffer_size": self._buffer.size,
            "run_config": dataclasses.asdict(self._run_cfg),
            **self._extra_meta(),
        }
        (out / META_FILE).write_text(json.dumps(meta, indent=2))
        print(f"checkpoint | {out} | env_steps={env_steps}")


def make_checkpointer(
    run_cfg: RunConfig, agent: TD3, buffer: ReplayBuffer
) -> Checkpointer | None:
    """Checkpointer for a TD3 agent, or None when checkpoint.dir is unset."""
    if not run_cfg.checkpoint.enabled:
        return None
    return Checkpointer(run_cfg.checkpoint, run_cfg, agent, buffer)


def _load_td3(path: Path, meta: dict[str, Any]) -> TD3:
    run = meta["run_config"]
    algo = run["algorithm"]
    rl = dict(algo["rl"])
    rl["critic_hidden_dims"] = tuple(rl["critic_hidden_dims"])
    dims = env_dims(run["env"]["id"])
    agent = TD3(
        dims.obs_dim,
        dims.action_dim,
        dims.action_limit,
        algo["embedding_dim"],
        rl_cfg=TD3RLConfig(**rl),
        key=jax.random.key(0),
        shadow=algo["shadow"]["enabled"],
    )
    agent.state = eqx.tree_deserialise_leaves(path / STATE_FILE, agent.state)
    return agent


def _load_xqc(path: Path, meta: dict[str, Any]) -> OfflineAgent:
    from xqc.agent import XQCAgent

    run = meta["run_config"]
    algo = dict(run["algorithm"])
    for name in ("actor_hidden_dims", "critic_hidden_dims"):
        algo[name] = tuple(algo[name])
    dims = env_dims(run["env"]["id"])
    agent = XQCAgent(
        dims.obs_dim,
        dims.action_dim,
        dims.action_limit,
        cfg=XQCConfig(**algo),
        total_steps=run["total_steps"],
        seed=run["seed"],
    )
    agent.state = eqx.tree_deserialise_leaves(path / STATE_FILE, agent.state)
    agent.reward_scale = meta["reward_scale"]
    return agent


def load_checkpoint(path: Path) -> LoadedCheckpoint:
    """Rebuild the agent of a checkpoint dir and restore its state."""
    meta_path = path / META_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"no {META_FILE} in {path}: pass a checkpoint dir such as "
            "<checkpoint.dir>/p050"
        )
    meta = json.loads(meta_path.read_text())
    algo = meta.get("algo", "td3")
    loaders = {"td3": _load_td3, "xqc": _load_xqc}
    if algo not in loaders:
        raise ValueError(
            f"unknown checkpoint algo {algo!r} in {meta_path}: "
            f"expected one of {sorted(loaders)}"
        )
    agent = loaders[algo](path, meta)
    with np.load(path / DATA_FILE, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    return LoadedCheckpoint(agent=agent, arrays=arrays, meta=meta)
