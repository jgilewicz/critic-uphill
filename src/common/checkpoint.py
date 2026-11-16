from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from common.replay_buffer import Buffer, BufferData
from common.td3 import TD3Networks
from modules.deep_modules import ActorHead, Critic, SharedStateEmbedding

NETS = "nets.eqx"
BUFFER = "buffer.npz"
META = "meta.json"
BUFFER_FIELDS = ("state", "action", "reward", "next_state", "done")


class Checkpoint(NamedTuple):
    nets: TD3Networks
    # buffer arrays cut to the filled size, in BufferData order
    data: BufferData
    # next write index: the newest transition sits at ptr - 1
    ptr: int
    meta: dict[str, Any]


def checkpoint_dir(root: Path, env: str, seed: int, step: int) -> Path:
    return root / f"{env}_s{seed}" / f"step_{step}"


def save_checkpoint(
    path: Path, nets: TD3Networks, buffer: Buffer, meta: dict[str, Any]
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path / NETS, nets)
    n = len(buffer)
    arrays = {f: np.asarray(getattr(buffer, f)[:n]) for f in BUFFER_FIELDS}
    np.savez(path / BUFFER, ptr=buffer.ptr, **arrays)
    obs_dim, action_dim = buffer.state.shape[1], buffer.action.shape[1]
    full = meta | {"obs_dim": obs_dim, "action_dim": action_dim, "size": n}
    (path / META).write_text(json.dumps(full, indent=1))


def _skeleton(meta: dict[str, Any]) -> TD3Networks:
    ek, ak, c1k, c2k = jax.random.split(jax.random.key(0), 4)
    obs, act = meta["obs_dim"], meta["action_dim"]
    hidden = tuple(meta["hidden_dims"])
    return TD3Networks(
        SharedStateEmbedding(obs, embedding_dim=meta["embedding_dim"], key=ek),
        ActorHead(
            meta["embedding_dim"],
            act,
            key=ak,
            action_limit=meta["action_limit"],
        ),
        Critic(obs, act, key=c1k, hidden_dims=hidden),
        Critic(obs, act, key=c2k, hidden_dims=hidden),
    )


def load_checkpoint(path: Path) -> Checkpoint:
    missing = [f for f in (NETS, BUFFER, META) if not (path / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"checkpoint {path} lacks {missing}. was the run trained with "
            "checkpoint.dir and checkpoint.at covering this step?"
        )
    meta = json.loads((path / META).read_text())
    nets = eqx.tree_deserialise_leaves(path / NETS, _skeleton(meta))
    with np.load(path / BUFFER) as f:
        state, action, reward, next_state, done = (
            jnp.asarray(f[k]) for k in BUFFER_FIELDS
        )
        ptr = int(f["ptr"])
    data = (state, action, reward, next_state, done)
    return Checkpoint(nets, data, ptr, meta)
