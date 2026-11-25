from __future__ import annotations

from importlib.metadata import version
from typing import Any

import numpy as np

from common.checkpoint import Checkpointer, buffer_tail
from common.replay_buffer import ReplayBuffer
from config import RunConfig
from xqc.agent import XQCAgent
from xqc.normalizer import RewardNormalizer

_VERSIONED = ("jax", "jaxlib", "optax", "equinox", "numpy")


class XQCCheckpointer(Checkpointer):
    """Checkpointer of the XQC training state.

    The buffer holds policy-scale actions in [-1, 1], the checkpoint stores
    them in env units so offline code sees the actions the env received.
    """

    def __init__(
        self,
        run_cfg: RunConfig,
        agent: XQCAgent,
        buffer: ReplayBuffer,
        normalizer: RewardNormalizer,
    ) -> None:
        super().__init__(run_cfg.checkpoint, run_cfg, agent, buffer)
        self._xqc = agent
        self._normalizer = normalizer

    def _arrays(self) -> dict[str, np.ndarray]:
        arrays = buffer_tail(self._buffer, self._cfg.buffer_tail)
        arrays["buffer_action"] = arrays["buffer_action"] * np.float32(
            self._xqc.action_limit
        )
        return arrays

    def _extra_meta(self) -> dict[str, Any]:
        return {
            "algo": "xqc",
            "gamma": self._xqc.gamma,
            "reward_scale": self._normalizer.scale,
            "reward_var": self._normalizer.var,
            "reward_count": self._normalizer.count,
            "action_limit": self._xqc.action_limit,
            "versions": {name: version(name) for name in _VERSIONED},
        }


def make_xqc_checkpointer(
    run_cfg: RunConfig,
    agent: XQCAgent,
    buffer: ReplayBuffer,
    normalizer: RewardNormalizer,
) -> XQCCheckpointer | None:
    """XQC checkpointer, or None when checkpoint.dir is unset."""
    if not run_cfg.checkpoint.enabled:
        return None
    return XQCCheckpointer(run_cfg, agent, buffer, normalizer)
