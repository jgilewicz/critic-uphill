from __future__ import annotations

import jax

from agents.base import RLAgent as RLAgent
from agents.sac import SAC as SAC
from agents.td3 import TD3 as TD3
from config import RLConfig

_AGENTS: dict[str, type[RLAgent]] = {
    "td3": TD3,
    "sac": SAC,
}


def make_agent(
    name: str,
    obs_dim: int,
    action_dim: int,
    action_limit: float,
    embedding_dim: int,
    *,
    rl_cfg: RLConfig,
    key: jax.Array,
) -> RLAgent:
    """Instantiate an RLAgent subclass by algorithm name."""
    agent_cls = _AGENTS.get(name)
    if agent_cls is None:
        choices = tuple(_AGENTS)
        raise ValueError(f"Unsupported agent={name!r}; choose from {choices}")
    return agent_cls(
        obs_dim,
        action_dim,
        action_limit,
        embedding_dim,
        rl_cfg=rl_cfg,
        key=key,
    )
