from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from agents.base import OfflineAgent
from common.td3 import FQESettings, TD3Hyperparams
from config import TD3RLConfig, XQCConfig
from xqc.layers import Dense
from xqc.learner import (
    XQCNets,
    XQCState,
    init_nets,
    init_state,
    make_spec,
    make_xqc_train,
)
from xqc.networks import CriticEnsemble, Trunk, sample_tanh_gaussian


class XQCActionHead(eqx.Module):
    """Mean layer and tanh of the XQC actor, scaled to the env bounds."""

    mean: Dense
    action_limit: float = eqx.field(static=True)

    def __call__(self, z: jax.Array) -> jax.Array:
        return jnp.tanh(self.mean(z)) * self.action_limit


class MinQ(eqx.Module):
    """Eval-mode expected Q of the lower critic, in policy action scale."""

    critic: CriticEnsemble
    action_limit: float = eqx.field(static=True)

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        q = self.critic.q_eval(state, action / self.action_limit)
        return jnp.min(q, axis=0)[None]


class MemberQ(eqx.Module):
    """Eval-mode expected Q of one critic member."""

    critic: CriticEnsemble
    action_limit: float = eqx.field(static=True)
    index: int = eqx.field(static=True)

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        q = self.critic.q_eval(state, action / self.action_limit)
        return q[self.index][None]


@eqx.filter_jit
def _act_sample(nets: XQCNets, state: jax.Array, key: jax.Array) -> jax.Array:
    mean, log_std, _ = nets.actor.forward(state[None], False)
    eps = jax.random.normal(key, mean.shape)
    return sample_tanh_gaussian(mean, log_std, eps)[0][0]


class XQCOfflineAgent(OfflineAgent):
    """Read-only XQC agent in the interface of the offline pipeline.

    Embedding is the actor trunk with running BatchNorm statistics, the
    head is the mean layer plus tanh. The critic the actor follows is the
    minimum over the 2 members (signal q1), member 0 is signal q2.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_limit: float,
        nets: XQCNets,
        *,
        reward_scale: float,
    ) -> None:
        super().__init__(obs_dim, action_dim, action_limit)
        self._nets = nets
        self.reward_scale = reward_scale
        self._unravel_head = ravel_pytree(self.actor_head)[1]

    @property
    def nets(self) -> XQCNets:
        return self._nets

    @property
    def embedding(self) -> Trunk:
        return self.nets.actor.trunk

    @property
    def actor_head(self) -> XQCActionHead:
        return XQCActionHead(self.nets.actor.mean, self.action_limit)

    @property
    def critic1(self) -> MinQ:
        return MinQ(self.nets.critic, self.action_limit)

    @property
    def critic2(self) -> MemberQ:
        return MemberQ(self.nets.critic, self.action_limit, 0)

    @property
    def signal_mapping(self) -> dict[str, str]:
        return {
            "q1": "min over the 2 critics (the critic the actor follows)",
            "q2": "critic member 0 alone",
            "shadow": "unavailable (no shadow critic)",
        }

    @property
    def fqe(self) -> FQESettings:
        """FQE with the TD3 defaults, identical for every algorithm."""
        rl = TD3RLConfig()
        hyper = TD3Hyperparams(
            gamma=rl.gamma,
            tau=rl.tau,
            policy_noise=rl.policy_noise,
            noise_clip=rl.noise_clip,
            action_limit=self.action_limit,
            policy_freq=rl.policy_freq,
            batch_size=rl.batch_size,
        )
        return FQESettings(hyper, rl.critic_hidden_dims, rl.critic_lr)

    def q1_at_actor(self, state: np.ndarray) -> float:
        """Q in raw reward units (the critic learns normalized rewards)."""
        return super().q1_at_actor(state) * self.reward_scale


class XQCAgent(XQCOfflineAgent):
    """XQC agent that also trains: owns the optimizers and the RNG."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_limit: float,
        *,
        cfg: XQCConfig,
        total_steps: int,
        seed: int,
    ) -> None:
        init_key, _ = jax.random.split(jax.random.key(seed))
        nets = init_nets(cfg, obs_dim, action_dim, init_key)
        self.cfg = cfg
        self.spec = make_spec(cfg, action_dim, total_steps)
        # raw key data, so the state serializes as plain arrays
        rng = jax.random.key_data(jax.random.fold_in(jax.random.key(seed), 1))
        self.state: XQCState = init_state(nets, self.spec.tx, rng)
        super().__init__(
            obs_dim, action_dim, action_limit, nets, reward_scale=1.0
        )
        self._train = make_xqc_train(
            self.spec,
            batch_size=cfg.batch_size,
            updates=cfg.updates_per_step,
        )

    @property
    def nets(self) -> XQCNets:
        return self.state.nets

    @property
    def gamma(self) -> float:
        return self.spec.hyper.gamma

    def act(self, state: np.ndarray, key: jax.Array) -> np.ndarray:
        """Sample a policy-scale action (temperature 1, eval-mode BN)."""
        action = _act_sample(self.nets, jnp.asarray(state), key)
        return np.asarray(action, dtype=np.float32)

    def update(
        self, data: Any, size: int, reward_scale: float
    ) -> dict[str, float]:
        """Run updates_per_step fused updates and return mean statistics."""
        self.reward_scale = reward_scale
        self.state, info = self._train(
            self.state,
            data,
            jnp.asarray(size),
            jnp.asarray(reward_scale, dtype=jnp.float32),
        )
        return {name: float(value) for name, value in info.items()}
