from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from common.replay_buffer import Buffer, Transition
from modules.deep_modules import ActorHead, SharedStateEmbedding

BatchPolicy = Callable[[jax.Array, jnp.ndarray], jnp.ndarray]
StepHook = Callable[
    [int, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], None
]


@jax.jit
def _update_returns(
    returns: jnp.ndarray,
    alive: jnp.ndarray,
    reward: jnp.ndarray,
    terminated: jnp.ndarray,
    truncated: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    returns = returns + reward * alive
    alive = alive & ~(terminated | truncated)
    return returns, alive


def collect_parallel_episode(
    vec_env: gym.vector.VectorEnv,
    key: jax.Array,
    policy: BatchPolicy,
    buffer: Buffer | None,
    horizon: int,
    *,
    on_step: StepHook | None = None,
    store_mask: np.ndarray | None = None,
) -> tuple[jnp.ndarray, jax.Array]:
    states, _ = vec_env.reset(
        seed=int(jax.random.randint(key, (), 0, 2**31 - 1))
    )
    states = jnp.asarray(states)
    num_envs = states.shape[0]
    returns = jnp.zeros(num_envs)
    alive = jnp.ones(num_envs, dtype=bool)
    prev_done = np.zeros(num_envs, dtype=bool)
    # envs outside store_mask still step (shadow rollout) but never reach
    # the buffer
    stored = (
        np.ones(num_envs, dtype=bool)
        if store_mask is None
        else np.asarray(store_mask, dtype=bool)
    )

    for step in range(horizon):
        key, act_key = jax.random.split(key)
        actions = policy(act_key, states)
        next_states, reward, terminated, truncated, _ = vec_env.step(actions)
        returns, alive = _update_returns(
            returns, alive, reward, terminated, truncated
        )
        if on_step is not None:
            on_step(step, reward, terminated, truncated, next_states)

        valid = ~prev_done & stored
        if buffer is not None and valid.any():
            idx = jnp.asarray(np.nonzero(valid)[0])
            buffer.add(
                Transition(
                    state=states[idx],
                    action=jnp.asarray(actions)[idx],
                    reward=reward[idx],
                    next_state=next_states[idx],
                    done=terminated[idx],
                )
            )

        prev_done = np.asarray(terminated | truncated)
        states = next_states

    return returns, key


@eqx.filter_jit
def heads_policy(
    embedding: SharedStateEmbedding, heads: ActorHead, states: jnp.ndarray
) -> jnp.ndarray:
    z = jax.vmap(embedding)(states)
    return jax.vmap(lambda head, zi: head(zi))(heads, z)


@eqx.filter_jit
def stacked_policy(
    embeddings: SharedStateEmbedding, heads: ActorHead, states: jnp.ndarray
) -> jnp.ndarray:
    return jax.vmap(lambda e, h, s: h(e(s)))(embeddings, heads, states)
