from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class Transition(NamedTuple):
    """Batch of environment transitions."""

    state: np.ndarray | jax.Array
    action: np.ndarray | jax.Array
    reward: np.ndarray | jax.Array
    next_state: np.ndarray | jax.Array
    done: np.ndarray | jax.Array


BufferData = tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]


@jax.jit
def _scatter(
    data: BufferData,
    indices: jnp.ndarray,
    batch: BufferData,
) -> BufferData:
    s, a, r, ns, d = data
    state, action, reward, next_state, done = batch
    return (
        s.at[indices].set(state),
        a.at[indices].set(action),
        r.at[indices].set(reward),
        ns.at[indices].set(next_state),
        d.at[indices].set(done),
    )


@jax.jit
def _gather_states(data: BufferData, indices: jnp.ndarray) -> jax.Array:
    return data[0][indices]


def sample_batch(
    data: BufferData, size: jax.Array, key: jax.Array, batch_size: int
) -> dict[str, jnp.ndarray]:
    """Sample a traceable uniform batch inside a jitted training loop."""
    indices = jax.random.randint(key, (batch_size,), 0, size)
    state, action, reward, next_state, done = data
    return {
        "state": state[indices],
        "action": action[indices],
        "reward": reward[indices],
        "next_state": next_state[indices],
        "done": done[indices],
    }


class ReplayBuffer:
    """Fixed-capacity ring replay buffer backed by JAX arrays."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self._data: BufferData = (
            jnp.zeros((capacity, state_dim), dtype=jnp.float32),
            jnp.zeros((capacity, action_dim), dtype=jnp.float32),
            jnp.zeros((capacity, 1), dtype=jnp.float32),
            jnp.zeros((capacity, state_dim), dtype=jnp.float32),
            jnp.zeros((capacity, 1), dtype=jnp.float32),
        )
        self.ptr = 0
        self.size = 0

    @property
    def data(self) -> BufferData:
        """Return underlying tuple of buffer arrays."""
        return self._data

    def add(self, transition: Transition) -> None:
        """Insert a batch of transitions into the ring buffer."""
        n = int(transition.state.shape[0])
        if n == 0:
            return
        indices = (self.ptr + jnp.arange(n)) % self.capacity
        batch: BufferData = (
            jnp.asarray(transition.state, dtype=jnp.float32),
            jnp.asarray(transition.action, dtype=jnp.float32),
            jnp.asarray(transition.reward, dtype=jnp.float32).reshape(n, 1),
            jnp.asarray(transition.next_state, dtype=jnp.float32),
            jnp.asarray(transition.done, dtype=jnp.float32).reshape(n, 1),
        )
        self._data = _scatter(self._data, indices, batch)
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample_states(self, key: jax.Array, count: int) -> jax.Array:
        """Sample `count` states uniformly with replacement."""
        if count <= 0:
            raise ValueError(f"count must be > 0, got {count}")
        if self.size == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        indices = jax.random.randint(key, (count,), 0, self.size)
        return _gather_states(self._data, indices)

    def __len__(self) -> int:
        return self.size
