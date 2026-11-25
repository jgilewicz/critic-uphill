"""Dump XQC reference updates for tests/xqc/test_fidelity.py.

Run with the interpreter of the reference checkout (flax 0.8.4, jax 0.4.30)
and that checkout as working directory, never from this repository:

    cd <xqc-ref> && .venv/bin/python <repo>/tests/xqc/ref_fidelity.py out.npz
"""

from __future__ import annotations

import sys
from typing import Any

import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from xqc.agents.xqc.xqc_learner import XQCLearner
from xqc.normalization import RewardNormalizer
from xqc.replay_buffer import Batch

OBS_DIM = 11
ACTION_DIM = 3
BATCH = 256
STEPS = 7
TOTAL_STEPS = 1000
GAMMA = 0.995


def make_learner() -> XQCLearner:
    return XQCLearner(
        seed=0,
        observations=jnp.zeros((1, OBS_DIM)),
        actions=jnp.zeros((1, ACTION_DIM)),
        actor_lr=3e-4,
        critic_lr=3e-4,
        temp_lr=3e-4,
        hidden_dims_critic=[512] * 4,
        policy_delay=3,
        hidden_dims_actor=[256] * 4,
        tau=0.005,
        target_entropy="auto",
        init_temperature=0.01,
        num_seeds=1,
        updates_per_step=2,
        max_v=5.0,
        min_v=-5.0,
        critic_loss="categorical",
        target_net_update_freq=1,
        n_critics=2,
        num_interactions=TOTAL_STEPS,
        lr_end=3e-5,
        pre_activation_bn=1,
        weight_decay=0.0,
        normalize_last_layer=1,
        use_layer_norm=0,
        use_batch_norm=1,
        decay_bn=0,
        use_weight_norm=1,
        skip_connections=0,
    )


def flat(prefix: str, tree: Any, out: dict[str, np.ndarray]) -> None:
    if tree is None:
        return
    for path, value in flax.traverse_util.flatten_dict(
        dict(tree), sep="/"
    ).items():
        out[f"{prefix}/{path}"] = np.asarray(value)[0]


def snapshot(prefix: str, learner: XQCLearner, out: dict[str, Any]) -> None:
    for name in ("actor", "critic", "target_critic", "temperature"):
        model = getattr(learner, name)
        flat(f"{prefix}/{name}/params", model.params, out)
        flat(f"{prefix}/{name}/stats", model.batch_stats, out)


def base_samples(
    learner: XQCLearner, batch: Batch, keys: tuple[Any, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Standard-normal draws behind the TFP samples of one update step."""
    actor = jax.tree.map(lambda x: x[0], learner.actor)
    critic_key, actor_key = keys
    next_dist = actor(batch.next_observations[0, 0])
    train_dist, _ = actor.apply(
        actor.params,
        batch.observations[0, 0],
        batch_stats=actor.batch_stats,
        mutable=["batch_stats", "activations"],
        training=True,
    )
    eps = []
    for dist, key in ((next_dist, critic_key), (train_dist, actor_key)):
        base = dist.distribution
        x = base.sample(seed=key)
        assert np.allclose(np.tanh(x), dist.sample(seed=key), atol=1e-6)
        eps.append(np.asarray((x - base.mean()) / base.stddev()))
    return eps[0], eps[1]


def make_batch(rng: np.random.Generator) -> Batch:
    def rand(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32)

    return Batch(
        observations=rand(1, 1, BATCH, OBS_DIM),
        actions=rng.uniform(-1, 1, (1, 1, BATCH, ACTION_DIM)).astype(
            np.float32
        ),
        rewards=0.5 * rand(1, 1, BATCH),
        masks=(rng.random((1, 1, BATCH)) < 0.9).astype(np.float32),
        next_observations=rand(1, 1, BATCH, OBS_DIM),
        discount=np.full((1, 1, BATCH, 1), GAMMA, np.float32),
    )


def dump_normalizer(
    rng: np.random.Generator, out: dict[str, Any], n_steps: int = 40
) -> None:
    norm = RewardNormalizer(n_seeds=1, gamma=GAMMA, max_v=5.0)
    rewards = rng.normal(loc=1.5, scale=2.0, size=n_steps).astype(np.float64)
    dones = (rng.random(n_steps) < 0.15).astype(bool)
    scales = np.empty(n_steps, dtype=np.float64)
    variances = np.empty(n_steps, dtype=np.float64)
    for idx, (reward, done) in enumerate(zip(rewards, dones)):
        norm.update(
            np.asarray([reward], dtype=np.float64),
            np.asarray([done], dtype=bool),
        )
        variances[idx] = norm.G_rms.var[0]
        scales[idx] = 1.0 / norm.normalize(np.ones((1, 1, 1)))[0, 0, 0]
    out["norm/rewards"] = rewards
    out["norm/dones"] = dones
    out["norm/scales"] = scales
    out["norm/vars"] = variances


def main(path: str) -> None:
    rng = np.random.default_rng(0)
    learner = make_learner()
    out: dict[str, Any] = {}
    snapshot("init", learner, out)
    for step in range(STEPS):
        batch = make_batch(rng)
        for name, value in batch._asdict().items():
            out[f"step{step}/batch/{name}"] = value[0, 0]
        _, critic_key, actor_key = jax.random.split(learner.rng[0], 3)
        eps_c, eps_a = base_samples(learner, batch, (critic_key, actor_key))
        out[f"step{step}/eps/critic"] = eps_c
        out[f"step{step}/eps/actor"] = eps_a
        info = learner.update(batch, num_updates=1)
        for name in ("critic_loss", "actor_loss", "entropy", "temperature"):
            out[f"step{step}/info/{name}"] = np.asarray(info[name])[0]
        snapshot(f"step{step}", learner, out)
    dump_normalizer(rng, out)
    np.savez(path, **out)
    print(f"wrote {len(out)} arrays to {path}")


if __name__ == "__main__":
    main(sys.argv[1])
