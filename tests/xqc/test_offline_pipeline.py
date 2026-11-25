from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from common.checkpoint import BUFFER_COLUMNS, load_checkpoint
from common.replay_buffer import ReplayBuffer, Transition
from config import (
    CheckpointConfig,
    DegConfig,
    EnvConfig,
    ProbeConfig,
    RunConfig,
    RunEvalConfig,
    WandbConfig,
    XQCConfig,
)
from environments import env_dims
from xqc.agent import XQCAgent
from xqc.checkpoint import XQCCheckpointer
from xqc.normalizer import RewardNormalizer

ROOT = Path(__file__).resolve().parents[2]
ENV_ID = "Hopper-v5"
SIGNALS = {
    "q1",
    "q2",
    "fqe",
    "own_q1",
    "prefix_return",
    "dist_action",
    "dist_weight",
}
TRAIN = [
    "algorithm=xqc",
    f"env.id={ENV_ID}",
    f"eval_env.id={ENV_ID}",
    "total_steps=600",
    "algorithm.start_training=200",
    "algorithm.actor_hidden_dims=[32,32]",
    "algorithm.critic_hidden_dims=[32,32]",
    "algorithm.batch_size=32",
    "eval.interval=300",
    "eval.episodes=1",
    "wandb.enabled=false",
]
OFFLINE = [
    "sigmas=[0.05,0.1]",
    "n_perturbations=6",
    "n_states=100",
    "truth_heads=3",
    "truth_episodes=3",
    "prefix_seeds=2",
    "prefix_h=30",
    "fqe_steps=100",
    "step_sigmas=[0.05]",
    "pair_sigmas=[0.05]",
    "n_pairs=3",
    "max_envs=8",
]


def _run(module: str, args: list[str], cwd: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _leaves(node: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, f"{path}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _leaves(value, f"{path}[{i}]")
    else:
        yield path, node


def _assert_finite(result: dict[str, Any], skip: tuple[str, ...]) -> None:
    for path, value in _leaves(result):
        if path.startswith(skip) or not isinstance(value, float):
            continue
        assert math.isfinite(value), f"non-finite {path}={value}"


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("xqc_run")
    ckpt = root / "ckpt"
    _run(
        "train",
        [*TRAIN, f"checkpoint.dir={ckpt}", f"hydra.run.dir={root}/hydra"],
        root,
    )
    return ckpt


def _measure(checkpoints: Path, kind: str) -> dict[str, Any]:
    out = checkpoints.parent / f"{kind}.json"
    _run(
        "measure_offline",
        [
            f"checkpoint={checkpoints / 'p100'}",
            f"out={out}",
            f"measurement={kind}",
            f"hydra.run.dir={checkpoints.parent}/hydra_{kind}",
            *OFFLINE,
        ],
        checkpoints.parent,
    )
    return json.loads(out.read_text())


def test_three_checkpoints_with_full_state(checkpoints: Path) -> None:
    names = sorted(p.name for p in checkpoints.iterdir())
    assert names == ["p010", "p050", "p100"]
    for name in names:
        ckpt = load_checkpoint(checkpoints / name)
        meta = ckpt.meta
        assert meta["algo"] == "xqc"
        assert 0.95 <= meta["gamma"] <= 0.995
        assert meta["reward_scale"] > 0.0
        assert {"jax", "optax", "equinox"} <= set(meta["versions"])
        for column in BUFFER_COLUMNS:
            assert ckpt.arrays[f"buffer_{column}"].shape[0] > 0
        assert isinstance(ckpt.agent, XQCAgent)
        assert np.asarray(ckpt.agent.state.rng).shape == (2,)
    assert np.abs(ckpt.arrays["buffer_action"]).max() <= 1.0


@pytest.mark.parametrize(
    ("env_id", "action_limit"),
    [("Hopper-v5", 1.0), ("Humanoid-v5", 0.4)],
)
def test_checkpoint_holds_env_scale_actions(
    tmp_path: Path, env_id: str, action_limit: float
) -> None:
    dims = env_dims(env_id)
    assert dims.action_limit == pytest.approx(action_limit)
    cfg = XQCConfig(
        actor_hidden_dims=(16, 16),
        critic_hidden_dims=(16, 16),
        start_training=1,
    )
    run_cfg = RunConfig(
        env=EnvConfig(env_id),
        eval_env=EnvConfig(env_id),
        eval=RunEvalConfig(),
        deg=DegConfig(),
        probe=ProbeConfig(),
        wandb=WandbConfig(enabled=False),
        checkpoint=CheckpointConfig(dir=str(tmp_path), fractions=(1.0,)),
        seed=0,
        total_steps=10,
        protocol=5,
        algorithm=cfg,
    )
    agent = XQCAgent(
        dims.obs_dim,
        dims.action_dim,
        dims.action_limit,
        cfg=cfg,
        total_steps=10,
        seed=0,
    )
    buffer = ReplayBuffer(32, dims.obs_dim, dims.action_dim)
    actions = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
    policy_actions = np.broadcast_to(
        actions[:, None], (10, dims.action_dim)
    ).copy()
    zeros_obs = np.zeros((10, dims.obs_dim), dtype=np.float32)
    buffer.add(
        Transition(
            state=zeros_obs,
            action=policy_actions,
            reward=np.ones((10, 1), dtype=np.float32),
            next_state=zeros_obs,
            done=np.zeros((10, 1), dtype=np.float32),
        )
    )
    ckpt_writer = XQCCheckpointer(
        run_cfg, agent, buffer, RewardNormalizer(agent.gamma)
    )
    ckpt_writer.maybe_save(10)
    ckpt = load_checkpoint(tmp_path / "p100")
    assert ckpt.meta["action_limit"] == pytest.approx(action_limit)
    np.testing.assert_allclose(
        ckpt.arrays["buffer_action"],
        policy_actions * np.float32(action_limit),
    )
    assert np.abs(ckpt.arrays["buffer_action"]).max() == pytest.approx(
        action_limit
    )


def test_signals_measurement(checkpoints: Path) -> None:
    result = _measure(checkpoints, "signals")
    assert result["algo"] == "xqc"
    assert set(result["perturbations"]["signals"]) == SIGNALS
    assert "shadow" not in result["perturbations"]["signals"]
    assert "unavailable" in result["signal_mapping"]["shadow"]
    assert result["population"] is None
    _assert_finite(result, skip=("/config",))


def test_extra_measurement(checkpoints: Path) -> None:
    _assert_finite(_measure(checkpoints, "extra"), skip=("/config",))


def test_pairs_measurement(checkpoints: Path) -> None:
    _assert_finite(_measure(checkpoints, "pairs"), skip=("/config",))
