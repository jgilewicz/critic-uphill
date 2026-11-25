"""Port vs the original XQC code (commit 9a6832b) over a few update steps.

The reference runs in its own virtualenv (flax 0.8.4, jax 0.4.30). Set
XQC_REF to the checkout, the test is skipped when it is missing. Both sides
start from the same parameters and see the same batches and standard-normal
draws, so any gap is numerical or structural.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from common.replay_buffer import sample_batch
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
from tests.xqc.conftest import random_buffer
from xqc.layers import trainable
from xqc.learner import (
    Noise,
    XQCNets,
    XQCState,
    discount_for_horizon,
    init_nets,
    init_state,
    make_optimizers,
    make_spec,
    make_xqc_train,
    update_step,
)
from xqc.normalizer import RewardNormalizer
from xqc.trainer import XQCTrainer

# must match tests/xqc/ref_fidelity.py (runs in another virtualenv)
OBS_DIM = 11
ACTION_DIM = 3
STEPS = 7
TOTAL_STEPS = 1000

REF = Path(os.environ.get("XQC_REF", "~/Home/research/xqc-ref")).expanduser()
REF_PYTHON = REF / ".venv" / "bin" / "python"
DRIVER = Path(__file__).with_name("ref_fidelity.py")
BN_FIELDS = {
    "scale": ("params", "scale"),
    "bias": ("params", "bias"),
    "running_mean": ("stats", "mean"),
    "running_var": ("stats", "var"),
}


# Adam divides by sqrt(v), so a parameter with a near-zero gradient moves by
# +-lr whichever way float noise points: parameter gaps stay below 2-3 lr.
SCALAR_ATOL, SCALAR_RTOL = 1e-4, 1e-3
PARAM_ATOL = 1e-3
STATS_ATOL = 2e-4
MIN_COSINE = 0.99
REAL_UPDATE = 1e-4


def _names(path: tuple[Any, ...]) -> list[str]:
    return [str(p.name) if hasattr(p, "name") else str(p.idx) for p in path]


def _ref_key(names: list[str], critic: bool) -> tuple[str, str]:
    """(collection, flax path) of one leaf of the port's actor or critic."""
    head = "VmapCritic_0/" if critic else ""
    names = names[1:] if critic else names
    if names[0] == "trunk":
        if names[1] == "embed_norm":
            leaf = "BatchNormEmbedder_0/BatchNorm_0"
            collection, field = BN_FIELDS[names[2]]
            return collection, f"{head}{leaf}/{field}"
        block = f"MLP_0/XQCBlock_{names[2]}"
        if names[3] == "dense":
            return "params", f"{head}{block}/Dense_0/{names[4]}"
        collection, field = BN_FIELDS[names[4]]
        return collection, f"{head}{block}/BatchNorm_0/{field}"
    layer = {"value": "predictor_scalar/value"}.get(
        names[0], f"predictor_tanh_gauss/{names[0]}"
    )
    return "params", f"{head}{layer}/{names[1]}"


def _load(template: Any, ref: Any, prefix: str, *, critic: bool) -> Any:
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    out = []
    for path, _ in leaves:
        collection, key = _ref_key(_names(path), critic)
        out.append(jnp.asarray(ref[f"{prefix}/{collection}/{key}"]))
    return jax.tree_util.tree_unflatten(treedef, out)


def _nets_from_ref(ref: Any, prefix: str, template: XQCNets) -> XQCNets:
    return XQCNets(
        actor=_load(template.actor, ref, f"{prefix}/actor", critic=False),
        critic=_load(template.critic, ref, f"{prefix}/critic", critic=True),
        target_critic=_load(
            template.target_critic, ref, f"{prefix}/target_critic", critic=True
        ),
        log_temp=jnp.asarray(ref[f"{prefix}/temperature/params/log_temp"]),
    )


class Gap(NamedTuple):
    abs: float
    rel: float


def _gap(port: Any, ref: Any) -> Gap:
    port, ref = np.asarray(port, np.float64), np.asarray(ref, np.float64)
    diff = np.abs(port - ref)
    return Gap(
        float(diff.max(initial=0.0)),
        float((diff / (np.abs(ref) + 1e-12)).max(initial=0.0)),
    )


class Leaf(NamedTuple):
    key: str
    kind: str
    port: np.ndarray


def _leaves(tree: Any, prefix: str, *, critic: bool) -> list[Leaf]:
    paths, _ = jax.tree_util.tree_flatten_with_path(tree)
    flags = jax.tree.leaves(trainable(tree))
    out = []
    for (path, value), is_param in zip(paths, flags, strict=True):
        collection, key = _ref_key(_names(path), critic)
        assert collection == ("params" if is_param else "stats")
        out.append(
            Leaf(f"{prefix}/{collection}/{key}", collection, np.asarray(value))
        )
    return out


def _port_leaves(nets: XQCNets, step: str) -> list[Leaf]:
    return [
        *_leaves(nets.actor, f"{step}/actor", critic=False),
        *_leaves(nets.critic, f"{step}/critic", critic=True),
        *_leaves(nets.target_critic, f"{step}/target_critic", critic=True),
    ]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _check_leaf(
    leaf: Leaf, ref: Any, before: Leaf | None
) -> tuple[Gap, float | None]:
    ref_now = ref[leaf.key]
    gap = _gap(leaf.port, ref_now)
    limit = PARAM_ATOL if leaf.kind == "params" else STATS_ATOL
    assert gap.abs <= limit, f"{leaf.key}: {gap}"
    if before is None or not leaf.key.endswith("kernel"):
        return gap, None
    ref_delta = ref_now - ref[before.key]
    if np.abs(ref_delta).max() < REAL_UPDATE:
        return gap, None
    cos = _cosine(leaf.port - before.port, ref_delta)
    assert cos >= MIN_COSINE, f"{leaf.key}: update cosine {cos:.4f}"
    return gap, cos


@pytest.fixture(scope="module")
def ref_dump(tmp_path_factory: pytest.TempPathFactory) -> Any:
    if not REF_PYTHON.is_file():
        pytest.skip(f"no reference virtualenv at {REF_PYTHON}, set XQC_REF")
    out = tmp_path_factory.mktemp("xqc_ref") / "dump.npz"
    env = {**os.environ, "PYTHONPATH": str(REF)}
    subprocess.run(
        [str(REF_PYTHON), str(DRIVER), str(out)],
        cwd=REF,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return np.load(out)


def _batch(ref: Any, step: int) -> dict[str, jax.Array]:
    pre = f"step{step}/batch"
    return {
        "state": jnp.asarray(ref[f"{pre}/observations"]),
        "action": jnp.asarray(ref[f"{pre}/actions"]),
        "reward": jnp.asarray(ref[f"{pre}/rewards"][:, None]),
        "next_state": jnp.asarray(ref[f"{pre}/next_observations"]),
        "done": jnp.asarray(1.0 - ref[f"{pre}/masks"][:, None]),
    }


def _check_scalars(
    info: dict[str, Any], ref: Any, step: int, report: dict[str, Gap]
) -> None:
    for name in ("critic_loss", "actor_loss", "entropy", "temperature"):
        expected = ref[f"step{step}/info/{name}"]
        gap = _gap(info[name], expected)
        report[f"step{step}/info/{name}"] = gap
        limit = SCALAR_ATOL + SCALAR_RTOL * abs(float(expected))
        assert gap.abs <= limit, f"step{step} {name}: {gap}"


def test_updates_match_original(ref_dump: Any, capsys: Any) -> None:
    cfg = XQCConfig()
    template = init_nets(cfg, OBS_DIM, ACTION_DIM, jax.random.key(0))
    spec = make_spec(cfg, ACTION_DIM, TOTAL_STEPS)
    nets = _nets_from_ref(ref_dump, "init", template)
    state: XQCState = init_state(nets, spec.tx, jnp.zeros(2, jnp.uint32))
    step_fn = eqx.filter_jit(update_step)
    before = {
        leaf.key.replace("init/", "", 1): leaf
        for leaf in _port_leaves(nets, "init")
    }
    report: dict[str, Gap] = {}
    cosines: list[float] = []
    for step in range(STEPS):
        noise = Noise(
            jnp.asarray(ref_dump[f"step{step}/eps/critic"]),
            jnp.asarray(ref_dump[f"step{step}/eps/actor"]),
        )
        state, info = step_fn(state, _batch(ref_dump, step), noise, spec)
        _check_scalars(info, ref_dump, step, report)
        prefix = f"step{step}"
        for leaf in _port_leaves(state.nets, prefix):
            rel = leaf.key.split("/", 1)[1]
            old = before.get(rel)
            report[leaf.key], cos = _check_leaf(leaf, ref_dump, old)
            if cos is not None:
                cosines.append(cos)
        before = {
            leaf.key.split("/", 1)[1]: leaf
            for leaf in _port_leaves(state.nets, prefix)
        }
        report[f"{prefix}/temperature/log_temp"] = _gap(
            state.nets.log_temp,
            ref_dump[f"{prefix}/temperature/params/log_temp"],
        )
    _print_report(report, cosines, capsys)


def _print_report(
    report: dict[str, Gap], cosines: list[float], capsys: Any
) -> None:
    groups: dict[str, list[Gap]] = {}
    for key, gap in report.items():
        parts = key.split("/")
        group = parts[2] if parts[1] == "info" else f"{parts[1]}/{parts[2]}"
        groups.setdefault(group, []).append(gap)
    with capsys.disabled():
        print("\nfidelity max gaps over all steps (abs, rel):")
        for group, gaps in sorted(groups.items()):
            print(
                f"  {group:26s} abs={max(g.abs for g in gaps):.3e} "
                f"rel={max(g.rel for g in gaps):.3e}"
            )
        print(
            f"  update cosine over {len(cosines)} kernels: "
            f"min={min(cosines):.5f}"
        )


def test_two_step_scan_matches_original(ref_dump: Any) -> None:
    cfg = XQCConfig()
    template = init_nets(cfg, OBS_DIM, ACTION_DIM, jax.random.key(0))
    spec = make_spec(cfg, ACTION_DIM, TOTAL_STEPS)
    init = init_state(
        _nets_from_ref(ref_dump, "init", template),
        spec.tx,
        jnp.zeros(2, jnp.uint32),
    )
    batches = jax.tree.map(
        lambda a, b: jnp.stack([a, b]), _batch(ref_dump, 0), _batch(ref_dump, 1)
    )
    noises = Noise(
        jnp.stack(
            [
                jnp.asarray(ref_dump["step0/eps/critic"]),
                jnp.asarray(ref_dump["step1/eps/critic"]),
            ]
        ),
        jnp.stack(
            [
                jnp.asarray(ref_dump["step0/eps/actor"]),
                jnp.asarray(ref_dump["step1/eps/actor"]),
            ]
        ),
    )

    def scan_body(
        carry: XQCState, inputs: tuple[dict[str, jax.Array], Noise]
    ) -> tuple[XQCState, dict[str, jax.Array]]:
        batch, noise = inputs
        return update_step(carry, batch, noise, spec)

    state, infos = eqx.filter_jit(
        lambda s, b, n: jax.lax.scan(scan_body, s, (b, n), length=2)
    )(init, batches, noises)
    report: dict[str, Gap] = {}
    for step in range(2):
        step_info = {k: v[step] for k, v in infos.items()}
        _check_scalars(step_info, ref_dump, step, report)
    for leaf in _port_leaves(state.nets, "step1"):
        _check_leaf(leaf, ref_dump, None)
    log_temp_gap = _gap(
        state.nets.log_temp, ref_dump["step1/temperature/params/log_temp"]
    )
    assert log_temp_gap.abs <= PARAM_ATOL


def test_make_xqc_train_scan_matches_sequential() -> None:
    cfg = XQCConfig(
        actor_hidden_dims=(16, 16),
        critic_hidden_dims=(16, 16),
        batch_size=32,
        updates_per_step=2,
    )
    spec = make_spec(cfg, ACTION_DIM, total_steps=50)
    nets = init_nets(cfg, OBS_DIM, ACTION_DIM, jax.random.key(0))
    init = init_state(nets, spec.tx, jax.random.key_data(jax.random.key(3)))
    data = random_buffer(jax.random.key(1))
    size = jnp.asarray(data[0].shape[0])
    scale = jnp.asarray(2.5, dtype=jnp.float32)
    train_fn = make_xqc_train(spec, batch_size=cfg.batch_size, updates=2)
    scanned_state, scanned_info = train_fn(init, data, size, scale)

    seq_state = init
    seq_infos = []
    step_fn = eqx.filter_jit(update_step)
    for _ in range(2):
        rng, batch_key, critic_key, actor_key = jax.random.split(
            seq_state.rng, 4
        )
        batch = sample_batch(data, size, batch_key, cfg.batch_size)
        batch["reward"] = batch["reward"] / scale
        shape = batch["action"].shape
        noise = Noise(
            jax.random.normal(critic_key, shape),
            jax.random.normal(actor_key, shape),
        )
        seq_state, info = step_fn(
            seq_state._replace(rng=rng), batch, noise, spec
        )
        seq_infos.append(info)

    for a, b in zip(
        jax.tree.leaves(scanned_state), jax.tree.leaves(seq_state), strict=True
    ):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=PARAM_ATOL)
    for name in scanned_info:
        expected = np.mean([np.asarray(x[name]) for x in seq_infos])
        np.testing.assert_allclose(scanned_info[name], expected, rtol=1e-4)


def test_lr_schedule_at_zero_mid_end() -> None:
    cfg = XQCConfig()
    total_steps = 50
    horizon = total_steps * cfg.updates_per_step
    tx = make_optimizers(cfg, total_steps)
    ref_sched = optax.linear_schedule(cfg.actor_lr, cfg.lr_end, horizon)
    assert float(ref_sched(0)) == pytest.approx(cfg.actor_lr)
    assert float(ref_sched(horizon // 2)) == pytest.approx(
        0.5 * (cfg.actor_lr + cfg.lr_end)
    )
    assert float(ref_sched(horizon)) == pytest.approx(cfg.lr_end)

    ref_adam = optax.adam(ref_sched)
    param = jnp.asarray(1.0, dtype=jnp.float32)
    grad = jnp.asarray(0.5, dtype=jnp.float32)
    c_opt = tx.critic.init(param)
    r_opt = ref_adam.init(param)
    for step in range(horizon + 1):
        c_upd, c_opt = tx.critic.update(grad, c_opt, param)
        r_upd, r_opt = ref_adam.update(grad, r_opt, param)
        if step in (0, horizon // 2, horizon):
            np.testing.assert_allclose(c_upd, r_upd, rtol=1e-7, atol=1e-9)

    a_opt = tx.actor.init(param)
    r_opt = ref_adam.init(param)
    for step in range(7):
        a_upd, a_opt = tx.actor.update(grad, a_opt, param)
        if step % cfg.policy_delay == 0:
            r_upd, r_opt = ref_adam.update(grad, r_opt, param)
            np.testing.assert_allclose(a_upd, r_upd, rtol=1e-7, atol=1e-9)
        else:
            np.testing.assert_array_equal(a_upd, jnp.zeros_like(param))


def test_reward_normalizer_matches_original(ref_dump: Any) -> None:
    norm = RewardNormalizer(discount_for_horizon(XQCConfig().horizon))
    rewards = ref_dump["norm/rewards"]
    dones = ref_dump["norm/dones"]
    ref_vars = ref_dump["norm/vars"]
    ref_scales = ref_dump["norm/scales"]
    for reward, done, ref_var, ref_scale in zip(
        rewards, dones, ref_vars, ref_scales, strict=True
    ):
        norm.update(float(reward), bool(done))
        assert norm.var == pytest.approx(float(ref_var), rel=1e-12, abs=1e-12)
        assert norm.scale == pytest.approx(
            float(ref_scale), rel=1e-12, abs=1e-12
        )


class _ScriptedEnv:
    """Deterministic Hopper-shaped env exercising termination and truncation."""

    def __init__(self, script: list[tuple[bool, bool]]) -> None:
        self._script = script
        self._idx = 0

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        del seed
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        del action
        term, trunc = self._script[self._idx]
        self._idx += 1
        obs = np.full(OBS_DIM, float(self._idx), dtype=np.float32)
        return obs, 1.0, term, trunc, {}

    def close(self) -> None:
        pass


def test_trainer_update_count_and_truncation_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = [
        (False, False),
        (True, False),
        (False, False),
        (False, True),
        (True, True),
    ] + [(False, False)] * 10
    monkeypatch.setattr(
        "xqc.trainer.make_env", lambda _env_id: _ScriptedEnv(script)
    )
    cfg = XQCConfig(
        actor_hidden_dims=(16, 16),
        critic_hidden_dims=(16, 16),
        batch_size=4,
        start_training=5,
        updates_per_step=2,
        buffer_capacity=32,
    )
    run_cfg = RunConfig(
        env=EnvConfig("Hopper-v5"),
        eval_env=EnvConfig("Hopper-v5"),
        eval=RunEvalConfig(episodes=1, interval=100),
        deg=DegConfig(),
        probe=ProbeConfig(),
        wandb=WandbConfig(enabled=False),
        checkpoint=CheckpointConfig(),
        seed=0,
        total_steps=12,
        protocol=5,
        algorithm=cfg,
    )
    trainer = XQCTrainer(run_cfg, cfg)
    update_steps: list[int] = []
    orig_update = trainer._agent.update

    def tracked_update(*args: Any, **kwargs: Any) -> Any:
        update_steps.append(trainer._budget.env_steps)
        return orig_update(*args, **kwargs)

    monkeypatch.setattr(trainer._agent, "update", tracked_update)
    trainer.train()

    assert update_steps == list(range(6, 13))
    assert int(trainer._agent.state.opt.critic[0].count) == 14
    dones = np.asarray(trainer._buffer.data[4][:12, 0])
    expected = np.array([0.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 7, dtype=np.float32)
    np.testing.assert_array_equal(dones, expected)
