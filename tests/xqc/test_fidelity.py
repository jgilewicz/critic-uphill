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
import pytest

from config import XQCConfig
from xqc.layers import trainable
from xqc.learner import (
    Noise,
    XQCNets,
    XQCState,
    init_nets,
    init_state,
    make_spec,
    update_step,
)

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
