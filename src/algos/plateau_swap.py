from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from analysis.runs import load_spec
from common.replay_buffer import Buffer
from common.td3 import FQETrainFn, TD3Config, TD3State, make_fqe_train
from offline.critic_degeneracy import perturb
from offline.curvature_es import cell_rng, curvature_on, flat_heads
from offline.gibo import (
    cost,
    line_search,
    new_arm,
    random_bases,
    run_rounds,
    verify,
)

Record = dict[str, Any]
Metrics = dict[str, float]
ROOT = Path(__file__).resolve().parents[2]
# slopes in return per 100k env steps
SLOPE_UNIT = 100_000


@dataclass(frozen=True)
class SwapConfig:
    enabled: bool = False
    # arm G-d of this spec, unchanged: [subspace] dim and key, [gibo],
    # [step], [verify]. path relative to the repo root
    phase_spec: str = "experiments/gibo-pilot-v1/experiment.toml"
    min_step: int = 200_000
    window: int = 100_000
    rel_slope: float = 0.005
    cooldown: int = 100_000
    max_phase_steps: int = 88_000
    states: int = 2000
    state_window: int = 20_000
    perturbations: int = 40
    sigma: float | None = None
    seed_base: int = 20_000_000
    key: int = 20261124
    critic_updates: int = 5000
    rollback_evals: int = 3

    def __post_init__(self) -> None:
        if self.enabled and (self.sigma is None or self.sigma <= 0):
            raise ValueError(
                f"swap.sigma={self.sigma}; swap.enabled=true needs the "
                "sigma-5% calibration of critic-degeneracy-v1 (> 0)"
            )


@dataclass
class Pending:
    # the TD3 state before the swap and the mean of the evals before it
    before: TD3State
    before_mean: float
    after: list[float] = field(default_factory=list)


@dataclass
class SwapState:
    cfg: SwapConfig
    spec: Record
    train_critic: FQETrainFn
    evals: list[tuple[int, float]] = field(default_factory=list)
    # (slope, window mean) of the trigger that fired, the phase runs next
    due: tuple[float, float] | None = None
    last_end: int | None = None
    pending: Pending | None = None
    phases: int = 0
    phase_steps: int = 0


def init_swap(
    cfg: SwapConfig,
    td3_cfg: TD3Config,
    critic_opt: optax.GradientTransformation,
) -> SwapState:
    path = ROOT / cfg.phase_spec
    if not path.is_file():
        raise FileNotFoundError(
            f"swap.phase_spec {path} missing. it holds the G-d phase "
            "hyperparameters of gibo-pilot-v1"
        )
    return SwapState(cfg, load_spec(path), make_fqe_train(td3_cfg, critic_opt))


def window_slope(
    evals: list[tuple[int, float]], t: int, window: int
) -> tuple[float, float]:
    # OLS slope per SLOPE_UNIT steps and mean of the evals in (t - window, t]
    pts = np.array([(s, v) for s, v in evals if t - window < s <= t], float)
    if len(pts) < 2:
        return np.inf, float("nan")
    slope = np.polyfit(pts[:, 0], pts[:, 1], 1)[0] * SLOPE_UNIT
    return float(slope), float(pts[:, 1].mean())


def trigger(sw: SwapState, t: int, budget: int) -> tuple[float, float] | None:
    c = sw.cfg
    cooled = sw.last_end is None or t - sw.last_end >= c.cooldown
    if t < c.min_step or not cooled or t + c.max_phase_steps > budget:
        return None
    slope, mean = window_slope(sw.evals, t, c.window)
    return (slope, mean) if slope <= c.rel_slope * abs(mean) else None


def restore_actor(td3: TD3State, old: TD3State) -> TD3State:
    # embedding + head, online and target, and the actor's Adam state
    online = td3.online._replace(
        embedding=old.online.embedding, actor=old.online.actor
    )
    target = td3.target._replace(
        embedding=old.target.embedding, actor=old.target.actor
    )
    return td3._replace(
        online=online, target=target, actor_opt_state=old.actor_opt_state
    )


def check_rollback(
    sw: SwapState, td3: TD3State, value: float
) -> tuple[TD3State, Metrics]:
    p = sw.pending
    assert p is not None
    p.after.append(value)
    if len(p.after) < sw.cfg.rollback_evals:
        return td3, {}
    sw.pending = None
    after = float(np.mean(p.after))
    rolled = after < p.before_mean
    print(
        f"phase rollback={rolled} after={after:.1f} before={p.before_mean:.1f}"
    )
    metrics = {
        "phase/rolled_back": float(rolled),
        "phase/after_mean": after,
        "phase/before_mean": p.before_mean,
    }
    return (restore_actor(td3, p.before) if rolled else td3), metrics


def after_eval(
    sw: SwapState, td3: TD3State, t: int, value: float, budget: int
) -> tuple[TD3State, Metrics]:
    # past evals only: the rollback decision of a pending swap, else the
    # trigger (the phase runs in the next iteration)
    sw.evals.append((t, value))
    if sw.pending is not None:
        return check_rollback(sw, td3, value)
    sw.due = trigger(sw, t, budget)
    return td3, {}


def buffer_states(
    buffer: Buffer, n: int, window: int, rng: np.random.Generator
) -> jax.Array:
    # n distinct rows of the newest `window` transitions
    newest = min(window, len(buffer))
    if newest < n:
        raise ValueError(
            f"{newest} buffer transitions < swap.states={n}. the trigger "
            "needs swap.min_step >> swap.states"
        )
    back = rng.choice(newest, n, replace=False)
    idx = (buffer.ptr - 1 - back) % buffer.capacity
    return buffer.state[jnp.asarray(idx)]


def phase_spec(pilot: Record, base: int) -> Record:
    # the pilot's sections, fresh reset seeds for this phase
    return pilot | {
        "gibo": pilot["gibo"] | {"seed_base": base},
        "step": pilot["step"] | {"seed_base": base + 1000},
        "verify": pilot["verify"] | {"seed_base": base + 2000},
    }


def prepare(
    sw: SwapState, td3: TD3State, buffer: Buffer, meta: Record
) -> Record:
    # curvature on buffer states, dist_H reference from sigma perturbations
    c = sw.cfg
    assert c.sigma is not None
    rng = cell_rng(c.key, meta)
    states = buffer_states(buffer, c.states, c.state_window, rng)
    cv = curvature_on(td3.online, states)
    key = jax.random.PRNGKey(int(rng.integers(2**31)))
    heads = perturb(td3.online.actor, c.sigma, c.perturbations, key)
    return {
        "cv": cv,
        "nets": td3.online,
        "meta": meta,
        "buffer": buffer,
        "ref": float(np.mean(cv["dist"](flat_heads(heads)))),
    }


def decide(sw: SwapState, pr: Record) -> tuple[Record, int]:
    # one G-d decision: the arm after verification and its env steps
    meta = pr["meta"]
    base = sw.cfg.seed_base + 1_000_000 * meta["seed"] + 10_000 * sw.phases
    spec = phase_spec(sw.spec, base)
    sub = spec["subspace"]
    p = len(pr["cv"]["theta"])
    basis = random_bases(p, sub["dim"], 1, meta, sub["key"])[0]
    arms = {"G-d": new_arm(pr, basis, "gibo")}
    rounds = run_rounds(pr, arms, spec)
    ls = line_search(pr, arms, spec)
    verify(pr, arms, spec)
    arm = arms["G-d"]
    return arm, cost(arm, rounds, ls)


def swap_actor(
    sw: SwapState, td3: TD3State, buffer: Buffer, head: Any, key: jax.Array
) -> tuple[TD3State, float]:
    # online and target head <- head, then critic-only updates
    td3 = td3._replace(
        online=td3.online._replace(actor=head),
        target=td3.target._replace(actor=head),
    )
    td3, loss = sw.train_critic(
        td3,
        buffer.data,
        jnp.asarray(len(buffer)),
        jnp.asarray(sw.cfg.critic_updates),
        key,
    )
    return td3, float(loss)


def run_phase(
    sw: SwapState, td3: TD3State, buffer: Buffer, meta: Record, key: jax.Array
) -> tuple[TD3State, int, Metrics]:
    assert sw.due is not None
    t, (slope, mean) = meta["env_steps"], sw.due
    pr = prepare(sw, td3, buffer, meta)
    arm, steps = decide(sw, pr)
    ver, chosen = arm["verify"], arm["candidates"][arm["chosen"]]
    accepted = bool(ver["accepted"])
    metrics: Metrics = {}
    if accepted:
        before = [v for _, v in sw.evals[-sw.cfg.rollback_evals :]]
        sw.pending = Pending(td3, float(np.mean(before)))
        head = pr["cv"]["to_head"](arm["thetas"][arm["chosen"]])
        td3, loss = swap_actor(sw, td3, buffer, head, key)
        metrics["phase/critic_loss"] = loss
    sw.due, sw.last_end = None, t + steps
    sw.phases += 1
    sw.phase_steps += steps
    record = {
        "t": t,
        "slope": slope,
        "window_mean": mean,
        "cost": steps,
        "accepted": accepted,
        "ci_lo": ver["ci"][0],
        "ci_hi": ver["ci"][1],
        "stage": ver["stage"],
        "multiple": chosen["multiple"],
        "candidate_y": float(np.mean(chosen["y"])),
        "fallback": chosen["fallback"],
        "fallback_axes": len(arm["fallback_axes"]),
        "ref_dist_h": pr["ref"],
        "h_a_not_concave": pr["cv"]["h_a_not_concave"],
        "count": sw.phases,
        "steps_frac": sw.phase_steps / (t + steps),
    }
    print("phase " + json.dumps(record))
    return (
        td3,
        steps,
        metrics | {f"phase/{k}": float(v) for k, v in record.items()},
    )
