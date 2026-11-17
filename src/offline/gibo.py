import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from analysis.runs import load_spec
from common import gp
from common.checkpoint import load_nets
from offline.critic_degeneracy import (
    PERTURB,
    as_batch,
    build_heads,
    checkpoint_key,
    episodes_matrix,
    stack_heads,
)
from offline.curvature_es import cell_rng, curvature, flat_heads, solve_alpha

Record = dict[str, Any]
# XLA CPU takes another matmul path below 16 rows: pad every batch to it
MIN_ROWS = 16
ARMS = ("G-b", "G-d", "ES-b")


def cell_name(meta: Record) -> str:
    return f"s{meta['seed']}_step_{meta['env_steps']}.json"


def behaviour_basis(cv: Record, dim: int) -> np.ndarray:
    # top-dim eigenvectors of M = S mean_s J_s^T J_s S, columns in u
    t = jnp.asarray(cv["theta"], "f4")
    jac = jax.vmap(jax.jacrev(cv["act"]), in_axes=(None, 0))(t, cv["z"])
    m = jnp.einsum("sap,saq->pq", jac, jac) / cv["z"].shape[0]
    s = cv["scale"]
    _, vecs = np.linalg.eigh(s[:, None] * np.asarray(m, "f8") * s[None])
    return vecs[:, ::-1][:, :dim]


def random_bases(p: int, dim: int, draws: int, meta: Record, key: int) -> list:
    rng = cell_rng(key, meta)
    return [np.linalg.qr(rng.normal(size=(p, dim)))[0] for _ in range(draws)]


def oracle_gradients(cv: Record, ctx: Record) -> list[np.ndarray]:
    # g_u per sigma level from the 30 local-search-v1 episodes per head
    heads_theta, levels, ep = ctx["heads_theta"], ctx["levels"], ctx["ep"]
    r0 = np.mean(ep["actor"]["returns"])
    out = []
    for level, sigma in enumerate(ctx["sigmas"]):
        idx = np.flatnonzero(levels[1:] == level)
        eps = (heads_theta[idx + 1] - cv["theta"]) / (sigma * cv["scale"])
        r = np.array([np.mean(ep["perturbations"][i]["returns"]) for i in idx])
        out.append((r - r0) @ eps / (len(idx) * sigma))
    return out


def energy(basis: np.ndarray, g: np.ndarray) -> float:
    return float(np.sum((basis.T @ g) ** 2) / np.sum(g**2))


def prepare(
    spec: Record, path: Path, sigmas: list[float], ep: Record
) -> Record:
    # checkpoint, curvature, both subspaces and the oracle gradients
    source = load_spec((spec["dir"] / spec["source"]).resolve())
    curv = load_spec((spec["dir"] / spec["steps_source"]).resolve())
    off = source["offline"]
    nets, meta = load_nets(path)
    key = jax.random.fold_in(checkpoint_key(off, meta), PERTURB)
    heads, levels = build_heads(nets.actor, sigmas, off["perturbations"], key)
    if len(ep["perturbations"]) != len(levels) - 1:
        raise ValueError(
            f"{path}: {len(ep['perturbations'])} local-search-v1 rows vs "
            f"{len(levels) - 1} regenerated perturbations"
        )
    cv = curvature(nets, meta, curv)
    heads_theta = flat_heads(heads)
    ctx = {"heads_theta": heads_theta, "levels": levels, "ep": ep}
    sub = spec["subspace"]
    p = len(cv["theta"])
    return {
        "nets": nets,
        "meta": meta,
        "cv": cv,
        "ref": float(cv["dist"](heads_theta[1:])[levels[1:] == 0].mean()),
        "b": behaviour_basis(cv, sub["dim"]),
        "random": random_bases(
            p, sub["dim"], sub["random_draws"], meta, sub["key"]
        ),
        "g_u": oracle_gradients(cv, ctx | {"sigmas": sigmas}),
    }


def gate_record(pr: Record) -> Record:
    levels = []
    for g in pr["g_u"]:
        rand = [energy(v, g) for v in pr["random"]]
        levels.append(
            {
                "energy_b": energy(pr["b"], g),
                "energy_random": rand,
                "ratio": energy(pr["b"], g) / float(np.mean(rand)),
            }
        )
    return {
        "meta": pr["meta"],
        "params": int(len(pr["cv"]["theta"])),
        "ref_dist_h": pr["ref"],
        "levels": levels,
    }


def gate_verdict(records: list[Record], hyp: Record) -> Record:
    walker = [r for r in records if r["meta"]["env"] == "Walker2d-v5"]
    ratios = [r["levels"][0]["ratio"] for r in walker]
    n = sum(x >= hyp["min_ratio"] for x in ratios)
    return {
        "count": n,
        "of": len(walker),
        "ratios": ratios,
        "passes": n >= hyp["min_cells"],
    }


def reach_alpha(
    cv: Record, dirs: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # alpha and dist_H per direction, nan where dist_H never reaches the
    # target (negative curvature of H_a). solve_alpha raises for the whole
    # batch, so on failure solve the directions one by one
    try:
        return solve_alpha(cv["dist"], cv["theta"], dirs, targets, -cv["h"])
    except ValueError:
        pass
    alpha, got = np.full(len(dirs), np.nan), np.full(len(dirs), np.nan)
    for i, (d, t) in enumerate(zip(dirs, targets)):
        try:
            a, g = solve_alpha(
                cv["dist"], cv["theta"], d[None], np.array([t]), -cv["h"]
            )
        except ValueError:
            continue
        alpha[i], got[i] = a[0], g[0]
    return alpha, got


def axis_map(
    cv: Record, basis: np.ndarray, ref: float
) -> tuple[np.ndarray, list[int]]:
    # B (P x dim): x_k = 1 moves dist_H by one reference along S v_k. axes
    # that never reach it take the median alpha of the others
    dirs = (cv["scale"][:, None] * basis).T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    alpha, _ = reach_alpha(cv, dirs, np.full(len(dirs), ref))
    fallback = np.flatnonzero(np.isnan(alpha))
    if len(fallback) == len(alpha):
        raise ValueError(
            "no basis direction reaches the dist_H reference. check H_a "
            "and the reference at this checkpoint"
        )
    alpha[fallback] = np.nanmedian(alpha)
    return (alpha[:, None] * dirs).T, fallback.tolist()


def play(
    pr: Record, thetas: list[np.ndarray], seeds: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    # rows 0..n-1 = thetas, padded with actor copies to MIN_ROWS
    cv = pr["cv"]
    # pr["buffer"] (online phase): the real rows' transitions are stored
    rows = list(thetas) + [cv["theta"]] * max(0, MIN_ROWS - len(thetas))
    batch = stack_heads([as_batch(cv["to_head"](t)) for t in rows])
    buffer = pr.get("buffer")
    returns, lengths = episodes_matrix(
        pr["meta"]["env"],
        pr["nets"].embedding,
        batch,
        seeds,
        pr["meta"]["horizon"],
        store=None if buffer is None else (buffer, len(thetas)),
    )
    return returns[: len(thetas)], lengths[: len(thetas)]


def bounds_of(cfg: Record) -> gp.Bounds:
    (l_lo, l_hi), (s_lo, s_hi) = cfg["length_bounds"], cfg["std_bounds"]
    return ((float(l_lo), float(l_hi)), (float(s_lo), float(s_hi)))


def arm_data(arm: Record) -> tuple[jax.Array, jax.Array]:
    # observations plus the exact point (0, 0), y scaled by its std
    y = np.asarray(arm["y"], "f8")
    scale = float(np.std(y)) if len(y) > 1 and np.std(y) > 0 else 1.0
    x = np.vstack([np.zeros((1, arm["dim"])), *arm["x"]])
    return jnp.asarray(x, "f8"), jnp.asarray(np.r_[0.0, y / scale], "f8")


def gp_params(arm: Record, cfg: Record, fitted: bool) -> gp.Params:
    bounds = bounds_of(cfg)
    raw = gp.init_raw(cfg["init"], bounds, arm["dim"])
    if fitted:
        x, y = arm_data(arm)
        raw = gp.fit(raw, x, y, bounds, cfg["fit_steps"], cfg["lr"])
    return gp.constrain(raw, bounds)


def hyper(p: gp.Params) -> Record:
    return {
        "length": np.asarray(p["length"]).tolist(),
        "signal": float(p["signal"]),
        "noise": float(p["noise"]),
    }


def propose(arm: Record, r: int, spec: Record, rng: np.random.Generator):
    # this round's queries in x, (n, dim)
    if arm["kind"] == "es":
        n, dim = spec["es"]["pairs"], arm["dim"]
        eps = rng.normal(size=(n, dim)) / np.sqrt(dim)
        arm["eps"].append(eps)
        return np.stack([eps, -eps], axis=1).reshape(2 * n, dim)
    cfg = spec["gibo"]
    key = jax.random.PRNGKey(int(rng.integers(2**31)))
    with jax.enable_x64(True):
        p = gp_params(arm, cfg, fitted=r > 0)
        x, _ = arm_data(arm)
        q = gp.acquire(p, x, cfg["queries"], key, cfg)
        arm["hyper"].append(hyper(p))
    return np.asarray(q, "f8")


def direction(arm: Record, spec: Record) -> np.ndarray:
    # ascent direction in x: ES estimate or GP posterior mean of grad f(0)
    y = np.asarray(arm["y"], "f8")
    if arm["kind"] == "es":
        eps = np.concatenate(arm["eps"])
        return (y[0::2] - y[1::2]) @ eps / (2 * len(eps) / arm["dim"])
    with jax.enable_x64(True):
        p = gp_params(arm, spec["gibo"], fitted=True)
        x, ys = arm_data(arm)
        arm["hyper"].append(hyper(p))
        return np.asarray(gp.grad_mean(p, x, ys), "f8")


def new_arm(pr: Record, basis: np.ndarray, kind: str) -> Record:
    axes, fallback = axis_map(pr["cv"], basis, pr["ref"])
    return {
        "kind": kind,
        "basis": basis,
        "map": axes,
        "fallback_axes": fallback,
        "dim": basis.shape[1],
        "x": [],
        "y": [],
        "lengths": [],
        "eps": [],
        "hyper": [],
    }


def make_arms(pr: Record) -> dict[str, Record]:
    bases = {"G-b": pr["b"], "G-d": pr["random"][0], "ES-b": pr["b"]}
    return {
        name: new_arm(pr, basis, "es" if name.startswith("ES") else "gibo")
        for name, basis in bases.items()
    }


def run_rounds(pr: Record, arms: dict[str, Record], spec: Record) -> Record:
    cfg, theta = spec["gibo"], pr["cv"]["theta"]
    rng = cell_rng(cfg["key"], pr["meta"])
    actor: Record = {"returns": [], "lengths": []}
    for r in range(cfg["rounds"]):
        queries = {n: propose(a, r, spec, rng) for n, a in arms.items()}
        thetas = [theta]
        for n, q in queries.items():
            thetas += [theta + arms[n]["map"] @ x for x in q]
        returns, lengths = play(pr, thetas, [cfg["seed_base"] + r])
        actor["returns"].append(float(returns[0, 0]))
        actor["lengths"].append(int(lengths[0, 0]))
        i = 1
        for n, q in queries.items():
            arm, j = arms[n], i + len(q)
            arm["x"] += list(q)
            arm["y"] += (returns[i:j, 0] - returns[0, 0]).tolist()
            arm["lengths"] += lengths[i:j, 0].tolist()
            i = j
    return actor


def line_search(pr: Record, arms: dict[str, Record], spec: Record) -> Record:
    st, cv = spec["step"], pr["cv"]
    dirs, targets = [], []
    for arm in arms.values():
        arm["g_x"] = direction(arm, spec)
        d = arm["map"] @ arm["g_x"]
        arm["d"] = d / np.linalg.norm(d)
        dirs += [arm["d"]] * len(st["multiples"])
        targets += [m * pr["ref"] for m in st["multiples"]]
    alpha, got = reach_alpha(cv, np.array(dirs), np.array(targets))
    # unreachable: the whitened step |x| = sqrt(multiple) along g_x
    fallback = np.isnan(alpha)
    scales = np.repeat(
        [
            np.linalg.norm(a["map"] @ a["g_x"]) / np.linalg.norm(a["g_x"])
            for a in arms.values()
        ],
        len(st["multiples"]),
    )
    roots = np.sqrt(np.tile(st["multiples"], len(arms)))
    alpha = np.where(fallback, roots * scales, alpha)
    thetas = list(cv["theta"] + alpha[:, None] * np.array(dirs))
    got = np.where(fallback, cv["dist"](np.array(thetas)), got)
    seeds = [st["seed_base"] + e for e in range(st["episodes"])]
    returns, lengths = play(pr, [cv["theta"], *thetas], seeds)
    k = len(st["multiples"])
    for a, arm in enumerate(arms.values()):
        rows = range(a * k, (a + 1) * k)
        arm["candidates"] = [
            {
                "multiple": st["multiples"][i - a * k],
                "alpha": float(alpha[i]),
                "dist_h": float(got[i]),
                "fallback": bool(fallback[i]),
                "y": (returns[i + 1] - returns[0]).tolist(),
                "lengths": lengths[i + 1].tolist(),
            }
            for i in rows
        ]
        arm["thetas"] = [thetas[i] for i in rows]
        means = [np.mean(c["y"]) for c in arm["candidates"]]
        arm["chosen"] = int(np.argmax(means))
    return {"returns": returns[0].tolist(), "lengths": lengths[0].tolist()}


def interval(d: np.ndarray) -> tuple[float, float]:
    # two-sided 95% paired t-interval of the mean difference
    half = stats.t.ppf(0.975, len(d) - 1) * np.std(d, ddof=1) / np.sqrt(len(d))
    return float(d.mean() - half), float(d.mean() + half)


def verify_stage(
    pr: Record, arms: list[Record], seeds: list[int]
) -> list[Record]:
    cv = pr["cv"]
    chosen = [a["thetas"][a["chosen"]] for a in arms]
    returns, lengths = play(pr, [cv["theta"], *chosen], seeds)
    return [
        {
            "y": (returns[i + 1] - returns[0]).tolist(),
            "lengths": lengths[i + 1].tolist(),
            "actor_lengths": lengths[0].tolist(),
        }
        for i in range(len(arms))
    ]


def verify(pr: Record, arms: dict[str, Record], spec: Record) -> None:
    v = spec["verify"]
    first = [v["seed_base"] + e for e in range(v["first"])]
    for arm, out in zip(
        arms.values(), verify_stage(pr, list(arms.values()), first), strict=True
    ):
        lo, hi = interval(np.array(out["y"]))
        out["stage"], out["ci"] = 1, [lo, hi]
        out["accepted"] = lo > 0 if lo > 0 or hi < 0 else None
        arm["verify"] = out
    open_ = [a for a in arms.values() if a["verify"]["accepted"] is None]
    if not open_:
        return
    more = [v["seed_base"] + e for e in range(v["first"], v["max"])]
    for arm, out in zip(open_, verify_stage(pr, open_, more), strict=True):
        ver = arm["verify"]
        for k in ("y", "lengths", "actor_lengths"):
            ver[k] += out[k]
        lo, hi = interval(np.array(ver["y"]))
        ver["stage"], ver["ci"], ver["accepted"] = 2, [lo, hi], lo > 0


def cost(arm: Record, actor_rounds: Record, actor_ls: Record) -> int:
    # every env step of the decision, the actor's episodes included
    ver = arm["verify"]
    parts = [
        arm["lengths"],
        actor_rounds["lengths"],
        *[c["lengths"] for c in arm["candidates"]],
        actor_ls["lengths"],
        ver["lengths"],
        ver["actor_lengths"],
    ]
    return int(sum(np.sum(p) for p in parts))


def oracle_steps(pr: Record, axes: np.ndarray) -> tuple[list, list[bool]]:
    # ES oracle step (sigma 5%) and its projection on b, both 1x reference.
    # unreachable: the median axis length of the b map
    cv, g = pr["cv"], pr["g_u"][0]
    dirs = np.array(
        [cv["scale"] * g, cv["scale"] * (pr["b"] @ (pr["b"].T @ g))]
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    alpha, _ = reach_alpha(cv, dirs, np.full(2, pr["ref"]))
    fallback = np.isnan(alpha)
    alpha[fallback] = np.median(np.linalg.norm(axes, axis=0))
    return list(cv["theta"] + alpha[:, None] * dirs), fallback.tolist()


def evaluate(pr: Record, arms: dict[str, Record], spec: Record) -> Record:
    ev, cv = spec["evaluation"], pr["cv"]
    heads = [t for a in arms.values() for t in a["thetas"]]
    oracle, fallback = oracle_steps(pr, arms["G-b"]["map"])
    seeds = [ev["seed_base"] + e for e in range(ev["episodes"])]
    returns, lengths = play(pr, [cv["theta"], *heads, *oracle], seeds)
    gain = returns[1:].mean(1) - returns[0].mean()
    k = len(heads) // len(arms)
    for a, arm in enumerate(arms.values()):
        for c, g in zip(arm["candidates"], gain[a * k : (a + 1) * k]):
            c["true_gain"] = float(g)
    return {
        "seeds": seeds,
        "actor": {
            "returns": returns[0].tolist(),
            "lengths": lengths[0].tolist(),
        },
        "oracle": [
            {"name": n, "returns": returns[i].tolist(), "fallback": f}
            for n, i, f in (
                ("es", len(heads) + 1, fallback[0]),
                ("es_b", len(heads) + 2, fallback[1]),
            )
        ],
    }


def arm_record(pr: Record, arm: Record, actor: Record) -> Record:
    g = pr["g_u"][0]
    proj = arm["basis"] @ (arm["basis"].T @ g)
    s = pr["cv"]["scale"]
    chosen = arm["candidates"][arm["chosen"]]
    accepted = bool(arm["verify"]["accepted"])
    return {
        "x": np.asarray(arm["x"]).tolist(),
        "y": arm["y"],
        "lengths": arm["lengths"],
        "hyper": arm["hyper"],
        "fallback_axes": arm["fallback_axes"],
        "g_x": arm["g_x"].tolist(),
        "cos_es": float(arm["d"] @ (s * g) / np.linalg.norm(s * g)),
        "cos_es_proj": float(arm["d"] @ (s * proj) / np.linalg.norm(s * proj)),
        "candidates": arm["candidates"],
        "chosen": arm["chosen"],
        "verify": arm["verify"] | {"accepted": accepted},
        "decision_gain": chosen["true_gain"] if accepted else 0.0,
        "cost": cost(arm, actor["rounds"], actor["line_search"]),
    }


def run_checkpoint(
    spec: Record, path: Path, sigmas: list[float], ep: Record
) -> Record:
    t0 = time.time()
    pr = prepare(spec, path, sigmas, ep)
    arms = make_arms(pr)
    t1 = time.time()
    actor = {"rounds": run_rounds(pr, arms, spec)}
    actor["line_search"] = line_search(pr, arms, spec)
    verify(pr, arms, spec)
    t2 = time.time()
    ev = evaluate(pr, arms, spec)
    return {
        "meta": pr["meta"],
        "params": int(len(pr["cv"]["theta"])),
        "ref_dist_h": pr["ref"],
        "actor": actor,
        "arms": {n: arm_record(pr, a, actor) for n, a in arms.items()},
        "evaluation": ev,
        "seconds": {
            "prepare": t1 - t0,
            "decision": t2 - t1,
            "evaluation": time.time() - t2,
        },
    }


def cells(spec: Record, root: Path, env: str) -> list[Record]:
    source = load_spec((spec["dir"] / spec["source"]).resolve())
    ls = load_spec((spec["dir"] / spec["episodes_source"]).resolve())
    offline = source["dir"] / source["analysis"]["offline_cache"] / env
    episodes_dir = ls["dir"] / ls["analysis"]["episodes_cache"] / env
    sigmas = json.loads((offline / "sigma.json").read_text())["sigmas"]
    out = []
    for seed in spec["seeds"]:
        for step in source["offline"]["checkpoint_at"]:
            n = int(step * source["total_steps"])
            name = f"s{seed}_step_{n}.json"
            path = root / f"{env}_s{seed}" / f"step_{n}"
            for f in (path / "nets.eqx", episodes_dir / name):
                if not f.is_file():
                    raise FileNotFoundError(
                        f"{f} missing. copy the critic-degeneracy-v1 "
                        "checkpoint and run local-search-v1 episodes first"
                    )
            ep = json.loads((episodes_dir / name).read_text())
            out.append({"name": name, "path": path, "sigmas": sigmas, "ep": ep})
    return out


def load_gate(spec: Record) -> list[Record]:
    root = spec["dir"] / spec["analysis"]["gate_cache"]
    files = sorted(root.glob("*/*.json"))
    return [json.loads(f.read_text()) for f in files]


def check_gate(spec: Record) -> None:
    hyp = next(h for h in spec["hypotheses"] if h["id"] == "G0")
    v = gate_verdict(load_gate(spec), hyp)
    if v["of"] != 6:
        raise FileNotFoundError(
            f"{v['of']}/6 Walker2d gate files. run python -m offline.gibo "
            "gate <toml> <ckpt_root> Walker2d-v5 first"
        )
    override = spec.get("deviations", {}).get("gate")
    if not v["passes"] and override:
        print(f"G0 fails ({v['count']}/6), run as deviation: {override}")
    elif not v["passes"]:
        raise SystemExit(
            f"G0 fails ({v['count']}/6 cells, ratios {v['ratios']}). stop "
            "rule: no rollouts"
        )


def write_cells(spec: Record, root: Path, env: str, command: str) -> None:
    key = "gate_cache" if command == "gate" else "cells_cache"
    out = spec["dir"] / spec["analysis"][key] / env
    out.mkdir(parents=True, exist_ok=True)
    for c in cells(spec, root, env):
        dest = out / c["name"]
        if dest.is_file():
            print(f"skip {dest}: exists")
            continue
        if command == "gate":
            pr = prepare(spec, c["path"], c["sigmas"], c["ep"])
            result = gate_record(pr)
        else:
            result = run_checkpoint(spec, c["path"], c["sigmas"], c["ep"])
        dest.write_text(json.dumps(result))
        print(f"-> {dest} {result.get('seconds', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="gibo-pilot-v1")
    parser.add_argument("command", choices=["gate", "run"])
    parser.add_argument("toml", type=Path)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("env")
    args = parser.parse_args()
    spec = load_spec(args.toml)
    if args.command == "run":
        check_gate(spec)
    write_cells(spec, args.checkpoint_root, args.env, args.command)


if __name__ == "__main__":
    main()
