import argparse
import json
import time
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

import environments
from analysis.runs import load_spec
from common.checkpoint import load_nets
from common.rollout import heads_policy
from modules.deep_modules import ActorHead, SharedStateEmbedding
from offline.critic_degeneracy import (
    PERTURB,
    Critics,
    as_batch,
    build_heads,
    checkpoint_key,
    episodes_matrix,
    stack_heads,
)

Record = dict[str, Any]
Act = Callable[[jax.Array, jax.Array], jax.Array]
Dist = Callable[[np.ndarray], np.ndarray]
DIRECTIONS = ("es", "newton")


def cell_rng(seed: int, meta: Record) -> np.random.Generator:
    env_id = zlib.crc32(meta["env"].encode()) & 0x7FFFFFFF
    return np.random.default_rng(
        [seed, env_id, meta["seed"], meta["env_steps"]]
    )


def onpolicy_states(
    st: Record,
    embedding: SharedStateEmbedding,
    actor: ActorHead,
    meta: Record,
) -> np.ndarray:
    # every alive state of `episodes` actor rollouts, then n of them
    n = st["episodes"]
    env = environments.make_vec_env(meta["env"], n, async_=False)
    heads = stack_heads([as_batch(actor)] * n)
    pool = []
    try:
        seeds = [st["seed_base"] + e for e in range(n)]
        states, _ = env.reset(seed=seeds)  # ty: ignore[invalid-argument-type]
        alive = np.ones(n, dtype=bool)
        for _ in range(meta["horizon"]):
            pool.append(np.asarray(states)[alive])
            actions = heads_policy(embedding, heads, jnp.asarray(states, "f4"))
            states, _, term, trunc, _ = env.step(np.asarray(actions))
            alive &= ~(np.asarray(term) | np.asarray(trunc))
            if not alive.any():
                break
    finally:
        env.close()
    pool = np.concatenate(pool)
    if len(pool) < st["n"]:
        raise ValueError(
            f"{len(pool)} on-policy states < {st['n']}. raise [states].episodes"
        )
    pick = cell_rng(st["key"], meta).choice(len(pool), st["n"], replace=False)
    return pool[pick].astype("f4")


def flat_head(
    head: ActorHead,
) -> tuple[np.ndarray, Act, Callable[[np.ndarray], ActorHead]]:
    # theta = head weights as one vector, act(theta, z) = action
    params, static = eqx.partition(head, eqx.is_array)
    theta, unravel = ravel_pytree(params)

    def to_head(t: jax.Array | np.ndarray) -> ActorHead:
        return eqx.combine(unravel(jnp.asarray(t, "f4")), static)

    def act(t: jax.Array, z: jax.Array) -> jax.Array:
        return to_head(t)(z)

    return np.asarray(theta, dtype=np.float64), act, to_head


def flat_heads(heads: ActorHead) -> np.ndarray:
    params = eqx.filter(heads, eqx.is_array)
    rows = jax.vmap(lambda p: ravel_pytree(p)[0])(params)
    return np.asarray(rows, dtype=np.float64)


def leaf_scale(head: ActorHead) -> np.ndarray:
    # S: std of each leaf of the actor, as perturb scales its noise
    params = eqx.filter(head, eqx.is_array)
    scaled = jax.tree.map(lambda x: jnp.full_like(x, jnp.std(x)), params)
    return np.asarray(ravel_pytree(scaled)[0], dtype=np.float64)


@eqx.filter_jit
def action_hessians(
    critics: Critics, states: jax.Array, a0: jax.Array
) -> jax.Array:
    c1, c2 = critics

    def q(s: jax.Array, a: jax.Array) -> jax.Array:
        return jnp.minimum(c1(s, a), c2(s, a))[0]

    return jax.vmap(jax.hessian(q, argnums=1))(states, a0)


def gauss_newton(
    act: Act, theta: np.ndarray, z: jax.Array, hess: jax.Array
) -> np.ndarray:
    # H_theta = mean_s J_s^T H_a,s J_s
    t = jnp.asarray(theta, "f4")
    jac = jax.vmap(jax.jacrev(act), in_axes=(None, 0))(t, z)
    hj = jnp.einsum("sab,sbp->sap", hess, jac)
    h = jnp.einsum("sap,saq->pq", jac, hj) / z.shape[0]
    return np.asarray(h, dtype=np.float64)


def dist_fn(act: Act, theta: np.ndarray, z: jax.Array, hess: jax.Array) -> Dist:
    # dist_H per row of thetas: mean_s -1/2 da^T H_a da
    def acts(thetas: jax.Array) -> jax.Array:
        return jax.vmap(lambda t: jax.vmap(lambda zi: act(t, zi))(z))(thetas)

    a0 = acts(jnp.asarray(theta, "f4")[None])[0]

    @jax.jit
    def dist(thetas: jax.Array) -> jax.Array:
        d = acts(thetas) - a0
        return -0.5 * jnp.mean(jnp.einsum("msa,sab,msb->ms", d, hess, d), 1)

    return lambda thetas: np.asarray(dist(jnp.asarray(thetas, "f4")))


def bracket(
    dist: Dist,
    theta: np.ndarray,
    dirs: np.ndarray,
    targets: np.ndarray,
    guess: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.zeros(len(dirs)), guess.copy()
    for _ in range(60):
        short = dist(theta + hi[:, None] * dirs) < targets
        if not short.any():
            return lo, hi
        lo = np.where(short, hi, lo)
        hi = np.where(short, 2 * hi, hi)
    raise ValueError(
        f"{int(short.sum())} directions never reach their dist_H target "
        "(negative curvature along them). check H_a at these states"
    )


def solve_alpha(
    dist: Dist,
    theta: np.ndarray,
    dirs: np.ndarray,
    targets: np.ndarray,
    curvature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # alpha with dist_H(theta + alpha d) = target, d of unit norm. start
    # from the quadratic model alpha^2 * 1/2 d^T (-H) d
    q = 0.5 * np.einsum("mp,pq,mq->m", dirs, curvature, dirs)
    guess = np.sqrt(targets / np.where(q > 0, q, targets))
    lo, hi = bracket(dist, theta, dirs, targets, guess)
    for _ in range(100):
        mid = (lo + hi) / 2
        got = dist(theta + mid[:, None] * dirs)
        if np.all(np.abs(got / targets - 1) <= 1e-3):
            return mid, got
        lo = np.where(got < targets, mid, lo)
        hi = np.where(got < targets, hi, mid)
    raise ValueError(
        "bisection on alpha did not converge in 100 halvings. dist_H is "
        "not monotone along some direction"
    )


def sources(st: Record, n_eps: int, meta: Record) -> list[Record]:
    # oracle = all episodes. k-draws: k columns shared by actor + 40
    rng = cell_rng(st["draw_seed"], meta)
    out = [{"source": "oracle", "draw": 0, "cols": list(range(n_eps))}]
    for k in st["ks"]:
        for r in range(st["draws"]):
            cols = sorted(rng.choice(n_eps, k, replace=False).tolist())
            out.append({"source": f"k{k}", "draw": r, "cols": cols})
    return out


def whitened(h: np.ndarray, scale: np.ndarray, damping: float) -> Record:
    # eigen-decomposition of -H_u = -S H_theta S and its lambda
    mu, vecs = np.linalg.eigh(-scale[:, None] * h * scale[None])
    lam = max(0.0, -mu[0]) + damping * mu[-1]
    return {"mu": mu, "vecs": vecs, "lambda": float(lam)}


def directions(
    g_u: np.ndarray, scale: np.ndarray, eig: Record
) -> dict[str, np.ndarray]:
    # unit vectors in theta: d_ES = S g_u, d_N = S (-H_u + lambda I)^-1 g_u
    v = eig["vecs"]
    d_n = v @ ((v.T @ g_u) / (eig["mu"] + eig["lambda"]))
    out = {"es": scale * g_u, "newton": scale * d_n}
    return {k: d / np.linalg.norm(d) for k, d in out.items()}


def level_steps(
    lv: Record, ep: Record, src: list[Record], multiples: list[float]
) -> tuple[list[Record], np.ndarray]:
    # one row per source x direction x multiple, with its unit direction
    pert = [ep["perturbations"][i] for i in lv["idx"]]
    r, n = (np.array([p[f] for p in pert]) for f in ("returns", "lengths"))
    r0, n0 = (np.array(ep["actor"][f]) for f in ("returns", "lengths"))
    steps, dirs = [], []
    for s in src:
        c = s["cols"]
        g_u = (r[:, c].mean(1) - r0[c].mean()) @ lv["eps"]
        g_u /= len(pert) * lv["sigma"]
        cost = int(n[:, c].sum() + n0[c].sum())
        for name, d in directions(g_u, lv["scale"], lv["eig"]).items():
            for m in multiples:
                steps.append(
                    {
                        "source": s["source"],
                        "draw": s["draw"],
                        "direction": name,
                        "multiple": m,
                        "cost": cost,
                    }
                )
                dirs.append(d)
    return steps, np.array(dirs)


def curvature(nets: Any, meta: Record, spec: Record) -> Record:
    # H_a and J on on-policy states, the Gauss-Newton H_theta, dist_H
    states = onpolicy_states(spec["states"], nets.embedding, nets.actor, meta)
    s = jnp.asarray(states)
    z = jax.vmap(nets.embedding)(s)
    theta, act, to_head = flat_head(nets.actor)
    a0 = jax.vmap(act, in_axes=(None, 0))(jnp.asarray(theta, "f4"), z)
    hess = action_hessians((nets.critic1, nets.critic2), s, a0)
    h = gauss_newton(act, theta, z, hess)
    scale = leaf_scale(nets.actor)
    top = np.asarray(jnp.linalg.eigvalsh(hess)[:, -1])
    return {
        "theta": theta,
        "to_head": to_head,
        "h": h,
        "scale": scale,
        "eig": whitened(h, scale, spec["steps"]["damping"]),
        "dist": dist_fn(act, theta, z, hess),
        "h_a_not_concave": float(np.mean(top > 0)),
    }


def level_record(cv: Record, lv: Record, ep: Record, cos: float) -> Record:
    truth = [np.mean(ep["perturbations"][i]["returns"]) for i in lv["idx"]]
    return {
        "level": lv["level"],
        "sigma": lv["sigma"],
        "best_row": int(lv["idx"][int(np.argmax(truth))]),
        "lambda": cv["eig"]["lambda"],
        "mu_min": float(cv["eig"]["mu"][0]),
        "mu_max": float(cv["eig"]["mu"][-1]),
        "cos_oracle_es_newton": cos,
    }


def build_steps(
    cv: Record, heads_theta: np.ndarray, ep: Record, ctx: Record
) -> tuple[list[Record], list[np.ndarray], list[Record]]:
    st, levels = ctx["spec"]["steps"], ctx["levels"][1:]
    ref = float(cv["dist"](heads_theta[1:])[levels == 0].mean())
    src = sources(st, len(ep["actor"]["returns"]), ctx["meta"])
    steps, thetas, info = [], [], []
    for level, sigma in enumerate(ctx["sigmas"]):
        idx = np.flatnonzero(levels == level)
        lv = {
            "level": level,
            "sigma": sigma,
            "idx": idx,
            "eps": (heads_theta[idx + 1] - cv["theta"]) / (sigma * cv["scale"]),
            "scale": cv["scale"],
            "eig": cv["eig"],
        }
        rows, dirs = level_steps(lv, ep, src, st["dist_multiples"])
        targets = ref * np.array([r["multiple"] for r in rows])
        alpha, got = solve_alpha(
            cv["dist"], cv["theta"], dirs, targets, -cv["h"]
        )
        for r, a, g in zip(rows, alpha, got, strict=True):
            r |= {"level": level, "alpha": float(a), "dist_h": float(g)}
        steps += rows
        thetas += list(cv["theta"] + alpha[:, None] * dirs)
        # rows 0 and len(multiples): oracle d_ES and oracle d_N
        cos = float(dirs[0] @ dirs[len(st["dist_multiples"])])
        info.append(level_record(cv, lv, ep, cos) | {"ref_dist_h": ref})
    return steps, thetas, info


def run_checkpoint(
    spec: Record, path: Path, sigmas: list[float], ep: Record
) -> Record:
    source = load_spec((spec["dir"] / spec["source"]).resolve())
    off = source["offline"]
    nets, meta = load_nets(path)
    key = jax.random.fold_in(checkpoint_key(off, meta), PERTURB)
    heads, levels = build_heads(nets.actor, sigmas, off["perturbations"], key)
    if len(ep["perturbations"]) != len(levels) - 1:
        raise ValueError(
            f"{path}: {len(ep['perturbations'])} local-search-v1 rows vs "
            f"{len(levels) - 1} regenerated perturbations"
        )
    t0 = time.time()
    cv = curvature(nets, meta, spec)
    heads_theta = flat_heads(heads)
    ctx = {"spec": spec, "meta": meta, "sigmas": sigmas, "levels": levels}
    steps, thetas, info = build_steps(cv, heads_theta, ep, ctx)
    best = [heads_theta[lv["best_row"] + 1] for lv in info]
    rows = [cv["theta"], *best, *thetas]
    batch = stack_heads([as_batch(cv["to_head"](t)) for t in rows])
    ev = spec["evaluation"]
    seeds = [ev["seed_base"] + e for e in range(ev["episodes"])]
    t1 = time.time()
    returns, lengths = episodes_matrix(
        meta["env"], nets.embedding, batch, seeds, meta["horizon"]
    )

    def episodes(i: int) -> Record:
        return {"returns": returns[i].tolist(), "lengths": lengths[i].tolist()}

    for lv, i in zip(info, range(1, 1 + len(best)), strict=True):
        lv["best"] = episodes(i)
    for s, i in zip(steps, range(1 + len(best), len(rows)), strict=True):
        s |= episodes(i)
    return {
        "meta": meta,
        "sigmas": sigmas,
        "seeds": seeds,
        "h_a_not_concave": cv["h_a_not_concave"],
        "actor": episodes(0),
        "levels": info,
        "steps": steps,
        "seconds": {"curvature": t1 - t0, "rollouts": time.time() - t1},
    }


def run(spec: Record, root: Path, env: str) -> None:
    source = load_spec((spec["dir"] / spec["source"]).resolve())
    ls = load_spec((spec["dir"] / spec["episodes_source"]).resolve())
    offline = source["dir"] / source["analysis"]["offline_cache"] / env
    episodes_dir = ls["dir"] / ls["analysis"]["episodes_cache"] / env
    out = spec["dir"] / spec["analysis"]["steps_cache"] / env
    out.mkdir(parents=True, exist_ok=True)
    sigmas = json.loads((offline / "sigma.json").read_text())["sigmas"]
    for seed in spec["seeds"]:
        for step in source["offline"]["checkpoint_at"]:
            n = int(step * source["total_steps"])
            name = f"s{seed}_step_{n}.json"
            dest = out / name
            if dest.is_file():
                print(f"skip {dest}: exists")
                continue
            path = root / f"{env}_s{seed}" / f"step_{n}"
            for f in (path / "nets.eqx", episodes_dir / name):
                if not f.is_file():
                    raise FileNotFoundError(
                        f"{f} missing. copy the critic-degeneracy-v1 "
                        "checkpoint and run local-search-v1 episodes first"
                    )
            ep = json.loads((episodes_dir / name).read_text())
            result = run_checkpoint(spec, path, sigmas, ep)
            dest.write_text(json.dumps(result))
            print(f"-> {dest} {result['seconds']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="critic-curvature-es-v1")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("toml", type=Path)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("env")
    args = parser.parse_args()
    run(load_spec(args.toml), args.checkpoint_root, args.env)


if __name__ == "__main__":
    main()
