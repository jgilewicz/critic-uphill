import argparse
import json
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import environments
from analysis.runs import load_spec
from common.checkpoint import Checkpoint, load_checkpoint
from common.replay_buffer import BufferData
from common.rollout import heads_policy
from common.td3 import TD3Config, TD3Networks, TD3State, make_mixed_fqe_train
from modules.deep_modules import ActorHead, Critic, SharedStateEmbedding
from offline.critic_degeneracy import (
    PERTURB,
    as_batch,
    build_heads,
    checkpoint_key,
    episodes_matrix,
    stack_heads,
)
from offline.curvature_es import (
    Act,
    cell_rng,
    curvature,
    flat_heads,
    level_steps,
    solve_alpha,
    sources,
)

Record = dict[str, Any]
# sigma levels whose landscape transitions each variant adds to the buffer
VARIANTS = {"A": (0,), "B": (1,), "C": (0, 1), "D": ()}


def landscape(
    env_name: str,
    embedding: SharedStateEmbedding,
    heads: ActorHead,
    seeds: list[int],
    steps: int,
) -> Record:
    # one episode per head, every alive transition as the TD3 buffer has it
    n = heads.out_layer.weight.shape[0]
    env = environments.make_vec_env(env_name, n, async_=False)
    cols: dict[str, list[np.ndarray]] = {
        k: [] for k in ("state", "action", "reward", "next_state", "done")
    }
    head_of: list[np.ndarray] = []
    returns, lengths = np.zeros(n), np.zeros(n, dtype=int)
    try:
        states, _ = env.reset(seed=seeds)  # ty: ignore[invalid-argument-type]
        alive = np.ones(n, dtype=bool)
        for _ in range(steps):
            actions = np.asarray(
                heads_policy(embedding, heads, jnp.asarray(states, "f4"))
            )
            nxt, reward, term, trunc, _ = env.step(actions)
            reward, term = np.asarray(reward), np.asarray(term)
            for k, v in zip(
                cols,
                (states, actions, reward[:, None], nxt, term[:, None]),
                strict=True,
            ):
                cols[k].append(np.asarray(v, "f4")[alive])
            head_of.append(np.flatnonzero(alive))
            returns += reward * alive
            lengths += alive
            alive &= ~(term | np.asarray(trunc))
            states = nxt
            if not alive.any():
                break
    finally:
        env.close()
    data = {k: np.concatenate(v) for k, v in cols.items()}
    return data | {
        "head": np.concatenate(head_of),
        "returns": returns,
        "lengths": lengths,
    }


def variant_data(
    land: Record, head_level: np.ndarray, use: tuple[int, ...]
) -> BufferData:
    keep = np.isin(head_level[land["head"]], use)
    fields = ("state", "action", "reward", "next_state", "done")
    state, action, reward, next_state, done = (
        jnp.asarray(land[f][keep]) for f in fields
    )
    return state, action, reward, next_state, done


def fine_tune(
    nets: TD3Networks,
    buffer: BufferData,
    extra: BufferData,
    spec: Record,
    key: jax.Array,
) -> tuple[dict[int, Critic], list[float]]:
    # checkpoint critics, target = online copy, snapshots of Q1
    cr = spec["critic"]
    opt = optax.adam(cr["lr"])
    critics = (nets.critic1, nets.critic2)
    state = TD3State(
        online=nets,
        target=nets,
        actor_opt_state=optax.EmptyState(),
        critic_opt_state=opt.init(eqx.filter(critics, eqx.is_array)),
    )
    cfg = TD3Config(
        gamma=cr["gamma"],
        tau=cr["tau"],
        policy_noise=cr["policy_noise"],
        noise_clip=cr["noise_clip"],
        action_limit=spec["action_limit"],
        batch_size=cr["batch_size"],
    )
    train = make_mixed_fqe_train(cfg, opt)
    chunk = cr["log_every"]
    snaps, losses = {}, []
    for i in range(max(cr["updates"]) // chunk):
        k = jax.random.fold_in(key, i)
        state, loss = train(state, buffer, extra, jnp.asarray(chunk), k)
        losses.append(float(loss))
        if (i + 1) * chunk in cr["updates"]:
            snaps[(i + 1) * chunk] = state.online.critic1
    return snaps, losses


@eqx.filter_jit
def q_gradient(
    critic: Critic, act: Act, theta: jax.Array, states: jax.Array, z: jax.Array
) -> jax.Array:
    # grad_theta mean_s Q1(s, actor_theta(s)), as the TD3 actor loss
    def mean_q(t: jax.Array) -> jax.Array:
        a = jax.vmap(lambda zi: act(t, zi))(z)
        return jnp.mean(jax.vmap(critic)(states, a))

    return jax.grad(mean_q)(theta)


def cosines(g_u: np.ndarray, levels: list[Record]) -> Record:
    # raw cos(g_crit, g_ES) and sketch cos in the span of the level's eps
    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    per = []
    for lv in levels:
        sketch = lv["eps"].T @ (lv["eps"] @ g_u) / len(lv["eps"])
        per.append(
            {"raw": cos(g_u, lv["g_es"]), "sketch": cos(sketch, lv["g_es"])}
        )
    return {
        "levels": per,
        "sketch": float(np.mean([p["sketch"] for p in per])),
        "raw": float(np.mean([p["raw"] for p in per])),
    }


def es_levels(
    cv: Record, heads_theta: np.ndarray, ep: Record, ctx: Record
) -> tuple[list[Record], list[Record], np.ndarray]:
    # per level: eps, oracle g_ES and the 1x ES steps (oracle + k = 1 draws)
    levels = ctx["levels"][1:]
    st = ctx["curv"]["steps"]
    src = [
        s
        for s in sources(st, len(ep["actor"]["returns"]), ctx["meta"])
        if s["source"] in ("oracle", "k1")
    ]
    r0 = np.mean(ep["actor"]["returns"])
    out, steps, dirs = [], [], []
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
        r = np.array([np.mean(ep["perturbations"][i]["returns"]) for i in idx])
        lv["g_es"] = (r - r0) @ lv["eps"] / (len(idx) * sigma)
        rows, d = level_steps(lv, ep, src, [ctx["spec"]["steps"]["multiple"]])
        es = [i for i, row in enumerate(rows) if row["direction"] == "es"]
        steps += [rows[i] | {"level": level} for i in es]
        dirs += [d[i] for i in es]
        out.append(lv)
    return out, steps, np.array(dirs)


def critic_rows(
    ck: Checkpoint, land: Record, cv: Record, ctx: Record
) -> list[Record]:
    # original critic, then each variant x updates: cos and step direction
    head_level = ctx["levels"][1:]
    theta = jnp.asarray(cv["theta"], "f4")

    def row(name: str, critic: Critic, extra: Record) -> Record:
        g = np.asarray(
            q_gradient(critic, cv["act"], theta, cv["states"], cv["z"])
        )
        g_u = cv["scale"] * g
        return extra | {
            "name": name,
            "cos": cosines(g_u, ctx["es"]),
            "direction": cv["scale"] * g_u,
        }

    rows = [row("T0", ck.nets.critic1, {"variant": "T0", "updates": 0})]
    key = jax.random.key(
        int(cell_rng(ctx["spec"]["critic"]["key"], ctx["meta"]).integers(2**31))
    )
    spec = ctx["spec"] | {"action_limit": ctx["meta"]["action_limit"]}
    for v, use in VARIANTS.items():
        extra = variant_data(land, head_level, use) if use else ck.data
        snaps, losses = fine_tune(ck.nets, ck.data, extra, spec, key)
        n_extra = int(extra[0].shape[0]) if use else 0
        for k, critic in snaps.items():
            info = {"variant": v, "updates": k, "losses": losses, "n": n_extra}
            rows.append(row(f"{v}{k // 1000}k", critic, info))
    return rows


def step_thetas(cv: Record, dirs: np.ndarray, ref: float) -> Record:
    unit = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    targets = np.full(len(unit), ref)
    alpha, got = solve_alpha(cv["dist"], cv["theta"], unit, targets, -cv["h"])
    return {
        "thetas": cv["theta"] + alpha[:, None] * unit,
        "alpha": alpha,
        "dist_h": got,
    }


def evaluate(
    nets: TD3Networks,
    meta: Record,
    cv: Record,
    thetas: list[np.ndarray],
    ev: Record,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    batch = stack_heads([as_batch(cv["to_head"](t)) for t in thetas])
    seeds = [ev["seed_base"] + e for e in range(ev["episodes"])]
    returns, lengths = episodes_matrix(
        meta["env"], nets.embedding, batch, seeds, meta["horizon"]
    )
    return seeds, returns, lengths


def run_checkpoint(
    spec: Record, path: Path, sigmas: list[float], ep: Record
) -> Record:
    source = load_spec((spec["dir"] / spec["source"]).resolve())
    curv = load_spec((spec["dir"] / spec["steps_source"]).resolve())
    off = source["offline"]
    ck = load_checkpoint(path)
    meta = ck.meta
    key = jax.random.fold_in(checkpoint_key(off, meta), PERTURB)
    heads, levels = build_heads(
        ck.nets.actor, sigmas, off["perturbations"], key
    )
    t0 = time.time()
    n = off["perturbations"]
    seeds = [
        spec["landscape"]["seed_base"] + i % n for i in range(len(levels) - 1)
    ]
    pert = jax.tree.map(lambda x: x[1:], heads)
    land = landscape(
        meta["env"], ck.nets.embedding, pert, seeds, meta["horizon"]
    )
    t1 = time.time()
    cv = curvature(ck.nets, meta, curv)
    heads_theta = flat_heads(heads)
    ctx: Record = {"spec": spec, "curv": curv, "meta": meta}
    ctx |= {"sigmas": sigmas, "levels": levels}
    ctx["es"], es_steps, es_dirs = es_levels(cv, heads_theta, ep, ctx)
    crit = critic_rows(ck, land, cv, ctx)
    t2 = time.time()
    ref = float(cv["dist"](heads_theta[1:])[levels[1:] == 0].mean())
    dirs = np.concatenate(
        [np.array([c.pop("direction") for c in crit]), es_dirs]
    )
    st = step_thetas(cv, dirs, ref * spec["steps"]["multiple"])
    seeds_ev, returns, lengths = evaluate(
        ck.nets, meta, cv, [cv["theta"], *st["thetas"]], spec["evaluation"]
    )
    for i, r in enumerate([*crit, *es_steps]):
        r |= {
            "alpha": float(st["alpha"][i]),
            "dist_h": float(st["dist_h"][i]),
            "returns": returns[i + 1].tolist(),
            "lengths": lengths[i + 1].tolist(),
        }
    return {
        "meta": meta,
        "sigmas": sigmas,
        "seeds": seeds_ev,
        "ref_dist_h": ref,
        "params": int(len(cv["theta"])),
        "landscape": {
            "levels": levels[1:].tolist(),
            "returns": land["returns"].tolist(),
            "lengths": land["lengths"].tolist(),
        },
        "actor": {
            "returns": returns[0].tolist(),
            "lengths": lengths[0].tolist(),
        },
        "critics": crit,
        "es_steps": es_steps,
        "seconds": {
            "landscape": t1 - t0,
            "critics": t2 - t1,
            "steps": time.time() - t2,
        },
    }


def run(spec: Record, root: Path, env: str) -> None:
    source = load_spec((spec["dir"] / spec["source"]).resolve())
    ls = load_spec((spec["dir"] / spec["episodes_source"]).resolve())
    offline = source["dir"] / source["analysis"]["offline_cache"] / env
    episodes_dir = ls["dir"] / ls["analysis"]["episodes_cache"] / env
    out = spec["dir"] / spec["analysis"]["cache"] / env
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
            for f in (path / "buffer.npz", episodes_dir / name):
                if not f.is_file():
                    raise FileNotFoundError(
                        f"{f} missing. rsync the critic-degeneracy-v1 "
                        "checkpoint buffers from PD and run local-search-v1 "
                        "episodes first"
                    )
            ep = json.loads((episodes_dir / name).read_text())
            result = run_checkpoint(spec, path, sigmas, ep)
            dest.write_text(json.dumps(result))
            print(f"-> {dest} {result['seconds']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="landscape-critic-v1")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("toml", type=Path)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("env")
    args = parser.parse_args()
    run(load_spec(args.toml), args.checkpoint_root, args.env)


if __name__ == "__main__":
    main()
