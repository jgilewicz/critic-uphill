import argparse
import json
import time
import zlib
from pathlib import Path
from typing import Any

import gymnasium as gym
import jax
import numpy as np

import environments
from analysis.runs import load_spec
from common.checkpoint import load_nets
from modules.deep_modules import ActorHead, SharedStateEmbedding
from offline.critic_degeneracy import (
    as_batch,
    episodes_matrix,
    perturb,
    rollout,
    stack_heads,
)

PROPOSE, ACCEPT = 0, 1


def temperature0(episodes: dict[str, Any], level: int, k: int) -> float:
    # noise of delta at k paired episodes: median sd over the perturbations
    actor = np.array(episodes["actor"]["returns"])
    sds = [
        np.std(np.array(r["returns"]) - actor, ddof=1)
        for r in episodes["perturbations"]
        if r["level"] == level
    ]
    return float(np.median(sds) / np.sqrt(k))


def proposal(head: ActorHead, sigma: float, key: jax.Array) -> ActorHead:
    return jax.tree.map(lambda x: x[0], perturb(head, sigma, 1, key))


def paired(
    env: gym.vector.VectorEnv,
    embedding: SharedStateEmbedding,
    pair: tuple[ActorHead, ActorHead],
    seeds: list[int],
    steps: int,
) -> tuple[float, int]:
    # rows: current x k seeds, then candidate x the same k seeds
    k = len(seeds)
    both = stack_heads([as_batch(h) for h in pair])
    heads = jax.tree.map(lambda x: x[np.repeat([0, 1], k)], both)
    returns, lengths = rollout(
        env, embedding, heads, seeds=seeds + seeds, steps=steps
    )
    return float(np.mean(returns[k:] - returns[:k])), int(lengths.sum())


def accept(delta: float, temperature: float, key: jax.Array) -> bool:
    if delta > 0:
        return True
    if temperature <= 0:
        return False
    return bool(float(jax.random.uniform(key)) < np.exp(delta / temperature))


def run_chain(
    env: gym.vector.VectorEnv,
    embedding: SharedStateEmbedding,
    actor: ActorHead,
    variant: dict[str, Any],
    ch: dict[str, Any],
) -> tuple[dict[str, Any], list[ActorHead]]:
    k, budget = ch["k"], ch["budget"]
    base = ch["seed_base"] + variant["chain"] * ch["seed_stride"]
    head, used, i = actor, 0, 0
    trace: dict[str, list[Any]] = {"used": [], "delta": [], "accepted": []}
    snaps, snap_used = [], []
    while used < budget:
        if k * (i + 1) > ch["seed_stride"]:
            raise ValueError(
                f"chain {variant} ran out of reset seeds at iteration {i}. "
                "raise [chain].seed_stride"
            )
        ikey = jax.random.fold_in(variant["key"], i)
        cand = proposal(
            head, variant["sigma"], jax.random.fold_in(ikey, PROPOSE)
        )
        seeds = [base + k * i + j for j in range(k)]
        delta, cost = paired(env, embedding, (head, cand), seeds, ch["horizon"])
        used += cost
        temp = variant["t0"] * max(0.0, 1 - used / budget)
        if variant["schedule"] == "greedy":
            temp = 0.0
        ok = accept(delta, temp, jax.random.fold_in(ikey, ACCEPT))
        if ok:
            head = cand
        trace["used"].append(used)
        trace["delta"].append(delta)
        trace["accepted"].append(ok)
        while (
            len(snaps) < len(ch["snapshots"])
            and used >= ch["snapshots"][len(snaps)]
        ):
            snaps.append(head)
            snap_used.append(used)
        i += 1
    record = {
        **{key: v for key, v in variant.items() if key != "key"},
        "iterations": i,
        "accepts": int(sum(trace["accepted"])),
        "used": used,
        "snapshot_used": snap_used,
        "trace": trace,
    }
    return record, snaps


def chain_key(spec: dict[str, Any], meta: dict[str, Any]) -> jax.Array:
    key = jax.random.PRNGKey(spec["chain"]["key"])
    env_id = zlib.crc32(meta["env"].encode()) & 0x7FFFFFFF
    for part in (env_id, meta["seed"], meta["env_steps"]):
        key = jax.random.fold_in(key, part)
    return key


def variants(
    spec: dict[str, Any], sigmas: list[float], t0: list[float], key: jax.Array
) -> list[dict[str, Any]]:
    ch = spec["chain"]
    out = []
    for level in ch["sigma_levels"]:
        for s, schedule in enumerate(ch["schedules"]):
            for c in range(ch["chains"]):
                vkey = jax.random.fold_in(
                    jax.random.fold_in(jax.random.fold_in(key, level), s), c
                )
                out.append(
                    {
                        "level": level,
                        "sigma": sigmas[level],
                        "t0": t0[level],
                        "schedule": schedule,
                        "chain": c,
                        "key": vkey,
                    }
                )
    return out


def evaluate_snapshots(
    spec: dict[str, Any],
    embedding: SharedStateEmbedding,
    actor: ActorHead,
    snaps: list[ActorHead],
    meta: dict[str, Any],
) -> dict[str, Any]:
    # one batch for the start actor and every snapshot: same numeric path
    ev = spec["evaluation"]
    seeds = [ev["seed_base"] + e for e in range(ev["episodes"])]
    heads = stack_heads([as_batch(h) for h in [actor, *snaps]])
    returns, lengths = episodes_matrix(
        meta["env"], embedding, heads, seeds, meta["horizon"]
    )
    return {
        "seeds": seeds,
        "actor": {
            "returns": returns[0].tolist(),
            "lengths": lengths[0].tolist(),
        },
        "snapshots": [
            {"returns": r.tolist(), "lengths": n.tolist()}
            for r, n in zip(returns[1:], lengths[1:], strict=True)
        ],
    }


def run_cell(
    spec: dict[str, Any],
    path: Path,
    sigmas: list[float],
    episodes: dict[str, Any],
) -> dict[str, Any]:
    nets, meta = load_nets(path)
    ch = {**spec["chain"], "horizon": meta["horizon"]}
    t0 = [temperature0(episodes, lv, ch["k"]) for lv in range(len(sigmas))]
    env = environments.make_vec_env(meta["env"], 2 * ch["k"], async_=False)
    records, snaps, start = [], [], time.time()
    try:
        for variant in variants(spec, sigmas, t0, chain_key(spec, meta)):
            record, chain_snaps = run_chain(
                env, nets.embedding, nets.actor, variant, ch
            )
            print(
                f"  {record['schedule']} level {record['level']} chain "
                f"{record['chain']}: {record['accepts']}/"
                f"{record['iterations']} accepted"
            )
            records.append(record)
            snaps.extend(chain_snaps)
    finally:
        env.close()
    evaluation = evaluate_snapshots(
        spec, nets.embedding, nets.actor, snaps, meta
    )
    return {
        "meta": meta,
        "sigmas": sigmas,
        "t0": t0,
        "snapshots": ch["snapshots"],
        "chains": records,
        "evaluation": evaluation,
        "seconds": time.time() - start,
    }


def cell_steps(spec: dict[str, Any], env: str) -> list[int]:
    for cell in spec["cells"]:
        if cell["env"] == env:
            return cell["steps"]
    raise ValueError(f"{env} has no [[cells]] entry in {spec['dir']}")


def run(spec: dict[str, Any], root: Path, env: str) -> None:
    source = load_spec((spec["dir"] / spec["source"]).resolve())
    ls = load_spec((spec["dir"] / spec["episodes_source"]).resolve())
    offline = source["dir"] / source["analysis"]["offline_cache"] / env
    episodes_dir = ls["dir"] / ls["analysis"]["episodes_cache"] / env
    out = spec["dir"] / spec["analysis"]["chains_cache"] / env
    out.mkdir(parents=True, exist_ok=True)
    for step in cell_steps(spec, env):
        for seed in spec["seeds"]:
            name = f"s{seed}_step_{step}.json"
            dest = out / name
            if dest.is_file():
                print(f"skip {dest}: exists")
                continue
            path = root / f"{env}_s{seed}" / f"step_{step}"
            for f in (path / "nets.eqx", offline / name, episodes_dir / name):
                if not f.is_file():
                    raise FileNotFoundError(
                        f"{f} missing. copy the critic-degeneracy-v1 "
                        "checkpoint and run local-search-v1 episodes first"
                    )
            sigmas = json.loads((offline / name).read_text())["sigmas"]
            episodes = json.loads((episodes_dir / name).read_text())
            print(f"{env} s{seed} {step}")
            result = run_cell(spec, path, sigmas, episodes)
            dest.write_text(json.dumps(result))
            print(f"-> {dest} {result['seconds']:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="td3-sa-pilot SA chains")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("toml", type=Path)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("env")
    args = parser.parse_args()
    run(load_spec(args.toml), args.checkpoint_root, args.env)


if __name__ == "__main__":
    main()
