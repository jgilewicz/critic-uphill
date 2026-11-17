import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

import environments
from analysis.runs import load_spec
from common.checkpoint import BUFFER, Checkpoint, load_checkpoint, load_nets
from common.replay_buffer import Buffer, Transition
from common.rollout import heads_policy
from common.td3 import TD3Config, TD3State, make_fqe_train
from modules.deep_modules import ActorHead, Critic, SharedStateEmbedding

Critics = tuple[Critic, Critic]
# independent key streams under [offline].key
PERTURB, STATES, FQE, CALIBRATE = 0, 1, 2, 3


def checkpoints(root: Path, env: str, seed: int) -> list[Path]:
    run_dir = root / f"{env}_s{seed}"
    found = sorted(
        run_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1])
    )
    if not found:
        raise FileNotFoundError(
            f"no step_* checkpoints in {run_dir}. did the td3 run finish "
            "with checkpoint.dir pointing at this root?"
        )
    return found


def perturb(head: ActorHead, sigma: float, n: int, key: jax.Array) -> ActorHead:
    # per leaf, as CEM over the head: dW ~ N(0, (sigma * std(W))^2)
    params, static = eqx.partition(head, eqx.is_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(key, len(leaves))
    noisy = [
        leaf + sigma * jnp.std(leaf) * jax.random.normal(k, (n, *leaf.shape))
        for leaf, k in zip(leaves, keys, strict=True)
    ]
    return eqx.combine(jax.tree.unflatten(treedef, noisy), static)


def stack_heads(heads: list[ActorHead]) -> ActorHead:
    params = [eqx.partition(h, eqx.is_array)[0] for h in heads]
    static = eqx.partition(heads[0], eqx.is_array)[1]
    joined = jax.tree.map(lambda *xs: jnp.concatenate(xs), *params)
    return eqx.combine(joined, static)


def as_batch(head: ActorHead) -> ActorHead:
    return jax.tree.map(lambda x: x[None], head)


def rollout(
    env: gym.vector.VectorEnv,
    embedding: SharedStateEmbedding,
    heads: ActorHead,
    *,
    seeds: list[int],
    steps: int,
    store: tuple[Buffer, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # one reset seed per env. store = (buffer, k): alive transitions of the
    # first k envs go to the buffer
    n = env.num_envs
    # Sync/AsyncVectorEnv take a per-env seed list, the base stub says int
    states, _ = env.reset(seed=seeds)  # ty: ignore[invalid-argument-type]
    returns = np.zeros(n)
    lengths = np.zeros(n, dtype=int)
    alive = np.ones(n, dtype=bool)
    for _ in range(steps):
        actions = heads_policy(embedding, heads, jnp.asarray(states, "f4"))
        prev = states
        states, reward, term, trunc, _ = env.step(np.asarray(actions))
        if store is not None:
            add_alive(store, alive, (prev, actions, reward, states, term))
        returns += np.asarray(reward) * alive
        lengths += alive
        alive &= ~(np.asarray(term) | np.asarray(trunc))
        if not alive.any():
            break
    return returns, lengths


def add_alive(
    store: tuple[Buffer, int], alive: np.ndarray, step: tuple[Any, ...]
) -> None:
    buffer, k = store
    idx = np.flatnonzero(alive[:k])
    if len(idx) == 0:
        return
    state, action, reward, next_state, term = (
        jnp.asarray(np.asarray(x)[idx]) for x in step
    )
    buffer.add(Transition(state, action, reward, next_state, term))


def episodes_matrix(
    env_name: str,
    embedding: SharedStateEmbedding,
    heads: ActorHead,
    seeds: list[int],
    steps: int,
    store: tuple[Buffer, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # returns and lengths, each (heads, episodes)
    n = heads.out_layer.weight.shape[0]
    env = environments.make_vec_env(env_name, n, async_=False)
    try:
        cols = [
            # every env resets with the same seed: all policies face one start
            rollout(
                env, embedding, heads, seeds=[s] * n, steps=steps, store=store
            )
            for s in seeds
        ]
    finally:
        env.close()
    returns, lengths = zip(*cols, strict=True)
    return np.stack(returns, axis=1), np.stack(lengths, axis=1)


def returns_matrix(
    env_name: str,
    embedding: SharedStateEmbedding,
    heads: ActorHead,
    seeds: list[int],
    steps: int,
) -> np.ndarray:
    return episodes_matrix(env_name, embedding, heads, seeds, steps)[0]


def interpolate_sigma(
    grid: list[float], drops: list[float], target: float
) -> float:
    # first crossing, linear in log sigma
    for i, d in enumerate(drops):
        if d < target:
            continue
        if i == 0:
            raise ValueError(
                f"drop {d:.3f} at sigma {grid[0]} already >= {target}. "
                "extend sigma_grid downward"
            )
        w = (target - drops[i - 1]) / (d - drops[i - 1])
        lo, hi = math.log(grid[i - 1]), math.log(grid[i])
        return math.exp(lo + w * (hi - lo))
    raise ValueError(
        f"max drop {max(drops):.3f} < {target}. extend sigma_grid upward"
    )


def calibrate(off: dict[str, Any], root: Path, out: Path, env: str) -> None:
    dest = out / env / "sigma.json"
    if dest.is_file():
        # sigma is fixed once per env: a rerun must not move it under
        # checkpoints already evaluated
        print(f"skip {dest}: exists")
        return
    path = checkpoints(root, env, off["calibration_seed"])[0]
    ck = load_checkpoint(path)
    emb, actor = ck.nets.embedding, ck.nets.actor
    key = jax.random.fold_in(jax.random.key(off["key"]), CALIBRATE)
    grid = off["sigma_grid"]
    n = off["calibration_perturbations"]
    heads = stack_heads(
        [as_batch(actor)]
        + [
            perturb(actor, s, n, jax.random.fold_in(key, i))
            for i, s in enumerate(grid)
        ]
    )
    seeds = [
        off["calibration_seed_base"] + e
        for e in range(off["calibration_episodes"])
    ]
    t0 = time.time()
    mean = returns_matrix(env, emb, heads, seeds, off["horizon"]).mean(axis=1)
    r_actor = float(mean[0])
    if r_actor <= 0:
        raise ValueError(
            f"{path}: actor return {r_actor:.1f} <= 0, relative drop is "
            "undefined. calibrate on a later checkpoint"
        )
    per_sigma = mean[1:].reshape(len(grid), n).mean(axis=1)
    drops = [float(1 - r / r_actor) for r in per_sigma]
    result = {
        "checkpoint": str(path),
        "actor_return": r_actor,
        "sigma_grid": grid,
        "drops": drops,
        "target_drops": off["target_drops"],
        "sigmas": [
            interpolate_sigma(grid, drops, t) for t in off["target_drops"]
        ],
        "seconds": time.time() - t0,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def recent_states(
    ck: Checkpoint, n: int, window: int, key: jax.Array
) -> jax.Array:
    # n distinct states among the newest `window` transitions
    size = ck.data[0].shape[0]
    window = min(window, size)
    if n > window:
        raise ValueError(
            f"{n} states requested from a {window}-transition window "
            f"(buffer size {size}). lower [offline].states"
        )
    back = jax.random.choice(key, window, (n,), replace=False) + 1
    return ck.data[0][(ck.ptr - back) % size]


def train_fqe(
    off: dict[str, Any], ck: Checkpoint, key: jax.Array
) -> tuple[Critics, list[float]]:
    fqe, meta = off["fqe"], ck.meta
    k1, k2, train_key = jax.random.split(key, 3)
    obs, act = meta["obs_dim"], meta["action_dim"]
    hidden = tuple(meta["hidden_dims"])
    c1 = Critic(obs, act, key=k1, hidden_dims=hidden)
    c2 = Critic(obs, act, key=k2, hidden_dims=hidden)
    nets = ck.nets._replace(critic1=c1, critic2=c2)
    opt = optax.adam(fqe["lr"])
    state = TD3State(
        online=nets,
        target=nets,
        actor_opt_state=optax.EmptyState(),
        critic_opt_state=opt.init(eqx.filter((c1, c2), eqx.is_array)),
    )
    cfg = TD3Config(
        gamma=fqe["gamma"],
        tau=fqe["tau"],
        policy_noise=fqe["policy_noise"],
        noise_clip=fqe["noise_clip"],
        action_limit=meta["action_limit"],
        batch_size=fqe["batch_size"],
    )
    train = make_fqe_train(cfg, opt)
    size = jnp.asarray(ck.data[0].shape[0])
    chunk = fqe["log_every"]
    losses = []
    for i in range(fqe["steps"] // chunk):
        k = jax.random.fold_in(train_key, i)
        state, loss = train(state, ck.data, size, jnp.asarray(chunk), k)
        losses.append(float(loss))
    return (state.online.critic1, state.online.critic2), losses


@eqx.filter_jit
def critic_terms(
    critics: Critics, states: jax.Array, a0: jax.Array, actions: jax.Array
) -> dict[str, jax.Array]:
    # actions (heads, states, A), a0 = actor actions (states, A)
    c1, c2 = critics

    def q(s: jax.Array, a: jax.Array) -> jax.Array:
        return jnp.minimum(c1(s, a), c2(s, a))[0]

    # derivatives of min(Q1, Q2) at a0 = those of the head that is the min
    g = jax.vmap(jax.grad(q, argnums=1))(states, a0)
    hess = jax.vmap(jax.hessian(q, argnums=1))(states, a0)

    def per_head(acts: jax.Array) -> jax.Array:
        d = acts - a0
        score = jnp.mean(jax.vmap(q)(states, acts))
        t1 = jnp.mean(jnp.einsum("sa,sa->s", d, g))
        t2 = 0.5 * jnp.mean(jnp.einsum("sa,sab,sb->s", d, hess, d))
        return jnp.stack([score, t1, t2])

    out = jax.lax.map(per_head, actions)
    return {"score": out[:, 0], "t1": out[:, 1], "t2": out[:, 2]}


@eqx.filter_jit
def head_actions(
    embedding: SharedStateEmbedding, heads: ActorHead, states: jax.Array
) -> jax.Array:
    z = jax.vmap(embedding)(states)
    return jax.vmap(lambda h: jax.vmap(h)(z))(heads)


def build_heads(
    actor: ActorHead, sigmas: list[float], n: int, key: jax.Array
) -> tuple[ActorHead, np.ndarray]:
    # row 0 = the actor, then n perturbations per sigma level
    pert = [
        perturb(actor, s, n, jax.random.fold_in(key, i))
        for i, s in enumerate(sigmas)
    ]
    levels = np.concatenate([[-1], np.repeat(np.arange(len(sigmas)), n)])
    return stack_heads([as_batch(actor), *pert]), levels


def critic_measures(
    off: dict[str, Any], ck: Checkpoint, heads: ActorHead, key: jax.Array
) -> tuple[dict[str, np.ndarray], list[float]]:
    states = recent_states(
        ck, off["states"], off["state_window"], jax.random.fold_in(key, STATES)
    )
    acts = head_actions(ck.nets.embedding, heads, states)
    a0 = acts[0]
    td3 = critic_terms((ck.nets.critic1, ck.nets.critic2), states, a0, acts)
    fqe_critics, losses = train_fqe(off, ck, jax.random.fold_in(key, FQE))
    fqe = critic_terms(fqe_critics, states, a0, acts)
    cols = {f"{k}_td3": v for k, v in td3.items()}
    cols |= {f"{k}_fqe": v for k, v in fqe.items()}
    cols["dist_l2"] = jnp.mean(jnp.sum((acts - a0) ** 2, axis=-1), axis=1)
    return {k: np.asarray(v) for k, v in cols.items()}, losses


def checkpoint_key(off: dict[str, Any], meta: dict[str, Any]) -> jax.Array:
    return jax.random.fold_in(
        jax.random.fold_in(jax.random.key(off["key"]), meta["seed"]),
        meta["env_steps"],
    )


def evaluate_checkpoint(
    off: dict[str, Any], path: Path, sigmas: list[float]
) -> dict[str, Any]:
    ck = load_checkpoint(path)
    meta = ck.meta
    key = checkpoint_key(off, meta)
    heads, levels = build_heads(
        ck.nets.actor,
        sigmas,
        off["perturbations"],
        jax.random.fold_in(key, PERTURB),
    )
    t0 = time.time()
    base = off["eval_seed"]
    n_eps = off["truth_episodes"]
    truth = returns_matrix(
        meta["env"],
        ck.nets.embedding,
        heads,
        [base + e for e in range(n_eps)],
        off["horizon"],
    )
    prefix = returns_matrix(
        meta["env"], ck.nets.embedding, heads, [base + n_eps], off["prefix_h"]
    )[:, 0]
    t1 = time.time()
    cols, losses = critic_measures(off, ck, heads, key)
    rows = [
        {
            "level": int(lv),
            "truth": truth[i].tolist(),
            "prefix": float(prefix[i]),
        }
        | {k: float(v[i]) for k, v in cols.items()}
        for i, lv in enumerate(levels)
    ]
    return {
        "meta": meta,
        "sigmas": sigmas,
        "actor": rows[0],
        "perturbations": rows[1:],
        "fqe_loss": losses,
        "seconds": {"rollouts": t1 - t0, "critics": time.time() - t1},
    }


def evaluate(off: dict[str, Any], root: Path, out: Path, env: str) -> None:
    sigma_file = out / env / "sigma.json"
    if not sigma_file.is_file():
        raise FileNotFoundError(f"{sigma_file} missing. run calibrate first")
    sigmas = json.loads(sigma_file.read_text())["sigmas"]
    for seed in off["seeds"]:
        for path in checkpoints(root, env, seed):
            dest = out / env / f"s{seed}_{path.name}.json"
            if dest.is_file():
                print(f"skip {dest}: exists")
                continue
            result = evaluate_checkpoint(off, path, sigmas)
            dest.write_text(json.dumps(result))
            print(f"-> {dest} {result['seconds']}")


def candidates(cell: dict[str, Any], top: int) -> dict[int, list[str]]:
    # perturbation row -> why it is re-evaluated, per sigma level
    rows = cell["perturbations"]
    actor = float(np.mean(cell["actor"]["truth"]))
    truth = [float(np.mean(r["truth"])) for r in rows]
    reasons: dict[int, list[str]] = {}
    for level in range(len(cell["sigmas"])):
        idx = [i for i, r in enumerate(rows) if r["level"] == level]
        picks = {
            "better": [i for i in idx if truth[i] > actor],
            "top_truth": sorted(idx, key=lambda i: -truth[i])[:1],
            "top_td3": sorted(idx, key=lambda i: -rows[i]["score_td3"])[:top],
            "top_prefix": sorted(idx, key=lambda i: -rows[i]["prefix"])[:top],
        }
        for why, ids in picks.items():
            for i in ids:
                reasons.setdefault(i, []).append(why)
    return reasons


def check_reproduction(
    off: dict[str, Any], path: Path, heads: ActorHead, cell: dict[str, Any]
) -> None:
    # same states as evaluate, so dist_L2 pins the regenerated perturbations.
    # rtol 5e-2: GPU (TF32) vs CPU matmuls leave <=2.2e-2, a wrong key >=0.28
    ck = load_checkpoint(path)
    key = jax.random.fold_in(checkpoint_key(off, ck.meta), STATES)
    states = recent_states(ck, off["states"], off["state_window"], key)
    acts = head_actions(ck.nets.embedding, heads, states)
    dist = np.asarray(jnp.mean(jnp.sum((acts - acts[0]) ** 2, -1), 1))[1:]
    stored = np.array([r["dist_l2"] for r in cell["perturbations"]])
    if not np.allclose(dist, stored, rtol=5e-2, atol=1e-8):
        worst = float(np.max(np.abs(dist - stored) / np.abs(stored)))
        raise ValueError(
            f"{path}: regenerated perturbations differ from the evaluated "
            f"ones (max rel. dist_L2 error {worst:.2e}). check [offline].key "
            "and sigma.json before validating"
        )


def validate_checkpoint(
    off: dict[str, Any], val: dict[str, Any], path: Path, cell: dict[str, Any]
) -> dict[str, Any]:
    nets, meta = load_nets(path)
    key = jax.random.fold_in(checkpoint_key(off, meta), PERTURB)
    heads, _ = build_heads(
        nets.actor, cell["sigmas"], off["perturbations"], key
    )
    reproduced = (path / BUFFER).is_file()
    if reproduced:
        check_reproduction(off, path, heads, cell)
    reasons = candidates(cell, val["top"])
    rows = sorted(reasons)
    pick = jnp.asarray([0] + [i + 1 for i in rows])
    subset = jax.tree.map(lambda x: x[pick], heads)
    seeds = [val["seed_base"] + e for e in range(val["episodes"])]
    t0 = time.time()
    returns = returns_matrix(
        meta["env"], nets.embedding, subset, seeds, off["horizon"]
    )
    return {
        "meta": meta,
        "sigmas": cell["sigmas"],
        "seeds": seeds,
        "reproduction_checked": reproduced,
        "actor": returns[0].tolist(),
        "candidates": [
            {
                "row": i,
                "level": cell["perturbations"][i]["level"],
                "reasons": reasons[i],
                "returns": returns[k + 1].tolist(),
            }
            for k, i in enumerate(rows)
        ],
        "seconds": time.time() - t0,
    }


def validate(
    off: dict[str, Any], val: dict[str, Any], root: Path, out: Path, env: str
) -> None:
    for seed in off["seeds"]:
        for path in checkpoints(root, env, seed):
            source = out / env / f"s{seed}_{path.name}.json"
            dest = out / env / f"validate_s{seed}_{path.name}.json"
            if dest.is_file():
                print(f"skip {dest}: exists")
                continue
            if not source.is_file():
                raise FileNotFoundError(
                    f"{source} missing. validate re-evaluates the "
                    "candidates of an evaluated checkpoint, run evaluate first"
                )
            cell = json.loads(source.read_text())
            result = validate_checkpoint(off, val, path, cell)
            dest.write_text(json.dumps(result))
            n = len(result["candidates"])
            print(f"-> {dest} {n} candidates {result['seconds']:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="critic-degeneracy-v1 offline measures"
    )
    parser.add_argument(
        "command", choices=["calibrate", "evaluate", "validate"]
    )
    parser.add_argument("toml", type=Path)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("env")
    args = parser.parse_args()
    spec = load_spec(args.toml)
    off = spec["offline"] | {"seeds": spec["seeds"]}
    paths = (args.checkpoint_root, args.out_dir, args.env)
    if args.command == "validate":
        validate(off, spec["validation"], *paths)
    elif args.command == "calibrate":
        calibrate(off, *paths)
    else:
        evaluate(off, *paths)


if __name__ == "__main__":
    main()
