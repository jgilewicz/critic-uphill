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
from common.checkpoint import Checkpoint, load_checkpoint
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
    seed: int,
    steps: int,
) -> np.ndarray:
    # every env resets with the same seed: all policies face one start state
    n = env.num_envs
    # Sync/AsyncVectorEnv take a per-env seed list, the base stub says int
    states, _ = env.reset(seed=[seed] * n)  # ty: ignore[invalid-argument-type]
    returns = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    for _ in range(steps):
        actions = heads_policy(embedding, heads, jnp.asarray(states, "f4"))
        states, reward, term, trunc, _ = env.step(np.asarray(actions))
        returns += np.asarray(reward) * alive
        alive &= ~(np.asarray(term) | np.asarray(trunc))
        if not alive.any():
            break
    return returns


def returns_matrix(
    env_name: str,
    embedding: SharedStateEmbedding,
    heads: ActorHead,
    seeds: list[int],
    steps: int,
) -> np.ndarray:
    # (heads, episodes)
    n = heads.out_layer.weight.shape[0]
    env = environments.make_vec_env(env_name, n, async_=False)
    try:
        cols = [
            rollout(env, embedding, heads, seed=s, steps=steps) for s in seeds
        ]
    finally:
        env.close()
    return np.stack(cols, axis=1)


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


def evaluate_checkpoint(
    off: dict[str, Any], path: Path, sigmas: list[float]
) -> dict[str, Any]:
    ck = load_checkpoint(path)
    meta = ck.meta
    key = jax.random.fold_in(
        jax.random.fold_in(jax.random.key(off["key"]), meta["seed"]),
        meta["env_steps"],
    )
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="critic-degeneracy-v1 offline measures"
    )
    parser.add_argument("command", choices=["calibrate", "evaluate"])
    parser.add_argument("toml", type=Path)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("env")
    args = parser.parse_args()
    spec = load_spec(args.toml)
    off = spec["offline"] | {"seeds": spec["seeds"]}
    command = calibrate if args.command == "calibrate" else evaluate
    command(off, args.checkpoint_root, args.out_dir, args.env)


if __name__ == "__main__":
    main()
