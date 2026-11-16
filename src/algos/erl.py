from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NamedTuple, get_args

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

import environments
from common.checkpoint import checkpoint_dir, save_checkpoint
from common.replay_buffer import Buffer
from common.rollout import (
    collect_parallel_episode,
    heads_policy,
    stacked_policy,
)
from common.surrogate import critic_fitness, policy_q, population_q
from common.td3 import TD3Config, TD3State, TrainFn, init_td3, make_td3_train
from common.utils import genetic_soft_update
from modules.deep_modules import ActorHead, SharedStateEmbedding
from modules.evo_module import CEM

Metrics = dict[str, float]
Policy = tuple[SharedStateEmbedding, ActorHead]
_EVAL_SEED_OFFSET = 10_000
# diagnostics draw from their own stream so they never shift training
_DIAG_KEY_OFFSET = 50_000
CHAMPION_EVAL_EPISODES = 10
DIAG_STATES = 1000

EvoTarget = Literal["head", "policy"]
MeanAnchor = Literal["none", "rl_actor"]
EAToRLMode = Literal["off", "soft", "soft_if_better"]
EvalMode = Literal["full", "prefix", "critic", "rl_only"]

# div/elite_origin codes
ORIGIN_RL = 0
ORIGIN_RL_OFFSPRING = 1
ORIGIN_POPULATION = 2


def _check_choice(name: str, value: str, kind: Any) -> None:
    choices = get_args(kind)
    if value not in choices:
        raise ValueError(f"{name}={value!r}; choose from {choices}")


@dataclass(frozen=True)
class CEMConfig:
    sigma_init: float = 1e-3
    damp: float = 1e-3
    damp_limit: float = 1e-5
    elitism: bool = True
    mean_anchor: MeanAnchor = "none"

    def __post_init__(self) -> None:
        _check_choice("cem.mean_anchor", self.mean_anchor, MeanAnchor)


@dataclass(frozen=True)
class RLConfig:
    enabled: bool = True
    critic_hidden_dims: tuple[int, int] = (400, 300)
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    exploration_noise: float = 0.1
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 256
    buffer_capacity: int = 1_000_000
    warmup_steps: int = 10_000
    train_ratio: float = 1.0  # gradient steps per env step collected


@dataclass(frozen=True)
class RLToEAConfig:
    period: int = 5  # generations between injections, 0 = never

    def __post_init__(self) -> None:
        if self.period < 0:
            raise ValueError(
                f"rl_to_ea.period={self.period}; use 0 (never) or more"
            )


@dataclass(frozen=True)
class EAToRLConfig:
    mode: EAToRLMode = "soft"
    tau: float = 0.005

    def __post_init__(self) -> None:
        _check_choice("ea_to_rl.mode", self.mode, EAToRLMode)


@dataclass(frozen=True)
class EvalConfig:
    mode: EvalMode = "full"
    prefix_h: int = 200
    p_real: float = 0.21  # critic mode: P(real rollout) per individual
    surrogate_batch: int = 1024

    def __post_init__(self) -> None:
        _check_choice("eval.mode", self.mode, EvalMode)
        if self.prefix_h < 1:
            raise ValueError(f"eval.prefix_h={self.prefix_h}; must be >= 1")
        if not 0.0 <= self.p_real <= 1.0:
            raise ValueError(f"eval.p_real={self.p_real}; must be in [0, 1]")


@dataclass(frozen=True)
class ERLConfig:
    env_name: str = "Swimmer-v5"
    seed: int = 0
    # env-step budget, not generations: generation cost varies by algorithm
    step_budget: int = 1_000_000
    horizon: int | None = None  # None: use the env's max_episode_steps
    async_env: bool = True
    embedding_dim: int = 256
    pop_size: int = 10
    evo_target: EvoTarget = "head"
    cem: CEMConfig = field(default_factory=CEMConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    rl_to_ea: RLToEAConfig = field(default_factory=RLToEAConfig)
    ea_to_rl: EAToRLConfig = field(default_factory=EAToRLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # performance evaluation, outside the step budget
    perf_env: str | None = None  # None: env_name
    perf_interval: int = 5_000
    perf_episodes: int = 5
    # TD3 nets + replay buffer saved at these fractions of step_budget
    checkpoint_dir: str | None = None
    checkpoint_at: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _check_choice("evo_target", self.evo_target, EvoTarget)
        if self.pop_size < 2:
            raise ValueError(f"pop_size={self.pop_size}; must be >= 2")
        if not self.rl.enabled:
            _check_without_rl(self)
        _check_checkpoint(self)


def _check_checkpoint(cfg: ERLConfig) -> None:
    if not cfg.checkpoint_at:
        return
    if cfg.checkpoint_dir is None or not cfg.rl.enabled:
        raise ValueError(
            f"checkpoint.at={list(cfg.checkpoint_at)} needs checkpoint.dir "
            "and rl.enabled=true (it saves the TD3 nets and replay buffer)"
        )
    bad = [f for f in cfg.checkpoint_at if not 0.0 < f <= 1.0]
    if bad:
        raise ValueError(f"checkpoint.at {bad}: fractions must be in (0, 1]")


def _check_without_rl(cfg: ERLConfig) -> None:
    uses_rl = {
        "rl_to_ea.period": cfg.rl_to_ea.period != 0,
        "ea_to_rl.mode": cfg.ea_to_rl.mode != "off",
        "cem.mean_anchor": cfg.cem.mean_anchor != "none",
        "eval.mode": cfg.eval.mode in ("critic", "rl_only"),
    }
    bad = [name for name, used in uses_rl.items() if used]
    if bad:
        raise ValueError(
            f"rl.enabled=false but {bad} need the RL actor; set "
            "rl_to_ea.period=0 ea_to_rl.mode=off cem.mean_anchor=none "
            "eval.mode=full|prefix"
        )


class Population(NamedTuple):
    flat: jax.Array
    # (pop,)-stacked when stacked, else the shared TD3 embedding
    embedding: SharedStateEmbedding
    heads: ActorHead
    stacked: bool


class Scored(NamedTuple):
    fitness: jax.Array
    pop_steps: int
    real_frac: float
    # the RL actor under the same estimator, for ea_to_rl.mode=soft_if_better
    rl_fitness: float = math.nan


@dataclass
class Run:
    cfg: ERLConfig
    action_limit: float
    horizon: int
    undiscount_scale: float
    # flat CEM vector -> ActorHead, or (embedding, actor) for evo_target=policy
    unravel: Callable[[jax.Array], Any]
    cem: CEM
    td3_state: TD3State
    train_td3: TrainFn
    buffer: Buffer | None
    key: jax.Array
    diag_key: jax.Array
    closers: list[Callable[[], None]] = field(default_factory=list)
    generation: int = 0
    env_steps: int = 0
    pending_injection: int | None = None
    # embedding the population was scored with in the current generation
    elite_embedding: SharedStateEmbedding | None = None
    origins: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    elite_origin: int = ORIGIN_POPULATION
    diag_states: jax.Array | None = None

    def close(self) -> None:
        for closer in self.closers:
            closer()


def env_dims(env_name: str, horizon: int | None) -> tuple[int, int, float, int]:
    env = environments.make_env(env_name)
    try:
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)
        obs_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.shape[0])
        action_limit = float(env.action_space.high[0])
        max_steps = env.spec.max_episode_steps if env.spec is not None else None
        resolved_horizon = (
            int(max_steps)
            if horizon is None and max_steps is not None
            else (horizon or 1000)
        )
        return obs_dim, action_dim, action_limit, resolved_horizon
    finally:
        env.close()


def _cem_space(
    cfg: ERLConfig, td3_state: TD3State, action_dim: int
) -> tuple[jax.Array, Callable[[jax.Array], Any]]:
    # (initial CEM mean, unravel)
    if cfg.evo_target == "policy":
        online = td3_state.online
        return ravel_pytree((online.embedding, online.actor))
    sample_head = ActorHead(
        cfg.embedding_dim, action_dim, key=jax.random.key(0)
    )
    flat, unravel = ravel_pytree(sample_head)
    return jnp.zeros_like(flat), unravel


def build_run(cfg: ERLConfig) -> Run:
    obs_dim, action_dim, action_limit, horizon = env_dims(
        cfg.env_name, cfg.horizon
    )
    if cfg.eval.mode == "prefix" and cfg.eval.prefix_h > horizon:
        raise ValueError(
            f"eval.prefix_h={cfg.eval.prefix_h} exceeds horizon={horizon}"
        )
    rl = cfg.rl
    undiscount_scale = (
        horizon
        if rl.gamma >= 1.0
        else (1.0 - rl.gamma**horizon) / (1.0 - rl.gamma)
    )
    key = jax.random.key(cfg.seed)
    key, td3_key, cem_key, _ = jax.random.split(key, 4)

    actor_opt = optax.adam(rl.actor_lr)
    critic_opt = optax.adam(rl.critic_lr)
    # built even with rl.enabled=false: the embedding is Z(s) for the heads
    td3_state = init_td3(
        obs_dim,
        action_dim,
        cfg.embedding_dim,
        key=td3_key,
        hidden_dims=rl.critic_hidden_dims,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
    )
    td3_cfg = TD3Config(
        gamma=rl.gamma,
        tau=rl.tau,
        policy_noise=rl.policy_noise,
        noise_clip=rl.noise_clip,
        action_limit=action_limit,
        policy_freq=rl.policy_freq,
        batch_size=rl.batch_size,
    )
    mean, unravel = _cem_space(cfg, td3_state, action_dim)
    cem = CEM(
        mean,
        cem_key,
        pop_size=cfg.pop_size,
        sigma_init=cfg.cem.sigma_init,
        damp=cfg.cem.damp,
        damp_limit=cfg.cem.damp_limit,
        elitism=cfg.cem.elitism,
    )
    buffer = (
        Buffer(rl.buffer_capacity, state_dim=obs_dim, action_dim=action_dim)
        if rl.enabled
        else None
    )
    return Run(
        cfg=cfg,
        action_limit=action_limit,
        horizon=horizon,
        undiscount_scale=undiscount_scale,
        unravel=unravel,
        cem=cem,
        td3_state=td3_state,
        train_td3=make_td3_train(td3_cfg, actor_opt, critic_opt),
        buffer=buffer,
        key=key,
        diag_key=jax.random.key(cfg.seed + _DIAG_KEY_OFFSET),
    )


def next_key(run: Run) -> jax.Array:
    run.key, k = jax.random.split(run.key)
    return k


def _buffer(run: Run) -> Buffer:
    if run.buffer is None:
        raise RuntimeError(
            "no replay buffer with rl.enabled=false; this path needs TD3"
        )
    return run.buffer


def _in_warmup(run: Run) -> bool:
    return run.buffer is not None and len(run.buffer) < run.cfg.rl.warmup_steps


def rl_target(run: Run) -> Any:
    # the RL actor in CEM's search space
    online = run.td3_state.online
    if run.cfg.evo_target == "policy":
        return (online.embedding, online.actor)
    return online.actor


def _set_rl_target(run: Run, target: Any) -> None:
    online = run.td3_state.online
    if run.cfg.evo_target == "policy":
        embedding, actor = target
        online = online._replace(embedding=embedding, actor=actor)
    else:
        online = online._replace(actor=target)
    run.td3_state = run.td3_state._replace(online=online)


def rl_flat(run: Run) -> jax.Array:
    return ravel_pytree(rl_target(run))[0]


def rl_policy(run: Run) -> Policy:
    online = run.td3_state.online
    return online.embedding, online.actor


def champion_policy(run: Run) -> Policy:
    member = run.unravel(run.cem.elite)
    if run.cfg.evo_target == "policy":
        return member
    assert run.elite_embedding is not None, "no generation scored yet"
    return run.elite_embedding, member


def ask_population(run: Run, warmup: bool) -> Population:
    cfg = run.cfg
    anchored = cfg.cem.mean_anchor == "rl_actor" and not warmup
    if anchored:
        run.cem.mu = rl_flat(run)
    flat_pop = run.cem.ask(next_key(run))
    origins = np.full(
        cfg.pop_size, ORIGIN_RL_OFFSPRING if anchored else ORIGIN_POPULATION
    )
    if cfg.cem.elitism:
        origins[-1] = run.elite_origin
    if run.pending_injection is not None:
        flat_pop = flat_pop.at[run.pending_injection].set(rl_flat(run))
        origins[run.pending_injection] = ORIGIN_RL
        run.pending_injection = None
    run.origins = origins
    members = jax.vmap(run.unravel)(flat_pop)
    if cfg.evo_target == "policy":
        embedding, heads = members
        return Population(flat_pop, embedding, heads, stacked=True)
    run.elite_embedding = run.td3_state.online.embedding
    return Population(flat_pop, run.elite_embedding, members, stacked=False)


def pop_actions(pop: Population, states: jnp.ndarray) -> jnp.ndarray:
    if pop.stacked:
        return stacked_policy(pop.embedding, pop.heads, states)
    return heads_policy(pop.embedding, pop.heads, states)


def train_td3(run: Run, gen_env_steps: int) -> Metrics:
    buffer = _buffer(run)
    num_updates = int(run.cfg.rl.train_ratio * gen_env_steps)
    run.td3_state, critic_loss, td_err = run.train_td3(
        run.td3_state,
        buffer.data,
        jnp.asarray(len(buffer)),
        jnp.asarray(num_updates),
        next_key(run),
    )
    return {
        "train/critic_loss": float(critic_loss),
        "train/td_error": float(td_err),
    }


def ea_to_rl(run: Run, fitness: jax.Array, rl_fitness: float) -> Metrics:
    mode = run.cfg.ea_to_rl.mode
    if mode == "off":
        return {}
    applied = mode == "soft" or float(jnp.max(fitness)) > rl_fitness
    if applied:
        champion = run.unravel(run.cem.elite)
        merged = genetic_soft_update(
            rl_target(run), champion, run.cfg.ea_to_rl.tau
        )
        _set_rl_target(run, merged)
    return {"gene/ea_to_rl": float(applied)}


@eqx.filter_jit
def _action_rms(a: Policy, b: Policy, states: jax.Array) -> jax.Array:
    def act(policy: Policy, state: jax.Array) -> jax.Array:
        embedding, actor = policy
        return actor(embedding(state))

    diff = jax.vmap(lambda s: act(a, s) - act(b, s))(states)
    return jnp.sqrt(jnp.mean(diff**2))


def divergence(run: Run) -> Metrics:
    # champion vs RL actor; actions live in [-1, 1] (ActorHead action_limit=1)
    if run.diag_states is None:
        buffer = _buffer(run)
        if len(buffer) < run.cfg.rl.warmup_steps:
            return {}
        run.diag_states = buffer.sample(run.diag_key, DIAG_STATES)["state"]
    champion = champion_policy(run)
    action_dist = _action_rms(champion, rl_policy(run), run.diag_states)
    param_dist = jnp.sqrt(jnp.mean((run.cem.elite - rl_flat(run)) ** 2))
    return {
        "div/action_dist": float(action_dist),
        "div/param_dist": float(param_dist),
    }


def maybe_evaluate(run: Run, steps_before: int) -> Metrics:
    cfg = run.cfg
    if steps_before // cfg.perf_interval == run.env_steps // cfg.perf_interval:
        return {}
    env_name = cfg.perf_env or cfg.env_name
    seed = cfg.seed + _EVAL_SEED_OFFSET
    has_pop = cfg.eval.mode != "rl_only"
    metrics: Metrics = {}
    if cfg.rl.enabled:
        metrics["perf/eval_rl"] = evaluate_policy(
            env_name, *rl_policy(run), episodes=cfg.perf_episodes, seed=seed
        )
    if has_pop:
        metrics["perf/eval_champion"] = evaluate_policy(
            env_name,
            *champion_policy(run),
            episodes=CHAMPION_EVAL_EPISODES,
            seed=seed,
        )
    metrics["perf/eval_best"] = max(metrics.values())
    if cfg.rl.enabled and has_pop:
        metrics |= divergence(run)
    return metrics


def maybe_checkpoint(run: Run, steps_before: int) -> None:
    cfg = run.cfg
    crossed = [
        f
        for f in cfg.checkpoint_at
        if steps_before < f * cfg.step_budget <= run.env_steps
    ]
    if not crossed or cfg.checkpoint_dir is None:
        return
    path = checkpoint_dir(
        Path(cfg.checkpoint_dir), cfg.env_name, cfg.seed, run.env_steps
    )
    meta = {
        "env": cfg.env_name,
        "seed": cfg.seed,
        "env_steps": run.env_steps,
        "fractions": crossed,
        "generation": run.generation,
        "embedding_dim": cfg.embedding_dim,
        "hidden_dims": list(cfg.rl.critic_hidden_dims),
        "action_limit": run.action_limit,
        "horizon": run.horizon,
    }
    save_checkpoint(path, run.td3_state.online, _buffer(run), meta)
    print(f"checkpoint -> {path}")


def print_generation(run: Run, metrics: Metrics, tag: str) -> None:
    def show(key: str, fmt: str = ".1f") -> str:
        return f"{metrics[key]:{fmt}}" if key in metrics else "n/a"

    print(
        f"gen {run.generation:4d} | steps {run.env_steps:7d} | "
        f"eval={show('perf/eval_rl')} | "
        f"champion={show('perf/eval_champion')} | "
        f"real_frac={show('cost/real_frac', '.2f')} | "
        f"critic_loss={show('train/critic_loss', '.4f')} | "
        f"td_err={show('train/td_error', '.3f')} | "
        f"pop_std={show('pop/std', '.4f')} | "
        f"origin={show('div/elite_origin', '.0f')} | tag={tag}"
    )


@eqx.filter_jit
def rl_policy_step(
    embedding: SharedStateEmbedding,
    actor: ActorHead,
    states: jnp.ndarray,
    act_key: jax.Array,
    action_limit: float,
    exploration_noise: float,
    warmup: bool,
) -> jnp.ndarray:
    noise_key, uniform_key = jax.random.split(act_key)
    z = embedding(states[0])
    action = actor(z)
    noise = jax.random.normal(noise_key, action.shape) * exploration_noise
    noisy_action = jnp.clip(action + noise, -action_limit, action_limit)
    uniform_action = jax.random.uniform(
        uniform_key, action.shape, minval=-action_limit, maxval=action_limit
    )
    return jnp.where(warmup, uniform_action, noisy_action)[None]


def _deterministic_action(
    embedding: SharedStateEmbedding, actor: ActorHead, state: jnp.ndarray
) -> jnp.ndarray:
    return actor(embedding(state))


def _collect_rl_episode(
    run: Run, rl_env: gym.vector.VectorEnv, warmup: bool
) -> tuple[float, float]:
    # (episode return, return over the first eval.prefix_h steps); the
    # prefix is tracked only where soft_if_better compares against it
    cfg = run.cfg
    online = run.td3_state.online

    def policy(act_key: jax.Array, states: jnp.ndarray) -> jnp.ndarray:
        return rl_policy_step(
            online.embedding,
            online.actor,
            states,
            act_key,
            run.action_limit,
            cfg.rl.exploration_noise,
            warmup,
        )

    prefix = np.full(1, math.nan)
    alive = np.ones(1, dtype=bool)

    def track_prefix(
        step: int,
        reward: jnp.ndarray,
        terminated: jnp.ndarray,
        truncated: jnp.ndarray,
        next_states: jnp.ndarray,
    ) -> None:
        if step == 0:
            prefix[0] = 0.0
        if step < cfg.eval.prefix_h:
            prefix[0] += float(reward[0]) * alive[0]
            alive[0] &= not bool((terminated | truncated)[0])

    tracks = cfg.eval.mode == "prefix" and cfg.ea_to_rl.mode == "soft_if_better"
    returns, _ = collect_parallel_episode(
        rl_env,
        next_key(run),
        policy,
        run.buffer,
        run.horizon,
        on_step=track_prefix if tracks else None,
    )
    return float(returns[0]), float(prefix[0])


def _step_rl_only(run: Run, rl_env: gym.vector.VectorEnv) -> Metrics:
    # control condition: same TD3 and budget, no population or gene flow
    warmup = _in_warmup(run)
    _collect_rl_episode(run, rl_env, warmup)
    run.env_steps += run.horizon
    metrics: Metrics = {"cost/real_frac": 0.0}
    if warmup:
        return metrics
    return metrics | train_td3(run, run.horizon)


def _score_rollout(
    run: Run, pop_env: gym.vector.VectorEnv, pop: Population, h: int
) -> Scored:
    fitness, _ = collect_parallel_episode(
        pop_env,
        next_key(run),
        lambda _, states: pop_actions(pop, states),
        run.buffer,
        h,
    )
    return Scored(fitness, run.cfg.pop_size * h, h / run.horizon)


def _real_mask(run: Run) -> jax.Array:
    pop_size, p_real = run.cfg.pop_size, run.cfg.eval.p_real
    if p_real == 0.0:
        return jnp.zeros(pop_size, dtype=bool)
    coin = jax.random.uniform(next_key(run), (pop_size,))
    return coin > 1.0 - p_real


def _score_critic(
    run: Run, pop_env: gym.vector.VectorEnv, pop: Population
) -> Scored:
    cfg = run.cfg
    buffer = _buffer(run)
    critic = run.td3_state.online.critic1
    states = buffer.sample(next_key(run), cfg.eval.surrogate_batch)["state"]
    q_key = next_key(run)
    mu = population_q(
        critic, pop.embedding, pop.heads, states, q_key, stacked=pop.stacked
    )
    real = _real_mask(run)
    n_real = int(jnp.sum(real))
    observed = jnp.zeros(cfg.pop_size)
    if n_real > 0:
        # every env steps, only the real ones count and reach the buffer
        observed, _ = collect_parallel_episode(
            pop_env,
            next_key(run),
            lambda _, s: pop_actions(pop, s),
            buffer,
            run.horizon,
            store_mask=np.asarray(real),
        )
    fitness, offset = critic_fitness(observed, real, mu, run.undiscount_scale)
    rl_fitness = math.nan
    if cfg.ea_to_rl.mode == "soft_if_better":
        rl_q = policy_q(critic, *rl_policy(run), states, q_key)
        rl_fitness = float(run.undiscount_scale * rl_q + offset)
    return Scored(
        fitness, n_real * run.horizon, n_real / cfg.pop_size, rl_fitness
    )


def _score(
    run: Run,
    pop_env: gym.vector.VectorEnv,
    pop: Population,
    rl_returns: tuple[float, float],
) -> Scored:
    # rl_returns carry exploration noise, so in full and prefix the RL
    # actor's soft_if_better fitness slightly understates it
    rl_return, rl_prefix = rl_returns
    match run.cfg.eval.mode:
        case "full":
            scored = _score_rollout(run, pop_env, pop, run.horizon)
            return scored._replace(rl_fitness=rl_return)
        case "prefix":
            scored = _score_rollout(run, pop_env, pop, run.cfg.eval.prefix_h)
            return scored._replace(rl_fitness=rl_prefix)
        case "critic":
            return _score_critic(run, pop_env, pop)
        case other:
            raise ValueError(f"eval.mode={other!r} has no population")


def _population_metrics(run: Run, scored: Scored) -> Metrics:
    run.elite_origin = int(run.origins[int(jnp.argmax(scored.fitness))])
    return {
        "cost/real_frac": scored.real_frac,
        "pop/std": float(jnp.sqrt(run.cem.cov).mean()),
        # cem.elite is the argmax of this fitness, on this condition's scale
        "pop/champion_fitness": float(jnp.max(scored.fitness)),
        "div/elite_origin": float(run.elite_origin),
    }


def _step_generation(
    run: Run,
    rl_env: gym.vector.VectorEnv | None,
    pop_env: gym.vector.VectorEnv,
) -> Metrics:
    warmup = _in_warmup(run)
    pop = ask_population(run, warmup)
    rl_steps, rl_returns = 0, (math.nan, math.nan)
    if rl_env is not None:
        # the RL actor always runs a full horizon into the replay buffer
        rl_returns = _collect_rl_episode(run, rl_env, warmup)
        rl_steps = run.horizon

    if warmup:
        # everyone runs a full rollout to seed the replay buffer
        scored = _score_rollout(run, pop_env, pop, run.horizon)
    else:
        scored = _score(run, pop_env, pop, rl_returns)
    gen_env_steps = rl_steps + scored.pop_steps
    run.env_steps += gen_env_steps
    run.cem.tell(scored.fitness, pop.flat)
    metrics = _population_metrics(run, scored)
    if warmup or rl_env is None:
        return metrics

    metrics |= train_td3(run, gen_env_steps)
    metrics |= ea_to_rl(run, scored.fitness, scored.rl_fitness)
    period = run.cfg.rl_to_ea.period
    if period > 0 and run.generation % period == 0:
        run.pending_injection = int(jnp.argmin(scored.fitness))
    return metrics


def _tag(cfg: ERLConfig) -> str:
    match cfg.eval.mode:
        case "prefix":
            return f"prefix-h{cfg.eval.prefix_h}"
        case "critic":
            return f"critic-p{cfg.eval.p_real}"
        case mode:
            return mode


def train(
    cfg: ERLConfig,
    *,
    on_generation: Callable[[Metrics], None] | None = None,
) -> Policy:
    # returns the RL actor, or the champion when rl.enabled=false
    run = build_run(cfg)
    rl_env = pop_env = None
    if cfg.rl.enabled:
        rl_env = environments.make_vec_env(
            cfg.env_name, 1, async_=cfg.async_env, to_jax=True
        )
        run.closers.append(rl_env.close)
    if cfg.eval.mode != "rl_only":
        pop_env = environments.make_vec_env(
            cfg.env_name, cfg.pop_size, async_=cfg.async_env, to_jax=True
        )
        run.closers.append(pop_env.close)
    tag = _tag(cfg)

    try:
        while run.env_steps < cfg.step_budget:
            steps_before = run.env_steps
            if pop_env is None:
                assert rl_env is not None
                metrics = _step_rl_only(run, rl_env)
            else:
                metrics = _step_generation(run, rl_env, pop_env)
            metrics |= maybe_evaluate(run, steps_before)
            maybe_checkpoint(run, steps_before)
            metrics |= {
                "generation": float(run.generation),
                "env_steps": float(run.env_steps),
            }
            print_generation(run, metrics, tag)
            if on_generation is not None:
                on_generation(metrics)
            run.generation += 1
    finally:
        run.close()

    return rl_policy(run) if cfg.rl.enabled else champion_policy(run)


def evaluate_policy(
    env_name: str,
    embedding: SharedStateEmbedding,
    actor: ActorHead,
    *,
    episodes: int,
    seed: int | None,
) -> float:
    env = environments.make_env(env_name)
    try:
        total_reward = 0.0
        for i in range(episodes):
            state, _ = env.reset(seed=seed if i == 0 else None)
            done = False
            while not done:
                action = np.asarray(
                    _deterministic_action(embedding, actor, jnp.asarray(state))
                )
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
        return total_reward / episodes
    finally:
        env.close()


if __name__ == "__main__":
    train(ERLConfig())
