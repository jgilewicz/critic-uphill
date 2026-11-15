from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from algos.erl import (
    CEMConfig,
    EAToRLConfig,
    ERLConfig,
    EvalConfig,
    RLConfig,
    RLToEAConfig,
    evaluate_policy,
)
from algos.erl import train as train_erl
from baselines.agents import build_agent
from baselines.envs import build_vec_env
from baselines.wandb_logging import WandbLoggingCallback


def _erl_config(cfg: DictConfig) -> ERLConfig:
    # unknown keys raise TypeError in the dataclass constructors
    algo = cast(
        "dict[str, Any]", OmegaConf.to_container(cfg.algorithm, resolve=True)
    )
    algo.pop("name")
    rl = algo.pop("rl")
    rl["critic_hidden_dims"] = tuple(rl["critic_hidden_dims"])
    return ERLConfig(
        env_name=cfg.env.id,
        seed=cfg.seed,
        step_budget=int(cfg.total_steps),
        cem=CEMConfig(**algo.pop("cem")),
        rl=RLConfig(**rl),
        rl_to_ea=RLToEAConfig(**algo.pop("rl_to_ea")),
        ea_to_rl=EAToRLConfig(**algo.pop("ea_to_rl")),
        eval=EvalConfig(**algo.pop("eval")),
        perf_env=cfg.eval_env.id,
        perf_interval=int(cfg.eval.interval),
        perf_episodes=int(cfg.eval.episodes),
        **algo,
    )


def _run_erl(cfg: DictConfig) -> float:
    policy = train_erl(_erl_config(cfg), on_generation=wandb.log)
    return evaluate_policy(
        cfg.eval_env.id,
        *policy,
        episodes=cfg.eval.episodes,
        seed=cfg.seed + 10_000,
    )


def _run_sb3(cfg: DictConfig) -> float:
    train_env = build_vec_env(cfg.env.id, cfg.n_envs, cfg.seed)
    eval_env = build_vec_env(cfg.eval_env.id, 1, cfg.seed + 10_000)
    model = build_agent(
        cfg.algorithm, train_env, seed=cfg.seed, device=cfg.device
    )
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=cfg.eval.episodes,
        eval_freq=max(cfg.eval.interval // cfg.n_envs, 1),
        deterministic=True,
        verbose=1,
    )
    try:
        model.learn(
            total_timesteps=cfg.total_steps,
            callback=CallbackList([WandbLoggingCallback(), eval_cb]),
        )
    finally:
        train_env.close()
        eval_env.close()
    return float(eval_cb.last_mean_reward)


def run_training(cfg: DictConfig) -> float:
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        tags=list(cfg.wandb.tags),
        mode="online" if cfg.wandb.enabled else "disabled",
        config=cast(
            "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
        ),
    )
    dispatch = {"erl": _run_erl}
    if cfg.algorithm.name in dispatch:
        # x-axis is interaction, not iteration: generation cost varies by algorithm
        wandb.define_metric("env_steps")
        wandb.define_metric("*", step_metric="env_steps")
    try:
        runner = dispatch.get(cfg.algorithm.name, _run_sb3)
        eval_reward = runner(cfg)
    except BaseException:
        # a bare `finally: run.finish()` logged crashes as "finished"
        run.finish(exit_code=1)
        raise
    run.finish()

    if cfg.result_file is not None:
        Path(cfg.result_file).write_text(
            json.dumps({"eval_reward": eval_reward})
        )
    return eval_reward
