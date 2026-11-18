from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from algos import EvoRainbowTrainer, TD3Trainer
from config import (
    CEMConfig,
    EAToRLConfig,
    EnvConfig,
    EvalConfig,
    EvoRainbowAlgoConfig,
    RLConfig,
    RLToEAConfig,
    RunConfig,
    RunEvalConfig,
    TD3AlgoConfig,
    WandbConfig,
)
from environments import validate_env_id

_ALLOWED_TOP_KEYS = frozenset(
    {
        "protocol",
        "seed",
        "total_steps",
        "env",
        "eval_env",
        "eval",
        "wandb",
        "algorithm",
    }
)


def _build_rl_config(raw: dict[str, Any]) -> RLConfig:
    raw = dict(raw)
    dims = raw.pop("critic_hidden_dims")
    raw["critic_hidden_dims"] = tuple(int(d) for d in dims)
    return RLConfig(**raw)


def _build_algorithm(
    raw: dict[str, Any],
) -> TD3AlgoConfig | EvoRainbowAlgoConfig:
    algo = dict(raw)
    name = algo.pop("name", None)
    rl = _build_rl_config(algo.pop("rl"))
    if name == "td3":
        return TD3AlgoConfig(rl=rl, **algo)
    if name == "evorainbow":
        return EvoRainbowAlgoConfig(
            rl=rl,
            cem=CEMConfig(**algo.pop("cem")),
            rl_to_ea=RLToEAConfig(**algo.pop("rl_to_ea")),
            ea_to_rl=EAToRLConfig(**algo.pop("ea_to_rl")),
            eval=EvalConfig(**algo.pop("eval")),
            **algo,
        )
    raise ValueError(
        f"Unsupported algorithm.name={name!r}; "
        "choose from ('td3', 'evorainbow')"
    )


def _build_wandb_config(raw: dict[str, Any]) -> WandbConfig:
    wb = dict(raw)
    if "tags" in wb:
        wb["tags"] = tuple(str(t) for t in wb["tags"])
    return WandbConfig(**wb)


_PRESET_KEYS: tuple[tuple[str, str], ...] = (
    ("cem", "damp"),
    ("rl_to_ea", "period"),
    ("ea_to_rl", "tau"),
    ("eval", "p_real"),
)


def _was_overridden(key: str) -> bool:
    if not HydraConfig.initialized():
        return False
    overrides = HydraConfig.get().overrides.task
    return any(
        item.lstrip("+").split("=", 1)[0].strip() == key for item in overrides
    )


def _apply_env_preset(cfg: DictConfig) -> None:
    if cfg.algorithm.name != "evorainbow":
        return
    env_id = str(cfg.env.id)
    preset_path = (
        Path(__file__).resolve().parent
        / "conf"
        / "algorithm"
        / "env"
        / f"{env_id}.yaml"
    )
    if not preset_path.is_file():
        raise FileNotFoundError(
            f"Missing per-env preset for {env_id!r}: {preset_path}"
        )
    preset = cast("DictConfig", OmegaConf.load(preset_path))
    for section, field in _PRESET_KEYS:
        if not _was_overridden(f"algorithm.{section}.{field}"):
            cfg.algorithm[section][field] = preset[section][field]


def _clear_preset_p_real_for_full(
    raw_algo: dict[str, Any], cfg: DictConfig
) -> None:
    eval_raw = raw_algo.get("eval")
    if not isinstance(eval_raw, dict) or eval_raw.get("mode") != "full":
        return
    if _was_overridden("algorithm.eval.p_real"):
        return
    eval_raw["p_real"] = 0.0
    cfg.algorithm.eval.p_real = 0.0


def build_run_config(cfg: DictConfig) -> RunConfig:
    """Validate Hydra DictConfig and convert it into a frozen RunConfig."""
    validate_env_id(str(cfg.env.id))
    validate_env_id(str(cfg.eval_env.id))
    _apply_env_preset(cfg)
    raw = cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True))
    unknown = sorted(set(raw) - _ALLOWED_TOP_KEYS)
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {unknown}")

    env = EnvConfig(**raw["env"])
    eval_env = EnvConfig(**raw["eval_env"])
    _clear_preset_p_real_for_full(raw["algorithm"], cfg)

    return RunConfig(
        env=env,
        eval_env=eval_env,
        eval=RunEvalConfig(**raw["eval"]),
        wandb=_build_wandb_config(raw["wandb"]),
        seed=int(raw["seed"]),
        total_steps=int(raw["total_steps"]),
        protocol=int(raw["protocol"]),
        algorithm=_build_algorithm(raw["algorithm"]),
    )


def run_training(cfg: DictConfig) -> float:
    """Initialize config, wandb logging, and the selected trainer."""
    run_cfg = build_run_config(cfg)
    if isinstance(run_cfg.algorithm, EvoRainbowAlgoConfig):
        algo = run_cfg.algorithm
        print(
            f"config | env={run_cfg.env.id} | mode={algo.eval.mode} | "
            f"damp={algo.cem.damp} | ea_to_rl.tau={algo.ea_to_rl.tau} | "
            f"p_real={algo.eval.p_real} | period={algo.rl_to_ea.period}"
        )
    run = wandb.init(
        project=run_cfg.wandb.project,
        entity=run_cfg.wandb.entity,
        name=run_cfg.wandb.name,
        tags=list(run_cfg.wandb.tags),
        mode="online" if run_cfg.wandb.enabled else "disabled",
        config=cast(
            "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
        ),
    )
    wandb.define_metric("env_steps")
    wandb.define_metric("*", step_metric="env_steps")
    try:
        if isinstance(run_cfg.algorithm, TD3AlgoConfig):
            trainer: TD3Trainer | EvoRainbowTrainer = TD3Trainer(
                run_cfg, run_cfg.algorithm, on_log=wandb.log
            )
        else:
            trainer = EvoRainbowTrainer(
                run_cfg, run_cfg.algorithm, on_log=wandb.log
            )
        eval_reward = trainer.train()
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish()
    return eval_reward
