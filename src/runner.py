from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from algos import EvoRainbowTrainer, RLTrainer
from config import (
    CEMConfig,
    CheckpointConfig,
    DegConfig,
    EAToRLConfig,
    EnvConfig,
    EvalConfig,
    EvoRainbowAlgoConfig,
    MatchedConfig,
    ProbeConfig,
    RLAlgoConfig,
    ResetConfig,
    RLToEAConfig,
    RunConfig,
    RunEvalConfig,
    ShadowConfig,
    TD3RLConfig,
    WandbConfig,
    XQCConfig,
)
from environments import validate_env_id
from environments.dmc_env import DMC_ENVS
from xqc.trainer import XQCTrainer

_ALLOWED_TOP_KEYS = frozenset(
    {
        "protocol",
        "seed",
        "total_steps",
        "env",
        "eval_env",
        "eval",
        "deg",
        "probe",
        "wandb",
        "checkpoint",
        "algorithm",
    }
)

_CONF_ALGO_DIR = Path(__file__).resolve().parent / "conf" / "algorithm"


def _check_unknown_keys(
    cls: type[Any], raw: dict[str, Any], label: str
) -> None:
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {label}: {unknown}; allowed: {sorted(allowed)}"
        )


def _build_rl_config(raw: dict[str, Any]) -> TD3RLConfig:
    raw = dict(raw)
    _check_unknown_keys(TD3RLConfig, raw, "algorithm.rl")
    dims = raw.pop("critic_hidden_dims")
    raw["critic_hidden_dims"] = tuple(int(d) for d in dims)
    return TD3RLConfig(**raw)


def _pop_section(
    algo: dict[str, Any], key: str, algo_name: str
) -> dict[str, Any]:
    section = algo.pop(key, None)
    if not isinstance(section, dict):
        raise ValueError(
            f"Missing or invalid 'algorithm.{key}' section for {algo_name!r}"
        )
    return section


def _build_section(cls: type[Any], raw: dict[str, Any], label: str) -> Any:
    _check_unknown_keys(cls, raw, label)
    return cls(**raw)


def _build_reset_config(raw: dict[str, Any]) -> ResetConfig:
    reset = dict(raw)
    _check_unknown_keys(ResetConfig, reset, "algorithm.reset")
    if "at_steps" in reset:
        reset["at_steps"] = tuple(int(x) for x in reset["at_steps"])
    return ResetConfig(**reset)


def _build_td3(algo: dict[str, Any]) -> RLAlgoConfig:
    rl = _build_rl_config(_pop_section(algo, "rl", "td3"))
    reset = _build_reset_config(algo.pop("reset", {}))
    shadow = _build_section(
        ShadowConfig, algo.pop("shadow", {}), "algorithm.shadow"
    )
    payload: dict[str, Any] = {
        "rl": rl,
        "reset": reset,
        "shadow": shadow,
        **algo,
    }
    _check_unknown_keys(RLAlgoConfig, payload, "algorithm (td3)")
    return RLAlgoConfig(**payload)


def _build_evorainbow(algo: dict[str, Any]) -> EvoRainbowAlgoConfig:
    if "reset" in algo:
        raise ValueError(
            "algorithm.reset is only supported for standalone td3. "
            "use algorithm=td3-reset"
        )
    algo.pop("name")
    rl = _build_rl_config(_pop_section(algo, "rl", "evorainbow"))
    sections: dict[str, type[Any]] = {
        "cem": CEMConfig,
        "rl_to_ea": RLToEAConfig,
        "ea_to_rl": EAToRLConfig,
        "eval": EvalConfig,
        "shadow": ShadowConfig,
        "matched": MatchedConfig,
    }
    payload: dict[str, Any] = {
        key: _build_section(
            cls, _pop_section(algo, key, "evorainbow"), f"algorithm.{key}"
        )
        for key, cls in sections.items()
    }
    payload |= {"rl": rl, **algo}
    _check_unknown_keys(EvoRainbowAlgoConfig, payload, "algorithm (evorainbow)")
    return EvoRainbowAlgoConfig(**payload)


def _build_xqc(algo: dict[str, Any]) -> XQCConfig:
    _check_unknown_keys(XQCConfig, algo, "algorithm (xqc)")
    for key in ("actor_hidden_dims", "critic_hidden_dims"):
        if key in algo:
            algo[key] = tuple(int(d) for d in algo[key])
    return XQCConfig(**algo)


_ALGORITHM_BUILDERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "td3": _build_td3,
    "evorainbow": _build_evorainbow,
    "xqc": _build_xqc,
}


def _build_algorithm(
    raw: dict[str, Any],
) -> RLAlgoConfig | EvoRainbowAlgoConfig | XQCConfig:
    algo = dict(raw)
    name = algo.get("name")
    if name not in _ALGORITHM_BUILDERS:
        choices = tuple(_ALGORITHM_BUILDERS)
        raise ValueError(
            f"Unsupported algorithm.name={name!r}; choose from {choices}"
        )
    return _ALGORITHM_BUILDERS[name](algo)


def _build_wandb_config(raw: dict[str, Any]) -> WandbConfig:
    wb = dict(raw)
    _check_unknown_keys(WandbConfig, wb, "wandb")
    if "tags" in wb:
        wb["tags"] = tuple(str(t) for t in wb["tags"])
    return WandbConfig(**wb)


def _build_checkpoint_config(raw: dict[str, Any]) -> CheckpointConfig:
    checkpoint = dict(raw)
    _check_unknown_keys(CheckpointConfig, checkpoint, "checkpoint")
    if "fractions" in checkpoint:
        checkpoint["fractions"] = tuple(
            float(x) for x in checkpoint["fractions"]
        )
    return CheckpointConfig(**checkpoint)


def _build_deg_config(raw: dict[str, Any]) -> DegConfig:
    deg = dict(raw)
    _check_unknown_keys(DegConfig, deg, "deg")
    if "sigmas" in deg:
        deg["sigmas"] = tuple(float(x) for x in deg["sigmas"])
    return DegConfig(**deg)


_PRESET_KEYS: tuple[tuple[str, str], ...] = (
    ("cem", "damp"),
    ("rl_to_ea", "period"),
    ("ea_to_rl", "tau"),
    ("eval", "p_real"),
)


def _validate_preset(preset: DictConfig, preset_path: Path) -> None:
    raw = OmegaConf.to_container(preset, resolve=False)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid preset {preset_path}: expected mapping")
    allowed: dict[str, set[str]] = {}
    for section, field in _PRESET_KEYS:
        allowed.setdefault(section, set()).add(field)
    unknown_sections = sorted(set(raw) - set(allowed))
    if unknown_sections:
        raise ValueError(
            f"Unknown preset section(s) {unknown_sections} in {preset_path}; "
            f"allowed: {sorted(allowed)}"
        )
    for section, values in raw.items():
        if not isinstance(values, dict):
            raise ValueError(
                f"Invalid preset section {section!r} in {preset_path}: "
                "expected mapping"
            )
        unknown_fields = sorted(set(values) - allowed[str(section)])
        if unknown_fields:
            bad = [f"{section}.{k}" for k in unknown_fields]
            raise ValueError(
                f"Unknown preset key(s) {bad} in {preset_path}; "
                f"allowed fields: {sorted(allowed[str(section)])}"
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
    preset_path = _CONF_ALGO_DIR / "env" / f"{env_id}.yaml"
    if env_id in DMC_ENVS and not preset_path.is_file():
        # DMC tasks have no official preset and keep the evorainbow.yaml values
        return
    if not preset_path.is_file():
        raise FileNotFoundError(
            f"no evorainbow preset for env {env_id!r}: expected {preset_path}"
        )
    preset = cast("DictConfig", OmegaConf.load(preset_path))
    _validate_preset(preset, preset_path)
    for section, field in _PRESET_KEYS:
        if not _was_overridden(f"algorithm.{section}.{field}"):
            cfg.algorithm[section][field] = preset[section][field]


def _clear_preset_p_real(raw_algo: dict[str, Any], cfg: DictConfig) -> None:
    eval_raw = raw_algo.get("eval")
    if not isinstance(eval_raw, dict) or eval_raw.get("mode") not in (
        "full",
        "prefix",
    ):
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

    env = _build_section(EnvConfig, raw["env"], "env")
    eval_env = _build_section(EnvConfig, raw["eval_env"], "eval_env")
    _clear_preset_p_real(raw["algorithm"], cfg)

    return RunConfig(
        env=env,
        eval_env=eval_env,
        eval=_build_section(RunEvalConfig, raw["eval"], "eval"),
        deg=_build_deg_config(raw["deg"]),
        probe=_build_section(ProbeConfig, raw["probe"], "probe"),
        wandb=_build_wandb_config(raw["wandb"]),
        checkpoint=_build_checkpoint_config(raw["checkpoint"]),
        seed=int(raw["seed"]),
        total_steps=int(raw["total_steps"]),
        protocol=int(raw["protocol"]),
        algorithm=_build_algorithm(raw["algorithm"]),
    )


def _make_trainer(
    run_cfg: RunConfig, on_log: Callable[[dict[str, float]], None]
) -> RLTrainer | EvoRainbowTrainer | XQCTrainer:
    algo = run_cfg.algorithm
    if isinstance(algo, RLAlgoConfig):
        return RLTrainer(run_cfg, algo, on_log=on_log)
    if isinstance(algo, XQCConfig):
        return XQCTrainer(run_cfg, algo, on_log=on_log)
    return EvoRainbowTrainer(run_cfg, algo, on_log=on_log)


def run_training(cfg: DictConfig) -> float:
    """Initialize config, wandb logging, and the selected trainer."""
    run_cfg = build_run_config(cfg)
    if isinstance(run_cfg.algorithm, EvoRainbowAlgoConfig):
        algo = run_cfg.algorithm
        print(
            f"config | env={run_cfg.env.id} | "
            f"mode={algo.eval.mode} | damp={algo.cem.damp} | "
            f"ea_to_rl.tau={algo.ea_to_rl.tau} | p_real={algo.eval.p_real} | "
            f"period={algo.rl_to_ea.period} | "
            f"shadow={algo.shadow.enabled} | "
            f"matched={algo.matched.generations}/{algo.matched.rl_updates}"
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
        eval_reward = _make_trainer(run_cfg, wandb.log).train()
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.summary["perf/eval_final"] = eval_reward
    run.finish()
    return eval_reward
