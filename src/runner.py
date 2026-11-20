from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, cast

import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from algos import EvoRainbowTrainer, RLTrainer
from config import (
    RL_CONFIG_CLASSES,
    CEMConfig,
    DegConfig,
    EAToRLConfig,
    EnvConfig,
    EvalConfig,
    EvoRainbowAlgoConfig,
    ProbeConfig,
    RLAlgoConfig,
    RLConfig,
    ResetConfig,
    RLToEAConfig,
    RunConfig,
    RunEvalConfig,
    ShadowConfig,
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
        "deg",
        "probe",
        "wandb",
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


def _build_rl_config(agent_name: str, raw: dict[str, Any]) -> RLConfig:
    cfg_cls = RL_CONFIG_CLASSES.get(agent_name)
    if cfg_cls is None:
        choices = tuple(RL_CONFIG_CLASSES)
        raise ValueError(
            f"Unsupported algorithm.agent={agent_name!r}; choose from {choices}"
        )
    raw = dict(raw)
    _check_unknown_keys(cfg_cls, raw, f"{agent_name}.rl")
    dims = raw.pop("critic_hidden_dims")
    raw["critic_hidden_dims"] = tuple(int(d) for d in dims)
    return cfg_cls(**raw)


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


def _build_algorithm(
    raw: dict[str, Any],
) -> RLAlgoConfig | EvoRainbowAlgoConfig:
    algo = dict(raw)
    name = algo.pop("name", None)
    if name in RL_CONFIG_CLASSES:
        rl = _build_rl_config(str(name), _pop_section(algo, "rl", str(name)))
        reset = _build_reset_config(algo.pop("reset", {}))
        payload: dict[str, Any] = {
            "name": name,
            "rl": rl,
            "reset": reset,
            **algo,
        }
        _check_unknown_keys(RLAlgoConfig, payload, f"algorithm ({name})")
        return RLAlgoConfig(**payload)
    if name == "evorainbow":
        if "reset" in algo:
            raise ValueError(
                "algorithm.reset is only supported for standalone td3. "
                "use algorithm=td3-reset"
            )
        agent = str(algo.get("agent", "td3"))
        rl = _build_rl_config(agent, _pop_section(algo, "rl", "evorainbow"))
        er_payload: dict[str, Any] = {
            "rl": rl,
            "cem": _build_section(
                CEMConfig,
                _pop_section(algo, "cem", "evorainbow"),
                "algorithm.cem",
            ),
            "rl_to_ea": _build_section(
                RLToEAConfig,
                _pop_section(algo, "rl_to_ea", "evorainbow"),
                "algorithm.rl_to_ea",
            ),
            "ea_to_rl": _build_section(
                EAToRLConfig,
                _pop_section(algo, "ea_to_rl", "evorainbow"),
                "algorithm.ea_to_rl",
            ),
            "eval": _build_section(
                EvalConfig,
                _pop_section(algo, "eval", "evorainbow"),
                "algorithm.eval",
            ),
            "shadow": _build_section(
                ShadowConfig,
                _pop_section(algo, "shadow", "evorainbow"),
                "algorithm.shadow",
            ),
            **algo,
        }
        _check_unknown_keys(
            EvoRainbowAlgoConfig, er_payload, "algorithm (evorainbow)"
        )
        return EvoRainbowAlgoConfig(**er_payload)
    choices = (*RL_CONFIG_CLASSES, "evorainbow")
    raise ValueError(
        f"Unsupported algorithm.name={name!r}; choose from {choices}"
    )


def _build_wandb_config(raw: dict[str, Any]) -> WandbConfig:
    wb = dict(raw)
    _check_unknown_keys(WandbConfig, wb, "wandb")
    if "tags" in wb:
        wb["tags"] = tuple(str(t) for t in wb["tags"])
    return WandbConfig(**wb)


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


def _apply_agent_rl_defaults(cfg: DictConfig) -> None:
    if cfg.algorithm.name != "evorainbow":
        return
    agent = str(cfg.algorithm.get("agent", ""))
    if agent not in RL_CONFIG_CLASSES:
        choices = tuple(RL_CONFIG_CLASSES)
        raise ValueError(
            f"Unsupported algorithm.agent={agent!r}; choose from {choices}"
        )
    if agent == "td3":
        return
    agent_yaml = _CONF_ALGO_DIR / f"{agent}.yaml"
    agent_cfg = cast("DictConfig", OmegaConf.load(agent_yaml))
    agent_rl = agent_cfg.rl
    with open_dict(cfg.algorithm.rl):
        for key in list(cfg.algorithm.rl.keys()):
            if key not in agent_rl and not _was_overridden(
                f"algorithm.rl.{key}"
            ):
                del cfg.algorithm.rl[key]
        for key, value in agent_rl.items():
            if not _was_overridden(f"algorithm.rl.{key}"):
                cfg.algorithm.rl[key] = value


def _apply_env_preset(cfg: DictConfig) -> None:
    if cfg.algorithm.name != "evorainbow":
        return
    env_id = str(cfg.env.id)
    preset_path = _CONF_ALGO_DIR / "env" / f"{env_id}.yaml"
    if not preset_path.is_file():
        raise FileNotFoundError(
            f"Missing per-env preset for {env_id!r}: {preset_path}"
        )
    preset = cast("DictConfig", OmegaConf.load(preset_path))
    _validate_preset(preset, preset_path)
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
    _apply_agent_rl_defaults(cfg)
    _apply_env_preset(cfg)
    raw = cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True))
    unknown = sorted(set(raw) - _ALLOWED_TOP_KEYS)
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {unknown}")

    env = _build_section(EnvConfig, raw["env"], "env")
    eval_env = _build_section(EnvConfig, raw["eval_env"], "eval_env")
    _clear_preset_p_real_for_full(raw["algorithm"], cfg)

    return RunConfig(
        env=env,
        eval_env=eval_env,
        eval=_build_section(RunEvalConfig, raw["eval"], "eval"),
        deg=_build_deg_config(raw["deg"]),
        probe=_build_section(ProbeConfig, raw["probe"], "probe"),
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
            f"config | env={run_cfg.env.id} | agent={algo.agent} | "
            f"mode={algo.eval.mode} | damp={algo.cem.damp} | "
            f"ea_to_rl.tau={algo.ea_to_rl.tau} | p_real={algo.eval.p_real} | "
            f"period={algo.rl_to_ea.period} | "
            f"shadow={algo.shadow.enabled} | "
            f"shadow_surrogate={algo.shadow.surrogate}"
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
        if isinstance(run_cfg.algorithm, RLAlgoConfig):
            trainer: RLTrainer | EvoRainbowTrainer = RLTrainer(
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
