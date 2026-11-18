from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, get_args

EvalMode = Literal["full", "critic", "hboot"]
AgentName = Literal["td3", "sac"]


def _check_choice(name: str, value: str, kind: Any) -> None:
    choices = get_args(kind)
    if value not in choices:
        raise ValueError(f"{name}={value!r}; choose from {choices}")


def _check_min(name: str, value: float, minimum: float) -> None:
    if value < minimum:
        raise ValueError(f"{name}={value}; must be >= {minimum}")


def _check_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name}={value}; must be > 0")


@dataclass(frozen=True)
class RLConfig:
    """Shared RL hyperparameters common to all off-policy agents."""

    critic_hidden_dims: tuple[int, int] = (400, 300)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 128
    buffer_capacity: int = 1_000_000
    warmup_steps: int = 10_000
    train_ratio: float = 1.0

    def __post_init__(self) -> None:
        if len(self.critic_hidden_dims) != 2:
            raise ValueError(
                "rl.critic_hidden_dims must have 2 entries, got "
                f"{self.critic_hidden_dims}"
            )
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"rl.gamma={self.gamma}; must be in [0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError(f"rl.tau={self.tau}; must be in (0, 1]")
        _check_min("rl.batch_size", self.batch_size, 1)
        _check_min("rl.buffer_capacity", self.buffer_capacity, 1)
        _check_min("rl.warmup_steps", self.warmup_steps, 0)
        _check_min("rl.train_ratio", self.train_ratio, 0.0)


@dataclass(frozen=True)
class TD3RLConfig(RLConfig):
    """TD3 hyperparameters shared between standalone TD3 and EvoRainbow."""

    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    exploration_noise: float = 0.1
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_min("rl.policy_noise", self.policy_noise, 0.0)
        _check_min("rl.noise_clip", self.noise_clip, 0.0)
        _check_min("rl.policy_freq", self.policy_freq, 1)
        _check_min("rl.exploration_noise", self.exploration_noise, 0.0)
        _check_positive("rl.actor_lr", self.actor_lr)
        _check_positive("rl.critic_lr", self.critic_lr)


@dataclass(frozen=True)
class SACRLConfig(RLConfig):
    """SAC hyperparameters shared between standalone SAC and EvoRainbow."""

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_positive("rl.actor_lr", self.actor_lr)
        _check_positive("rl.critic_lr", self.critic_lr)
        _check_positive("rl.alpha_lr", self.alpha_lr)


RL_CONFIG_CLASSES: dict[str, type[RLConfig]] = {
    "td3": TD3RLConfig,
    "sac": SACRLConfig,
}


@dataclass(frozen=True)
class ResetConfig:
    """Full network resets of standalone TD3 at fixed env steps."""

    enabled: bool = False
    at_steps: tuple[int, ...] = (200_000, 400_000, 600_000)

    def __post_init__(self) -> None:
        if any(step < 1 for step in self.at_steps):
            raise ValueError(f"reset.at_steps={self.at_steps}; must be >= 1")
        if list(self.at_steps) != sorted(set(self.at_steps)):
            raise ValueError(
                f"reset.at_steps={self.at_steps}; must be strictly increasing"
            )


@dataclass(frozen=True)
class RLAlgoConfig:
    """Configuration for standalone off-policy RL agents (td3 | sac)."""

    name: AgentName
    horizon: int = 1000
    embedding_dim: int = 300
    rl: RLConfig = field(default_factory=TD3RLConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)

    def __post_init__(self) -> None:
        _check_choice("algorithm.name", self.name, AgentName)
        if self.reset.enabled and self.name != "td3":
            raise ValueError(
                "algorithm.reset.enabled=true is only supported for td3, "
                f"not {self.name!r}. set algorithm.reset.enabled=false"
            )
        if self.horizon < 1:
            raise ValueError(f"horizon={self.horizon}; must be >= 1")
        if self.embedding_dim < 1:
            raise ValueError(
                f"embedding_dim={self.embedding_dim}; must be >= 1"
            )


@dataclass(frozen=True)
class CEMConfig:
    """Cross-Entropy Method hyperparameters."""

    sigma_init: float = 1e-3
    damp: float = 1e-3
    damp_limit: float = 1e-5
    elitism: bool = False

    def __post_init__(self) -> None:
        if self.sigma_init <= 0.0:
            raise ValueError(f"cem.sigma_init={self.sigma_init}; must be > 0")


@dataclass(frozen=True)
class RLToEAConfig:
    """RL-to-EA injection schedule."""

    period: int = 1

    def __post_init__(self) -> None:
        if self.period < 0:
            raise ValueError(
                f"rl_to_ea.period={self.period}; use 0 (never) or more"
            )


@dataclass(frozen=True)
class EAToRLConfig:
    """EA-to-RL genetic soft update hyperparameters."""

    tau: float = 0.3

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError(f"ea_to_rl.tau={self.tau}; must be in [0, 1]")


@dataclass(frozen=True)
class EvalConfig:
    """Population fitness estimator configuration."""

    mode: EvalMode = "hboot"
    h: int = 200
    critic_states: int = 50_000
    p_real: float = 0.0

    def __post_init__(self) -> None:
        _check_choice("eval.mode", self.mode, EvalMode)
        if self.h < 1:
            raise ValueError(f"eval.h={self.h}; must be >= 1")
        if self.critic_states < 1:
            raise ValueError(
                f"eval.critic_states={self.critic_states}; must be >= 1"
            )
        if not 0.0 <= self.p_real <= 1.0:
            raise ValueError(f"eval.p_real={self.p_real}; must be in [0, 1]")
        if self.mode == "full" and self.p_real != 0.0:
            raise ValueError(
                f"eval.p_real={self.p_real} is only valid for eval.mode in "
                "('critic', 'hboot'), not 'full'"
            )


@dataclass(frozen=True)
class EvoRainbowAlgoConfig:
    """Configuration for EvoRainbow (RL agent + CEM + surrogate fitness)."""

    agent: AgentName = "td3"
    horizon: int = 1000
    embedding_dim: int = 300
    pop_size: int = 5
    cem: CEMConfig = field(default_factory=CEMConfig)
    rl: RLConfig = field(default_factory=TD3RLConfig)
    rl_to_ea: RLToEAConfig = field(default_factory=RLToEAConfig)
    ea_to_rl: EAToRLConfig = field(default_factory=EAToRLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self) -> None:
        _check_choice("algorithm.agent", self.agent, AgentName)
        if self.horizon < 1:
            raise ValueError(f"horizon={self.horizon}; must be >= 1")
        if self.embedding_dim < 1:
            raise ValueError(
                f"embedding_dim={self.embedding_dim}; must be >= 1"
            )
        if self.pop_size < 2:
            raise ValueError(f"pop_size={self.pop_size}; must be >= 2")
        if self.eval.mode == "hboot" and self.eval.h > self.horizon:
            raise ValueError(
                f"eval.h={self.eval.h} exceeds horizon={self.horizon}"
            )


@dataclass(frozen=True)
class EnvConfig:
    """Environment selection."""

    id: str


@dataclass(frozen=True)
class RunEvalConfig:
    """Periodic evaluation schedule."""

    episodes: int = 10
    interval: int = 5000

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError(f"eval.interval={self.interval}; must be >= 1")
        if self.episodes < 1:
            raise ValueError(f"eval.episodes={self.episodes}; must be >= 1")


@dataclass(frozen=True)
class DegConfig:
    """Critic degeneration measurement schedule and size."""

    interval: int = 50_000
    n_states: int = 2000
    n_perturbations: int = 40
    sigmas: tuple[float, ...] = (0.05, 0.2)

    def __post_init__(self) -> None:
        _check_min("deg.interval", self.interval, 1)
        _check_min("deg.n_states", self.n_states, 1)
        _check_min("deg.n_perturbations", self.n_perturbations, 1)
        if not self.sigmas:
            raise ValueError("deg.sigmas must have at least 1 entry")
        for sigma in self.sigmas:
            _check_positive("deg.sigmas entry", sigma)


@dataclass(frozen=True)
class ProbeConfig:
    """Truth probe: real paired episodes outside the step budget."""

    enabled: bool = True
    interval: int = 100_000
    n_perturbations: int = 8
    episodes: int = 3

    def __post_init__(self) -> None:
        _check_min("probe.interval", self.interval, 1)
        _check_min("probe.n_perturbations", self.n_perturbations, 1)
        _check_min("probe.episodes", self.episodes, 1)


@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases logging configuration."""

    enabled: bool = True
    project: str = "critic-uphill"
    entity: str | None = None
    name: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunConfig:
    """Top-level training configuration resolved from Hydra."""

    env: EnvConfig
    eval_env: EnvConfig
    eval: RunEvalConfig
    deg: DegConfig
    probe: ProbeConfig
    wandb: WandbConfig
    seed: int
    total_steps: int
    protocol: int
    algorithm: RLAlgoConfig | EvoRainbowAlgoConfig

    def __post_init__(self) -> None:
        if self.total_steps < 1:
            raise ValueError(f"total_steps={self.total_steps}; must be >= 1")
        candidates = self.deg.n_perturbations * len(self.deg.sigmas)
        if self.probe.enabled and self.probe.n_perturbations > candidates:
            raise ValueError(
                f"probe.n_perturbations={self.probe.n_perturbations} exceeds "
                f"the {candidates} perturbations measured by deg"
            )
        if isinstance(self.algorithm, RLAlgoConfig) and (
            self.algorithm.reset.enabled
        ):
            late = [
                s
                for s in self.algorithm.reset.at_steps
                if s >= self.total_steps
            ]
            if late:
                raise ValueError(
                    f"algorithm.reset.at_steps={late} never fire within "
                    f"total_steps={self.total_steps}. lower them or raise "
                    "total_steps"
                )
