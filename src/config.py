from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, get_args

EvalMode = Literal["full", "critic", "hboot", "prefix"]
AgentName = Literal["td3"]
MeasurementName = Literal["signals", "extra", "pairs"]
# top-1 by Q1, Q2 and shadow, plus the nearest head (min dist_H)
SHADOW_PROBE_PICKS = 4


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
class TD3RLConfig:
    """TD3 hyperparameters shared between standalone TD3 and EvoRainbow."""

    critic_hidden_dims: tuple[int, int] = (400, 300)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 128
    buffer_capacity: int = 1_000_000
    warmup_steps: int = 10_000
    train_ratio: float = 1.0
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    exploration_noise: float = 0.1
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3

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
        _check_min("rl.policy_noise", self.policy_noise, 0.0)
        _check_min("rl.noise_clip", self.noise_clip, 0.0)
        _check_min("rl.policy_freq", self.policy_freq, 1)
        _check_min("rl.exploration_noise", self.exploration_noise, 0.0)
        _check_positive("rl.actor_lr", self.actor_lr)
        _check_positive("rl.critic_lr", self.critic_lr)


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
class ShadowConfig:
    """Shadow twin critic: TD-trained, never in the actor loss."""

    enabled: bool = False


@dataclass(frozen=True)
class RLAlgoConfig:
    """Configuration for standalone TD3."""

    name: AgentName
    horizon: int = 1000
    embedding_dim: int = 300
    rl: TD3RLConfig = field(default_factory=TD3RLConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)

    def __post_init__(self) -> None:
        _check_choice("algorithm.name", self.name, AgentName)
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
        if self.mode in ("full", "prefix") and self.p_real != 0.0:
            raise ValueError(
                f"eval.p_real={self.p_real} is only valid for eval.mode in "
                f"('critic', 'hboot'), not {self.mode!r}"
            )


@dataclass(frozen=True)
class MatchedConfig:
    """Stop at a fixed generation count with a fixed total of RL updates.

    0/0 is off (run to the step budget). Generations include warmup.
    """

    generations: int = 0
    rl_updates: int = 0

    def __post_init__(self) -> None:
        _check_min("matched.generations", self.generations, 0)
        _check_min("matched.rl_updates", self.rl_updates, 0)
        if (self.generations == 0) != (self.rl_updates == 0):
            raise ValueError(
                f"matched.generations={self.generations} and "
                f"matched.rl_updates={self.rl_updates}: set both or neither"
            )

    @property
    def enabled(self) -> bool:
        return self.generations > 0


@dataclass(frozen=True)
class EvoRainbowAlgoConfig:
    """Configuration for EvoRainbow (TD3 + CEM + surrogate fitness)."""

    horizon: int = 1000
    embedding_dim: int = 300
    pop_size: int = 5
    cem: CEMConfig = field(default_factory=CEMConfig)
    rl: TD3RLConfig = field(default_factory=TD3RLConfig)
    rl_to_ea: RLToEAConfig = field(default_factory=RLToEAConfig)
    ea_to_rl: EAToRLConfig = field(default_factory=EAToRLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)
    matched: MatchedConfig = field(default_factory=MatchedConfig)

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon={self.horizon}; must be >= 1")
        if self.embedding_dim < 1:
            raise ValueError(
                f"embedding_dim={self.embedding_dim}; must be >= 1"
            )
        if self.pop_size < 2:
            raise ValueError(f"pop_size={self.pop_size}; must be >= 2")
        if self.eval.mode in ("hboot", "prefix") and self.eval.h > self.horizon:
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
class CheckpointConfig:
    """Offline-analysis checkpoints at fractions of the step budget."""

    dir: str | None = None
    fractions: tuple[float, ...] = (0.1, 0.5, 1.0)
    buffer_tail: int = 200_000
    generations: int = 4

    def __post_init__(self) -> None:
        if not self.fractions:
            raise ValueError("checkpoint.fractions must have at least 1 entry")
        if list(self.fractions) != sorted(set(self.fractions)):
            raise ValueError(
                f"checkpoint.fractions={list(self.fractions)} must be "
                "strictly increasing"
            )
        for fraction in self.fractions:
            if not 0.0 < fraction <= 1.0:
                raise ValueError(
                    f"checkpoint.fractions entry {fraction} not in (0, 1]"
                )
        _check_min("checkpoint.buffer_tail", self.buffer_tail, 1)
        _check_min("checkpoint.generations", self.generations, 1)

    @property
    def enabled(self) -> bool:
        return self.dir is not None


@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases logging configuration."""

    enabled: bool = True
    project: str = "critic_uphill"
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
    checkpoint: CheckpointConfig
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
        if (
            self.algorithm.shadow.enabled
            and self.probe.enabled
            and self.probe.n_perturbations < SHADOW_PROBE_PICKS + 1
        ):
            raise ValueError(
                f"probe.n_perturbations={self.probe.n_perturbations} leaves "
                f"no random pick next to the {SHADOW_PROBE_PICKS} shadow "
                f"probe picks. use >= {SHADOW_PROBE_PICKS + 1}"
            )
        if (
            self.checkpoint.enabled
            and isinstance(self.algorithm, EvoRainbowAlgoConfig)
            and self.algorithm.matched.enabled
        ):
            raise ValueError(
                "checkpoint.dir with algorithm.matched: a matched run stops "
                "before the budget fractions are reached"
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


@dataclass(frozen=True)
class OfflineConfig:
    """Offline measurement on one checkpoint, outside any budget."""

    checkpoint: str
    out: str
    measurement: MeasurementName = "signals"
    sigmas: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2)
    n_perturbations: int = 40
    n_states: int = 2000
    truth_heads: int = 20
    truth_episodes: int = 30
    prefix_seeds: int = 5
    prefix_h: int = 200
    fqe_steps: int = 100_000
    fqe_batch_size: int = 256
    step_sigmas: tuple[float, ...] = (0.02, 0.05, 0.1)
    pair_sigmas: tuple[float, ...] = (0.05, 0.1)
    n_pairs: int = 20
    max_envs: int = 256

    def __post_init__(self) -> None:
        _check_choice("measurement", self.measurement, MeasurementName)
        if not self.sigmas or not self.pair_sigmas:
            raise ValueError("sigmas and pair_sigmas need at least 1 entry")
        for sigma in (*self.sigmas, *self.step_sigmas, *self.pair_sigmas):
            _check_positive("sigmas/step_sigmas/pair_sigmas entry", sigma)
        for name in (
            "n_perturbations",
            "n_states",
            "truth_heads",
            "truth_episodes",
            "prefix_seeds",
            "prefix_h",
            "fqe_steps",
            "fqe_batch_size",
            "n_pairs",
            "max_envs",
        ):
            _check_min(name, getattr(self, name), 1)
        if self.truth_heads > self.n_perturbations:
            raise ValueError(
                f"truth_heads={self.truth_heads} exceeds the "
                f"n_perturbations={self.n_perturbations} per sigma"
            )
