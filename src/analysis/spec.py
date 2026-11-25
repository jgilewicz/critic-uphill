from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Comparison:
    a: str
    b: str


@dataclass(frozen=True)
class EnvGroup:
    """Envs analyzed together. Holm and verdicts only when tested."""

    name: str
    envs: tuple[str, ...]
    conditions: tuple[str, ...]
    tested: bool


@dataclass(frozen=True)
class SuiteSpec:
    """Pre-registered analysis of one training suite's final scores."""

    entity: str
    project: str
    tags: tuple[str, ...]
    # short label -> run condition name (<cond> in <cond>_<env>_s<seed>)
    conditions: dict[str, str]
    reference: str
    groups: tuple[EnvGroup, ...]
    primary: tuple[Comparison, ...]
    secondary: tuple[Comparison, ...]
    alpha: float
    margin: float
    budget: int
    rng_seed: int
    history_keys: tuple[str, ...]
    surrogate_keys: tuple[str, ...]
    # surrogate means skip rows logged at or before this env step (warmup)
    surrogate_after_steps: int

    @classmethod
    def load(cls, path: Path) -> SuiteSpec:
        raw = tomllib.loads(path.read_text())
        try:
            return cls._parse(raw)
        except KeyError as e:
            raise ValueError(
                f"{path}: missing key {e}, see src/analysis/specs/ for the "
                "expected layout"
            ) from e

    @classmethod
    def _parse(cls, raw: dict[str, Any]) -> SuiteSpec:
        conds = dict(raw["conditions"])
        a = raw["analysis"]
        groups = tuple(
            EnvGroup(
                name=g["name"],
                envs=tuple(g["envs"]),
                conditions=tuple(g.get("conditions", conds)),
                tested=g["tested"],
            )
            for g in raw["groups"]
        )
        spec = cls(
            entity=raw["wandb"]["entity"],
            project=raw["wandb"]["project"],
            tags=tuple(raw["wandb"]["tags"]),
            conditions=conds,
            reference=a["reference"],
            groups=groups,
            primary=tuple(Comparison(*c) for c in a["primary"]),
            secondary=tuple(Comparison(*c) for c in a["secondary"]),
            alpha=a["alpha"],
            margin=a["margin"],
            budget=a["budget"],
            rng_seed=a["rng_seed"],
            history_keys=tuple(a["history_keys"]),
            surrogate_keys=tuple(a["surrogate_keys"]),
            surrogate_after_steps=a["surrogate_after_steps"],
        )
        spec._check_labels()
        return spec

    def _check_labels(self) -> None:
        used = {self.reference}
        for c in (*self.primary, *self.secondary):
            used |= {c.a, c.b}
        for g in self.groups:
            used |= set(g.conditions)
        unknown = used - set(self.conditions)
        if unknown:
            raise ValueError(
                f"labels {sorted(unknown)} are not in [conditions], add them "
                "or fix the comparison"
            )
