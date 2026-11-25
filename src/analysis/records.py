from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from analysis.stats import SeedValues, seed_bootstrap

CORE_ENVS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "Ant-v5")
ENVS = (*CORE_ENVS, "Swimmer-v5")
ALGO_LABELS = {
    "td3": "TD3",
    "evorainbow-td3-full": "ERL",
    "evorainbow-td3-critic": "SC-ERL",
}
RUN_NAME = re.compile(r"^(?P<cond>.+)_(?P<env>[^_]+)_s(?P<seed>\d+)$")
Record = dict[str, Any]


def run_name(d: Record) -> str:
    """<cond>_<env>_s<seed>, the checkpoint directory's parent."""
    return Path(d["checkpoint"]).parent.name


def file_name(d: Record) -> str:
    """<run>_p<percent>.json, how every measurement output is named."""
    return f"{run_name(d)}_p{round(d['fraction'] * 100):03d}.json"


def load_records(directory: Path) -> list[Record]:
    """Offline JSONs sorted by file name, each tagged with its condition."""
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no offline JSONs in {directory}")
    out = []
    for path in paths:
        d = json.loads(path.read_text())
        m = RUN_NAME.match(run_name(d))
        if m is None:
            raise ValueError(
                f"{path}: run name {run_name(d)!r} is not <cond>_<env>_s<seed>"
            )
        d["cond"] = m["cond"]
        out.append(d)
    return out


class SeedTable:
    """table[env][key][seed] -> values, summarized by seed bootstrap."""

    def __init__(self) -> None:
        self._cells: dict[str, dict[str, SeedValues]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

    def __contains__(self, env: str) -> bool:
        return env in self._cells

    def add(self, env: str, seed: int, row: Mapping[str, float | None]) -> None:
        for key, value in row.items():
            self._cells[env][key][seed].append(
                np.nan if value is None else value
            )

    def summarize(
        self, rng: np.random.Generator
    ) -> dict[str, dict[str, tuple[float, float, float]]]:
        """Median and seed-bootstrap CI per (env, key), insertion order."""
        return {
            env: {k: seed_bootstrap(v, rng) for k, v in keys.items()}
            for env, keys in self._cells.items()
        }
