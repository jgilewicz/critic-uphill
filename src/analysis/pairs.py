from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analysis.records import Record, SeedTable, run_name
from analysis.stats import SeedValues, ols_slope, seed_bootstrap, spearman


def _odd(x: np.ndarray, level: int) -> np.ndarray:
    return 0.5 * (x[level, 0] - x[level, 1])


def _even(x: np.ndarray, level: int) -> np.ndarray:
    return 0.5 * (x[level, 0] + x[level, 1])


def pair_row(d: Record) -> dict[str, float]:
    """Spearman and OLS slope truth ~ critic of odd and even parts."""
    critic, truth = np.asarray(d["critic"]), np.asarray(d["truth"])
    out = {}
    for level, sigma in enumerate(d["sigmas"]):
        for name, part in (("odd", _odd), ("even", _even)):
            c, t = part(critic, level), part(truth, level)
            out[f"{name}/sp/{sigma}"] = spearman(c, t)
            out[f"{name}/slope/{sigma}"] = ols_slope(c, t)
    return out


def odd_share(d: Record, level: int) -> list[float]:
    """|odd| / (|odd| + |even|) of the true return gain, per pair."""
    truth = np.asarray(d["truth"])
    odd = np.abs(_odd(truth, level))
    even = np.abs(_even(truth, level))
    if np.any(odd + even == 0):
        raise ValueError(
            f"{run_name(d)} sigma {d['sigmas'][level]}: a pair with zero "
            "gain on both sides, the odd share is undefined"
        )
    return (odd / (odd + even)).tolist()


class PairAnalysis:
    """Odd (linear) and even (curvature) parts of antithetic pairs.

    Checks that every pair record used the same seed set B as the signal
    record of the final checkpoint. The odd/even table and the odd share
    each draw from their own rng 3.
    """

    def __init__(self, records: list[Record], signals_dir: Path) -> None:
        self._records = records
        self._signals_dir = signals_dir

    def check_actor(self, d: Record) -> float:
        """Max actor return gap vs the signal record, which should be 0."""
        path = self._signals_dir / f"{run_name(d)}_p100.json"
        signals = json.loads(path.read_text())
        if signals["seeds_b"] != d["seeds_b"]:
            raise ValueError(
                f"{path.name}: seed set B differs between the pair and "
                "signal records, rerun both with the same offline.seed"
            )
        return float(
            np.abs(
                np.subtract(signals["actor_returns"], d["actor_returns"])
            ).max()
        )

    def _table(self) -> dict[str, Any]:
        table = SeedTable()
        for d in self._records:
            table.add(d["env_id"], d["seed"], pair_row(d))
        return table.summarize(np.random.default_rng(3))

    def _odd_share(self) -> dict[str, dict[str, Any]]:
        shares: dict[str, dict[float, SeedValues]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for d in self._records:
            for level, sigma in enumerate(d["sigmas"]):
                shares[d["env_id"]][sigma][d["seed"]] += odd_share(d, level)
        rng = np.random.default_rng(3)
        out: dict[str, dict[str, Any]] = {}
        for env in sorted(shares):
            out[env] = {}
            for sigma in sorted(shares[env]):
                by_seed = shares[env][sigma]
                out[env][str(sigma)] = {
                    "share": seed_bootstrap(by_seed, rng),
                    "n_seeds": len(by_seed),
                    "n_pairs": sum(len(v) for v in by_seed.values()),
                }
        return out

    def run(self) -> dict[str, Any]:
        return {
            "actor_return_gap": {
                run_name(d): self.check_actor(d) for d in self._records
            },
            "odd_even": self._table(),
            "odd_share": self._odd_share(),
        }
