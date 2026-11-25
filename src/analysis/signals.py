from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from analysis.records import (
    ALGO_LABELS,
    CORE_ENVS,
    ENVS,
    Record,
    SeedTable,
)

SIGNALS = (
    "q1",
    "q2",
    "shadow",
    "fqe",
    "own_q1",
    "prefix_return",
    "dist_action",
    "dist_weight",
)
PER_SIGMA_SIGNALS = ("q1", "q2", "fqe", "own_q1", "dist_action")
SAME_AS_NEAREST_SIGNALS = SIGNALS[:6]
GAIN = "top1_minus_nearest_action"
ERL_CONDS = ("evorainbow-td3-full", "evorainbow-td3-critic")
# (point, lo, hi) from seed_bootstrap
Interval = tuple[float, float, float]
Summary = dict[str, dict[str, Interval]]


def _sigma_mean(per_sigma: list[dict], signal: str, field: str) -> float:
    values = [
        p["signals"][signal][field] for p in per_sigma if signal in p["signals"]
    ]
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else np.nan


def perturbation_row(d: Record) -> dict[str, float]:
    """Spearman and top-1 gain per signal, averaged over sigmas."""
    per_sigma = d["perturbations"]["per_sigma"]
    row = {}
    for s in SIGNALS:
        row[f"sp/{s}"] = _sigma_mean(per_sigma, s, "spearman")
        row[f"gain/{s}"] = _sigma_mean(per_sigma, s, GAIN)
    for i, p in enumerate(per_sigma):
        for s in PER_SIGMA_SIGNALS:
            if s in p["signals"]:
                row[f"sp_s{i}/{s}"] = p["signals"][s]["spearman"]
                row[f"gain_s{i}/{s}"] = p["signals"][s][GAIN]
    return _with_diffs(row)


def population_row(d: Record) -> dict[str, float]:
    summary = d["population"]["summary"]
    row = {}
    for s in SIGNALS:
        if s in summary:
            row[f"sp/{s}"] = summary[s]["spearman"]
            row[f"gain/{s}"] = summary[s][GAIN]
    return _with_diffs(row)


def _with_diffs(row: dict[str, float]) -> dict[str, float]:
    if "sp/q1" in row:
        row["sp/q1-dist"] = row["sp/q1"] - row["sp/dist_action"]
        row["sp/own-q1"] = row["sp/own_q1"] - row["sp/q1"]
    return row


def ratio_slope(d: Record) -> float:
    """Log-log slope of the Q1 deg ratio over sigma."""
    sigmas = np.asarray(d["config"]["sigmas"])
    ratio = np.asarray([d["deg"][f"q1/ratio_s{i}"] for i in range(len(sigmas))])
    if (ratio <= 0).any():
        return np.nan
    return float(np.polyfit(np.log(sigmas), np.log(ratio), 1)[0])


def gradient_row(d: Record) -> dict[str, float]:
    g = d["gradient"]
    row = {
        "norm_ratio": g["grad_norm_fresh"] / g["grad_norm_buffer"],
        "cosine": g["cosine"],
        **{f"step{i}": gain for i, gain in enumerate(g["step_gains"])},
    }
    row["slope"] = ratio_slope(d)
    return row


def _count(
    st: Summary,
    key: str,
    test: Callable[[Interval], bool],
    envs: tuple[str, ...] = CORE_ENVS,
) -> tuple[int, list[str]]:
    hits = [e for e in envs if e in st and key in st[e] and test(st[e][key])]
    return len(hits), hits


def _covers0(ci: Interval) -> bool:
    return ci[1] <= 0 <= ci[2]


def _above0(ci: Interval) -> bool:
    return ci[1] > 0


def _not_above0(ci: Interval) -> bool:
    return ci[1] <= 0


def _like_inverse(ci: Interval) -> bool:
    return ci[1] <= -1 <= ci[2] and ci[2] < 0


def ranking_verdicts(st: Summary) -> dict[str, Any]:
    """Pre-registered rules on one perturbation or population table."""
    gain_covers0 = _count(st, "gain/q1", _covers0)
    sp_not_above = _count(st, "sp/q1-dist", _not_above0)
    q2 = _count(st, "gain/q2", _above0)
    fqe = _count(st, "gain/fqe", _above0)
    own = _count(st, "sp/own-q1", _above0)
    return {
        "q1_no_better_than_nearest": {
            "gain_covers_0": gain_covers0,
            "spearman_not_above_dist": sp_not_above,
            "holds": gain_covers0[0] >= 3 and sp_not_above[0] >= 3,
        },
        "independent_critic_helps": {
            "q2": q2,
            "fqe": fqe,
            "holds": q2[0] >= 2 or fqe[0] >= 2,
        },
        "fresh_states_help": {
            "own_above_buffer": own,
            "holds": own[0] >= 2,
        },
        "n_core_envs": sum(e in st for e in CORE_ENVS),
    }


def gradient_verdicts(st: Summary) -> dict[str, Any]:
    steps = {
        f"step{i}": {
            "all_envs": _count(st, f"step{i}", _above0, ENVS),
            "core_envs": _count(st, f"step{i}", _above0, CORE_ENVS),
        }
        for i in range(3)
    }
    return {
        "ratio_falls_like_1_over_sigma": _count(
            st, "slope", _like_inverse, ENVS
        ),
        "gradient_steps": steps,
        "gradient_step_helps_2_envs": any(
            v["all_envs"][0] >= 2 for v in steps.values()
        ),
    }


class SignalAnalysis:
    """Ranking quality of every signal, verdicts and deg ratio per sigma.

    Tables are medians with 95% seed-bootstrap CIs over checkpoints. The
    main tables use rng 0, the deg ratio and top-1 == nearest shares rng 1.
    """

    def __init__(self, records: list[Record]) -> None:
        self._records = records

    def _table(self, cond: str, row: Callable[[Record], dict]) -> SeedTable:
        table = SeedTable()
        for d in self._records:
            if d["cond"] == cond:
                table.add(d["env_id"], d["seed"], row(d))
        return table

    def _populations(self, rng: np.random.Generator) -> dict[str, Summary]:
        out = {}
        for cond in ERL_CONDS:
            table = SeedTable()
            for d in self._records:
                if d["cond"] == cond and d["population"] is not None:
                    table.add(d["env_id"], d["seed"], population_row(d))
            out[ALGO_LABELS[cond]] = table.summarize(rng)
        return out

    def _main(self, rng: np.random.Generator) -> dict[str, Any]:
        pert = self._table("td3", perturbation_row).summarize(rng)
        grad = self._table("td3", gradient_row).summarize(rng)
        pops = self._populations(rng)
        actor_pert = {
            ALGO_LABELS[c]: self._table(c, perturbation_row).summarize(rng)
            for c in ERL_CONDS
        }
        return {
            "verdicts": {
                "td3_perturbations": ranking_verdicts(pert),
                "populations": {
                    name: ranking_verdicts(st) for name, st in pops.items()
                },
                "gradient": gradient_verdicts(grad),
            },
            "tables": {
                "td3_perturbations": pert,
                "gradient": grad,
                "populations": pops,
                "actor_perturbations": actor_pert,
            },
        }

    def _deg_ratio(self, rng: np.random.Generator) -> Summary:
        table = SeedTable()
        for d in self._records:
            if d["cond"] == "td3":
                ratios = {
                    str(s): d["deg"][f"q1/ratio_s{i}"]
                    for i, s in enumerate(d["config"]["sigmas"])
                }
                table.add(d["env_id"], d["seed"], ratios)
        return table.summarize(rng)

    def _same_as_nearest(self) -> dict[str, dict[str, float]]:
        """Share of picks where the signal's top-1 is the nearest member."""
        same: dict[str, dict[str, list[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for d in self._records:
            algo, env = ALGO_LABELS[d["cond"]], d["env_id"]
            if algo == "TD3":
                for p in d["perturbations"]["per_sigma"]:
                    for s in SAME_AS_NEAREST_SIGNALS:
                        same[f"TD3 pert | {env}"][s].append(
                            p["signals"][s][GAIN] == 0
                        )
            if d["population"] is not None:
                summary = d["population"]["summary"]
                for s in SAME_AS_NEAREST_SIGNALS:
                    same[f"{algo} pop | {env}"][s].append(summary[s][GAIN] == 0)
        return {
            k: {s: float(np.mean(v)) for s, v in sig.items()}
            for k, sig in same.items()
        }

    def _cost(self) -> dict[str, dict[str, float]]:
        acc: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for d in self._records:
            key = f"{ALGO_LABELS.get(d['cond'], d['cond'])} {d['env_id']}"
            acc[key]["env_steps"].append(sum(d["cost"]["env_steps"].values()))
            acc[key]["seconds"].append(d["cost"]["seconds"]["total"])
        return {
            k: {m: float(np.mean(v)) for m, v in ms.items()}
            | {"n": len(ms["seconds"])}
            for k, ms in acc.items()
        }

    def run(self) -> dict[str, Any]:
        main = self._main(np.random.default_rng(0))
        return {
            "n_checkpoints": len(self._records),
            **main,
            "deg_ratio": self._deg_ratio(np.random.default_rng(1)),
            "same_as_nearest": self._same_as_nearest(),
            "cost": self._cost(),
        }
