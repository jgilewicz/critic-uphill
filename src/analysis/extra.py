from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analysis.records import ALGO_LABELS, Record, SeedTable, file_name
from analysis.stats import spearman

FRACTIONS = (0.5, 1.0)


def _top1_minus_nearest(
    signal: np.ndarray, dist: np.ndarray, gains: np.ndarray
) -> float:
    return float(gains[np.argmax(signal)] - gains[np.argmax(dist)])


def per_sigma_mean(d: Record, signals: dict[str, np.ndarray]) -> dict:
    """Spearman and top-1 gain vs nearest per sigma on truth picks, averaged."""
    p = d["perturbations"]
    picks = np.asarray(p["truth_index"])
    gains = np.asarray(p["truth_gain"])
    levels = np.asarray(p["levels"])[picks]
    dist = np.asarray(p["signals"]["dist_action"])[picks]
    acc: dict[str, list[float]] = defaultdict(list)
    for level in np.unique(levels):
        m = levels == level
        for name, values in signals.items():
            v = values[picks][m]
            acc[f"sp/{name}"].append(spearman(v, gains[m]))
            acc[f"gain/{name}"].append(
                _top1_minus_nearest(v, dist[m], gains[m])
            )
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def curvature_row(d: Record, extra: Record) -> dict[str, float]:
    """Ranking by dS (Q1), by the curvature term t2 and by action distance."""
    sig = d["perturbations"]["signals"]
    q1 = np.asarray(sig["q1"])
    terms = extra["terms"]
    drift = np.asarray(terms["q1_signal"]) - q1
    if np.abs(drift).max() > 1e-3 * (np.abs(q1).max() + 1):
        raise ValueError(
            f"{extra['checkpoint']}: extra q1 differs from signals q1 by "
            f"{drift.max()}, the two runs did not use the same heads"
        )
    row = per_sigma_mean(
        d,
        {
            "a_q1": np.asarray(terms["delta_s"]),
            "b_t2": np.asarray(terms["t2"]),
            "c_dist": np.asarray(sig["dist_action"]),
        },
    )
    row["sp/a-b"] = row["sp/a_q1"] - row["sp/b_t2"]
    row["sp/b-c"] = row["sp/b_t2"] - row["sp/c_dist"]
    return row


def _with_prefix_diffs(row: dict[str, float]) -> dict[str, float]:
    row["sp/own-prefix"] = row["sp/own"] - row["sp/prefix"]
    row["gain/own-prefix"] = row["gain/own"] - row["gain/prefix"]
    return row


def prefix_row_perturbations(d: Record) -> dict[str, float]:
    sig = d["perturbations"]["signals"]
    row = per_sigma_mean(
        d,
        {
            "own": np.asarray(sig["own_q1"]),
            "prefix": np.asarray(sig["prefix_return"]),
        },
    )
    return _with_prefix_diffs(row)


def prefix_row_population(d: Record) -> dict[str, float]:
    s = d["population"]["summary"]
    row = {
        "sp/own": s["own_q1"]["spearman"],
        "sp/prefix": s["prefix_return"]["spearman"],
        "gain/own": s["own_q1"]["top1_minus_nearest_action"],
        "gain/prefix": s["prefix_return"]["top1_minus_nearest_action"],
    }
    return _with_prefix_diffs(row)


def noise_row(extra: Record) -> dict[str, float]:
    n = extra["noise"]
    cross_cos = float(np.mean(n["cross"]["cosine"]))
    return {
        "cos/buffer-buffer": n["buffer"]["cosine"],
        "cos/fresh-fresh": n["fresh"]["cosine"],
        "cos/buffer-fresh": cross_cos,
        "ratio/buffer-buffer": n["buffer"]["norm_ratio"],
        "ratio/fresh-fresh": n["fresh"]["norm_ratio"],
        "ratio/fresh-buffer": float(np.mean(n["cross"]["norm_ratio"])),
        "coh/buffer": float(np.mean(n["buffer"]["coherence"])),
        "coh/fresh": float(np.mean(n["fresh"]["coherence"])),
        "n_fresh": float(n["fresh"]["n"]),
        "cos/bb-bf": n["buffer"]["cosine"] - cross_cos,
        "cos/ff-bf": n["fresh"]["cosine"] - cross_cos,
    }


class ExtraAnalysis:
    """Curvature, prefix-return control and gradient noise floor.

    Reads signal records at FRACTIONS and, for TD3, the extra record of the
    same checkpoint. Tables are seed-bootstrapped with rng 2.
    """

    def __init__(self, records: list[Record], extra_dir: Path) -> None:
        self._records = [d for d in records if d["fraction"] in FRACTIONS]
        self._extra_dir = extra_dir

    def _extra(self, d: Record) -> Record:
        path = self._extra_dir / file_name(d)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing, run the extra measurement on "
                f"{d['checkpoint']}"
            )
        return json.loads(path.read_text())

    def run(self) -> dict[str, Any]:
        tables: dict[str, SeedTable] = defaultdict(SeedTable)
        for d in self._records:
            env, seed = d["env_id"], d["seed"]
            algo = ALGO_LABELS[d["cond"]]
            if algo != "TD3":
                tables[f"prefix {algo} pop"].add(
                    env, seed, prefix_row_population(d)
                )
                continue
            tables["prefix TD3 pert"].add(
                env, seed, prefix_row_perturbations(d)
            )
            extra = self._extra(d)
            tables["curvature"].add(env, seed, curvature_row(d, extra))
            tables["noise"].add(env, seed, noise_row(extra))
        rng = np.random.default_rng(2)
        return {name: t.summarize(rng) for name, t in tables.items()}
