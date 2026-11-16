import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from analysis.runs import load_spec

Cell = dict[str, Any]
# predictors ranked against the truth; -dist_H = t2_td3, -dist_L2 below
PREDICTORS = ("score_td3", "score_fqe", "neg_dist_l2", "neg_dist_h", "prefix")


def rho(x: np.ndarray, y: np.ndarray, where: str) -> float:
    r = float(spearmanr(x, y).statistic)
    if math.isnan(r):
        raise ValueError(f"{where}: Spearman undefined (constant input)")
    return r


def partial(x: np.ndarray, y: np.ndarray, z: np.ndarray, where: str) -> float:
    # partial Spearman of x and y given z, from the three pairwise rhos
    rxy, rxz, ryz = rho(x, y, where), rho(x, z, where), rho(y, z, where)
    return (rxy - rxz * ryz) / math.sqrt((1 - rxz**2) * (1 - ryz**2))


def ratio(t1: np.ndarray, t2: np.ndarray) -> float:
    return float(np.median(np.abs(t1)) / np.median(np.abs(t2)))


def cell_stats(rows: list[dict[str, Any]], actor: dict, where: str) -> Cell:
    col = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "truth"}
    episodes = np.array([r["truth"] for r in rows])
    truth = episodes.mean(axis=1)
    half = episodes.shape[1] // 2
    r_half = rho(episodes[:, :half].mean(1), episodes[:, half:].mean(1), where)
    neg_dist_h = col["t2_td3"]
    x = {
        "score_td3": col["score_td3"],
        "score_fqe": col["score_fqe"],
        "neg_dist_l2": -col["dist_l2"],
        "neg_dist_h": neg_dist_h,
        "prefix": col["prefix"],
    }
    return {
        "drop": float(1 - truth.mean() / np.mean(actor["truth"])),
        "rho_truth": {k: rho(v, truth, where) for k, v in x.items()},
        "rho_td3_dist": rho(col["score_td3"], neg_dist_h, where),
        "rho_fqe_dist": rho(col["score_fqe"], neg_dist_h, where),
        "partial_td3": partial(col["score_td3"], truth, neg_dist_h, where),
        "partial_fqe": partial(col["score_fqe"], truth, neg_dist_h, where),
        "ratio_td3": ratio(col["t1_td3"], col["t2_td3"]),
        "ratio_fqe": ratio(col["t1_fqe"], col["t2_fqe"]),
        # split-half reliability of the truth, Spearman-Brown to 10 episodes
        "ceiling": 2 * r_half / (1 + r_half),
    }


def load_cells(results: Path, envs: list[str]) -> list[Cell]:
    cells = []
    for env in envs:
        for path in sorted((results / env).glob("s*_step_*.json")):
            d = json.loads(path.read_text())
            meta = d["meta"]
            for level, sigma in enumerate(d["sigmas"]):
                rows = [r for r in d["perturbations"] if r["level"] == level]
                where = f"{path.name} level {level}"
                cells.append(
                    {
                        "env": env,
                        "seed": meta["seed"],
                        "step": meta["env_steps"],
                        "level": level,
                        "sigma": sigma,
                        "n": len(rows),
                        "actor_return": float(np.mean(d["actor"]["truth"])),
                    }
                    | cell_stats(rows, d["actor"], where)
                )
    return sorted(
        cells, key=lambda c: (c["env"], c["seed"], c["step"], c["level"])
    )


def med(cells: list[Cell], key: str) -> float:
    return float(np.median([c[key] for c in cells]))


def hypotheses(cells: list[Cell], envs: list[str]) -> dict[str, Any]:
    h1 = med(cells, "rho_td3_dist") >= 0.8 and med(cells, "ratio_td3") <= 0.2
    h2 = abs(med(cells, "partial_td3")) <= 0.1
    gain = {
        e: float(
            np.median(
                [
                    c["partial_fqe"] - c["partial_td3"]
                    for c in cells
                    if c["env"] == e
                ]
            )
        )
        for e in envs
        if any(c["env"] == e for c in cells)
    }
    envs_up = sum(g >= 0.15 for g in gain.values())
    ratio_up = med(cells, "ratio_fqe") >= 2 * med(cells, "ratio_td3")
    h3 = envs_up >= math.ceil(2 * len(envs) / 3) and ratio_up
    if not h1:
        decision = "noH1"
    else:
        decision = "H1+H3" if h3 else "H1-noH3"
    return {
        "medians": {
            k: med(cells, k)
            for k in (
                "rho_td3_dist",
                "rho_fqe_dist",
                "ratio_td3",
                "ratio_fqe",
                "partial_td3",
                "partial_fqe",
                "ceiling",
            )
        },
        "partial_gain_per_env": gain,
        "H1": bool(h1),
        "H2": bool(h2),
        "H3": bool(h3),
        "decision": decision,
    }


def print_table(cells: list[Cell]) -> None:
    head = (
        "env              seed    step lv  drop  ceil  rTD3  rFQE  r-dL2"
        "  r-dH  rPre  TD3~dH pTD3  pFQE  q1/2TD3 q1/2FQE"
    )
    print(head)
    for c in cells:
        r = c["rho_truth"]
        print(
            f"{c['env']:<16} {c['seed']:>4} {c['step']:>7} {c['level']:>2}"
            f" {c['drop']:5.2f} {c['ceiling']:5.2f}"
            + "".join(f" {r[k]:5.2f}" for k in PREDICTORS)
            + f"  {c['rho_td3_dist']:5.2f} {c['partial_td3']:5.2f}"
            f" {c['partial_fqe']:5.2f}  {c['ratio_td3']:6.3f}"
            f"  {c['ratio_fqe']:6.3f}"
        )


def main(toml_path: Path) -> None:
    spec = load_spec(toml_path)
    off = spec["offline"]
    results = spec["dir"] / spec["analysis"]["offline_cache"]
    cells = load_cells(results, spec["envs"])
    expected = (
        len(spec["envs"])
        * len(spec["seeds"])
        * len(off["checkpoint_at"])
        * len(off["target_drops"])
    )
    if not cells:
        raise FileNotFoundError(f"no s*_step_*.json under {results}")
    print_table(cells)
    verdict = hypotheses(cells, spec["envs"])
    verdict["coverage"] = f"{len(cells)}/{expected} cells"
    verdict["decision_text"] = spec["decision"][verdict["decision"]]
    print(json.dumps(verdict, indent=1))
    out = spec["dir"] / "degeneracy.json"
    out.write_text(json.dumps({"verdict": verdict, "cells": cells}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
