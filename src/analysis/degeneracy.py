import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr, t as student_t

from analysis.runs import load_spec

TEMPLATE = Path(__file__).resolve().parent / "templates" / "degeneracy.html"
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


def points(rows: list[dict[str, Any]], actor: dict) -> list[list[float]]:
    # scatter rows: dist_H, score_TD3 and score_FQE relative to the actor,
    # truth
    return [
        [
            -r["t2_td3"],
            r["score_td3"] - actor["score_td3"],
            r["score_fqe"] - actor["score_fqe"],
            float(np.mean(r["truth"])),
        ]
        for r in rows
    ]


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
                        "fqe_loss": d["fqe_loss"][-1],
                        "points": points(rows, d["actor"]),
                    }
                    | cell_stats(rows, d["actor"], where)
                )
    return sorted(
        cells, key=lambda c: (c["env"], c["seed"], c["step"], c["level"])
    )


def med(cells: list[Cell], key: str) -> float:
    return float(np.median([c[key] for c in cells]))


MEDIANS = (
    "rho_td3_dist",
    "rho_fqe_dist",
    "ratio_td3",
    "ratio_fqe",
    "partial_td3",
    "partial_fqe",
    "ceiling",
)


def per_env(cells: list[Cell], envs: list[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for e in envs:
        mine = [c for c in cells if c["env"] == e]
        out[e] = {k: med(mine, k) for k in MEDIANS} | {
            f"rho_truth_{k}": float(
                np.median([c["rho_truth"][k] for c in mine])
            )
            for k in PREDICTORS
        }
    return out


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
        "medians": {k: med(cells, k) for k in MEDIANS},
        "partial_gain_per_env": gain,
        "H1": bool(h1),
        "H2": bool(h2),
        "H3": bool(h3),
        "decision": decision,
    }


def paired_ci(d: np.ndarray, alpha: float) -> tuple[float, float, float]:
    # two-sided t-interval of the mean paired difference
    half = (
        student_t.ppf(1 - alpha / 2, len(d) - 1)
        * d.std(ddof=1)
        / math.sqrt(len(d))
    )
    return float(d.mean()), float(d.mean() - half), float(d.mean() + half)


def gain_of(scored: list[dict[str, Any]], reason: str) -> float | None:
    # every top-1 pick is re-evaluated by construction, None if absent
    hits = [c["gain"] for c in scored if reason in c["reasons"]]
    return hits[0] if hits else None


def validation_cell(
    d: dict[str, Any], level: int, alpha: float, n_level: int
) -> Cell:
    actor = np.array(d["actor"])
    base = abs(float(actor.mean()))
    mine = [c for c in d["candidates"] if c["level"] == level]
    k = len(mine)
    scored: list[dict[str, Any]] = []
    for c in mine:
        diff = np.array(c["returns"]) - actor
        mean, lo, hi = paired_ci(diff, alpha)
        bonf_lo = paired_ci(diff, alpha / k)[1]
        scored.append(
            {
                "row": c["row"],
                "reasons": c["reasons"],
                "gain": mean / base,
                "lo": lo / base,
                "hi": hi / base,
                "confirmed": lo > 0,
                "confirmed_bonf": bonf_lo > 0,
            }
        )
    better = [c for c in scored if "better" in c["reasons"]]
    top1 = {r: gain_of(scored, f"top_{r}") for r in ("truth", "td3", "prefix")}
    return {
        "env": d["meta"]["env"],
        "seed": d["meta"]["seed"],
        "step": d["meta"]["env_steps"],
        "level": level,
        "actor_return": float(actor.mean()),
        "n_level": n_level,
        "n_candidates": k,
        "n_better": len(better),
        "n_confirmed": sum(c["confirmed"] for c in scored),
        "n_confirmed_better": sum(c["confirmed"] for c in better),
        "passes": any(c["confirmed"] for c in scored),
        "passes_bonf": any(c["confirmed_bonf"] for c in scored),
        "best_gain": max(c["gain"] for c in scored),
        "top1_truth_gain": top1["truth"],
        "top1_td3_gain": top1["td3"],
        "top1_prefix_gain": top1["prefix"],
        "candidates": scored,
    }


def load_validation(
    results: Path, envs: list[str], val: dict[str, Any], n_level: int
) -> list[Cell]:
    cells = []
    for env in envs:
        for path in (results / env).glob("validate_s*_step_*.json"):
            d = json.loads(path.read_text())
            for level in range(len(d["sigmas"])):
                cells.append(validation_cell(d, level, val["alpha"], n_level))
    return sorted(
        cells, key=lambda c: (c["env"], c["seed"], c["step"], c["level"])
    )


def validation_verdict(
    vcells: list[Cell], envs: list[str], val: dict[str, Any]
) -> dict[str, Any]:
    out = {}
    for e in envs:
        mine = [c for c in vcells if c["env"] == e]
        share = float(np.mean([c["passes"] for c in mine]))
        groups = {}
        for level in sorted({c["level"] for c in mine}):
            g = [c for c in mine if c["level"] == level]
            n_better = sum(c["n_better"] for c in g)
            groups[level] = {
                "cells": len(g),
                "confirmed_of_all": sum(c["n_confirmed"] for c in g)
                / sum(c["n_level"] for c in g),
                "confirmed_of_better": (
                    sum(c["n_confirmed_better"] for c in g) / n_better
                    if n_better
                    else None
                ),
                "n_better": n_better,
                "passing_cells": sum(c["passes"] for c in g),
                "passing_cells_bonf": sum(c["passes_bonf"] for c in g),
            } | {
                f"median_{k}": float(np.median([c[k] for c in g]))
                for k in (
                    "best_gain",
                    "top1_truth_gain",
                    "top1_td3_gain",
                    "top1_prefix_gain",
                )
            }
        out[e] = {
            "passing_share": share,
            "passing_share_bonf": float(
                np.mean([c["passes_bonf"] for c in mine])
            ),
            "real": share >= val["min_cell_share"],
            "levels": groups,
        }
    return out


def print_validation(verdict: dict[str, Any]) -> None:
    print(
        "env              lv cells  ok/all ok/better  pass bonf"
        "   best  top1tr top1td3 top1pre"
    )
    for e, v in verdict.items():
        for level, g in v["levels"].items():
            ob = g["confirmed_of_better"]
            print(
                f"{e:<16} {level:>2} {g['cells']:>5}"
                f"  {g['confirmed_of_all']:6.3f}"
                f"  {'  n/a' if ob is None else f'{ob:5.2f}'}"
                f"  {g['passing_cells']:>4} {g['passing_cells_bonf']:>4}"
                + "".join(
                    f" {g[f'median_{k}'] * 100:+6.1f}%"
                    for k in (
                        "best_gain",
                        "top1_truth_gain",
                        "top1_td3_gain",
                        "top1_prefix_gain",
                    )
                )
            )
        print(
            f"{e:<16} passing {v['passing_share']:.2f}"
            f" (bonf {v['passing_share_bonf']:.2f}) real={v['real']}"
        )


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
    val = spec["validation"]
    vcells = load_validation(results, spec["envs"], val, off["perturbations"])
    if len(vcells) != expected:
        raise FileNotFoundError(
            f"{len(vcells)}/{expected} validation cells under {results}. "
            "run python -m offline.critic_degeneracy validate per env"
        )
    vverdict = validation_verdict(vcells, spec["envs"], val)
    print_validation(vverdict)
    data = {
        "verdict": verdict,
        "validation": {"verdict": vverdict, "cells": vcells, "spec": val},
        "per_env": per_env(cells, spec["envs"]),
        "cells": cells,
        "hypotheses": spec["hypotheses"],
        "sigmas": {
            e: json.loads((results / e / "sigma.json").read_text())["sigmas"]
            for e in spec["envs"]
        },
    }
    out = spec["dir"] / "degeneracy.json"
    out.write_text(json.dumps(data, indent=1))
    page = spec["dir"] / "degeneracy.html"
    page.write_text(
        TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data))
    )
    print(f"-> {out}\n-> {page}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
