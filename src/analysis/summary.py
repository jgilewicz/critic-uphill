import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from analysis.runs import (
    Hist,
    Run,
    auc,
    cache_dirs,
    cells,
    complete_envs,
    final_start,
    load,
    load_spec,
    select,
    series,
)
from analysis.stats import boot_iqm, ci, iqm, normalize, pooled

TEMPLATE = Path(__file__).resolve().parent / "templates" / "summary.html"
NOT_METRICS = {"env_steps", "generation", "_runtime", "_step", "_timestamp"}


def run_summary(hist: Hist, metrics: list[str], start: float) -> dict:
    out: dict[str, float | None] = {}
    for key in metrics:
        x, y = series(hist, key)
        tail = y[x >= start]
        out[key] = float(tail.mean()) if len(tail) else None
    primary = metrics[0]
    out["auc"] = auc(hist, primary)
    out["max_eval"] = float(series(hist, primary)[1].max())
    _, gens = series(hist, "generation")
    _, rt = series(hist, "_runtime")
    out["generations"] = float(gens.max() + 1) if len(gens) else None
    out["runtime_h"] = float(rt.max() / 3600) if len(rt) else None
    return out


def on_grid(hist: Hist, key: str, grid: np.ndarray) -> np.ndarray:
    x, y = series(hist, key)
    if len(y) < 2:
        return np.full(len(grid), np.nan)
    v = np.interp(grid, x, y)
    v[(grid < x[0]) | (grid > x[-1])] = np.nan
    return v


def _rounded(v: np.ndarray) -> list[float | None]:
    return [None if not np.isfinite(x) else round(float(x), 4) for x in v]


def curves(runs: list[Run], metrics: list[str], grid: np.ndarray) -> dict:
    out: dict[str, Any] = {}
    by_cell: dict[tuple[str, str], list[Hist]] = {}
    for r in runs:
        by_cell.setdefault((r["env"], r["condition"]), []).append(r["hist"])
    for (env, cond), hists in by_cell.items():
        for key in metrics:
            m = np.vstack([on_grid(h, key, grid) for h in hists])
            ok = np.isfinite(m)
            n = ok.sum(axis=0)
            if not n.any():
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(ok, m, 0.0).sum(axis=0) / n
                sq = np.where(ok, (m - mean) ** 2, 0.0).sum(axis=0)
                half = 1.96 * np.sqrt(sq / (n - 1)) / np.sqrt(n)
            out.setdefault(env, {}).setdefault(cond, {})[key] = {
                "mean": _rounded(mean),
                "ci": _rounded(half),
            }
    return out


def ranking(runs: list[Run], envs: list[str], field: str) -> list[dict]:
    conds = sorted({r["condition"] for r in runs if r["env"] in envs})
    scores = normalize(cells(runs, envs, conds, lambda r: r["summary"][field]))
    draws = boot_iqm(scores, conds, np.random.default_rng(0))
    rows = []
    for c in conds:
        lo, hi = ci(draws[c])
        point = float(iqm(pooled(scores, c)))
        rows.append({"condition": c, "iqm": point, "lo": lo, "hi": hi})
    return sorted(rows, key=lambda r: -r["iqm"])


def gate(runs: list[Run], env: str, rule: dict[str, Any], key: str) -> dict:
    final: dict[str, dict[int, float]] = {}
    for r in runs:
        if r["env"] == env and r["seed"] in rule["seeds"]:
            by_seed = final.setdefault(r["condition"], {})
            by_seed[r["seed"]] = r["summary"][key]
    ref = final[rule["reference"]]
    ref_mean = float(np.mean(list(ref.values())))
    variants = []
    for cond, by_seed in final.items():
        if cond == rule["reference"]:
            continue
        m = float(np.mean(list(by_seed.values())))
        shared = by_seed.keys() & ref.keys()
        wins = sum(by_seed[s] > ref[s] for s in shared)
        gain = (m - ref_mean) / abs(ref_mean)
        passed = gain >= rule["min_rel_gain"] and wins >= rule["min_wins"]
        variants.append(
            {
                "condition": cond,
                "mean": m,
                "gain": gain,
                "wins": int(wins),
                "n": len(shared),
                "pass": bool(passed),
            }
        )
    variants.sort(key=lambda v: -float(v["mean"]))
    return {
        "env": env,
        "ref_mean": ref_mean,
        "variants": variants,
        "pass": any(v["pass"] for v in variants),
    }


def group_rankings(
    runs: list[Run], groups: dict[str, list[str]], primary: str
) -> dict[str, Any]:
    out = {}
    for name, envs in groups.items():
        conds = sorted({r["condition"] for r in runs})
        present = complete_envs(runs, envs, conds)
        if not present:
            continue
        out[name] = {
            "envs": present,
            "seeds": len({r["seed"] for r in runs if r["env"] in present}),
            "final": ranking(runs, present, primary),
            "auc": ranking(runs, present, "auc"),
        }
    return out


def page_data(spec: dict[str, Any], runs: list[Run]) -> dict[str, Any]:
    primary = spec["metric"]["keys"][0]
    keys = {k for r in runs for k in r["hist"]} - NOT_METRICS - {primary}
    metrics = [primary, *sorted(keys)]
    start = final_start(spec)
    for r in runs:
        r["summary"] = run_summary(r["hist"], metrics, start)
    envs = [e for e in spec["envs"] if any(r["env"] == e for r in runs)]
    groups = spec.get("groups", {"all": spec["envs"]})
    rule = spec.get("gate")
    conds = [*spec["conditions"], *spec.get("controls", [])]
    grid = np.linspace(0, spec["total_steps"], 51)
    return {
        "name": spec["name"],
        "question": spec["question"],
        "protocol": spec["protocol"],
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "tags": [spec["wandb"]["tag"], *spec["wandb"].get("join", [])],
        "conditions": [
            c for c in conds if any(r["condition"] == c for r in runs)
        ],
        "metrics": metrics,
        "final_start": start,
        "total_steps": spec["total_steps"],
        "envs": envs,
        "groups": groups,
        "grid": grid.tolist(),
        "gate_rule": rule,
        "gate": [gate(runs, e, rule, primary) for e in groups["hard"]]
        if rule
        else [],
        "ranking": group_rankings(runs, groups, primary),
        "curves": curves(runs, metrics, grid),
        "runs": [{k: v for k, v in r.items() if k != "hist"} for r in runs],
    }


def main(toml_path: Path) -> None:
    spec = load_spec(toml_path)
    conds = [*spec["conditions"], *spec.get("controls", [])]
    runs = load(cache_dirs(spec), spec["wandb"]["run_name"], spec["protocol"])
    runs = select(runs, conds, spec["envs"])
    data = page_data(spec, runs)
    out = spec["dir"] / "summary.html"
    page = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data))
    out.write_text(page)
    print(f"{len(runs)} runs, {len(data['envs'])} envs -> {out}")
    for name, g in data["ranking"].items():
        print(f"{name}: {len(g['envs'])} envs x {g['seeds']} seeds")
        for row in g["final"]:
            print(
                f"  {row['condition']:<16} IQM {row['iqm']:6.3f}"
                f" [{row['lo']:.3f}, {row['hi']:.3f}]"
            )
    for g in data["gate"]:
        verdict = "PASS" if g["pass"] else "FAIL"
        print(f"gate {g['env']:<14} ref {g['ref_mean']:8.1f}  {verdict}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
