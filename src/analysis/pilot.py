import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from analysis.runs import (
    Hist,
    Run,
    cache_dirs,
    final,
    final_start,
    load,
    load_spec,
    series,
)

TEMPLATE = Path(__file__).resolve().parent / "templates" / "pilot.html"


def on_grid(hist: Hist, key: str, grid: np.ndarray) -> list[float]:
    x, y = series(hist, key)
    return np.interp(grid, x, y, left=np.nan).round(2).tolist()


def pilot_cell(run: Run, start: float, grid: np.ndarray) -> dict[str, Any]:
    h = run["hist"]
    rl, ch = h["perf/eval_rl"], h["perf/eval_champion"]
    late = h["env_steps"] >= start
    rho = spearmanr(h["pop/champion_fitness"], ch).statistic
    return {
        "rl": final(h, "perf/eval_rl", start),
        "champion": final(h, "perf/eval_champion", start),
        "best": final(h, "perf/eval_best", start),
        "fitness": final(h, "pop/champion_fitness", start),
        "win_all": float(np.mean(ch > rl)),
        "win_final": float(np.mean(ch[late] > rl[late])),
        "n_evals": int(len(rl)),
        "rho_fit_eval": float(rho),
        "curves": {
            k: on_grid(h, f"perf/eval_{k}", grid)
            for k in ("rl", "champion", "best")
        }
        | {"fitness": on_grid(h, "pop/champion_fitness", grid)},
    }


def verdict(rel: float, wins: int, rule: dict[str, Any]) -> str:
    if rel >= rule["min_rel_gain"] and wins >= rule["min_wins"]:
        return "clearly_above"
    if abs(rel) < rule["min_rel_gain"]:
        return "about_equal"
    return "clearly_below" if rel < 0 and wins == 0 else "mixed"


def summary_row(
    spec: dict[str, Any], env: str, cond: str, cells: dict, old: dict
) -> dict[str, Any]:
    seeds = spec["seeds"]
    cs = {s: cells[f"{env}|{cond}|{s}"] for s in seeds}
    td3 = {s: old[f"{env}|td3|{s}"]["rl"] for s in seeds}
    old_rl = {s: old[f"{env}|{cond}|{s}"]["rl"] for s in seeds}
    first = next(iter(cs.values()))
    mean = {
        k: float(np.mean([c[k] for c in cs.values()]))
        for k in first
        if k != "curves"
    }
    wins = sum(c["champion"] > c["rl"] for c in cs.values())
    rel = (mean["champion"] - mean["rl"]) / abs(mean["rl"])
    return {
        "env": env,
        "condition": cond,
        **mean,
        "old_rl": float(np.mean(list(old_rl.values()))),
        "td3": float(np.mean(list(td3.values()))),
        "rel": rel,
        "wins": int(wins),
        "verdict": verdict(rel, wins, spec["decision"]),
        "beats_td3_rl": sum(cs[s]["rl"] > td3[s] for s in cs),
        "beats_td3_champion": sum(cs[s]["champion"] > td3[s] for s in cs),
        "seeds": {
            s: {k: v for k, v in c.items() if k != "curves"}
            | {"td3": td3[s], "old_rl": old_rl[s]}
            for s, c in cs.items()
        },
    }


def print_summary(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        print(
            f"{r['env']:<11} {r['condition']:<14} rl {r['rl']:8.1f}"
            f" (old {r['old_rl']:7.1f}) champ {r['champion']:8.1f}"
            f" best {r['best']:8.1f} fit {r['fitness']:8.1f}"
            f" td3 {r['td3']:7.1f} diff {r['rel']:+.0%} {r['wins']}/3"
            f" {r['verdict']} win% {r['win_all']:.0%}/{r['win_final']:.0%}"
            f" rho {r['rho_fit_eval']:.2f}"
        )


def main(toml_path: Path) -> None:
    spec = load_spec(toml_path)
    pattern, protocol = spec["wandb"]["run_name"], spec["protocol"]
    start = final_start(spec)
    grid = np.linspace(0, spec["total_steps"], 51)
    cells = {
        f"{r['env']}|{r['condition']}|{r['seed']}": pilot_cell(r, start, grid)
        for r in load(cache_dirs(spec), pattern, protocol)
    }
    # td3 and pre-pilot runs of the same cells, from earlier experiments
    old = {}
    keep = [*spec["conditions"], *spec["controls"]]
    for r in load(cache_dirs(spec, "reference_caches"), pattern, protocol):
        if r["env"] in spec["envs"] and r["seed"] in spec["seeds"]:
            if r["condition"] in keep:
                old[f"{r['env']}|{r['condition']}|{r['seed']}"] = {
                    "rl": final(r["hist"], "perf/eval_rl", start),
                    "curve": on_grid(r["hist"], "perf/eval_rl", grid),
                }
    rows = [
        summary_row(spec, env, cond, cells, old)
        for env in spec["envs"]
        for cond in spec["conditions"]
    ]
    data = {
        "grid": grid.tolist(),
        "envs": spec["envs"],
        "conditions": spec["conditions"],
        "seeds": spec["seeds"],
        "rule": spec["decision"],
        "rows": rows,
        "curves": {k: v["curves"] for k, v in cells.items()},
        "td3_curves": {
            k: v["curve"] for k, v in old.items() if k.split("|")[1] == "td3"
        },
    }
    out = spec["dir"] / "report.html"
    out.write_text(
        TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data))
    )
    print_summary(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
