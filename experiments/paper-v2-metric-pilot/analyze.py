import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
# td3 and pre-pilot cells (same seeds, eval_rl only) from earlier experiments
OLD_CACHES = {
    "Swimmer-v5": ROOT.parent / "paper-v2-stage2" / "cache" / "full",
    "Ant-v5": ROOT.parent / "paper-v2" / "cache" / "full",
}
GRID = np.linspace(0, 1_000_000, 51)


def load(cache: Path, pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    runs = []
    for f in sorted(cache.glob("*.json")):
        data = json.loads(f.read_text())
        m = pattern.match(data["name"])
        if m is None:
            continue
        hist = {
            k: np.array([np.nan if v is None else v for v in vals], float)
            for k, vals in data["hist"].items()
        }
        runs.append({**m.groupdict(), "seed": int(m["seed"]), "hist": hist})
    return runs


def final(hist: dict[str, np.ndarray], key: str, start: float) -> float:
    x = hist["env_steps"]
    y = hist[key]
    mask = (x >= start) & np.isfinite(y)
    if not mask.any():
        raise ValueError(f"no finite {key} after {start} steps")
    return float(y[mask].mean())


def on_grid(hist: dict[str, np.ndarray], key: str) -> list[float]:
    x = hist["env_steps"]
    y = hist[key]
    ok = np.isfinite(y)
    return np.interp(GRID, x[ok], y[ok], left=np.nan).round(2).tolist()


def pilot_cell(run: dict[str, Any], start: float) -> dict[str, Any]:
    h = run["hist"]
    rl, ch = h["perf/eval_rl"], h["perf/eval_champion"]
    late = h["env_steps"] >= start
    fit, ev = h["pop/champion_fitness"], h["perf/eval_champion"]
    rho = spearmanr(fit, ev).statistic
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
            k: on_grid(h, f"perf/eval_{k}") for k in ("rl", "champion", "best")
        }
        | {"fitness": on_grid(h, "pop/champion_fitness")},
    }


def main(toml_path: Path) -> None:
    spec = tomllib.loads(toml_path.read_text())
    pattern = re.compile(spec["wandb"]["run_name"])
    start = (1 - spec["metric"]["final_window"]) * spec["total_steps"]
    seeds = set(spec["seeds"])
    cells: dict[str, Any] = {}
    for run in load(ROOT / "cache" / "pilot", pattern):
        key = f"{run['env']}|{run['condition']}|{run['seed']}"
        cells[key] = pilot_cell(run, start)
    old: dict[str, Any] = {}
    for env, cache in OLD_CACHES.items():
        for run in load(cache, pattern):
            if run["env"] != env or run["seed"] not in seeds:
                continue
            cond = run["condition"]
            if cond not in [*spec["conditions"], *spec["controls"]]:
                continue
            old[f"{env}|{cond}|{run['seed']}"] = {
                "rl": final(run["hist"], "perf/eval_rl", start),
                "curve": on_grid(run["hist"], "perf/eval_rl"),
            }
    rows = summarize(spec, cells, old)
    data = {
        "grid": GRID.tolist(),
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
    template = (ROOT / "report_template.html").read_text()
    out = ROOT / "report.html"
    out.write_text(template.replace("/*DATA*/null", json.dumps(data)))
    print_summary(rows)
    print(f"-> {out}")


def verdict(rel: float, wins: int, rule: dict[str, Any]) -> str:
    if rel >= rule["min_rel_gain"] and wins >= rule["min_wins"]:
        return "clearly_above"
    if abs(rel) < rule["min_rel_gain"]:
        return "about_equal"
    return "clearly_below" if rel < 0 and wins == 0 else "mixed"


def summarize(
    spec: dict[str, Any], cells: dict[str, Any], old: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for env in spec["envs"]:
        td3 = {s: old[f"{env}|td3|{s}"]["rl"] for s in spec["seeds"]}
        for cond in spec["conditions"]:
            cs = {s: cells[f"{env}|{cond}|{s}"] for s in spec["seeds"]}
            mean = {
                k: float(np.mean([c[k] for c in cs.values()]))
                for k in next(iter(cs.values()))
                if k != "curves"
            }
            wins = sum(c["champion"] > c["rl"] for c in cs.values())
            rel = (mean["champion"] - mean["rl"]) / abs(mean["rl"])
            rows.append(
                {
                    "env": env,
                    "condition": cond,
                    **mean,
                    "old_rl": float(
                        np.mean([old[f"{env}|{cond}|{s}"]["rl"] for s in cs])
                    ),
                    "td3": float(np.mean(list(td3.values()))),
                    "rel": rel,
                    "wins": int(wins),
                    "verdict": verdict(rel, wins, spec["decision"]),
                    "beats_td3_rl": sum(cs[s]["rl"] > td3[s] for s in cs),
                    "beats_td3_champion": sum(
                        cs[s]["champion"] > td3[s] for s in cs
                    ),
                    "seeds": {
                        s: {k: v for k, v in c.items() if k != "curves"}
                        | {
                            "td3": td3[s],
                            "old_rl": old[f"{env}|{cond}|{s}"]["rl"],
                        }
                        for s, c in cs.items()
                    },
                }
            )
    return rows


def print_summary(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        print(
            f"{r['env']:<11} {r['condition']:<14} rl {r['rl']:8.1f}"
            f" (old {r['old_rl']:7.1f}) champ {r['champion']:8.1f}"
            f" best {r['best']:8.1f} fit {r['fitness']:8.1f} td3 {r['td3']:7.1f}"
            f" diff {r['rel']:+.0%} {r['wins']}/3 {r['verdict']}"
            f" win% {r['win_all']:.0%}/{r['win_final']:.0%}"
            f" rho {r['rho_fit_eval']:.2f}"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
