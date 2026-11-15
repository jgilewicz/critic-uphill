import json
import re
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import wandb

# reuse stage-1 fetch, per-run summary and curve code unchanged
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper-v2"))
from summary import GRID, curves, fetch_history, ranking, run_summary  # noqa: E402

# same hue per condition as the stage-1 summary page
COLOR = {"prefix-h200": 0, "sc_erl-always": 1, "erl": 2, "sc_erl-random": 3}
COLOR["td3"] = 6


def run_index(api: Any, path: str, spec: dict[str, Any]) -> list[dict]:
    wb = spec["wandb"]
    pattern = re.compile(wb["run_name"])
    tags = [wb["tag"], *wb.get("join", [])]
    filters = {"tags": {"$in": tags}, "state": "finished"}
    found: dict[tuple[str, str, int], Any] = {}
    for run in api.runs(path, filters=filters):
        m = pattern.match(run.name)
        if not m or m["condition"] not in spec["conditions"]:
            continue
        if m["env"] not in spec["envs"]:
            continue
        key = (m["condition"], m["env"], int(m["seed"]))
        # duplicates resolve to the newest finished run
        if key in found and found[key].created_at > run.created_at:
            continue
        found[key] = run
    return [
        {
            "id": run.id,
            "name": run.name,
            "url": run.url,
            "condition": c,
            "env": e,
            "seed": s,
            "commit": (run.metadata or {}).get("git", {}).get("commit"),
        }
        for (c, e, s), run in sorted(found.items())
    ]


def gate(
    runs: list[dict[str, Any]], env: str, rule: dict[str, Any]
) -> dict[str, Any]:
    final: dict[str, dict[int, float]] = {}
    for r in runs:
        if r["env"] == env and r["seed"] in rule["seeds"]:
            final.setdefault(r["condition"], {})[r["seed"]] = r["summary"][
                "perf/eval_rl"
            ]
    ref = final[rule["reference"]]
    ref_mean = float(np.mean(list(ref.values())))
    variants = []
    for cond, by_seed in final.items():
        if cond == rule["reference"]:
            continue
        m = float(np.mean(list(by_seed.values())))
        wins = sum(by_seed[s] > ref[s] for s in by_seed.keys() & ref.keys())
        gain = (m - ref_mean) / abs(ref_mean)
        variants.append(
            {
                "condition": cond,
                "mean": m,
                "gain": gain,
                "wins": int(wins),
                "n": len(by_seed.keys() & ref.keys()),
                "pass": bool(
                    gain >= rule["min_rel_gain"] and wins >= rule["min_wins"]
                ),
            }
        )
    variants.sort(key=lambda v: -v["mean"])
    return {
        "env": env,
        "ref_mean": ref_mean,
        "variants": variants,
        "pass": any(v["pass"] for v in variants),
    }


def group_rankings(
    runs: list[dict[str, Any]], groups: dict[str, list[str]]
) -> dict[str, Any]:
    out = {}
    for name, envs in groups.items():
        present = [e for e in envs if any(r["env"] == e for r in runs)]
        out[name] = {
            "envs": present,
            "seeds": len({r["seed"] for r in runs if r["env"] in present}),
            "final": ranking(runs, present, "perf/eval_rl"),
            "auc": ranking(runs, present, "auc"),
        }
    return out


def main(toml_path: Path) -> None:
    spec = tomllib.loads(toml_path.read_text())
    wb = spec["wandb"]
    cache = toml_path.parent / "cache" / "full"
    cache.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    path = f"{wb['entity']}/{wb['project']}"
    index = run_index(api, path, spec)
    with ThreadPoolExecutor(8) as pool:
        fulls = list(
            pool.map(
                lambda r: fetch_history(api.run(f"{path}/{r['id']}"), cache),
                index,
            )
        )
    runs = []
    for meta, full in zip(index, fulls, strict=True):
        runs.append(
            {
                **{k: v for k, v in meta.items() if k != "id"},
                "commit": (meta.get("commit") or "")[:7],
                "hist": full["hist"],
                "summary": run_summary(full["hist"]),
            }
        )
    groups = spec["groups"]
    envs = [e for e in spec["envs"] if any(r["env"] == e for r in runs)]
    data = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "tags": [wb["tag"], *wb.get("join", [])],
        "conditions": spec["conditions"],
        "color": COLOR,
        "envs": envs,
        "groups": groups,
        "grid": GRID.tolist(),
        "gate_rule": spec["gate"],
        "gate": [gate(runs, e, spec["gate"]) for e in groups["hard"]],
        "ranking": group_rankings(runs, groups),
        "curves": curves(runs),
        "runs": [{k: v for k, v in r.items() if k != "hist"} for r in runs],
    }
    template = (toml_path.parent / "summary_template.html").read_text()
    out = toml_path.parent / "summary.html"
    out.write_text(template.replace("/*DATA*/null", json.dumps(data)))
    print(f"{len(runs)} runs, {len(envs)} envs -> {out}")
    for g in data["gate"]:
        best = g["variants"][0]
        print(
            f"{g['env']:<14} td3 {g['ref_mean']:8.1f}  best {best['condition']}"
            f" {best['mean']:8.1f} ({best['gain']:+.0%}, {best['wins']}/"
            f"{best['n']})  gate {'PASS' if g['pass'] else 'FAIL'}"
        )
        for v in g["variants"]:
            print(
                f"    {v['condition']:<14} {v['mean']:8.1f} {v['gain']:+7.0%}"
                f"  wins {v['wins']}/{v['n']}  {'PASS' if v['pass'] else ''}"
            )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
