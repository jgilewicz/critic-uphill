import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from analysis.runs import (
    Run,
    cache_dirs,
    cells,
    final,
    final_start,
    load,
    load_spec,
    select,
    series,
)
from analysis.stats import (
    boot_iqm,
    ci,
    iqm,
    normalize,
    pooled,
    prob_greater_ci,
)

TEMPLATE = Path(__file__).resolve().parent / "templates" / "cost.html"


def real_frac(run: Run, warmup: int) -> float:
    x, y = series(run["hist"], "cost/real_frac")
    return float(y[x > warmup].mean())


def cost_rows(
    runs: list[Run], envs: list[str], conds: list[str], cfg: dict[str, Any]
) -> dict[str, dict[str, float]]:
    out = {}
    for c in conds:
        rf = [
            real_frac(r, cfg["warmup"])
            for r in runs
            if r["condition"] == c and r["env"] in envs
        ]
        f = float(np.mean(rf))
        n = cfg["pop_size"]
        out[c] = {"real_frac": f, "pop_budget": n * f / (1 + n * f)}
    return out


def group_stats(
    spec: dict[str, Any], runs: list[Run], envs: list[str]
) -> dict[str, Any]:
    conds = spec["conditions"]
    key = spec["metric"]["keys"][0]
    start = final_start(spec)
    raw = cells(runs, envs, conds, lambda r: final(r["hist"], key, start))
    s = normalize(raw) if len(envs) > 1 else raw
    rng = np.random.default_rng(0)
    draws = boot_iqm(s, conds, rng)
    ranking = {
        c: {"iqm": float(iqm(pooled(s, c))), "ci": ci(draws[c])} for c in conds
    }
    pairs = {
        f"{a}|{b}": prob_greater_ci(s, a, b, rng)
        for a in conds
        for b in conds
        if a != b
    }
    return {
        "envs": envs,
        "n_seeds": {c: int(min(len(s[e][c]) for e in envs)) for c in conds},
        "ranking": ranking,
        "pairs": pairs,
        "cost": cost_rows(runs, envs, conds, spec["cost"]),
        "raw": {
            e: {c: v.tolist() for c, v in d.items()} for e, d in raw.items()
        },
    }


def dominance(runs: list[Run], start: float) -> list[dict[str, Any]]:
    # share of eval points where the population champion beats the RL actor
    rows = []
    for r in runs:
        h = r["hist"]
        rl, ch = h["perf/eval_rl"], h["perf/eval_champion"]
        late = h["env_steps"] >= start
        rows.append(
            {
                "condition": r["condition"],
                "env": r["env"],
                "seed": r["seed"],
                "share_all": float(np.mean(ch > rl)),
                "share_final": float(np.mean(ch[late] > rl[late])),
                "n": int(len(rl)),
            }
        )
    return rows


def print_group(name: str, g: dict[str, Any], conds: list[str]) -> None:
    print(f"\n{name} {g['envs']} seeds {g['n_seeds']}")
    for c in conds:
        rk, co = g["ranking"][c], g["cost"][c]
        print(
            f"  {c:<14} IQM {rk['iqm']:8.3f} [{rk['ci'][0]:.3f},"
            f" {rk['ci'][1]:.3f}]  real_frac {co['real_frac']:.3f}"
            f"  pop budget {co['pop_budget']:.2f}"
        )
    ref = conds[0]
    for c in conds[1:]:
        p = g["pairs"][f"{c}|{ref}"]
        print(f"  P({c} > {ref}) {p['p']:.2f} {np.round(p['ci'], 2)}")


def main(toml_path: Path) -> None:
    spec = load_spec(toml_path)
    pattern, protocol = spec["wandb"]["run_name"], spec["protocol"]
    conds = spec["conditions"]
    runs = select(
        load(cache_dirs(spec), pattern, protocol), conds, spec["envs"]
    )
    champion = load(cache_dirs(spec, "champion_caches"), pattern, protocol)
    data: dict[str, Any] = {"conditions": conds}
    for name, envs in spec["cost"]["groups"].items():
        data[name] = group_stats(spec, runs, envs)
        print_group(name, data[name], conds)
    start = final_start(spec)
    data["dominance"] = dominance(select(champion, conds, spec["envs"]), start)
    out = spec["dir"] / "cost.html"
    out.write_text(
        TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data))
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
