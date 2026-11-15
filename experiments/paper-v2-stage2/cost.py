import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHES = {
    "easy": ROOT.parent / "paper-v2" / "cache" / "full",
    "swimmer": ROOT / "cache" / "full",
}
# eval_champion exists only in the metric pilot (seeds 0-2) and step B
CHAMPION = [
    ROOT.parent / "paper-v2-metric-pilot" / "cache" / "pilot",
    ROOT / "cache" / "champion",
]
EASY = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "Ant-v5"]
CONDITIONS = ["td3", "erl", "sc_erl-always", "sc_erl-random", "prefix-h200"]
NAME = re.compile(r"^(?P<condition>.+)_(?P<env>[^_]+)_s(?P<seed>\d+)$")
FINAL_START = 900_000
WARMUP = 10_000
POP_SIZE = 10
N_BOOT = 10_000


def column(hist: dict[str, list[Any]], key: str) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in hist[key]], float)


def run_scores(path: Path) -> dict[str, Any] | None:
    data = json.loads(path.read_text())
    m = NAME.match(data["name"])
    if m is None or m["condition"] not in CONDITIONS:
        return None
    h = data["hist"]
    x = column(h, "env_steps")
    rl = column(h, "perf/eval_rl")
    rf = column(h, "cost/real_frac")
    late = (x >= FINAL_START) & np.isfinite(rl)
    if not late.any():
        raise ValueError(f"{data['name']}: no eval_rl after {FINAL_START}")
    return {
        "condition": m["condition"],
        "env": m["env"],
        "seed": int(m["seed"]),
        "final": float(rl[late].mean()),
        "real_frac": float(np.nanmean(rf[x > WARMUP])),
    }


def iqm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    lo, hi = np.percentile(x, [25, 75], axis=axis, keepdims=True)
    inside = (x >= lo) & (x <= hi)
    return np.sum(np.where(inside, x, 0), axis=axis) / inside.sum(axis=axis)


def normalize(scores: dict[str, dict[str, np.ndarray]]) -> None:
    # per env: 0 = worst condition mean, 1 = best condition mean
    for env, by_cond in scores.items():
        means = [v.mean() for v in by_cond.values()]
        lo, hi = min(means), max(means)
        for c in by_cond:
            by_cond[c] = (by_cond[c] - lo) / (hi - lo)


def boot(
    scores: dict[str, dict[str, np.ndarray]], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    # unpaired: seeds resampled independently per (env, condition)
    out = {}
    for c in CONDITIONS:
        draws = []
        for by_cond in scores.values():
            v = by_cond[c]
            idx = rng.integers(0, len(v), (N_BOOT, len(v)))
            draws.append(v[idx])
        out[c] = np.concatenate(draws, axis=1)
    return out


def prob_greater(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    gt = (a[..., :, None] > b[..., None, :]).mean(axis=(-1, -2))
    eq = (a[..., :, None] == b[..., None, :]).mean(axis=(-1, -2))
    return gt + 0.5 * eq


def group_stats(runs: list[dict[str, Any]], envs: list[str]) -> dict[str, Any]:
    scores: dict[str, dict[str, np.ndarray]] = {}
    for env in envs:
        scores[env] = {
            c: np.array(
                [
                    r["final"]
                    for r in sorted(runs, key=lambda r: r["seed"])
                    if r["env"] == env and r["condition"] == c
                ]
            )
            for c in CONDITIONS
        }
    raw = {e: {c: v.tolist() for c, v in d.items()} for e, d in scores.items()}
    if len(envs) > 1:
        normalize(scores)
    rng = np.random.default_rng(0)
    draws = boot(scores, rng)
    pooled = {
        c: np.concatenate([scores[e][c] for e in envs]) for c in CONDITIONS
    }
    ranking = {
        c: {
            "iqm": float(iqm(pooled[c])),
            "ci": np.percentile(iqm(draws[c]), [2.5, 97.5]).tolist(),
        }
        for c in CONDITIONS
    }
    pairs = {}
    for a in CONDITIONS:
        for b in CONDITIONS:
            if a == b:
                continue
            point = np.mean(
                [prob_greater(scores[e][a], scores[e][b]) for e in envs]
            )
            bs = []
            for e in envs:
                va, vb = scores[e][a], scores[e][b]
                ia = rng.integers(0, len(va), (N_BOOT, len(va)))
                ib = rng.integers(0, len(vb), (N_BOOT, len(vb)))
                bs.append(prob_greater(va[ia], vb[ib]))
            ci = np.percentile(np.mean(bs, axis=0), [2.5, 97.5])
            pairs[f"{a}|{b}"] = {"p": float(point), "ci": ci.tolist()}
    real = {}
    for c in CONDITIONS:
        rf = [
            r["real_frac"]
            for r in runs
            if r["condition"] == c and r["env"] in envs
        ]
        f = float(np.mean(rf))
        real[c] = {
            "real_frac": f,
            "pop_budget": POP_SIZE * f / (1 + POP_SIZE * f),
        }
    n_seeds = {c: int(min(len(scores[e][c]) for e in envs)) for c in CONDITIONS}
    return {
        "envs": envs,
        "n_seeds": n_seeds,
        "ranking": ranking,
        "pairs": pairs,
        "cost": real,
        "raw": raw,
    }


def dominance() -> list[dict[str, Any]]:
    rows = []
    paths = [p for d in CHAMPION for p in sorted(d.glob("*.json"))]
    for path in paths:
        data = json.loads(path.read_text())
        m = NAME.match(data["name"])
        if m is None:
            raise ValueError(f"champion run {data['name']!r} does not parse")
        if m["condition"] not in CONDITIONS:
            continue
        h = data["hist"]
        x = column(h, "env_steps")
        rl = column(h, "perf/eval_rl")
        ch = column(h, "perf/eval_champion")
        late = x >= FINAL_START
        rows.append(
            {
                "condition": m["condition"],
                "env": m["env"],
                "seed": int(m["seed"]),
                "share_all": float(np.mean(ch > rl)),
                "share_final": float(np.mean(ch[late] > rl[late])),
                "n": int(len(rl)),
            }
        )
    return rows


def main(out: Path) -> None:
    runs: list[dict[str, Any]] = []
    for group, cache in CACHES.items():
        for path in sorted(cache.glob("*.json")):
            r = run_scores(path)
            if r is None:
                continue
            if group == "easy" and r["env"] in EASY:
                runs.append(r)
            if group == "swimmer" and r["env"] == "Swimmer-v5":
                runs.append(r)
    data: dict[str, Any] = {
        "conditions": CONDITIONS,
        "easy": group_stats(runs, EASY),
        "swimmer": group_stats(runs, ["Swimmer-v5"]),
        "dominance": dominance(),
    }
    template = (ROOT / "cost_template.html").read_text()
    out.write_text(template.replace("/*DATA*/null", json.dumps(data)))
    for g in ("easy", "swimmer"):
        s = data[g]
        print(f"\n{g} {s['envs']} seeds {s['n_seeds']}")
        for c in CONDITIONS:
            rk, co = s["ranking"][c], s["cost"][c]
            print(
                f"  {c:<14} IQM {rk['iqm']:8.3f} [{rk['ci'][0]:.3f},"
                f" {rk['ci'][1]:.3f}]  real_frac {co['real_frac']:.3f}"
                f"  pop budget {co['pop_budget']:.2f}"
            )
        for c in CONDITIONS[1:]:
            p = s["pairs"][f"{c}|td3"]
            q = s["pairs"][f"{c}|erl"] if c != "erl" else None
            extra = f"  P(>erl) {q['p']:.2f}" if q else ""
            print(f"  P({c} > td3) {p['p']:.2f} {np.round(p['ci'], 2)}{extra}")
    print(f"-> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cost.html")
