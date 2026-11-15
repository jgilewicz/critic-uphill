import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import wandb

from cost import CONDITIONS, FINAL_START, NAME, iqm, run_scores

ROOT = Path(__file__).resolve().parent
ENV = "Swimmer-v5"
CONFIG_TAG = "paper-v2-config"
KEYS = ["env_steps", "perf/eval_rl", "perf/eval_champion"]
N_BOOT = 10_000


def fetch_champion(run: Any, cache: Path) -> dict[str, Any]:
    out = cache / f"{run.id}.json"
    if out.exists():
        return json.loads(out.read_text())
    rows = list(run.scan_history(keys=KEYS, page_size=10_000))
    hist = {k: [r[k] for r in rows] for k in KEYS}
    data = {"id": run.id, "name": run.name, "url": run.url, "hist": hist}
    out.write_text(json.dumps(data))
    return data


def champion_runs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    wb = spec["wandb"]
    cache = ROOT / "cache" / "champion"
    cache.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    filters = {
        "tags": {"$in": [wb["tag"], CONFIG_TAG]},
        "state": "finished",
        "display_name": {"$regex": f"_{ENV}_s"},
    }
    rows = []
    for run in api.runs(f"{wb['entity']}/{wb['project']}", filters=filters):
        if "perf/eval_champion" not in run.summary:
            continue
        m = NAME.match(run.name)
        if m is None:
            raise ValueError(f"run {run.name!r} does not parse")
        h = fetch_champion(run, cache)["hist"]
        x = np.array(h["env_steps"], float)
        rl = np.array(h["perf/eval_rl"], float)
        ch = np.array(h["perf/eval_champion"], float)
        late = x >= FINAL_START
        rows.append(
            {
                "condition": m["condition"],
                "seed": int(m["seed"]),
                "rl": float(rl[late].mean()),
                "champion": float(ch[late].mean()),
                "share_all": float(np.mean(ch > rl)),
                "share_final": float(np.mean(ch[late] > rl[late])),
            }
        )
    return rows


def iqm_diff(
    a: np.ndarray, b: np.ndarray, level: float, rng: np.random.Generator
) -> dict[str, Any]:
    ia = rng.integers(0, len(a), (N_BOOT, len(a)))
    ib = rng.integers(0, len(b), (N_BOOT, len(b)))
    d = iqm(a[ia]) - iqm(b[ib])
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(d, [tail, 100 - tail])
    return {
        "diff": float(iqm(a) - iqm(b)),
        "ci": [float(lo), float(hi)],
        "pass": bool(lo > 0),
    }


def hypotheses(scores: dict[str, np.ndarray]) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    variants = [c for c in CONDITIONS if c != "td3"]
    h6 = {
        c: iqm_diff(scores[c], scores["td3"], 1 - 0.05 / 4, rng)
        for c in variants
    }
    cheap = ["prefix-h200", "sc_erl-random", "sc_erl-always"]
    h7 = {
        c: iqm_diff(scores[c], scores["erl"], 1 - 0.05 / 3, rng) for c in cheap
    }
    return {"H6": h6, "H7": h7, "H6_pass": any(v["pass"] for v in h6.values())}


def main(toml_path: Path) -> None:
    spec = tomllib.loads(toml_path.read_text())
    runs = [run_scores(p) for p in sorted((ROOT / "cache" / "full").glob("*"))]
    swim = [r for r in runs if r is not None and r["env"] == ENV]
    scores = {
        c: np.array([r["final"] for r in swim if r["condition"] == c])
        for c in CONDITIONS
    }
    seeds = {
        c: sorted(r["seed"] for r in swim if r["condition"] == c)
        for c in CONDITIONS
    }
    for c in CONDITIONS:
        v = scores[c]
        print(
            f"{c:<14} n={len(v):2d} seeds {seeds[c]}  IQM {iqm(v):6.1f}"
            f"  mean {v.mean():6.1f}  min {v.min():6.1f}  max {v.max():6.1f}"
        )
    hyp = hypotheses(scores)
    for h in ("H6", "H7"):
        ref = "td3" if h == "H6" else "erl"
        for c, r in hyp[h].items():
            lo, hi = r["ci"]
            verdict = "PASS" if r["pass"] else "-"
            print(
                f"{h} {c:<14} - {ref}: {r['diff']:+7.1f} [{lo:+7.1f}, {hi:+7.1f}]"
                f"  {verdict}"
            )
    champ = champion_runs(spec)
    for r in sorted(champ, key=lambda r: (r["condition"], r["seed"])):
        print(
            f"champ {r['condition']:<14} s{r['seed']}  rl {r['rl']:6.1f}"
            f"  champion {r['champion']:6.1f}  share {r['share_all']:.2f}"
            f" / {r['share_final']:.2f}"
        )
    out = {"scores": {c: v.tolist() for c, v in scores.items()}, **hyp}
    out["champion"] = champ
    (ROOT / "cache" / "stepb.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
