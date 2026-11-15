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

from analyze import boot_iqm, flat, iqm, normalize

METRICS = (
    "perf/eval_rl",
    "cost/real_frac",
    "p_surr",
    "prefix/h",
    "screen/cem_std",
    "train/critic_loss",
    "train/td_error",
    "train/td_error_ema",
)
GRID = np.linspace(0, 1_000_000, 51)
# out of the spec's conditions (ERL-only analysis) but shown on this page
CONTROLS = ("td3",)
Hist = dict[str, list[float]]


def control_index(api: Any, path: str, wb: dict[str, Any]) -> list[dict]:
    pattern = re.compile(wb["run_name"])
    out = []
    filters = {"tags": wb["tag"], "state": "finished"}
    for run in api.runs(path, filters=filters):
        m = pattern.match(run.name)
        if not m or m["condition"] not in CONTROLS:
            continue
        out.append(
            {
                "id": run.id,
                "name": run.name,
                "url": run.url,
                "condition": m["condition"],
                "env": m["env"],
                "seed": int(m["seed"]),
                "commit": (run.metadata or {}).get("git", {}).get("commit"),
            }
        )
    return out


def fetch_history(run: Any, cache: Path) -> dict[str, Any]:
    path = cache / f"{run.id}.json"
    if path.exists():
        return json.loads(path.read_text())
    keys = ("env_steps", "generation", "_runtime", *METRICS)
    rows = list(run.scan_history(page_size=1000))
    hist: Hist = {k: [] for k in keys}
    for row in rows:
        if row.get("env_steps") is None:
            continue
        for k in keys:
            v = row.get(k)
            hist[k].append(float(v) if isinstance(v, (int, float)) else np.nan)
    out = {"id": run.id, "name": run.name, "url": run.url, "hist": hist}
    path.write_text(json.dumps(out))
    return out


def series(hist: Hist, key: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(hist["env_steps"])
    y = np.asarray(hist[key])
    ok = np.isfinite(y)
    return x[ok], y[ok]


def run_summary(hist: Hist) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in METRICS:
        x, y = series(hist, key)
        if not len(y):
            out[key] = None
            continue
        tail = y[x >= 0.9 * 1_000_000]
        out[key] = float(tail.mean()) if len(tail) else None
    x, y = series(hist, "perf/eval_rl")
    out["auc"] = float(np.trapezoid(y, x) / (x[-1] - x[0]))
    out["max_eval"] = float(y.max())
    _, gens = series(hist, "generation")
    _, rt = series(hist, "_runtime")
    out["generations"] = float(gens.max() + 1) if len(gens) else None
    out["runtime_h"] = float(rt.max() / 3600) if len(rt) else None
    return out


def on_grid(hist: Hist, key: str) -> np.ndarray:
    x, y = series(hist, key)
    if len(y) < 2:
        return np.full(len(GRID), np.nan)
    v = np.interp(GRID, x, y)
    v[(GRID < x[0]) | (GRID > x[-1])] = np.nan
    return v


def curves(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cells: dict[tuple[str, str], list[Hist]] = {}
    for r in runs:
        cells.setdefault((r["env"], r["condition"]), []).append(r["hist"])
    for (env, cond), hists in cells.items():
        for key in METRICS:
            m = np.vstack([on_grid(h, key) for h in hists])
            ok = np.isfinite(m)
            n = ok.sum(axis=0)
            if not n.any():
                continue
            filled = np.where(ok, m, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = filled.sum(axis=0) / n
                sq = np.where(ok, (m - mean) ** 2, 0.0).sum(axis=0)
                ci = 1.96 * np.sqrt(sq / (n - 1)) / np.sqrt(n)
            out.setdefault(env, {}).setdefault(cond, {})[key] = {
                "mean": [
                    None if not np.isfinite(v) else round(v, 4) for v in mean
                ],
                "ci": [None if not np.isfinite(v) else round(v, 4) for v in ci],
            }
    return out


def ranking(
    runs: list[dict[str, Any]], envs: list[str], field: str
) -> list[dict[str, Any]]:
    raw: dict[str, dict[str, dict[int, float]]] = {}
    for r in runs:
        raw.setdefault(r["condition"], {}).setdefault(r["env"], {})
        raw[r["condition"]][r["env"]][r["seed"]] = r["summary"][field]
    norm = normalize(raw, envs)
    rows = []
    for c in norm:
        draws = boot_iqm(norm, c, envs, np.random.default_rng(0))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append(
            {
                "condition": c,
                "iqm": iqm(flat(norm, c, envs)),
                "lo": float(lo),
                "hi": float(hi),
            }
        )
    return sorted(rows, key=lambda r: -r["iqm"])


def main(toml_path: Path) -> None:
    spec = tomllib.loads(toml_path.read_text())
    wb = spec["wandb"]
    cache = toml_path.parent / "cache" / "full"
    cache.mkdir(parents=True, exist_ok=True)
    index = [
        json.loads(p.read_text())
        for p in sorted((toml_path.parent / "cache" / "runs").glob("*.json"))
    ]
    index = [r for r in index if r["condition"] in spec["conditions"]]
    api = wandb.Api(timeout=60)
    path = f"{wb['entity']}/{wb['project']}"
    index += control_index(api, path, wb)
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
                "name": meta["name"],
                "url": meta["url"],
                "condition": meta["condition"],
                "env": meta["env"],
                "seed": meta["seed"],
                "commit": (meta.get("commit") or "")[:7],
                "hist": full["hist"],
                "summary": run_summary(full["hist"]),
            }
        )
    envs = [e for e in spec["envs"] if any(r["env"] == e for r in runs)]
    data = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "tags": [wb["tag"], *wb.get("join", [])],
        "conditions": [*spec["conditions"], *CONTROLS],
        "envs": envs,
        "grid": GRID.tolist(),
        "ranking": {
            "final": ranking(runs, envs, "perf/eval_rl"),
            "auc": ranking(runs, envs, "auc"),
        },
        "curves": curves(runs),
        "runs": [{k: v for k, v in r.items() if k != "hist"} for r in runs],
    }
    template = (toml_path.parent / "summary_template.html").read_text()
    out = toml_path.parent / "summary.html"
    out.write_text(template.replace("/*DATA*/null", json.dumps(data)))
    print(f"{len(runs)} runs, {len(envs)} envs -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
