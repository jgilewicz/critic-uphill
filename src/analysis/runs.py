import json
import re
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from analysis.stats import Cells

Hist = dict[str, np.ndarray]
Run = dict[str, Any]
# runs logged before the protocol key existed (paper-core-v1, paper-v2)
LEGACY_PROTOCOL = 2


def load_spec(toml_path: Path) -> dict[str, Any]:
    spec = tomllib.loads(toml_path.read_text())
    spec["dir"] = toml_path.resolve().parent
    spec["protocol"] = spec.get("protocol", LEGACY_PROTOCOL)
    return spec


def cache_dirs(spec: dict[str, Any], key: str = "caches") -> list[Path]:
    names = spec.get("analysis", {}).get(key, ["cache/full"])
    return [(spec["dir"] / n).resolve() for n in names]


def column(hist: dict[str, list[Any]], key: str) -> np.ndarray:
    n = len(hist["env_steps"])
    if key not in hist:
        return np.full(n, np.nan)
    return np.array([np.nan if v is None else v for v in hist[key]], float)


def load(dirs: list[Path], pattern: str, protocol: int) -> list[Run]:
    # a run id cached in several dirs comes from the first dir listed
    name = re.compile(pattern)
    seen: dict[str, Run] = {}
    for d in dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"cache dir {d} missing. run fetch first")
        for path in sorted(d.glob("*.json")):
            data = json.loads(path.read_text())
            if data["id"] in seen:
                continue
            m = name.match(data["name"])
            if m is None:
                raise ValueError(f"{path}: run name {data['name']!r} unparsed")
            if data.get("protocol", LEGACY_PROTOCOL) != protocol:
                continue
            raw = data["hist"]
            seen[data["id"]] = {
                "name": data["name"],
                "url": data.get("url"),
                "commit": (data.get("commit") or "")[:7],
                "condition": m["condition"],
                "env": m["env"],
                "seed": int(m["seed"]),
                "hist": {k: column(raw, k) for k in raw},
            }
    return sorted(
        seen.values(), key=lambda r: (r["condition"], r["env"], r["seed"])
    )


def select(
    runs: list[Run], conditions: list[str], envs: list[str]
) -> list[Run]:
    return [
        r for r in runs if r["condition"] in conditions and r["env"] in envs
    ]


def series(hist: Hist, key: str) -> tuple[np.ndarray, np.ndarray]:
    x = hist["env_steps"]
    y = hist.get(key, np.full(len(x), np.nan))
    ok = np.isfinite(y)
    return x[ok], y[ok]


def final(hist: Hist, key: str, start: float) -> float:
    x, y = series(hist, key)
    late = x >= start
    if not late.any():
        raise ValueError(f"no finite {key} after {start:.0f} steps")
    return float(y[late].mean())


def auc(hist: Hist, key: str) -> float:
    x, y = series(hist, key)
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def final_start(spec: dict[str, Any]) -> float:
    return (1 - spec["metric"]["final_window"]) * spec["total_steps"]


def cells(
    runs: list[Run], envs: list[str], conds: list[str], score: Any
) -> Cells:
    out: Cells = {}
    for env in envs:
        out[env] = {}
        for c in conds:
            rs = sorted(
                (r for r in runs if r["env"] == env and r["condition"] == c),
                key=lambda r: r["seed"],
            )
            if not rs:
                raise ValueError(f"no runs for {c} on {env}")
            out[env][c] = np.array([score(r) for r in rs])
    return out


def complete_envs(
    runs: list[Run], envs: list[str], conds: list[str]
) -> list[str]:
    have = {(r["env"], r["condition"]) for r in runs}
    return [e for e in envs if all((e, c) in have for c in conds)]
