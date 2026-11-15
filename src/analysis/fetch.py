import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import wandb

from analysis.runs import LEGACY_PROTOCOL, cache_dirs, load_spec


def run_index(api: Any, spec: dict[str, Any]) -> list[Any]:
    wb = spec["wandb"]
    pattern = re.compile(wb["run_name"])
    tags = [wb["tag"], *wb.get("join", [])]
    filters = {"tags": {"$in": tags}, "state": "finished"}
    conditions = [*spec["conditions"], *spec.get("controls", [])]
    found: dict[tuple[str, str, int], Any] = {}
    path = f"{wb['entity']}/{wb['project']}"
    for run in api.runs(path, filters=filters):
        m = pattern.match(run.name)
        if m is None:
            raise ValueError(f"run {run.id} name {run.name!r} does not parse")
        if m["condition"] not in conditions or m["env"] not in spec["envs"]:
            continue
        key = (m["condition"], m["env"], int(m["seed"]))
        # duplicates resolve to the newest finished run
        if key in found and found[key].created_at > run.created_at:
            continue
        found[key] = run
    return [found[k] for k in sorted(found)]


def fetch_history(run: Any, cache: Path) -> dict[str, Any]:
    path = cache / f"{run.id}.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = [
        r
        for r in run.scan_history(page_size=1000)
        if r.get("env_steps") is not None
    ]
    keys = sorted({k for r in rows for k, v in r.items() if _numeric(v)})
    hist = {k: [_value(r.get(k)) for r in rows] for k in keys}
    data = {
        "id": run.id,
        "name": run.name,
        "url": run.url,
        "protocol": run.config.get("protocol", LEGACY_PROTOCOL),
        "commit": (run.metadata or {}).get("git", {}).get("commit"),
        "hist": hist,
    }
    path.write_text(json.dumps(data))
    return data


def _numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _value(v: Any) -> float | None:
    return float(v) if _numeric(v) else None


def main(toml_path: Path) -> None:
    spec = load_spec(toml_path)
    cache = cache_dirs(spec)[0]
    cache.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    index = run_index(api, spec)
    with ThreadPoolExecutor(8) as pool:
        fetched = list(pool.map(lambda r: fetch_history(r, cache), index))
    protocols = sorted({d["protocol"] for d in fetched})
    print(f"{len(fetched)} runs, protocols {protocols} -> {cache}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
