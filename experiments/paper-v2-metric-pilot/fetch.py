import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import wandb

KEYS = [
    "env_steps",
    "perf/eval_rl",
    "perf/eval_champion",
    "perf/eval_best",
    "pop/champion_fitness",
]


def fetch(run: Any, cache: Path) -> dict[str, Any]:
    out = cache / f"{run.id}.json"
    if out.exists():
        return json.loads(out.read_text())
    # rows holding every key = eval rows; champion fitness is from that gen
    rows = list(run.scan_history(keys=KEYS, page_size=10_000))
    hist = {k: [r[k] for r in rows] for k in KEYS}
    data = {"id": run.id, "name": run.name, "url": run.url, "hist": hist}
    out.write_text(json.dumps(data))
    return data


def main(toml_path: Path) -> None:
    spec = tomllib.loads(toml_path.read_text())
    wb = spec["wandb"]
    pattern = re.compile(wb["run_name"])
    cache = toml_path.parent / "cache" / "pilot"
    cache.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    filters = {"tags": {"$in": [wb["tag"]]}, "state": "finished"}
    runs = api.runs(f"{wb['entity']}/{wb['project']}", filters=filters)
    names = []
    for run in runs:
        if not pattern.match(run.name):
            raise ValueError(f"run {run.id} name {run.name!r} does not parse")
        n = len(fetch(run, cache)["hist"]["env_steps"])
        names.append(f"{run.name} ({n} evals)")
    print(f"{len(names)} runs")
    print("\n".join(sorted(names)))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
