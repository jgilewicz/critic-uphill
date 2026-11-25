from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import wandb

from analysis.records import RUN_NAME
from analysis.spec import SuiteSpec

FINAL_KEY = "perf/eval_final"
# (cond, env, seed)
RunKey = tuple[str, str, int]


def slurm_job(run: Any) -> str | None:
    """<array job>_<task>, the name of the run's slurm_run_array.sh log."""
    slurm = (run.metadata or {}).get("slurm")
    if slurm is None or "array_job_id" not in slurm:
        return None
    return f"{slurm['array_job_id']}_{slurm['array_task_id']}"


def parse_name(name: str) -> RunKey:
    m = RUN_NAME.match(name)
    if m is None:
        raise ValueError(f"run name {name!r} is not <cond>_<env>_s<seed>")
    return m["cond"], m["env"], int(m["seed"])


class RunCache:
    """Finished wandb runs of a suite, one JSON per run name.

    Each file holds the run's id, url, tags, SLURM array job, final eval
    from the summary (None for runs logged before perf/eval_final existed)
    and the history of every spec history key against env_steps. Duplicate
    names resolve to the run whose tag comes first in the spec, then to the
    newest.
    """

    def __init__(self, spec: SuiteSpec, directory: Path) -> None:
        self._spec = spec
        self._dir = directory

    def _index(self, api: Any) -> dict[str, Any]:
        found: dict[str, Any] = {}
        filters = {"tags": {"$in": list(self._spec.tags)}, "state": "finished"}
        path = f"{self._spec.entity}/{self._spec.project}"
        for run in api.runs(path, filters=filters):
            parse_name(run.name)
            old = found.get(run.name)
            if old is None or self._preferred(run, old):
                found[run.name] = run
        return found

    def _tag_rank(self, run: Any) -> int:
        return min(
            self._spec.tags.index(t) for t in run.tags if t in self._spec.tags
        )

    def _preferred(self, run: Any, old: Any) -> bool:
        if self._tag_rank(run) != self._tag_rank(old):
            return self._tag_rank(run) < self._tag_rank(old)
        return run.created_at > old.created_at

    def _fetch_one(self, run: Any) -> None:
        path = self._dir / f"{run.name}.json"
        if path.exists() and json.loads(path.read_text())["id"] == run.id:
            return
        series = {}
        for key in self._spec.history_keys:
            rows = run.scan_history(keys=["env_steps", key], page_size=5000)
            pts = [(r["env_steps"], r.get(key)) for r in rows]
            series[key] = {"x": [p[0] for p in pts], "y": [p[1] for p in pts]}
        data = {
            "id": run.id,
            "name": run.name,
            "url": run.url,
            "tags": list(run.tags),
            "slurm_job": slurm_job(run),
            "final": run.summary.get(FINAL_KEY),
            "series": series,
        }
        path.write_text(json.dumps(data))

    def fetch(self) -> int:
        self._dir.mkdir(parents=True, exist_ok=True)
        runs = self._index(wandb.Api(timeout=120))
        with ThreadPoolExecutor(16) as pool:
            list(pool.map(self._fetch_one, runs.values()))
        return len(runs)

    def load(self) -> dict[RunKey, dict[str, Any]]:
        paths = sorted(self._dir.glob("*.json"))
        if not paths:
            raise FileNotFoundError(
                f"no cached runs in {self._dir}, run analysis.wandb_runs first"
            )
        out = {}
        for path in paths:
            d = json.loads(path.read_text())
            out[parse_name(d["name"])] = d
        return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache finished wandb runs of a suite spec."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    cache = RunCache(SuiteSpec.load(args.spec), args.cache)
    print(f"cached {cache.fetch()} finished runs in {args.cache}")


if __name__ == "__main__":
    main()
