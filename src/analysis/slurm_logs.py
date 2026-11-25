from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

TASK = re.compile(
    r"^Task \d+ \| (?P<cond>\S+) \| (?P<env>\S+) \| seed (?P<seed>\d+)$"
)
DONE = re.compile(
    r"^done \| \S+ \| env_steps=(?P<env_steps>\d+) \| "
    r"rl_steps=(?P<rl_steps>\d+) \| pop_steps=(?P<pop_steps>\d+) \| "
    r"eval=(?P<eval>\S+)$"
)
LOG_NAME = re.compile(r"^slurm-(?P<job>\d+)_(?P<index>\d+)\.out$")


def parse_log(path: Path) -> dict[str, Any] | None:
    """Final row of one slurm_run_array.sh log, None if the run never ended.

    eval is the final evaluation rounded to one decimal by print_summary.
    """
    task = done = None
    for line in path.read_text(errors="replace").splitlines():
        task = task or TASK.match(line)
        done = DONE.match(line) or done
    if task is None:
        raise ValueError(f"{path}: no 'Task N | cond | env | seed S' line")
    if done is None or done["eval"] == "n/a":
        return None
    name = LOG_NAME.match(path.name)
    return {
        "cond": task["cond"],
        "env": task["env"],
        "seed": int(task["seed"]),
        "final": float(done["eval"]),
        "env_steps": int(done["env_steps"]),
        "rl_steps": int(done["rl_steps"]),
        "pop_steps": int(done["pop_steps"]),
        "job": None if name is None else f"{name['job']}_{name['index']}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final evals from SLURM logs, for runs without "
        "perf/eval_final in the wandb summary."
    )
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, unfinished = [], []
    for path in args.logs:
        row = parse_log(path)
        if row is None:
            unfinished.append(path.name)
        else:
            rows.append(row)
    args.out.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} finished runs -> {args.out}")
    if unfinished:
        print(f"{len(unfinished)} logs without a final eval: {unfinished}")


if __name__ == "__main__":
    main()
