import itertools
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from scipy.stats import t as student_t

from analysis.runs import cache_dirs, load, load_spec, select, series

TEMPLATE = Path(__file__).parent / "templates" / "local_search.html"
METRIC = "perf/eval_rl"
N_CHECKPOINTS = 6

Record = dict[str, Any]
Slopes = dict[str, dict[int, dict[int, float]]]


class Cell(NamedTuple):
    env: str
    seed: int
    step: int
    level: int
    # (41, episodes): row 0 = actor, then the level's 40 perturbations
    returns: np.ndarray
    lengths: np.ndarray


class Procedure(NamedTuple):
    kind: str
    k: int
    n: int
    paired: bool

    @property
    def name(self) -> str:
        if self.kind == "single":
            return f"single k={self.k}"
        mode = "paired" if self.paired else "unpaired"
        return f"best-of-{self.n} k={self.k} {mode}"


def checkpoint_cells(d: dict[str, Any], env: str) -> list[Cell]:
    rows = [d["actor"], *d["perturbations"]]
    cells = []
    for level in range(len(d["sigmas"])):
        keep = [rows[0]] + [r for r in rows[1:] if r["level"] == level]
        cells.append(
            Cell(
                env,
                d["meta"]["seed"],
                d["meta"]["env_steps"],
                level,
                np.array([r["returns"] for r in keep], float),
                np.array([r["lengths"] for r in keep], float),
            )
        )
    return cells


def load_cells(spec: dict[str, Any]) -> tuple[list[Cell], list[Record]]:
    root = spec["dir"] / spec["analysis"]["episodes_cache"]
    cells: list[Cell] = []
    agreement: list[Record] = []
    for env in spec["envs"]:
        files = sorted((root / env).glob("s*_step_*.json"))
        if len(files) != N_CHECKPOINTS:
            raise FileNotFoundError(
                f"{root / env}: {len(files)} of {N_CHECKPOINTS} checkpoint "
                "files. run python -m offline.local_search episodes "
                f"<toml> <checkpoint_root> {env}"
            )
        for f in files:
            d = json.loads(f.read_text())
            cells += checkpoint_cells(d, env)
            agreement.append(
                {"env": env, "seed": d["meta"]["seed"]}
                | {"step": d["meta"]["env_steps"]}
                | {"reproduction_checked": d["reproduction_checked"]}
                | d["agreement"]
            )
    return cells, agreement


def truth(cell: Cell, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    # true gain, and "better" = paired t-interval entirely above zero
    d = cell.returns[1:] - cell.returns[0]
    n = d.shape[1]
    gain = d.mean(axis=1)
    half = student_t.ppf(1 - alpha / 2, n - 1) * d.std(1, ddof=1) / n**0.5
    return gain, gain - half > 0


def draws(
    rng: np.random.Generator, shape: tuple[int, ...], k: int, n: int
) -> np.ndarray:
    # k of range(n) without replacement, independently per leading index
    return rng.random((*shape, n)).argsort(axis=-1)[..., :k]


def boot_lower(
    x: np.ndarray, alpha: float, y: np.ndarray | None = None
) -> np.ndarray:
    # exact percentile bootstrap over all k^k resamples of each row of x;
    # with y, the two-sample difference over all k^k x k^k pairs
    k = x.shape[1]
    idx = np.array(list(itertools.product(range(k), repeat=k)))
    means = x[:, idx].mean(axis=-1)
    if y is not None:
        other = y[:, idx].mean(axis=-1)
        means = (means[:, :, None] - other[:, None, :]).reshape(len(x), -1)
    return np.quantile(means, alpha / 2, axis=1)


def metrics(
    accept: np.ndarray, better: np.ndarray, gain: np.ndarray, cost: float
) -> Record:
    hits = accept & better
    return {
        "p_accept": float(accept.mean()),
        "p_correct": float((accept == better).mean()),
        "p_false": float((accept & ~better).mean()),
        "p_accepted_better": (
            float(hits.sum() / accept.sum()) if accept.any() else None
        ),
        "gain": float(np.mean(accept * gain)),
        "cost": cost,
    }


def single(
    cell: Cell, proc: Procedure, rng: np.random.Generator, an: Record
) -> Record:
    gain, better = truth(cell, an["alpha"])
    d = cell.returns[1:] - cell.returns[0]
    n_pert, n_eps = d.shape
    idx = draws(rng, (n_pert, an["reps"]), proc.k, n_eps)
    rows = np.arange(n_pert)[:, None, None]
    accept = d[rows, idx].mean(axis=-1) > 0
    steps = cell.lengths[1:][rows, idx].sum(-1) + cell.lengths[0][idx].sum(-1)
    return metrics(accept, better[:, None], gain[:, None], float(steps.mean()))


def best_of(
    cell: Cell, proc: Procedure, rng: np.random.Generator, an: Record
) -> Record:
    gain, better = truth(cell, an["alpha"])
    n_pert, n_eps = cell.returns[1:].shape
    reps = np.arange(an["reps"])
    cand = draws(rng, (an["reps"],), proc.n, n_pert) + 1
    if proc.paired:
        idx = np.repeat(
            draws(rng, (an["reps"], 1), proc.k, n_eps), proc.n, axis=1
        )
        actor_idx = idx[:, 0]
    else:
        idx = draws(rng, (an["reps"], proc.n), proc.k, n_eps)
        actor_idx = draws(rng, (an["reps"],), proc.k, n_eps)
    x = cell.returns[cand[:, :, None], idx]
    y = cell.returns[0][actor_idx]
    win = x.mean(-1).argmax(axis=1)
    if proc.paired:
        lower = boot_lower(x[reps, win] - y, an["alpha"])
    else:
        lower = boot_lower(x[reps, win], an["alpha"], y)
    steps = cell.lengths[cand[:, :, None], idx].sum((1, 2))
    steps += cell.lengths[0][actor_idx].sum(-1)
    w = cand[reps, win] - 1
    return metrics(lower > 0, better[w], gain[w], float(steps.mean()))


def procedures(an: Record) -> list[Procedure]:
    procs = [Procedure("single", k, 1, True) for k in an["single_k"]]
    for n, k, paired in itertools.product(
        an["best_n"], an["best_k"], (True, False)
    ):
        procs.append(Procedure("best", k, n, paired))
    return procs


def simulate(
    cells: list[Cell], an: Record
) -> tuple[list[Record], list[Record]]:
    procs = procedures(an)
    records, summary = [], []
    for ci, cell in enumerate(cells):
        gain, better = truth(cell, an["alpha"])
        where = {
            "env": cell.env,
            "seed": cell.seed,
            "step": cell.step,
            "level": cell.level,
        }
        summary.append(
            where
            | {
                "actor": float(cell.returns[0].mean()),
                "n_better": int(better.sum()),
                "max_gain": float(gain.max()),
                "mean_gain": float(gain.mean()),
            }
        )
        for pi, proc in enumerate(procs):
            rng = np.random.default_rng([an["rng_seed"], ci, pi])
            run = single if proc.kind == "single" else best_of
            m = run(cell, proc, rng, an)
            m["per_100k"] = m["gain"] / m["cost"] * an["per_steps"]
            records.append(where | proc._asdict() | {"name": proc.name} | m)
    return records, summary


def td3_slopes(source: Record, an: Record, steps: list[int]) -> Slopes:
    # env -> seed -> phase -> OLS slope of eval_rl per `per_steps` steps
    runs = load(
        cache_dirs(source), source["wandb"]["run_name"], source["protocol"]
    )
    runs = select(runs, source["conditions"], source["envs"])
    win, total = an["td3_window"], source["total_steps"]
    out: Slopes = {}
    for run in runs:
        x, y = series(run["hist"], METRIC)
        per: dict[int, float] = {}
        for t in steps:
            m = (x >= max(0, t - win)) & (x <= min(total, t + win))
            if m.sum() < 3:
                raise ValueError(
                    f"{run['name']}: {m.sum()} {METRIC} points around {t}. "
                    "re-run analysis.fetch for the source experiment"
                )
            per[t] = float(np.polyfit(x[m], y[m], 1)[0] * an["per_steps"])
        out.setdefault(run["env"], {})[run["seed"]] = per
    return out


def hc_false(records: list[Record], env: str) -> dict[tuple[str, int], float]:
    # mean false acceptance per (procedure, level) over the env's 6 cells
    groups: dict[tuple[str, int], list[float]] = {}
    for r in records:
        if r["env"] == env:
            groups.setdefault((r["name"], r["level"]), []).append(r["p_false"])
    return {key: float(np.mean(v)) for key, v in groups.items()}


def phase_entries(
    records: list[Record],
    slopes: Slopes,
    false: dict[tuple[str, int], float],
    fa_max: float,
) -> list[Record]:
    # one entry per (env, phase, procedure, level), seeds side by side
    groups: dict[tuple[str, int, str, int], dict[int, Record]] = {}
    for r in records:
        key = (r["env"], r["step"], r["name"], r["level"])
        groups.setdefault(key, {})[r["seed"]] = r
    out = []
    for (env, step, name, level), by_seed in groups.items():
        td3 = {s: slopes[env][s][step] for s in by_seed}
        per = {s: r["per_100k"] for s, r in by_seed.items()}
        beats = all(v > 0 and v > td3[s] for s, v in per.items())
        out.append(
            {
                "env": env,
                "step": step,
                "procedure": name,
                "level": level,
                "per_100k": per,
                "td3": td3,
                "mean_per_100k": float(np.mean(list(per.values()))),
                "gain": float(np.mean([r["gain"] for r in by_seed.values()])),
                "cost": float(np.mean([r["cost"] for r in by_seed.values()])),
                "beats_td3": beats and len(per) == 2,
                "hc_false": false[(name, level)],
                "hc_ok": false[(name, level)] <= fa_max,
            }
        )
    return out


def phase_table(entries: list[Record], spec: Record) -> list[Record]:
    # per env x phase: best procedure by mean gain per 100k, overall and
    # among those with HalfCheetah false acceptance within the rule
    table = []
    phases = sorted({e["step"] for e in entries})
    for env, step in itertools.product(spec["envs"], phases):
        here = [e for e in entries if e["env"] == env and e["step"] == step]
        ok = [e for e in here if e["hc_ok"]]
        top = max(here, key=lambda e: e["mean_per_100k"])
        table.append(
            {
                "env": env,
                "step": step,
                "td3": top["td3"],
                "best": top,
                "best_hc_ok": (
                    max(ok, key=lambda e: e["mean_per_100k"]) if ok else None
                ),
                "n_beating": sum(e["beats_td3"] for e in here),
                "n_passing": sum(e["beats_td3"] and e["hc_ok"] for e in here),
            }
        )
    return table


def verdict(entries: list[Record], spec: Record) -> Record:
    targets = spec["envs"][1:]
    passing = [
        {k: e[k] for k in ("env", "step", "procedure", "level", "hc_false")}
        for e in entries
        if e["env"] in targets and e["beats_td3"] and e["hc_ok"]
    ]
    return {"pays_off": bool(passing), "passing": passing}


def procedure_summary(records: list[Record]) -> list[Record]:
    # per env x level x procedure: means over the env's 6 cells
    keys = ("p_accept", "p_correct", "p_false", "gain", "cost", "per_100k")
    groups: dict[tuple[str, int, str], list[Record]] = {}
    for r in records:
        groups.setdefault((r["env"], r["level"], r["name"]), []).append(r)
    out = []
    for (env, level, name), rs in groups.items():
        acc = [r["p_accepted_better"] for r in rs]
        acc = [a for a in acc if a is not None]
        first = rs[0]
        out.append(
            {"env": env, "level": level, "procedure": name}
            | {k: first[k] for k in ("kind", "k", "n", "paired")}
            | {k: float(np.mean([r[k] for r in rs])) for k in keys}
            | {"p_accepted_better": float(np.mean(acc)) if acc else None}
        )
    return out


def fmt(v: float | None, spec: str = ".3f") -> str:
    return "-" if v is None else format(v, spec)


def print_report(data: Record) -> None:
    print("agreement (rerun vs critic-degeneracy-v1):")
    for a in data["agreement"]:
        print(
            f"  {a['env']:15} s{a['seed']} {a['step']:>7} "
            f"val max|diff| {a['validation_max_abs_diff']:.2e} "
            f"truth corr {a['truth_mean_corr']:.4f}"
        )
    print("\nprocedures (mean over 6 cells):")
    for s in data["summary"]:
        print(
            f"  {s['env']:15} L{s['level']} {s['procedure']:26} "
            f"acc {fmt(s['p_accept'])} right {fmt(s['p_correct'])} "
            f"false {fmt(s['p_false'])} acc->better "
            f"{fmt(s['p_accepted_better'])} gain {fmt(s['gain'], '.1f')} "
            f"cost {s['cost']:.0f} /100k {fmt(s['per_100k'], '.1f')}"
        )
    print("\nphases (best mean gain per 100k vs td3 per seed):")
    for p in data["phases"]:
        b, ok = p["best"], p["best_hc_ok"]
        td3 = " ".join(f"s{s}={v:.1f}" for s, v in p["td3"].items())
        ok_name = f"{ok['procedure']} L{ok['level']}" if ok else "-"
        print(
            f"  {p['env']:15} {p['step']:>7} td3 {td3} | best "
            f"{b['procedure']} L{b['level']} {b['mean_per_100k']:.1f} | "
            f"hc-ok {ok_name} {fmt(ok and ok['mean_per_100k'], '.1f')} | "
            f"beat {p['n_beating']} pass {p['n_passing']}"
        )
    v = data["verdict"]
    print(f"\nverdict: pays_off={v['pays_off']} ({len(v['passing'])} passing)")
    for e in v["passing"]:
        print(f"  {e}")


def main() -> None:
    spec = load_spec(Path(sys.argv[1]))
    source = load_spec((spec["dir"] / spec["source"]).resolve())
    an = spec["analysis"]
    cells, agreement = load_cells(spec)
    records, cell_summary = simulate(cells, an)
    slopes = td3_slopes(source, an, sorted({c.step for c in cells}))
    false = hc_false(records, spec["envs"][0])
    entries = phase_entries(
        records, slopes, false, spec["rule"]["false_accept_max"]
    )
    data = {
        "verdict": verdict(entries, spec),
        "phases": phase_table(entries, spec),
        "entries": entries,
        "summary": procedure_summary(records),
        "cells": cell_summary,
        "td3": slopes,
        "agreement": agreement,
        "spec": an | {"rule": spec["rule"], "envs": spec["envs"]},
    }
    out = spec["dir"] / "local_search.json"
    out.write_text(json.dumps(data, indent=1))
    page = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data))
    html = spec["dir"] / "local_search.html"
    html.write_text(page)
    print_report(data)
    print(f"\n-> {out}\n-> {html}")


if __name__ == "__main__":
    main()
