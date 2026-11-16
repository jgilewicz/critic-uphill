import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from analysis.runs import cache_dirs, load, load_spec, select, series

TEMPLATE = Path(__file__).parent / "templates" / "sa_pilot.html"
METRIC = "perf/eval_rl"
HORIZON = 100_000
Record = dict[str, Any]


def load_cells(spec: Record) -> list[Record]:
    root = spec["dir"] / spec["analysis"]["chains_cache"]
    cells, missing = [], []
    for entry in spec["cells"]:
        for step in entry["steps"]:
            for seed in spec["seeds"]:
                f = root / entry["env"] / f"s{seed}_step_{step}.json"
                if not f.is_file():
                    missing.append(str(f))
                    continue
                d = json.loads(f.read_text())
                cells.append(
                    {"env": entry["env"], "seed": seed, "step": step, **d}
                )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} chain files missing, first {missing[0]}. run "
            "python -m offline.sa_chain run for every env in [[cells]]"
        )
    return cells


def window_mean(x: np.ndarray, y: np.ndarray, t: int, w: int) -> float:
    m = (x >= t - w) & (x <= t + w)
    if m.sum() < 3:
        raise ValueError(
            f"{m.sum()} {METRIC} points in [{t - w}, {t + w}]. re-run "
            "analysis.fetch for the source experiment"
        )
    return float(y[m].mean())


def td3_gains(source: Record, cells: list[Record], w: int) -> dict[str, Any]:
    # (env, seed, step) -> eval_rl at t, at t + 100k and the gain. none past
    # the run's end
    runs = load(
        cache_dirs(source), source["wandb"]["run_name"], source["protocol"]
    )
    runs = select(runs, source["conditions"], source["envs"])
    curves = {(r["env"], r["seed"]): series(r["hist"], METRIC) for r in runs}
    out: dict[str, Any] = {}
    for c in cells:
        x, y = curves[(c["env"], c["seed"])]
        key = f"{c['env']}|{c['seed']}|{c['step']}"
        start = window_mean(x, y, c["step"], w)
        if c["step"] + HORIZON + w > source["total_steps"]:
            out[key] = {"start": start, "end": None, "gain": None}
            continue
        end = window_mean(x, y, c["step"] + HORIZON, w)
        out[key] = {"start": start, "end": end, "gain": end - start}
    return out


def accepts_until(trace: Record, budget: int) -> int:
    return int(
        sum(
            a
            for u, a in zip(trace["used"], trace["accepted"], strict=True)
            if u <= budget
        )
    )


def variant_rows(cell: Record) -> list[Record]:
    # per (level, schedule): gain per chain at each snapshot vs start actor
    ev = cell["evaluation"]
    actor = np.array(ev["actor"]["returns"])
    n_snap = len(cell["snapshots"])
    groups: dict[tuple[int, str], list[Record]] = {}
    for i, ch in enumerate(cell["chains"]):
        snaps = ev["snapshots"][i * n_snap : (i + 1) * n_snap]
        gains = [float(np.mean(np.array(s["returns"]) - actor)) for s in snaps]
        groups.setdefault((ch["level"], ch["schedule"]), []).append(
            {
                "chain": ch["chain"],
                "gains": gains,
                "accepts": [
                    accepts_until(ch["trace"], u) for u in ch["snapshot_used"]
                ],
                "iterations": ch["iterations"],
                "used": ch["used"],
                "snapshot_used": ch["snapshot_used"],
            }
        )
    rows = []
    for (level, schedule), chains in sorted(groups.items()):
        g = np.array([c["gains"] for c in chains])
        rows.append(
            {
                "level": level,
                "schedule": schedule,
                "sigma": cell["sigmas"][level],
                "t0": cell["t0"][level],
                "gain": g.mean(0).tolist(),
                "gain_min": g.min(0).tolist(),
                "gain_max": g.max(0).tolist(),
                "accepts": np.mean([c["accepts"] for c in chains], 0).tolist(),
                "iterations": float(np.mean([c["iterations"] for c in chains])),
                "chains": chains,
            }
        )
    return rows


def summarize(cells: list[Record], td3: dict[str, Any]) -> list[Record]:
    out = []
    for c in cells:
        actor = np.array(c["evaluation"]["actor"]["returns"])
        out.append(
            {
                "env": c["env"],
                "seed": c["seed"],
                "step": c["step"],
                "start": float(actor.mean()),
                "start_len": float(
                    np.mean(c["evaluation"]["actor"]["lengths"])
                ),
                "td3": td3[f"{c['env']}|{c['seed']}|{c['step']}"],
                "variants": variant_rows(c),
                "seconds": c["seconds"],
            }
        )
    return out


def find(rows: list[Record], env: str, seed: int, step: int) -> Record:
    for r in rows:
        if (r["env"], r["seed"], r["step"]) == (env, seed, step):
            return r
    raise KeyError(f"no cell {env} s{seed} {step}")


def beats(cell: Record, v: int) -> bool:
    gain, td3 = cell["variants"][v]["gain"][-1], cell["td3"]["gain"]
    return gain > 0 and td3 is not None and gain > td3


def verdict(rows: list[Record], spec: Record) -> Record:
    rule, seeds = spec["rule"], spec["seeds"]
    swim = [
        find(rows, "Swimmer-v5", s, t)
        for t in (300_000, 600_000)
        for s in seeds
    ]
    walker = find(rows, "Walker2d-v5", rule["stuck_seed"], 600_000)
    hc = [find(rows, "HalfCheetah-v5", s, 600_000) for s in seeds]
    out = []
    for v, var in enumerate(walker["variants"]):
        n_swim = sum(beats(c, v) for c in swim)
        hc_rel = [c["variants"][v]["gain"][-1] / abs(c["start"]) for c in hc]
        checks = {
            "swimmer": n_swim >= 3,
            "walker": beats(walker, v),
            "halfcheetah": all(abs(r) < rule["hc_max_rel"] for r in hc_rel),
        }
        out.append(
            {
                "level": var["level"],
                "schedule": var["schedule"],
                "swimmer_pairs": n_swim,
                "walker_gain": walker["variants"][v]["gain"][-1],
                "walker_td3": walker["td3"]["gain"],
                "hc_rel": hc_rel,
                "checks": checks,
                "passes": all(checks.values()),
            }
        )
    return {"sa_beats_td3": any(r["passes"] for r in out), "variants": out}


def print_report(data: Record) -> None:
    lv = ["5%", "20%"]
    print(f"SA beats TD3: {data['verdict']['sa_beats_td3']}")
    for r in data["verdict"]["variants"]:
        hc = " ".join(f"{100 * x:+.2f}%" for x in r["hc_rel"])
        print(
            f"  {r['schedule']:6s} σ{lv[r['level']]:3s} swimmer "
            f"{r['swimmer_pairs']}/4 walker {r['walker_gain']:8.1f} vs "
            f"{r['walker_td3']:8.1f} hc {hc} -> {r['passes']}"
        )
    for c in data["cells"]:
        t = c["td3"]["gain"]
        td3 = "   n/a" if t is None else f"{t:7.1f}"
        print(
            f"{c['env'][:8]} s{c['seed']} {c['step'] // 1000}k "
            f"start {c['start']:8.1f} td3 {td3}"
        )
        for v in c["variants"]:
            g = " ".join(f"{x:7.1f}" for x in v["gain"])
            a = " ".join(f"{x:4.1f}" for x in v["accepts"])
            print(
                f"    {v['schedule']:6s} σ{lv[v['level']]:3s} gain {g} "
                f"acc {a} / {v['iterations']:.0f} it"
            )


def main() -> None:
    spec = load_spec(Path(sys.argv[1]))
    source = load_spec((spec["dir"] / spec["source"]).resolve())
    cells = load_cells(spec)
    td3 = td3_gains(source, cells, spec["compare"]["window"])
    rows = summarize(cells, td3)
    data = {
        "verdict": verdict(rows, spec),
        "cells": rows,
        "spec": {
            "chain": spec["chain"],
            "evaluation": spec["evaluation"],
            "compare": spec["compare"],
            "rule": spec["rule"],
            "seeds": spec["seeds"],
        },
    }
    out = spec["dir"] / "sa_pilot.json"
    out.write_text(json.dumps(data, indent=1))
    page = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data))
    html = spec["dir"] / "sa_pilot.html"
    html.write_text(page)
    print_report(data)
    print(f"\n-> {out}\n-> {html}")


if __name__ == "__main__":
    main()
