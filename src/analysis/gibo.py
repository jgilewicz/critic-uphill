import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from analysis.curvature_es import paired
from analysis.local_search import td3_slopes
from analysis.runs import load_spec

TEMPLATE = Path(__file__).parent / "templates" / "gibo.html"
ARMS = ("G-b", "G-d", "ES-b")
Record = dict[str, Any]


def load_files(spec: Record, source: Record) -> list[Record]:
    root = spec["dir"] / spec["analysis"]["cells_cache"]
    steps = [
        int(f * source["total_steps"])
        for f in source["offline"]["checkpoint_at"]
    ]
    files = [
        root / env / f"s{seed}_step_{n}.json"
        for env in spec["envs"]
        for seed in spec["seeds"]
        for n in steps
    ]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} cell files missing, first {missing[0]}. run "
            "python -m offline.gibo run for every env"
        )
    return [json.loads(f.read_text()) for f in files]


def gate_ratio(spec: Record, meta: Record) -> float | None:
    root = spec["dir"] / spec["analysis"]["gate_cache"] / meta["env"]
    f = root / f"s{meta['seed']}_step_{meta['env_steps']}.json"
    if not f.is_file():
        return None
    return json.loads(f.read_text())["levels"][0]["ratio"]


def arm_row(a: Record, slope: float, actor: float) -> Record:
    chosen = a["candidates"][a["chosen"]]
    gain = a["decision_gain"]
    td3 = slope * a["cost"]
    return {
        "accepted": a["verify"]["accepted"],
        "stage": a["verify"]["stage"],
        "ci": a["verify"]["ci"],
        "multiple": chosen["multiple"],
        "chosen_gain": chosen["true_gain"],
        "candidates": [
            {
                "multiple": c["multiple"],
                "true_gain": c["true_gain"],
                "ls_y": float(np.mean(c["y"])),
                "fallback": c.get("fallback", False),
            }
            for c in a["candidates"]
        ],
        "decision_gain": gain,
        "rel": gain / abs(actor),
        "cost": a["cost"],
        "td3": td3,
        "beats": gain > 0 and gain > td3,
        "false_accept": a["verify"]["accepted"] and chosen["true_gain"] <= 0,
        "cos_es": a["cos_es"],
        "cos_es_proj": a["cos_es_proj"],
        "fallback_axes": len(a.get("fallback_axes", [])),
        "hyper": a["hyper"][-1] if a["hyper"] else None,
        "query_y": float(np.mean(a["y"])),
    }


def cell(d: Record, spec: Record, slopes: Record) -> Record:
    meta = d["meta"]
    ev = d["evaluation"]
    actor = np.array(ev["actor"]["returns"])
    slope = slopes[meta["env"]][meta["seed"]][meta["env_steps"]]
    return {
        "env": meta["env"],
        "seed": meta["seed"],
        "step": meta["env_steps"],
        "actor": float(actor.mean()),
        "td3_slope": slope,
        "gate_ratio": gate_ratio(spec, meta),
        "arms": {
            n: arm_row(d["arms"][n], slope, float(actor.mean())) for n in ARMS
        },
        "oracle": {
            o["name"]: paired(o["returns"], actor)
            | {"fallback": o.get("fallback", False)}
            for o in ev["oracle"]
        },
        "seconds": sum(d["seconds"].values()),
    }


def arm_summary(rows: list[Record], arm: str) -> Record:
    r = [c["arms"][arm] for c in rows]
    return {
        "beats": sum(a["beats"] for a in r),
        "accepted": sum(a["accepted"] for a in r),
        "false_accept": sum(a["false_accept"] for a in r),
        "positive": sum(a["decision_gain"] > 0 for a in r),
        "of": len(r),
        "median_rel": float(np.median([a["rel"] for a in r])),
        "mean_gain": float(np.mean([a["decision_gain"] for a in r])),
    }


def verdicts(rows: list[Record], spec: Record) -> Record:
    hyp = {h["id"]: h for h in spec["hypotheses"]}
    walker = [c for c in rows if c["env"] == "Walker2d-v5"]
    cheetah = [c for c in rows if c["env"] == "HalfCheetah-v5"]
    ratios = [c["gate_ratio"] for c in walker]
    n_gate = sum(x is not None and x >= hyp["G0"]["min_ratio"] for x in ratios)
    n_l = sum(c["arms"]["G-b"]["beats"] for c in walker)
    med = float(np.median([c["arms"]["G-b"]["rel"] for c in cheetah]))
    return {
        "G0": {
            "count": n_gate,
            "of": len(walker),
            "ratios": ratios,
            "passes": n_gate >= hyp["G0"]["min_cells"],
        },
        "L": {
            "count": n_l,
            "of": len(walker),
            "passes": n_l >= hyp["L"]["min_cells"],
        },
        "HC": {"median_rel": med, "passes": med >= hyp["HC"]["hc_min_rel"]},
        "arms": {
            env: {n: arm_summary(sel, n) for n in ARMS}
            for env in spec["envs"]
            if (sel := [c for c in rows if c["env"] == env])
        },
    }


def print_report(data: Record) -> None:
    v = data["verdict"]
    print(f"G0: {v['G0']['count']}/{v['G0']['of']} -> {v['G0']['passes']}")
    print(f"L: {v['L']['count']}/{v['L']['of']} -> {v['L']['passes']}")
    print(
        f"HC: median rel {100 * v['HC']['median_rel']:+.2f}% -> "
        f"{v['HC']['passes']}"
    )
    for env, arms in v["arms"].items():
        for n, s in arms.items():
            print(
                f"{env[:8]} {n:4s} beats {s['beats']}/{s['of']} acc "
                f"{s['accepted']} false {s['false_accept']} mean gain "
                f"{s['mean_gain']:7.1f} median rel {100 * s['median_rel']:+.2f}%"
            )
    for c in data["cells"]:
        arms = " | ".join(
            f"{n} {a['decision_gain']:6.1f}/{a['td3']:6.1f}"
            f"{'*' if a['accepted'] else ' '}"
            for n, a in c["arms"].items()
        )
        o = c["oracle"]
        print(
            f"{c['env'][:8]} s{c['seed']} {c['step'] // 1000:4d}k actor "
            f"{c['actor']:7.1f} | {arms} | ES {o['es']['gain']:6.1f} "
            f"ES-b {o['es_b']['gain']:6.1f}"
        )


def main() -> None:
    spec = load_spec(Path(sys.argv[1]))
    source = load_spec((spec["dir"] / spec["source"]).resolve())
    files = load_files(spec, source)
    steps = sorted({d["meta"]["env_steps"] for d in files})
    an = {"td3_window": spec["compare"]["td3_window"], "per_steps": 1}
    slopes = td3_slopes(source, an, steps)
    rows = [cell(d, spec, slopes) for d in files]
    data = {
        "verdict": verdicts(rows, spec),
        "cells": rows,
        "deviations": spec.get("deviations", {}),
        "spec": {
            k: spec[k]
            for k in (
                "subspace",
                "gibo",
                "es",
                "step",
                "verify",
                "evaluation",
                "hypotheses",
            )
        },
    }
    out = spec["dir"] / "gibo.json"
    out.write_text(json.dumps(data, indent=1))
    page = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data))
    html = spec["dir"] / "gibo.html"
    html.write_text(page)
    print_report(data)
    print(f"\n-> {out}\n-> {html}")


if __name__ == "__main__":
    main()
