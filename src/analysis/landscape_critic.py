import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from analysis.curvature_es import paired
from analysis.runs import load_spec

TEMPLATE = Path(__file__).parent / "templates" / "landscape_critic.html"
LANDSCAPE = ("A", "B", "C")
Record = dict[str, Any]


def load_files(spec: Record, source: Record) -> list[Record]:
    root = spec["dir"] / spec["analysis"]["cache"]
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
            "python -m offline.landscape_critic run for every env"
        )
    return [json.loads(f.read_text()) for f in files]


def critic_row(c: Record, actor: np.ndarray, log_every: int) -> Record:
    return {
        "name": c["name"],
        "variant": c["variant"],
        "updates": c["updates"],
        "sketch": c["cos"]["sketch"],
        "raw": c["cos"]["raw"],
        "levels": c["cos"]["levels"],
        "loss": c["losses"][c["updates"] // log_every - 1]
        if "losses" in c
        else None,
        "n": c.get("n", 0),
        "alpha": c["alpha"],
        "step": paired(c["returns"], actor),
    }


def es_rows(d: Record, actor: np.ndarray) -> list[Record]:
    out = []
    for level in (0, 1):
        at = [s for s in d["es_steps"] if s["level"] == level]
        oracle = next(s for s in at if s["source"] == "oracle")
        k1 = [
            paired(s["returns"], actor)["gain"] for s in at if s is not oracle
        ]
        out.append(
            {
                "level": level,
                "sigma": d["sigmas"][level],
                "oracle": paired(oracle["returns"], actor),
                "k1": float(np.mean(k1)),
                "k1_min": float(np.min(k1)),
                "k1_max": float(np.max(k1)),
                "k1_cost": float(
                    np.mean([s["cost"] for s in at if s is not oracle])
                ),
            }
        )
    return out


def cell(d: Record, log_every: int) -> Record:
    meta, actor = d["meta"], np.array(d["actor"]["returns"])
    crit = [critic_row(c, actor, log_every) for c in d["critics"]]
    land = [c for c in crit if c["variant"] in LANDSCAPE]
    best = max(land, key=lambda c: c["sketch"])
    control = [c for c in crit if c["variant"] == "D"]
    paired_d = next(c for c in control if c["updates"] == best["updates"])
    n = len(d["landscape"]["levels"]) // 2
    return {
        "env": meta["env"],
        "seed": meta["seed"],
        "step": meta["env_steps"],
        "actor": float(actor.mean()),
        "params": d["params"],
        "ceiling": float(np.sqrt(n / (n + d["params"]))),
        "landscape_cost": int(np.sum(d["landscape"]["lengths"])),
        "critics": crit,
        "es": es_rows(d, actor),
        "x_star": best["name"],
        "d_star": paired_d["name"],
        "d_best_sketch": max(c["sketch"] for c in control),
        "seconds": d["seconds"],
    }


def by_name(c: Record, name: str) -> Record:
    return next(r for r in c["critics"] if r["name"] == name)


def verdicts(rows: list[Record], spec: Record) -> Record:
    hyp = {h["id"]: h for h in spec["hypotheses"]}
    walker = [c for c in rows if c["env"] == "Walker2d-v5"]
    for c in walker:
        x, d = by_name(c, c["x_star"]), by_name(c, c["d_star"])
        c["l1"] = (
            x["sketch"] >= hyp["L1"]["min_cos"]
            and x["sketch"] >= c["d_best_sketch"] + hyp["L1"]["min_margin"]
        )
        c["l2"] = x["step"]["gain"] > max(0.0, d["step"]["gain"])
    n1, n2 = sum(c["l1"] for c in walker), sum(c["l2"] for c in walker)
    rel = [
        by_name(c, c["x_star"])["step"]["gain"] / abs(c["actor"])
        for c in rows
        if c["env"] == "HalfCheetah-v5"
    ]
    med = float(np.median(rel))
    return {
        "L1": {
            "count": n1,
            "of": len(walker),
            "passes": n1 >= hyp["L1"]["min_cells"],
        },
        "L2": {
            "count": n2,
            "of": len(walker),
            "passes": n2 >= hyp["L2"]["min_cells"],
        },
        "HC": {
            "median_rel": med,
            "rel": rel,
            "passes": med >= hyp["HC"]["hc_min_rel"],
        },
    }


def print_report(data: Record) -> None:
    v = data["verdict"]
    for h in ("L1", "L2"):
        print(f"{h}: {v[h]['count']}/{v[h]['of']} -> {v[h]['passes']}")
    print(
        f"HC: median rel {100 * v['HC']['median_rel']:+.2f}% -> "
        f"{v['HC']['passes']}"
    )
    for c in data["cells"]:
        x, t0 = by_name(c, c["x_star"]), by_name(c, "T0")
        d = by_name(c, c["d_star"])
        es = " ".join(f"{e['oracle']['gain']:7.1f}" for e in c["es"])
        print(
            f"{c['env'][:8]} s{c['seed']} {c['step'] // 1000:4d}k "
            f"actor {c['actor']:7.1f} | sketch T0 {t0['sketch']:+.2f} "
            f"X* {c['x_star']} {x['sketch']:+.2f} D {c['d_best_sketch']:+.2f}"
            f" | gain T0 {t0['step']['gain']:7.1f} X* {x['step']['gain']:7.1f}"
            f" D* {d['step']['gain']:7.1f} ES {es}"
        )


def main() -> None:
    spec = load_spec(Path(sys.argv[1]))
    source = load_spec((spec["dir"] / spec["source"]).resolve())
    every = spec["critic"]["log_every"]
    rows = [cell(d, every) for d in load_files(spec, source)]
    data = {
        "verdict": verdicts(rows, spec),
        "cells": rows,
        "spec": {
            k: spec[k]
            for k in (
                "landscape",
                "critic",
                "steps",
                "evaluation",
                "hypotheses",
            )
        },
    }
    out = spec["dir"] / "landscape_critic.json"
    out.write_text(json.dumps(data, indent=1))
    page = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data))
    html = spec["dir"] / "landscape_critic.html"
    html.write_text(page)
    print_report(data)
    print(f"\n-> {out}\n-> {html}")


if __name__ == "__main__":
    main()
