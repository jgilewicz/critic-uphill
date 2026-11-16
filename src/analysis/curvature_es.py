import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from analysis.local_search import td3_slopes
from analysis.runs import load_spec

TEMPLATE = Path(__file__).parent / "templates" / "curvature_es.html"
PLATEAU = ("Walker2d-v5", 0, 600_000)
Record = dict[str, Any]


def load_files(spec: Record, source: Record) -> list[Record]:
    root = spec["dir"] / spec["analysis"]["steps_cache"]
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
            f"{len(missing)} step files missing, first {missing[0]}. run "
            "python -m offline.curvature_es run for every env"
        )
    return [json.loads(f.read_text()) for f in files]


def paired(head: list[float], actor: np.ndarray) -> Record:
    # mean paired difference and its two-sided 95% t-interval
    d = np.array(head) - actor
    half = student_t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return {
        "gain": float(d.mean()),
        "lo": float(d.mean() - half),
        "hi": float(d.mean() + half),
    }


def procedure_rows(steps: list[Record], actor: np.ndarray) -> list[Record]:
    # (source, direction, multiple) -> gain per draw, mean cost
    groups: dict[tuple[str, str, float], list[Record]] = {}
    for s in steps:
        key = (s["source"], s["direction"], s["multiple"])
        groups.setdefault(key, []).append(s)
    out = []
    for (src, direction, m), rows in groups.items():
        gains = [paired(r["returns"], actor)["gain"] for r in rows]
        out.append(
            {
                "source": src,
                "direction": direction,
                "multiple": m,
                "gain": float(np.mean(gains)),
                "gain_min": float(np.min(gains)),
                "gain_max": float(np.max(gains)),
                "cost": float(np.mean([r["cost"] for r in rows])),
                "alpha": float(np.mean([r["alpha"] for r in rows])),
                "ci": paired(rows[0]["returns"], actor)
                if src == "oracle"
                else None,
            }
        )
    return out


def cells(files: list[Record], slopes: Record) -> list[Record]:
    out = []
    for d in files:
        meta, actor = d["meta"], np.array(d["actor"]["returns"])
        slope = slopes[meta["env"]][meta["seed"]][meta["env_steps"]]
        for lv in d["levels"]:
            steps = [s for s in d["steps"] if s["level"] == lv["level"]]
            out.append(
                {
                    "env": meta["env"],
                    "seed": meta["seed"],
                    "step": meta["env_steps"],
                    "level": lv["level"],
                    "sigma": lv["sigma"],
                    "actor": float(actor.mean()),
                    "td3_slope": slope,
                    "best": paired(lv["best"]["returns"], actor),
                    "lambda": lv["lambda"],
                    "mu": [lv["mu_min"], lv["mu_max"]],
                    "cos": lv["cos_oracle_es_newton"],
                    "ref_dist_h": lv["ref_dist_h"],
                    "h_a_not_concave": d["h_a_not_concave"],
                    "procedures": procedure_rows(steps, actor),
                }
            )
    return out


def proc(cell: Record, src: str, direction: str, m: float) -> Record:
    for p in cell["procedures"]:
        if (p["source"], p["direction"], p["multiple"]) == (src, direction, m):
            return p
    raise KeyError(f"no procedure {src} {direction} {m}")


def h6_h7(rows: list[Record], multiples: list[float]) -> Record:
    ws = [c for c in rows if c["env"] != "HalfCheetah-v5"]
    for c in ws:
        es = [proc(c, "oracle", "es", m)["gain"] for m in multiples]
        nw = [proc(c, "oracle", "newton", m)["gain"] for m in multiples]
        c["h6"] = max(es) > c["best"]["gain"]
        c["h7"] = sum(n > e for n, e in zip(nw, es, strict=True)) >= 2
    n6, n7 = sum(c["h6"] for c in ws), sum(c["h7"] for c in ws)
    return {
        "H6": {"count": n6, "of": len(ws), "passes": n6 >= 16},
        "H7": {"count": n7, "of": len(ws), "passes": n7 >= 16},
    }


def beats_td3(cell: Record, p: Record) -> bool:
    return p["gain"] > 0 and p["gain"] > cell["td3_slope"] * p["cost"]


def h8(rows: list[Record], spec: Record) -> Record:
    st = spec["steps"]
    found = []
    for level in (0, 1):
        at = [c for c in rows if c["level"] == level]
        walker = next(
            c for c in at if (c["env"], c["seed"], c["step"]) == PLATEAU
        )
        swim = [c for c in at if c["env"] == "Swimmer-v5"]
        for k in st["ks"]:
            for direction in ("es", "newton"):
                for m in st["dist_multiples"]:
                    key = (f"k{k}", direction, m)
                    w = beats_td3(walker, proc(walker, *key))
                    n = sum(beats_td3(c, proc(c, *key)) for c in swim)
                    found.append(
                        {
                            "level": level,
                            "k": k,
                            "direction": direction,
                            "multiple": m,
                            "walker": w,
                            "swimmer": n,
                            "passes": w and n >= 4,
                        }
                    )
    return {"passes": any(f["passes"] for f in found), "procedures": found}


def hc(rows: list[Record], spec: Record) -> Record:
    rel = []
    for c in rows:
        if c["env"] != "HalfCheetah-v5":
            continue
        best = max(
            p["gain"] for p in c["procedures"] if p["source"] == "oracle"
        )
        rel.append(best / abs(c["actor"]))
    hyp = next(h for h in spec["hypotheses"] if h["id"] == "HC")
    med = float(np.median(rel))
    return {"median_rel": med, "rel": rel, "passes": med <= hyp["hc_max_rel"]}


def print_report(data: Record) -> None:
    v = data["verdict"]
    for h in ("H6", "H7"):
        print(f"{h}: {v[h]['count']}/{v[h]['of']} -> {v[h]['passes']}")
    best = max(v["H8"]["procedures"], key=lambda f: (f["walker"], f["swimmer"]))
    print(f"H8: {v['H8']['passes']} (best {best})")
    print(
        f"HC: median rel {100 * v['HC']['median_rel']:+.2f}% -> "
        f"{v['HC']['passes']}"
    )
    for c in data["cells"]:
        es = " ".join(
            f"{proc(c, 'oracle', 'es', m)['gain']:7.1f}"
            for m in (0.5, 1.0, 2.0)
        )
        nw = " ".join(
            f"{proc(c, 'oracle', 'newton', m)['gain']:7.1f}"
            for m in (0.5, 1.0, 2.0)
        )
        print(
            f"{c['env'][:8]} s{c['seed']} {c['step'] // 1000:4d}k L{c['level']}"
            f" actor {c['actor']:8.1f} best {c['best']['gain']:7.1f} "
            f"es {es} | N {nw} cos {c['cos']:.3f}"
        )


def main() -> None:
    spec = load_spec(Path(sys.argv[1]))
    source = load_spec((spec["dir"] / spec["source"]).resolve())
    files = load_files(spec, source)
    steps = sorted({d["meta"]["env_steps"] for d in files})
    an = {"td3_window": spec["compare"]["td3_window"], "per_steps": 1}
    rows = cells(files, td3_slopes(source, an, steps))
    verdict = h6_h7(rows, spec["steps"]["dist_multiples"])
    verdict |= {"H8": h8(rows, spec), "HC": hc(rows, spec)}
    data = {
        "verdict": verdict,
        "cells": rows,
        "spec": {
            k: spec[k]
            for k in ("states", "steps", "evaluation", "compare", "hypotheses")
        },
    }
    out = spec["dir"] / "curvature_es.json"
    out.write_text(json.dumps(data, indent=1))
    page = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data))
    html = spec["dir"] / "curvature_es.html"
    html.write_text(page)
    print_report(data)
    print(f"\n-> {out}\n-> {html}")


if __name__ == "__main__":
    main()
