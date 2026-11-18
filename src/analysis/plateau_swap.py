import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from analysis.compare import check_budget
from analysis.runs import (
    Hist,
    Run,
    auc,
    cache_dirs,
    final,
    final_start,
    load,
    load_spec,
    select,
    series,
)
from analysis.stats import Cells, boot_iqm, ci, iqm, prob_greater_ci

TEMPLATE = Path(__file__).parent / "templates" / "plateau_swap.html"
KEY = "perf/eval_rl"
BASE, ARM = "td3", "td3-swap"
PHASE_KEYS = (
    "t",
    "slope",
    "window_mean",
    "cost",
    "accepted",
    "ci_lo",
    "ci_hi",
    "stage",
    "multiple",
    "candidate_y",
    "fallback",
    "fallback_axes",
    "ref_dist_h",
    "h_a_not_concave",
    "critic_loss",
)
ROLLBACK_KEYS = ("rolled_back", "after_mean", "before_mean")
Record = dict[str, Any]


def _num(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


def _row(hist: Hist, keys: tuple[str, ...], i: int) -> Record:
    return {
        k: _num(hist[f"phase/{k}"][i]) if f"phase/{k}" in hist else None
        for k in keys
    }


def logged(hist: Hist, key: str) -> np.ndarray:
    return np.flatnonzero(np.isfinite(hist.get(key, np.array([]))))


def phases(hist: Hist) -> list[Record]:
    # a rollback decision always lands before the next phase (cooldown >
    # rollback evals), so decisions pair with accepted phases in order
    out = [
        _row(hist, PHASE_KEYS, i) | {"end": float(hist["env_steps"][i])}
        for i in logged(hist, "phase/t")
    ]
    decisions = iter(logged(hist, "phase/rolled_back"))
    for p in out:
        p["accepted"] = bool(p["accepted"])
        p["fallback"] = bool(p["fallback"])
        p |= dict.fromkeys(ROLLBACK_KEYS)
        i = next(decisions, None) if p["accepted"] else None
        if i is not None:
            p |= _row(hist, ROLLBACK_KEYS, int(i))
            p["rolled_back"] = bool(p["rolled_back"])
    return out


def run_row(r: Run, start: float, min_step: int) -> Record:
    h = r["hist"]
    ph = phases(h)
    end = float(h["env_steps"].max())
    x, y = series(h, KEY)
    return {
        "name": r["name"],
        "url": r["url"],
        "commit": r["commit"],
        "condition": r["condition"],
        "env": r["env"],
        "seed": r["seed"],
        "final": final(h, KEY, start),
        "auc": auc(h, KEY),
        # no phase can start before min_step: arm gap from seed luck alone
        "pre": float(y[x <= min_step].mean()),
        "end": end,
        "phases": ph,
        "n_phases": len(ph),
        "accepted": sum(p["accepted"] for p in ph),
        "rolled_back": sum(p["rolled_back"] is True for p in ph),
        "undecided": sum(
            p["accepted"] and p["rolled_back"] is None for p in ph
        ),
        "steps_frac": sum(p["cost"] for p in ph) / end,
        "curve": {"x": x.astype(int).tolist(), "y": np.round(y, 1).tolist()},
    }


def scores(rows: list[Record], envs: list[str], field: str) -> Cells:
    out: Cells = {}
    for env in envs:
        out[env] = {}
        for c in (BASE, ARM):
            v = [
                r[field]
                for r in rows
                if r["env"] == env and r["condition"] == c
            ]
            if not v:
                raise ValueError(f"no {c} runs on {env}. run fetch first")
            out[env][c] = np.array(v)
    return out


def env_stats(final_: Cells, auc_: Cells, env: str) -> Record:
    rng = np.random.default_rng(0)
    draws = boot_iqm({env: final_[env]}, [BASE, ARM], rng)
    iqms = {c: float(iqm(v)) for c, v in final_[env].items()}
    by: Record = {
        c: {
            "iqm": iqms[c],
            "ci": ci(draws[c]),
            "auc_iqm": float(iqm(auc_[env][c])),
            "q1": float(np.percentile(v, 25)),
            "n": len(v),
        }
        for c, v in final_[env].items()
    }
    p = prob_greater_ci({env: final_[env]}, ARM, BASE, rng)
    return by | {"p_greater": p, "ratio": iqms[ARM] / iqms[BASE]}


def pooled_q1(final_: Cells, per_env: Record) -> Record:
    # lower quartile of final / IQM(td3) of its env, pooled over envs
    return {
        c: float(
            np.percentile(
                np.concatenate(
                    [v[c] / per_env[e][BASE]["iqm"] for e, v in final_.items()]
                ),
                25,
            )
        )
        for c in (BASE, ARM)
    }


def verdict(per_env: Record, p: float, rule: Record) -> Record:
    ge = all(s[ARM]["iqm"] >= s[BASE]["iqm"] for s in per_env.values())
    drop = all(s["ratio"] >= 1 - rule["max_env_drop"] for s in per_env.values())
    return {
        "p": p,
        "p_ok": p >= rule["p_min"],
        "iqm_ge": ge,
        "no_drop": drop,
        "signal": p >= rule["p_min"] and ge and drop,
    }


def trigger_stats(rows: list[Record]) -> Record:
    sw = [r for r in rows if r["condition"] == ARM]
    ph = [p for r in sw for p in r["phases"]]
    zero = sum(r["n_phases"] == 0 for r in sw)
    return {
        "runs": len(sw),
        "zero": zero,
        "mostly_zero": zero > len(sw) / 2,
        "phases": len(ph),
        "per_run": [r["n_phases"] for r in sw],
        "accepted": sum(p["accepted"] for p in ph),
        "rolled_back": sum(p["rolled_back"] is True for p in ph),
        "undecided": sum(r["undecided"] for r in sw),
        "fallback": sum(p["fallback"] for p in ph),
        "steps_frac": [r["steps_frac"] for r in sw],
        "cost": [p["cost"] for p in ph],
    }


def expected_swap(spec: Record, env: str) -> Record:
    ph = spec["phase"]
    root = spec["dir"].parents[1]
    source = (spec["dir"] / spec["phase_source"]).resolve()
    return (
        {"enabled": True, "phase_spec": str(source.relative_to(root))}
        | spec["trigger"]
        | {k: ph[k] for k in ("states", "state_window", "perturbations")}
        | {k: ph[k] for k in ("seed_base", "key")}
        | {"sigma": ph["sigma"][env]}
        | spec["swap"]
    )


def config_mismatches(spec: Record, runs: list[Run]) -> list[str]:
    out = []
    for r in runs:
        got = r["config"].get("algorithm", {}).get("swap")
        if got is None:
            out.append(f"{r['name']}: no algorithm.swap in the cached config")
            continue
        want = (
            expected_swap(spec, r["env"])
            if r["condition"] == ARM
            else {"enabled": False}
        )
        out += [
            f"{r['name']}: swap.{k} = {got.get(k)!r}, prereg {v!r}"
            for k, v in want.items()
            if got.get(k) != v
        ]
    return out


def page_data(spec: Record, runs: list[Run]) -> Record:
    min_step = spec["trigger"]["min_step"]
    rows = [run_row(r, final_start(spec), min_step) for r in runs]
    envs = spec["envs"]
    final_, auc_ = scores(rows, envs, "final"), scores(rows, envs, "auc")
    per_env = {e: env_stats(final_, auc_, e) for e in envs}
    p: Record = prob_greater_ci(final_, ARM, BASE, np.random.default_rng(0))
    return {
        "name": spec["name"],
        "question": spec["question"],
        "envs": envs,
        "total_steps": spec["total_steps"],
        "final_start": final_start(spec),
        "verdict": verdict(per_env, p["p"], spec["decision"]),
        "per_env": per_env,
        "pooled": {"p_greater": p, "q1": pooled_q1(final_, per_env)},
        "trigger": trigger_stats(rows),
        "config_mismatches": config_mismatches(spec, runs),
        "commits": sorted({r["commit"] for r in rows}),
        "runs": rows,
        "spec": {k: spec[k] for k in ("trigger", "phase", "swap", "decision")},
    }


def print_report(d: Record) -> None:
    v, t = d["verdict"], d["trigger"]
    print(
        f"SIGNAL={v['signal']}: P {v['p']:.3f} (>= 0.65 {v['p_ok']}), "
        f"IQM >= in both envs {v['iqm_ge']}, no env drop > 5% {v['no_drop']}"
    )
    for env, s in d["per_env"].items():
        print(
            f"{env:12s} IQM td3 {s[BASE]['iqm']:7.1f} swap {s[ARM]['iqm']:7.1f}"
            f" ratio {s['ratio']:.3f} P {s['p_greater']['p']:.2f} AUC "
            f"{s[BASE]['auc_iqm']:7.1f}/{s[ARM]['auc_iqm']:7.1f} Q1 "
            f"{s[BASE]['q1']:7.1f}/{s[ARM]['q1']:7.1f}"
        )
    print(
        f"trigger: {t['zero']}/{t['runs']} swap runs with 0 phases, "
        f"{t['phases']} phases, {t['accepted']} accepted, "
        f"{t['rolled_back']} rolled back, {t['undecided']} undecided"
    )
    for r in d["runs"]:
        marks = " ".join(
            f"{p['t'] / 1000:.0f}k{'+' if p['accepted'] else '-'}"
            f"{'r' if p['rolled_back'] else ''}"
            for p in r["phases"]
        )
        print(
            f"{r['condition']:9s} {r['env'][:8]} s{r['seed']} final "
            f"{r['final']:7.1f} frac {100 * r['steps_frac']:4.1f}% {marks}"
        )
    for m in d["config_mismatches"]:
        print(f"CONFIG MISMATCH {m}")


def main() -> None:
    spec = load_spec(Path(sys.argv[1]))
    runs = load(cache_dirs(spec), spec["wandb"]["run_name"], spec["protocol"])
    runs = select(runs, spec["conditions"], spec["envs"])
    check_budget(runs, spec["total_steps"])
    data = page_data(spec, runs)
    out = spec["dir"] / "plateau_swap.json"
    out.write_text(json.dumps(data, indent=1))
    page = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(data))
    html = spec["dir"] / "plateau_swap.html"
    html.write_text(page)
    print_report(data)
    print(f"\n-> {out}\n-> {html}")


if __name__ == "__main__":
    main()
