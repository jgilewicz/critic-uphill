import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from analysis.runs import (
    Run,
    cache_dirs,
    cells,
    complete_envs,
    final,
    final_start,
    load,
    load_spec,
    select,
)
from analysis.stats import (
    Cells,
    boot_iqm,
    ci,
    holm,
    iqm,
    normalize,
    p_equivalence,
    p_superiority,
    pooled,
    prob_greater_ci,
)


def check_budget(runs: list[Run], total: int) -> None:
    short = [
        r["name"] for r in runs if r["hist"]["env_steps"].max() < 0.99 * total
    ]
    if short:
        raise ValueError(
            f"{len(short)} runs stop before 99% of {total} steps: "
            f"{short[:5]}. rerun fetch or drop them from the caches"
        )


def scores(spec: dict[str, Any], runs: list[Run], envs: list[str]) -> Cells:
    key = spec["metric"]["keys"][0]
    start = final_start(spec)
    raw = cells(
        runs, envs, spec["conditions"], lambda r: final(r["hist"], key, start)
    )
    # one env: keep raw returns, normalizing would only rescale them
    return normalize(raw) if len(envs) > 1 else raw


def test(spec: dict[str, Any], runs: list[Run], comp: dict) -> dict:
    envs = complete_envs(
        runs, comp.get("envs", spec["envs"]), spec["conditions"]
    )
    if not envs:
        return {**comp, "verdict": "not tested: no env has every condition"}
    s = scores(spec, runs, envs)
    a, b = comp["a"], comp["b"]
    rng = np.random.default_rng(0)
    draws = boot_iqm(s, [a, b], rng)
    diff = draws[a] - draws[b]
    margin = comp.get("equivalence_margin")
    p = p_superiority(diff) if margin is None else p_equivalence(diff, margin)
    return {
        **comp,
        "envs": envs,
        "normalized": len(envs) > 1,
        "diff": float(iqm(pooled(s, a)) - iqm(pooled(s, b))),
        "ci": ci(diff),
        "p": p,
        "p_greater": prob_greater_ci(s, a, b, rng),
        "per_env": {
            e: {
                c: [float(s[e][c].mean()), float(s[e][c].std())] for c in (a, b)
            }
            for e in envs
        },
    }


def verdict(r: dict[str, Any], alpha: float) -> str:
    if r["p_holm"] >= alpha:
        if "equivalence_margin" in r:
            return "equivalence not shown"
        return "no difference detected"
    if "equivalence_margin" in r:
        return f"equivalent within ±{r['equivalence_margin']}"
    return f"{r['a']} > {r['b']}" if r["diff"] > 0 else f"{r['b']} > {r['a']}"


def run_family(
    spec: dict[str, Any], runs: list[Run], comps: list[dict], alpha: float
) -> list[dict[str, Any]]:
    results = [test(spec, runs, c) for c in comps]
    tested = [r for r in results if "p" in r]
    for r, p in zip(tested, holm([r["p"] for r in tested]), strict=True):
        r["p_holm"] = p
        r["verdict"] = verdict(r, alpha)
        r["pass"] = p < alpha and ("equivalence_margin" in r or r["diff"] > 0)
    return results


def run_all(spec: dict[str, Any], runs: list[Run]) -> list[dict[str, Any]]:
    alpha = spec["analysis"]["alpha"]
    families: dict[str, list[dict]] = {}
    for c in spec["comparisons"]:
        families.setdefault(c.get("family", "primary"), []).append(c)
    out: list[dict[str, Any]] = []
    passed: set[str] = set()
    for name, comps in families.items():
        # gatekeeping: a family is tested only if the one it requires passed
        need = {c["requires"] for c in comps if "requires" in c}
        if need - passed:
            out += [
                {**c, "verdict": f"not tested: {need} failed"} for c in comps
            ]
            continue
        results = run_family(spec, runs, comps, alpha)
        if any(r.get("pass", False) for r in results):
            passed.add(name)
        out += results
    return out


def print_result(r: dict[str, Any]) -> None:
    print(r.get("question", f"{r['a']} vs {r['b']}"))
    if "p" not in r:
        print(f"  -> {r['verdict']}")
        return
    lo, hi = r["ci"]
    pg = r["p_greater"]
    scale = "norm" if r["normalized"] else "raw"
    print(
        f"  dIQM ({scale}) {r['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        f"  p {r['p']:.4f}  Holm {r['p_holm']:.4f}"
        f"  P(A>B) {pg['p']:.2f} [{pg['ci'][0]:.2f}, {pg['ci'][1]:.2f}]"
    )
    print(f"  -> {r['verdict']}  ({len(r['envs'])} envs)")


def main(toml_path: Path) -> None:
    spec = load_spec(toml_path)
    runs = load(cache_dirs(spec), spec["wandb"]["run_name"], spec["protocol"])
    runs = select(runs, spec["conditions"], spec["envs"])
    check_budget(runs, spec["total_steps"])
    results = run_all(spec, runs)
    for r in results:
        print_result(r)
    out = spec["dir"] / "compare.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
