import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

N_BOOT = 10_000
Scores = dict[str, dict[str, dict[int, float]]]  # cond -> env -> seed -> v


def load_runs(cache: Path) -> list[dict[str, Any]]:
    runs = [json.loads(p.read_text()) for p in sorted(cache.glob("*.json"))]
    return [r for r in runs if r["metric"] and r["state"] == "finished"]


def run_scores(
    run: dict[str, Any], total: int, window: float
) -> tuple[float, float]:
    x = np.asarray(run["metric"]["x"])
    y = np.asarray(run["metric"]["y"])
    keep = x <= total * 1.01
    x, y = x[keep], y[keep]
    if x[-1] < 0.99 * total:
        raise ValueError(
            f"{run['name']} stops at {x[-1]:.0f} of {total} steps. "
            "rerun fetch.py or drop the run"
        )
    final = float(y[x >= (1 - window) * total].mean())
    auc = float(np.trapezoid(y, x) / (x[-1] - x[0]))
    return final, auc


def collect(
    runs: list[dict[str, Any]], total: int, window: float
) -> tuple[Scores, Scores]:
    final: Scores = {}
    auc: Scores = {}
    for r in runs:
        f, a = run_scores(r, total, window)
        final.setdefault(r["condition"], {}).setdefault(r["env"], {})
        auc.setdefault(r["condition"], {}).setdefault(r["env"], {})
        final[r["condition"]][r["env"]][r["seed"]] = f
        auc[r["condition"]][r["env"]][r["seed"]] = a
    return final, auc


def normalize(scores: Scores, envs: list[str]) -> Scores:
    out: Scores = {c: {} for c in scores}
    for env in envs:
        means = [np.mean(list(scores[c][env].values())) for c in scores]
        lo, hi = min(means), max(means)
        for c in scores:
            out[c][env] = {
                s: (v - lo) / (hi - lo) for s, v in scores[c][env].items()
            }
    return out


def iqm(values: np.ndarray) -> float:
    q25, q75 = np.percentile(values, [25, 75])
    return float(values[(values >= q25) & (values <= q75)].mean())


def boot_iqm(
    norm: Scores, cond: str, envs: list[str], rng: np.random.Generator
) -> np.ndarray:
    # same rng seed for every condition + seed order: resamples are paired
    per_env = [
        np.asarray([norm[cond][e][s] for s in sorted(norm[cond][e])])
        for e in envs
    ]
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sample = [v[rng.integers(0, len(v), len(v))] for v in per_env]
        draws[i] = iqm(np.concatenate(sample))
    return draws


def flat(norm: Scores, cond: str, envs: list[str]) -> np.ndarray:
    return np.concatenate([list(norm[cond][e].values()) for e in envs])


def p_greater(norm: Scores, a: str, b: str, envs: list[str]) -> float:
    wins = []
    for e in envs:
        for s in norm[a][e].keys() & norm[b][e].keys():
            d = norm[a][e][s] - norm[b][e][s]
            wins.append(1.0 if d > 0 else 0.5 if d == 0 else 0.0)
    return float(np.mean(wins))


def ranking(norm: Scores, envs: list[str], seed: int) -> dict[str, np.ndarray]:
    print(f"{'condition':<15}{'IQM':>7}  95% CI")
    draws = {}
    for c in norm:
        draws[c] = boot_iqm(norm, c, envs, np.random.default_rng(seed))
    order = sorted(norm, key=lambda c: -iqm(flat(norm, c, envs)))
    for c in order:
        lo, hi = np.percentile(draws[c], [2.5, 97.5])
        point = iqm(flat(norm, c, envs))
        print(f"{c:<15}{point:7.2f}  [{lo:.2f}, {hi:.2f}]")
    return draws


def compare(
    norm: Scores,
    draws: dict[str, np.ndarray],
    pair: tuple[str, str],
    envs: list[str],
) -> tuple[float, float, float]:
    a, b = pair
    diff_point = iqm(flat(norm, a, envs)) - iqm(flat(norm, b, envs))
    lo, hi = np.percentile(draws[a] - draws[b], [2.5, 97.5])
    print(
        f"  {a} - {b}: dIQM {diff_point:+.2f} [{lo:+.2f}, {hi:+.2f}], "
        f"P(A>B) {p_greater(norm, a, b, envs):.2f}"
    )
    return diff_point, lo, hi


def per_env_table(raw: Scores, a: str, b: str, envs: list[str]) -> None:
    print(f"  {'env':<16}{a:>18}{b:>18}  wins A")
    for e in envs:
        va, vb = raw[a][e], raw[b][e]
        wins = sum(va[s] > vb[s] for s in va.keys() & vb.keys())
        n = len(va.keys() & vb.keys())
        print(
            f"  {e:<16}{np.mean(list(va.values())):9.0f} ± "
            f"{np.std(list(va.values())):<6.0f}"
            f"{np.mean(list(vb.values())):9.0f} ± "
            f"{np.std(list(vb.values())):<6.0f}  {wins}/{n}"
        )


def verdict(comparison: dict[str, Any], d: float, lo: float) -> str:
    margin = comparison.get("equivalence_margin")
    if margin is not None:
        ok = abs(d) < margin
        return f"|dIQM| < {margin}: {'PASS' if ok else 'FAIL'}"
    return f"CI of dIQM above 0: {'PASS' if lo > 0 else 'FAIL'}"


def report(
    raw: Scores,
    envs: list[str],
    comparisons: list[dict[str, Any]],
    label: str,
) -> None:
    norm = normalize(raw, envs)
    print(f"\n== {label} (normalized over {len(raw)} conditions) ==")
    draws = ranking(norm, envs, seed=0)
    for c in comparisons:
        print(c.get("question", ""))
        d, lo, _ = compare(norm, draws, (c["a"], c["b"]), envs)
        print(f"  -> {verdict(c, d, lo)}")
        per_env_table(raw, c["a"], c["b"], envs)


def main(toml_path: Path) -> None:
    spec = tomllib.loads(toml_path.read_text())
    runs = [
        r
        for r in load_runs(toml_path.parent / "cache" / "runs")
        if r["condition"] in spec["conditions"]
    ]
    final, auc = collect(
        runs, spec["total_steps"], spec["metric"]["final_window"]
    )
    envs = [
        e
        for e in spec["envs"]
        if all(len(final.get(c, {}).get(e, {})) for c in spec["conditions"])
    ]
    print(f"envs with all {len(spec['conditions'])} conditions: {envs}")
    comparisons = spec.get("comparisons", [])
    report(final, envs, comparisons, "final score")
    report(auc, envs, comparisons, "AUC")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
