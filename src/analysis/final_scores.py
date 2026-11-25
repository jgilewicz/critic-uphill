from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from analysis.slurm_logs import parse_log
from analysis.spec import Comparison, EnvGroup, SuiteSpec
from analysis.stats import (
    holm,
    iqm,
    iqm_ci,
    iqm_diff,
    mean_ci,
    prob_improvement,
)
from analysis.wandb_runs import RunCache, RunKey

# Scores[env][label] -> {seed: value}
Scores = dict[str, dict[str, dict[int, float]]]
Series = dict[str, list[float]]


def auc(series: Series, budget: int) -> float:
    """Trapezoid area under eval vs env_steps, divided by the budget.

    NaN for runs stopped before the budget (matched runs), whose area is not
    comparable.
    """
    x = np.array(series["x"], dtype=float)
    y = np.array([np.nan if v is None else v for v in series["y"]], float)
    keep = np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 2 or x.max() < budget:
        return np.nan
    order = np.argsort(x)
    return float(np.trapezoid(y[order], x[order]) / budget)


def run_mean(series: Series, after: int) -> float:
    y = np.array(
        [
            v
            for x, v in zip(series["x"], series["y"])
            if v is not None and x > after
        ],
        dtype=float,
    )
    y = y[np.isfinite(y)]
    return float(y.mean()) if len(y) else np.nan


class FinalScoreAnalysis:
    """Normalized IQM, pre-registered comparisons, AUC and surrogate means.

    Scores are divided by the reference condition's IQM per env. Every
    bootstrap draws from one rng seeded by the spec, in spec order.
    """

    def __init__(
        self,
        spec: SuiteSpec,
        runs: dict[RunKey, dict[str, Any]],
        log_rows: list[dict[str, Any]],
    ) -> None:
        self._spec = spec
        self._runs = runs
        self._logs = {r["job"]: r for r in log_rows}
        self._label = {c: label for label, c in spec.conditions.items()}
        self._rng = np.random.default_rng(spec.rng_seed)

    def finals(self) -> dict[RunKey, float]:
        out = {}
        # logs only fill in runs of the spec tags, never add runs
        for key in sorted(self._runs):
            if key[0] not in self._label:
                continue
            final = self._runs[key]["final"]
            if final is None:
                final = self._log_final(key)
            out[key] = float(final)
        return out

    def _log_final(self, key: RunKey) -> float:
        job = self._runs[key]["slurm_job"]
        row = self._logs.get(job)
        if row is None:
            raise ValueError(
                f"{key}: no perf/eval_final in the wandb summary and no "
                f"SLURM log of job {job}, pass slurm-{job}.out to --logs"
            )
        if (row["cond"], row["env"], row["seed"]) != key:
            raise ValueError(
                f"slurm-{job}.out runs {row['cond']} {row['env']} seed "
                f"{row['seed']}, but wandb says the job ran {key}"
            )
        return row["final"]

    def _scores(self, values: dict[RunKey, float]) -> Scores:
        raw: Scores = defaultdict(lambda: defaultdict(dict))
        for (cond, env, seed), v in values.items():
            if np.isfinite(v):
                raw[env][self._label[cond]][seed] = v
        return raw

    def _normalize(self, raw: Scores, group: EnvGroup) -> Scores:
        out: Scores = {}
        for env in group.envs:
            missing = [c for c in group.conditions if c not in raw.get(env, {})]
            if missing:
                raise ValueError(f"{env}: no runs of {missing}")
            ref = iqm(np.array(list(raw[env][self._spec.reference].values())))
            out[env] = {
                c: {s: v / ref for s, v in d.items()}
                for c, d in raw[env].items()
            }
        return out

    @staticmethod
    def _strata(norm: Scores, label: str) -> list[np.ndarray]:
        return [
            np.array([d[label][s] for s in sorted(d[label])])
            for d in norm.values()
        ]

    def _iqm_table(self, norm: Scores, group: EnvGroup) -> dict[str, Any]:
        return {
            c: iqm_ci(self._strata(norm, c), self._rng)
            for c in group.conditions
        }

    def _vs_reference(self, norm: Scores, group: EnvGroup) -> dict[str, Any]:
        ref = self._strata(norm, self._spec.reference)
        out = {}
        for c in group.conditions:
            if c != self._spec.reference:
                out[c] = prob_improvement(self._strata(norm, c), ref, self._rng)
        return out

    def _comparisons(
        self,
        norm: Scores,
        group: EnvGroup,
        pairs: tuple[Comparison, ...],
        tested: bool,
    ) -> list[dict[str, Any]]:
        # Bonferroni CI over this family, an outer bound of the Holm interval
        family = self._spec.alpha / max(1, len(pairs))
        rows: list[dict[str, Any]] = [
            {
                "a": c.a,
                "b": c.b,
                **iqm_diff(
                    self._strata(norm, c.a),
                    self._strata(norm, c.b),
                    self._rng,
                    family,
                ),
            }
            for c in pairs
            if c.a in group.conditions and c.b in group.conditions
        ]
        if tested:
            ps = [r["p"] for r in rows]
            for r, reject in zip(rows, holm(ps, self._spec.alpha)):
                r["holm_reject"] = reject
                r["verdict"] = self._verdict(r, reject)
        return rows

    def _verdict(self, d: dict[str, Any], rejected: bool) -> str:
        m = self._spec.margin
        if -m <= d["lo"] and d["hi"] <= m:
            return "equivalent"
        if rejected:
            return "better" if d["point"] > 0 else "worse"
        return "inconclusive"

    def _per_env_raw(self, raw: Scores, group: EnvGroup) -> dict[str, Any]:
        return {
            env: {
                c: {
                    "mean": float(np.mean(list(raw[env][c].values()))),
                    "iqm": iqm(np.array(list(raw[env][c].values()))),
                    "n": len(raw[env][c]),
                }
                for c in group.conditions
            }
            for env in group.envs
        }

    def _aucs(self) -> dict[RunKey, float]:
        return {
            key: auc(d["series"]["perf/eval_rl"], self._spec.budget)
            for key, d in self._runs.items()
            if key[0] in self._label
        }

    def _group(self, group: EnvGroup, finals: Scores) -> dict[str, Any]:
        norm = self._normalize(finals, group)
        out = {
            "table": self._iqm_table(norm, group),
            "per_env_raw": self._per_env_raw(finals, group),
            "p_improve_vs_reference": self._vs_reference(norm, group),
            "primary": self._comparisons(
                norm, group, self._spec.primary, group.tested
            ),
            "secondary": self._comparisons(
                norm, group, self._spec.secondary, tested=False
            ),
        }
        aucs = self._scores(self._aucs())
        full_budget = tuple(
            c
            for c in group.conditions
            if all(c in aucs.get(env, {}) for env in group.envs)
        )
        auc_group = replace(group, conditions=full_budget)
        out["auc"] = self._iqm_table(
            self._normalize(aucs, auc_group), auc_group
        )
        return out

    def _surrogate(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        envs = [e for g in self._spec.groups for e in g.envs]
        for key in self._spec.surrogate_keys:
            values: dict[str, dict[str, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for (cond, env, _), d in sorted(self._runs.items()):
                if cond in self._label and env in envs:
                    v = run_mean(
                        d["series"][key], self._spec.surrogate_after_steps
                    )
                    values[self._label[cond]][env].append(v)
            out[key] = {
                label: {
                    env: mean_ci(x[np.isfinite(x)], self._rng)
                    for env, vs in by_env.items()
                    if np.isfinite(x := np.array(vs)).any()
                }
                for label, by_env in values.items()
            }
        return out

    def run(self) -> dict[str, Any]:
        finals = self.finals()
        scores = self._scores(finals)
        return {
            "n_runs": len(finals),
            "final_from_logs": sum(
                self._runs[k]["final"] is None for k in finals
            ),
            "groups": {
                g.name: self._group(g, scores) for g in self._spec.groups
            },
            "surrogate": self._surrogate(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final-score statistics of a training suite."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--logs", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = SuiteSpec.load(args.spec)
    rows = [r for p in args.logs if (r := parse_log(p)) is not None]
    result = FinalScoreAnalysis(spec, RunCache(spec, args.cache).load(), rows)
    summary = result.run()
    args.out.write_text(json.dumps(summary, indent=1))
    for name, group in summary["groups"].items():
        for r in group["primary"]:
            print(
                f"{name} | {r['a']} vs {r['b']} {r['point']:+.3f} "
                f"[{r['lo']:+.3f}, {r['hi']:+.3f}] p={r['p']:.4f} "
                f"{r.get('verdict', '')}"
            )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
