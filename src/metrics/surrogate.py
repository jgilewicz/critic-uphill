from __future__ import annotations

import numpy as np

from fitness.estimators import FitnessResult
from metrics.stats import spearman

MIN_REAL_EVALS = 3


def _elite_overlap(surrogate: np.ndarray, truth: np.ndarray) -> float:
    k = max(1, len(truth) // 2)
    top_surrogate = set(np.argsort(-surrogate, kind="stable")[:k])
    top_truth = set(np.argsort(-truth, kind="stable")[:k])
    return len(top_surrogate & top_truth) / k


class SurrogateMetrics:
    """Block 2: surrogate ranking quality and population geometry (ERL)."""

    def measure(
        self,
        flat_pop: np.ndarray,
        actor_flat: np.ndarray,
        result: FitnessResult,
        injected: int | None,
    ) -> dict[str, float]:
        """Metrics of one evaluated generation, using no env steps."""
        dist = np.linalg.norm(flat_pop - actor_flat, axis=1)
        others = np.arange(len(dist)) != injected
        metrics = {
            "pop/dist_to_actor": float(dist[others].mean()),
            "pop/diversity": self._diversity(flat_pop),
        }
        if injected is not None:
            fitness = np.asarray(result.fitness)
            metrics["pop/actor_rank"] = float(
                1 + np.sum(fitness > fitness[injected])
            )
        if result.critic_share is not None:
            metrics["surr/critic_share"] = result.critic_share
        metrics |= self._ranking(result, dist)
        return metrics

    @staticmethod
    def _diversity(flat_pop: np.ndarray) -> float:
        diffs = flat_pop[:, None, :] - flat_pop[None, :, :]
        pair = np.linalg.norm(diffs, axis=-1)
        upper = np.triu_indices(len(flat_pop), k=1)
        return float(pair[upper].mean())

    @staticmethod
    def _ranking(result: FitnessResult, dist: np.ndarray) -> dict[str, float]:
        mask = result.real_mask
        if result.surrogate is None or int(mask.sum()) < MIN_REAL_EVALS:
            return {}
        truth = np.asarray(result.fitness)[mask]
        surrogate = result.surrogate[mask]
        chosen = int(np.argmax(surrogate))
        ranking = {
            "surr/rank_corr": spearman(surrogate, truth),
            "surr/top1_regret": float(truth[chosen] - truth.max()),
            "surr/elite_overlap": _elite_overlap(surrogate, truth),
            "surr/dist_corr": spearman(surrogate, -dist[mask]),
        }
        return {k: v for k, v in ranking.items() if np.isfinite(v)}
