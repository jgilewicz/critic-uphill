from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from common.replay_buffer import BufferData
from common.td3 import TargetPolicy, init_shadow, make_fqe_train
from common.utils import make_adam
from metrics.degeneration import measure_terms, term_metrics
from metrics.stats import spearman
from modules.protocols import QCritic
from offline.gradient import GradientProbe
from offline.setup import OfflineSetup
from offline.signals import Signals, SignalScorer


def _take(heads: Any, idx: np.ndarray) -> Any:
    return jax.tree.map(lambda x: x[idx], heads)


def _summarize(
    signals: Signals, gains: np.ndarray, index: np.ndarray
) -> dict[str, dict[str, float]]:
    """Per signal: Spearman with truth, top-1 pick and gain vs nearest."""
    nearest = {
        name: float(gains[np.argmax(signals[name])])
        for name in ("dist_action", "dist_weight")
    }
    out: dict[str, dict[str, float]] = {}
    for name, values in signals.items():
        top = int(np.argmax(values))
        gain = float(gains[top])
        out[name] = {
            "spearman": spearman(values, gains),
            "top1": int(index[top]),
            "top1_gain": gain,
            "top1_minus_nearest_action": gain - nearest["dist_action"],
            "top1_minus_nearest_weight": gain - nearest["dist_weight"],
        }
    return out


def _truth_picks(
    signals: Signals,
    levels: np.ndarray,
    n_truth: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per level: top-1 of every signal, random fill up to n_truth."""
    picks: list[int] = []
    for level in np.unique(levels):
        idx = np.flatnonzero(levels == level)
        top = sorted({int(idx[np.argmax(v[idx])]) for v in signals.values()})
        others = np.setdiff1d(idx, top)
        n_random = min(max(n_truth - len(top), 0), len(others))
        picks += top + rng.choice(others, n_random, replace=False).tolist()
    return np.sort(np.asarray(picks))


class SignalMeasurement:
    """Signal measurement of one TD3 checkpoint.

    Scores perturbed actor heads and the stored EvoRainbow population with
    every signal, compares them with truth gains and runs the gradient probe.
    """

    def __init__(self, setup: OfflineSetup) -> None:
        self._setup = setup
        self._cfg = setup.cfg
        self._agent = setup.agent
        self._rng = setup.rng
        self._steps = {"prefix": 0, "truth": 0}
        self._seconds: dict[str, float] = {}

    def _fit_fqe(self, data: BufferData) -> QCritic:
        """Fresh twin critic fit by TD to the frozen checkpoint actor."""
        agent, cfg = self._agent, self._cfg
        settings = agent.fqe
        init_key, train_key = jax.random.split(self._setup.fqe_key)
        optimizer = make_adam(settings.critic_lr)
        hyperparams = settings.hyperparams._replace(
            batch_size=cfg.fqe_batch_size
        )
        fqe = init_shadow(
            agent.obs_dim,
            agent.action_dim,
            key=init_key,
            hidden_dims=settings.hidden_dims,
            optimizer=optimizer,
        )
        policy = TargetPolicy(agent.embedding, agent.actor_head, hyperparams)
        train = make_fqe_train(hyperparams, optimizer)
        fqe = train(fqe, policy, data, jnp.asarray(cfg.fqe_steps), train_key)
        return fqe.online[0]

    def _truth_gains(self, heads: Any, actor_returns: np.ndarray) -> np.ndarray:
        agent = self._agent
        truth = self._setup.runner.run(
            agent.embedding, agent.critic1, heads, self._setup.seeds_b
        )
        self._steps["truth"] += truth.steps
        return (truth.returns - actor_returns).mean(1)

    def _score(self, scorer: SignalScorer, heads: Any) -> Signals:
        signals, steps = scorer.score(heads)
        self._steps["prefix"] += steps
        return signals

    def _perturbations(
        self,
        scorer: SignalScorer,
        critics: dict[str, QCritic],
        states: jax.Array,
        actor_returns: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        cfg, agent = self._cfg, self._agent
        levels = np.repeat(np.arange(len(cfg.sigmas)), cfg.n_perturbations)
        terms = {
            # same key: every critic scores the same perturbed heads
            name: measure_terms(
                agent.embedding,
                critic,
                agent.actor_head,
                states,
                self._setup.perturb_key,
                sigmas=cfg.sigmas,
                n=cfg.n_perturbations,
            )
            for name, critic in critics.items()
        }
        deg: dict[str, float] = {}
        for name, t in terms.items():
            deg |= term_metrics(t, name, levels)
        heads = terms["q1"].heads
        signals = self._score(scorer, heads)
        picks = _truth_picks(signals, levels, cfg.truth_heads, self._rng)
        gains = self._truth_gains(_take(heads, picks), actor_returns)
        per_sigma = []
        for level, sigma in enumerate(cfg.sigmas):
            m = levels[picks] == level
            sub = {name: v[picks][m] for name, v in signals.items()}
            per_sigma.append(
                {
                    "sigma": sigma,
                    "n_truth": int(m.sum()),
                    "signals": _summarize(sub, gains[m], picks[m]),
                }
            )
        return deg, {
            "levels": levels.tolist(),
            "signals": {k: v.tolist() for k, v in signals.items()},
            "truth_index": picks.tolist(),
            "truth_gain": gains.tolist(),
            "per_sigma": per_sigma,
        }

    def _population(
        self, scorer: SignalScorer, actor_returns: np.ndarray
    ) -> dict[str, Any] | None:
        arrays = self._setup.ckpt.arrays
        if "pop_flat" not in arrays:
            return None
        flat = arrays["pop_flat"]
        n_gen, pop_size, dim = flat.shape
        heads = jax.vmap(self._agent.unravel_head)(
            jnp.asarray(flat.reshape(n_gen * pop_size, dim))
        )
        signals = self._score(scorer, heads)
        gains = self._truth_gains(heads, actor_returns)
        slots = np.tile(np.arange(pop_size), n_gen)
        injected = np.repeat(arrays["pop_injected"], pop_size)
        generation = np.repeat(arrays["pop_generation"], pop_size)
        return {
            "generation": generation.tolist(),
            "slot": slots.tolist(),
            "injected": (slots == injected).tolist(),
            "fitness": arrays["pop_fitness"].reshape(-1).tolist(),
            "signals": {k: v.tolist() for k, v in signals.items()},
            "truth_gain": gains.tolist(),
            "summary": _summarize(signals, gains, np.arange(len(gains))),
        }

    def _tick(self, name: str, start: float) -> float:
        now = time.perf_counter()
        self._seconds[name] = now - start
        return now

    def run(self) -> dict[str, Any]:
        """Measure every signal against truth and the gradient probe."""
        setup, cfg, agent = self._setup, self._cfg, self._agent
        start = clock = time.perf_counter()
        data = setup.buffer_data()
        states = setup.buffer_states(data)
        critics = {"q1": agent.critic1, "q2": agent.critic2}
        if agent.shadow_critic is not None:
            critics["shadow"] = agent.shadow_critic
        critics["fqe"] = self._fit_fqe(data)
        clock = self._tick("fqe", clock)
        scorer = SignalScorer(
            agent,
            critics,
            states,
            setup.runner,
            seeds=setup.seeds_a,
            h=cfg.prefix_h,
        )
        actor = setup.actor_episodes()
        self._steps["truth"] += actor.steps
        deg, perturbations = self._perturbations(
            scorer, critics, states, actor.returns
        )
        clock = self._tick("perturbations", clock)
        population = self._population(scorer, actor.returns)
        clock = self._tick("population", clock)
        assert actor.states is not None
        n_fresh = len(actor.states)
        fresh = actor.states[
            self._rng.choice(n_fresh, min(cfg.n_states, n_fresh), False)
        ]
        gradient, gradient_steps = GradientProbe(
            agent,
            setup.runner,
            seeds=setup.seeds_b,
            step_sigmas=cfg.step_sigmas,
        ).measure(states, jnp.asarray(fresh), actor.returns)
        self._tick("gradient", clock)
        self._tick("total", start)
        return setup.header() | {
            "seeds_a": setup.seeds_a.tolist(),
            "actor_returns": actor.returns[0].tolist(),
            "deg": deg,
            "perturbations": perturbations,
            "population": population,
            "gradient": gradient,
            "cost": {
                "env_steps": self._steps | {"gradient": gradient_steps},
                "seconds": self._seconds,
            },
        }
