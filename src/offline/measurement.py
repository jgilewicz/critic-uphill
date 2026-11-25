from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from common.checkpoint import BUFFER_COLUMNS, LoadedCheckpoint
from common.replay_buffer import BufferData
from common.td3 import init_shadow, make_fqe_train
from common.utils import make_adam
from config import OfflineConfig
from metrics.degeneration import measure_terms, term_metrics
from metrics.stats import spearman
from modules.deep_modules import Critic
from offline.gradient import GradientProbe
from offline.rollouts import HeadRunner
from offline.signals import Signals, SignalScorer

# own stream: offline draws never overlap training or metric keys
OFFLINE_STREAM = 15_485_863


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


class OfflineMeasurement:
    """E1 offline measurement of one TD3 checkpoint (points 3 and 4).

    Uses its own RNG stream and no training budget. Truth is real episodes
    on seed set B paired with the checkpoint actor, prefix signals use the
    disjoint seed set A.
    """

    def __init__(self, cfg: OfflineConfig, ckpt: LoadedCheckpoint) -> None:
        self._cfg = cfg
        self._ckpt = ckpt
        self._agent = ckpt.agent
        run = ckpt.meta["run_config"]
        base = jax.random.fold_in(jax.random.key(run["seed"]), OFFLINE_STREAM)
        key = jax.random.fold_in(base, ckpt.meta["env_steps"])
        np_key, self._perturb_key, self._fqe_key = jax.random.split(key, 3)
        self._rng = np.random.default_rng(
            int(jax.random.randint(np_key, (), 0, 2**31 - 1))
        )
        n_seeds = cfg.truth_episodes + cfg.prefix_seeds
        seeds = int(self._rng.integers(0, 2**31 - 1 - n_seeds)) + np.arange(
            n_seeds
        )
        self._seeds_b = seeds[: cfg.truth_episodes]
        self._seeds_a = seeds[cfg.truth_episodes :]
        self._runner = HeadRunner(run["env"]["id"], cfg.max_envs)
        self._steps = {"prefix": 0, "truth": 0}
        self._seconds: dict[str, float] = {}

    def _buffer_data(self) -> BufferData:
        arrays = self._ckpt.arrays
        s, a, r, ns, d = (
            jnp.asarray(arrays[f"buffer_{name}"]) for name in BUFFER_COLUMNS
        )
        return s, a, r, ns, d

    def _fit_fqe(self, data: BufferData) -> Critic:
        """Fresh twin critic fit by TD to the frozen checkpoint actor."""
        agent, cfg = self._agent, self._cfg
        init_key, train_key = jax.random.split(self._fqe_key)
        optimizer = make_adam(agent.rl_cfg.critic_lr)
        fqe = init_shadow(
            agent.obs_dim,
            agent.action_dim,
            key=init_key,
            hidden_dims=agent.rl_cfg.critic_hidden_dims,
            optimizer=optimizer,
        )
        train = make_fqe_train(
            agent.hyperparams._replace(batch_size=cfg.fqe_batch_size),
            optimizer,
        )
        fqe = train(
            fqe, agent.state.online, data, jnp.asarray(cfg.fqe_steps), train_key
        )
        return fqe.online[0]

    def _truth_gains(self, heads: Any, actor_returns: np.ndarray) -> np.ndarray:
        agent = self._agent
        truth = self._runner.run(
            agent.embedding, agent.critic1, heads, self._seeds_b
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
        critics: dict[str, Critic],
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
                self._perturb_key,
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
        arrays = self._ckpt.arrays
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
        cfg, agent, rng = self._cfg, self._agent, self._rng
        start = clock = time.perf_counter()
        data = self._buffer_data()
        n_rows = int(data[0].shape[0])
        states = data[0][rng.choice(n_rows, min(cfg.n_states, n_rows), False)]
        critics = {"q1": agent.critic1, "q2": agent.critic2}
        if agent.shadow_critic is not None:
            critics["shadow"] = agent.shadow_critic
        critics["fqe"] = self._fit_fqe(data)
        clock = self._tick("fqe", clock)
        scorer = SignalScorer(
            agent,
            critics,
            states,
            self._runner,
            seeds=self._seeds_a,
            h=cfg.prefix_h,
        )
        actor = self._runner.run(
            agent.embedding,
            agent.critic1,
            jax.tree.map(lambda x: x[None], agent.actor_head),
            self._seeds_b,
            keep_states=True,
        )
        self._steps["truth"] += actor.steps
        assert actor.states is not None
        deg, perturbations = self._perturbations(
            scorer, critics, states, actor.returns
        )
        clock = self._tick("perturbations", clock)
        population = self._population(scorer, actor.returns)
        clock = self._tick("population", clock)
        n_fresh = len(actor.states)
        fresh = actor.states[
            rng.choice(n_fresh, min(cfg.n_states, n_fresh), False)
        ]
        gradient, gradient_steps = GradientProbe(
            agent,
            self._runner,
            seeds=self._seeds_b,
            step_sigmas=cfg.step_sigmas,
        ).measure(states, jnp.asarray(fresh), actor.returns)
        self._tick("gradient", clock)
        self._tick("total", start)
        meta = self._ckpt.meta
        return {
            "checkpoint": cfg.checkpoint,
            "env_id": meta["run_config"]["env"]["id"],
            "seed": meta["run_config"]["seed"],
            "fraction": meta["fraction"],
            "env_steps": meta["env_steps"],
            "seeds_a": self._seeds_a.tolist(),
            "seeds_b": self._seeds_b.tolist(),
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
