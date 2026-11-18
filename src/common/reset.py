from __future__ import annotations

from dataclasses import dataclass

import jax

from agents.td3 import TD3
from common.replay_buffer import ReplayBuffer
from common.training import LogFn
from config import ResetConfig
from metrics.diagnostics import Diagnostics
from metrics.perf import PerfMetrics

DEG_OFFSETS: tuple[int, ...] = (5_000, 20_000, 50_000)
BEFORE_OFFSET = -1
_RESET_STREAM = 104_729


@dataclass
class _Reset:
    """Bookkeeping of one fired reset until its recovery is logged."""

    index: int
    step: int
    eval_before: float
    recovered: bool = False


def _row(env_steps: int, *parts: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {"env_steps": float(env_steps)}
    for part in parts:
        merged |= part
    return merged


class ResetController:
    """Fires full TD3 resets and logs the reset/* and surrounding metrics.

    Fresh networks use keys folded from the run seed, so the training RNG
    stream is untouched. Eval and deg measurements are outside the budget.
    """

    def __init__(
        self,
        cfg: ResetConfig,
        agent: TD3,
        buffer: ReplayBuffer,
        *,
        seed: int,
        perf: PerfMetrics,
        diagnostics: Diagnostics,
        on_log: LogFn | None,
    ) -> None:
        self._cfg = cfg
        self._agent = agent
        self._buffer = buffer
        self._seed = seed
        self._perf = perf
        self._diagnostics = diagnostics
        self._on_log = on_log
        self._fired: list[_Reset] = []
        self._pending_deg: list[tuple[int, int, int]] = []

    def _log(self, row: dict[str, float]) -> None:
        if self._on_log is not None:
            self._on_log(row)

    def _fresh_key(self, index: int) -> jax.Array:
        base = jax.random.fold_in(jax.random.key(self._seed), _RESET_STREAM)
        return jax.random.fold_in(base, index)

    def maybe_reset(self, env_steps: int) -> None:
        """Reset at the first boundary past the next configured step."""
        index = len(self._fired)
        if index >= len(self._cfg.at_steps):
            return
        if env_steps < self._cfg.at_steps[index]:
            return
        agent, buffer = self._agent, self._buffer
        # no cache: the regular eval at this step must see the new networks
        before = self._perf.evaluate(agent)
        deg_before = self._diagnostics.measure_deg(agent, buffer, env_steps)
        agent.reset(self._fresh_key(index))
        after = self._perf.evaluate(agent, env_steps=env_steps)
        deg_after = self._diagnostics.measure_deg(agent, buffer, env_steps)
        record = _Reset(index, env_steps, before.rl_return)
        self._fired.append(record)
        self._pending_deg += [
            (env_steps + offset, index, offset) for offset in DEG_OFFSETS
        ]
        tag = {"reset/index": float(index)}
        self._log(
            _row(
                env_steps,
                tag,
                {"reset/deg_offset": float(BEFORE_OFFSET)},
                before.as_metrics(),
                deg_before,
            )
        )
        event = {
            "reset/event": float(env_steps),
            "reset/deg_offset": 0.0,
            "reset/eval_before": before.rl_return,
            "reset/eval_after": after.rl_return,
            "reset/drop": before.rl_return - after.rl_return,
        }
        recovery = self._recovery(record, env_steps, after.rl_return)
        self._log(
            _row(env_steps, tag, event, recovery, after.as_metrics(), deg_after)
        )

    def _recovery(
        self, record: _Reset, env_steps: int, eval_rl: float
    ) -> dict[str, float]:
        if record.recovered or eval_rl < record.eval_before:
            return {}
        record.recovered = True
        return {"reset/recovery_steps": float(env_steps - record.step)}

    def after_iteration(
        self, env_steps: int, eval_metrics: dict[str, float] | None
    ) -> None:
        """Log due post-reset deg rows and recovery rows."""
        due = [p for p in self._pending_deg if p[0] <= env_steps]
        self._pending_deg = [p for p in self._pending_deg if p[0] > env_steps]
        for _, index, offset in due:
            deg = self._diagnostics.measure_deg(
                self._agent, self._buffer, env_steps
            )
            tag = {
                "reset/index": float(index),
                "reset/deg_offset": float(offset),
            }
            self._log(_row(env_steps, tag, deg))
        if eval_metrics is None:
            return
        for record in self._fired:
            if record.step < env_steps:
                recovery = self._recovery(
                    record, env_steps, eval_metrics["perf/eval_rl"]
                )
                self._log_recovery(record, env_steps, recovery)

    def _log_recovery(
        self, record: _Reset, env_steps: int, recovery: dict[str, float]
    ) -> None:
        if recovery:
            tag = {"reset/index": float(record.index)}
            self._log(_row(env_steps, tag, recovery))

    def finish(self, env_steps: int) -> None:
        """Log recovery_steps up to the run end for unrecovered resets."""
        for record in self._fired:
            if not record.recovered:
                record.recovered = True
                steps = {"reset/recovery_steps": float(env_steps - record.step)}
                self._log_recovery(record, env_steps, steps)
