# CLAUDE.md — hboot-erl (usc_erl_jax)

Extends the global CLAUDE.md. Project-specific deltas only.

## Stack

- Python 3.12+, JAX + Equinox + Optax for ERL, [SBX](https://github.com/araffin/sbx) for baselines, Hydra for config, wandb for logging.
- Package manager: `uv`. Quality tools: `ruff`, `ty` (type checking). No pytest suite; verify via smoke runs.

## Layout

```text
src/
  train.py           # Thin Hydra entrypoint (env workers re-import it)
  runner.py          # run_training + dispatch (ERL, SC-ERL, SBX baselines)
  algos/erl.py       # ERL loop: eval modes, knobs (config dataclasses), diagnostics
  analysis/          # Experiment analysis: fetch, summary, compare, cost, pilot, stats
  baselines/         # SBX agent builders, vec-env wrapping, wandb logging callback
  common/            # Buffer, fused TD3, rollout, critic surrogate, utils
  modules/           # Equinox modules (ActorHead, SharedStateEmbedding) + CEM evolutionary module
  environments/      # Environment registry: mujoco (v5), dmc (dog-*), myosuite
  conf/              # Hydra configs (config.yaml, algorithm/*.yaml)
```

## Conventions

- **Functional JAX style**: Pure functions across `algos/`, `common/`, `modules/`. Classes hold state (e.g. `CEM`), not logic.
- **Unified ERL Runner**: fitness modes `eval.mode` = `full`, `prefix` (`prefix_h`), `critic` (`p_real`: 0 = SC-ERL always, 0.21 = random gate), `rl_only` in [`src/algos/erl.py`](src/algos/erl.py). Knobs are nested frozen dataclasses (`ERLConfig.cem/rl/rl_to_ea/ea_to_rl/eval`) mirrored by `conf/algorithm/erl.yaml`, defaults = paper-v2 `erl`. Table in README.
- **Gene Flow**:
  - RL $\to$ EA: TD3 actor injected into the weakest member every `rl_to_ea.period = 5` gens (0 = never).
  - EA $\to$ RL: `ea_to_rl.mode` = `off | soft | soft_if_better`, `ea_to_rl.tau = 0.005` via `genetic_soft_update`.
  - `rl.enabled=false` = pure CEM, rejects any knob that needs the RL actor.
- **Rollout Separation**: 1 vectorized env for the RL actor (always full $H$, seeds buffer); `pop_size` vectorized envs for the ES population.
- **Thesis**: gain comes from cheap fitness evaluation, not the critic (prefix-h200 ≈ SC-ERL). `rl_only` = TD3 control (no population/gene flow). h-bootstrap, the TD horizon controller and critic uncertainty are negative results (code removed in protocol 3). Plan: `TODO.md`.
- **Metrics**: WandB x-axis is always `env_steps` (`wandb.define_metric("*", step_metric="env_steps")`). Diagnostics outside the budget: `perf/eval_rl`, `perf/eval_champion`, `perf/eval_best`, `div/action_dist`, `div/param_dist`, `div/elite_origin` (0 RL actor, 1 its descendant, 2 population), `pop/std`. Also `cost/real_frac`, `gene/ea_to_rl`, `train/critic_loss`, `train/td_error`.
- **Protocol**: runs log `config.protocol` (3 now). Analysis loads one protocol per toml, so protocol-2 runs (`paper-core-v1`, `paper-v2`) stay a separate reference set.
- **Analysis**: `python -m analysis.{fetch,summary,compare,cost,pilot} <experiment.toml>` from `src/`. One stats module (`analysis/stats.py`: IQM, unpaired stratified bootstrap, P(A>B), TOST, Holm). Prereg template `analysis/templates/experiment.toml`. HTML templates live in `analysis/templates/`, outputs next to the toml.
- **Verification**: Run `ruff check .`, `ty check`, and a tiny-budget smoke run:
  ```bash
  uv run python -m train total_steps=3000 algorithm.horizon=100 algorithm.pop_size=4 eval.interval=1000 eval.episodes=1 wandb.enabled=false
  ```

## Slurm (WCSS Lem)

- Single benchmark runner: [`slurm_run_array.sh`](slurm_run_array.sh).
- Suites (`SUITE`, task = cond×10+seed): `core` (erl, sc_erl-random, prefix-h200, 0-29), `controls` (td3 = rl_only, sc_erl-always, 0-19), `ablation-horizon` (h100/400/800, 0-29), `config-check` (erl-evorainbow, 0-9), `all` (0-79). Submit with `scripts/submit_suite.sh <suite> <env>...`.
- Paper envs: MuJoCo v5 HalfCheetah, Hopper, Walker2d, Ant, Swimmer, Humanoid. DMC Dog and MyoSuite (`.venv-myosuite`) stay registered.
- Run `uv sync` on the login node prior to submissions. Jobs execute `.venv/bin/python` directly on `lem-gpu` (1x NVIDIA H100, 8 CPUs, 16GB RAM).
- Online wandb logging default; syncs to Lustre `$PD_ROOT/wandb_logs` when offline.
