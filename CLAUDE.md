# CLAUDE.md - critic-uphill

Extends `~/.claude/CLAUDE.md`. Overrides the global "functional JAX style" rule: this repository uses modular, class-based Python with one responsibility per class, keeping pure jitted JAX math inside focused helper functions called by the classes.

## Study Context

- **Research question**: Why do critic surrogates in evolutionary RL (ERL) fail to beat plain TD3 under a fair step budget?
- **Thesis**: The TD3 actor is trained to sit at the top of its own critic's estimate, so the critic sees every change to the actor as worse and ranks population members only by their distance from the actor, not by whether they are actually better.
- **Planned follow-ups** (do not stub or implement unless asked):
  - First experiment after diagnostics: network resets in TD3 to test if resets briefly break `deg/*` degeneration.
- **Experimental discipline**: Simple clean code, every population step counted in the budget, one frozen commit per run series, no hyperparameter changes mid-run.

## Stack and Quality Tools

- Python 3.12+, JAX + Equinox + Optax, Gymnasium MuJoCo v5, Hydra, wandb.
- Package manager: `uv`. Linters/typecheck: `ruff check .`, `ruff format --check .`, `ty check src`.
- No pytest suite yet: verify with `just check` and `just smoke`.

## Architecture and Layout

- `src/train.py`: Thin Hydra entrypoint delegating to `runner.run_training`.
- `src/runner.py`: Validates Hydra `DictConfig` into `config.RunConfig`, applies `algorithm/<agent>.yaml` `rl` defaults when `algorithm.agent != td3` and `algorithm/env/<env.id>.yaml` presets (unless CLI-overridden), initializes wandb (`env_steps` metric), and dispatches `RLTrainer` (`td3` or `sac`) or `EvoRainbowTrainer`. Under `algorithm=evorainbow algorithm.agent=sac`, `evorainbow.yaml` still carries TD3's `rl` schema at Hydra compose time, so overriding SAC-only `rl` keys requires Hydra's `+` prefix (e.g. `+algorithm.rl.alpha_lr=1e-4`).
- `src/config.py`: Frozen dataclasses (`RLConfig`, `TD3RLConfig`, `SACRLConfig`, `RLAlgoConfig`, `CEMConfig`, `RLToEAConfig`, `EAToRLConfig`, `EvalConfig`, `DegConfig`, `ProbeConfig`, `EvoRainbowAlgoConfig`, `RunConfig`) mirrored 1:1 by `src/conf/config.yaml`, `src/conf/algorithm/{td3,sac,evorainbow}.yaml`, and `src/conf/algorithm/env/<env>.yaml`. Unknown keys and invalid `algorithm.agent` fail fast.
- `src/environments/`: MuJoCo v5 registry (`HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`, `Swimmer-v5` only, others fail fast) and `EnvPool` (steps only active envs, no auto-reset).
- `src/modules/`: Equinox `SharedStateEmbedding`, `ActorHead`, `GaussianActorHead` (`log_std` clipped to `[-20, 2]`, deterministic `tanh(mean) * action_limit`), `Critic`, plus `CEM` with pure jitted `_cem_ask`/`_cem_tell`.
- `src/common/`:
  - `replay_buffer.py`: `ReplayBuffer`, `Transition`, `sample_batch`.
  - `td3.py`: Pure jitted TD3 training loop (`make_td3_train`).
  - `sac.py`: Pure jitted SAC training loop (`make_sac_train`) with automatic entropy tuning (`target_entropy = -act_dim`, init `alpha = 1.0`) and target critics only.
  - `rollout.py`: `Budget`, `rollout_single_episode`, `rollout_population`.
  - `training.py`: `run_warmup`, `print_summary`.
  - `utils.py`: `make_adam`, `critic_point`, `critic_loss`, `genetic_soft_update`.
- `src/agents/`:
  - `__init__.py`: `make_agent` factory dispatching `td3` and `sac`.
  - `base.py`: Abstract `RLAgent` owning shared `act_deterministic`, `act_population`, `q1_population_mean`, `q1_population_states`, `actor_head_flat`, and abstract action/update/head-setter interface.
  - `td3.py`: `TD3(RLAgent)` owning networks, targets, optimizers, noisy exploration, fused updates, and `soft_update_actor_from` (online + target actor heads).
  - `sac.py`: `SAC(RLAgent)` owning networks, target critics (no actor target), optimizers, stochastic exploration, fused updates, and `soft_update_actor_from` (online actor head only).
- `src/metrics/`: one class per metric group, none consume training RNG (`schedule.metric_key`) or budget steps:
  - `perf.py`: `PerfMetrics` (eval, overestimation, champion, train return) and `budget_metrics`.
  - `surrogate.py`: `SurrogateMetrics` (`pop/*`, `surr/*`, ranking keys need >=3 real evals).
  - `degeneration.py`: `DegenerationProbe.measure(agent, buffer, key)` (`deg/*`, schedule lives outside).
  - `truth.py`: `TruthProbe` (paired real episodes on `eval_env`, `probe/*`).
  - `diagnostics.py`: `Diagnostics` schedules deg and truth probes. `schedule.py`: `Schedule` (interval boundaries), `metric_key`. `stats.py`: numpy Spearman.
- `src/fitness/estimators.py`: `FitnessEstimator` protocol, `FullEpisodeFitness`, `CriticBufferFitness` (Q1 over `eval.critic_states`), `HStepBootstrapFitness` (H-step discounted return + `gamma^H * Q1`), and `IndividualControlFitness` (per-individual `eval.p_real`).
- `src/algos/`: `RLTrainer` (generic episode-by-episode RL trainer) and `EvoRainbowTrainer` (supports `algorithm.agent=td3|sac`).
- `src/analysis/`: Kept utilities (`stats.py`, `fetch.py`, `runs.py`).

## Conventions

- **Budget accounting**: Every real env step goes through one `Budget` instance (`rl_steps + pop_steps == env_steps <= total_steps`). Rollouts stop stepping finished envs immediately and stop at `budget.remaining == 0` as truncation (`done=False`), so `final env_steps <= total_steps` holds strictly.
- **Warmup**: Standalone `td3`/`sac` collects uniform-random RL episodes until `len(buffer) >= rl.warmup_steps`. `evorainbow` matches official `agent.py`: while `rl_steps < rl.warmup_steps`, runs CEM `ask`/`tell(zeros)` and uniform-random RL episodes with no population rollouts or RL updates.
- **EvoRainbow generation order**: CEM `ask` (keeping injected RL slot) -> population fitness -> CEM `tell` -> genetic soft update of RL actor head toward generation's best (skipped if best is the injected RL slot) -> one RL episode -> RL updates (`int(train_ratio * gen_steps)`) -> RL-to-EA injection into `argmin(fitness)` every `rl_to_ea.period` generations.
- **Standalone RL parity**: Standalone `td3` and `sac` run episode-by-episode (collect one RL episode, run `int(train_ratio * ep_steps)` updates), matching EvoRainbow's RL side (`embedding_dim=300`, `batch_size=128`, grad-norm clip `10`, separate embedding optimizer, `0.1`-scaled `ActorHead`/`GaussianActorHead` and `Critic` outputs, `eval.episodes=10`). SAC uses `actor_lr=critic_lr=alpha_lr=3e-4`.
- **Logging**: Runs log `config.protocol = 5` and x-axis `env_steps` (`wandb.define_metric("*", step_metric="env_steps")`). Key groups: `budget/*` (every iteration), `perf/*` (eval every `eval.interval`), `critic/*` and `actor/*` (`td_loss`, `q_mean`, `overestimation`, `loss`, SAC `alpha`), `pop/*` and `surr/*` (EvoRainbow, ranking keys need >=3 real evals), `deg/*` (every `deg.interval`, pooled plus `_s{i}` per sigma), `probe/*` (every `probe.interval`). Full key list in README.
- **Style**: 80-char line length, absolute imports only, type annotations everywhere, <=100 lines and cyclomatic complexity <=8 per function, <=5 positional parameters. No em dashes or semicolons in prose.
- **Slurm (WCSS Lem)**: `slurm_run_array.sh` and `scripts/submit_suite.sh` run suite `main` (`td3`, `sac`, `evorainbow-td3-{full,critic,hboot}`, `evorainbow-sac-{full,critic,hboot}`, array `0-79`).
