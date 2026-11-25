# CLAUDE.md - critic-uphill

Extends `~/.claude/CLAUDE.md`. Overrides the global "functional JAX style" rule: this repository uses modular, class-based Python with one responsibility per class, keeping pure jitted JAX math inside focused helper functions called by the classes.

## Study Context

- **Research question**: Why do critic surrogates in evolutionary RL (ERL) fail to beat plain TD3 under a fair step budget?
- **Thesis**: The TD3 actor is trained to sit at the top of its own critic's estimate, so the critic sees every change to the actor as worse and ranks population members only by their distance from the actor, not by whether they are actually better.
- **First experiment after diagnostics**: `td3-reset` (full network resets in TD3) tests if resets briefly break `deg/*` degeneration.
  - Pre-registered prediction (recorded before any run): "deg/ratio wyraźnie rośnie zaraz po resecie i spada w ciągu kolejnych 50k kroków. Wynik końcowy jest celem drugorzędnym."
  - English: `deg/ratio` rises clearly right after the reset and falls within the following 50k steps. The final score is a secondary target.
  - Scope: 5 envs x 10 seeds, `td3` vs `td3-reset`, suite `reset`.
- **Experimental discipline**: Simple clean code, every population step counted in the budget, one frozen commit per run series, no hyperparameter changes mid-run.

## Stack and Quality Tools

- Python 3.12+, JAX + Equinox + Optax, Gymnasium MuJoCo v5, DMC suite via shimmy (`mujoco==3.6.0`, `dm_control==1.0.38`, `shimmy==2.0.1`), Hydra, wandb.
- Package manager: `uv`. Linters/typecheck: `ruff check .`, `ruff format --check .`, `ty check src`.
- No pytest suite yet: verify with `just check` and `just smoke`.

## Architecture and Layout

- `src/train.py`: Thin Hydra entrypoint delegating to `runner.run_training`.
- `src/measure_offline.py`: Thin Hydra entrypoint (`conf/offline.yaml`, `OfflineConfig`) running `measurement=signals|extra|pairs` (`SignalMeasurement`, `ExtraMeasurement`, `PairMeasurement`) on one checkpoint dir and writing JSON.
- `src/runner.py`: Validates Hydra `DictConfig` into `config.RunConfig`, applies `algorithm/env/<env.id>.yaml` presets (unless CLI-overridden, envs without a file such as DMC keep the `evorainbow.yaml` values), initializes wandb (`env_steps` metric, final eval in summary `perf/eval_final`), and dispatches `RLTrainer` or `EvoRainbowTrainer`.
- `src/config.py`: Frozen dataclasses (`TD3RLConfig`, `ResetConfig`, `ShadowConfig`, `RLAlgoConfig`, `CEMConfig`, `RLToEAConfig`, `EAToRLConfig`, `EvalConfig`, `MatchedConfig`, `EvoRainbowAlgoConfig`, `DegConfig`, `ProbeConfig`, `CheckpointConfig`, `RunConfig`, plus `OfflineConfig` for `conf/offline.yaml`) mirrored 1:1 by `src/conf/config.yaml`, `src/conf/algorithm/{td3,td3-reset,evorainbow}.yaml`, and `src/conf/algorithm/env/<env>.yaml`. Unknown keys fail fast.
- `src/environments/`: registry (MuJoCo v5 `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`, `Swimmer-v5`, `Humanoid-v5` in `mujoco_env.py`, DMC `dmc-{cheetah,quadruped,humanoid,dog}-run` in `dmc_env.py`, others fail fast) and `EnvPool` (steps only active envs, no auto-reset). `DmcStateEnv` wraps shimmy: sorted-key float32 flat obs, actions `[-1, 1]` rescaled to the task bounds, truncation only (termination raises), `MUJOCO_GL=disable` set before the lazy `dm_control` import. Env ids must not contain `_` (run-name regex).
- `src/modules/`: Equinox `SharedStateEmbedding`, `ActorHead`, `Critic`, plus `CEM` with pure jitted `_cem_ask`/`_cem_tell`.
- `src/common/`:
  - `replay_buffer.py`: `ReplayBuffer`, `Transition`, `sample_batch`. `add` donates the buffer arrays (in-place scatter), so never keep `buffer.data` across an `add`.
  - `td3.py`: Pure jitted TD3 training loop (`make_td3_train`). Optional `ShadowState` (`init_shadow`) in `TD3State.shadow`: TD-trained on the same batches and next action, own targets, never in the actor loss. `make_fqe_train`: same twin-critic TD update for a frozen policy (offline FQE).
  - `rollout.py`: `Budget`, `rollout_single_episode`, `rollout_population`.
  - `training.py`: `run_warmup`, `print_summary`.
  - `reset.py`: `ResetController` (standalone TD3 only): fires resets at the first episode/update boundary past `reset.at_steps`, measures eval + `deg/*` before/after and at +5k/+20k/+50k via `Diagnostics.measure_deg`, logs `reset/*`. Reset keys come from `fold_in(key(seed), stream)`, never from the training key.
  - `checkpoint.py`: `Checkpointer` (td3 agents, `checkpoint.dir`): `TD3State` (`state.eqx`), buffer tail plus last EvoRainbow generations (`data.npz`), `meta.json` at `checkpoint.fractions` of the budget. `load_checkpoint` rebuilds the agent. Read-only.
  - `utils.py`: `make_adam`, `critic_point`, `critic_loss`, `genetic_soft_update`.
- `src/agents/`:
  - `base.py`: Abstract `RLAgent` owning shared `act_deterministic`, `act_population`, `q1_population_mean`, `q1_population_states`, `actor_head_flat`, and abstract action/update/head-setter interface.
  - `td3.py`: `TD3(RLAgent)` owning networks, targets, optimizers, noisy exploration, fused updates, `reset(key)` (fresh networks, targets and optimizer states), `soft_update_actor_from` (online + target actor heads), `critic2` and, with `shadow=True`, `shadow_critic` (first online shadow head, init key `fold_in(key, stream)`).
- `src/metrics/`: one class per metric group, none consume training RNG (`schedule.metric_key`) or budget steps:
  - `perf.py`: `PerfMetrics` (eval, overestimation, champion, train return) and `budget_metrics`.
  - `surrogate.py`: `SurrogateMetrics` (`pop/*`, `surr/*`, ranking keys need >=3 real evals, `full` mode scores a metrics-only Q1 surrogate for all individuals).
  - `degeneration.py`: `DegenerationProbe.measure(agent, buffer, key)` (`deg/*`, with `shadow=True` also `deg_q2/*` and `shadow/*` on the same heads, schedule lives outside).
  - `truth.py`: `TruthProbe` (paired real episodes on `eval_env`, `probe/*`, shadow mode picks top-1 by Q1, Q2, shadow and nearest plus random, `probe/gain_*`, `probe/pick_*` indices).
  - `diagnostics.py`: `Diagnostics` schedules deg and truth probes. `schedule.py`: `Schedule` (interval boundaries), `metric_key`. `stats.py`: numpy Spearman.
- `src/fitness/estimators.py`: `FitnessEstimator` protocol, `FullEpisodeFitness`, `CriticBufferFitness` (Q1 over `eval.critic_states`, shadow head with `shadow=True`), `HStepBootstrapFitness` (H-step discounted return + `gamma^H * Q1`), `PrefixFitness` (`eval.mode=prefix`, undiscounted return of the first `eval.h` steps, no critic), and `IndividualControlFitness` (per-individual `eval.p_real`).
- `src/algos/`: `RLTrainer` (generic episode-by-episode RL trainer) and `EvoRainbowTrainer`, both building a `TD3` agent.
- `src/offline/` (never imported by training):
  - `setup.py`: `OfflineSetup` (loads checkpoint, buffer states, actor episodes on seed set B, JSON header), shared by all measurements.
  - `rollouts.py`: `HeadRunner` (every stacked head on every seed, chunked over `max_envs` envs, sums Q1 over visited states, no buffer or budget).
  - `signals.py`: `SignalScorer` (`q1`, `q2`, `shadow`, `fqe`, `own_q1`, `prefix_return`, `dist_action`, `dist_weight`, higher is better).
  - `gradient.py`: `GradientProbe` (Q1 head gradient on buffer vs fresh states, steps along the fresh gradient).
  - `measurement.py`: `SignalMeasurement` (own stream `fold_in(key(seed), OFFLINE_STREAM)` then `env_steps`, seed sets A and B disjoint, `deg` via `metrics.degeneration.measure_terms`/`term_metrics`, truth picks, population, cost).
  - `extra.py`: `ExtraMeasurement` (`dS`, `t1`, `t2` of the same perturbed heads, gradient noise floor on buffer vs fresh state halves).
  - `pairs.py`: `PairMeasurement` (antithetic `+-eps` pairs per `pair_sigmas`, `Q1` and truth on seed set B).
- `src/analysis/` (CLIs run from `src/` or via `just`):
  - `stats.py`: seed bootstrap, Spearman, OLS slope, IQM, stratified IQM CI, `iqm_diff` (two-sided p + Bonferroni CI), `holm`, `prob_improvement`, `mean_ci`.
  - `records.py`: loads measurement JSONs (`<cond>_<env>_s<seed>_p<pct>.json`), `SeedTable` (per env median over checkpoints, seed bootstrap CI).
  - `signals.py`, `extra.py`, `pairs.py`: `SignalAnalysis` (rng 0, `deg_ratio` rng 1), `ExtraAnalysis` (rng 2), `PairAnalysis` (rng 3). `offline.py` CLI writes `signals.json`, `extra.json`, `pairs.json`. RNG order is fixed, reordering changes the numbers.
  - `spec.py`: `SuiteSpec` from `specs/<suite>.toml` (tags, condition labels, reference, env groups, primary/secondary comparisons, margin). `wandb_runs.py`: `RunCache` (finished runs of the tags, duplicate names resolve to the first spec tag then newest, history of `history_keys`, `slurm_job` from wandb metadata). `slurm_logs.py`: final eval from `done |` log lines for runs without `perf/eval_final`, matched to runs by `slurm_job` only.
  - `final_scores.py`: `FinalScoreAnalysis` (IQM normalized by the reference per env, Holm on tested groups, verdict `equivalent` within `+-margin`, AUC, surrogate key means after warmup).

## Conventions

- **Budget accounting**: Every real env step goes through one `Budget` instance (`rl_steps + pop_steps == env_steps <= total_steps`). Rollouts stop stepping finished envs immediately and stop at `budget.remaining == 0` as truncation (`done=False`), so `final env_steps <= total_steps` holds strictly.
- **Reset**: `algorithm=td3-reset` sets `algorithm.reset.enabled=true` (`td3.yaml` has it `false`, bit-identical to pre-reset td3). Resets fire between updates, keep buffer, step counter, noise and the running episode, and re-init embedding, actor, critics, targets and all optimizers. `enabled=true` with `evorainbow` or `at_steps >= total_steps` fails fast. Keys: `reset/{index,event,deg_offset,eval_before,eval_after,drop,recovery_steps,recovered}`, `recovered` is `1` when an eval reached `eval_before`, `0` for the run-end censored `recovery_steps`. Reset rows carry no `perf/*`, the regular eval at the reset step logs `perf/eval_rl` from the cached post-reset eval. `deg_offset` is `-1` (before), `0` (after), `5000/20000/50000`.
- **Shadow critic**: `algorithm.shadow.enabled=true` (standalone `td3` or evorainbow + td3) trains a shadow twin critic in the agent and logs `deg_q2/*`, `shadow/*`, `probe/gain_*`. Pure measurement, never in the actor loss or fitness. Default `false`.
- **Warmup**: Standalone `td3` collects uniform-random RL episodes until `len(buffer) >= rl.warmup_steps`. `evorainbow` matches official `agent.py`: while `rl_steps < rl.warmup_steps`, runs CEM `ask`/`tell(zeros)` and uniform-random RL episodes with no population rollouts or RL updates.
- **Matched runs**: `algorithm.matched.{generations,rl_updates}` (both or neither, `0` off) stop EvoRainbow after `G` generations (warmup included) with exactly `U` RL updates. Logged as `budget/generation` and `budget/rl_updates` (EvoRainbow only, so TD3 logs stay unchanged).
- **EvoRainbow generation order**: CEM `ask` (keeping injected RL slot) -> population fitness -> CEM `tell` -> genetic soft update of RL actor head toward generation's best (skipped if best is the injected RL slot) -> one RL episode -> RL updates (`int(train_ratio * gen_steps)`, matched runs `(U - done) // (G - g)`) -> RL-to-EA injection into `argmin(fitness)` every `rl_to_ea.period` generations.
- **Standalone RL parity**: Standalone `td3` runs episode-by-episode (collect one RL episode, run `int(train_ratio * ep_steps)` updates), matching EvoRainbow's RL side (`embedding_dim=300`, `batch_size=128`, grad-norm clip `10`, separate embedding optimizer, `0.1`-scaled `ActorHead` and `Critic` outputs, `eval.episodes=10`).
- **Logging**: Runs log `config.protocol = 5` and x-axis `env_steps` (`wandb.define_metric("*", step_metric="env_steps")`). Key groups: `budget/*` (every iteration), `perf/*` (eval every `eval.interval`), `critic/*` and `actor/*` (`td_loss`, `q_mean`, `overestimation`, `loss`), `pop/*` and `surr/*` (EvoRainbow, ranking keys need >=3 real evals, `full` logs them every generation, `critic_share` is hboot only), `deg/*` (every `deg.interval`, pooled plus `_s{i}` per sigma), `probe/*` (every `probe.interval`), summary `perf/eval_final`. Full key list in README.
- **Style**: 80-char line length, absolute imports only, type annotations everywhere, <=100 lines and cyclomatic complexity <=8 per function, <=5 positional parameters. No em dashes or semicolons in prose.
- **Slurm (WCSS Lem)**: `slurm_run_array.sh` and `scripts/submit_suite.sh` run suite `main` (`td3`, `evorainbow-td3-{full,critic,hboot}`, array `0-39`), suite `reset` (`td3` `0-9`, `td3-reset` `10-19`, array `0-19` per env), suite `measure` (`td3`, `evorainbow-td3-{full,critic}`, shadow metric and checkpoints, array `0-29`) and suite `surrogate` (`evorainbow-td3-critic-p0`, `evorainbow-td3-prefix-h{100,200,400}`, `evorainbow-td3-prefix-h200-matched` needing `MATCH_GENERATIONS`/`MATCH_UPDATES`, array `0-49`, one env per submit). Tags `critic-uphill-<suite>-v1`. `SEEDS="10 11 12 13 14"` swaps the seed list (task ID = condition index * number of seeds + seed position).
