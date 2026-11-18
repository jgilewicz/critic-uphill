# CLAUDE.md - critic-uphill

Extends `~/.claude/CLAUDE.md`. Overrides the global "functional JAX style" rule: this repository uses modular, class-based Python with one responsibility per class, keeping pure jitted JAX math inside focused helper functions called by the classes.

## Study Context

- **Research question**: Why do critic surrogates in evolutionary RL (ERL) fail to beat plain TD3 under a fair step budget?
- **Thesis**: The TD3 actor is trained to sit at the top of its own critic's estimate, so the critic sees every change to the actor as worse and ranks population members only by their distance from the actor, not by whether they are actually better.
- **Planned follow-ups** (do not stub or implement unless asked):
  - Surrogate ranking quality metrics and critic degeneration metrics (`deg/*`).
  - First experiment after diagnostics: network resets in TD3 to test if resets briefly break `deg/*` degeneration.
- **Experimental discipline**: Simple clean code, every population step counted in the budget, one frozen commit per run series, no hyperparameter changes mid-run.

## Stack and Quality Tools

- Python 3.12+, JAX + Equinox + Optax, Gymnasium MuJoCo v5, Hydra, wandb.
- Package manager: `uv`. Linters/typecheck: `ruff check .`, `ruff format --check .`, `ty check src`.
- No pytest suite yet: verify with `just check` and `just smoke`.

## Architecture and Layout

- `src/train.py`: Thin Hydra entrypoint delegating to `runner.run_training`.
- `src/runner.py`: Validates Hydra `DictConfig` into `config.RunConfig`, applies `algorithm/env/<env.id>.yaml` presets (unless CLI-overridden), initializes wandb (`env_steps` metric), and dispatches `TD3Trainer` or `EvoRainbowTrainer`.
- `src/config.py`: Frozen dataclasses (`RLConfig`, `TD3AlgoConfig`, `CEMConfig`, `RLToEAConfig`, `EAToRLConfig`, `EvalConfig`, `EvoRainbowAlgoConfig`, `RunConfig`) mirrored 1:1 by `src/conf/config.yaml`, `src/conf/algorithm/{td3,evorainbow}.yaml`, and `src/conf/algorithm/env/<env>.yaml`. Unknown keys fail fast.
- `src/environments/`: MuJoCo v5 registry (`HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`, `Swimmer-v5` only, others fail fast) and `EnvPool` (steps only active envs, no auto-reset).
- `src/modules/`: Equinox `SharedStateEmbedding`, `ActorHead`, `Critic`, plus `CEM` with pure jitted `_cem_ask`/`_cem_tell`.
- `src/common/`:
  - `replay_buffer.py`: `ReplayBuffer`, `Transition`, `sample_batch`.
  - `td3.py`: Pure jitted TD3 training loop (`make_td3_train`) and `critic_point`.
  - `rollout.py`: `Budget`, `rollout_single_episode`, `rollout_population`.
  - `training.py`: `PerfEvaluator`, `run_warmup`, `print_summary`.
  - `utils.py`: `genetic_soft_update`.
- `src/agents/td3.py`: `TD3` class owning networks, targets, optimizers, action sampling, Q1 queries, fused updates, and `soft_update_actor_from` (updates both online and target actor heads).
- `src/fitness/estimators.py`: `FitnessEstimator` protocol, `FullEpisodeFitness`, `CriticBufferFitness` (Q1 over `eval.critic_states`), `HStepBootstrapFitness` (H-step discounted return + `gamma^H * Q1`), and `IndividualControlFitness` (per-individual `eval.p_real`).
- `src/algos/`: `TD3Trainer` and `EvoRainbowTrainer`.
- `src/analysis/`: Kept utilities (`stats.py`, `fetch.py`, `runs.py`).

## Conventions

- **Budget accounting**: Every real env step goes through one `Budget` instance (`rl_steps + pop_steps == env_steps <= total_steps`). Rollouts stop stepping finished envs immediately and stop at `budget.remaining == 0` as truncation (`done=False`), so `final env_steps <= total_steps` holds strictly.
- **Warmup**: `td3` collects uniform-random RL episodes until `len(buffer) >= rl.warmup_steps`. `evorainbow` matches official `agent.py`: while `rl_steps < rl.warmup_steps`, runs CEM `ask`/`tell(zeros)` and uniform-random RL episodes with no population rollouts or TD3 updates.
- **EvoRainbow generation order**: CEM `ask` (keeping injected RL slot) -> population fitness -> CEM `tell` -> genetic soft update of online + target RL actor head toward generation's best (skipped if best is the injected RL slot) -> one noisy RL episode -> TD3 updates (`int(train_ratio * gen_steps)`) -> RL-to-EA injection into `argmin(fitness)` every `rl_to_ea.period` generations.
- **TD3 parity**: Standalone `td3` runs episode-by-episode (collect one RL episode, run `int(train_ratio * ep_steps)` updates), matching EvoRainbow's RL side (`embedding_dim=300`, `batch_size=128`, grad-norm clip `10`, separate `1e-3` embedding optimizer, `0.1`-scaled `ActorHead` and `Critic` outputs, `eval.episodes=10`).
- **Logging**: Runs log `config.protocol = 4` and x-axis `env_steps` (`wandb.define_metric("*", step_metric="env_steps")`). Currently logged: `env_steps`, `perf/eval_rl`, `cost/pop_steps`, and `train/critic_loss`.
- **Style**: 80-char line length, absolute imports only, type annotations everywhere, <=100 lines and cyclomatic complexity <=8 per function, <=5 positional parameters. No em dashes or semicolons in prose.
- **Slurm (WCSS Lem)**: `slurm_run_array.sh` and `scripts/submit_suite.sh` run suite `main` (`td3`, `evorainbow-full`, `evorainbow-critic`, `evorainbow-hboot`, array `0-39`).
