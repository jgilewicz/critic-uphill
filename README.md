# critic-uphill

JAX and Equinox codebase studying why critic surrogates in evolutionary RL (ERL) do not beat plain TD3 under a fair step budget.

## Research Question and Thesis

- **Question**: Why do critic surrogates in evolutionary RL (ERL) fail to beat plain TD3 under a fair 1M-step budget?
- **Thesis**: The TD3 actor is trained to sit at the top of its own critic's estimate, so the critic sees every change to the actor as worse and ranks population members only by their distance from the actor, not by whether they are actually better.
- **Measurement plan**: Compare returns of TD3 and EvoRainbow (`full`, `critic`, `hboot`) on 5 MuJoCo v5 tasks under the same 1M-step budget, alongside surrogate ranking quality and critic degeneration (`deg/*`) in a follow-up task.
- **Priorities**: Simple clean code, every population step counted in the budget, one frozen commit per run series, no hyperparameter changes mid-run.

## Implemented Algorithms

Paper reference: EvoRainbow (Li et al., ICML 2024, PMLR v235).

- **TD3** (`algorithm=td3`): Standalone TD3 using the same networks (`SharedStateEmbedding` with `tanh` after both `LayerNorm` layers and `embedding_dim=300`, `ActorHead` with `0.1`-scaled RL init, twin `Critic` `[obs+act -> 400 -> LN -> leaky_relu -> 300 -> LN -> leaky_relu -> 1]` with `0.1`-scaled output layer), separate embedding Adam optimizer (`1e-3`), grad-norm clipping at `10`, `batch_size=128`, and episode-by-episode RL rollout loop.
- **EvoRainbow** (`algorithm=evorainbow`):
  - Parallel mode: RL actor and CEM population share one replay buffer.
  - Shared architecture: `SharedStateEmbedding` Z(s) trained by TD3 actor loss, CEM over per-individual `ActorHead` (`mu_init` from an unscaled `ActorHead`, `elitism=false`, antithetic sampling iff `pop_size` is even, `damp` decaying `0.95 * damp + 0.05 * damp_limit`).
  - Warmup (`rl_steps < rl.warmup_steps`): runs CEM `ask()`/`tell(zeros)` and uniform-random RL episodes without population rollouts or TD3 updates.
  - Generation order: CEM `ask()` (keeping injected RL slot) -> population fitness -> CEM `tell()` -> genetic soft update of online and target RL actor head toward generation's best (skipped if the best is the injected RL slot) -> one noisy RL episode -> TD3 updates (`int(train_ratio * gen_steps)`) -> RL-to-EA injection into `argmin(fitness)` every `rl_to_ea.period` generations.
- **Population fitness modes** (`algorithm.eval.mode`):
  - `full`: One full real episode (up to `horizon=1000` steps) per individual.
  - `critic`: Critic(Buffer) from EvoRainbow. Mean of online `Q1(s, pi_i(s))` over `eval.critic_states` (default `50_000`) replay-buffer states, 0 real steps.
  - `hboot`: Critic(H-step Bootstrap) from EvoRainbow. `sum_{t=0}^{H-1} gamma^t r_t + gamma^H Q1(s_H, pi_i(s_H))` with `H = eval.h` (default `200`). Episodes terminating before `H` omit the bootstrap term. All real steps go to the buffer and count toward the budget.
  - **Per-individual gating**: In `critic` and `hboot` modes, each individual independently receives a real full episode with probability `eval.p_real` (per-env official `theta`), otherwise the surrogate. Invalid when `eval.mode=full`.

## Hyperparameters (`algorithm.*`)

| Key | Description | Default |
|---|---|---|
| `pop_size` | CEM population size | `5` |
| `horizon` | Max steps per episode | `1000` |
| `embedding_dim` | Shared state embedding Z(s) size | `300` |
| `cem.sigma_init` | Initial CEM diagonal variance | `1e-3` |
| `cem.damp`, `cem.damp_limit`, `cem.elitism` | CEM covariance damping and elitism | per-env, `1e-5`, `false` |
| `rl.critic_hidden_dims` | Twin critic MLP widths | `[400, 300]` |
| `rl.gamma`, `rl.tau`, `rl.policy_freq` | Discount, Polyak tau, actor delay | `0.99`, `0.005`, `2` |
| `rl.policy_noise`, `rl.noise_clip`, `rl.exploration_noise` | Target and exploration noise | `0.2`, `0.5`, `0.1` |
| `rl.actor_lr`, `rl.critic_lr`, `rl.batch_size` | Adam LRs and batch size | `1e-3`, `1e-3`, `128` |
| `rl.buffer_capacity`, `rl.warmup_steps`, `rl.train_ratio` | Replay capacity, uniform warmup, updates/step | `1_000_000`, `10_000`, `1.0` |
| `rl_to_ea.period` | Generations between RL injections (0 disables) | per-env |
| `ea_to_rl.tau` | Genetic Soft Update weight on online + target actor | per-env |
| `eval.mode`, `eval.h`, `eval.critic_states`, `eval.p_real` | Population fitness estimator and per-individual gate | `hboot`, `200`, `50_000`, per-env |

## Per-Env EvoRainbow Presets (`src/conf/algorithm/env/`)

Selected automatically from `env.id` and individually overridable via CLI:

| Env | `cem.damp` | `ea_to_rl.tau` | `eval.p_real` | `rl_to_ea.period` |
|---|---|---|---|---|
| `Ant-v5` | `1e-4` | `0.1` | `0.2` | `1` |
| `Walker2d-v5` | `1e-1` | `0.1` | `0.6` | `1` |
| `Swimmer-v5` | `1e-3` | `0.3` | `0.4` | `10` |
| `HalfCheetah-v5` | `1e-3` | `0.3` | `0.5` | `1` |
| `Hopper-v5` | `1e-3` | `0.3` | `0.5` | `1` |

## Budget Accounting Rule

- `total_steps` (default `1_000_000`) is a hard cap on real environment steps.
- Every real step (warmup, RL actor rollouts, `full` episodes, `hboot` prefixes, generation-control full evals) goes through one `Budget` counter.
- Finished environments stop stepping immediately (no auto-reset dummy steps).
- Any rollout reaching the remaining budget stops immediately and marks the last transition as truncated (`done=False`), so TD targets still bootstrap.
- Result: `final env_steps <= total_steps` holds strictly, and `rl_steps + pop_steps == env_steps`.
- Deterministic performance evaluations (`perf/eval_rl` every `eval.interval=5000` steps for `eval.episodes=10` episodes) run on a separate environment outside the budget.

## Environments

MuJoCo v5 only (any other ID fails fast):
- `HalfCheetah-v5`
- `Hopper-v5`
- `Walker2d-v5`
- `Ant-v5`
- `Swimmer-v5`

## Running

- Install and verify:
  ```bash
  uv sync
  just check
  ```
- Train locally:
  ```bash
  just train td3 Hopper-v5
  just train evorainbow HalfCheetah-v5 algorithm.eval.mode=hboot
  just train evorainbow HalfCheetah-v5 algorithm.eval.mode=critic
  just train evorainbow HalfCheetah-v5 algorithm.eval.mode=full
  ```
- Smoke run (`wandb.enabled=false`, 3000 steps):
  ```bash
  just smoke evorainbow Hopper-v5 algorithm.eval.mode=hboot algorithm.eval.h=50
  ```
- SLURM array on WCSS Lem (suite `main`: `td3`, `evorainbow-full`, `evorainbow-critic`, `evorainbow-hboot`, array `0-39`):
  ```bash
  ./scripts/submit_suite.sh main HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5 Swimmer-v5
  ```

## Repository Layout

```text
├── CLAUDE.md               # Project conventions and architecture
├── README.md               # Study overview and usage
├── justfile                # Dev, lint, typecheck, train, smoke recipes
├── pyproject.toml          # Dependencies and hatch/ruff/ty config
├── slurm_run_array.sh      # WCSS SLURM array runner (suite: main)
├── scripts/submit_suite.sh # Submit suite across envs
└── src/
    ├── train.py            # Thin Hydra entrypoint
    ├── runner.py           # Hydra config validation and wandb/trainer dispatch
    ├── config.py           # Frozen config dataclasses (1:1 with YAML)
    ├── agents/             # TD3 class wrapping jitted updates and actor/critic calls
    ├── algos/              # TD3Trainer and EvoRainbowTrainer loops
    ├── analysis/           # Wandb run fetching, loading, and bootstrap stats
    ├── common/             # ReplayBuffer, Budget, PerfEvaluator, rollouts, jitted TD3
    ├── conf/               # Hydra config.yaml, algorithm/{td3,evorainbow}.yaml, algorithm/env/
    ├── environments/       # MuJoCo v5 registry and EnvPool
    ├── fitness/            # FullEpisode, CriticBuffer, HStepBootstrap, IndividualControl
    └── modules/            # Equinox networks and CEM optimizer
```

## Planned Next Steps

- Online training diagnostics (separate task): surrogate ranking quality metrics and critic degeneration metrics (`deg/*`).
- First experiment after diagnostics: periodic network resets in TD3 (testing whether resets briefly break degeneration as measured by `deg/*`).
