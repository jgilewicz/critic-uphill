# critic-uphill

JAX and Equinox codebase studying why critic surrogates in evolutionary RL (ERL) do not beat plain TD3 under a fair step budget.

## Research Question and Thesis

- **Question**: Why do critic surrogates in evolutionary RL (ERL) fail to beat plain TD3 under a fair 1M-step budget?
- **Thesis**: The TD3 actor is trained to sit at the top of its own critic's estimate, so the critic sees every change to the actor as worse and ranks population members only by their distance from the actor, not by whether they are actually better.
- **Measurement plan**: Compare returns of TD3 and EvoRainbow (`full`, `critic`, `hboot`) on 5 MuJoCo v5 tasks under the same 1M-step budget, alongside surrogate ranking quality and critic degeneration (`deg/*`) in a follow-up task.
- **Priorities**: Simple clean code, every population step counted in the budget, one frozen commit per run series, no hyperparameter changes mid-run.

## Implemented Algorithms

Paper reference: EvoRainbow (Li et al., ICML 2024, PMLR v235).

- **TD3** (`algorithm=td3`): Standalone TD3 using the same networks (`SharedStateEmbedding`, `ActorHead`, twin `Critic`), same TD3 hyperparameters, same warmup, and same episode-by-episode RL rollout loop as EvoRainbow's RL side.
- **EvoRainbow** (`algorithm=evorainbow`):
  - Parallel mode: RL actor and CEM population share one replay buffer.
  - Shared architecture: `SharedStateEmbedding` Z(s) trained by TD3, CEM over per-individual `ActorHead`.
  - Genetic Soft Update: `theta_rl = tau * theta_elite + (1 - tau) * theta_rl` applied to both online and target RL actor heads (`ea_to_rl.tau = 0.3`).
  - RL-to-EA injection: RL actor head injected into the weakest individual every `rl_to_ea.period` generations (`1`).
- **Population fitness modes** (`algorithm.eval.mode`):
  - `full`: One full real episode (up to `horizon=1000` steps) per individual.
  - `critic`: Critic(Buffer) from EvoRainbow. Mean of online `Q1(s, pi_i(s))` over `eval.critic_states` (default `50_000`) replay-buffer states, 0 real steps.
  - `hboot`: Critic(H-step Bootstrap) from EvoRainbow. `sum_{t=0}^{H-1} gamma^t r_t + gamma^H Q1(s_H, pi_i(s_H))` with `H = eval.h` (default `200`). Episodes terminating before `H` omit the bootstrap term. All real steps go to the buffer and count toward the budget.
  - **Generation control**: With probability `eval.p_real` (default `0.0`) per generation, the whole population is scored with `full` instead of the surrogate. Valid only for `critic` and `hboot`.

## Hyperparameters (`algorithm.*`)

| Key | Description | Default |
|---|---|---|
| `pop_size` | CEM population size | `5` |
| `horizon` | Max steps per episode | `1000` |
| `embedding_dim` | Shared state embedding Z(s) size | `256` |
| `cem.sigma_init` | Initial CEM diagonal variance | `1e-3` |
| `cem.damp`, `cem.damp_limit`, `cem.elitism` | CEM covariance damping and elitism | `1e-3`, `1e-5`, `true` |
| `rl.critic_hidden_dims` | Twin critic MLP widths | `[400, 300]` |
| `rl.gamma`, `rl.tau`, `rl.policy_freq` | Discount, Polyak tau, actor delay | `0.99`, `0.005`, `2` |
| `rl.policy_noise`, `rl.noise_clip`, `rl.exploration_noise` | Target and exploration noise | `0.2`, `0.5`, `0.1` |
| `rl.actor_lr`, `rl.critic_lr`, `rl.batch_size` | Adam LRs and batch size | `1e-3`, `1e-3`, `256` |
| `rl.buffer_capacity`, `rl.warmup_steps`, `rl.train_ratio` | Replay capacity, uniform warmup, updates/step | `1_000_000`, `10_000`, `1.0` |
| `rl_to_ea.period` | Generations between RL injections (0 disables) | `1` |
| `ea_to_rl.tau` | Genetic Soft Update weight on online + target actor | `0.3` |
| `eval.mode`, `eval.h`, `eval.critic_states`, `eval.p_real` | Population fitness estimator and generation control | `hboot`, `200`, `50_000`, `0.0` |

## Budget Accounting Rule

- `total_steps` (default `1_000_000`) is a hard cap on real environment steps.
- Every real step (warmup, RL actor rollouts, `full` episodes, `hboot` prefixes, generation-control full evals) goes through one `Budget` counter.
- Finished environments stop stepping immediately (no auto-reset dummy steps).
- Any rollout reaching the remaining budget stops immediately and marks the last transition as truncated (`done=False`), so TD targets still bootstrap.
- Result: `final env_steps <= total_steps` holds strictly, and `rl_steps + pop_steps == env_steps`.
- Deterministic performance evaluations (`perf/eval_rl` every `eval.interval` steps for `eval.episodes` episodes) run on a separate environment outside the budget.

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
    ├── conf/               # Hydra config.yaml and algorithm/{td3,evorainbow}.yaml
    ├── environments/       # MuJoCo v5 registry and EnvPool
    ├── fitness/            # FullEpisode, CriticBuffer, HStepBootstrap, GenerationControl
    └── modules/            # Equinox networks and CEM optimizer
```

## Planned Next Steps

- Online training diagnostics (separate task): surrogate ranking quality metrics and critic degeneration metrics (`deg/*`).
- First experiment after diagnostics: periodic network resets in TD3 (testing whether resets briefly break degeneration as measured by `deg/*`).
