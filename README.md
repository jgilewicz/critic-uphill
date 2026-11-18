# critic-uphill

[![JAX](https://img.shields.io/badge/JAX-0.11+-blue.svg)](https://github.com/google/jax)
[![Equinox](https://img.shields.io/badge/Equinox-0.13+-9cf.svg)](https://github.com/patrick-kidger/equinox)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)

JAX and Equinox codebase studying why critic surrogates in evolutionary RL (ERL) fail to beat plain TD3 under a fair step budget. Implements standalone TD3 and EvoRainbow (shared embedding, CEM population, and `full` / `critic` / `hboot` fitness modes) on MuJoCo v5 with Hydra configs.

---

## Research question

### Why critic surrogates stall
- Under a shared 1M-step budget, EvoRainbow replaces full population rollouts with critic-based or `H`-step bootstrapped fitness (`critic` or `hboot`) to run more CEM generations.
- Comparing `td3` against `evorainbow` (`full`, `critic`, `hboot`) tests whether surrogate selection helps under a fixed budget or fails because of critic geometry around the RL actor.

### Thesis
> The TD3 actor is trained to sit at the top of its own critic's estimate, so the critic sees every change to the actor as worse and ranks population members only by their distance from the actor, not by whether they are actually better.

---

## System Architecture

```text
               ┌──────────────────────────────────────────────────┐
               │           SharedStateEmbedding  Z(s)             │
               │        (Linear->LN->tanh x2, dim=300)            │
               └────────────────────────┬─────────────────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
     ┌───────────────────────┐                     ┌───────────────────────┐
     │   TD3 Gradient Agent  │◄─ ea_to_rl (soft) ──┤   CEM ES Population   │
     │   (ActorHead, 0.1x)   ├── rl_to_ea (period)►│  (5x ActorHead, 1.0x) │
     └───────────┬───────────┘                     └───────────┬───────────┘
                 │ (1x Env, up to H=1000)                      │ (5x EnvPool)
                 ▼                                             ▼
     ┌───────────────────────┐                     ┌───────────────────────┐
     │  Replay Buffer (1M)   │◄─ real pop steps ───┤ Fitness Evaluation:   │
     └───────────┬───────────┘                     │ • full (H=1000)       │
                 │ jitted TD3 updates              │ • critic (Q1 buffer)  │
                 ▼                                 │ • hboot (H=200 + Q1)  │
     ┌───────────────────────┐                     │   (+ p_real gate)     │
     │ Twin Critic Q1 / Q2   ├────────────────────►│                       │
     └───────────────────────┘                     └───────────────────────┘
```

Generation loop in [src/algos/evorainbow.py](src/algos/evorainbow.py):
- **Warmup** (`rl_steps < 10_000`): CEM `ask` and `tell(zeros)` plus one uniform-random RL episode per generation, with no population rollouts or TD3 updates.
- **CEM `ask()`**: Sample `pop_size` heads, writing the current RL `ActorHead` into the slot scheduled from the previous generation (`pending_injection`) if set.
- **Population fitness**: Evaluate heads with `full`, `critic`, or `hboot` (plus per-individual `p_real` gating).
- **CEM `tell()`**: Update mean and diagonal covariance from fitness (skipped if budget exhausted during evaluation).
- **EA-to-RL**: Genetic soft update (`ea_to_rl.tau`) of online and target RL `ActorHead` toward `argmax(fitness)` (skipped in warmup, on exhaustion, or if best is the injected RL slot).
- **RL episode and TD3 updates**: Roll out one noisy RL episode into `ReplayBuffer`, then run `int(rl.train_ratio * gen_steps)` jitted TD3 updates (where `gen_steps` is this generation's population plus RL steps) once cumulative `rl_steps >= rl.warmup_steps`.
- **RL-to-EA**: Schedule RL `ActorHead` injection into `argmin(fitness)` for the next generation every `rl_to_ea.period` generations (if `period > 0`).

---

## Algorithms and Fitness (`algorithm`, `algorithm.eval.mode`)

| Variant | Fitness and role | Real pop steps / gen |
|---|---|---|
| `td3` | Standalone TD3 baseline (episode-by-episode rollout and updates), no population | `0` |
| `evorainbow` (`full`) | Full real episode up to `horizon=1000`, undiscounted return (`p_real=0.0`) | up to `pop_size * 1000` |
| `evorainbow` (`critic`) | Mean online `Q1(s, pi_i(s))` over `50_000` replay states, or full episode w.p. `p_real` | `0` (+ full episode w.p. `p_real`) |
| `evorainbow` (`hboot`) | `H`-step discounted return (`H=200`) + `gamma^H Q1(s_H, pi_i(s_H))`, or full episode w.p. `p_real` | up to `pop_size * 200` (+ full episode w.p. `p_real`) |

Shared defaults (`src/conf/algorithm/{td3,evorainbow}.yaml`): `horizon=1000`, `embedding_dim=300`, `rl.gamma=0.99`, `rl.tau=0.005`, `rl.batch_size=128`, `rl.warmup_steps=10_000`, `rl.train_ratio=1.0`, `rl.actor_lr=rl.critic_lr=1e-3`. EvoRainbow defaults (`src/conf/algorithm/evorainbow.yaml`): `pop_size=5`, `cem.sigma_init=1e-3`, `cem.damp_limit=1e-5`, `cem.elitism=false`.

---

## Per-Env Presets (`src/conf/algorithm/env/*.yaml`)

MuJoCo v5 only (`HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`, `Swimmer-v5`). Applied automatically from `env.id` for `evorainbow` unless overridden via CLI:

| Environment | `cem.damp` | `ea_to_rl.tau` | `eval.p_real` | `rl_to_ea.period` |
|---|---|---|---|---|
| `Ant-v5` | `1e-4` | `0.1` | `0.2` | `1` |
| `HalfCheetah-v5` | `1e-3` | `0.3` | `0.5` | `1` |
| `Hopper-v5` | `1e-3` | `0.3` | `0.5` | `1` |
| `Swimmer-v5` | `1e-3` | `0.3` | `0.4` | `10` |
| `Walker2d-v5` | `1e-1` | `0.1` | `0.6` | `1` |

---

## Step Budget

- `total_steps` (default `1_000_000`) is a hard cap enforced by one `Budget` counter (`rl_steps + pop_steps == env_steps <= total_steps`).
- Finished environments in `EnvPool` stop stepping immediately (no auto-reset dummy steps), and any rollout reaching `budget.remaining == 0` stops as truncated (`done=False`).
- Periodic deterministic evaluation (`perf/eval_rl`, `eval.episodes=10` every `eval.interval=5000` steps) runs on a separate `eval_env` outside the budget.

---

## Quickstart

```bash
uv sync
just check                                                # ruff check, ruff format --check, ty check src

just train td3 Hopper-v5                                  # standalone TD3
just train evorainbow HalfCheetah-v5                      # EvoRainbow (default eval.mode=hboot)
just train evorainbow Walker2d-v5 algorithm.eval.mode=critic
```

Smoke runs (`total_steps=3000`, `algorithm.rl.warmup_steps=1000`, `eval.interval=1000`, `eval.episodes=1`, `wandb.enabled=false`):

```bash
just smoke td3 Hopper-v5
just smoke evorainbow Hopper-v5 algorithm.eval.mode=full
just smoke evorainbow Hopper-v5 algorithm.eval.mode=critic
just smoke evorainbow Hopper-v5 algorithm.eval.mode=hboot algorithm.eval.h=50
```

---

## Cluster (SLURM on WCSS Lem)

`slurm_run_array.sh` runs suite `main` (`1_000_000` steps on `lem-gpu`, 1x NVIDIA Hopper GPU, seeds `0-9`) with array `0-39` (`0-9`: `td3`, `10-19`: `evorainbow-full`, `20-29`: `evorainbow-critic`, `30-39`: `evorainbow-hboot`). Reads `WANDB_API_KEY` from the environment or `wandb login`.

```bash
./scripts/submit_suite.sh main HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5 Swimmer-v5
```

---

## Logged Metrics (wandb)

All runs log `config.protocol = 4` with `env_steps` as the x-axis (`wandb.define_metric("*", step_metric="env_steps")`):

- `env_steps`: total real environment steps consumed (`rl_steps + pop_steps`).
- `perf/eval_rl`: mean deterministic RL return over `eval.episodes` every `eval.interval` steps.
- `cost/pop_steps`: cumulative real environment steps used by the population.
- `train/critic_loss`: mean TD3 critic loss over the episode or generation updates.

---

## Layout (`src/`)

```text
src/
├── train.py          # Hydra entrypoint
├── runner.py         # Config validation, preset loading, wandb init, trainer dispatch
├── config.py         # Frozen config dataclasses matching src/conf/ YAML
├── agents/           # TD3 agent (networks, targets, optimizers, jitted updates)
├── algos/            # TD3Trainer and EvoRainbowTrainer loops
├── analysis/         # Wandb run fetching, loading, and bootstrap stats
├── common/           # ReplayBuffer, Budget, PerfEvaluator, rollouts, jitted TD3 math
├── conf/             # Hydra config.yaml, algorithm/{td3,evorainbow}.yaml, algorithm/env/
├── environments/     # MuJoCo v5 registry and EnvPool
├── fitness/          # FullEpisode, CriticBuffer, HStepBootstrap, IndividualControl
└── modules/          # Equinox SharedStateEmbedding, ActorHead, Critic, and CEM
```
