# critic-uphill

[![JAX](https://img.shields.io/badge/JAX-0.11+-blue.svg)](https://github.com/google/jax)
[![Equinox](https://img.shields.io/badge/Equinox-0.13+-9cf.svg)](https://github.com/patrick-kidger/equinox)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)

JAX and Equinox codebase studying why critic surrogates in evolutionary RL (ERL) fail to beat plain TD3 under a fair step budget. Implements standalone TD3 and EvoRainbow (shared embedding, CEM population, and `full` / `critic` / `hboot` fitness modes) on MuJoCo v5 with Hydra configs.

---

## Research question

### Why critic surrogates stall
- Under a shared 1M-step budget, EvoRainbow replaces real population rollouts with critic-based fitness (`critic` or `hboot`) to run more CEM generations.
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
     │   (ActorHead, 0.1x)   │── rl_to_ea (period)►│  (5x ActorHead, 1.0x) │
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
- **CEM `ask()`**: Sample `pop_size` heads, restoring the pending RL-injected slot if scheduled from the previous generation.
- **Population fitness**: Evaluate heads with `full`, `critic`, or `hboot` (plus per-individual `p_real` gating).
- **CEM `tell()`**: Update mean and diagonal covariance from fitness (skipped if budget exhausted during evaluation).
- **EA-to-RL**: Genetic soft update (`ea_to_rl.tau`) of online and target RL `ActorHead` toward `argmax(fitness)` (skipped in warmup, on exhaustion, or if best is the injected RL slot).
- **RL episode and TD3 updates**: Roll out one noisy RL episode into `ReplayBuffer`, then run `int(rl.train_ratio * (pop_steps + rl_steps))` jitted TD3 updates once `rl_steps >= rl.warmup_steps`.
- **RL-to-EA**: Schedule RL `ActorHead` injection into `argmin(fitness)` for the next generation every `rl_to_ea.period` generations (if `period > 0`).

---

## Algorithms and Fitness (`algorithm`, `algorithm.eval.mode`)

| Variant | Fitness and role | Real pop steps / gen |
|---|---|---|
| `td3` | Standalone TD3 baseline (episode-by-episode rollout and updates), no population | `0` |
| `evorainbow` (`full`) | Full real episode up to `horizon=1000`, undiscounted return (`p_real=0.0`) | up to `pop_size * 1000` |
| `evorainbow` (`critic`) | Mean online `Q1(s, pi_i(s))` over `50_000` replay states, or full episode w.p. `p_real` | `0` (+ `p_real` full evals) |
| `evorainbow` (`hboot`) | `H`-step discounted return (`H=200`) + `gamma^H Q1(s_H, pi_i(s_H))`, or full episode w.p. `p_real` | up to `pop_size * 200` (+ `p_real` full evals) |

Shared defaults (`src/conf/algorithm/{td3,evorainbow}.yaml`): `pop_size=5`, `horizon=1000`, `embedding_dim=300`, `cem.sigma_init=1e-3`, `cem.damp_limit=1e-5`, `cem.elitism=false`, `rl.gamma=0.99`, `rl.tau=0.005`, `rl.batch_size=128`, `rl.warmup_steps=10_000`, `rl.train_ratio=1.0`, `rl.actor_lr=rl.critic_lr=1e-3`.

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
- Periodic deterministic evaluation (`perf/eval_rl`, `eval.episodes=10` every `eval.interval=5000` steps, seeds `seed + i`) runs on a separate `eval_env` outside the budget.
- `deg/*` measurements and the truth probe (`probe/*`) use no budget steps: the probe runs paired episodes on `eval_env` only.

---

## Quickstart

```bash
uv sync
just check                                                # ruff check, ruff format --check, ty check src

just train td3 Hopper-v5                                  # standalone TD3
just train evorainbow HalfCheetah-v5                      # EvoRainbow (default eval.mode=hboot)
just train evorainbow Walker2d-v5 algorithm.eval.mode=critic
```

Smoke runs (`total_steps=3000`, `algorithm.rl.warmup_steps=1000`, `wandb.enabled=false`):

```bash
just smoke td3 Hopper-v5
just smoke evorainbow Hopper-v5 algorithm.eval.mode=full
just smoke evorainbow Hopper-v5 algorithm.eval.mode=critic
just smoke evorainbow Hopper-v5 algorithm.eval.mode=hboot algorithm.eval.h=50
```

---

## Cluster (SLURM on WCSS Lem)

`slurm_run_array.sh` runs suite `main` (`1_000_000` steps on `lem-gpu`, 1x Hopper GPU, seeds `0-9`) with array `0-39` (`0-9`: `td3`, `10-19`: `evorainbow-full`, `20-29`: `evorainbow-critic`, `30-39`: `evorainbow-hboot`). Reads `WANDB_API_KEY` from the environment or `wandb login`.

```bash
./scripts/submit_suite.sh main HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5 Swimmer-v5
```

---

## Logged Metrics (wandb)

All runs log `config.protocol = 5` with `env_steps` as the x-axis (`wandb.define_metric("*", step_metric="env_steps")`). Metrics use no RNG from training (separate `metric_key`).

**Budget** (every iteration)

- `env_steps`: total real environment steps consumed.
- `budget/steps_total`, `budget/steps_rl`, `budget/steps_pop`: cumulative steps (`rl + pop == total`).
- `budget/real_frac`: share of population individuals scored by a real episode (0 for td3/sac, 1 for `full`).

**Performance** (every `eval.interval` steps)

- `perf/eval_rl`, `perf/eval_rl_std`: deterministic RL actor return, mean and std over `eval.episodes`.
- `perf/train_return`: mean training RL episode return since the last eval.
- `perf/eval_champion`, `perf/champion_beats_actor`: best population head of the last evaluated generation vs the RL actor (EvoRainbow). The champion runs on the embedding snapshot it was selected with.

**Critic and actor** (per update call, overestimation at eval)

- `critic/td_loss`, `critic/q_mean`: mean over the updates.
- `critic/overestimation`: mean `Q1(s0, pi(s0))` minus the discounted eval return.
- `actor/loss` (SAC also `actor/alpha`).

**Surrogate quality** (EvoRainbow, per evaluated generation)

- `pop/dist_to_actor`, `pop/diversity`, `pop/actor_rank`: population geometry (rank only when the RL actor is injected).
  - `dist_to_actor` excludes the injected RL slot, `diversity` includes it.
  - `actor_rank` is `1 + count(fitness > actor)`, so ties favour the actor.
  - SAC head distances include the unused `log_std_layer` dimensions.
- `surr/critic_share`: `mean|gamma^H Q| / mean|total score|` over the surrogate-scored individuals (H-bootstrap only).
- `surr/rank_corr`, `surr/top1_regret`, `surr/elite_overlap`, `surr/dist_corr`: only with at least 3 real evals in the generation (no pooling), on the really evaluated subset. `top1_regret` is `truth[argmax surrogate] - max(truth)` (<= 0, 0 is a perfect pick), elite size `max(1, n_real // 2)`, `dist_corr` is Spearman of surrogate vs `-dist_to_actor`. Flat generations (constant scores) log no ranking keys.

**Degeneration** (every `deg.interval=50000` steps, `deg.n_states=2000`, `deg.n_perturbations=40` per sigma in `deg.sigmas=[0.05, 0.2]`)

- `deg/ratio`: median `|first-order term|` over median `|second-order term|` of the Q1 change along each perturbation (`dS ~ t1 + t2`, `-dist_H = t2`). Not logged when the median second-order term is 0.
- `deg/rho_H`: Spearman of `dS` (critic gain of the perturbed head) vs `t2`.
- `deg/frac_critic_better`: share of perturbations with `dS > 0`.
- All three pooled over sigmas, plus `_s0`, `_s1` per sigma.
- `deg/grad_norm`, `deg/action_grad_norm`: critic gradient norms at the actor.
- `deg/*` is also measured at probe steps (the truth probe picks from the same perturbations). `DegenerationProbe.measure()` sets `.last` as a side channel.

**Truth probe** (every `probe.interval=100000` steps, `probe.n_perturbations=8`: top-1 by critic gain plus random, `probe.episodes=3` paired seeds)

- `probe/top1_true_gain`, `probe/frac_true_better`: real return gain over the actor for the perturbed heads.

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
├── common/           # ReplayBuffer, Budget, rollouts, jitted TD3 and SAC math
├── metrics/          # PerfMetrics, SurrogateMetrics, DegenerationProbe, TruthProbe, Schedule
├── conf/             # Hydra config.yaml, algorithm/{td3,evorainbow}.yaml, algorithm/env/
├── environments/     # MuJoCo v5 registry and EnvPool
├── fitness/          # FullEpisode, CriticBuffer, HStepBootstrap, IndividualControl
└── modules/          # Equinox SharedStateEmbedding, ActorHead, Critic, and CEM
```
