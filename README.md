# critic-uphill

[![JAX](https://img.shields.io/badge/JAX-0.11+-blue.svg)](https://github.com/google/jax)
[![Equinox](https://img.shields.io/badge/Equinox-0.13+-9cf.svg)](https://github.com/patrick-kidger/equinox)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)

JAX and Equinox codebase studying why critic surrogates in evolutionary RL (ERL) fail to beat plain TD3 under a fair step budget. Implements standalone TD3, standalone SAC, and EvoRainbow (TD3 or SAC RL agent, shared embedding, CEM population, and `full` / `critic` / `hboot` fitness modes) on MuJoCo v5 with Hydra configs.

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
     │  RL Agent (TD3 | SAC) │◄─ ea_to_rl (soft) ──┤   CEM ES Population   │
     │ (Actor head, 0.1x)    ├── rl_to_ea (period)►│  (5x ActorHead, 1.0x) │
     └───────────┬───────────┘                     └───────────┬───────────┘
                 │ (1x Env, up to H=1000)                      │ (5x EnvPool)
                 ▼                                             ▼
     ┌───────────────────────┐                     ┌───────────────────────┐
     │  Replay Buffer (1M)   │◄─ real pop steps ───┤ Fitness Evaluation:   │
     └───────────┬───────────┘                     │ • full (H=1000)       │
                 │ jitted TD3/SAC updates          │ • critic (Q1 buffer)  │
                 ▼                                 │ • hboot (H=200 + Q1)  │
     ┌───────────────────────┐                     │   (+ p_real gate)     │
     │ Twin Critic Q1 / Q2   ├────────────────────►│                       │
     └───────────────────────┘                     └───────────────────────┘
```

Generation loop in [src/algos/evorainbow.py](src/algos/evorainbow.py):
- **Warmup** (`rl_steps < 10_000`): CEM `ask` and `tell(zeros)` plus one uniform-random RL episode per generation, with no population rollouts or RL updates.
- **CEM `ask()`**: Sample `pop_size` heads, writing the current RL actor head into the slot scheduled from the previous generation (`pending_injection`) if set.
- **Population fitness**: Evaluate heads with `full`, `critic`, or `hboot` (plus per-individual `p_real` gating).
- **CEM `tell()`**: Update mean and diagonal covariance from fitness (skipped if budget exhausted during evaluation).
- **EA-to-RL**: Genetic soft update (`ea_to_rl.tau`) of the RL actor head (online and target for TD3, online only for SAC) toward `argmax(fitness)` (skipped in warmup, on exhaustion, or if best is the injected RL slot).
- **RL episode and updates**: Roll out one exploration episode into `ReplayBuffer`, then run `int(rl.train_ratio * gen_steps)` jitted RL updates (where `gen_steps` is this generation's population plus RL steps) once cumulative `rl_steps >= rl.warmup_steps`.
- **RL-to-EA**: Schedule RL actor head injection into `argmin(fitness)` for the next generation every `rl_to_ea.period` generations (if `period > 0`).

---

## Algorithms and Fitness (`algorithm`, `algorithm.eval.mode`)

| Variant | Fitness and role | Real pop steps / gen |
|---|---|---|
| `td3` | Standalone TD3 baseline (episode-by-episode rollout and updates), no population | `0` |
| `td3-reset` (`algorithm=td3-reset`) | `td3` plus a full network reset at `algorithm.reset.at_steps=[200_000, 400_000, 600_000]` | `0` |
| `sac` | Standalone SAC baseline (same episode loop, automatic entropy tuning), no population | `0` |
| `evorainbow` (`full`) | Full real episode up to `horizon=1000`, undiscounted return (`p_real=0.0`) | up to `pop_size * 1000` |
| `evorainbow` (`critic`) | Mean online `Q1(s, pi_i(s))` over `50_000` replay states, or full episode w.p. `p_real` | `0` (+ full episode w.p. `p_real`) |
| `evorainbow` (`hboot`) | `H`-step discounted return (`H=200`) + `gamma^H Q1(s_H, pi_i(s_H))`, or full episode w.p. `p_real` | up to `pop_size * 200` (+ full episode w.p. `p_real`) |

`td3-reset` is `td3.yaml` with `algorithm.reset.enabled=true` (`td3.yaml` has `reset.enabled=false`, so `td3` is unchanged and bit-identical to the pre-reset code for a fixed seed). `reset.enabled=true` with `sac` or `evorainbow` fails fast.

- At the first episode/update boundary with `env_steps >= at_step` (never inside the fused update scan) the embedding, actor head, both critics, all target networks and all optimizer states are re-initialized with the init functions used at run start (actor head scale `0.1` etc.).
- Fresh networks take their key from `fold_in(fold_in(key(seed), stream), reset_index)`, so the training RNG stream is untouched, before and after a reset.
- Replay buffer, step counter, exploration noise and the running episode are kept. Updates continue immediately, no warmup.

**Shadow critic** (`algorithm.shadow.enabled=true`, standalone `td3` or `evorainbow` with `agent=td3`, default `false`):

- A second twin critic with the TD3 critic architecture, its own init (key `fold_in(agent_key, stream)`), own target networks and own optimizer state.
- TD-trained on the same batches and the same target-policy-smoothed next action as the main critic, target `min` of its own two target heads. Targets soft-update at the `policy_freq` cadence.
- Never enters the actor loss. Logs `shadow/*` and `deg_q2/*` (see Logged Metrics) and switches the truth probe to the shadow picks.
- `algorithm.shadow.surrogate=true` (`evorainbow` only, needs `eval.mode=critic`) scores the population with the shadow's first head instead of Q1. The `p_real` gate is unchanged. No suite uses it.

`evorainbow` takes `algorithm.agent=td3|sac` (default `td3`). With SAC, CEM evolves the full `GaussianActorHead` through its deterministic `tanh(mean)` path (rollouts and surrogates), and the genetic soft update blends into the online head only.

TD3 and EvoRainbow defaults (`src/conf/algorithm/{td3,evorainbow}.yaml`): `horizon=1000`, `embedding_dim=300`, `rl.gamma=0.99`, `rl.tau=0.005`, `rl.batch_size=128`, `rl.warmup_steps=10_000`, `rl.train_ratio=1.0`, `rl.actor_lr=rl.critic_lr=1e-3`. EvoRainbow-only defaults (`src/conf/algorithm/evorainbow.yaml`): `pop_size=5`, `cem.sigma_init=1e-3`, `cem.damp_limit=1e-5`, `cem.elitism=false`.

SAC defaults (`src/conf/algorithm/sac.yaml`): `rl.actor_lr=rl.critic_lr=rl.alpha_lr=3e-4`, learned `alpha` (init `1.0`, target entropy `-act_dim`), `log_std` clipped to `[-20, 2]`, target critics only. Under `evorainbow algorithm.agent=sac` these apply unless overridden, and SAC-only keys need Hydra's `+` prefix (e.g. `+algorithm.rl.alpha_lr=1e-4`).

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

## Experiment Notes: `td3-reset`

Pre-registered prediction (recorded before any `td3-reset` run, Polish original):

> deg/ratio wyraźnie rośnie zaraz po resecie i spada w ciągu kolejnych 50k kroków. Wynik końcowy jest celem drugorzędnym.

English translation: `deg/ratio` rises clearly right after the reset and falls within the following 50k steps. The final score is a secondary target.

- Scope: 5 envs (HalfCheetah, Hopper, Walker2d, Ant, Swimmer `-v5`) x 10 seeds, `td3` and `td3-reset`, one frozen commit, no hyperparameter changes mid-run.
- Suite `reset` (see Cluster) runs both conditions.

---

## Step Budget

- `total_steps` (default `1_000_000`) is a hard cap enforced by one `Budget` counter (`rl_steps + pop_steps == env_steps <= total_steps`).
- Finished environments in `EnvPool` stop stepping immediately (no auto-reset dummy steps), and any rollout reaching `budget.remaining == 0` stops as truncated (`done=False`).
- Periodic deterministic evaluation (`perf/eval_rl`, `eval.episodes=10` every `eval.interval=5000` steps, seeds `seed + i`) runs on a separate `eval_env` outside the budget.
- Reset measurements (`reset/*` rows: eval and `deg/*` before and after a reset and at +5k, +20k, +50k) also use no budget steps.
- `deg/*` measurements and the truth probe (`probe/*`) use no budget steps: the probe runs paired episodes on `eval_env` only.

---

## Quickstart

```bash
uv sync
just check                                                # ruff check, ruff format --check, ty check src

just train td3 Hopper-v5                                  # standalone TD3
just train sac Hopper-v5                                  # standalone SAC
just train td3-reset Hopper-v5                            # TD3 with full network resets
just train evorainbow HalfCheetah-v5                      # EvoRainbow (default eval.mode=hboot)
just train evorainbow Walker2d-v5 algorithm.eval.mode=critic
just train evorainbow Hopper-v5 algorithm.agent=sac       # EvoRainbow with SAC agent
```

Smoke runs (`total_steps=3000`, `algorithm.rl.warmup_steps=1000`, `eval.interval=1000`, `eval.episodes=1`, `wandb.enabled=false`):

```bash
just smoke td3 Hopper-v5
just smoke sac Hopper-v5
just smoke evorainbow Hopper-v5 algorithm.eval.mode=full
just smoke evorainbow Hopper-v5 algorithm.eval.mode=critic
just smoke evorainbow Hopper-v5 algorithm.eval.mode=hboot algorithm.eval.h=50
just smoke evorainbow Hopper-v5 algorithm.agent=sac algorithm.eval.mode=critic
```

---

## Cluster (SLURM on WCSS Lem)

`slurm_run_array.sh` runs suite `main` (`1_000_000` steps on `lem-gpu`, 1x NVIDIA Hopper GPU, seeds `0-9`) with array `0-79` (task ID = condition index * 10 + seed):
- `0-9`: `td3`
- `10-19`: `sac`
- `20-29`: `evorainbow-td3-full`, `30-39`: `evorainbow-td3-critic`, `40-49`: `evorainbow-td3-hboot`
- `50-59`: `evorainbow-sac-full`, `60-69`: `evorainbow-sac-critic`, `70-79`: `evorainbow-sac-hboot`

Suite `reset` (array `0-19` per env, task ID = condition index * 10 + seed): `0-9` `td3`, `10-19` `td3-reset`.

Reads `WANDB_API_KEY` from the environment or `wandb login`.

```bash
./scripts/submit_suite.sh main HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5 Swimmer-v5
./scripts/submit_suite.sh reset HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5 Swimmer-v5
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
- `surr/rank_corr`, `surr/top1_regret`, `surr/elite_overlap`, `surr/dist_corr`: only with at least 3 real evals in the generation (no pooling), on the really evaluated subset. In `full` mode the surrogate is mean Q1 over `eval.critic_states` (same score as `critic` mode, no env steps, metrics only, CEM still ranks the real returns), so every evaluated generation logs the keys. `top1_regret` is `truth[argmax surrogate] - max(truth)` (<= 0, 0 is a perfect pick), elite size `max(1, n_real // 2)`, `dist_corr` is Spearman of surrogate vs `-dist_to_actor`. Flat generations (constant scores) log no ranking keys.

**Degeneration** (every `deg.interval=50000` steps, `deg.n_states=2000`, `deg.n_perturbations=40` per sigma in `deg.sigmas=[0.05, 0.2]`)

- `deg/ratio`: median `|first-order term|` over median `|second-order term|` of the Q1 change along each perturbation (`dS ~ t1 + t2`, `-dist_H = t2`). Not logged when the median second-order term is 0.
- `deg/rho_H`: Spearman of `dS` (critic gain of the perturbed head) vs `t2`.
- `deg/frac_critic_better`: share of perturbations with `dS > 0`.
- All three pooled over sigmas, plus `_s0`, `_s1` per sigma.
- `deg/grad_norm`, `deg/action_grad_norm`: critic gradient norms at the actor.
- `deg/*` is also measured at probe steps (the truth probe picks from the same perturbations). `DegenerationProbe.measure()` sets `.last` as a side channel.
- With `algorithm.shadow.enabled=true` the same states and perturbed heads are also scored by Q2 (`deg_q2/*`) and the shadow's first head (`shadow/*`), each with the same keys as `deg/*` (`ratio`, `rho_H`, `frac_critic_better`, `_s{i}`, `grad_norm`, `action_grad_norm`). `t1`, `t2` and `dS` come from that critic.

**Reset** (`td3-reset` only, each reset logs a `before` row and an `event` row at the same `env_steps`, then later rows)

- `reset/index`: 0, 1, 2 (order of the reset), set on every `reset/*` row.
- `reset/event`: actual reset step (first boundary at or past `at_step`). The event row also holds `reset/eval_before`, `reset/eval_after`, `reset/drop` (`eval_before - eval_after`) and, if `eval_after >= eval_before`, `reset/recovery_steps = 0`.
- `reset/deg_offset`: marks the `deg/*` rows of a reset: `-1` right before (old networks), `0` right after (fresh networks, same `env_steps` and same `metric_key` as before), `5000`, `20000`, `50000` at the first boundary past `reset_step + offset`. All carry `deg/ratio`, `deg/rho_H`, `deg/frac_critic_better`, `deg/grad_norm`, `deg/action_grad_norm` (plus `_s{i}`).
- `reset/recovery_steps`: `eval_step - reset_step` of the first eval with `perf/eval_rl >= reset/eval_before`, logged once per reset. If it never happens, logged at run end as `env_steps - reset_step`.
- `reset/recovered`: on every row with `reset/recovery_steps`. `1` if an eval reached `eval_before`, `0` if `recovery_steps` is the censored run-end value.
- Reset rows carry no `perf/*`. The regular eval at the reset step logs `perf/eval_rl` of the fresh networks (cached), so `perf/eval_rl` has one value per eval step.
- The regular `eval.interval` and `deg.interval` schedules keep running, so a regular row can duplicate a reset row at the same `env_steps`.

**Truth probe** (every `probe.interval=100000` steps, `probe.n_perturbations=8`: top-1 by critic gain plus random, `probe.episodes=3` paired seeds)

- `probe/top1_true_gain`, `probe/frac_true_better`: real return gain over the actor for the perturbed heads.
- With `algorithm.shadow.enabled=true` the picks from all `deg` perturbations (both sigmas) are top-1 by Q1, Q2 and shadow `dS`, the nearest head (max Q1 `t2`, i.e. min `dist_H`), then distinct random others up to `probe.n_perturbations` (needs `>= 5`). Shared picks run once. Logs `probe/gain_q1`, `probe/gain_q2`, `probe/gain_shadow`, `probe/gain_nearest`, `probe/gain_random` (mean over the random picks), plus `top1_true_gain` (= `gain_q1`) and `frac_true_better` over all distinct picks. `probe/pick_{q1,q2,shadow,nearest}` log the picked index into the `deg` perturbations (level-major: index `// deg.n_perturbations` is the sigma level).

---

## Layout (`src/`)

```text
src/
├── train.py          # Hydra entrypoint
├── runner.py         # Config validation, preset loading, wandb init, trainer dispatch
├── config.py         # Frozen config dataclasses matching src/conf/ YAML
├── agents/           # RLAgent base, TD3 (with reset and shadow critic), SAC, make_agent factory
├── algos/            # RLTrainer (td3, sac) and EvoRainbowTrainer loops
├── analysis/         # Wandb run fetching, loading, and bootstrap stats
├── common/           # ReplayBuffer, Budget, rollouts, ResetController, jitted TD3 and SAC math
├── conf/             # Hydra config.yaml, algorithm/{td3,td3-reset,sac,evorainbow}.yaml, algorithm/env/
├── environments/     # MuJoCo v5 registry and EnvPool
├── fitness/          # FullEpisode, CriticBuffer, HStepBootstrap, IndividualControl
├── metrics/          # PerfMetrics, SurrogateMetrics, DegenerationProbe, TruthProbe, Schedule
└── modules/          # Equinox SharedStateEmbedding, ActorHead, GaussianActorHead, Critic, and CEM
```
