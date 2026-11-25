# critic-uphill

[![JAX](https://img.shields.io/badge/JAX-0.11+-blue.svg)](https://github.com/google/jax)
[![Equinox](https://img.shields.io/badge/Equinox-0.13+-9cf.svg)](https://github.com/patrick-kidger/equinox)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)

Why do critic surrogates in evolutionary RL (ERL) fail to beat plain TD3 under a fair step budget? This repository holds the training code (TD3 and EvoRainbow on MuJoCo v5 and DeepMind Control tasks), the online metrics logged during training, the offline measurements run on saved checkpoints, and the scripts that turn both into the tables of the paper.

---

## Contents

- [Research question](#research-question)
- [Quickstart](#quickstart)
- [Algorithms](#algorithms)
- [Training details](#training-details)
- [Online metrics](#online-metrics-wandb)
- [Offline measurements](#offline-measurements)
- [Analysis](#analysis)
- [Suites on the cluster](#suites-on-the-cluster)
- [Layout](#layout)

---

## Research question

- Under a shared 1M-step budget, ERL can replace real population rollouts with a cheap fitness: mean `Q1` of the RL critic (`critic`), `H` real steps plus a bootstrap (`hboot`), or a short real prefix (`prefix`). Cheaper fitness buys more CEM generations.
- In practice these surrogates do not beat TD3. The question is why.

**Thesis**

> The TD3 actor is trained to sit at the top of its own critic's estimate, so the critic sees every change to the actor as worse and ranks population members only by their distance from the actor, not by whether they are actually better.

**How the code tests it**

- Online, during training: `deg/*` measures how flat or curved `Q1` is around the actor, `probe/*` checks the critic's top pick against real episodes, `surr/*` scores the surrogate's ranking of the population.
- Offline, on checkpoints: every candidate signal (`Q1`, `Q2`, an independent shadow critic, FQE, own-state `Q1`, prefix return, plain distance to the actor) ranks the same perturbed heads and population members, and real episodes say which one is right.
- Training suites: final scores of TD3, ERL and surrogate variants under the same budget, compared with pre-registered tests.

---

## Quickstart

```bash
uv sync
just check                                  # ruff check, ruff format --check, ty check src

just train td3 Hopper-v5                    # standalone TD3
just train td3-reset Hopper-v5              # TD3 with full network resets
just train evorainbow HalfCheetah-v5        # EvoRainbow (default eval.mode=hboot)
just train evorainbow Walker2d-v5 algorithm.eval.mode=critic
```

Smoke runs (`total_steps=3000`, warmup `1000`, one eval episode, wandb off):

```bash
just smoke td3 Hopper-v5
just smoke evorainbow Hopper-v5 algorithm.eval.mode=full
just smoke evorainbow Hopper-v5 algorithm.eval.mode=critic
just smoke evorainbow Hopper-v5 algorithm.eval.mode=hboot algorithm.eval.h=50
just smoke evorainbow Hopper-v5 algorithm.eval.mode=prefix algorithm.eval.h=50
```

Any Hydra key can be appended as an override. Configs live in `src/conf/` and are validated into frozen dataclasses in `src/config.py`, unknown keys fail fast.

---

## Algorithms

```text
               ┌──────────────────────────────────────────────────┐
               │           SharedStateEmbedding  Z(s)             │
               │        (Linear->LN->tanh x2, dim=300)            │
               └────────────────────────┬─────────────────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
     ┌───────────────────────┐                     ┌───────────────────────┐
     │       TD3 agent       │◄─ ea_to_rl (soft) ──┤   CEM ES Population   │
     │ (Actor head, 0.1x)    ├── rl_to_ea (period)►│  (5x ActorHead, 1.0x) │
     └───────────┬───────────┘                     └───────────┬───────────┘
                 │ (1x Env, up to H=1000)                      │ (5x EnvPool)
                 ▼                                             ▼
     ┌───────────────────────┐                     ┌───────────────────────┐
     │  Replay Buffer (1M)   │◄─ real pop steps ───┤ Fitness Evaluation:   │
     └───────────┬───────────┘                     │ • full (H=1000)       │
                 │ jitted TD3 updates              │ • critic (Q1 buffer)  │
                 ▼                                 │ • hboot (H=200 + Q1)  │
     ┌───────────────────────┐                     │ • prefix (first h)    │
     │ Twin Critic Q1 / Q2   ├────────────────────►│   (+ p_real gate)     │
     └───────────────────────┘                     └───────────────────────┘
```

| `algorithm` | Fitness of a population member | Real pop steps per generation |
|---|---|---|
| `td3` | Standalone TD3, no population | `0` |
| `td3-reset` | `td3` with full network resets at `reset.at_steps=[200k, 400k, 600k]` | `0` |
| `evorainbow`, `eval.mode=full` | Return of one full episode (up to `1000` steps) | up to `pop_size * 1000` |
| `evorainbow`, `eval.mode=critic` | Mean `Q1(s, pi_i(s))` over `50_000` replay states, or a full episode with probability `p_real` | `0` (+ full episodes) |
| `evorainbow`, `eval.mode=hboot` | `H=200`-step discounted return plus `gamma^H Q1(s_H, pi_i(s_H))`, or a full episode with probability `p_real` | up to `pop_size * 200` (+ full episodes) |
| `evorainbow`, `eval.mode=prefix` | Undiscounted return of the first `eval.h` real steps, no critic | up to `pop_size * h` |

**EvoRainbow generation** ([src/algos/evorainbow.py](src/algos/evorainbow.py)), matching the official implementation:

1. Warmup (`rl_steps < 10_000`): CEM `ask` and `tell(zeros)` plus one uniform-random RL episode, no population rollouts, no RL updates.
2. CEM `ask`: sample `pop_size` heads, writing the RL actor head into the slot chosen at the previous injection.
3. Population fitness with the selected mode (plus the per-individual `p_real` gate).
4. CEM `tell`: update mean and diagonal covariance.
5. EA to RL: genetic soft update (`ea_to_rl.tau`) of the online and target actor heads toward the generation's best (skipped when the best is the injected RL head).
6. One RL exploration episode, then `int(rl.train_ratio * gen_steps)` jitted TD3 updates.
7. RL to EA: every `rl_to_ea.period` generations, schedule the RL head for injection into `argmin(fitness)`.

**Per-env presets** (`src/conf/algorithm/env/*.yaml`, applied to `evorainbow` from `env.id` unless set on the CLI):

| Environment | `cem.damp` | `ea_to_rl.tau` | `eval.p_real` | `rl_to_ea.period` |
|---|---|---|---|---|
| `Ant-v5` | `1e-4` | `0.1` | `0.2` | `1` |
| `HalfCheetah-v5` | `1e-3` | `0.3` | `0.5` | `1` |
| `Hopper-v5` | `1e-3` | `0.3` | `0.5` | `1` |
| `Humanoid-v5` | `1e-2` | `0.1` | `0.8` | `1` |
| `Swimmer-v5` | `1e-3` | `0.3` | `0.4` | `10` |
| `Walker2d-v5` | `1e-1` | `0.1` | `0.6` | `1` |

`Humanoid-v5` values are carried over from the official `Humanoid-v2` entries in `run.sh` (`-damp=1e-2 -EA_tau=0.1 -theta=0.8`) and `parameters.py` (sync period 1). The DMC tasks have no official preset, so `evorainbow` keeps the defaults below for them.

**Environments** (`src/environments/`, any other id fails fast):

- MuJoCo v5: `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`, `Swimmer-v5`, `Humanoid-v5`.
- DMC suite through `shimmy`: `dmc-cheetah-run`, `dmc-quadruped-run`, `dmc-humanoid-run`, `dmc-dog-run` (`dmc-<domain>-<task>`, no slashes or underscores because ids end up in run names and file names).
- DMC adapter (`DmcStateEnv`): observation dict concatenated in sorted-key order as one float32 vector, actions in `[-1, 1]` mapped affinely onto the task bounds (`quadruped` has non-unit bounds), 1000-step episodes ending by truncation only (a termination raises), no action repeat, `reset(seed=...)` seeds the task's RNG.
- State only: `MUJOCO_GL=disable` is set (unless already set) before `dm_control` is imported, so no GL backend is loaded on headless nodes.
- Pinned `mujoco==3.6.0` and `dm_control==1.0.38`, the newest release that accepts it. Newer `dm_control` needs `mujoco>=3.7`, and `mujoco==4.0.1` changes fixed-seed MuJoCo v5 trajectories.

**Defaults** (`src/conf/algorithm/{td3,evorainbow}.yaml`): `horizon=1000`, `embedding_dim=300`, `gamma=0.99`, `tau=0.005`, `batch_size=128`, `warmup_steps=10_000`, `train_ratio=1.0`, `actor_lr=critic_lr=1e-3`, grad-norm clip `10`. EvoRainbow adds `pop_size=5`, `cem.sigma_init=1e-3`, `cem.damp_limit=1e-5`, `cem.elitism=false`.

---

## Training details

**Step budget**

- `total_steps` (default `1_000_000`) is a hard cap kept by one `Budget` counter: `rl_steps + pop_steps == env_steps <= total_steps`.
- Finished envs in `EnvPool` stop stepping (no auto-reset), and a rollout that hits `budget.remaining == 0` stops as truncated.
- Evaluation, `deg/*`, `probe/*`, reset measurements and offline measurements step a separate `eval_env` or no env at all, never the budget.

**Optional features, all off by default**

| Flag | What it does |
|---|---|
| `algorithm=td3-reset` (`reset.enabled=true`) | At the first episode boundary past each `reset.at_steps`, re-initializes embedding, actor, critics, targets and all optimizers. Buffer, step counter, noise and the running episode are kept. Fresh keys come from `fold_in(key(seed), stream)`, so the training RNG is untouched. Logs `reset/*`. Fails fast with `evorainbow`. |
| `algorithm.shadow.enabled=true` | Trains a second twin critic next to the TD3 critic: own init, own targets, same batches and next actions, never in the actor loss. Pure measurement: logs `deg_q2/*`, `shadow/*`, `probe/gain_*`. Works with `td3` and `evorainbow`. |
| `algorithm.matched.generations=G algorithm.matched.rl_updates=U` | Stops EvoRainbow after `G` generations with exactly `U` RL updates (post-warmup generation `g` runs `(U - done) // (G - g)` updates). Used to match ERL's generations and updates under a cheaper fitness. |
| `checkpoint.dir=<path>` | Saves `p010`, `p050`, `p100` at those fractions of the budget: `state.eqx` (full `TD3State`), `data.npz` (last `200_000` transitions, for EvoRainbow also the last 4 generations of population heads and fitness), `meta.json`. Read-only, logs are bit-identical with or without it. |

Pre-registered prediction for `td3-reset` (recorded before any run, Polish original): "deg/ratio wyraźnie rośnie zaraz po resecie i spada w ciągu kolejnych 50k kroków. Wynik końcowy jest celem drugorzędnym." In English: `deg/ratio` rises clearly right after the reset and falls within the following 50k steps. The final score is a secondary target.

---

## Online metrics (wandb)

Every run logs `config.protocol = 5` against `env_steps` (`wandb.define_metric("*", step_metric="env_steps")`). Metrics draw from their own RNG (`metric_key`) and use no budget steps. The final evaluation is stored in the run summary as `perf/eval_final`.

**Budget** (every iteration)

- `budget/steps_total`, `budget/steps_rl`, `budget/steps_pop`: cumulative steps.
- `budget/real_frac`: share of population members scored by a real episode (`1` for `full`, `0` for `td3`).
- `budget/generation`, `budget/rl_updates` (EvoRainbow only).

**Performance** (every `eval.interval=5000` steps, `eval.episodes=10`)

- `perf/eval_rl`, `perf/eval_rl_std`: deterministic RL actor return.
- `perf/train_return`: mean training episode return since the last eval.
- `perf/eval_champion`, `perf/champion_beats_actor` (EvoRainbow): best head of the last generation vs the RL actor.

**Critic and actor**

- `critic/td_loss`, `critic/q_mean`, `actor/loss`: means over each update call.
- `critic/overestimation`: `Q1(s0, pi(s0))` minus the discounted eval return.

**Surrogate quality** (EvoRainbow, per evaluated generation)

- `pop/dist_to_actor`, `pop/diversity`, `pop/actor_rank`: population geometry around the RL actor.
- `surr/rank_corr`, `surr/top1_regret`, `surr/elite_overlap`, `surr/dist_corr`: surrogate vs real return on the members with a real episode (needs at least 3). In `full` mode a metrics-only `Q1` surrogate scores every member, so these are logged every generation. `dist_corr` is Spearman of the surrogate vs `-dist_to_actor`.
- `surr/critic_share` (`hboot` only): share of the score that comes from the bootstrap term.

**Degeneration** (every `deg.interval=50_000` steps, `40` perturbed actor heads per sigma in `deg.sigmas=[0.05, 0.2]`, `2000` buffer states)

- The `Q1` change of a perturbed head splits into a first-order term `t1` and a second-order term `t2` (`dS ~ t1 + t2`, `t2 = -dist_H`).
- `deg/ratio`: median `|t1|` over median `|t2|`. Small means the actor sits at a critic maximum, so `Q1` only sees distance.
- `deg/rho_H`: Spearman of `dS` vs `t2`. `deg/frac_critic_better`: share of perturbations with `dS > 0`.
- Pooled over sigmas, plus `_s0`, `_s1` per sigma. `deg/grad_norm`, `deg/action_grad_norm`: critic gradient norms at the actor.
- With the shadow critic: the same heads and states scored by `Q2` (`deg_q2/*`) and the shadow (`shadow/*`).

**Truth probe** (every `probe.interval=100_000` steps, `8` perturbed heads, `3` paired real episodes)

- `probe/top1_true_gain`, `probe/frac_true_better`: real return gain over the actor of the critic's top pick and of all picks.
- With the shadow critic: picks are top-1 by `Q1`, `Q2`, shadow, the nearest head and random heads, logged as `probe/gain_{q1,q2,shadow,nearest,random}` and `probe/pick_*`.

**Reset** (`td3-reset` only)

- `reset/index`, `reset/event` (actual reset step), `reset/eval_before`, `reset/eval_after`, `reset/drop`.
- `reset/deg_offset`: tags the `deg/*` rows of a reset, `-1` before, `0` right after, then `5000`, `20000`, `50000` steps later.
- `reset/recovery_steps`: steps until an eval reaches `eval_before`. `reset/recovered` is `0` when the run ended first (censored).

---

## Offline measurements

`measure_offline` reads one checkpoint dir and writes one JSON. It uses no budget, its own RNG stream (`fold_in(key(seed), stream)` then `env_steps`), and its output is deterministic.

```bash
just offline <ckpt>/p050 data/signals/<run>_p050.json
just offline <ckpt>/p100 data/extra/<run>_p100.json measurement=extra
just offline <ckpt>/p100 data/pairs/<run>_p100.json measurement=pairs
```

Settings in `src/conf/offline.yaml`. Output files are named `<cond>_<env>_s<seed>_p<percent>.json`, which the analysis relies on.

**`measurement=signals`** (all checkpoints)

- Perturbed heads: `40` per sigma in `[0.02, 0.05, 0.1, 0.2]`, noise scaled by each leaf's std. For EvoRainbow checkpoints also every saved population member.
- Signals per head, higher is better:
  - `q1`, `q2`, `shadow` (if trained), `fqe`: mean critic value over `2000` buffer states. `fqe` is a fresh twin critic fit for `100_000` steps to the frozen actor.
  - `own_q1`: mean `Q1` over the head's own states (seed set A, `200`-step prefixes). `prefix_return`: return of those prefixes.
  - `dist_action`, `dist_weight`: minus the distance to the actor in action space and weight space.
- Truth: per sigma, top-1 of every signal plus random heads up to `20`, each run for `30` full episodes on seed set B, paired with the actor. Gain is the mean return difference.
- Per signal: Spearman with truth, top-1 gain, and top-1 gain minus the gain of the nearest head (`top1_minus_nearest_action`).
- `deg`: the online degeneration terms for `q1`, `q2`, `shadow`, `fqe` per sigma. `gradient`: `Q1` head gradient on buffer vs fresh actor states and the real gain of three steps along it. `cost`: env steps and seconds per phase.

**`measurement=extra`** (TD3, p050 and p100)

- `terms`: `dS`, `t1`, `t2` of the same perturbed heads (same key and buffer states as `signals`, checked against its `q1`).
- `noise`: mean `Q1` gradient on two disjoint halves of buffer states and of fresh states, cosine and norm ratio within and across, the noise floor for the `gradient` result.

**`measurement=pairs`** (TD3, p100)

- `20` antithetic pairs `+eps`, `-eps` per sigma in `[0.05, 0.1]`, each scored by `Q1` and by truth on the same seed set B.
- The odd part `(f(+eps) - f(-eps)) / 2` is the linear response, the even part the curvature.

---

## Analysis

Scripts in `src/analysis/` turn measurement JSONs and wandb runs into tables. Every interval is a 95% bootstrap CI with `10_000` draws and a fixed RNG seed, so reruns give the same numbers.

**Offline tables** (`python -m analysis.offline`)

```bash
just analyze-offline data/signals data/tables --extra data/extra --pairs data/pairs
```

| Output | Content |
|---|---|
| `signals.json` | Per env: median Spearman and top-1 gain of every signal on TD3 perturbations, ERL and SC-ERL populations, and actor perturbations of ERL checkpoints. Gradient table, `deg` ratio per sigma, share of picks equal to the nearest head, cost. |
| `signals.json` → `verdicts` | Pre-registered rules: `q1_no_better_than_nearest` (gain CI covers 0 and Spearman not above distance in at least 3 of 4 core envs), `independent_critic_helps` (`Q2` or FQE gain above 0 in at least 2), `fresh_states_help` (`own_q1` above buffer `Q1` in at least 2), `ratio_falls_like_1_over_sigma`, `gradient_step_helps_2_envs`. |
| `extra.json` | Curvature (ranking by `dS`, by `t2`, by distance), own-state `Q1` vs prefix return, gradient noise floor. |
| `pairs.json` | Spearman and slope of truth on `Q1` for the odd and even parts, the odd share of the true gain, and a check that pairs and signals used the same seed set B. |

Medians pool all checkpoints of an env, CIs resample seeds.

**Final scores** (`python -m analysis.final_scores`)

```bash
just fetch-runs                                   # cache finished runs of the spec tags into data/runs
just analyze-final data/surrogate.json            # runs with perf/eval_final in the summary
just analyze-final data/surrogate.json logs/slurm-*.out   # older runs: final eval from SLURM logs
```

- The suite spec ([src/analysis/specs/surrogate.toml](src/analysis/specs/surrogate.toml)) lists wandb tags, condition labels, the reference, env groups and the pre-registered comparisons.
- A run name found under several tags resolves to the first tag in the spec. A SLURM log fills in only the run whose wandb metadata names that log's array job and task, so logs of other suites cannot leak in.
- Scores are divided by the reference's (TD3) IQM per env. Per condition: IQM with a stratified bootstrap CI, probability of improvement over the reference, normalized AUC of `perf/eval_rl` (only conditions whose runs reach the budget, so matched runs get none).
- Primary comparisons: IQM difference, two-sided bootstrap p, Holm at `alpha=0.05`. Verdict `equivalent` if the CI lies within `±0.10`, else `better`/`worse` if Holm rejects, else `inconclusive`. A Bonferroni CI bounds the Holm interval.
- Secondary comparisons get the same numbers without a verdict. Surrogate keys (`surr/*`, `budget/real_frac`, `perf/champion_beats_actor`) are averaged per run after warmup.

---

## Suites on the cluster

`slurm_run_array.sh` runs one condition and seed per array task on WCSS Lem (`lem-gpu`, 1 GPU). Task ID = condition index × number of seeds + seed position. `SEEDS` (default `0 1 2 3 4 5 6 7 8 9`) changes the seed list. `scripts/submit_suite.sh <suite> <env>...` submits one array per env with tag `critic-uphill-<suite>-v1`.

| Suite | Array | Conditions | Purpose |
|---|---|---|---|
| `main` | `0-39` | `td3`, `evorainbow-td3-{full,critic,hboot}` | Baseline comparison |
| `reset` | `0-19` | `td3`, `td3-reset` | Do resets break degeneration? |
| `measure` | `0-29` | `td3`, `evorainbow-td3-full` (ERL), `evorainbow-td3-critic` (SC-ERL), all with shadow critic and checkpoints | Online metrics and checkpoints for the offline measurements |
| `surrogate` | `0-49` | `evorainbow-td3-critic-p0`, `evorainbow-td3-prefix-h{100,200,400}`, `evorainbow-td3-prefix-h200-matched` | Surrogate type and budget |

```bash
./scripts/submit_suite.sh main HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5 Swimmer-v5
SEEDS="10 11 12 13 14" ./scripts/submit_suite.sh measure Hopper-v5
MATCH_GENERATIONS=<G> MATCH_UPDATES=<U> ./scripts/submit_suite.sh surrogate Hopper-v5
```

- `surrogate` takes one env per call: the matched condition needs that env's median generations and RL updates of the `measure` ERL runs.
- `WANDB_API_KEY` comes from the environment or `wandb login`.

---

## Layout

```text
src/
├── train.py            # Hydra entrypoint: training
├── measure_offline.py  # Hydra entrypoint: offline measurement of one checkpoint
├── runner.py           # config validation, presets, wandb init, trainer dispatch
├── config.py           # frozen config dataclasses mirroring src/conf/
├── conf/               # config.yaml, offline.yaml, algorithm/{td3,td3-reset,evorainbow}.yaml, algorithm/env/
├── agents/             # RLAgent base, TD3 (reset, shadow critic)
├── algos/              # RLTrainer (td3) and EvoRainbowTrainer
├── common/             # replay buffer, budget and rollouts, jitted TD3 math, ResetController, checkpoints
├── environments/       # MuJoCo v5 and DMC registry, DmcStateEnv, EnvPool
├── fitness/            # full, critic, hboot, prefix fitness and the p_real gate
├── metrics/            # perf, surrogate, degeneration, truth probe, schedules
├── modules/            # Equinox embedding, actor head, critic, CEM
├── offline/            # signals, extra and pairs measurements on checkpoints
└── analysis/           # offline tables, wandb fetch, SLURM log parser, final-score statistics
```
