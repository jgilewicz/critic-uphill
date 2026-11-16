# Do We Need the Critic? Truncated Fitness Evaluation in Budget-Constrained ERL

[![JAX](https://img.shields.io/badge/JAX-0.11+-blue.svg)](https://github.com/google/jax)
[![Equinox](https://img.shields.io/badge/Equinox-0.13+-9cf.svg)](https://github.com/patrick-kidger/equinox)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

JAX + Equinox ERL (TD3 + CEM population on a shared embedding) with every fitness
estimator behind one `eval.mode` switch: full rollouts, truncated prefixes,
SC-ERL critic surrogates and a TD3-only control. Gene flow, CEM and the RL
side are Hydra knobs (protocol 3).

Open work: [TODO.md](TODO.md).

---

## Research question

### The bottleneck
- Step budget $10^6$, population $P=10$, horizon $H=1000$ → one generation costs $11{,}000$ steps → **~90 generations**.
- Prefix $h=200$ → $3{,}000$ steps per generation → **~330 generations** in the same budget.

### Thesis
> Under a fixed environment-step budget, the gain of surrogate-assisted ERL comes
> from cheaper fitness evaluation, not from the critic. A plain truncated prefix
> ($h=200$, no critic in selection) matches SC-ERL; critic bootstrap on the
> prefix ($h$-bootstrap) adds nothing detectable.

### Paper axis
1. **Main comparison**: ERL vs SC-ERL vs prefix-h200, TD3 (`rl_only`) as reference.
2. **Where the gain comes from**: `rl_only` (no population) and `sc_erl_always` controls.
3. **Horizon tradeoff**: $h$ → generations → score (`ablation-horizon`: $h \in \{100, 200, 400, 800\}$).
4. **Does the critic help selection**: `critic` mode (`p_real` 0 or 0.21) vs `prefix`.

Dropped: h-bootstrap and the TD-error horizon controller (negative results in `paper-core-v1`, code removed in protocol 3).

### Results so far (`paper-core-v1`, 4 envs × 10 seeds)
- Normalized final score, IQM [95% CI]: SC-ERL 0.85 [0.67, 1.00], prefix-h200 0.84 [0.67, 0.95], prefix-td 0.81 [0.63, 0.99], hboot-td 0.73 [0.53, 0.92], ERL 0.08 [−0.23, 0.37].
- All cheap estimators ≫ ERL; no separation among them.
- `hboot_td` vs `prefix_td`: P(A>B) = 0.46.
- TD controller saturates ($\delta_{\text{ema}}$ grows with return scale) → $h \approx 700$–$800$, 128–148 generations vs 330.
- Report: https://claude.ai/code/artifact/0a041ade-99cb-4345-b742-48dc4ba03d7f

### Prior negative result
- Critic uncertainty ($\sigma$ from MC-dropout / ensembles / evidential) does not track $Q$-error (AUC ≈ 0.5); uncertainty gating ≈ random gate.

---

## System Architecture

```text
               ┌──────────────────────────────────────────────────┐
               │           SharedStateEmbedding  Z(s)             │
               └────────────────────────┬─────────────────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
     ┌───────────────────────┐                     ┌───────────────────────┐
     │   TD3 Gradient Agent  │◄─ ea_to_rl (soft, tau) ─┤   CEM ES Population   │
     │      (RL Actor)       │─ rl_to_ea (every 5 gen) ►│ (10x ActorHead)       │
     └───────────┬───────────┘                     └───────────┬───────────┘
                 │ (1x VecEnv, full H)                         │ (10x VecEnv)
                 ▼                                             ▼
     ┌───────────────────────┐                     ┌───────────────────────┐
     │  Replay Buffer (1M)   │                     │ Fitness Evaluation:   │
     └───────────────────────┘                     │ • Full (H=1000)       │
                 ▲                                 │ • Prefix (h=100-800)  │
                 │ updates (fused JAX JIT)         │ • Critic surrogate    │
     ┌───────────┴───────────┐                     │   (p_real gate)       │
     │ Twin Critic Q1 / Q2   │────────────────────►│                       │
     └───────────────────────┘                     └───────────────────────┘
```

---

## Fitness evaluation (`algorithm.eval.mode`)

| Mode | Evaluation | Critic role | Real ES steps per gen |
|---|---|---|---|
| `full` | Full rollout ($H=1000$) | none | $P \times 1000$ |
| `prefix` | Truncated rollout, fixed $h$ (`eval.prefix_h`) | none | $P \times h$ |
| `critic` | Real rollout with prob. `eval.p_real`, else calibrated $Q$ on a replay batch | surrogate fitness | $p_{\text{real}} \times P \times 1000$ |
| `rl_only` | No population, no gene flow: TD3 control | none | 0 (RL actor only) |

- `critic` with `p_real=0.21` = SC-ERL random gate, `p_real=0` = always surrogate (after warmup).
- The RL actor always rolls out the full $H$ into the replay buffer (all modes).
- `rl_only` trains TD3 with `train_ratio × H` updates per episode, same networks, budget and evaluation as ERL.

## Knobs (`algorithm.*`, defaults = paper-v2 `erl`)

| Key | Values | Default |
|---|---|---|
| `pop_size` | int ≥ 2 | 10 |
| `evo_target` | `head` (CEM over ActorHead) · `policy` (embedding + head) | `head` |
| `cem.sigma_init`, `cem.damp`, `cem.damp_limit` | float | 1e-3, 1e-3, 1e-5 |
| `cem.mean_anchor` | `none` · `rl_actor` (CEM mean reset to the RL actor before every ask) | `none` |
| `rl_to_ea.period` | generations between injections, 0 = never | 5 |
| `ea_to_rl.mode` | `off` · `soft` · `soft_if_better` (only when the champion beats the RL actor under the same estimator) | `soft` |
| `ea_to_rl.tau` | float | 0.005 |
| `rl.enabled` | `false` = pure CEM, no TD3 (needs `rl_to_ea.period=0`, `ea_to_rl.mode=off`, `cem.mean_anchor=none`, `eval.mode` full or prefix) | `true` |
| `rl.*` | TD3 hyperparameters (gamma, lr, batch, warmup, train_ratio, ...) | see `conf/algorithm/erl.yaml` |
| `eval.mode`, `eval.prefix_h`, `eval.p_real` | see above | `full`, 200, 0.21 |

- Unknown keys fail at startup (Hydra struct config + dataclass constructors).
- Checkpoints (top-level, needs `rl.enabled`): `checkpoint.dir=<root> checkpoint.at=[0.3,0.6,1.0]` saves the TD3 nets (embedding, actor, both critics) and the filled replay buffer to `<root>/<env>_s<seed>/step_<N>/` (`src/common/checkpoint.py`).

---

## Environments

| Suite | Environments | Role |
|---|---|---|
| **MuJoCo v5** | `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`, `Swimmer-v5`, `Humanoid-v5` | paper benchmark |
| **DMC Dog** | `dog-stand`, `dog-walk`, `dog-trot`, `dog-run`, `dog-fetch` | registered, not in the paper plan |
| **MyoSuite** | `myoElbowPose1D6MRandom-v0`, `myoHandReachRandom-v0`, `myoHandPenTwirlRandom-v0`, `myoHandObjHoldRandom-v0`, `myoLegWalk-v0` | registered, needs `.venv-myosuite` |

---

## Metrics

### Online (wandb, x-axis `env_steps`, runs carry `config.protocol = 3`)
- Diagnostics run outside the step budget, every `eval.interval` steps.
- `perf/eval_rl`, `perf/eval_champion`, `perf/eval_best`: deterministic return of the RL actor, the CEM champion and the best of the two.
- `div/action_dist`: RMS action gap, champion vs RL actor, on 1000 fixed buffer states.
- `div/param_dist`: RMS parameter gap, CEM elite vs RL actor.
- `div/elite_origin`: champion lineage, 0 = RL actor injected this gen, 1 = descendant of an injected actor, 2 = population.
- `pop/std`, `pop/champion_fitness`: CEM dispersion and the champion's training fitness.
- `cost/real_frac`: real-step fraction used by the population.
- `gene/ea_to_rl`: 1 when the champion updated the RL actor this generation.
- `train/critic_loss`, `train/td_error`.

### Analysis (`src/analysis`, from `src/`)
```bash
python -m analysis.fetch   ../experiments/<name>/experiment.toml  # wandb -> cache
python -m analysis.summary ../experiments/<name>/experiment.toml  # summary.html
python -m analysis.compare ../experiments/<name>/experiment.toml  # compare.json
python -m analysis.cost    ../experiments/<name>/experiment.toml  # cost.html ([cost])
python -m analysis.pilot   ../experiments/<name>/experiment.toml  # report.html
```
- Final score: mean primary metric over the last `final_window` of the budget.
- Per-env normalization (0 = worst condition mean, 1 = best) when > 1 env, IQM with a stratified bootstrap CI, seeds drawn independently per condition (no pairing).
- P(A>B) over all seed pairs, two-sided bootstrap p (TOST for equivalence), Holm within a family, gatekeeping via `requires`.
- Only runs of the toml `protocol` are loaded (default 2: `paper-core-v1`, `paper-v2`).
- Preregistration template: `src/analysis/templates/experiment.toml`.

### Offline critic measures (`critic-degeneracy-v1`)
```bash
python -m offline.critic_degeneracy calibrate <toml> <ckpt_root> <out_dir> <env>  # sigma.json
python -m offline.critic_degeneracy evaluate  <toml> <ckpt_root> <out_dir> <env>  # per checkpoint JSON
python -m offline.critic_degeneracy validate  <toml> <ckpt_root> <out_dir> <env>  # re-evaluate candidates, validate_*.json (CPU, nets only)
python -m analysis.degeneracy ../experiments/critic-degeneracy-v1/experiment.toml # H1-H3, validation, degeneracy.{json,html}
```
- Actor-head perturbations at two calibrated sigma levels, true return (10 episodes, common seeds), prefix-200, TD3 and FQE critic scores with 1st/2nd-order Taylor terms around the actor.
- Validation (`[validation]`): perturbations better than the actor plus top-k by truth, TD3 and prefix, re-evaluated on fresh reset seeds with the actor (paired t CI). Regenerates perturbations from `[offline].key`, checks dist_L2 where a buffer exists.
- All parameters in the toml `[offline]`, cluster: `scripts/submit_degeneracy.sh <env>...` (td3 with checkpoints, then `slurm_offline_degeneracy.sh` via `afterok`).

---

## Quickstart

```bash
uv sync
just train erl HalfCheetah-v5                                        # full ERL
just train erl HalfCheetah-v5 algorithm.eval.mode=prefix algorithm.eval.prefix_h=200
just train erl HalfCheetah-v5 algorithm.eval.mode=rl_only            # TD3 control
just train sac Hopper-v5                                             # SBX baseline
```

Smoke run:
```bash
uv run python -m train total_steps=3000 algorithm.horizon=100 \
  algorithm.pop_size=4 eval.interval=1000 eval.episodes=1 wandb.enabled=false
```

## Cluster (SLURM on WCSS Lem)

`slurm_run_array.sh` picks conditions via `SUITE`; task ID = condition × 10 + seed.

| Suite | Conditions | Array |
|---|---|---|
| `core` | erl, sc_erl-random, prefix-h200 | 0-29 |
| `controls` | td3 (`rl_only`), sc_erl-always | 0-19 |
| `ablation-horizon` | prefix-h100, prefix-h400, prefix-h800 | 0-29 |
| `config-check` | erl-evorainbow | 0-9 |
| `all` | core + controls + ablation-horizon | 0-79 |

```bash
uv sync                                                    # login node, once
TARGET_ENV=Humanoid-v5 TAG=<tag> sbatch --array=0-29 slurm_run_array.sh
./scripts/submit_suite.sh controls HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5
```
- `submit_suite.sh` sets the array range per suite; tag defaults to `paper-<suite>-v1` (override with `TAG=`).
- `WANDB_API_KEY` is read from the environment or `wandb login` / netrc.

---

## Repository Structure

```text
├── CLAUDE.md              # Developer rules and project architecture
├── README.md              # This file
├── TODO.md                # Experiment plan and open work
├── slurm_run_array.sh     # SLURM array (suites: core, controls, ablation-horizon, all)
├── scripts/submit_suite.sh# Submit one suite across envs
├── slurm_offline_degeneracy.sh, scripts/submit_degeneracy.sh  # critic-degeneracy-v1
├── src/
│   ├── train.py           # Hydra entrypoint
│   ├── runner.py          # Training dispatch and wandb logging
│   ├── algos/erl.py       # ERL loop, eval modes, knobs, diagnostics
│   ├── analysis/          # fetch, summary, compare, cost, pilot, stats, degeneracy
│   ├── offline/           # offline critic measures on checkpoints
│   ├── baselines/         # SBX agents and logging callbacks
│   ├── common/            # Buffer, TD3, rollout, surrogate, utils
│   ├── modules/           # Equinox networks and CEM
│   ├── environments/      # MuJoCo, DMC Dog, MyoSuite registries
│   └── conf/              # Hydra configs
```

---

## License
MIT License. See [LICENSE](LICENSE) for details.
