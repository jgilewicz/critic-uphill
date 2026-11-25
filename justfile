set dotenv-load := false

# list recipes
default:
    @just --list

# sync the venv from the lockfile
install:
    uv sync

# lint + autofix + format
lint:
    uv run ruff check --fix .
    uv run ruff format .

# lint without writing changes (CI)
lint-check:
    uv run ruff check .
    uv run ruff format --check .

# static type check
types:
    uv run ty check src

# unit tests (CPU)
test *args:
    uv run pytest -q {{args}}

# everything: lint-check, types
check: lint-check types

# train one algorithm (td3 | td3-reset | evorainbow); extra tokens are Hydra overrides
#   just train td3 Hopper-v5
#   just train td3-reset HalfCheetah-v5
#   just train evorainbow Swimmer-v5 algorithm.eval.mode=critic
#   just train evorainbow Hopper-v5 algorithm.shadow.enabled=true seed=3
train algo="evorainbow" env="HalfCheetah-v5" *overrides:
    uv run python -m train \
      algorithm={{algo}} env.id={{env}} eval_env.id={{env}} \
      {{overrides}}

# quick smoke run with wandb disabled
smoke algo="evorainbow" env="Hopper-v5" *overrides:
    uv run python -m train \
      algorithm={{algo}} env.id={{env}} eval_env.id={{env}} \
      total_steps=3000 algorithm.rl.warmup_steps=1000 \
      eval.interval=1000 eval.episodes=1 wandb.enabled=false \
      {{overrides}}

# quick XQC smoke run (3 checkpoints in CKPT) with wandb disabled
smoke-xqc env="Hopper-v5" ckpt="/tmp/xqc-smoke" *overrides:
    uv run python -m train \
      algorithm=xqc env.id={{env}} eval_env.id={{env}} \
      total_steps=3000 algorithm.start_training=1000 \
      eval.interval=1000 eval.episodes=1 wandb.enabled=false \
      checkpoint.dir={{ckpt}} {{overrides}}

# offline measurement of one checkpoint dir (signals | extra | pairs)
#   just offline checkpoints/run/p050 results/run_p050.json
#   just offline checkpoints/run/p100 results/pairs.json measurement=pairs
offline checkpoint out *overrides:
    uv run python -m measure_offline \
      checkpoint={{checkpoint}} out={{out}} {{overrides}}

# tables and verdicts from offline measurement JSONs
#   just analyze-offline data/signals data/tables
#   just analyze-offline data/signals data/tables --extra data/extra --pairs data/pairs
analyze-offline signals out *args:
    uv run python -m analysis.offline \
      --signals {{signals}} --out {{out}} {{args}}

# cache finished wandb runs (histories + perf/eval_final) of a suite spec
fetch-runs spec="src/analysis/specs/surrogate.toml" cache="data/runs":
    uv run python -m analysis.wandb_runs --spec {{spec}} --cache {{cache}}

# final-score statistics of a suite, SLURM logs fill in runs without perf/eval_final
#   just analyze-final data/surrogate.json
#   just analyze-final data/surrogate.json logs/slurm-*.out
analyze-final out *logs:
    uv run python -m analysis.final_scores \
      --spec src/analysis/specs/surrogate.toml --cache data/runs \
      --out {{out}} --logs {{logs}}
