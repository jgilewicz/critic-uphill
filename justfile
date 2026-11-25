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

# everything: lint-check, types
check: lint-check types

# train one algorithm (td3 | sac | evorainbow); extra tokens are Hydra overrides
#   just train td3 Hopper-v5
#   just train sac HalfCheetah-v5
#   just train evorainbow Swimmer-v5 algorithm.eval.mode=critic
#   just train evorainbow Hopper-v5 algorithm.agent=sac seed=3
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

# offline E1 measurement of one checkpoint dir; extra tokens are Hydra overrides
#   just offline checkpoints/run/p050 results/run_p050.json
offline checkpoint out *overrides:
    uv run python -m measure_offline \
      checkpoint={{checkpoint}} out={{out}} {{overrides}}
