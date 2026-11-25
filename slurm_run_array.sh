#!/bin/bash -l
# critic-uphill benchmark array on WCSS Lem (MuJoCo v5).
# Envs: HalfCheetah-v5, Hopper-v5, Walker2d-v5, Ant-v5, Swimmer-v5
#
# Suite: main (task ID = condition_index * 10 + seed, array 0-39)
#   0-9   td3
#   10-19 evorainbow-td3-full
#   20-29 evorainbow-td3-critic
#   30-39 evorainbow-td3-hboot
#
# Suite: reset (task ID = condition_index * 10 + seed, array 0-19)
#   0-9   td3
#   10-19 td3-reset
#
# Suite: measure (array 0-29), shadow critic as a metric, checkpoints saved to
# <PD>/checkpoints/<TAG>/<run> at 10%, 50%, 100% of the budget
#   0-9   td3
#   10-19 evorainbow-td3-full (ERL)
#   20-29 evorainbow-td3-critic (SC-ERL, per-env p_real gate)
#
# Suite: surrogate (array 0-49), shadow critic as a metric
#   0-9   evorainbow-td3-critic-p0 (Q1 surrogate always)
#   10-19 evorainbow-td3-prefix-h100
#   20-29 evorainbow-td3-prefix-h200
#   30-39 evorainbow-td3-prefix-h400
#   40-49 evorainbow-td3-prefix-h200-matched (needs MATCH_GENERATIONS and
#         MATCH_UPDATES: median generations and RL updates of the measure ERL
#         runs in this env)
#
# Example usage:
#   TARGET_ENV=Hopper-v5 TAG=main-v1 sbatch --array=0-39 slurm_run_array.sh
#   SUITE=measure SEEDS="10 11 12 13 14" TARGET_ENV=Hopper-v5 \
#     sbatch --array=0-14 slurm_run_array.sh
#   ./scripts/submit_suite.sh main HalfCheetah-v5 Hopper-v5

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH --time=0-12:00:00
#SBATCH --job-name=critic-uphill
#SBATCH -p lem-gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

set -euo pipefail

ENV="${TARGET_ENV:?set TARGET_ENV, e.g. TARGET_ENV=HalfCheetah-v5}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?run sbatch from the checkout or set PROJECT_DIR}}"
TOTAL_STEPS="${TOTAL_STEPS:-1_000_000}"
TAG="${TAG:-critic-uphill-main-v1}"
SUITE="${SUITE:-main}"

MAIN=(
  "td3|algorithm=td3"
  "evorainbow-td3-full|algorithm=evorainbow algorithm.eval.mode=full"
  "evorainbow-td3-critic|algorithm=evorainbow algorithm.eval.mode=critic"
  "evorainbow-td3-hboot|algorithm=evorainbow algorithm.eval.mode=hboot"
)

RESET=(
  "td3|algorithm=td3"
  "td3-reset|algorithm=td3-reset"
)

ER_TD3="algorithm=evorainbow algorithm.shadow.enabled=true"
MEASURE=(
  "td3|algorithm=td3 algorithm.shadow.enabled=true"
  "evorainbow-td3-full|${ER_TD3} algorithm.eval.mode=full"
  "evorainbow-td3-critic|${ER_TD3} algorithm.eval.mode=critic"
)

PREFIX="${ER_TD3} algorithm.eval.mode=prefix"
SURROGATE=(
  "evorainbow-td3-critic-p0|${ER_TD3} algorithm.eval.mode=critic algorithm.eval.p_real=0"
  "evorainbow-td3-prefix-h100|${PREFIX} algorithm.eval.h=100"
  "evorainbow-td3-prefix-h200|${PREFIX} algorithm.eval.h=200"
  "evorainbow-td3-prefix-h400|${PREFIX} algorithm.eval.h=400"
  "evorainbow-td3-prefix-h200-matched|${PREFIX} algorithm.eval.h=200"
)

case "${SUITE}" in
main) CONDITIONS=("${MAIN[@]}") ;;
reset) CONDITIONS=("${RESET[@]}") ;;
measure) CONDITIONS=("${MEASURE[@]}") ;;
surrogate) CONDITIONS=("${SURROGATE[@]}") ;;
*)
  echo "ERROR: Unknown SUITE '${SUITE}'. Choose 'main', 'reset', 'measure' or 'surrogate'." >&2
  exit 1
  ;;
esac

# SEEDS="10 11 12 13 14" runs other seeds: task ID = condition_index *
# number of seeds + position of the seed in SEEDS
read -r -a SEEDS <<<"${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
N_SEEDS=${#SEEDS[@]}

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
COND_IDX=$((TASK_ID / N_SEEDS))
((COND_IDX < ${#CONDITIONS[@]})) || {
  echo "ERROR: task ${TASK_ID} out of range for SUITE=${SUITE} (${#CONDITIONS[@]} conditions x ${N_SEEDS} seeds)" >&2
  exit 1
}
CONDITION="${CONDITIONS[${COND_IDX}]}"
SEED="${SEEDS[$((TASK_ID % N_SEEDS))]}"
NAME="${CONDITION%%|*}"
read -r -a OVERRIDES <<<"${CONDITION#*|} ${EXTRA:-}"
ENV_SLUG="${ENV//\//_}"
RUN_NAME="${NAME}_${ENV_SLUG}_s${SEED}"
if [[ "${NAME}" == *-matched ]]; then
  OVERRIDES+=(
    "algorithm.matched.generations=${MATCH_GENERATIONS:?set MATCH_GENERATIONS (median measure ERL generations in ${ENV})}"
    "algorithm.matched.rl_updates=${MATCH_UPDATES:?set MATCH_UPDATES (median measure ERL RL updates in ${ENV})}"
  )
fi

echo "Task ${TASK_ID} | ${NAME} | ${ENV} | seed ${SEED}"

cd "${PROJECT_DIR}"

VENV="${PROJECT_DIR}/.venv/bin"
[[ -x "${VENV}/python" ]] || {
  echo "no venv at ${VENV} - run 'uv sync' in ${PROJECT_DIR} on the login node" >&2
  exit 1
}

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY="$("${VENV}/python" -c 'import wandb; print(wandb.api.api_key or "")' 2>/dev/null || true)"
  export WANDB_API_KEY
fi
: "${WANDB_API_KEY:?export WANDB_API_KEY before submitting or run wandb login}"
export WANDB_MODE="${WANDB_MODE:-online}"
PD_ROOT="/lustre/pd03/${SLURM_JOB_ACCOUNT:-hpc-jakgil6519-1780267238}"
if [[ -d "${PD_ROOT}" && -w "${PD_ROOT}" ]]; then
  DATA_ROOT="${PD_ROOT}"
else
  DATA_ROOT="${PROJECT_DIR}"
fi
export WANDB_DIR="${DATA_ROOT}/wandb_logs/${RUN_NAME}"
mkdir -p logs "${WANDB_DIR}"
if [[ "${SUITE}" == measure ]]; then
  OVERRIDES+=("checkpoint.dir=${DATA_ROOT}/checkpoints/${TAG}/${RUN_NAME}")
fi

if "${VENV}/python" -m train \
  "${OVERRIDES[@]}" \
  seed="${SEED}" \
  env.id="${ENV}" \
  eval_env.id="${ENV}" \
  total_steps="${TOTAL_STEPS}" \
  wandb.enabled=true \
  "wandb.name=${RUN_NAME}" \
  "wandb.tags=[${TAG},${NAME},${ENV_SLUG}]" \
  hydra.run.dir="outputs/${RUN_NAME}"; then
  if [[ "${WANDB_MODE}" == offline ]]; then
    for d in "${WANDB_DIR}"/wandb/offline-run-*; do
      [[ -d "$d" ]] && "${VENV}/wandb" sync "$d"
    done
  fi
else
  echo "ERROR: ${RUN_NAME} failed" >&2
  exit 1
fi
