#!/bin/bash -l
# critic-uphill benchmark array on WCSS Lem (MuJoCo v5).
# Envs: HalfCheetah-v5, Hopper-v5, Walker2d-v5, Ant-v5, Swimmer-v5
#
# Suite: main (task ID = condition_index * 10 + seed, array 0-39)
#   0-9   td3
#   10-19 evorainbow-full
#   20-29 evorainbow-critic
#   30-39 evorainbow-hboot
#
# Example usage:
#   TARGET_ENV=Hopper-v5 TAG=main-v1 sbatch --array=0-39 slurm_run_array.sh
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
  "evorainbow-full|algorithm=evorainbow algorithm.eval.mode=full"
  "evorainbow-critic|algorithm=evorainbow algorithm.eval.mode=critic"
  "evorainbow-hboot|algorithm=evorainbow algorithm.eval.mode=hboot"
)

case "${SUITE}" in
main) CONDITIONS=("${MAIN[@]}") ;;
*)
  echo "ERROR: Unknown SUITE '${SUITE}'. Only 'main' is supported." >&2
  exit 1
  ;;
esac

SEEDS=(0 1 2 3 4 5 6 7 8 9)
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
  export WANDB_DIR="${PD_ROOT}/wandb_logs/${RUN_NAME}"
else
  export WANDB_DIR="${PROJECT_DIR}/wandb_logs/${RUN_NAME}"
fi
mkdir -p logs "${WANDB_DIR}"

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
