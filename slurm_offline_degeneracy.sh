#!/bin/bash -l
# critic-degeneracy-v1 offline measures for one env: calibrate sigma on
# seed 0's first checkpoint, then evaluate every checkpoint of both seeds.
# Reads checkpoints from and writes JSONs to $PD_ROOT/critic-degeneracy-v1.
#   TARGET_ENV=Swimmer-v5 sbatch --dependency=afterok:<train job> slurm_offline_degeneracy.sh

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH --time=0-03:00:00
#SBATCH --job-name=crit-degen
#SBATCH -p lem-gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

ENV="${TARGET_ENV:?set TARGET_ENV, e.g. TARGET_ENV=Swimmer-v5}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?run sbatch from the checkout or set PROJECT_DIR}}"
PD_ROOT="/lustre/pd03/${SLURM_JOB_ACCOUNT:-hpc-jakgil6519-1780267238}"
ROOT="${PD_ROOT}/critic-degeneracy-v1"
TOML="${PROJECT_DIR}/experiments/critic-degeneracy-v1/experiment.toml"
PY="${PROJECT_DIR}/.venv/bin/python"
[[ -x "${PY}" ]] || {
  echo "no venv at ${PY} - run 'uv sync' in ${PROJECT_DIR} on the login node" >&2
  exit 1
}

cd "${PROJECT_DIR}/src"
for cmd in calibrate evaluate; do
  "${PY}" -m offline.critic_degeneracy "${cmd}" "${TOML}" \
    "${ROOT}/ckpt" "${ROOT}/out" "${ENV}"
done
