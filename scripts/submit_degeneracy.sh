#!/usr/bin/env bash
# critic-degeneracy-v1: per env, td3 seeds 0-1 with checkpoints, then the
# offline measures once both seeds finish. Prints "<env> <train> <offline>".
set -euo pipefail

(($# > 0)) || {
  echo "usage: submit_degeneracy.sh <env>..." >&2
  exit 1
}
pd_root="/lustre/pd03/${ACCOUNT:-hpc-jakgil6519-1780267238}"
extra="checkpoint.dir=${pd_root}/critic-degeneracy-v1/ckpt checkpoint.at=[0.3,0.6,1.0]"

for env in "$@"; do
  train=$(SUITE=controls TARGET_ENV="${env}" TAG=critic-degeneracy-v1 \
    EXTRA="${extra}" sbatch --parsable --array=0-1 slurm_run_array.sh)
  offline=$(TARGET_ENV="${env}" sbatch --parsable \
    --dependency="afterok:${train}" slurm_offline_degeneracy.sh)
  echo "${env} ${train} ${offline}"
done
