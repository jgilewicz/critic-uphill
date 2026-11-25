#!/usr/bin/env bash
set -euo pipefail

suite="${1:?usage: submit_suite.sh <main|reset|measure|surrogate|xqc> <env>...}"
shift
(($# > 0)) || {
  echo "usage: submit_suite.sh ${suite} <env>..." >&2
  exit 1
}

sbatch_args=()
case "${suite}" in
main) array=0-39 ;;
reset) array=0-19 ;;
measure) array=0-29 ;;
surrogate)
  array=0-49
  # matched condition: MATCH_* are per-env medians of the measure ERL runs
  : "${MATCH_GENERATIONS:?set MATCH_GENERATIONS for surrogate}"
  : "${MATCH_UPDATES:?set MATCH_UPDATES for surrogate}"
  (($# == 1)) || {
    echo "surrogate takes one env per call (MATCH_* differ per env)" >&2
    exit 1
  }
  ;;
xqc)
  read -r -a seeds <<<"${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
  array="0-$((${#seeds[@]} - 1))"
  # 1M-step replay buffers (Humanoid-v5 about 28 GB) exceed the 16 GB default
  sbatch_args=(--mem=48gb)
  ;;
*)
  echo "unknown suite '${suite}'; choose 'main', 'reset', 'measure', 'surrogate' or 'xqc'" >&2
  exit 1
  ;;
esac

tag="${TAG:-critic-uphill-${suite}-v1}"
for env in "$@"; do
  echo "--> ${env}: SUITE=${suite} array=${array} TAG=${tag}"
  SUITE="${suite}" TARGET_ENV="${env}" TAG="${tag}" \
    sbatch --parsable ${sbatch_args[@]+"${sbatch_args[@]}"} --array="${array}" slurm_run_array.sh
done
