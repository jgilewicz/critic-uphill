#!/usr/bin/env bash
set -euo pipefail

suite="${1:?usage: submit_suite.sh <main|reset|measure|surrogate> <env>...}"
shift
(($# > 0)) || {
  echo "usage: submit_suite.sh ${suite} <env>..." >&2
  exit 1
}

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
*)
  echo "unknown suite '${suite}'; choose 'main', 'reset', 'measure' or 'surrogate'" >&2
  exit 1
  ;;
esac

tag="${TAG:-critic-uphill-${suite}-v1}"
for env in "$@"; do
  echo "--> ${env}: SUITE=${suite} array=${array} TAG=${tag}"
  SUITE="${suite}" TARGET_ENV="${env}" TAG="${tag}" \
    sbatch --parsable --array="${array}" slurm_run_array.sh
done
