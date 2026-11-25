#!/usr/bin/env bash
set -euo pipefail

suite="${1:?usage: submit_suite.sh <main|reset|e1|e2> <env>...}"
shift
(($# > 0)) || {
  echo "usage: submit_suite.sh ${suite} <env>..." >&2
  exit 1
}

case "${suite}" in
main) array=0-79 ;;
reset) array=0-19 ;;
e1) array=0-29 ;;
e2)
  array=0-49
  # matched condition: MATCH_* are per-env medians of the e1 ERL runs
  : "${MATCH_GENERATIONS:?set MATCH_GENERATIONS for e2}"
  : "${MATCH_UPDATES:?set MATCH_UPDATES for e2}"
  (($# == 1)) || {
    echo "e2 takes one env per call (MATCH_* differ per env)" >&2
    exit 1
  }
  ;;
*)
  echo "unknown suite '${suite}'; choose 'main', 'reset', 'e1' or 'e2'" >&2
  exit 1
  ;;
esac

tag="${TAG:-critic-uphill-${suite}-v1}"
for env in "$@"; do
  echo "--> ${env}: SUITE=${suite} array=${array} TAG=${tag}"
  SUITE="${suite}" TARGET_ENV="${env}" TAG="${tag}" \
    sbatch --parsable --array="${array}" slurm_run_array.sh
done
