#!/usr/bin/env bash
set -euo pipefail

suite="${1:?usage: submit_suite.sh <main> <env>...}"
shift
(($# > 0)) || {
  echo "usage: submit_suite.sh ${suite} <env>..." >&2
  exit 1
}

case "${suite}" in
main) array=0-79 ;;
*)
  echo "unknown suite '${suite}'; choose 'main'" >&2
  exit 1
  ;;
esac

tag="${TAG:-critic-uphill-${suite}-v1}"
for env in "$@"; do
  echo "--> ${env}: SUITE=${suite} array=${array} TAG=${tag}"
  SUITE="${suite}" TARGET_ENV="${env}" TAG="${tag}" \
    sbatch --parsable --array="${array}" slurm_run_array.sh
done
