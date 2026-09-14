#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${QWEN35_SUPERVISOR_LOG:-${HERE}/qwen35_supervisor.log}"
QUICK_ALLOCATION_RETRIES="${QWEN35_QUICK_ALLOCATION_RETRIES:-5}"
QUICK_ALLOCATION_DELAY="${QWEN35_QUICK_ALLOCATION_DELAY:-10}"
SLOW_ALLOCATION_DELAY="${QWEN35_SLOW_ALLOCATION_DELAY:-3600}"
allocation_failures=0

while true; do
  printf '[%s] Starting or resuming Qwen validation.\n' \
    "$(date --iso-8601=seconds)" | tee -a "${LOG}"
  "${HERE}/run_colab.sh"
  status=$?
  if [[ "${status}" -eq 0 ]]; then
    printf '[%s] Qwen validation completed.\n' \
      "$(date --iso-8601=seconds)" | tee -a "${LOG}"
    exit 0
  fi
  if [[ "${status}" -eq 76 ]]; then
    allocation_failures=$((allocation_failures + 1))
    if (( allocation_failures <= QUICK_ALLOCATION_RETRIES )); then
      delay="${QUICK_ALLOCATION_DELAY}"
    else
      delay="${SLOW_ALLOCATION_DELAY}"
    fi
    printf '[%s] T4 allocation failed on the current account (%d); retrying in %d seconds.\n' \
      "$(date --iso-8601=seconds)" \
      "${allocation_failures}" \
      "${delay}" | tee -a "${LOG}"
  elif [[ "${status}" -eq 77 ]]; then
    allocation_failures=0
    delay=10
  elif [[ "${status}" -eq 75 ]]; then
    allocation_failures=0
    delay=60
  else
    allocation_failures=0
    delay=300
  fi
  printf '[%s] Launcher exited %d; retrying in %d seconds.\n' \
    "$(date --iso-8601=seconds)" "${status}" "${delay}" | tee -a "${LOG}"
  sleep "${delay}"
done
