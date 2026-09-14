#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUN="/mnt/storage/projects/agentic/images/sdxl_littlequeen_v1/outputs/storybook_mvp_v1/the_witches_trick_20260728"
RUN_ROOT="${1:-${DEFAULT_RUN}}"
LOG_FILE="${RUN_ROOT}/qwen35_local_storybook.log"
PID_FILE="/tmp/lq_qwen35_storybook_run.pid"

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  cmdline=""
  if [[ -r "/proc/${pid}/cmdline" ]]; then
    cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
  fi
  if kill -0 "${pid}" 2>/dev/null && [[ "${cmdline}" == *"render_storybook_qwen35.py"* ]]; then
    printf 'Qwen storybook picker is already running (PID %s).\n' "${pid}"
    printf 'Log: %s\n' "${LOG_FILE}"
    exit 0
  fi
fi

printf '\n[%s] Launching Qwen3.5-4B Q6_K_L storybook picker\n' \
  "$(date --iso-8601=seconds)" >>"${LOG_FILE}"
setsid "${WORKFLOW_DIR}/run_qwen35_storybook.sh" "${RUN_ROOT}" \
  >>"${LOG_FILE}" 2>&1 &
pid=$!
disown "${pid}"
printf '%s\n' "${pid}" >"${PID_FILE}"

printf 'Started Qwen storybook picker (PID %s).\n' "${pid}"
printf 'Log: %s\n' "${LOG_FILE}"
