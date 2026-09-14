#!/usr/bin/env bash
set -euo pipefail

PID_FILE="/tmp/lq_storybook_gemma.pid"
if [[ ! -f "${PID_FILE}" ]]; then
  exit 0
fi

pid="$(cat "${PID_FILE}")"
if kill -0 "${pid}" 2>/dev/null; then
  kill "${pid}"
  for _ in {1..30}; do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 1
  done
fi
rm -f "${PID_FILE}"
