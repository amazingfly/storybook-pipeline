#!/usr/bin/env bash
set -euo pipefail

PID_FILE="/tmp/lq_storybook_qwen35.pid"
[[ -f "${PID_FILE}" ]] || exit 0

pid="$(cat "${PID_FILE}")"
cmdline=""
if [[ -r "/proc/${pid}/cmdline" ]]; then
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
fi
if [[ "${cmdline}" == *"llama-server"* && "${cmdline}" == *"Qwen_Qwen3.5-4B-Q6_K_L.gguf"* ]]; then
  kill "${pid}" 2>/dev/null || true
  for _ in {1..30}; do
    kill -0 "${pid}" 2>/dev/null || break
    sleep 1
  done
fi
rm -f "${PID_FILE}"
