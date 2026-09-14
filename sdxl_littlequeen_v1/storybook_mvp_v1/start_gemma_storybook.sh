#!/usr/bin/env bash
set -euo pipefail

SERVER="/home/derek/.local/share/bonsai-q1-crack/llama.cpp/build/bin/llama-server"
MODEL="/home/derek/projects/hug/gemma4_uncensored/models/gemma4/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf"
MMPROJ="/home/derek/projects/hug/gemma4_uncensored/models/gemma4/mmproj-google_gemma-4-E4B-it-f16.gguf"
PID_FILE="/tmp/lq_storybook_gemma.pid"
LOG_FILE="/tmp/lq_storybook_gemma.log"
HOST="127.0.0.1"
PORT="8081"

[[ -x "${SERVER}" ]] || { printf 'Missing llama-server: %s\n' "${SERVER}" >&2; exit 1; }
[[ -f "${MODEL}" ]] || { printf 'Missing Gemma model: %s\n' "${MODEL}" >&2; exit 1; }
[[ -f "${MMPROJ}" ]] || { printf 'Missing Gemma projector: %s\n' "${MMPROJ}" >&2; exit 1; }

if curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  printf 'Storybook Gemma is already running on port %s.\n' "${PORT}"
  exit 0
fi

setsid "${SERVER}" \
  --model "${MODEL}" \
  --mmproj "${MMPROJ}" \
  --alias gemma4-storybook \
  --host "${HOST}" \
  --port "${PORT}" \
  --ctx-size 8192 \
  --n-predict 2048 \
  --threads 8 \
  --threads-batch 12 \
  --parallel 1 \
  --n-gpu-layers 0 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-ram 512 \
  --batch-size 2048 \
  --ubatch-size 2048 \
  --temp 0 \
  --top-p 0.9 \
  --mmap \
  --jinja \
  --cont-batching \
  >"${LOG_FILE}" 2>&1 &

pid=$!
disown "${pid}"
printf '%s\n' "${pid}" >"${PID_FILE}"

for _ in {1..180}; do
  if curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    printf 'Storybook Gemma is ready on port %s.\n' "${PORT}"
    exit 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    printf 'Storybook Gemma exited; see %s\n' "${LOG_FILE}" >&2
    exit 1
  fi
  sleep 1
done

printf 'Storybook Gemma startup timed out; see %s\n' "${LOG_FILE}" >&2
exit 1
