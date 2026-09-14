#!/usr/bin/env bash
set -euo pipefail

SERVER="/home/derek/.local/share/bonsai-q1-crack/llama.cpp/build/bin/llama-server"
MODEL_DIR="/mnt/storage/projects/agentic/models/qwen35-4b-q6kl"
MODEL="${MODEL_DIR}/Qwen_Qwen3.5-4B-Q6_K_L.gguf"
MMPROJ="${MODEL_DIR}/mmproj-Qwen_Qwen3.5-4B-bf16.gguf"
PID_FILE="/tmp/lq_storybook_qwen35.pid"
LOG_FILE="/tmp/lq_storybook_qwen35_server.log"
HOST="127.0.0.1"
PORT="8082"

[[ -x "${SERVER}" ]] || { printf 'Missing llama-server: %s\n' "${SERVER}" >&2; exit 1; }
[[ -f "${MODEL}" ]] || { printf 'Missing Qwen model: %s\n' "${MODEL}" >&2; exit 1; }
[[ -f "${MMPROJ}" ]] || { printf 'Missing Qwen projector: %s\n' "${MMPROJ}" >&2; exit 1; }

if curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  printf 'Storybook Qwen is already running on port %s.\n' "${PORT}"
  exit 0
fi

setsid "${SERVER}" \
  --model "${MODEL}" \
  --mmproj "${MMPROJ}" \
  --no-mmproj-offload \
  --alias qwen35-4b-q6kl-storybook \
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
  --batch-size 1024 \
  --ubatch-size 512 \
  --image-min-tokens 1024 \
  --image-max-tokens 2048 \
  --reasoning on \
  --reasoning-budget 192 \
  --reasoning-format deepseek \
  --temp 0 \
  --top-p 0.9 \
  --mmap \
  --jinja \
  --cont-batching \
  >"${LOG_FILE}" 2>&1 &

pid=$!
disown "${pid}"
printf '%s\n' "${pid}" >"${PID_FILE}"

for _ in {1..300}; do
  if curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    printf 'Storybook Qwen is ready on port %s (PID %s).\n' "${PORT}" "${pid}"
    exit 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    printf 'Storybook Qwen exited; see %s\n' "${LOG_FILE}" >&2
    exit 1
  fi
  sleep 1
done

printf 'Storybook Qwen startup timed out; see %s\n' "${LOG_FILE}" >&2
exit 1
