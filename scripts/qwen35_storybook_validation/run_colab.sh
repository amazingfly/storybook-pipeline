#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
SESSION="${QWEN35_COLAB_SESSION:-lq-qwen35-validation-v3}"
GPU="${QWEN35_COLAB_GPU:-T4}"
COLAB="${COLAB_BIN:-${HOME}/.local/bin/colab}"
ALLOCATION_TIMEOUT="${QWEN35_COLAB_ALLOCATION_TIMEOUT:-900}"
QUICK_ALLOCATION_RETRIES="${QWEN35_QUICK_ALLOCATION_RETRIES:-5}"
QUICK_ALLOCATION_DELAY="${QWEN35_QUICK_ALLOCATION_DELAY:-10}"
SLOW_ALLOCATION_DELAY="${QWEN35_SLOW_ALLOCATION_DELAY:-3600}"
ALLOCATED=0
SUCCESS=0
SESSION_BUSY=0
FRONTEND_PID=""
TUNNEL_PID=""
EXEC_PID=""
LOG_TAIL_PID=""
LOG="${QWEN35_COLAB_LOG:-${HERE}/qwen35_colab.log}"
RCLONE_CONFIG="${RCLONE_CONFIG:-${HOME}/.config/rclone/rclone.conf}"
FRONTEND_READY="${HERE}/qwen35_frontend_keepalive.ready.json"
FRONTEND_LOG="${HERE}/qwen35_frontend_keepalive.log"
TUNNEL_LOG="${HERE}/qwen35_tunnel_keepalive.log"
FRONTEND_START_TIMEOUT="${QWEN35_FRONTEND_KEEPALIVE_START_TIMEOUT:-60}"
TUNNEL_INTERVAL="${QWEN35_TUNNEL_KEEPALIVE_INTERVAL:-45}"
TUNNEL_TIMEOUT="${QWEN35_TUNNEL_KEEPALIVE_TIMEOUT:-10}"
HEALTH_INTERVAL="${QWEN35_SESSION_HEALTH_INTERVAL:-30}"
HEALTH_FAILURE_LIMIT="${QWEN35_SESSION_HEALTH_FAILURE_LIMIT:-3}"

resolve_colab_python() {
  local shebang
  shebang="$(head -n 1 "${COLAB}" 2>/dev/null || true)"
  if [[ "${shebang}" == "#!"* ]] && [[ "${shebang}" != *"/usr/bin/env "* ]]; then
    printf '%s\n' "${shebang#\#!}"
  else
    printf '%s\n' "python3"
  fi
}

resolve_frontend_python() {
  local candidate
  for candidate in \
      "${QWEN35_FRONTEND_PYTHON:-}" \
      "${HOME}/miniforge3/bin/python3" \
      "${HOME}/miniconda3/bin/python3" \
      "$(command -v python3 2>/dev/null || true)"; do
    [[ -x "${candidate}" ]] || continue
    if "${candidate}" -c 'import playwright.async_api' >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

COLAB_PYTHON="${COLAB_PYTHON:-$(resolve_colab_python)}"
COLAB_CMD=("${COLAB_PYTHON}" "${HERE}/colab_ipv4.py")
FRONTEND_PYTHON="$(resolve_frontend_python || true)"

stop_process() {
  local pid="$1" process_state
  [[ -n "${pid}" ]] || return 0
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do
      process_state="$(ps -o stat= -p "${pid}" 2>/dev/null || true)"
      [[ -n "${process_state}" && "${process_state}" != Z* ]] || break
      sleep 0.5
    done
    process_state="$(ps -o stat= -p "${pid}" 2>/dev/null || true)"
    if [[ -n "${process_state}" && "${process_state}" != Z* ]]; then
      kill -KILL "${pid}" >/dev/null 2>&1 || true
    fi
  fi
  wait "${pid}" >/dev/null 2>&1 || true
}

stop_local_execution() {
  stop_process "${EXEC_PID}"
  EXEC_PID=""
  stop_process "${LOG_TAIL_PID}"
  LOG_TAIL_PID=""
}

cleanup() {
  stop_local_execution
  if [[ -n "${TUNNEL_PID}" ]]; then
    stop_process "${TUNNEL_PID}"
    TUNNEL_PID=""
  fi
  if [[ -n "${FRONTEND_PID}" ]]; then
    stop_process "${FRONTEND_PID}"
    FRONTEND_PID=""
  fi
  if [[ "${ALLOCATED}" -eq 1 && "${SUCCESS}" -eq 1 ]]; then
    timeout 60s "${COLAB_CMD[@]}" stop -s "${SESSION}" >/dev/null 2>&1 || true
  elif [[ "${ALLOCATED}" -eq 1 ]]; then
    printf 'Preserving Colab session %s after interruption for diagnosis/resume.\n' \
      "${SESSION}" >&2
  fi
}
trap cleanup EXIT

session_is_registered() {
  timeout 20s "${COLAB_CMD[@]}" sessions </dev/null 2>&1 \
    | grep -Fq "[${SESSION}]"
}

open_frontend_url() {
  local frontend_url="$1"
  python3 - "${frontend_url}" "${QWEN35_COLAB_AUTHUSER:-0}" <<'PY'
import sys
import urllib.parse
import webbrowser

url, authuser = sys.argv[1:]
parts = urllib.parse.urlsplit(url)
query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
query = [(key, value) for key, value in query if key != "authuser"]
query.append(("authuser", authuser))
webbrowser.open(
    urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urllib.parse.urlencode(query), parts.fragment)
    )
)
PY
}

start_frontend_keepalive() {
  [[ "${QWEN35_COLAB_FRONTEND_KEEPALIVE:-1}" == "1" ]] || return 0
  local frontend_url started_at
  if ! frontend_url="$(
      timeout 30s "${COLAB_CMD[@]}" url -s "${SESSION}" </dev/null
    )"; then
    printf 'Unable to resolve the Colab frontend URL; tunnel keep-alive remains active.\n' \
      >&2
    return 0
  fi
  if [[ -z "${FRONTEND_PYTHON}" ]]; then
    printf 'No Python interpreter with Playwright is available; opening the runtime in the browser.\n' \
      >&2
    open_frontend_url "${frontend_url}"
    return 0
  fi
  rm -f "${FRONTEND_READY}" "${FRONTEND_LOG}"
  "${FRONTEND_PYTHON}" "${ROOT}/scripts/colab_frontend_keepalive.py" \
    --url "${frontend_url}" \
    --authuser "${QWEN35_COLAB_AUTHUSER:-0}" \
    --ready-file "${FRONTEND_READY}" \
    >"${FRONTEND_LOG}" 2>&1 &
  FRONTEND_PID=$!
  started_at="$(date +%s)"
  while [[ ! -s "${FRONTEND_READY}" ]]; do
    if ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
      wait "${FRONTEND_PID}" >/dev/null 2>&1 || true
      FRONTEND_PID=""
      printf 'Authenticated frontend keep-alive failed; opening the runtime in the browser. See %s\n' \
        "${FRONTEND_LOG}" >&2
      open_frontend_url "${frontend_url}"
      return 0
    fi
    if (( $(date +%s) - started_at > FRONTEND_START_TIMEOUT )); then
      kill "${FRONTEND_PID}" >/dev/null 2>&1 || true
      wait "${FRONTEND_PID}" >/dev/null 2>&1 || true
      FRONTEND_PID=""
      printf 'Authenticated frontend keep-alive timed out; opening the runtime in the browser. See %s\n' \
        "${FRONTEND_LOG}" >&2
      open_frontend_url "${frontend_url}"
      return 0
    fi
    sleep 2
  done
  printf 'Authenticated Colab frontend keep-alive is active.\n'
}

start_tunnel_keepalive() {
  [[ "${QWEN35_COLAB_TUNNEL_KEEPALIVE:-1}" == "1" ]] || return 0
  rm -f "${TUNNEL_LOG}"
  "${COLAB_PYTHON}" "${ROOT}/scripts/colab_tunnel_keepalive.py" \
    --session "${SESSION}" \
    --authuser "${QWEN35_COLAB_AUTHUSER:-0}" \
    --auth-provider oauth2 \
    --ipv4 \
    --request-timeout "${TUNNEL_TIMEOUT}" \
    --interval "${TUNNEL_INTERVAL}" \
    --loop \
    >"${TUNNEL_LOG}" 2>&1 &
  TUNNEL_PID=$!
  sleep 2
  if ! kill -0 "${TUNNEL_PID}" 2>/dev/null; then
    wait "${TUNNEL_PID}" >/dev/null 2>&1 || true
    TUNNEL_PID=""
    printf 'Colab tunnel keep-alive failed to start. See %s\n' \
      "${TUNNEL_LOG}" >&2
    return 0
  fi
  printf 'Colab tunnel keep-alive is active.\n'
}

if session_is_registered; then
  if timeout 30s "${COLAB_CMD[@]}" ls -s "${SESSION}" /content \
      </dev/null >/dev/null 2>&1; then
    printf 'Reusing Colab session %s.\n' "${SESSION}"
    ALLOCATED=1
    if timeout 30s "${COLAB_CMD[@]}" status -s "${SESSION}" </dev/null 2>&1 \
        | grep -Fq "Status: BUSY"; then
      SESSION_BUSY=1
    fi
  else
    printf 'Discarding stale Colab session %s.\n' "${SESSION}" >&2
    timeout 60s "${COLAB_CMD[@]}" stop -s "${SESSION}" </dev/null >/dev/null 2>&1 \
      || true
  fi
fi
if [[ "${ALLOCATED}" -eq 0 ]]; then
  allocation_failures=0
  while [[ "${ALLOCATED}" -eq 0 ]]; do
    if timeout --foreground "${ALLOCATION_TIMEOUT}s" \
        "${COLAB_CMD[@]}" new -s "${SESSION}" --gpu "${GPU}"; then
      ALLOCATED=1
      break
    fi

    status=$?
    if session_is_registered \
        && timeout 30s "${COLAB_CMD[@]}" ls -s "${SESSION}" /content \
          </dev/null >/dev/null 2>&1; then
      printf 'Colab session %s became ready after the allocation command exited.\n' \
        "${SESSION}"
      ALLOCATED=1
      break
    fi

    allocation_failures=$((allocation_failures + 1))
    printf 'Colab %s allocation failed or exceeded %ss (status %d, failure %d).\n' \
      "${GPU}" \
      "${ALLOCATION_TIMEOUT}" \
      "${status}" \
      "${allocation_failures}" >&2
    if (( allocation_failures <= QUICK_ALLOCATION_RETRIES )); then
      delay="${QUICK_ALLOCATION_DELAY}"
      printf 'Retrying the same Colab account in %ss (quick retry %d/%d).\n' \
        "${delay}" \
        "${allocation_failures}" \
        "${QUICK_ALLOCATION_RETRIES}" >&2
    else
      delay="${SLOW_ALLOCATION_DELAY}"
      printf 'Quick retries exhausted; retrying the same Colab account in %ss.\n' \
        "${delay}" >&2
    fi
    sleep "${delay}"
  done
fi

start_tunnel_keepalive
start_frontend_keepalive

if [[ "${SESSION_BUSY}" -eq 1 ]]; then
  printf 'Colab session %s already has an active cell; guarding it until it exits.\n' \
    "${SESSION}" >&2
  while timeout 30s "${COLAB_CMD[@]}" status -s "${SESSION}" </dev/null 2>&1 \
      | grep -Fq "Status: BUSY"; do
    sleep 60
  done
  exit 75
fi

if [[ ! -f "${RCLONE_CONFIG}" ]]; then
  printf 'Local rclone config is unavailable.\n' >&2
  exit 1
fi
"${COLAB_CMD[@]}" upload -s "${SESSION}" "${RCLONE_CONFIG}" /content/rclone.conf
"${COLAB_CMD[@]}" upload -s "${SESSION}" \
  "${HERE}/remote_run.py" \
  /content/qwen35_storybook_validation.py
set +e
tail -n 0 -F "${LOG}" &
LOG_TAIL_PID=$!
"${COLAB_CMD[@]}" exec -s "${SESSION}" \
  -f "${HERE}/launch.py" \
  --timeout "${QWEN35_COLAB_TIMEOUT:-43200}" \
  >>"${LOG}" 2>&1 &
EXEC_PID=$!
set -e

health_failures=0
last_health_check=0
while kill -0 "${EXEC_PID}" >/dev/null 2>&1; do
  sleep 5
  now_epoch="$(date +%s)"
  if (( now_epoch - last_health_check < HEALTH_INTERVAL )); then
    continue
  fi
  last_health_check="${now_epoch}"
  if session_is_registered; then
    if (( health_failures > 0 )); then
      printf '[%s] Colab session health restored after %d failed probe(s).\n' \
        "$(date --iso-8601=seconds)" "${health_failures}"
    fi
    health_failures=0
    continue
  fi
  health_failures=$((health_failures + 1))
  printf '[%s] Colab session health probe failed %d/%d.\n' \
    "$(date --iso-8601=seconds)" \
    "${health_failures}" \
    "${HEALTH_FAILURE_LIMIT}" >&2
  if (( health_failures >= HEALTH_FAILURE_LIMIT )); then
    printf 'Colab runtime %s disappeared; terminating the stale local execution stream.\n' \
      "${SESSION}" >&2
    stop_local_execution
    exit 77
  fi
done

set +e
wait "${EXEC_PID}"
EXEC_STATUS=$?
set -e
EXEC_PID=""
stop_process "${LOG_TAIL_PID}"
LOG_TAIL_PID=""
if [[ "${EXEC_STATUS}" -ne 0 ]]; then
  exit "${EXEC_STATUS}"
fi
if ! tail -n 200 "${LOG}" | grep -Fq "[validation complete]"; then
  printf 'Colab cell exited without the validator completion marker.\n' >&2
  exit 1
fi

SUCCESS=1
printf 'Qwen3.5 storybook validation completed; releasing %s.\n' "${SESSION}"
