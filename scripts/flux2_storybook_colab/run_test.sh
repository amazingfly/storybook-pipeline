#!/usr/bin/env bash
set -Eeuo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG="$HERE/config.json"
MANIFEST="$HERE/test_manifest.json"
COLAB_PY="$HERE/../qwen35_storybook_validation/colab_ipv4.py"
COLAB_PYTHON=/home/derek/.local/share/uv/tools/google-colab-cli/bin/python3
COLAB=("$COLAB_PYTHON" "$COLAB_PY")
SESSION=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["session"])' "$CONFIG")
LOCAL_ROOT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["local_output_root"])' "$CONFIG")
STOP_AFTER_RECOVERY=$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1])).get("stop_session_after_recovery", True)).lower())' "$CONFIG")
TOKEN=${HF_TOKEN_FILE:-/home/derek/projects/hug/token}
RUN_ID=$(date -u +%Y%m%d_%H%M%S)
RUN_DIR="$LOCAL_ROOT/$RUN_ID"
LOG="$RUN_DIR/colab.log"
KEEPALIVE_PID=""

mkdir -p "$RUN_DIR"

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"
}

cleanup() {
    if [[ -n "$KEEPALIVE_PID" ]]; then
        kill "$KEEPALIVE_PID" 2>/dev/null || true
        wait "$KEEPALIVE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

[[ -s "$TOKEN" ]] || { log "ERROR Hugging Face token not found at $TOKEN"; exit 1; }
python3 "$HERE/test_package.py" 2>&1 | tee -a "$LOG"

STATUS_OUTPUT=$("${COLAB[@]}" status -s "$SESSION" 2>&1 || true)
printf '%s\n' "$STATUS_OUTPUT" >>"$LOG"
if grep -q "not found" <<<"$STATUS_OUTPUT"; then
    log "Requesting Colab T4 session $SESSION"
    "${COLAB[@]}" new -s "$SESSION" --gpu T4 2>&1 | tee -a "$LOG"
else
    log "Reusing active Colab session $SESSION"
fi

log "Starting tunnel keepalive"
"$COLAB_PYTHON" "$HERE/../colab_tunnel_keepalive.py" \
    --session "$SESSION" --loop --interval 45 --ipv4 >>"$LOG" 2>&1 &
KEEPALIVE_PID=$!

log "Uploading test inputs"
"${COLAB[@]}" upload "$CONFIG" /content/flux2_storybook_config.json -s "$SESSION" 2>&1 | tee -a "$LOG"
"${COLAB[@]}" upload "$MANIFEST" /content/flux2_storybook_manifest.json -s "$SESSION" 2>&1 | tee -a "$LOG"
"${COLAB[@]}" upload "$TOKEN" /content/hf_token -s "$SESSION" 2>&1 | tee -a "$LOG"

log "Installing inference dependencies"
"${COLAB[@]}" exec -s "$SESSION" -f "$HERE/bootstrap.py" --timeout 1200 2>&1 | tee -a "$LOG"

log "Starting resumable comparison generation"
set +e
"${COLAB[@]}" exec -s "$SESSION" -f "$HERE/remote_generate.py" --timeout 7200 2>&1 | tee -a "$LOG"
GENERATE_RC=${PIPESTATUS[0]}
set -e

log "Recovering current result archive"
if "${COLAB[@]}" download /content/flux2_storybook_results.tar.gz "$RUN_DIR/results.tar.gz" -s "$SESSION" 2>&1 | tee -a "$LOG"; then
    mkdir -p "$RUN_DIR/results"
    tar -xzf "$RUN_DIR/results.tar.gz" -C "$RUN_DIR/results"
fi

if [[ -f "$RUN_DIR/results/failure.json" ]]; then
    log "ERROR remote generation wrote failure.json"
    GENERATE_RC=1
fi

ln -sfn "$RUN_DIR" "$LOCAL_ROOT/latest"
log "Run directory: $RUN_DIR"
if [[ "$STOP_AFTER_RECOVERY" == "true" ]]; then
    log "Stopping recovered Colab session $SESSION"
    "${COLAB[@]}" stop -s "$SESSION" 2>&1 | tee -a "$LOG" || true
else
    log "Colab session left active: $SESSION"
fi
exit "$GENERATE_RC"
