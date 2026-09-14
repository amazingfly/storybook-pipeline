#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-lq-storybook-mvp-v1}"
GPU="${COLAB_GPU:-T4}"
BUNDLE="${ROOT}/storybook_mvp_v1/colab/storybook_mvp_v1_bundle.tar.gz"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS="${COLAB_RESULTS:-${ROOT}/outputs/storybook_mvp_v1/colab_run_${RUN_STAMP}}"
COMPILED_STORY="${COMPILED_STORY:?COMPILED_STORY must point to a compiled story JSON file}"
SUCCESS=0
ALLOCATED=0
EXEC_PID=""
SYNC_PID=""
SYNC_STOP_FILE=""

upload_chunked() {
  local source="$1"
  local remote_prefix="$2"
  local chunk_dir
  chunk_dir="$(mktemp -d)"
  split -b 40M -d -a 2 "${source}" "${chunk_dir}/part_"
  for part in "${chunk_dir}"/part_*; do
    uploaded=0
    for attempt in 1 2 3; do
      if colab upload -s "${SESSION}" "${part}" \
          "/content/${remote_prefix}$(basename "${part}")"; then
        uploaded=1
        break
      fi
      printf 'Upload failed for %s (attempt %s/3); retrying in 15 seconds.\n' \
        "$(basename "${part}")" "${attempt}" >&2
      sleep 15
    done
    if [[ "${uploaded}" -ne 1 ]]; then
      printf 'Unable to upload %s after three attempts.\n' "${part}" >&2
      rm -rf "${chunk_dir}"
      return 1
    fi
  done
  rm -rf "${chunk_dir}"
}

refresh_colab_file_token() {
  local colab_python
  colab_python="$(sed -n '1s/^#!//p' "$(command -v colab)")"
  "${colab_python}" "${ROOT}/scripts/refresh_colab_runtime_token.py" \
    --session "${SESSION}"
}

cleanup() {
  if [[ -n "${SYNC_PID}" ]]; then
    kill "${SYNC_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${SYNC_STOP_FILE}" ]]; then
    rm -f "${SYNC_STOP_FILE}"
  fi
  if [[ "${ALLOCATED}" -eq 0 ]]; then
    return
  elif [[ "${SUCCESS}" -eq 1 || "${STOP_COLAB_ON_ERROR:-0}" -eq 1 ]]; then
    timeout 60s colab stop -s "${SESSION}" >/dev/null 2>&1 || true
  else
    printf 'Preserving Colab session %s after failure for recovery.\n' "${SESSION}" >&2
  fi
}
trap cleanup EXIT

python "${ROOT}/scripts/build_storybook_mvp_v1_bundle.py" \
  --compiled-story "${COMPILED_STORY}"
COMPILED_SHA256="$(sha256sum "${COMPILED_STORY}" | awk '{print $1}')"
CHECKPOINT_SLUG="$(
  jq -r '.title' "${COMPILED_STORY}" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/-/g; s/^-//; s/-$//'
)"
[[ -n "${CHECKPOINT_SLUG}" ]] || CHECKPOINT_SLUG="storybook"
CHECKPOINT_NAME="${CHECKPOINT_SLUG}_${COMPILED_SHA256:0:16}"
LOCAL_CHECKPOINT_DIR="${RESULTS}/durable_checkpoint/${CHECKPOINT_NAME}"
REMOTE_CHECKPOINT_DIR="/content/storybook_checkpoints/${CHECKPOINT_NAME}"
if timeout 30s colab sessions </dev/null 2>&1 | grep -Fq "[${SESSION}]"; then
  printf 'Reusing preserved Colab session %s.\n' "${SESSION}"
  RESUMING=1
else
  ALLOCATED=0
  for attempt in $(seq 1 "${COLAB_ALLOCATION_ATTEMPTS:-4}"); do
    if colab new -s "${SESSION}" --gpu "${GPU}"; then
      ALLOCATED=1
      break
    fi
    printf 'Colab allocation attempt %s failed; retrying in %s seconds.\n' \
      "${attempt}" "${COLAB_ALLOCATION_DELAY:-60}" >&2
    sleep "${COLAB_ALLOCATION_DELAY:-60}"
  done
  if [[ "${ALLOCATED}" -eq 0 ]]; then
    printf 'Unable to allocate a Colab %s session after all retries.\n' "${GPU}" >&2
    exit 1
  fi
  RESUMING=0
fi
ALLOCATED=1
upload_chunked "${BUNDLE}" "storybook_mvp_v1_bundle.tar.gz.part_"
if [[ -f "${LOCAL_CHECKPOINT_DIR}/checkpoint_manifest.json" ]]; then
  colab upload -s "${SESSION}" \
    "${LOCAL_CHECKPOINT_DIR}/checkpoint_manifest.json" \
    /content/storybook_checkpoint_manifest.json
  printf 'Uploaded checkpoint manifest only (%s completed files indexed).\n' \
    "$(jq '.files | length' "${LOCAL_CHECKPOINT_DIR}/checkpoint_manifest.json")"
fi
if [[ -n "${COLAB_REUSE_BASES:-}" ]]; then
  [[ -d "${COLAB_REUSE_BASES}" ]] || {
    printf 'COLAB_REUSE_BASES is not a directory: %s\n' "${COLAB_REUSE_BASES}" >&2
    exit 1
  }
  BASE_ARCHIVE="$(mktemp --suffix=.tar.gz)"
  tar -czf "${BASE_ARCHIVE}" -C "$(dirname "${COLAB_REUSE_BASES}")" \
    "$(basename "${COLAB_REUSE_BASES}")"
  upload_chunked "${BASE_ARCHIVE}" "storybook_mvp_v1_bases.tar.gz.part_"
  rm -f "${BASE_ARCHIVE}"
fi
printf 'LoRAs will be downloaded and checksum-verified from Google Drive.\n'
colab upload -s "${SESSION}" \
  "${ROOT}/colab_storybook_mvp_v1/remote_run.py" \
  /content/remote_storybook_mvp_v1.py
set +e
colab exec -s "${SESSION}" \
  -f "${ROOT}/colab_storybook_mvp_v1/launch.py" \
  --timeout 21600 &
EXEC_PID=$!
SYNC_STOP_FILE="$(mktemp)"
rm -f "${SYNC_STOP_FILE}"
python "${ROOT}/scripts/sync_storybook_checkpoints.py" \
  --session "${SESSION}" \
  --remote-dir "${REMOTE_CHECKPOINT_DIR}" \
  --local-dir "${LOCAL_CHECKPOINT_DIR}" \
  --compiled-sha256 "${COMPILED_SHA256}" \
  --watch-pid "${EXEC_PID}" \
  --stop-file "${SYNC_STOP_FILE}" &
SYNC_PID=$!
wait "${EXEC_PID}"
EXEC_STATUS=$?
if [[ "${EXEC_STATUS}" -eq 0 ]]; then
  touch "${SYNC_STOP_FILE}"
fi
wait "${SYNC_PID}"
SYNC_PID=""
rm -f "${SYNC_STOP_FILE}"
SYNC_STOP_FILE=""
set -e
if [[ "${EXEC_STATUS}" -ne 0 ]]; then
  printf 'Colab execution disconnected; attempting artifact recovery before failing.\n' >&2
fi

mkdir -p "${RESULTS}"
if [[ "${STORYBOOK_GENERATION_ONLY:-1}" == "1" ]]; then
  if [[ "${EXEC_STATUS}" -ne 0 ]]; then
    printf 'Base generation interrupted; durable local batches are preserved.\n' >&2
    exit "${EXEC_STATUS}"
  fi
  if timeout 60s colab stop -s "${SESSION}"; then
    ALLOCATED=0
    printf 'Released Colab session before local post-processing.\n'
  else
    printf '%s\n' \
      'Warning: unable to release Colab before local post-processing; cleanup will retry when the workflow exits.' >&2
  fi
  python "${ROOT}/scripts/run_storybook_q8.py" \
    "${COMPILED_STORY}" \
    "${RESULTS}" \
    --checkpoint-dir "${LOCAL_CHECKPOINT_DIR}" \
    --limit-new 0
  python "${ROOT}/scripts/finalize_storybook_q8.py" \
    "${COMPILED_STORY}" \
    "${RESULTS}"
  SUCCESS=1
  printf 'Downloaded missing Colab images and finished validation locally at %s\n' \
    "${RESULTS}"
  exit 0
fi

refresh_colab_file_token
COLAB_PYTHON="$(sed -n '1s/^#!//p' "$(command -v colab)")"
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" \
  /content/lq_storybook_mvp_v1_results.tar.gz \
  "${RESULTS}/results.tar.gz"
rm -rf \
  "${RESULTS}/base" \
  "${RESULTS}/composite" \
  "${RESULTS}/regalia" \
  "${RESULTS}/wand_only" \
  "${RESULTS}/grip_repaired" \
  "${RESULTS}/final" \
  "${RESULTS}/metadata" \
  "${RESULTS}/review" \
  "${RESULTS}/logs" \
  "${RESULTS}/summary.json" \
  "${RESULTS}/validation_report.json" \
  "${RESULTS}/compiled_story.json"
tar -xzf "${RESULTS}/results.tar.gz" -C "${RESULTS}"
python "${ROOT}/scripts/validate_storybook_accessories_v4.py" \
  --run-root "${RESULTS}" \
  --summary "${RESULTS}/summary.json" \
  --output "${RESULTS}/validation_report_local_recheck.json"
SUCCESS=1
printf 'Downloaded JSON-driven storybook run to %s\n' "${RESULTS}"
