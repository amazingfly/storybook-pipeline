#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-lq-storybook-refinement-v1}"
GPU="${COLAB_GPU:-T4}"
COMPILED_STORY="${COMPILED_STORY:?COMPILED_STORY must point to compiled_story.json}"
RESULTS="${COLAB_RESULTS:?COLAB_RESULTS must point to the existing storybook run}"
INPUT_ARCHIVE="${RESULTS}/logs/refinement/storybook_refinement_inputs.tar"
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
  split -b 40M -d -a 3 "${source}" "${chunk_dir}/part_"
  for part in "${chunk_dir}"/part_*; do
    local uploaded=0
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

cleanup() {
  if [[ -n "${SYNC_PID}" ]]; then
    kill "${SYNC_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${SYNC_STOP_FILE}" ]]; then
    rm -f "${SYNC_STOP_FILE}"
  fi
  if [[ "${ALLOCATED}" -eq 0 ]]; then
    return
  fi
  if [[ "${SUCCESS}" -eq 1 || "${STOP_COLAB_ON_ERROR:-0}" -eq 1 ]]; then
    timeout 60s colab stop -s "${SESSION}" >/dev/null 2>&1 || true
  else
    printf 'Preserving Colab session %s after failure for recovery.\n' \
      "${SESSION}" >&2
  fi
}
trap cleanup EXIT

COMPILED_SHA256="$(sha256sum "${COMPILED_STORY}" | awk '{print $1}')"
CHECKPOINT_SLUG="$(
  jq -r '.title' "${COMPILED_STORY}" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/-/g; s/^-//; s/-$//'
)"
[[ -n "${CHECKPOINT_SLUG}" ]] || CHECKPOINT_SLUG="storybook"
CHECKPOINT_NAME="${CHECKPOINT_SLUG}_${COMPILED_SHA256:0:16}"
LOCAL_CHECKPOINT_DIR="${RESULTS}/accessory_refinement_checkpoint/${CHECKPOINT_NAME}"
REMOTE_CHECKPOINT_DIR="/content/storybook_refinement_checkpoints/${CHECKPOINT_NAME}"
CHECKPOINT_MANIFEST="${LOCAL_CHECKPOINT_DIR}/checkpoint_manifest.json"

mkdir -p "${RESULTS}/logs/refinement" "${LOCAL_CHECKPOINT_DIR}"
if [[ -f "${CHECKPOINT_MANIFEST}" ]]; then
  backup_dir="${RESULTS}/backups/refinement_manifest_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${backup_dir}"
  cp -a "${CHECKPOINT_MANIFEST}" "${backup_dir}/checkpoint_manifest.json"
fi

prepare_args=(
  --compiled-story "${COMPILED_STORY}"
  --run-root "${RESULTS}"
  --output "${INPUT_ARCHIVE}"
)
if [[ -f "${CHECKPOINT_MANIFEST}" ]]; then
  prepare_args+=(--checkpoint-manifest "${CHECKPOINT_MANIFEST}")
fi
python "${ROOT}/scripts/prepare_storybook_refinement_inputs.py" \
  "${prepare_args[@]}"
INPUT_SHA256="$(sha256sum "${INPUT_ARCHIVE}" | awk '{print $1}')"
printf '%s\n' "${INPUT_SHA256}" > "${RESULTS}/logs/refinement/input.sha256"

if timeout 30s colab sessions </dev/null 2>&1 | grep -Fq "[${SESSION}]"; then
  printf 'Reusing preserved Colab session %s.\n' "${SESSION}"
  ALLOCATED=1
else
  for attempt in $(seq 1 "${COLAB_ALLOCATION_ATTEMPTS:-3}"); do
    if colab new -s "${SESSION}" --gpu "${GPU}"; then
      ALLOCATED=1
      break
    fi
    printf 'Colab allocation attempt %s failed; retrying in %s seconds.\n' \
      "${attempt}" "${COLAB_ALLOCATION_DELAY:-60}" >&2
    sleep "${COLAB_ALLOCATION_DELAY:-60}"
  done
fi
if [[ "${ALLOCATED}" -ne 1 ]]; then
  printf 'Unable to allocate a Colab %s session.\n' "${GPU}" >&2
  exit 1
fi

upload_chunked \
  "${INPUT_ARCHIVE}" \
  "storybook_refinement_inputs_${INPUT_SHA256:0:16}.tar.part_"
colab upload -s "${SESSION}" \
  "${RESULTS}/logs/refinement/input.sha256" \
  /content/storybook_refinement_input.sha256
colab upload -s "${SESSION}" \
  "${ROOT}/colab_storybook_mvp_v1/remote_run.py" \
  /content/remote_storybook_mvp_v1_shared.py
colab upload -s "${SESSION}" \
  "${ROOT}/colab_storybook_refinement_v1/remote_run.py" \
  /content/remote_storybook_refinement_v1.py
if [[ -f "${CHECKPOINT_MANIFEST}" ]]; then
  colab upload -s "${SESSION}" \
    "${CHECKPOINT_MANIFEST}" \
    /content/storybook_refinement_checkpoint_manifest.json
fi

set +e
colab exec -s "${SESSION}" \
  -f "${ROOT}/colab_storybook_refinement_v1/launch.py" \
  --timeout 43200 &
EXEC_PID=$!
SYNC_STOP_FILE="$(mktemp)"
rm -f "${SYNC_STOP_FILE}"
python "${ROOT}/scripts/sync_storybook_checkpoints.py" \
  --session "${SESSION}" \
  --remote-dir "${REMOTE_CHECKPOINT_DIR}" \
  --local-dir "${LOCAL_CHECKPOINT_DIR}" \
  --compiled-sha256 "${COMPILED_SHA256}" \
  --extract-root "${RESULTS}" \
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
  printf 'Colab refinement interrupted; downloaded checkpoints are preserved.\n' >&2
  exit "${EXEC_STATUS}"
fi
if [[ "$(jq -r '.status // empty' "${CHECKPOINT_MANIFEST}")" != "complete" ]]; then
  printf 'Refinement process exited without a complete durable manifest.\n' >&2
  exit 1
fi

if timeout 60s colab stop -s "${SESSION}"; then
  ALLOCATED=0
  printf 'Released Colab session before local validation and rendering.\n'
fi
python "${ROOT}/scripts/finalize_storybook_fp16_refinement.py" \
  --compiled-story "${COMPILED_STORY}" \
  --run-root "${RESULTS}"
python "${ROOT}/scripts/render_storybook_mvp.py" \
  --run-root "${RESULTS}" \
  --compiled-story "${COMPILED_STORY}"
SUCCESS=1
printf 'FP16 accessory refinement and local storybook rendering complete at %s\n' \
  "${RESULTS}"
