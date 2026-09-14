#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-lq-storybook-accessories-v4-auto}"
GPU="${COLAB_GPU:-T4}"
BUNDLE="${ROOT}/storybook_accessories_v4/colab/storybook_accessories_v4_auto_bundle.tar.gz"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS="${COLAB_RESULTS:-${ROOT}/outputs/storybook_accessories_v4_auto/colab_run_${RUN_STAMP}}"
SUCCESS=0
ALLOCATED=0

upload_chunked() {
  local source="$1"
  local remote_prefix="$2"
  local chunk_dir
  chunk_dir="$(mktemp -d)"
  split -b 40M -d -a 2 "${source}" "${chunk_dir}/part_"
  for part in "${chunk_dir}"/part_*; do
    colab upload -s "${SESSION}" "${part}" "/content/${remote_prefix}$(basename "${part}")"
  done
  rm -rf "${chunk_dir}"
}

cleanup() {
  if [[ "${ALLOCATED}" -eq 0 ]]; then
    return
  elif [[ "${SUCCESS}" -eq 1 || "${STOP_COLAB_ON_ERROR:-0}" -eq 1 ]]; then
    timeout 60s colab stop -s "${SESSION}" >/dev/null 2>&1 || true
  else
    printf 'Preserving Colab session %s after failure for recovery.\n' "${SESSION}" >&2
  fi
}
trap cleanup EXIT

python "${ROOT}/scripts/build_storybook_accessories_v4_auto_bundle.py"
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
upload_chunked "${BUNDLE}" "storybook_accessories_v4_auto_bundle.tar.gz.part_"
if [[ -n "${COLAB_REUSE_BASES:-}" ]]; then
  [[ -d "${COLAB_REUSE_BASES}" ]] || {
    printf 'COLAB_REUSE_BASES is not a directory: %s\n' "${COLAB_REUSE_BASES}" >&2
    exit 1
  }
  BASE_ARCHIVE="$(mktemp --suffix=.tar.gz)"
  tar -czf "${BASE_ARCHIVE}" -C "$(dirname "${COLAB_REUSE_BASES}")" \
    "$(basename "${COLAB_REUSE_BASES}")"
  upload_chunked "${BASE_ARCHIVE}" "storybook_accessories_v4_auto_bases.tar.gz.part_"
  rm -f "${BASE_ARCHIVE}"
fi
if [[ "${RESUMING}" -eq 0 ]]; then
  upload_chunked "${ROOT}/models/loras/lqxl_sdxl_v2.safetensors" "lqxl_sdxl_v2.safetensors.part_"
  upload_chunked "${ROOT}/models/loras/lqmoonfit_sdxl_v1.safetensors" "lqmoonfit_sdxl_v1.safetensors.part_"
  upload_chunked "${ROOT}/models/loras/lqmoonregalia_sdxl_v4.safetensors" "lqmoonregalia_sdxl_v4.safetensors.part_"
  upload_chunked "${ROOT}/models/loras/lqmoonwand_sdxl_v4.safetensors" "lqmoonwand_sdxl_v4.safetensors.part_"
fi
colab upload -s "${SESSION}" \
  "${ROOT}/colab_storybook_accessories_v4_auto/remote_run.py" \
  /content/remote_storybook_accessories_v4_auto.py
if ! colab exec -s "${SESSION}" \
  -f "${ROOT}/colab_storybook_accessories_v4_auto/launch.py" \
  --timeout 21600; then
  printf 'Colab execution disconnected; attempting artifact recovery before failing.\n' >&2
fi

mkdir -p "${RESULTS}"
COLAB_PYTHON="$(sed -n '1s/^#!//p' "$(command -v colab)")"
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" \
  /content/lq_storybook_accessories_v4_auto_results.tar.gz \
  "${RESULTS}/results.tar.gz"
rm -rf \
  "${RESULTS}/base" \
  "${RESULTS}/composite" \
  "${RESULTS}/regalia" \
  "${RESULTS}/final" \
  "${RESULTS}/metadata" \
  "${RESULTS}/review" \
  "${RESULTS}/logs" \
  "${RESULTS}/summary.json" \
  "${RESULTS}/validation_report.json" \
  "${RESULTS}/auto_pipeline_config.json"
tar -xzf "${RESULTS}/results.tar.gz" -C "${RESULTS}"
python "${ROOT}/scripts/validate_storybook_accessories_v4.py" \
  --run-root "${RESULTS}" \
  --summary "${RESULTS}/summary.json" \
  --output "${RESULTS}/validation_report_local_recheck.json"
SUCCESS=1
printf 'Downloaded unattended V4 storybook run to %s\n' "${RESULTS}"
