#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${COLAB_SESSION:-lq-storybook-accessories-v2}"
GPU="${COLAB_GPU:-T4}"
BUNDLE="${ROOT}/storybook_accessories_v1/colab/storybook_accessories_v2_bundle.tar.gz"
RESULTS="${ROOT}/outputs/storybook_accessories_v2/colab_bare_base_832x1216_20260722"
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

"${ROOT}/scripts/build_storybook_accessory_v2_bundle.py"
colab new -s "${SESSION}" --gpu "${GPU}"
ALLOCATED=1
upload_chunked "${BUNDLE}" "storybook_accessories_v2_bundle.tar.gz.part_"
upload_chunked "${ROOT}/models/loras/lqxl_sdxl_v2.safetensors" "lqxl_sdxl_v2.safetensors.part_"
colab upload -s "${SESSION}" "${ROOT}/colab_storybook_accessories_v2/remote_run.py" /content/remote_storybook_accessories_v2.py
colab exec -s "${SESSION}" -f "${ROOT}/colab_storybook_accessories_v2/launch.py" --timeout 21600

mkdir -p "${RESULTS}"
COLAB_PYTHON="$(sed -n '1s/^#!//p' "$(command -v colab)")"
"${COLAB_PYTHON}" "${ROOT}/scripts/recover_colab_artifact.py" \
  --session "${SESSION}" /content/lq_storybook_accessories_v2_results.tar.gz "${RESULTS}/results.tar.gz"
rm -rf "${RESULTS}/base" "${RESULTS}/composite" "${RESULTS}/final" \
  "${RESULTS}/metadata" "${RESULTS}/review" "${RESULTS}/summary.json"
tar -xzf "${RESULTS}/results.tar.gz" -C "${RESULTS}"
SUCCESS=1
printf 'Downloaded bare-base accessory validation to %s\n' "${RESULTS}"
