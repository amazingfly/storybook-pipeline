#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORY="${1:?Usage: run_storybook.sh STORY.json [OUTPUT_DIRECTORY]}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${2:-${ROOT}/outputs/storybook_mvp_v1/run_${RUN_STAMP}}"
COMPILED="${OUTPUT}/compiled_story.json"

mkdir -p "${OUTPUT}"
python "${ROOT}/scripts/compile_storybook_script.py" \
  "${STORY}" \
  --output "${COMPILED}"

if [[ "${SKIP_COLAB:-0}" != "1" ]]; then
  COMPILED_STORY="${COMPILED}" \
  COLAB_RESULTS="${OUTPUT}" \
  COLAB_SESSION="${COLAB_SESSION:-lq-storybook-mvp-v1}" \
    "${ROOT}/colab_storybook_mvp_v1/run_check.sh"
fi

[[ -f "${OUTPUT}/summary.json" ]] || {
  printf 'Missing Colab summary: %s\n' "${OUTPUT}/summary.json" >&2
  exit 1
}

python "${ROOT}/scripts/render_storybook_mvp.py" \
  --run-root "${OUTPUT}" \
  --compiled-story "${OUTPUT}/compiled_story.json"

printf 'Storybook video: %s/story/storybook.mp4\n' "${OUTPUT}"
