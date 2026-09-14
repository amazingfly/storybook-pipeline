#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORY="${1:?Usage: run_storybook_q8.sh STORY.json OUTPUT_DIRECTORY [generator options]}"
OUTPUT="${2:?Usage: run_storybook_q8.sh STORY.json OUTPUT_DIRECTORY [generator options]}"
shift 2
COMPILED="${OUTPUT}/compiled_story.json"

mkdir -p "${OUTPUT}"
python "${ROOT}/scripts/compile_storybook_script.py" \
  "${STORY}" \
  --output "${COMPILED}"
python -u "${ROOT}/scripts/run_storybook_q8.py" \
  "${COMPILED}" \
  "${OUTPUT}" \
  "$@"

if [[ "$(jq -r '.status' "${OUTPUT}/q8_generation_state.json")" == "base_complete" ]]; then
  python -u "${ROOT}/scripts/finalize_storybook_q8.py" \
    "${COMPILED}" \
    "${OUTPUT}"
  python -u "${ROOT}/scripts/render_storybook_mvp.py" \
    --run-root "${OUTPUT}" \
    --compiled-story "${COMPILED}"
fi
