#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:?Usage: run_storybook_v2.sh EXISTING_RUN_DIRECTORY}"
COMPILED="${OUTPUT}/compiled_story.json"

[[ -f "${OUTPUT}/summary.json" ]] || {
  printf 'Missing candidate summary: %s\n' "${OUTPUT}/summary.json" >&2
  exit 1
}
[[ -f "${COMPILED}" ]] || {
  printf 'Missing compiled story: %s\n' "${COMPILED}" >&2
  exit 1
}

python -u "${ROOT}/scripts/render_storybook_mvp_v2.py" \
  --run-root "${OUTPUT}" \
  --compiled-story "${COMPILED}"

printf 'V2 storybook video: %s/story_v2/storybook.mp4\n' "${OUTPUT}"
