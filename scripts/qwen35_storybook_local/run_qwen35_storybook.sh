#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUN="/mnt/storage/projects/agentic/images/sdxl_littlequeen_v1/outputs/storybook_mvp_v1/the_witches_trick_20260728"
RUN_ROOT="${1:-${DEFAULT_RUN}}"

[[ -f "${RUN_ROOT}/summary.json" ]] || {
  printf 'Missing summary: %s/summary.json\n' "${RUN_ROOT}" >&2
  exit 1
}
[[ -f "${RUN_ROOT}/compiled_story.json" ]] || {
  printf 'Missing compiled story: %s/compiled_story.json\n' "${RUN_ROOT}" >&2
  exit 1
}

exec env PYTHONUNBUFFERED=1 python3 \
  "${WORKFLOW_DIR}/render_storybook_qwen35.py" \
  --run-root "${RUN_ROOT}" \
  --compiled-story "${RUN_ROOT}/compiled_story.json"
