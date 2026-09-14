#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/storage/projects/agentic/images/sdxl_littlequeen_v1"
RUN="${ROOT}/outputs/storybook_mvp_v1/the_witches_trick_20260728"
COMPILED="${RUN}/compiled_story.json"
MASK_WORKERS="${STORYBOOK_MASK_WORKERS:-2}"
LOG="${RUN}.log"

exec > >(tee -a "${LOG}") 2>&1

python "${ROOT}/scripts/finalize_storybook_q8.py" \
  "${COMPILED}" \
  "${RUN}" \
  --mask-workers "${MASK_WORKERS}"

python "${ROOT}/scripts/render_storybook_mvp.py" \
  --run-root "${RUN}" \
  --compiled-story "${COMPILED}"
