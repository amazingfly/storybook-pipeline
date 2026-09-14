#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/storage/projects/agentic/images/sdxl_littlequeen_v1"
RUN="${ROOT}/outputs/storybook_mvp_v1/the_witches_trick_20260728"
COMPILED="${RUN}/compiled_story.json"
LOG="${RUN}.log"

exec > >(tee -a "${LOG}") 2>&1
exec env \
  COMPILED_STORY="${COMPILED}" \
  COLAB_RESULTS="${RUN}" \
  COLAB_SESSION="${COLAB_SESSION:-lq-witches-refinement-v1}" \
  COLAB_ALLOCATION_ATTEMPTS="${COLAB_ALLOCATION_ATTEMPTS:-2}" \
  COLAB_ALLOCATION_DELAY="${COLAB_ALLOCATION_DELAY:-30}" \
  "${ROOT}/colab_storybook_refinement_v1/run_check.sh"
