#!/usr/bin/env bash
set -euo pipefail

TOKEN_CYCLE="/home/derek/projects/agentic/ltxVideo/scripts/colab_token_cycle.py"
CONFIG_DIR="/home/derek/.config/colab-cli"
RUNNER="/mnt/storage/projects/agentic/images/sdxl_littlequeen_v1/storybook_mvp_v1/run_storybook.sh"
STORY="/mnt/storage/projects/agentic/images/theWitchesTrick/theWitchesTrickStory.json"
OUTPUT="/mnt/storage/projects/agentic/images/sdxl_littlequeen_v1/outputs/storybook_mvp_v1/the_witches_trick_20260728"
LOG="${OUTPUT}.log"

python "${TOKEN_CYCLE}" --config-dir "${CONFIG_DIR}"
colab whoami

mkdir -p "${OUTPUT}"
exec env \
  COLAB_SESSION="lq-witches-trick-v1" \
  COLAB_ALLOCATION_ATTEMPTS="2" \
  COLAB_ALLOCATION_DELAY="30" \
  STORYBOOK_GENERATION_ONLY="1" \
  STORYBOOK_MASK_WORKERS="2" \
  "${RUNNER}" "${STORY}" "${OUTPUT}" 2>&1 \
  | tee -a "${LOG}"
