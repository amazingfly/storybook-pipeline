#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config_local_storybook_v2.json}"

python3 "$SCRIPT_DIR/run_storybook_accessories_v2_local.py" "$CONFIG"
