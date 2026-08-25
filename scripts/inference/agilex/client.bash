#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
cd "${REPO_ROOT}"
python projects/vla/giga-brain-0/scripts/inference/inference_agilex_client_unified_smooth.py \
  --host 127.0.0.1 \
  --port 8081 \
  --task-name "Put the cucumber into the woven basket." \
  --image-mode float_native \
  --force-rgb \
  --apply-gripper-rescale \
  --publish-rate 30 \
  --chunk-size 50 \
  --inference-trigger-remaining 30 \
  --max-action-execute-horizon 50 \
  --distance-thresh 0.5 \
  --align-search-window 8
