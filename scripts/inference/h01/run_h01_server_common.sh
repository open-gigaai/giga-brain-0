#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
CODE_ROOT=$(dirname "${REPO_ROOT}")
PROJECT_ROOT="${REPO_ROOT}/projects/vla/giga-brain-0"
GIGA_TRAIN_ROOT=${GIGA_TRAIN_ROOT:-"${CODE_ROOT}/giga-train"}
GIGA_DATASETS_ROOT=${GIGA_DATASETS_ROOT:-"${CODE_ROOT}/giga-datasets"}

: "${MODEL_PATH:?Set MODEL_PATH to the resolved checkpoint model_ema directory}"
: "${PRETRAINED_PATH:?Set PRETRAINED_PATH to the PaliGemma2 pretrained directory}"
: "${FAST_TOKENIZER_PATH:?Set FAST_TOKENIZER_PATH to the FAST tokenizer directory}"
: "${NORM_STATS_PATH:?Set NORM_STATS_PATH to the variant norm stats JSON}"
: "${ROBOT_TYPE:?ROBOT_TYPE must be set by the variant wrapper}"
: "${EMBODIMENT_ID:?EMBODIMENT_ID must be set by the variant wrapper}"
: "${ORIGINAL_ACTION_DIM:?ORIGINAL_ACTION_DIM must be set by the variant wrapper}"
: "${EXPECTED_STATE_DIM:?EXPECTED_STATE_DIM must be set by the variant wrapper}"

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8011}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
DTYPE=${DTYPE:-bf16}

case "${DTYPE}" in
  bf16)
    DTYPE_ARGS=(--use-bf16)
    ;;
  fp32)
    DTYPE_ARGS=(--no-use-bf16)
    ;;
  *)
    echo "Unsupported DTYPE=${DTYPE}; expected bf16 or fp32" >&2
    exit 2
    ;;
esac

MOVEMENT_ARGS=()
if [[ ${IS_ROBOT_MOVING:-0} == 1 ]]; then
  MOVEMENT_ARGS=(--is-robot-moving)
elif [[ ${IS_BODY_MOVING:-0} == 1 ]]; then
  MOVEMENT_ARGS=(--is-body-moving)
fi

SAVE_ARGS=()
if [[ -n ${SAVE_DIR:-} ]]; then
  SAVE_ARGS=(--save-dir "${SAVE_DIR}")
fi

export CUDA_VISIBLE_DEVICES
export GIGA_TRAIN_ROOT GIGA_DATASETS_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${REPO_ROOT}:${GIGA_TRAIN_ROOT}:${GIGA_DATASETS_ROOT}:${PYTHONPATH:-}"
export GIGA_UNIFIED_SERVER_SCRIPT="${PROJECT_ROOT}/scripts/inference/inference_agilex_server_unified.py"

exec "${PYTHON_BIN}" -u \
  "${GIGA_UNIFIED_SERVER_SCRIPT}" \
  --model-path "${MODEL_PATH}" \
  --pretrained-path "${PRETRAINED_PATH}" \
  --fast-tokenizer-path "${FAST_TOKENIZER_PATH}" \
  --embodiment-id "${EMBODIMENT_ID}" \
  --norm-stats-path "${NORM_STATS_PATH}" \
  --robot-type "${ROBOT_TYPE}" \
  --original-action-dim "${ORIGINAL_ACTION_DIM}" \
  --expected-state-dim "${EXPECTED_STATE_DIM}" \
  --host "${HOST}" \
  --port "${PORT}" \
  "${DTYPE_ARGS[@]}" \
  "${SAVE_ARGS[@]}" \
  "${MOVEMENT_ARGS[@]}"
