#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CKPT_PATH="${CKPT_PATH:-model-repos/spirit-v1.5}"
DATASET_PATH="${DATASET_PATH:-datasets/public_datasets/VLM/vqa/EmbSpatial-Bench}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/embspatial_vlm}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXTRA_ARGS=()
if [[ -n "${BACKBONE_PATH:-}" ]]; then
  EXTRA_ARGS+=(--backbone-path "${BACKBONE_PATH}")
fi
if [[ -n "${PROCESSOR_PATH:-}" ]]; then
  EXTRA_ARGS+=(--processor-path "${PROCESSOR_PATH}")
fi

if [[ ! -f "${CKPT_PATH}/model.safetensors" ]]; then
  echo "[ERROR] model.safetensors not found in CKPT_PATH: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CKPT_PATH}/config.json" ]]; then
  echo "[ERROR] config.json not found in CKPT_PATH: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_PATH}/data/test-00000-of-00001.parquet" ]]; then
  echo "[ERROR] EmbSpatial-Bench test parquet not found: ${DATASET_PATH}/data/test-00000-of-00001.parquet" >&2
  exit 1
fi

echo "[INFO] CKPT_PATH=${CKPT_PATH}"
echo "[INFO] DATASET_PATH=${DATASET_PATH}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "[INFO] DEVICE=${DEVICE}"
echo "[INFO] DTYPE=${DTYPE}"
echo "[INFO] BATCH_SIZE=${BATCH_SIZE}"
echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"
if [[ -n "${BACKBONE_PATH:-}" ]]; then
  echo "[INFO] BACKBONE_PATH=${BACKBONE_PATH}"
fi
if [[ -n "${PROCESSOR_PATH:-}" ]]; then
  echo "[INFO] PROCESSOR_PATH=${PROCESSOR_PATH}"
fi

"${PYTHON_BIN}" -m evaluations.embspatial_vlm \
  --ckpt-path "${CKPT_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
