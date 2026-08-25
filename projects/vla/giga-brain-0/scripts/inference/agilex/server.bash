#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
CODE_ROOT=$(dirname "${REPO_ROOT}")
PROJECT_ROOT="${REPO_ROOT}/projects/vla/giga-brain-0"
GIGA_TRAIN_ROOT=${GIGA_TRAIN_ROOT:-"${CODE_ROOT}/giga-train"}
GIGA_DATASETS_ROOT=${GIGA_DATASETS_ROOT:-"${CODE_ROOT}/giga-datasets"}
export GIGA_TRAIN_ROOT GIGA_DATASETS_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${REPO_ROOT}:${GIGA_TRAIN_ROOT}:${GIGA_DATASETS_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES=0 python projects/vla/giga-brain-0/scripts/inference/inference_agilex_server_unified.py \
  --model-path /home/ubuntu/users/yunmo.wang/codes_release/gb1_pg2_pick_and_place_30k_from_200k_release/model_ema \
  --pretrained-path /home/ubuntu/users/peng.li/model/huggingface/models--google--paligemma2-3b-pt-224 \
  --fast-tokenizer-path /home/ubuntu/users/yunmo.wang/dit/models--physical-intelligence--fast \
  --norm-stats-path /home/ubuntu/users/yunmo.wang/codes_release/pick_and_place.json \
  --embodiment-id 6 \
  --robot-type agilex_cobot_magic \
  --original-action-dim 14 \
  --expected-state-dim 14 \
  --host 127.0.0.1 \
  --port 8081
