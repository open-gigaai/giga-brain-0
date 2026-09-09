#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
CODE_ROOT=$(dirname "${REPO_ROOT}")
GIGA_TRAIN_ROOT=${GIGA_TRAIN_ROOT:-"${CODE_ROOT}/giga-train"}
GIGA_DATASETS_ROOT=${GIGA_DATASETS_ROOT:-"${CODE_ROOT}/giga-datasets"}
export GIGA_TRAIN_ROOT GIGA_DATASETS_ROOT
export PYTHONPATH="${REPO_ROOT}:${GIGA_TRAIN_ROOT}:${GIGA_DATASETS_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES=0 python scripts/inference/inference_agilex_server_unified.py \
  --model-path /path/to/your/model/gb1_pg2_pick_and_place_30k_release/model_ema \
  --pretrained-path /path/to/your/model/models--google--paligemma2-3b-pt-224 \
  --fast-tokenizer-path /path/to/your/model/models--physical-intelligence--fast \
  --norm-stats-path /path/to/your/norm/norm_stat_giga_robot_1_0_agilex.json \
  --embodiment-id 6 \
  --robot-type agilex_cobot_magic \
  --original-action-dim 14 \
  --expected-state-dim 14 \
  --host 127.0.0.1 \
  --port 8081
