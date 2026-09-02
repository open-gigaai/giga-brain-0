#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(pwd)

# GigaBrain 0.7 仓库目录
GIGABRAIN_ROOT=${GIGABRAIN_ROOT:-${SCRIPT_DIR}/../../giga-brain-0}  
SERVER_SCRIPT=${GIGABRAIN_ROOT}/scripts/inference/inference_agilex_server_unified.py
# 模型目录
MODEL_PATH=${MODEL_PATH:-}  
NORM_STATS_PATH=${NORM_STATS_PATH:-}
TOKENIZER_PATH=/shared_disk/models/huggingface/models--google--paligemma2-3b-pt-224
FAST_TOKENIZER_PATH=/shared_disk/models/huggingface/models--physical-intelligence--fast
# 起始服务端口
BASE_PORT=${BASE_PORT:-8000}
NUM_PORTS=${NUM_PORTS:-8}

LOG_DIR=${SCRIPT_DIR}/logs/giga/$(date +%Y%m%d_%H%M%S)
IFS=',' read -ra gpu_list <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
mkdir -p "${LOG_DIR}"

pids=()
tail_pid=""
trap 'kill "${pids[@]}" "${tail_pid}" 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM

for ((index = 0; index < NUM_PORTS; index++)); do
  gpu=${gpu_list[$((index % ${#gpu_list[@]}))]}
  port=$((BASE_PORT + index))
  log_file=${LOG_DIR}/gpu${gpu}_port${port}.log
  touch "${log_file}"
  echo "[${index}] GPU ${gpu}, port ${port} -> ${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu}" python -u "${SERVER_SCRIPT}" \
    --model-path "${MODEL_PATH}" \
    --norm-stats-path "${NORM_STATS_PATH}" \
    --pretrained-path "${TOKENIZER_PATH}" \
    --fast-tokenizer-path "${FAST_TOKENIZER_PATH}" \
    --embodiment-id 0 \
    --original-action-dim 17 \
    --expected-state-dim 14 \
    --is-robot-moving \
    --host 0.0.0.0 \
    --port "${port}" \
    >"${log_file}" 2>&1 &
  pids+=("$!")
done

tail -n 100 -F "${log_file}" &
tail_pid=$!

wait -n "${pids[@]}"
