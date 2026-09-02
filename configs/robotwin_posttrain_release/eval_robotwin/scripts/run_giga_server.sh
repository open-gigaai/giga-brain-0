#!/usr/bin/env bash
# GigaBrain-0.7 RoboTwin 评测服务端 (单端口 ZMQ policy server)
#
# 用法:
#   export GIGABRAIN_ROOT=/path/to/giga-brain-0
#   export MODEL_PATH=/path/to/checkpoint/model_ema
#   export NORM_STATS_PATH=/path/to/giga_norm_stats.json
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_giga_server.sh
#
# 协议 (必须与训练 config 和客户端一致):
#   state  = left_joints(6) + left_gripper(1) + right_joints(6) + right_gripper(1) = 14D
#   action = 同上                                                                  = 14D
#
# 一个端口服务多个并行客户端：客户端请求在 server 端串行排队，
# 单条推理约 7~12s，客户端侧超时已按并发排队留了余量 (300s)。
# 想提高吞吐就在多张卡上起多个 server (换 PORT)，客户端按 SERVER_PORT 分流。

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# ========== 用户配置区 ==========

# giga-brain-0 仓库根目录
GIGABRAIN_ROOT=${GIGABRAIN_ROOT:-$(cd "${EVAL_DIR}/../../.." && pwd)}
SERVER_SCRIPT=${GIGABRAIN_ROOT}/scripts/inference/inference_agilex_server_unified.py

# 训练输出的 checkpoint 目录 (含 config.json + diffusion_pytorch_model.bin)
MODEL_PATH=${MODEL_PATH:-}
# 训练时用的 norm_stats.json (14 维)
NORM_STATS_PATH=${NORM_STATS_PATH:-}
# PaliGemma2 HF tokenizer 目录 (对应训练 config 的 prompt_cfg.tokenizer_model_path)
PRETRAINED_PATH=${PRETRAINED_PATH:-/path/to/huggingface/models--google--paligemma2-3b-pt-224}
# FAST action tokenizer 目录 (对应 prompt_cfg.fast_tokenizer_path)
FAST_TOKENIZER_PATH=${FAST_TOKENIZER_PATH:-/path/to/huggingface/models--physical-intelligence--fast}

# embodiment id：训练 config 未开启 robot_type_embodiment_id_overrides，用默认 0。
# 若你训练时把 agilex_cobot_magic 映射到 6，这里也要改成 6。
EMBODIMENT_ID=${EMBODIMENT_ID:-0}
# delta-mask 的 robot_type 键，对应训练 config 的 delta_action_cfg.mask
ROBOT_TYPE=${ROBOT_TYPE:-agilex_cobot_magic}
# RoboTwin 双臂有效维度；模型内部 pad 到 max_action_dim=32
ORIGINAL_ACTION_DIM=${ORIGINAL_ACTION_DIM:-14}
EXPECTED_STATE_DIM=${EXPECTED_STATE_DIM:-14}

# ZMQ 绑定地址
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8081}

# bf16 推理 (默认)。设为 false 走 fp32，显存翻倍。
USE_BF16=${USE_BF16:-true}
# 可选：dump 每步推理 IO 用于离线对比
SAVE_DIR=${SAVE_DIR:-}

# ========== 参数校验 ==========

if [ -z "${MODEL_PATH}" ]; then
  echo "[ERROR] 请设置 MODEL_PATH (checkpoint 目录，例如 .../model_ema)" >&2
  exit 1
fi
if [ -z "${NORM_STATS_PATH}" ]; then
  echo "[ERROR] 请设置 NORM_STATS_PATH (训练时的 norm_stats.json)" >&2
  exit 1
fi
if [ ! -f "${SERVER_SCRIPT}" ]; then
  echo "[ERROR] 找不到 server 脚本: ${SERVER_SCRIPT}" >&2
  echo "        请设置 export GIGABRAIN_ROOT=/path/to/giga-brain-0" >&2
  exit 1
fi

export PYTHONPATH=${GIGABRAIN_ROOT}:${PYTHONPATH:-}

echo "=================================================="
echo "GigaBrain-0.7 RoboTwin Inference Server"
echo "=================================================="
echo "Model Path      : ${MODEL_PATH}"
echo "Pretrained Path : ${PRETRAINED_PATH}"
echo "Norm Stats      : ${NORM_STATS_PATH}"
echo "Embodiment ID   : ${EMBODIMENT_ID}"
echo "Robot Type      : ${ROBOT_TYPE}"
echo "Action Dim      : ${ORIGINAL_ACTION_DIM}"
echo "State Dim       : ${EXPECTED_STATE_DIM}"
echo "ZMQ Address     : ${HOST}:${PORT}"
echo "CUDA devices    : ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "=================================================="

CMD=(python -u "${SERVER_SCRIPT}"
  --model-path "${MODEL_PATH}"
  --pretrained-path "${PRETRAINED_PATH}"
  --fast-tokenizer-path "${FAST_TOKENIZER_PATH}"
  --norm-stats-path "${NORM_STATS_PATH}"
  --embodiment-id "${EMBODIMENT_ID}"
  --robot-type "${ROBOT_TYPE}"
  --original-action-dim "${ORIGINAL_ACTION_DIM}"
  --expected-state-dim "${EXPECTED_STATE_DIM}"
  --host "${HOST}"
  --port "${PORT}"
)

if [ "${USE_BF16}" = false ]; then
  CMD+=(--no-use-bf16)
fi
if [ -n "${SAVE_DIR}" ]; then
  CMD+=(--save-dir "${SAVE_DIR}")
fi

echo "执行命令:"
printf '%q ' "${CMD[@]}"
echo
echo

exec "${CMD[@]}"
