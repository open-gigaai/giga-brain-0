#!/usr/bin/env bash
# GigaBrain-0.7 RoboTwin 2.0 50-task 后训练启动脚本
#
# 用法:
#   export GIGABRAIN_ROOT=/path/to/giga-brain-0
#   bash scripts/train_robotwin.sh
#   # 或指定别的 config
#   bash scripts/train_robotwin.sh configs/gb07_pg2_robotwin_50task_80k.py
#
# 运行前请先改好 config 里的用户配置区 (DATA_ROOT / NORM_STATS_PATH /
# PRETRAINED_CKPT / PROJECT_DIR / tokenizer 路径)。

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RELEASE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# giga-brain-0 仓库根目录
GIGABRAIN_ROOT=${GIGABRAIN_ROOT:-$(cd "${RELEASE_DIR}/../.." && pwd)}
CONFIG=${1:-${RELEASE_DIR}/configs/gb07_pg2_robotwin_50task_80k.py}

if [ ! -f "${CONFIG}" ]; then
  echo "[ERROR] config not found: ${CONFIG}" >&2
  exit 1
fi
if [ ! -f "${GIGABRAIN_ROOT}/scripts/train.py" ]; then
  echo "[ERROR] GIGABRAIN_ROOT 不像 giga-brain-0 仓库: ${GIGABRAIN_ROOT}" >&2
  echo "        请设置 export GIGABRAIN_ROOT=/path/to/giga-brain-0" >&2
  exit 1
fi

export PYTHONPATH=${GIGABRAIN_ROOT}:${PYTHONPATH:-}

echo "=================================================="
echo "GigaBrain-0.7 RoboTwin post-training"
echo "=================================================="
echo "GIGABRAIN_ROOT : ${GIGABRAIN_ROOT}"
echo "CONFIG         : ${CONFIG}"
echo "=================================================="

cd "${GIGABRAIN_ROOT}"
exec python scripts/train.py --config "${CONFIG}"
