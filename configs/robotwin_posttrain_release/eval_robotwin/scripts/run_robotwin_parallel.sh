#!/usr/bin/env bash
# GigaBrain-0.7 RoboTwin 2.0 50-task 并行评测客户端
#
# 用法: bash scripts/run_robotwin_parallel.sh [NUM_CLIENTS=8] [TEST_NUM=100] [SAVE_VIDEO=0] [POS_LOOKAHEAD_STEP=50]
#   bash scripts/run_robotwin_parallel.sh 24              # 24 客户端, 每任务 100 集, 不存视频, 每次执行 50 步
#   bash scripts/run_robotwin_parallel.sh 24 100 0 30     # 每次推理只执行 30 步 (更高频重规划)
#
# 服务端需先在 GPU 上起好 (见 scripts/run_giga_server.sh)，然后:
#   SERVER_HOST=<server-ip> SERVER_PORT=8081 bash scripts/run_robotwin_parallel.sh 24 100 0 30
#
# 每个任务跑 demo_clean 和 demo_randomized 两种配置。
# 任务分配策略：LPT 贪心装箱，按 step_limit 权重把 50 个任务均分到 N 个客户端。

set -euo pipefail

NUM_CLIENTS=${1:-8}
TEST_NUM=${2:-100}
SAVE_VIDEO=${3:-0}
POS_LOOKAHEAD_STEP=${4:-50}

# policy server 地址
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
SERVER_PORT=${SERVER_PORT:-8081}

# RoboTwin 仓库根目录 (客户端必须在这里运行)
ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-$(pwd)}
# RoboTwin conda 环境的 python
PYTHON=${PYTHON:-python}
# 本目录 (eval_robotwin)，用于定位 client 脚本
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CLIENT_SCRIPT=${CLIENT_SCRIPT:-${SCRIPT_DIR}/robotwin_eval_client.py}

# 可分配给客户端的 GPU (渲染用，不含服务端那张卡)，多个客户端轮询共享
IFS=',' read -ra CLIENT_GPUS <<< "${CLIENT_GPUS:-0,1,2,3,4,5,6,7}"

if [ ! -f "${ROBOTWIN_ROOT}/envs/__init__.py" ]; then
  echo "[ERROR] ROBOTWIN_ROOT 不像 RoboTwin 仓库: ${ROBOTWIN_ROOT}" >&2
  echo "        请 cd 到 RoboTwin 根目录运行，或设置 export ROBOTWIN_ROOT=/path/to/RoboTwin" >&2
  exit 1
fi
if [ ! -f "${CLIENT_SCRIPT}" ]; then
  echo "[ERROR] 找不到 client 脚本: ${CLIENT_SCRIPT}" >&2
  exit 1
fi

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
SAVE_ROOT=${SAVE_ROOT:-${ROBOTWIN_ROOT}/eval_result_gigabrain07_parallel/${TIMESTAMP}}
LOG_DIR=${SAVE_ROOT}/logs
mkdir -p "${LOG_DIR}"

# 用 LPT 贪心算法按 step_limit 权重均分任务
ASSIGNMENT_FILE=$(mktemp)
trap 'rm -f "${ASSIGNMENT_FILE}"' EXIT

"${PYTHON}" - "${NUM_CLIENTS}" <<'EOF' > "${ASSIGNMENT_FILE}"
import heapq
import sys

# (task_name, step_limit 权重)：权重越大越先分配，用于让各客户端总耗时接近
TASKS_WITH_WEIGHTS = [
    ("adjust_bottle",             400),
    ("beat_block_hammer",         400),
    ("blocks_ranking_rgb",       1200),
    ("blocks_ranking_size",      1200),
    ("click_alarmclock",          400),
    ("click_bell",                400),
    ("dump_bin_bigbin",           600),
    ("grab_roller",               400),
    ("handover_block",            800),
    ("handover_mic",              600),
    ("hanging_mug",               900),
    ("lift_pot",                  400),
    ("move_can_pot",              400),
    ("move_pillbottle_pad",       400),
    ("move_playingcard_away",     400),
    ("move_stapler_pad",          400),
    ("open_laptop",               700),
    ("open_microwave",           1500),
    ("pick_diverse_bottles",      400),
    ("pick_dual_bottles",         400),
    ("place_a2b_left",            400),
    ("place_a2b_right",           400),
    ("place_bread_basket",        700),
    ("place_bread_skillet",       500),
    ("place_burger_fries",        500),
    ("place_can_basket",          700),
    ("place_cans_plasticbox",     800),
    ("place_container_plate",     400),
    ("place_dual_shoes",          600),
    ("place_empty_cup",           500),
    ("place_fan",                 400),
    ("place_mouse_pad",           400),
    ("place_object_basket",       700),
    ("place_object_scale",        400),
    ("place_object_stand",        400),
    ("place_phone_stand",         400),
    ("place_shoe",                500),
    ("press_stapler",             400),
    ("put_bottles_dustbin",      1700),
    ("put_object_cabinet",        700),
    ("rotate_qrcode",             400),
    ("scan_object",               500),
    ("shake_bottle",              700),
    ("shake_bottle_horizontally", 700),
    ("stack_blocks_three",       1200),
    ("stack_blocks_two",          800),
    ("stack_bowls_three",        1200),
    ("stack_bowls_two",           900),
    ("stamp_seal",                400),
    ("turn_switch",               400),
]

n_clients = max(1, min(int(sys.argv[1]), len(TASKS_WITH_WEIGHTS)))

# LPT：权重降序，依次分配给当前负载最轻的客户端
heap = [(0, i, []) for i in range(n_clients)]
heapq.heapify(heap)
for task, weight in sorted(TASKS_WITH_WEIGHTS, key=lambda x: -x[1]):
    load, cid, task_list = heapq.heappop(heap)
    task_list.append(task)
    heapq.heappush(heap, (load + weight, cid, task_list))

for cid, task_list, load in sorted((c, t, l) for l, c, t in heap):
    print(f"{cid}\t{','.join(task_list)}\t{load}")
EOF

echo "=========================================="
echo "GigaBrain-0.7 RoboTwin 并行评测"
echo "=========================================="
echo "总任务数        : 50 (x2 配置: demo_clean + demo_randomized)"
echo "并行客户端数    : ${NUM_CLIENTS}"
echo "每任务集数      : ${TEST_NUM}"
echo "Policy server   : tcp://${SERVER_HOST}:${SERVER_PORT}"
echo "Lookahead step  : ${POS_LOOKAHEAD_STEP}"
echo "保存视频        : $([ "${SAVE_VIDEO}" -eq 1 ] && echo '是' || echo '否')"
echo "结果目录        : ${SAVE_ROOT}"
echo "=========================================="
echo

PIDS=()
while IFS=$'\t' read -r CLIENT_ID TASKS_STR LOAD; do
  GPU=${CLIENT_GPUS[$((CLIENT_ID % ${#CLIENT_GPUS[@]}))]}
  IFS=',' read -ra SUBSET <<< "${TASKS_STR}"

  echo "客户端 ${CLIENT_ID} → GPU ${GPU} → 预估负载 ${LOAD} → ${#SUBSET[@]} 个任务: ${SUBSET[*]}"

  (
    export CUDA_VISIBLE_DEVICES=${GPU}
    cd "${ROBOTWIN_ROOT}"

    if [ "${SAVE_VIDEO}" -eq 1 ]; then
      CONFIG_CLEAN="task_config/demo_clean.yml"
      CONFIG_RAND="task_config/demo_randomized.yml"
    else
      # 关掉视频保存：复制一份临时 config 到日志目录
      CONFIG_CLEAN="${LOG_DIR}/demo_clean_novideo_${CLIENT_ID}.yml"
      CONFIG_RAND="${LOG_DIR}/demo_randomized_novideo_${CLIENT_ID}.yml"
      sed 's/eval_video_log: true/eval_video_log: false/' \
        task_config/demo_clean.yml > "${CONFIG_CLEAN}"
      sed 's/eval_video_log: true/eval_video_log: false/' \
        task_config/demo_randomized.yml > "${CONFIG_RAND}"
    fi

    for TASK in "${SUBSET[@]}"; do
      for SETTING in demo_clean demo_randomized; do
        if [ "${SETTING}" = demo_clean ]; then
          CFG="${CONFIG_CLEAN}"
        else
          CFG="${CONFIG_RAND}"
        fi
        TASK_LOG="${LOG_DIR}/${TASK}_${SETTING}.log"
        echo "[客户端${CLIENT_ID} | GPU${GPU}] 开始: $(date '+%F %T')" > "${TASK_LOG}"
        "${PYTHON}" -u "${CLIENT_SCRIPT}" \
          --config "${CFG}" \
          --task_name "${TASK}" \
          --ckpt_setting "${SETTING}" \
          --test_num "${TEST_NUM}" \
          --seed 0 \
          --instruction_type unseen \
          --host "${SERVER_HOST}" \
          --port "${SERVER_PORT}" \
          --image_mode float_native \
          --pos_lookahead_step "${POS_LOOKAHEAD_STEP}" \
          --save_dir "${SAVE_ROOT}" \
          >> "${TASK_LOG}" 2>&1 || \
          echo "[客户端${CLIENT_ID}] ${TASK}/${SETTING} 异常退出" >> "${TASK_LOG}"
        echo "[客户端${CLIENT_ID} | GPU${GPU}] 结束: $(date '+%F %T')" >> "${TASK_LOG}"
      done
    done

    [ "${SAVE_VIDEO}" -eq 0 ] && rm -f "${CONFIG_CLEAN}" "${CONFIG_RAND}"
  ) &

  PIDS+=($!)
done < "${ASSIGNMENT_FILE}"

echo
echo "已启动 ${#PIDS[@]} 个并行客户端，PID: ${PIDS[*]}"
echo "任务日志目录: ${LOG_DIR}"
echo "等待所有客户端完成..."

FAILED=0
for (( i=0; i<${#PIDS[@]}; i++ )); do
  if wait "${PIDS[$i]}"; then
    echo "客户端 ${i} (PID ${PIDS[$i]}) 完成"
  else
    echo "客户端 ${i} (PID ${PIDS[$i]}) 异常退出"
    FAILED=$((FAILED + 1))
  fi
done

echo
echo "=========================================="
echo "评测完成。失败客户端数: ${FAILED}"
echo "结果保存在: ${SAVE_ROOT}"
echo "汇总成功率: python scripts/parse_eval_results.py ${SAVE_ROOT} --accepted-totals ${TEST_NUM}"
echo "=========================================="
