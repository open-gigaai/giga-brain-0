#!/bin/bash
set -euo pipefail

# H01 远程推理 client —— smooth(平滑滑动窗口)版启动脚本(在 Jetson 上直接运行)。
# 对应 nvidia_client_smooth.py；网络和调度参数可通过环境变量覆盖。

# ===== 开关 =====
# ready pose 默认关闭；确认环境安全后设置 INIT_POSE=--init_pose，读取任务首帧 NPY 并移动（2s 平滑）。
INIT_POSE=${INIT_POSE:---no_init_pose}
case "${INIT_POSE}" in
  --init_pose|--no_init_pose) ;;
  *) echo "INIT_POSE must be --init_pose or --no_init_pose" >&2; exit 2 ;;
esac
READY_POSE_NPY=${READY_POSE_NPY:-}
READY_POSE_ARGS=()
if [[ -n ${READY_POSE_NPY} ]]; then
  READY_POSE_ARGS=(--ready_pose_npy "${READY_POSE_NPY}")
fi
# 默认 dry-run；确认 22D state / 16D action topic 映射和动作数值后，显式设置 EXECUTE_ACTION=--execute_action。
EXECUTE_ACTION=${EXECUTE_ACTION:-}
# dump 发给 server 的前几帧 obs 到 debug/obs_dump：要开就改成  DUMP="--dump_obs"
DUMP=""
# server 加载 push_button/beside_plate 模型，按需改下面 --prompt（用本任务对应指令）。

# ===== ROS2 环境 =====
# ROS generated setup files may probe unset hook variables.
set +u
if [[ -f /opt/ros/humble/setup.bash ]]; then
  source /opt/ros/humble/setup.bash
elif [[ -f /opt/ros/foxy/setup.bash ]]; then
  source /opt/ros/foxy/setup.bash
else
  echo "ROS2 setup.bash was not found" >&2
  exit 2
fi

# 当前 H01 镜像把 fixed_image_msgs/cv_bridge 放在 xrobotoolkit 的 bundled overlay。
ROS_INTERFACES_SETUP=${ROS_INTERFACES_SETUP:-/home/nvidia/miniconda3/envs/xrobotoolkit-humble/lib/python3.10/site-packages/xrobotoolkit_teleop/_bundled/ros_interfaces/install/setup.bash}
IMAGE_BRIDGE_SETUP=${IMAGE_BRIDGE_SETUP:-/home/nvidia/workspace/sax/GigaAI_Image_Bridge/install/setup.bash}
if [[ -f ${ROS_INTERFACES_SETUP} ]]; then
  source "${ROS_INTERFACES_SETUP}"
elif [[ -f ${IMAGE_BRIDGE_SETUP} ]]; then
  source "${IMAGE_BRIDGE_SETUP}"
else
  echo "H01 ROS interface overlay was not found" >&2
  exit 2
fi
set -u

# 仅在 UDP allowlist 覆盖本机接口时使用机器人 FastDDS profile。
# IP 变化后继续强制加载旧 profile 会导致 FastDDS 过滤全部网卡。
FAST_DDS_PROFILE=${FAST_DDS_PROFILE:-/etc/robot/fastdds/shm_fastdds_optimized.xml}
USE_FAST_DDS_PROFILE=0
if [[ -f ${FAST_DDS_PROFILE} ]]; then
  if ! grep -q '<interfaceWhiteList>' "${FAST_DDS_PROFILE}"; then
    USE_FAST_DDS_PROFILE=1
  else
    for LOCAL_IP in $(hostname -I); do
      if grep -Fq "<address>${LOCAL_IP}</address>" "${FAST_DDS_PROFILE}"; then
        USE_FAST_DDS_PROFILE=1
        break
      fi
    done
  fi
fi
if (( USE_FAST_DDS_PROFILE )); then
  export FASTRTPS_DEFAULT_PROFILES_FILE="${FAST_DDS_PROFILE}"
  export RMW_FASTRTPS_USE_QOS_FROM_XML=1
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
else
  unset FASTRTPS_DEFAULT_PROFILES_FILE RMW_FASTRTPS_USE_QOS_FROM_XML
  echo "[WARN] FastDDS profile missing or its allowlist does not match local IPs; using builtin transports" >&2
fi

# openpi_client 源码目录（image_tools / websocket_client_policy）
OPENPI_CLIENT_SRC=${OPENPI_CLIENT_SRC:-/home/nvidia/Gigabrain-Client/openpi-client/src}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${SCRIPT_DIR}:${OPENPI_CLIENT_SRC}:/usr/lib/python3/dist-packages:${PYTHONPATH:-}"
: "${SERVER_HOST:?SERVER_HOST is required; set it to the inference server IP address}"
SERVER_PORT=${SERVER_PORT:-8011}
DEFAULT_PYTHON=/home/nvidia/miniconda3/envs/xrobotoolkit-humble/bin/python
if [[ -x ${DEFAULT_PYTHON} ]]; then
  PYTHON_BIN=${PYTHON_BIN:-${DEFAULT_PYTHON}}
else
  PYTHON_BIN=${PYTHON_BIN:-python3}
fi
CHUNK_SIZE=${CHUNK_SIZE:-50}
SERVER_STATE_DIM=${SERVER_STATE_DIM:-22}
ACTION_DIM=${ACTION_DIM:-16}
PROMPT=${PROMPT:-Push the yellow button next to the plate}
INFERENCE_TRIGGER_REMAINING=${INFERENCE_TRIGGER_REMAINING:-30}
MAX_ACTION_EXECUTE_HORIZON=${MAX_ACTION_EXECUTE_HORIZON:-35}

# 平滑调度参数（对齐参考 smooth 版；卡顿就调大 --inference_trigger_remaining）：
#   --publish_rate 匀速发布频率  --chunk_size 服务端单次返回动作数
#   --inference_trigger_remaining num1(剩余这么多提前重推理)  --max_action_execute_horizon 单次最多执行
#   --distance_thresh 自适应horizon阈值  --continuity_blend_steps 拼接连续性衰减步数  --align_search_window 就近对齐半窗
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/nvidia_client_smooth.py" \
  --server_host "${SERVER_HOST}" \
  --server_port "${SERVER_PORT}" \
  --robot_type h01 \
  --fisheye_mode 3fisheye \
  --policy_action_space qpose \
  --server_state_dim "${SERVER_STATE_DIM}" \
  --action_dim "${ACTION_DIM}" \
  --transport zmq \
  --prompt "${PROMPT}" \
  --publish_rate 30 \
  --chunk_size "${CHUNK_SIZE}" \
  --inference_trigger_remaining "${INFERENCE_TRIGGER_REMAINING}" \
  --max_action_execute_horizon "${MAX_ACTION_EXECUTE_HORIZON}" \
  --distance_thresh 0.5 \
  --continuity_blend_steps 10 \
  --align_search_window 8 \
  "${READY_POSE_ARGS[@]}" \
  ${INIT_POSE} ${EXECUTE_ACTION} ${DUMP}
