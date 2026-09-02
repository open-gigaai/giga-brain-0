#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(pwd)

# policy server 地址
MODEL_HOST=${MODEL_HOST:-127.0.0.1}
BASE_PORT=${BASE_PORT:-8000}
NUM_PORTS=${NUM_PORTS:-8}
HORIZON=${HORIZON:-20}
# 评测服务地址
EVAL_ENDPOINT=${EVAL_ENDPOINT:-http://127.0.0.1:8087}
# 在线评测 token
EBENCH_SUBMIT_TOKEN=${EBENCH_SUBMIT_TOKEN:-}
# 评测运行 ID
RUN_ID=${RUN_ID:-}

LOG_DIR=${SCRIPT_DIR}/logs/ebench/$(date +%Y%m%d_%H%M%S)
mkdir -p "${LOG_DIR}"

pids=()
log_files=()
tail_pid=""
trap 'kill "${pids[@]}" "${tail_pid}" 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM

for ((index = 0; index < NUM_PORTS; index++)); do
  port=$((BASE_PORT + index))
  log_file=${LOG_DIR}/ebench_port${port}_worker${index}.log
  touch "${log_file}"
  echo "[${index}] ${MODEL_HOST}:${port} -> ${log_file}"

  python -u "${SCRIPT_DIR}/scripts/ebench_eval_client.py" \
    --run-id "${RUN_ID}" \
    --worker-id "${index}" \
    --eval-endpoint "${EVAL_ENDPOINT}" \
    --eval-token "${EBENCH_SUBMIT_TOKEN}" \
    --model-host "${MODEL_HOST}" \
    --model-port "${port}" \
    --horizon "${HORIZON}" \
    >"${log_file}" 2>&1 &
  pids+=("$!")
  log_files+=("${log_file}")
done

tail -n 100 -F "${log_file}" &
tail_pid=$!

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[index]}"; then
    status=1
    echo "[ERROR] worker ${index} exited, log: ${log_files[index]}" >&2
    tail -n 50 "${log_files[index]}" >&2
  fi
done

kill "${tail_pid}" 2>/dev/null || true
tail_pid=""
exit "${status}"
