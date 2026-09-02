#!/usr/bin/env bash
# Standalone MiMo-Embodied VQA runner for the VLM tower inside Hy-VLA.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIMO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCHMARK_ROOT="$(cd "${MIMO_ROOT}/.." && pwd)"
HY_VLA_REPO="${HY_VLA_REPO:-${BENCHMARK_ROOT}/model-repos/Hy-Embodied-0.5-VLA-main}"
HY_VLA_DEFAULT_VENV="${HY_VLA_REPO}/.venv-shared"

DEFAULT_CHECKPOINT="${BENCHMARK_ROOT}/model-repos/hy-vla/Hy-Embodied-0.5-VLA-RoboTwin"
DEFAULT_OUT_DIR="${MIMO_ROOT}/eval_results"
DEFAULT_GPU_IDS="0"
DEFAULT_TASKS="where2place_point,roboafford,part_affordance,roborefit,vabench_point_box,cvbench_boxed,erqa_boxed,embspatialbench,sat,robospatial,refspatialbench,crpe_relation,metavqa_eval"

CHECKPOINT="${HY_VLA_CHECKPOINT:-${DEFAULT_CHECKPOINT}}"
OUT_DIR="${HY_VLA_VQA_OUT_DIR:-${DEFAULT_OUT_DIR}}"
DATA_ROOT="${MIMO_DATA_ROOT:-${MIMO_ROOT}/datasets/public_datasets/VLM}"
GPU_IDS="${GPU_IDS:-${DEFAULT_GPU_IDS}}"
TASK_OVERRIDE=""
LIMIT="${HY_VLA_VQA_LIMIT:-}"
GEN_KWARGS="${HY_VLA_VQA_GEN_KWARGS:-}"
DRY_RUN=0
pids=()
LOG_DIR=""
FAILURE_DIR=""

usage() {
    cat <<USAGE
Usage:
  bash runners/mimo_hy_vla.sh [--checkpoint DIR] [--out-dir DIR] [--data-root DIR] [--gpus 0,1] [--tasks TASK1,TASK2] [--limit N] [--dry-run]

Environment:
  HY_VLA_PYTHON_BIN=model-repos/Hy-Embodied-0.5-VLA-main/.venv-shared/bin/python
  HY_VLA_REPO=${HY_VLA_REPO}
  HY_VLA_CHECKPOINT=${DEFAULT_CHECKPOINT}
  HY_VLA_VQA_RESULT_NAME=hy-vla-robotwin-vqa
  HY_VLA_VQA_IMAGE_MODE=all  # all|first
  HY_VLA_VQA_MAX_NEW_TOKENS=128
  HY_VLA_VQA_TEMPERATURE=0.0
USAGE
}

kill_tree() {
    local signal="$1" pid="$2" child
    while read -r child; do
        [ -n "${child}" ] || continue
        kill_tree "${signal}" "${child}"
    done < <(pgrep -P "${pid}" 2>/dev/null || true)
    kill "-${signal}" "${pid}" 2>/dev/null || true
}

cleanup() {
    local status="${1:-130}" pid
    trap - INT TERM
    for pid in "${pids[@]}"; do kill_tree TERM "${pid}"; done
    sleep 2
    for pid in "${pids[@]}"; do kill_tree KILL "${pid}"; done
    [ -n "${LOG_DIR}" ] && echo "Logs: ${LOG_DIR}" >&2
    exit "${status}"
}
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --checkpoint) CHECKPOINT="${2:-}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
        --data-root) DATA_ROOT="${2:-}"; shift 2 ;;
        --gpus) GPU_IDS="${2:-}"; shift 2 ;;
        --tasks) TASK_OVERRIDE="${2:-}"; shift 2 ;;
        --limit) LIMIT="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

read -r -a GPUS <<< "${GPU_IDS//,/ }"
for gpu in "${GPUS[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid GPU id: ${gpu}" >&2; exit 2; }
done
[ "${#GPUS[@]}" -gt 0 ] || { echo "ERROR: no GPUs provided" >&2; exit 2; }

TASK_SOURCE="${TASK_OVERRIDE:-${DEFAULT_TASKS}}"
read -r -a TASKS <<< "${TASK_SOURCE//,/ }"
[ "${#TASKS[@]}" -gt 0 ] || { echo "ERROR: no tasks provided" >&2; exit 2; }

if [ "${DRY_RUN}" -eq 1 ]; then
    printf 'Model: hy_vla_vqa\nCheckpoint: %s\nRepo: %s\nData: %s\nGPUs: %s\nTasks: %s\n' \
        "${CHECKPOINT}" "${HY_VLA_REPO}" "${DATA_ROOT}" "${GPUS[*]}" "${TASKS[*]}"
    exit 0
fi

[ -n "${CHECKPOINT}" ] || { echo "ERROR: empty checkpoint" >&2; exit 2; }
[ -f "${CHECKPOINT}/model.safetensors" ] || {
    echo "ERROR: model.safetensors not found in ${CHECKPOINT}" >&2
    exit 2
}
[ -d "${DATA_ROOT}" ] || { echo "ERROR: dataset root not found: ${DATA_ROOT}" >&2; exit 2; }

if [ -n "${HY_VLA_PYTHON_BIN:-}" ]; then
    PYTHON="${HY_VLA_PYTHON_BIN}"
elif [ -x "${HY_VLA_DEFAULT_VENV}/bin/python" ]; then
    PYTHON="${HY_VLA_DEFAULT_VENV}/bin/python"
else
    echo "ERROR: set HY_VLA_PYTHON_BIN to an environment containing Hy-VLA and lmms_eval dependencies" >&2
    exit 2
fi

if ! "${PYTHON}" -B - <<'PY' >/dev/null; then
import accelerate
import safetensors
import torch
import transformers
PY
    echo "ERROR: selected Python is missing Hy-VLA dependencies: ${PYTHON}" >&2
    exit 2
fi

export GIGA_BENCHMARK_ROOT="${MIMO_ROOT}"
export PYTHONPATH="${MIMO_ROOT}:${MIMO_ROOT}/patches:${HY_VLA_REPO}:${PYTHONPATH:-}"
export MIMO_DATA_ROOT="${DATA_ROOT}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export QWEN_RESIZE_MAX_PIXELS="${QWEN_RESIZE_MAX_PIXELS:-50176}"

MODEL_ARGS="checkpoint_path=${CHECKPOINT},repo_path=${HY_VLA_REPO}"
MODEL_ARGS="${MODEL_ARGS},max_new_tokens=${HY_VLA_VQA_MAX_NEW_TOKENS:-128}"
MODEL_ARGS="${MODEL_ARGS},temperature=${HY_VLA_VQA_TEMPERATURE:-0.0},top_p=${HY_VLA_VQA_TOP_P:-1.0}"
MODEL_ARGS="${MODEL_ARGS},thinking=${HY_VLA_VQA_THINKING:-false},image_mode=${HY_VLA_VQA_IMAGE_MODE:-all}"

RESULT_NAME="${HY_VLA_VQA_RESULT_NAME:-hy-vla-robotwin-vqa}"
mkdir -p "${OUT_DIR}/${RESULT_NAME}"
LOG_DIR="${OUT_DIR}/logs/$(date +%Y%m%d_%H%M%S)_hy_vla"
mkdir -p "${LOG_DIR}"
FAILURE_DIR="$(mktemp -d "${LOG_DIR}/failures.XXXXXX")"

LIMIT_ARGS=()
[ -n "${LIMIT}" ] && LIMIT_ARGS=(--limit "${LIMIT}")
GEN_ARGS=()
[ -n "${GEN_KWARGS}" ] && GEN_ARGS=(--gen_kwargs "${GEN_KWARGS}")

cat <<SUMMARY
Runner: runners/mimo_hy_vla.sh
Python: ${PYTHON}
Backend: hy_vla_vqa
Checkpoint: ${CHECKPOINT}
Output: ${OUT_DIR}/${RESULT_NAME}
GPUs: ${GPUS[*]}
Tasks: ${TASKS[*]}
Limit: ${LIMIT:-none}
Model args: ${MODEL_ARGS}
Logs: ${LOG_DIR}
SUMMARY

run_worker() {
    local worker_idx="$1" gpu="$2" task_idx task task_log task_output task_status worker_status failure_file
    worker_status=0
    failure_file="${FAILURE_DIR}/gpu_${gpu}.txt"
    for task_idx in "${!TASKS[@]}"; do
        [ $((task_idx % ${#GPUS[@]})) -eq "${worker_idx}" ] || continue
        task="${TASKS[$task_idx]}"
        task_log="${LOG_DIR}/gpu_${gpu}_${task}.log"
        task_output="${OUT_DIR}/${RESULT_NAME}/${task}"
        if [ -e "${task_output}" ]; then
            worker_status=2
            echo "[GPU ${gpu}] FAIL ${task}: output already exists: ${task_output}" | tee -a "${failure_file}" >&2
            continue
        fi
        echo "[GPU ${gpu}] START ${task} -> ${task_log}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m accelerate.commands.launch \
            --num_processes=1 \
            --main_process_port "$((29900 + gpu))" \
            -m lmms_eval \
            --model hy_vla_vqa \
            --model_args "${MODEL_ARGS}" \
            --tasks "${task}" \
            --batch_size 1 \
            --log_samples \
            --log_samples_suffix "${RESULT_NAME}" \
            --output_path "${task_output}" \
            "${LIMIT_ARGS[@]}" \
            "${GEN_ARGS[@]}" \
            >"${task_log}" 2>&1
        task_status=$?
        if [ "${task_status}" -eq 0 ]; then
            if ! find "${task_output}" -type f -name '*_results.json' -print -quit 2>/dev/null | grep -q . \
                || ! find "${task_output}" -type f -name '*_samples_*.jsonl' -print -quit 2>/dev/null | grep -q .; then
                task_status=1
                echo "[GPU ${gpu}] evaluation returned success without results and samples" >>"${task_log}"
            fi
        fi
        if [ "${task_status}" -ne 0 ]; then
            worker_status="${task_status}"
            echo "[GPU ${gpu}] FAIL ${task} status=${task_status}" | tee -a "${failure_file}" >&2
        else
            echo "[GPU ${gpu}] DONE ${task}"
        fi
    done
    exit "${worker_status}"
}

for worker_idx in "${!GPUS[@]}"; do
    run_worker "${worker_idx}" "${GPUS[$worker_idx]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then status=1; fi
done
trap - INT TERM

if [ "${status}" -ne 0 ]; then
    echo "Some tasks failed. Logs: ${LOG_DIR}" >&2
    find "${FAILURE_DIR}" -maxdepth 1 -type f -print -exec sed -n '1,120p' {} \; >&2
    exit "${status}"
fi
echo "All tasks finished. Logs: ${LOG_DIR}"
