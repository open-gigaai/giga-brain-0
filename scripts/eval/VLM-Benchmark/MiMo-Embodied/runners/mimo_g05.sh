#!/usr/bin/env bash
# 用途：运行 OpenGalaxea G05 在 MiMo-Embodied 上的 VQA/选择题评测。
# Self-contained MiMo-Embodied runner for OpenGalaxea G05.
# This script intentionally does not call any other .sh runner.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIMO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCHMARK_ROOT="$(cd "${MIMO_ROOT}/.." && pwd)"
G05_REPO="${G05_REPO:-${BENCHMARK_ROOT}/model-repos/GalaxeaVLA-main}"
G05_MODEL_ROOT="${G05_MODEL_ROOT:-${BENCHMARK_ROOT}/model-repos/g05/G05-local}"
G05_DEFAULT_VENV="${G05_REPO}/.venv-$(hostname -s)"
G05_FALLBACK_VENVS=(
    "${G05_DEFAULT_VENV}"
    "${G05_REPO}/.venv"
    "${G05_REPO}/.venv-shared"
)

DEFAULT_OUT_DIR="${MIMO_ROOT}/eval_results"
DEFAULT_GPU_IDS="0"
DEFAULT_TASKS="where2place_point,roboafford,part_affordance,roborefit,vabench_point_box,cvbench_boxed,erqa_boxed,embspatialbench,sat,robospatial,refspatialbench,crpe_relation,metavqa_eval"

OUT_DIR="${G05_OUT_DIR:-${DEFAULT_OUT_DIR}}"
DATA_ROOT="${MIMO_DATA_ROOT:-${MIMO_ROOT}/datasets/public_datasets/VLM}"
GPU_IDS="${GPU_IDS:-${DEFAULT_GPU_IDS}}"
TASK_OVERRIDE=""
LIMIT="${G05_LIMIT:-}"
GEN_KWARGS="${G05_GEN_KWARGS:-}"
DRY_RUN=0
pids=()
LOG_DIR=""
FAILURE_DIR=""

usage() {
    cat <<USAGE
Usage:
  bash runners/mimo_g05.sh [--out-dir DIR] [--data-root DIR] [--gpus 0,1] [--tasks TASK1,TASK2] [--limit N] [--dry-run]

Defaults:
  out:   ${DEFAULT_OUT_DIR}
  gpus:  ${DEFAULT_GPU_IDS}
  tasks: ${DEFAULT_TASKS}

Env overrides:
  G05_REPO=${G05_REPO}
  G05_PYTHON_BIN=model-repos/GalaxeaVLA-main/.venv/bin/python
  G05_SHARED_SITE_PACKAGES=model-repos/GalaxeaVLA-main/.venv/lib/python3.10/site-packages
  G05_LOCAL_ROOT=/tmp/g05-local
  G05_MODEL_ROOT=${G05_MODEL_ROOT}
  G05_VARIANT=base  # base|droid|libero|robotwin20|so101
  G05_OUT_DIR=${DEFAULT_OUT_DIR}
  G05_MAX_NEW_TOKENS=512
  G05_TEMPERATURE=0.0
  G05_GEN_KWARGS='max_new_tokens=512'
USAGE
}

kill_tree() {
    local signal="$1"
    local pid="$2"
    local child
    while read -r child; do
        [ -n "${child}" ] || continue
        kill_tree "${signal}" "${child}"
    done < <(pgrep -P "${pid}" 2>/dev/null || true)
    kill "-${signal}" "${pid}" 2>/dev/null || true
}

cleanup() {
    local status="${1:-130}"
    local pid
    trap - INT TERM
    echo "Stopping GPU workers..." >&2
    for pid in "${pids[@]}"; do
        kill_tree TERM "${pid}"
    done
    sleep 2
    for pid in "${pids[@]}"; do
        kill_tree KILL "${pid}"
    done
    [ -n "${LOG_DIR}" ] && echo "Logs: ${LOG_DIR}" >&2
    exit "${status}"
}
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --out-dir) OUT_DIR="${2:-}"; [ -n "${OUT_DIR}" ] || { echo "ERROR: --out-dir requires a value" >&2; exit 2; }; shift 2 ;;
        --data-root) DATA_ROOT="${2:-}"; [ -n "${DATA_ROOT}" ] || { echo "ERROR: --data-root requires a value" >&2; exit 2; }; shift 2 ;;
        --gpus) GPU_IDS="${2:-}"; [ -n "${GPU_IDS}" ] || { echo "ERROR: --gpus requires a value" >&2; exit 2; }; shift 2 ;;
        --tasks) TASK_OVERRIDE="${2:-}"; [ -n "${TASK_OVERRIDE}" ] || { echo "ERROR: --tasks requires a value" >&2; exit 2; }; shift 2 ;;
        --limit) LIMIT="${2:-}"; [ -n "${LIMIT}" ] || { echo "ERROR: --limit requires a value" >&2; exit 2; }; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

GPU_IDS_NORMALIZED="${GPU_IDS//,/ }"
read -r -a RAW_GPUS <<< "${GPU_IDS_NORMALIZED}"
GPUS=()
for gpu in "${RAW_GPUS[@]}"; do
    [ -n "${gpu}" ] || continue
    [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid GPU id: ${gpu}" >&2; exit 2; }
    GPUS+=("${gpu}")
done
[ "${#GPUS[@]}" -gt 0 ] || { echo "ERROR: no valid GPU ids provided" >&2; exit 2; }

TASK_SOURCE="${TASK_OVERRIDE:-${DEFAULT_TASKS}}"
TASK_SOURCE="${TASK_SOURCE//,/ }"
read -r -a TASK_CANDIDATES <<< "${TASK_SOURCE}"
TASKS=()
for task in "${TASK_CANDIDATES[@]}"; do
    [ -n "${task}" ] || continue
    TASKS+=("${task}")
done
[ "${#TASKS[@]}" -gt 0 ] || { echo "ERROR: no tasks to run" >&2; exit 2; }

if [ "${DRY_RUN}" -eq 1 ]; then
    printf 'Model: g05_vqa\nModel root: %s\nRepo: %s\nData: %s\nGPUs: %s\nTasks: %s\n' \
        "${G05_MODEL_ROOT}" "${G05_REPO}" "${DATA_ROOT}" "${GPUS[*]}" "${TASKS[*]}"
    exit 0
fi

[ -d "${DATA_ROOT}" ] || { echo "ERROR: dataset root not found: ${DATA_ROOT}" >&2; exit 2; }

cd "${MIMO_ROOT}" || exit 1
if [ -n "${G05_PYTHON_BIN:-}" ]; then
    PYTHON="${G05_PYTHON_BIN}"
else
    PYTHON=""
    for candidate in "${G05_FALLBACK_VENVS[@]}"; do
        if [ -x "${candidate}/bin/python" ]; then
            PYTHON="${candidate}/bin/python"
            break
        fi
    done
    if [ -z "${PYTHON}" ]; then
        echo "ERROR: cannot find a G05 Python env under ${G05_REPO}" >&2
        echo "Checked:" >&2
        for candidate in "${G05_FALLBACK_VENVS[@]}"; do
            echo "  ${candidate}/bin/python" >&2
        done
        echo "Create it on this machine first:" >&2
        echo "  bash scripts/setup_g05_env.sh" >&2
        echo "Or set G05_PYTHON_BIN to a Python executable with G05 dependencies." >&2
        exit 2
    fi
fi

if [ -n "${PYTHON_BIN:-}" ] && [ "${PYTHON_BIN}" != "${PYTHON}" ]; then
    echo "WARNING: ignoring PYTHON_BIN=${PYTHON_BIN}; G05 runner uses ${PYTHON}" >&2
    echo "Use G05_PYTHON_BIN if you need to override the G05 Python executable." >&2
fi

G05_SHARED_SITE_PACKAGES="${G05_SHARED_SITE_PACKAGES:-}"
if [ -n "${G05_SHARED_SITE_PACKAGES}" ] && [ -d "${G05_SHARED_SITE_PACKAGES}" ]; then
    export PYTHONPATH="${G05_SHARED_SITE_PACKAGES}:${PYTHONPATH:-}"
fi

if ! "${PYTHON}" - <<'PY' >/dev/null; then
import diskcache  # noqa: F401
import decord  # noqa: F401
import torch  # noqa: F401
PY
    echo "ERROR: selected Python is missing required packages: ${PYTHON}" >&2
    echo "Expected a Python env with G05 dependencies. Create it with:" >&2
    echo "  bash scripts/setup_g05_env.sh" >&2
    exit 2
fi

export GIGA_BENCHMARK_ROOT="${MIMO_ROOT}"
export PYTHONPATH="${MIMO_ROOT}:${MIMO_ROOT}/patches:${G05_REPO}/src:${PYTHONPATH:-}"
export MIMO_DATA_ROOT="${DATA_ROOT}"
export QWEN_RESIZE_MAX_PIXELS="${QWEN_RESIZE_MAX_PIXELS:-50176}"

MODEL_ARGS="model_root=${G05_MODEL_ROOT}"
MODEL_ARGS="${MODEL_ARGS},variant=${G05_VARIANT:-base}"
MODEL_ARGS="${MODEL_ARGS},repo_path=${G05_REPO}"
MODEL_ARGS="${MODEL_ARGS},max_new_tokens=${G05_MAX_NEW_TOKENS:-512},temperature=${G05_TEMPERATURE:-0.0}"
MODEL_ARGS="${MODEL_ARGS},top_p=${G05_TOP_P:-1.0},top_k=${G05_TOP_K:-1}"

RESULT_NAME="${G05_RESULT_NAME:-g05-vqa}"
mkdir -p "${OUT_DIR}/${RESULT_NAME}"
LOG_DIR="${OUT_DIR}/logs/$(date +%Y%m%d_%H%M%S)_g05"
mkdir -p "${LOG_DIR}"
FAILURE_DIR="$(mktemp -d "${LOG_DIR}/failures.XXXXXX")"

LIMIT_ARGS=()
[ -n "${LIMIT}" ] && LIMIT_ARGS=(--limit "${LIMIT}")
GEN_ARGS=()
[ -n "${GEN_KWARGS}" ] && GEN_ARGS=(--gen_kwargs "${GEN_KWARGS}")

cat <<SUMMARY
Runner: runners/mimo_g05.sh
Python: ${PYTHON}
Backend: g05_vqa
Output: ${OUT_DIR}/${RESULT_NAME}
GPUs: ${GPUS[*]}
Tasks: ${TASKS[*]}
Limit: ${LIMIT:-none}
Generation kwargs: ${GEN_KWARGS:-task defaults}
Model args: ${MODEL_ARGS}
Logs: ${LOG_DIR}
SUMMARY

run_worker() {
    local worker_idx="$1"
    local gpu="$2"
    local task_idx task task_log task_output task_status worker_status failure_file
    worker_status=0
    failure_file="${FAILURE_DIR}/gpu_${gpu}.txt"
    for task_idx in "${!TASKS[@]}"; do
        if [ $((task_idx % ${#GPUS[@]})) -ne "${worker_idx}" ]; then
            continue
        fi
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
            --main_process_port "$((29700 + gpu))" \
            -m lmms_eval \
            --model g05_vqa \
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
            worker_status=${task_status}
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
    if ! wait "${pid}"; then
        status=1
    fi
done

trap - INT TERM
if [ "${status}" -ne 0 ]; then
    echo "Some tasks failed. Logs: ${LOG_DIR}" >&2
    find "${FAILURE_DIR}" -type f -maxdepth 1 -print -exec cat {} \; >&2
    exit "${status}"
fi
echo "All tasks finished. Logs: ${LOG_DIR}"
