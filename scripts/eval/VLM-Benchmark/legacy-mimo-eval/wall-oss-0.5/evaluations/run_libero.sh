#!/usr/bin/env bash
set -euo pipefail

# LIBERO evaluation launcher. Configure with environment variables:
#
#   CHECKPOINT_PATH=model-repos/wall-oss-0.5 bash scripts/run_libero.sh
#   bash scripts/run_libero.sh model-repos/wall-oss-0.5
#   CHECKPOINT_PATH=model-repos/wall-oss-0.5 SMOKE=1 bash scripts/run_libero.sh
#   CONFIG=configs/libero-eval.yml bash scripts/run_libero.sh
#
# Optional knobs:
#   LIBERO_PATH=third_party/LIBERO     # optional when not cloned to third_party/LIBERO
#   CUDA_ID=0
#   DRIVER_MODE=in_process
#   NUM_WORKERS=1
#   MAX_BATCH_SIZE=1
#   ALL_SUITES=1                    # run all standard LIBERO suites (40 tasks)
#   TASK_SUITES="libero_spatial ..." # custom suite list (space- or comma-separated)
#   TASK_SUITE_NAME=libero_spatial  # single suite when ALL_SUITES=0 and TASK_SUITES unset
#   NUM_TRIALS_PER_TASK=50
#   TASK_INDICES=0,1,2              # omit to run every task in the suite
#   MAX_INFER_TIMES=52
#   NORM_KEY=libero_all
#   ROLLOUT_BASE=outputs/libero   # per-suite logs under ${ROLLOUT_BASE}/${suite}/
#   LOG_DIR=outputs/libero/logs       # overrides ROLLOUT_BASE when set (single suite)
#   SKIP_LIBERO_DEP_CHECK=1         # bypass dependency preflight

# export ALL_SUITES=1
# export TASK_SUITE_NAME=libero_10
# export CUDA_ID=0
# export NUM_WORKERS=10
# export NUM_TRIALS_PER_TASK=20
# export CHECKPOINT_PATH=model-repos/wall-oss-0.5
# export ROLLOUT_BASE=outputs/libero

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_libero.sh [CHECKPOINT_PATH]

Environment:
  CHECKPOINT_PATH=model-repos/wall-oss-0.5
  CONFIG=configs/libero-eval.yml
  LIBERO_PATH=third_party/LIBERO
  CUDA_ID=0
  SMOKE=1
EOF
}

if [[ -d "${SOURCE_ROOT}/third_party/LIBERO" ]]; then
    export PYTHONPATH="${SOURCE_ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
fi
if [[ -n "${LIBERO_PATH:-}" ]]; then
    export PYTHONPATH="${LIBERO_PATH}:${PYTHONPATH:-}"
fi

if [[ -f "${SCRIPT_DIR}/infer_libero.py" ]]; then
    INFER_LIBERO_CMD=("${SCRIPT_DIR}/infer_libero.py")
elif [[ -f "${SOURCE_ROOT}/scripts/infer_libero.py" ]]; then
    INFER_LIBERO_CMD=(python "${SOURCE_ROOT}/scripts/infer_libero.py")
elif command -v infer_libero.py >/dev/null 2>&1; then
    INFER_LIBERO_CMD=(infer_libero.py)
else
    echo "infer_libero.py is not available. Install Wall-X first." >&2
    exit 2
fi

DEFAULT_ALL_SUITES=(
    libero_spatial
    libero_object
    libero_goal
    libero_10
)

case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
esac

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then
    CHECKPOINT_PATH="$1"
fi

CUDA_IDS="${CUDA_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_IDS}"

# MuJoCo offscreen rendering via NVIDIA EGL (required on headless GPU nodes).
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# LIBERO init-state files are trusted simulator assets. PyTorch 2.6 changed
# torch.load() defaults in a way that breaks LIBERO's upstream loader.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
EGL_VENDOR_DIR="${HOME}/.config/glvnd/egl_vendor.d"
EGL_VENDOR_FILE="${EGL_VENDOR_DIR}/10_nvidia.json"
if [[ ! -f "${EGL_VENDOR_FILE}" ]]; then
    mkdir -p "${EGL_VENDOR_DIR}"
    cat > "${EGL_VENDOR_FILE}" <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
fi
export __EGL_VENDOR_LIBRARY_FILENAMES="${EGL_VENDOR_FILE}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

# robosuite validates this value against CUDA_VISIBLE_DEVICES on common
# headless EGL stacks, so default to the first requested physical GPU id.
FIRST_CUDA_ID="${CUDA_IDS%%,*}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${FIRST_CUDA_ID:-0}}"
if [[ -d "${SOURCE_ROOT}/third_party/harrix/python" ]]; then
    export PYTHONPATH="${SOURCE_ROOT}/third_party/harrix/python:${PYTHONPATH:-}"
fi
if [[ -d "${SOURCE_ROOT}/wall_x" ]]; then
    export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"
fi

check_libero_dependencies() {
    local missing
    if missing="$(
        python - <<'PY'
import importlib.util

checks = [
    ("libero.libero", "LIBERO"),
    ("robosuite", "robosuite"),
    ("mujoco", "mujoco"),
    ("OpenGL", "PyOpenGL"),
    ("bddl", "bddl"),
    ("gym", "gym"),
    ("h5py", "h5py"),
]

missing = [label for module, label in checks if importlib.util.find_spec(module) is None]
if missing:
    print(", ".join(missing))
    raise SystemExit(1)
PY
    )"; then
        return 0
    fi

    cat >&2 <<EOF
LIBERO simulator dependencies are missing: ${missing}

Install the optional LIBERO simulator stack from the Wall-X repository root:

  pip install -r requirements-libero.txt
  mkdir -p third_party
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO

If LIBERO is cloned elsewhere, pass it with:

  LIBERO_PATH=third_party/LIBERO bash scripts/run_libero.sh ...

Set SKIP_LIBERO_DEP_CHECK=1 only if you intentionally manage these dependencies elsewhere.
EOF
    exit 2
}

if [[ "${SKIP_LIBERO_DEP_CHECK:-0}" != "1" ]]; then
    check_libero_dependencies
fi

resolve_task_suites() {
    local suites=()
    if [[ "${ALL_SUITES:-0}" == "1" ]]; then
        if [[ -n "${TASK_SUITES:-}" ]]; then
            TASK_SUITES="${TASK_SUITES//,/ }"
            read -r -a suites <<< "${TASK_SUITES}"
        else
            suites=("${DEFAULT_ALL_SUITES[@]}")
        fi
    elif [[ -n "${TASK_SUITES:-}" ]]; then
        TASK_SUITES="${TASK_SUITES//,/ }"
        read -r -a suites <<< "${TASK_SUITES}"
    else
        suites=("${TASK_SUITE_NAME:-libero_spatial}")
    fi
    echo "${suites[@]}"
}

resolve_log_dirs() {
    local suite="$1"
    local multi_suite="$2"
    local log_dir rollout_dir

    if [[ -n "${LOG_DIR:-}" ]]; then
        if [[ "${multi_suite}" == "1" ]]; then
            log_dir="${LOG_DIR}/${suite}"
        else
            log_dir="${LOG_DIR}"
        fi
    elif [[ -n "${ROLLOUT_BASE:-}" ]]; then
        if [[ "${multi_suite}" == "1" ]]; then
            log_dir="${ROLLOUT_BASE}/${suite}/json"
        else
            log_dir="${ROLLOUT_BASE}/json"
        fi
    else
        log_dir="outputs/libero/logs"
    fi

    if [[ -n "${WALLX_ROLLOUT_DIR:-}" ]]; then
        if [[ "${multi_suite}" == "1" ]]; then
            rollout_dir="${WALLX_ROLLOUT_DIR}/${suite}"
        else
            rollout_dir="${WALLX_ROLLOUT_DIR}"
        fi
    elif [[ -n "${ROLLOUT_BASE:-}" ]]; then
        if [[ "${multi_suite}" == "1" ]]; then
            rollout_dir="${ROLLOUT_BASE}/${suite}/videos"
        else
            rollout_dir="${ROLLOUT_BASE}/videos"
        fi
    else
        rollout_dir=""
    fi

    mkdir -p "${log_dir}"
    if [[ -n "${rollout_dir}" ]]; then
        mkdir -p "${rollout_dir}"
    fi
    LOG_DIR_RESOLVED="${log_dir}"
    WALLX_ROLLOUT_DIR_RESOLVED="${rollout_dir}"
}

run_suite() {
    local suite="$1"
    local multi_suite="$2"

    resolve_log_dirs "${suite}" "${multi_suite}"
    if [[ -n "${WALLX_ROLLOUT_DIR_RESOLVED}" ]]; then
        export WALLX_ROLLOUT_DIR="${WALLX_ROLLOUT_DIR_RESOLVED}"
    else
        unset WALLX_ROLLOUT_DIR
    fi

    local args=()
    if [[ -n "${CONFIG:-}" ]]; then
        args+=(--config "${CONFIG}")
    else
        if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
            echo "CHECKPOINT_PATH is required when CONFIG is not set." >&2
            exit 2
        fi
        args+=(--checkpoint-path "${CHECKPOINT_PATH}")
    fi

    if [[ -n "${TRAIN_CONFIG_PATH:-}" ]]; then
        args+=(--train-config-path "${TRAIN_CONFIG_PATH}")
    fi
    if [[ "${SMOKE:-0}" == "1" ]]; then
        args+=(--smoke)
    fi
    if [[ "${DET:-0}" == "1" ]]; then
        args+=(--deterministic-model)
    fi
    if [[ -n "${TASK_INDICES:-}" ]]; then
        args+=(--task-indices "${TASK_INDICES}")
    fi
    if [[ -n "${MAX_INFER_TIMES:-}" ]]; then
        args+=(--max-infer-times "${MAX_INFER_TIMES}")
    fi

    args+=(--driver-mode "${DRIVER_MODE:-in_process}")
    args+=(--num-workers "${NUM_WORKERS:-1}")
    args+=(--max-batch-size "${MAX_BATCH_SIZE:-1}")
    args+=(--task-suite-name "${suite}")
    args+=(--num-trials-per-task "${NUM_TRIALS_PER_TASK:-50}")
    args+=(--norm-key "${NORM_KEY:-libero_all}")
    args+=(--log-dir "${LOG_DIR_RESOLVED}")

    echo "=== LIBERO eval: suite=${suite} log_dir=${LOG_DIR_RESOLVED} ==="
    "${INFER_LIBERO_CMD[@]}" "${args[@]}"
}

suites=($(resolve_task_suites))
multi_suite=0
if [[ "${#suites[@]}" -gt 1 ]]; then
    multi_suite=1
fi

for suite in "${suites[@]}"; do
    run_suite "${suite}" "${multi_suite}"
done
