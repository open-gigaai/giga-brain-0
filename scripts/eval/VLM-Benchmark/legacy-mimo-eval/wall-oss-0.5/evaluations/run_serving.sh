#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_serving.sh --checkpoint-path model-repos/wall-oss-0.5 [options]

Required:
  --checkpoint-path PATH         Checkpoint directory or checkpoint file.

Common options:
  --train-config-path PATH       Training config used by the checkpoint.
  --port PORT                    WebSocket port. Default: 32195.
  --host HOST                    Bind host. Default: 0.0.0.0.
  --env X2ROBOT|LIBERO           Serving environment. Default: X2ROBOT.
  --cuda-id ID                   Sets CUDA_VISIBLE_DEVICES. Default: 0.
  --image-passing-mode MODE      base64 or numpy. Default: base64.
  --action-horizon N             Model action horizon. Default: 32.
  --robot-type TYPE              desktop, turtle, or ex001. Default: desktop.
  --raw-actions                  Alias for --no-serialize-actions.
  --serialize-actions            Return robot-serialized actions.
  --no-serialize-actions         Return raw model action chunks. Default.
  --max-batch-size N             Enable dynamic batching.
  --enable-cuda-graph            Enable CUDA graph in the serving runtime.
  --enable-experimental-engine   Enable the experimental inference engine.
  --debug                        Enable debug logging.
  --dry-run                      Print the command without running it.

Additional arguments after "--" are forwarded to launch_serving.py, for example:
  bash scripts/run_serving.sh --checkpoint-path /ckpt -- \
    --model-config.norm-key libero_all

Environment variables can also be used, e.g. CHECKPOINT_PATH,
TRAIN_CONFIG_PATH, PORT, CUDA_ID, ACTION_HORIZON, WALLX_ENV.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -d "${SOURCE_ROOT}/wall_x" ]]; then
    export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-}"
PORT="${PORT:-32195}"
HOST="${HOST:-0.0.0.0}"
WALLX_ENV="${WALLX_ENV:-X2ROBOT}"
CUDA_ID="${CUDA_ID:-0}"
IMAGE_PASSING_MODE="${IMAGE_PASSING_MODE:-base64}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
ROBOT_TYPE="${ROBOT_TYPE:-desktop}"
ROBOT_ACTION_INTERPOLATE_MULTIPLIER="${ROBOT_ACTION_INTERPOLATE_MULTIPLIER:-1}"
ROBOT_ACTION_END_RATIO="${ROBOT_ACTION_END_RATIO:-1.0}"
MODEL_DEVICE="${MODEL_DEVICE:-cuda}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-}"
SERIALIZE_ACTIONS="${SERIALIZE_ACTIONS:-0}"
ENABLE_CUDA_GRAPH="${ENABLE_CUDA_GRAPH:-0}"
ENABLE_EXPERIMENTAL_ENGINE="${ENABLE_EXPERIMENTAL_ENGINE:-0}"
DEBUG="${DEBUG:-0}"
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --checkpoint-path)
            CHECKPOINT_PATH="${2:?missing value for --checkpoint-path}"
            shift 2
            ;;
        --checkpoint-path=*)
            CHECKPOINT_PATH="${1#*=}"
            shift
            ;;
        --train-config-path)
            TRAIN_CONFIG_PATH="${2:?missing value for --train-config-path}"
            shift 2
            ;;
        --train-config-path=*)
            TRAIN_CONFIG_PATH="${1#*=}"
            shift
            ;;
        --port)
            PORT="${2:?missing value for --port}"
            shift 2
            ;;
        --port=*)
            PORT="${1#*=}"
            shift
            ;;
        --host)
            HOST="${2:?missing value for --host}"
            shift 2
            ;;
        --host=*)
            HOST="${1#*=}"
            shift
            ;;
        --env)
            WALLX_ENV="${2:?missing value for --env}"
            shift 2
            ;;
        --env=*)
            WALLX_ENV="${1#*=}"
            shift
            ;;
        --cuda-id)
            CUDA_ID="${2:?missing value for --cuda-id}"
            shift 2
            ;;
        --cuda-id=*)
            CUDA_ID="${1#*=}"
            shift
            ;;
        --image-passing-mode)
            IMAGE_PASSING_MODE="${2:?missing value for --image-passing-mode}"
            shift 2
            ;;
        --image-passing-mode=*)
            IMAGE_PASSING_MODE="${1#*=}"
            shift
            ;;
        --action-horizon)
            ACTION_HORIZON="${2:?missing value for --action-horizon}"
            shift 2
            ;;
        --action-horizon=*)
            ACTION_HORIZON="${1#*=}"
            shift
            ;;
        --robot-type)
            ROBOT_TYPE="${2:?missing value for --robot-type}"
            shift 2
            ;;
        --robot-type=*)
            ROBOT_TYPE="${1#*=}"
            shift
            ;;
        --robot-action-interpolate-multiplier)
            ROBOT_ACTION_INTERPOLATE_MULTIPLIER="${2:?missing value for --robot-action-interpolate-multiplier}"
            shift 2
            ;;
        --robot-action-interpolate-multiplier=*)
            ROBOT_ACTION_INTERPOLATE_MULTIPLIER="${1#*=}"
            shift
            ;;
        --robot-action-end-ratio)
            ROBOT_ACTION_END_RATIO="${2:?missing value for --robot-action-end-ratio}"
            shift 2
            ;;
        --robot-action-end-ratio=*)
            ROBOT_ACTION_END_RATIO="${1#*=}"
            shift
            ;;
        --model-device)
            MODEL_DEVICE="${2:?missing value for --model-device}"
            shift 2
            ;;
        --model-device=*)
            MODEL_DEVICE="${1#*=}"
            shift
            ;;
        --max-batch-size)
            MAX_BATCH_SIZE="${2:?missing value for --max-batch-size}"
            shift 2
            ;;
        --max-batch-size=*)
            MAX_BATCH_SIZE="${1#*=}"
            shift
            ;;
        --default-prompt)
            DEFAULT_PROMPT="${2:?missing value for --default-prompt}"
            shift 2
            ;;
        --default-prompt=*)
            DEFAULT_PROMPT="${1#*=}"
            shift
            ;;
        --serialize-actions)
            SERIALIZE_ACTIONS=1
            shift
            ;;
        --no-serialize-actions|--raw-actions)
            SERIALIZE_ACTIONS=0
            shift
            ;;
        --enable-cuda-graph)
            ENABLE_CUDA_GRAPH=1
            shift
            ;;
        --enable-experimental-engine)
            ENABLE_EXPERIMENTAL_ENGINE=1
            shift
            ;;
        --debug)
            DEBUG=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${CHECKPOINT_PATH}" ]]; then
    echo "error: --checkpoint-path is required." >&2
    echo >&2
    usage >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_ID}"
export ENABLE_FAST_PREPROCESS="${ENABLE_FAST_PREPROCESS:-true}"

CMD=(
    "${PYTHON_BIN}" -m wall_x._vendor.harrix.serving.launch_serving
    --env "${WALLX_ENV}"
    --host "${HOST}"
    --port "${PORT}"
    --image-passing-mode "${IMAGE_PASSING_MODE}"
)

if [[ "${SERIALIZE_ACTIONS}" == "1" ]]; then
    CMD+=(--serialize-actions)
else
    CMD+=(--no-serialize-actions)
fi
if [[ -n "${MAX_BATCH_SIZE}" ]]; then
    CMD+=(--max-batch-size "${MAX_BATCH_SIZE}")
fi
if [[ -n "${DEFAULT_PROMPT}" ]]; then
    CMD+=(--default-prompt "${DEFAULT_PROMPT}")
fi
if [[ "${ENABLE_CUDA_GRAPH}" == "1" ]]; then
    CMD+=(--enable-cuda-graph)
fi
if [[ "${ENABLE_EXPERIMENTAL_ENGINE}" == "1" ]]; then
    CMD+=(--enable-experimental-engine)
fi
if [[ "${DEBUG}" == "1" ]]; then
    CMD+=(--debug)
fi

CMD+=(
    model-config:server-model-config
    --model-config.checkpoint-path "${CHECKPOINT_PATH}"
    --model-config.action-horizon "${ACTION_HORIZON}"
    --model-config.robot-type "${ROBOT_TYPE}"
    --model-config.robot-action-interpolate-multiplier "${ROBOT_ACTION_INTERPOLATE_MULTIPLIER}"
    --model-config.robot-action-end-ratio "${ROBOT_ACTION_END_RATIO}"
    --model-config.model-device "${MODEL_DEVICE}"
)

if [[ -n "${TRAIN_CONFIG_PATH}" ]]; then
    CMD+=(--model-config.train-config-path "${TRAIN_CONFIG_PATH}")
fi

CMD+=("${EXTRA_ARGS[@]}")

printf 'Launching Wall-X serving:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

exec "${CMD[@]}"
