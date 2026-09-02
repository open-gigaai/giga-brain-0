#!/usr/bin/env bash

set -Eeuo pipefail

# Resolve package files relative to this launcher; external resources are explicit.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_ROOT=${SCRIPT_DIR}
: "${GIGABRAIN_PROJECT_ROOT:?Set GIGABRAIN_PROJECT_ROOT to the GigaBrain-0 project directory}"
: "${GIGA_DATASETS_ROOT:?Set GIGA_DATASETS_ROOT to the giga-datasets main checkout}"
: "${GIGA_TRAIN_ROOT:?Set GIGA_TRAIN_ROOT to the giga-train 1.1.0 checkout}"
: "${GIGA_MODELS_ROOT:?Set GIGA_MODELS_ROOT to the giga-models 1.1.0 checkout}"
: "${GIGABRAIN_PRETRAINED_CKPT:?Set GIGABRAIN_PRETRAINED_CKPT to a model_ema directory}"
: "${ROBOCOLISEUM_PROCESSED_DATA_ROOT:?Set ROBOCOLISEUM_PROCESSED_DATA_ROOT to the processed task_suite directory}"
: "${ROBOCOLISEUM_NORM_STATS_ROOT:?Set ROBOCOLISEUM_NORM_STATS_ROOT to the norm-stats directory}"
: "${ROBOCOLISEUM_OUTPUT_ROOT:?Set ROBOCOLISEUM_OUTPUT_ROOT to a writable output directory}"
: "${PALIGEMMA2_TOKENIZER_PATH:?Set PALIGEMMA2_TOKENIZER_PATH to the tokenizer directory}"
: "${FAST_TOKENIZER_PATH:?Set FAST_TOKENIZER_PATH to the FAST tokenizer directory}"
TRAIN_SCRIPT=${GIGABRAIN_PROJECT_ROOT}/scripts/train.py
CONFIG_RELATIVE_PATH=configs/robocoliseum_manip.py
CONFIG_PATH=${EXPERIMENT_ROOT}/${CONFIG_RELATIVE_PATH}
NORM_STATS_PATH=${ROBOCOLISEUM_NORM_STATS_ROOT}/robocoliseum_manip_17d.json
MANIP_TRAINING_VARIANT=${MANIP_TRAINING_VARIANT:-first_step}
case "${MANIP_TRAINING_VARIANT}" in
    first_step|concat_subinstructions)
        ;;
    *)
        echo "Unsupported MANIP_TRAINING_VARIANT: ${MANIP_TRAINING_VARIANT}" >&2
        exit 1
        ;;
esac
MANIP_GPU_IDS=${MANIP_GPU_IDS:-0,1,2,3}
MODEL_DIR=${GIGABRAIN_PRETRAINED_CKPT}
MODEL_CONFIG=${MODEL_DIR}/config.json
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_DIR=${ROBOCOLISEUM_OUTPUT_ROOT}/runs/robocoliseum_manip_${MANIP_TRAINING_VARIANT}
TENSORBOARD_LOG_DIR=${RUN_DIR}/logs
CACHE_ROOT=${ROBOCOLISEUM_OUTPUT_ROOT}/.cache_train
LOG_ROOT=${ROBOCOLISEUM_OUTPUT_ROOT}/logs/train_manip

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

for required_file in \
    "${TRAIN_SCRIPT}" \
    "${GIGABRAIN_PROJECT_ROOT}/gigabrain07.py" \
    "${GIGA_DATASETS_ROOT}/giga_datasets/__init__.py" \
    "${GIGA_TRAIN_ROOT}/giga_train/__init__.py" \
    "${GIGA_MODELS_ROOT}/giga_models/__init__.py" \
    "${CONFIG_PATH}" \
    "${NORM_STATS_PATH}" \
    "${PALIGEMMA2_TOKENIZER_PATH}/tokenizer.json" \
    "${PALIGEMMA2_TOKENIZER_PATH}/tokenizer_config.json" \
    "${FAST_TOKENIZER_PATH}/tokenizer.json" \
    "${FAST_TOKENIZER_PATH}/tokenizer_config.json" \
    "${MODEL_CONFIG}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done
for required_dir in "${PALIGEMMA2_TOKENIZER_PATH}" "${FAST_TOKENIZER_PATH}"; do
    if [[ ! -d "${required_dir}" ]]; then
        echo "Required directory not found: ${required_dir}" >&2
        exit 1
    fi
done

"${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version

from accelerate.tracking import is_tensorboard_available

assert version("tensorboard") == "2.20.0", "Expected tensorboard 2.20.0"
assert version("protobuf") == "4.25.6", "Expected protobuf 4.25.6"
assert version("grpcio") == "1.70.0", "Expected grpcio 1.70.0"
assert is_tensorboard_available(), "Accelerate cannot detect TensorBoard"
PY

mkdir -p "${CACHE_ROOT}/huggingface/datasets" "${LOG_ROOT}" "${RUN_DIR}" "${TENSORBOARD_LOG_DIR}"

export PYTHONPATH="${GIGABRAIN_PROJECT_ROOT}:${EXPERIMENT_ROOT}:${GIGA_DATASETS_ROOT}:${GIGA_TRAIN_ROOT}:${GIGA_MODELS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=${CACHE_ROOT}/huggingface
export HF_DATASETS_CACHE=${CACHE_ROOT}/huggingface/datasets
export XDG_CACHE_HOME=${CACHE_ROOT}
# Keep the validated model path and the trainer checkpoint in sync.
export GIGABRAIN_PRETRAINED_CKPT=${MODEL_DIR}
export MANIP_TRAINING_VARIANT
export MANIP_GPU_IDS

GPU_COUNT=$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')
if (( GPU_COUNT < 4 )); then
    echo "Expected at least 4 visible GPUs, found ${GPU_COUNT}." >&2
    exit 1
fi

TIMESTAMP=$(date '+%Y%m%d_%H%M%S_%Z')
LOG_PATH=${LOG_ROOT}/train_${TIMESTAMP}.log
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "purpose=RoboColiseum manipulation unified-17D/effective-17D post-training"
echo "reference=GigaBrain-0.7 open-source training stack"
echo "training_variant=${MANIP_TRAINING_VARIANT}"
echo "gpu_ids=${MANIP_GPU_IDS}"
echo "config=${CONFIG_PATH}"
echo "norm_stats=${NORM_STATS_PATH}"
echo "model_config=${MODEL_CONFIG}"
echo "run_dir=${RUN_DIR}"
echo "tensorboard_log_dir=${TENSORBOARD_LOG_DIR}"
echo "tensorboard_command=tensorboard --logdir ${TENSORBOARD_LOG_DIR} --host 0.0.0.0 --port 6006"
echo "visible_gpus=${GPU_COUNT}"
echo "started_at=$(date --iso-8601=seconds)"

cd "${EXPERIMENT_ROOT}"
exec "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" --config "${CONFIG_RELATIVE_PATH}"
