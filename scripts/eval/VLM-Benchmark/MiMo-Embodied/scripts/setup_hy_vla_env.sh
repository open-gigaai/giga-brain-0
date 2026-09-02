#!/usr/bin/env bash
# 用途：创建 Hy-Embodied-0.5-VLA 专用共享 Python 环境。
# 关键约束：环境固定在 shared disk，且 Python 本体也由 uv 安装到 shared disk；
#          不使用 hostname 后缀，不使用 system-site-packages，避免只能当前容器可用。
# 默认位置：model-repos/Hy-Embodied-0.5-VLA-main/.venv-shared
# 默认安装模式：runtime，只安装推理/action smoke 必需依赖；如需官方全量训练/可视化依赖，
#              使用 HY_VLA_INSTALL_MODE=full bash scripts/setup_hy_vla_env.sh。
# 使用示例：
#   bash scripts/setup_hy_vla_env.sh
#   bash scripts/setup_hy_vla_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIMO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCHMARK_ROOT="$(cd "${MIMO_ROOT}/.." && pwd)"
HY_VLA_REPO="${HY_VLA_REPO:-${BENCHMARK_ROOT}/model-repos/Hy-Embodied-0.5-VLA-main}"
VENV_DIR="${HY_VLA_VENV_DIR:-${HY_VLA_REPO}/.venv-shared}"
PYTHON_VERSION="${HY_VLA_PYTHON_VERSION:-3.11}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${BENCHMARK_ROOT}/model-repos/.cache/uv}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${BENCHMARK_ROOT}/model-repos/.cache/uv-python-shared}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
HY_VLA_INSTALL_MODE="${HY_VLA_INSTALL_MODE:-runtime}"
HY_VLA_FROZEN="${HY_VLA_FROZEN:-1}"
HY_VLA_INSTALL_BENCH_DEPS="${HY_VLA_INSTALL_BENCH_DEPS:-1}"
HY_VLA_TORCH_VERSION="${HY_VLA_TORCH_VERSION:-2.7.1}"
HY_VLA_TORCHVISION_VERSION="${HY_VLA_TORCHVISION_VERSION:-0.22.1}"
HY_VLA_TRANSFORMERS_VERSION="${HY_VLA_TRANSFORMERS_VERSION:-4.57.0}"
HY_VLA_FLASH_ATTN_VERSION="${HY_VLA_FLASH_ATTN_VERSION:-2.7.4.post1}"
HY_VLA_FLASH_ATTN_FROM_SITE="${HY_VLA_FLASH_ATTN_FROM_SITE:-}"
HY_VLA_DOWNLOAD_DIR="${HY_VLA_DOWNLOAD_DIR:-${BENCHMARK_ROOT}/model-repos/.cache/hy-vla-downloads}"
HY_VLA_FLASH_ATTN_BUILD_FROM_SOURCE="${HY_VLA_FLASH_ATTN_BUILD_FROM_SOURCE:-0}"

if [ ! -d "${HY_VLA_REPO}" ]; then
    echo "ERROR: cannot find HY-VLA repo: ${HY_VLA_REPO}" >&2
    exit 2
fi

if [ ! -f "${HY_VLA_REPO}/pyproject.toml" ]; then
    echo "ERROR: cannot find pyproject.toml under HY-VLA repo: ${HY_VLA_REPO}" >&2
    exit 2
fi

export UV_CACHE_DIR
export UV_PYTHON_INSTALL_DIR
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

case "${HY_VLA_INSTALL_MODE}" in
    runtime|full) ;;
    *) echo "ERROR: HY_VLA_INSTALL_MODE must be runtime or full, got: ${HY_VLA_INSTALL_MODE}" >&2; exit 2 ;;
esac

install_flash_attn_from_source() {
    echo "Building flash-attn from the PyPI mirror with ${HY_VLA_FLASH_ATTN_MAX_JOBS:-8} jobs."
    FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="${HY_VLA_FLASH_ATTN_MAX_JOBS:-8}" \
        uv pip install --python "${PYTHON}" \
            --default-index "${PYPI_INDEX}" \
            --no-build-isolation \
            "flash-attn==${HY_VLA_FLASH_ATTN_VERSION}"
}

if [ "${HY_VLA_INSTALL_MODE}" = "full" ]; then
    SYNC_ARGS=(
        --project "${HY_VLA_REPO}"
        --python "${PYTHON_VERSION}"
        --managed-python
        --default-index "${PYPI_INDEX}"
        --link-mode copy
        --no-dev
    )

    if [ "${HY_VLA_FROZEN}" = "1" ]; then
        SYNC_ARGS+=(--frozen)
    fi

    uv sync "${SYNC_ARGS[@]}"
else
    if [ ! -x "${VENV_DIR}/bin/python" ]; then
        uv venv "${VENV_DIR}" \
            --python "${PYTHON_VERSION}" \
            --managed-python \
            --seed \
            --link-mode copy
    fi
fi

PYTHON="${VENV_DIR}/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: uv sync finished but Python is missing: ${PYTHON}" >&2
    exit 2
fi

if [ "${HY_VLA_INSTALL_MODE}" = "runtime" ]; then
    CONSTRAINTS_FILE="${VENV_DIR}/hy_vla_runtime_constraints.txt"
cat >"${CONSTRAINTS_FILE}" <<CONSTRAINTS
torch==${HY_VLA_TORCH_VERSION}
torchvision==${HY_VLA_TORCHVISION_VERSION}
CONSTRAINTS

    uv pip install --python "${PYTHON}" \
        --index "${PYTORCH_INDEX}" \
        --default-index "${PYPI_INDEX}" \
        --index-strategy unsafe-best-match \
        "torch==${HY_VLA_TORCH_VERSION}" \
        "torchvision==${HY_VLA_TORCHVISION_VERSION}"

    uv pip install --python "${PYTHON}" --default-index "${PYPI_INDEX}" \
        --constraints "${CONSTRAINTS_FILE}" \
        "numpy>=1.24,<2.0" \
        "transformers==${HY_VLA_TRANSFORMERS_VERSION}" \
        "safetensors>=0.4" \
        "huggingface-hub>=0.23,<1.0" \
        "scipy>=1.11" \
        pillow

    uv pip install --python "${PYTHON}" --default-index "${PYPI_INDEX}" \
        --no-deps \
        "timm==1.0.21"

    PY_TAG="$("${PYTHON}" - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
)"
    case "${PY_TAG}" in
        cp310|cp311|cp312) ;;
        *) echo "ERROR: unsupported Python ABI for prebuilt flash-attn: ${PY_TAG}" >&2; exit 2 ;;
    esac
    if ! "${PYTHON}" - <<'PY' >/dev/null 2>&1; then
import flash_attn  # noqa: F401
PY
        if [ -z "${HY_VLA_FLASH_ATTN_FROM_SITE}" ]; then
            DEFAULT_FLASH_SITE="/opt/conda/lib/python3.11/site-packages"
            if [ -d "${DEFAULT_FLASH_SITE}/flash_attn" ]; then
                HY_VLA_FLASH_ATTN_FROM_SITE="${DEFAULT_FLASH_SITE}"
            fi
        fi

        if [ -n "${HY_VLA_FLASH_ATTN_FROM_SITE}" ] && [ -d "${HY_VLA_FLASH_ATTN_FROM_SITE}/flash_attn" ]; then
            TARGET_SITE="$("${PYTHON}" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
            cp -a "${HY_VLA_FLASH_ATTN_FROM_SITE}/flash_attn" "${TARGET_SITE}/"
            cp -a "${HY_VLA_FLASH_ATTN_FROM_SITE}"/flash_attn-*.dist-info "${TARGET_SITE}/" 2>/dev/null || true
            cp -a "${HY_VLA_FLASH_ATTN_FROM_SITE}"/flash_attn_2_cuda*.so "${TARGET_SITE}/" 2>/dev/null || true
        fi
    fi

    if ! "${PYTHON}" - <<'PY' >/dev/null 2>&1; then
import flash_attn  # noqa: F401
PY
        FLASH_ATTN_WHEEL="flash_attn-${HY_VLA_FLASH_ATTN_VERSION}+cu12torch2.7cxx11abiTRUE-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
        FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${HY_VLA_FLASH_ATTN_VERSION}/${FLASH_ATTN_WHEEL}"
        FLASH_ATTN_PATH="${HY_VLA_DOWNLOAD_DIR}/${FLASH_ATTN_WHEEL}"
        mkdir -p "${HY_VLA_DOWNLOAD_DIR}"

        if [ "${HY_VLA_FLASH_ATTN_BUILD_FROM_SOURCE}" = "1" ]; then
            install_flash_attn_from_source
        elif [ ! -f "${FLASH_ATTN_PATH}" ]; then
            command -v curl >/dev/null 2>&1 || {
                echo "ERROR: curl is required to download flash-attn" >&2
                exit 2
            }
            echo "Downloading flash-attn with retries and resume support: ${FLASH_ATTN_PATH}"
            if curl --fail --location --http1.1 \
                --retry 20 --retry-all-errors --retry-delay 5 \
                --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
                --continue-at - --output "${FLASH_ATTN_PATH}.part" \
                "${FLASH_ATTN_URL}"; then
                mv "${FLASH_ATTN_PATH}.part" "${FLASH_ATTN_PATH}"
            else
                echo "WARNING: GitHub wheel download failed; building flash-attn from the PyPI mirror." >&2
                install_flash_attn_from_source
            fi
        fi

        if [ -f "${FLASH_ATTN_PATH}" ]; then
            uv pip install --python "${PYTHON}" "${FLASH_ATTN_PATH}"
        fi
    fi
    uv pip install --python "${PYTHON}" --default-index "${PYPI_INDEX}" --no-deps -e "${HY_VLA_REPO}"
fi

if [ "${HY_VLA_INSTALL_BENCH_DEPS}" = "1" ]; then
    uv pip install --python "${PYTHON}" --default-index "${PYPI_INDEX}" \
        diskcache evaluate sqlitedict pytablewriter sacrebleu decord hf_transfer \
        loguru "tenacity==8.3.0"
fi

PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" -B - <<'PY'
import flash_attn  # noqa: F401
import hy_vla  # noqa: F401
import safetensors  # noqa: F401
import torch
import transformers

print("HY-VLA env import check OK")
print("python torch", torch.__version__)
print("transformers", transformers.__version__)
PY

cat <<SUMMARY
HY-VLA shared env ready.
Python: ${PYTHON}
Repo: ${HY_VLA_REPO}
Install mode: ${HY_VLA_INSTALL_MODE}

Use:
  HY_VLA_PYTHON_BIN=${PYTHON} bash runners/mimo_hy_vla.sh --gpus 0 --tasks cvbench_boxed

Notes:
  - This env is shared-path based: no hostname suffix and no system-site-packages.
  - All machines must still be Linux x86_64 with compatible NVIDIA driver/CUDA runtime for the installed torch/flash-attn wheels.
SUMMARY
