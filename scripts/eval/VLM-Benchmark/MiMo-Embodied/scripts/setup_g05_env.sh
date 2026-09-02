#!/usr/bin/env bash
# 用途：在当前机器上创建 OpenGalaxea G05 专用 Python 环境。
# 说明：venv 名称包含 hostname，避免 shared disk 上的 .venv 软链指向其他机器的 /root。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIMO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCHMARK_ROOT="$(cd "${MIMO_ROOT}/.." && pwd)"
G05_REPO="${G05_REPO:-${BENCHMARK_ROOT}/model-repos/GalaxeaVLA-main}"
HOST_TAG="$(hostname -s)"
VENV_DIR="${G05_VENV_DIR:-${G05_REPO}/.venv-${HOST_TAG}}"
PYTHON_VERSION="${G05_PYTHON_VERSION:-3.10.16}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${BENCHMARK_ROOT}/model-repos/.cache/uv}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${BENCHMARK_ROOT}/model-repos/.cache/uv-python-${HOST_TAG}}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

export UV_CACHE_DIR
export UV_PYTHON_INSTALL_DIR
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    uv venv "${VENV_DIR}" \
        --python "${PYTHON_VERSION}" \
        --seed \
        --link-mode copy
fi

PYTHON="${VENV_DIR}/bin/python"

uv pip install --python "${PYTHON}" --index-url "${PYPI_INDEX}" \
    -e "${G05_REPO}" \
    hf_transfer diskcache evaluate sqlitedict pytablewriter sacrebleu "tenacity==8.3.0" decord

cat <<SUMMARY
G05 env ready.
Python: ${PYTHON}
Use:
  G05_PYTHON_BIN=${PYTHON} bash runners/mimo_g05.sh --gpus 0 --tasks cvbench_boxed
SUMMARY
