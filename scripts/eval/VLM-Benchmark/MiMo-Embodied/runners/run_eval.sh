#!/usr/bin/env bash

# Public MiMo-Embodied evaluation entry point.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIMO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -n "${PYTHON_BIN:-}" ]; then
    PYTHON="${PYTHON_BIN}"
elif [ -x "${MIMO_ROOT}/.venv/bin/python3" ]; then
    PYTHON="${MIMO_ROOT}/.venv/bin/python3"
else
    PYTHON=python3
fi

exec "${PYTHON}" "${MIMO_ROOT}/tools/run_eval.py" "$@"
