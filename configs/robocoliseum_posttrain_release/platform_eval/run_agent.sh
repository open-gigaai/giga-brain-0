#!/usr/bin/env bash
set -euo pipefail

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
}

for name in \
  GIGA_MODELS_ROOT \
  ROBOCOLISEUM_MODEL_PATH \
  ROBOCOLISEUM_NORM_STATS_PATH \
  PALIGEMMA2_TOKENIZER_PATH \
  FAST_TOKENIZER_PATH \
  CHALLENGE_TOKEN \
  JOB_UUID \
  TUNNEL_ENDPOINT; do
  require_env "$name"
done

for path_name in \
  GIGA_MODELS_ROOT \
  ROBOCOLISEUM_MODEL_PATH \
  ROBOCOLISEUM_NORM_STATS_PATH \
  PALIGEMMA2_TOKENIZER_PATH \
  FAST_TOKENIZER_PATH; do
  if [[ ! -e "${!path_name}" ]]; then
    echo "Path does not exist (${path_name}): ${!path_name}" >&2
    exit 2
  fi
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${script_dir}:${GIGA_MODELS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python_bin="${PYTHON_BIN:-python}"
device="${DEVICE:-cuda}"

exec "$python_bin" "${script_dir}/robocoliseum_challenge_agent.py" \
  --job-uuid "$JOB_UUID" \
  --gateway-url "$TUNNEL_ENDPOINT" \
  --model-path "$ROBOCOLISEUM_MODEL_PATH" \
  --norm-stats-path "$ROBOCOLISEUM_NORM_STATS_PATH" \
