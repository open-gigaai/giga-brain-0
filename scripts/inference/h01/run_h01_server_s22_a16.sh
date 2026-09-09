#!/usr/bin/env bash
set -euo pipefail

# H01 push-buttons model: observed state=22, predicted/published action=16.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ROBOT_TYPE=h01_robot
export EMBODIMENT_ID=7
export ORIGINAL_ACTION_DIM=16
export EXPECTED_STATE_DIM=22
export IS_ROBOT_MOVING=0
export IS_BODY_MOVING=0
export MODEL_PATH=/path/to/your/model/gb1_pg2_push_buttons_h01_260815_30k_release/model_ema
export PRETRAINED_PATH=/path/to/your/model/models--google--paligemma2-3b-pt-224
export FAST_TOKENIZER_PATH=/path/to/your/model/models--physical-intelligence--fast
export NORM_STATS_PATH=/path/to/your/norm/norm_stat_giga_robot_1_0_agilex.json
exec "${SCRIPT_DIR}/run_h01_server_common.sh"
