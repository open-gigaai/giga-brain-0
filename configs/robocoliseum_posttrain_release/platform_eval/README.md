# Platform evaluation connector

Before evaluation, read the platform-provided skills and official documentation
first. In particular, start with `challenge-help`, then consult
`challenge-login`, `challenge-submit-job`, `challenge-run-agent`,
`challenge-poll-result`, and `challenge-inference-protocol` as needed. Platform
rules, endpoints, protocol details, and quotas can change; the platform guidance
is authoritative.

This directory contains only the runtime connector for an already submitted
instruction, robust, or spatial evaluation job. It does not submit a job. Submission
consumes platform quota and should be performed separately with explicit
confirmation through the platform workflow.

## Files

- `robocoliseum_challenge_agent.py`: reverse WebSocket tunnel client.
- `robocoliseum_challenge_adapter.py`: observation and action protocol mapping.
- `robocoliseum_challenge_policy.py`: GigaBrain-0.7 inference wrapper.
- `run_agent.sh`: environment-based launcher without stored credentials or
  model paths.

## Launch

Use the same `giga_torch:v1.5` environment as training. Supply all runtime paths
and platform values from the current cluster session:

```bash
export GIGA_MODELS_ROOT=
export ROBOCOLISEUM_MODEL_PATH=
export ROBOCOLISEUM_NORM_STATS_PATH=
export PALIGEMMA2_TOKENIZER_PATH=
export FAST_TOKENIZER_PATH=

export CHALLENGE_TOKEN=
export JOB_UUID=
export TUNNEL_ENDPOINT=

# Optional. Defaults to python and cuda.
export PYTHON_BIN=
export DEVICE=

./run_agent.sh
```

`ROBOCOLISEUM_MODEL_PATH` must point to the selected checkpoint's complete
`model_ema` directory. `ROBOCOLISEUM_NORM_STATS_PATH` must match the submitted
board: use instruction/robust statistics for the instruction or robust board,
and spatial statistics for the spatial board.

When multiple GPUs are visible, the policy distributes decoder layers across
them. Restrict the process explicitly when needed:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_agent.sh
```

The connector maps the gateway's 21-D state to the trained 17-D layout and
returns 50-step dual-arm action chunks using native Python lists. It intentionally
does not include platform submission, polling, ranking, or login implementations;
use the current platform skills and official documentation for those operations.
