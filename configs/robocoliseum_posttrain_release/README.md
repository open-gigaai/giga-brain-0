# RoboColiseum post-training

Minimal GigaBrain-0.7 post-training bundle for RoboColiseum instruction,
spatial, and manipulation suites. Model, checkpoint, and tokenizer paths are
not stored in this directory; callers must provide them through environment
variables.

## Contents

- `configs/`, `robocoliseum_ext/`, `preprocess_*.py`, and `train_*.sh` contain
  the post-training implementation and launchers.
- `training_logs/` contains compact, de-duplicated loss histories for the
  completed instruction/robust and spatial runs.
- `platform_eval/` contains the GigaBrain-0.7 connector for an already submitted
  platform evaluation job.

## Requirements

Use `giga-brain-0` main with `giga-datasets` main and the `1.1.0` releases of
`giga-train` and `giga-models`.

Set all required paths before launching:

```bash
export GIGABRAIN_PROJECT_ROOT=
export GIGA_DATASETS_ROOT=
export GIGA_TRAIN_ROOT=
export GIGA_MODELS_ROOT=

export ROBOCOLISEUM_DATA_ROOT=
export ROBOCOLISEUM_PROCESSED_DATA_ROOT=
export ROBOCOLISEUM_NORM_STATS_ROOT=
export ROBOCOLISEUM_OUTPUT_ROOT=

export GIGABRAIN_PRETRAINED_CKPT=
export PALIGEMMA2_TOKENIZER_PATH=
export FAST_TOKENIZER_PATH=

# Optional; defaults to python from the active environment.
export PYTHON_BIN=
```

`GIGABRAIN_PRETRAINED_CKPT` must be a complete `model_ema` directory.
`ROBOCOLISEUM_DATA_ROOT` is the raw task-suite root, while
`ROBOCOLISEUM_PROCESSED_DATA_ROOT` is the offline 17-D task-suite root.
Missing paths fail before training starts.

## Launch

```bash
./train_instruction_robust.sh
./train_spatial.sh
./train_manip.sh
```

Manipulation defaults to the first instruction segment. To concatenate all
sub-instructions instead:

```bash
MANIP_TRAINING_VARIANT=concat_subinstructions ./train_manip.sh
```

## Platform evaluation

Before connecting to the evaluation platform, read the platform-provided skills
and official documentation first. Start with `challenge-help`; use the dedicated
login, submission, agent, polling, and inference-protocol skills for the current
workflow and rules. Job submission consumes platform quota and requires explicit
confirmation.

The local connector does not submit jobs and stores no credentials or model
paths. See `platform_eval/README.md` after obtaining a valid token and job UUID.
