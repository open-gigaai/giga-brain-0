<div align="center">
  <h1>VLM Benchmark</h1>
  <p>Reproducible MiMo-Embodied evaluation for Gigabrain0.7 and five embodied VLM baselines.</p>

  <p><b>English</b> | <a href="README_zh.md">中文</a></p>

  <a href="../../../LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Benchmark-MiMo--Embodied-2E8B57" alt="MiMo-Embodied">
</div>

## ✨ Overview

This directory provides evaluation code for Gigabrain0.7 and the following comparison models:

| Model | MiMo-Embodied | Entry point |
| --- | :---: | --- |
| Gigabrain0.7 | Yes | `MiMo-Embodied/runners/run_eval.sh` |
| Xiaomi-Robotics-0 | Yes | `legacy-mimo-eval/xiaomi-robotics-0/eval_vlm/` |
| Spirit-v1.5 | Yes | `legacy-mimo-eval/spirit-v1.5/scripts/` |
| Wall-OSS-0.5 | Yes | `legacy-mimo-eval/wall-oss-0.5/evaluations/` |
| G0.5-Base | Yes | `MiMo-Embodied/runners/mimo_g05.sh` |
| Hy-Embodied-0.5-VLA-UMI | Yes | `MiMo-Embodied/runners/mimo_hy_vla.sh` |

`MiMo-Embodied/` contains the shared task definitions and the Gigabrain0.7, G0.5, and Hy-VLA adapters. `legacy-mimo-eval/` keeps the model-specific Xiaomi, Spirit, and Wall evaluation programs.

## 📁 Layout

```text
VLM-Benchmark/
├── README.md
├── README_zh.md
├── model-repos/                  # Local source trees, checkpoints, and tokenizers
├── datasets/                     # Legacy model dataset defaults
├── MiMo-Embodied/                # MiMo tasks, adapters, and runners
└── legacy-mimo-eval/             # Xiaomi, Spirit, and Wall MiMo programs
```

Model weights, datasets, generated results, and virtual environments are local resources and must not be committed.

## ⚡ Installation

Run all commands from `scripts/eval/VLM-Benchmark` unless a section says otherwise. From the repository root:

```bash
cd scripts/eval/VLM-Benchmark
```

Use a separate environment for each model family because their PyTorch and Transformers requirements may conflict.

Install the common benchmark packages:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e MiMo-Embodied
```

Prepare the dedicated G0.5 and Hy-VLA environments after placing their source repositories under `model-repos/`:

```bash
bash MiMo-Embodied/scripts/setup_g05_env.sh
bash MiMo-Embodied/scripts/setup_hy_vla_env.sh
```

Spirit dependencies are listed separately:

```bash
python -m pip install -r legacy-mimo-eval/spirit-v1.5/requirements.txt
```

Install Xiaomi and Wall dependencies from their upstream model repositories. Wall also requires its source package to be importable in the selected environment.

## 💾 Resources

The runners use repository-relative paths. Place resources in the locations below or override the corresponding command-line argument or environment variable.

```text
model-repos/
├── Gigabrain0.7/                         # Gigabrain0.7 checkpoint
├── gigabrain/                            # giga_models Python package
├── tokenizers/
│   ├── paligemma2-3b-pt-224/
│   └── fast/
├── GalaxeaVLA-main/                      # G0.5 source repository
├── g05/G05-local/g05-base/               # G0.5-Base weights and processor
├── Hy-Embodied-0.5-VLA-main/             # Hy-VLA source repository
├── hy-vla/Hy-Embodied-0.5-VLA-UMI/       # Hy-VLA UMI checkpoint
├── xiaomi-robotics-0/                    # Xiaomi checkpoint
├── spirit-v1.5/                          # Spirit checkpoint
├── qwen3-vl-4b-instruct/                 # Optional Spirit backbone
└── wall-oss-0.5/                         # Wall checkpoint
```

Gigabrain0.7 expects at least:

```text
model-repos/Gigabrain0.7/
├── config.json
├── inference_config.json
└── diffusion_pytorch_model.bin
```

Default dataset roots:

```text
MiMo-Embodied/datasets/public_datasets/VLM/
datasets/public_datasets/VLM/
```

The legacy scripts use `datasets/public_datasets/VLM/`; MiMo runners use `MiMo-Embodied/datasets/public_datasets/VLM/`. Pass `--data-root`, `DATASET_PATH`, or `MIMO_DATA_ROOT` when using another layout.

## 🚀 Quick Start

### 1. Validate configurations

Dry runs resolve profiles, resources, GPUs, and tasks without loading model weights:

```bash
bash MiMo-Embodied/runners/run_eval.sh \
  --model gigabrain0.7 \
  --gpus 0 \
  --dry-run

G05_VARIANT=base \
bash MiMo-Embodied/runners/mimo_g05.sh \
  --gpus 0 \
  --dry-run

bash MiMo-Embodied/runners/mimo_hy_vla.sh \
  --checkpoint model-repos/hy-vla/Hy-Embodied-0.5-VLA-UMI \
  --gpus 0 \
  --dry-run
```

### 2. Run MiMo-Embodied

Gigabrain0.7 smoke test:

```bash
bash MiMo-Embodied/runners/run_eval.sh \
  --model gigabrain0.7 \
  --gpus 0 \
  --tasks roboafford \
  --limit 10 \
  --output-root MiMo-Embodied/eval_results/gigabrain0.7/roboafford_smoke
```

G0.5-Base and Hy-VLA UMI:

```bash
G05_VARIANT=base \
bash MiMo-Embodied/runners/mimo_g05.sh --gpus 0

HY_VLA_VQA_RESULT_NAME=hy-vla-umi-vqa \
bash MiMo-Embodied/runners/mimo_hy_vla.sh \
  --checkpoint model-repos/hy-vla/Hy-Embodied-0.5-VLA-UMI \
  --gpus 0
```

Xiaomi, Spirit, and Wall use their model-specific MiMo programs:

```bash
python legacy-mimo-eval/xiaomi-robotics-0/eval_vlm/main.py \
  --model-path model-repos/xiaomi-robotics-0 \
  --data-root datasets/public_datasets/VLM/vqa/benchmarks/RealWorldQA \
  --output legacy-mimo-eval/xiaomi-robotics-0/eval_vlm/results/realworldqa.jsonl \
  --limit 10

CKPT_PATH=model-repos/spirit-v1.5 \
DATASET_PATH=datasets/public_datasets/VLM/vqa/RoboSpatial-Home \
bash legacy-mimo-eval/spirit-v1.5/scripts/run_robospatial_home_vlm.sh \
  --max-samples 10

python legacy-mimo-eval/wall-oss-0.5/evaluations/eval_robospatial_home_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/RoboSpatial-Home \
  --output-jsonl legacy-mimo-eval/wall-oss-0.5/outputs/robospatial_home.jsonl \
  --limit 10
```

## 📊 Outputs

Each runner writes into its own ignored output directory:

```text
MiMo-Embodied/eval_results/<model-or-run>/
legacy-mimo-eval/xiaomi-robotics-0/eval_vlm/results/
legacy-mimo-eval/spirit-v1.5/outputs/
legacy-mimo-eval/wall-oss-0.5/outputs/
```

Gigabrain0.7 MiMo runs include `summary.json`, `summary.md`, `run_plan.json`, per-task results, samples, audit records, and logs. Legacy artifacts depend on the model-specific evaluator.

## ✅ Validation

The release does not include a standalone `MiMo-Embodied/tests` test suite. Use the `--dry-run` commands in [Quick Start](#-quick-start) to validate the tracked runner configurations before a full evaluation.

A successful dry run verifies configuration expansion only; it does not validate checkpoint compatibility, dataset contents, CUDA kernels, or full inference.

## 📚 Documentation

- [MiMo-Embodied](MiMo-Embodied/README.md)
- [Xiaomi, Spirit, and Wall MiMo evaluation](legacy-mimo-eval/README.md)
