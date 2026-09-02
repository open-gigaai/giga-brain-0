<div align="center">
  <h1>VLM Benchmark</h1>
  <p>面向 Gigabrain0.7 与五个具身 VLM 基线的 MiMo-Embodied 可复现测评代码。</p>

  <p><a href="README.md">English</a> | <b>中文</b></p>

  <a href="../../../LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Benchmark-MiMo--Embodied-2E8B57" alt="MiMo-Embodied">
</div>

## ✨ 概览

本目录提供 Gigabrain0.7 及以下对比模型的测评代码：

| 模型 | MiMo-Embodied | 入口 |
| --- | :---: | --- |
| Gigabrain0.7 | 支持 | `MiMo-Embodied/runners/run_eval.sh` |
| Xiaomi-Robotics-0 | 支持 | `legacy-mimo-eval/xiaomi-robotics-0/eval_vlm/` |
| Spirit-v1.5 | 支持 | `legacy-mimo-eval/spirit-v1.5/scripts/` |
| Wall-OSS-0.5 | 支持 | `legacy-mimo-eval/wall-oss-0.5/evaluations/` |
| G0.5-Base | 支持 | `MiMo-Embodied/runners/mimo_g05.sh` |
| Hy-Embodied-0.5-VLA-UMI | 支持 | `MiMo-Embodied/runners/mimo_hy_vla.sh` |

`MiMo-Embodied/` 提供公共任务定义以及 Gigabrain0.7、G0.5 和 Hy-VLA 适配器；`legacy-mimo-eval/` 保留 Xiaomi、Spirit 和 Wall 的模型专用程序。

## 📁 目录结构

```text
VLM-Benchmark/
├── README.md
├── README_zh.md
├── model-repos/                  # 本地模型源码、权重和 tokenizer
├── datasets/                     # legacy 模型的默认数据目录
├── MiMo-Embodied/                # MiMo 任务、适配器和 runner
└── legacy-mimo-eval/             # Xiaomi、Spirit 和 Wall 的 MiMo 程序
```

模型权重、数据集、评测结果和虚拟环境都是本地资源，不应提交到仓库。

## ⚡ 环境安装

除非另有说明，所有命令均从 `scripts/eval/VLM-Benchmark` 运行。从仓库根目录进入：

```bash
cd scripts/eval/VLM-Benchmark
```

不同模型的 PyTorch 和 Transformers 依赖可能冲突，建议按模型系列使用独立环境。

安装公共测评包：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e MiMo-Embodied
```

将模型源码放入 `model-repos/` 后，分别准备 G0.5 和 Hy-VLA 环境：

```bash
bash MiMo-Embodied/scripts/setup_g05_env.sh
bash MiMo-Embodied/scripts/setup_hy_vla_env.sh
```

Spirit 依赖单独维护：

```bash
python -m pip install -r legacy-mimo-eval/spirit-v1.5/requirements.txt
```

Xiaomi 和 Wall 的依赖请按各自上游模型仓库安装。Wall 还要求所选环境能够导入其模型源码包。

## 💾 资源准备

runner 默认使用仓库相对路径。可以按下面的目录放置资源，也可以通过命令行参数或环境变量覆盖。

```text
model-repos/
├── Gigabrain0.7/                         # Gigabrain0.7 checkpoint
├── gigabrain/                            # giga_models Python 包
├── tokenizers/
│   ├── paligemma2-3b-pt-224/
│   └── fast/
├── GalaxeaVLA-main/                      # G0.5 模型源码
├── g05/G05-local/g05-base/               # G0.5-Base 权重和 processor
├── Hy-Embodied-0.5-VLA-main/             # Hy-VLA 模型源码
├── hy-vla/Hy-Embodied-0.5-VLA-UMI/       # Hy-VLA UMI checkpoint
├── xiaomi-robotics-0/                    # Xiaomi checkpoint
├── spirit-v1.5/                          # Spirit checkpoint
├── qwen3-vl-4b-instruct/                 # Spirit 可选 backbone
└── wall-oss-0.5/                         # Wall checkpoint
```

Gigabrain0.7 目录至少包含：

```text
model-repos/Gigabrain0.7/
├── config.json
├── inference_config.json
└── diffusion_pytorch_model.bin
```

默认数据目录：

```text
MiMo-Embodied/datasets/public_datasets/VLM/
datasets/public_datasets/VLM/
```

legacy 脚本使用 `datasets/public_datasets/VLM/`，MiMo runner 使用 `MiMo-Embodied/datasets/public_datasets/VLM/`。采用其他目录布局时，通过 `--data-root`、`DATASET_PATH` 或 `MIMO_DATA_ROOT` 覆盖。

## 🚀 快速开始

### 1. 检查配置

dry-run 只解析 profile、资源、GPU 和任务，不加载模型权重：

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

### 2. 运行 MiMo-Embodied

Gigabrain0.7 smoke test：

```bash
bash MiMo-Embodied/runners/run_eval.sh \
  --model gigabrain0.7 \
  --gpus 0 \
  --tasks roboafford \
  --limit 10 \
  --output-root MiMo-Embodied/eval_results/gigabrain0.7/roboafford_smoke
```

运行 G0.5-Base 和 Hy-VLA UMI：

```bash
G05_VARIANT=base \
bash MiMo-Embodied/runners/mimo_g05.sh --gpus 0

HY_VLA_VQA_RESULT_NAME=hy-vla-umi-vqa \
bash MiMo-Embodied/runners/mimo_hy_vla.sh \
  --checkpoint model-repos/hy-vla/Hy-Embodied-0.5-VLA-UMI \
  --gpus 0
```

Xiaomi、Spirit 和 Wall 使用各自的 MiMo 测评程序：

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

## 📊 输出

不同入口分别写入各自已忽略的结果目录：

```text
MiMo-Embodied/eval_results/<model-or-run>/
legacy-mimo-eval/xiaomi-robotics-0/eval_vlm/results/
legacy-mimo-eval/spirit-v1.5/outputs/
legacy-mimo-eval/wall-oss-0.5/outputs/
```

Gigabrain0.7 的 MiMo 结果包含 `summary.json`、`summary.md`、`run_plan.json`、各任务结果、样本、审计记录和日志。legacy 结果格式由各模型脚本决定。

## ✅ 验证

本次发布不包含独立的 `MiMo-Embodied/tests` 测试集。完整测评前，请使用[快速开始](#-快速开始)中的 `--dry-run` 命令检查已提交的 runner 配置。

dry-run 成功只代表配置展开正常，不代表 checkpoint、数据内容、CUDA 算子或完整推理已经验证。

## 📚 详细文档

- [MiMo-Embodied](MiMo-Embodied/README.md)
- [Xiaomi、Spirit 和 Wall MiMo 测评](legacy-mimo-eval/README.md)
