
# MiMo-Embodied 测评

本目录提供 Gigabrain0.7、OpenGalaxea G05 和 Hy-VLA 在 MiMo-Embodied 上的公开测评入口。


## 目录结构

本文命令均从 `scripts/eval/VLM-Benchmark` 目录运行。从仓库根目录进入：

```bash
cd scripts/eval/VLM-Benchmark
```

以下路径均相对于该目录：

```text
VLM-Benchmark/
├── model-repos/
│   ├── gigabrain/                     # GigaModels 代码
│   ├── Gigabrain0.7/                  # 模型权重
│   ├── GalaxeaVLA-main/               # G05 模型源码
│   ├── g05/G05-local/                 # G05 权重与 processor
│   ├── Hy-Embodied-0.5-VLA-main/      # Hy-VLA 模型源码
│   ├── hy-vla/Hy-Embodied-0.5-VLA-RoboTwin/
│   └── tokenizers/
│       ├── paligemma2-3b-pt-224/
│       └── fast/
└── MiMo-Embodied/
    ├── runners/
    │   ├── run_eval.sh
    │   ├── mimo_g05.sh
    │   ├── mimo_hy_vla.sh
    │   └── model_configs/gigabrain0.7.sh
    ├── scripts/
    │   ├── setup_g05_env.sh
    │   └── setup_hy_vla_env.sh
    ├── lmms_eval/
    ├── tools/
    ├── datasets/
    └── eval_results/
```

## 环境配置

推荐使用 Python 3.10：

```bash
python3.10 -m venv MiMo-Embodied/.venv
source MiMo-Embodied/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e MiMo-Embodied
```

将包含 `giga_models/` Python 包的 GigaModels 源码放到 `model-repos/gigabrain/`。该路径不是本仓库的 Git submodule，需要单独准备。

多卡测评时，每张 GPU 独立加载一份模型，并行处理不同数据集。模型首次加载较慢属于正常现象。

G05 和 Hy-VLA 使用各自的模型环境。将模型源码放入 `model-repos/` 后执行：

```bash
bash MiMo-Embodied/scripts/setup_g05_env.sh
bash MiMo-Embodied/scripts/setup_hy_vla_env.sh
```

也可以分别通过 `G05_PYTHON_BIN` 和 `HY_VLA_PYTHON_BIN` 指定已经安装好依赖的 Python。
G05 的 Triton 算子会在首次推理时编译扩展，因此 Python 环境必须包含对应版本的 `Python.h`，不要混用不同 Python 版本的 site-packages。

## 模型准备

将 Gigabrain0.7 权重放在：

```text
model-repos/Gigabrain0.7/
├── config.json
├── inference_config.json
└── diffusion_pytorch_model.bin
```

准备 tokenizer：

```text
model-repos/tokenizers/paligemma2-3b-pt-224/
model-repos/tokenizers/fast/
```

权重和 tokenizer 不提交到 Git。

G05 资源目录：

```text
model-repos/GalaxeaVLA-main/
model-repos/g05/G05-local/
├── g05-base/
├── g05-droid/
├── g05-libero/
├── g05-robotwin20/
├── g05-so101/
├── qwen3_5_2b_base_processor/
└── action_tokenizer.pt
```

Hy-VLA 资源目录：

```text
model-repos/Hy-Embodied-0.5-VLA-main/
model-repos/hy-vla/Hy-Embodied-0.5-VLA-RoboTwin/
├── config.json
├── model.safetensors
├── tokenizer.json
└── preprocessor_config.json
```

建议单张 GPU 至少准备 40 GB 显存。脚本固定 `batch_size=1`；多卡时每张卡各加载一份模型，然后从同一个任务队列领取完整数据集。

## 数据准备

默认数据根目录为：

```text
MiMo-Embodied/datasets/public_datasets/VLM/
```

数据位于其他目录时，运行时使用 `--data-root` 指定。

## 运行测评

检查 17 个任务及其生成配置，不加载模型：

```bash
bash MiMo-Embodied/runners/run_eval.sh \
  --model gigabrain0.7 \
  --gpus 0,1,2,3 \
  --dry-run
```

完整测评：

```bash
bash MiMo-Embodied/runners/run_eval.sh \
  --model gigabrain0.7 \
  --gpus 0,1,2,3
```

使用自定义资源路径：

```bash
bash MiMo-Embodied/runners/run_eval.sh \
  --model gigabrain0.7 \
  --gpus 0,1,2,3 \
  --model-path model-repos/Gigabrain0.7 \
  --giga-models model-repos/gigabrain \
  --tokenizer model-repos/tokenizers/paligemma2-3b-pt-224 \
  --fast-tokenizer model-repos/tokenizers/fast \
  --data-root MiMo-Embodied/datasets/public_datasets/VLM \
  --output-root MiMo-Embodied/eval_results/gigabrain0.7/manual_run
```

单任务 smoke test：

```bash
bash MiMo-Embodied/runners/run_eval.sh \
  --model gigabrain0.7 \
  --gpus 0 \
  --tasks roboafford \
  --limit 10 \
  --output-root MiMo-Embodied/eval_results/gigabrain0.7/roboafford_smoke
```

检查 G05 和 Hy-VLA 配置，不加载模型：

```bash
bash MiMo-Embodied/runners/mimo_g05.sh --gpus 0,1 --dry-run
bash MiMo-Embodied/runners/mimo_hy_vla.sh --gpus 0,1 --dry-run
```

运行完整测评：

```bash
bash MiMo-Embodied/runners/mimo_g05.sh --gpus 0,1,2,3
bash MiMo-Embodied/runners/mimo_hy_vla.sh --gpus 0,1,2,3
```

G05 默认使用 `base` 权重，可通过环境变量选择其他权重：

```bash
G05_VARIANT=robotwin20 \
bash MiMo-Embodied/runners/mimo_g05.sh --gpus 0,1,2,3
```

两者均支持 `--tasks`、`--limit`、`--data-root` 和 `--out-dir`。单任务检查示例：

```bash
bash MiMo-Embodied/runners/mimo_g05.sh \
  --gpus 0 \
  --tasks roboafford \
  --limit 2

bash MiMo-Embodied/runners/mimo_hy_vla.sh \
  --gpus 0 \
  --tasks roboafford \
  --limit 2
```

Gigabrain0.7 的输出目录必须不存在，脚本不会覆盖已有结果。G05 和 Hy-VLA 建议每次通过 `--out-dir` 使用新的输出目录。

## 结果文件

每个底层任务单独保存结果、样本和日志：

```text
MiMo-Embodied/eval_results/gigabrain0.7/<run_time>/
├── summary.json
├── summary.md
├── run_plan.json
├── roboafford/
│   ├── results.json
│   ├── samples.jsonl
│   ├── audit.json
│   ├── run_config.json
│   └── run.log
├── roborefit/
│   └── ...
└── ...
```

`results.json` 和 `samples.jsonl` 是 lmms-eval 对该数据集生成的原始正式结果，只统一了文件名，没有修改其中的模型答案。`summary.json` 和 `summary.md` 根据本次实际结果现场生成，不与预设精度比较。

G05 和 Hy-VLA 保留 lmms-eval 原始文件名，默认分别写入：

```text
MiMo-Embodied/eval_results/g05-vqa/<task>/
MiMo-Embodied/eval_results/hy-vla-robotwin-vqa/<task>/
```

每个任务均保存对应的 results 和 samples 文件，运行日志位于 `MiMo-Embodied/eval_results/logs/`。

查看汇总：

```bash
python -m json.tool MiMo-Embodied/eval_results/gigabrain0.7/<run_time>/summary.json
```

## 指标

| 指标 | 底层任务 |
|---|---|
| EmbSpatial | `embspatialbench_robust` |
| ERQA | `erqa_boxed` |
| CVBench | `cvbench_boxed_robust` |
| SAT | `sat_robust` |
| MetaVQA | `metavqa_eval_robust` |
| CRPE | `crpe_relation_robust` |
| RoboSpatial | Compatibility、Configuration、Context 的算术平均 |
| RefSpatial | Location、Placement、Unseen 的算术平均 |
| RoboAfford | `roboafford` |
| VABench | `vabench_point_box` |
| Where2Place | `where2place_point` |
| PartAffordance | `part_affordance` |
| RoboRefIt | `roborefit` |

13 项指标等权计算宏平均。

## 验证

本次发布不包含 `MiMo-Embodied/tests` 下的独立测试文件。使用上文的 `--dry-run` 命令验证已提交的 runner 配置；dry-run 不会加载模型或检查数据集内容。
