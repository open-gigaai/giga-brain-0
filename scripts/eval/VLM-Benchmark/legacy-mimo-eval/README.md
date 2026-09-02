# Legacy MiMo-Embodied Evaluation Scripts

This directory contains the MiMo-Embodied evaluation scripts for Xiaomi-Robotics-0, Spirit-v1.5, and Wall-OSS-0.5. Run commands from the `VLM-Benchmark` directory so the repository-relative defaults resolve correctly.

## Local Resources

The repository does not contain model weights or datasets. Put local resources in the following locations, or override the corresponding command-line argument or environment variable:

| Path | Fill with |
| --- | --- |
| `model-repos/xiaomi-robotics-0/` | Xiaomi-Robotics-0 checkpoint |
| `model-repos/spirit-v1.5/` | Spirit-v1.5 checkpoint |
| `model-repos/qwen3-vl-4b-instruct/` | Qwen3-VL backbone and processor used by Spirit |
| `model-repos/wall-oss-0.5/` | Wall-OSS-0.5 checkpoint |
| `datasets/public_datasets/VLM/` | MiMo-Embodied benchmark datasets |
| `datasets/public_datasets/MSCOCO/` | COCO images required by CRPE |

Do not commit machine-specific paths, checkpoints, datasets, or generated outputs.

## Entrypoints

- Xiaomi: scripts under `xiaomi-robotics-0/eval_vlm/`; pass `--model-path` and `--data-root` when overriding defaults.
- Spirit: launchers under `spirit-v1.5/scripts/`; override `CKPT_PATH`, `DATASET_PATH`, `BACKBONE_PATH`, or `PROCESSOR_PATH` as needed.
- Wall: launchers under `wall-oss-0.5/scripts/`; pass checkpoint and dataset arguments documented in `wall-oss-0.5/evaluations/README.md`.

Generated results belong under `eval_results/` or `outputs/`, both of which must remain untracked.

## LLaVA JSONL Inputs

Most legacy scripts read Hugging Face parquet datasets. A few scripts read, or optionally support, LLaVA-style JSONL files as evaluation input.

Input JSONL files are newline-delimited JSON. Each line is one sample. Blank lines are ignored.

### Common Layout

The common LLaVA format used by the legacy scripts is:

```json
{
  "id": "sample-id",
  "image": "relative/or/absolute/image.png",
  "conversations": [
    {"from": "human", "value": "<image>\nQuestion text ..."},
    {"from": "gpt", "value": "A"}
  ]
}
```

Notes:

- `image` may be a string or, for some Xiaomi scripts, a list of image paths.
- Relative image paths are resolved against `--image-root` when that argument exists; otherwise they are resolved relative to the JSONL file or dataset root.
- For Xiaomi CRPE, `--image-root` is treated as the COCO root/prefix. For a JSONL image such as `coco/val2017/000000000139.jpg`, the script tries the CRPE dataset root, `CRPE/abnormal_images/`, `--image-root/coco/val2017/...`, `--image-root/val2017/...`, and `dirname(--image-root)/coco/val2017/...`.
- The first `human` message is treated as the prompt/question.
- The first `gpt` message is treated as the reference answer.
- For multiple-choice tasks, the answer is usually a letter such as `A`, `B`, `C`, or `D`.
- For bbox tasks, the answer should contain four normalized coordinates, for example `[0.12, 0.20, 0.56, 0.78]`.

### Tasks Using JSONL As Input

CRPE uses official JSONL files:

```text
CRPE/
├── crpe_exist.jsonl
└── crpe_relation.jsonl
```

These are read by the AceBrain, RoboBrain, Spirit, Xiaomi, and Wall-OSS CRPE scripts.

MetaVQA-Eval uses:

```text
MetaVQA-Eval/
├── test.jsonl
├── test.json          # some scripts also use this metadata file
└── obs/
```

These rows generally contain fields such as `question_id`, `question`, `answer`, `obs`, `options`, and `type`. Some scripts accept `--data-file` or `--annotation-file` to point to another JSONL file.

Spirit RoboRefIt supports a LLaVA JSONL fallback. It looks for:

```text
roborefit_test_llava.jsonl
roborefit_<split>_llava.jsonl
<split>_llava.jsonl
<split>.jsonl
```

Spirit VABench point/bbox supports a LLaVA JSONL fallback. It looks for:

```text
vabench_point_bbox_llava.jsonl
vabench_point_bbox_<split>_llava.jsonl
<split>_llava.jsonl
<split>.jsonl
```

Xiaomi also has a dedicated LLaVA-format ERQA script:

```text
benchmarks/ERQA/llava_json/erqa_test_llava.jsonl
```

This is used by `erqa_llava.py` through `--data-path`.

The previous RealWorldQA LLaVA script is archived under `tmp/realworldqa-eobench/` together with the other inactive RealWorldQA and EO-Bench scripts.

### Conversion Guidance

When converting parquet or other annotations to JSONL:

1. Write one sample per line as compact JSON.
2. Preserve stable IDs in `id` or `question_id`.
3. Store image paths relative to the JSONL location or a known `--image-root`.
4. Put the model-facing question in `conversations[0].value`; include `<image>` if the original script expects LLaVA-style prompts.
5. Put the scorer-facing answer in `conversations[1].value`.
6. For bbox tasks, store normalized `[x1, y1, x2, y2]` coordinates in the answer text.
7. Keep generated prediction files separate from input JSONL files; `predictions.jsonl` is output, not input.
