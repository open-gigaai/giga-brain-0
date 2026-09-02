# VLM Evaluation

This folder evaluates the VLM ability of Xiaomi-Robotics-0-Pretrain on RealWorldQA.

RealWorldQA in this workspace is stored as Hugging Face parquet files:

```text
datasets/public_datasets/VLM/vqa/benchmarks/RealWorldQA
└── data
    ├── test-00000-of-00002.parquet
    └── test-00001-of-00002.parquet
```

Run:

```bash
python eval_vlm/main.py \
  --model-path XiaomiRobotics/Xiaomi-Robotics-0-Pretrain \
  --data-root datasets/public_datasets/VLM/vqa/benchmarks/RealWorldQA \
  --batch-size 1 \
  --attn-implementation flash_attention_2 \
  --output eval_vlm/results/realworldqa_predictions.jsonl
```

If `flash-attn` is not available, use:

```bash
python eval_vlm/main.py \
  --model-path XiaomiRobotics/Xiaomi-Robotics-0-Pretrain \
  --attn-implementation sdpa
```

The script writes per-sample predictions to JSONL and summary metrics to
`eval_vlm/results/realworldqa_predictions.metrics.json`.

Scoring uses choice-letter exact match when the ground truth is an option such as `A`, `B`, `C`, or `D`; otherwise it uses normalized exact match for short answers such as `Yes`, `No`, numbers, and colors.

## ERQA

ERQA is stored at:

```text
datasets/public_datasets/VLM/vqa/benchmarks/ERQA
└── data
    └── test-00000-of-00001.parquet
```

Run:

```bash
python eval_vlm/erqa.py \
  --model-path XiaomiRobotics/Xiaomi-Robotics-0-Pretrain \
  --data-root datasets/public_datasets/VLM/vqa/benchmarks/ERQA \
  --batch-size 1 \
  --attn-implementation flash_attention_2 \
  --output eval_vlm/results/erqa_predictions.jsonl
```

If `flash-attn` is not available:

```bash
python eval_vlm/erqa.py --attn-implementation sdpa
```

ERQA has 400 A/B/C/D multiple-choice examples. Some examples contain multiple images, so the script inserts one image token per image and also reports accuracy by `question_type`.

## EO-Bench

EO-Bench is stored at:

```text
datasets/public_datasets/VLM/vqa/benchmarks/EO-Bench
├── data
│   └── test-00000-of-00001.parquet
└── images
```

Run:

```bash
python eval_vlm/eobench.py \
  --model-path XiaomiRobotics/Xiaomi-Robotics-0-Pretrain \
  --data-root datasets/public_datasets/VLM/vqa/benchmarks/EO-Bench \
  --batch-size 1 \
  --attn-implementation flash_attention_2 \
  --output eval_vlm/results/eobench_predictions.jsonl
```

If `flash-attn` is not available:

```bash
python eval_vlm/eobench.py --attn-implementation sdpa
```

EO-Bench has 600 embodied-reasoning examples. It includes both single-choice and multiple-choice answers, including options beyond `D`, so scoring compares the predicted option-letter set with the ground-truth option-letter set exactly. The script also reports accuracy by `question_type`.

## Additional VQA / Spatial Benchmarks

All scripts share the same model arguments:

```bash
--model-path XiaomiRobotics/Xiaomi-Robotics-0-Pretrain
--batch-size 1
--attn-implementation flash_attention_2
```

Use `--attn-implementation sdpa` if Flash Attention is unavailable, and `--limit 10` for smoke tests.

Multiple-choice / text-answer benchmarks:

```bash
python eval_vlm/cvbench.py
python eval_vlm/embspatial.py
python eval_vlm/sat.py
python eval_vlm/metavqa.py
python eval_vlm/crpe.py
```

Spatial point / mask benchmarks:

```bash
python eval_vlm/where2place.py
python eval_vlm/part_affordance.py
python eval_vlm/roboafford.py
python eval_vlm/refspatial.py
python eval_vlm/robospatial_home.py
```

BBox / point-in-box benchmarks:

```bash
python eval_vlm/roborefit.py
python eval_vlm/vabench_point_bbox.py
```

Default dataset roots:

| Script | Dataset |
| --- | --- |
| `where2place.py` | `datasets/public_datasets/VLM/vqa/benchmarks/Where2Place` |
| `roborefit.py` | `datasets/public_datasets/VLM/vqa/roborefit` |
| `vabench_point_bbox.py` | `datasets/public_datasets/VLM/vqa/vabench-point-bbox` |
| `part_affordance.py` | `datasets/public_datasets/VLM/vqa/Part-Affordance-2K` |
| `roboafford.py` | `datasets/public_datasets/VLM/vqa/roboafford-eval` |
| `cvbench.py` | `datasets/public_datasets/VLM/vqa/CV-Bench` |
| `embspatial.py` | `datasets/public_datasets/VLM/vqa/EmbSpatial-Bench` |
| `sat.py` | `datasets/public_datasets/VLM/vqa/benchmarks/SAT` |
| `robospatial_home.py` | `datasets/public_datasets/VLM/vqa/RoboSpatial-Home` |
| `refspatial.py` | `datasets/public_datasets/VLM/vqa/RefSpatial-Bench` |
| `crpe.py` | `datasets/public_datasets/VLM/vqa/CRPE` |
| `metavqa.py` | `datasets/public_datasets/VLM/vqa/MetaVQA-Eval` |

Notes:

- Mask/point tasks report `success_rate`, meaning at least one predicted point lands inside the ground-truth mask, plus `mean_point_hit_rate`.
- BBox tasks report `mean_iou` and accuracy at `--iou-threshold 0.5`; they also report `point_hit_rate` when the model outputs a point instead of a box.
- `crpe.py` defaults to skipping missing images. This local CRPE copy includes `abnormal_images`, while many official examples reference paths such as `coco/val2017/*.jpg`; pass `--image-root datasets/public_datasets/MSCOCO` to evaluate those too.
