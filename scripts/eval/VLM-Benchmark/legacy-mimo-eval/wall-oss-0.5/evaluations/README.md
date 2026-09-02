# Scripts

This directory contains the public Wall-X command-line helpers. Run the examples
below from the repository root, using `python scripts/...` and `bash scripts/...`.
Pass file and directory paths explicitly.

## Inference smoke test

Use `fake_inference.py` to verify that a checkpoint can be loaded and can
produce one action chunk from a synthetic LIBERO-style observation.

```bash
python scripts/fake_inference.py --checkpoint-path model-repos/wall-oss-0.5
```

If the training config is not stored next to the checkpoint as `config.yml` or
`config.yaml`, pass it explicitly:

```bash
python scripts/fake_inference.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --train-config-path model-repos/wall-oss-0.5/config.yml
```

## LIBERO evaluation

`run_libero.sh` is a small shell wrapper around `infer_libero.py`. It requires
the optional LIBERO simulator stack:

```bash
pip install -r requirements-libero.txt
mkdir -p third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
```

The launcher checks for LIBERO, robosuite, MuJoCo, PyOpenGL, BDDL, Gym, and
h5py before loading the model. If LIBERO is cloned elsewhere, pass
`LIBERO_PATH=third_party/LIBERO`.

```bash
bash scripts/run_libero.sh model-repos/wall-oss-0.5
```

Useful environment variables:

```bash
CHECKPOINT_PATH=model-repos/wall-oss-0.5
TRAIN_CONFIG_PATH=model-repos/wall-oss-0.5/config.yml
TASK_SUITE_NAME=libero_spatial
TASK_INDICES=0,1,2
NUM_TRIALS_PER_TASK=50
CUDA_ID=0
SMOKE=1
MAX_INFER_TIMES=52
```

`MAX_INFER_TIMES` is optional. When omitted, the launcher uses suite-specific
defaults aligned with the LIBERO evaluator: spatial 22, object 28, goal 30,
libero_10 52, and libero_90 40 action chunks.

For full control, call the Python entry directly:

```bash
python scripts/infer_libero.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --task-suite-name libero_spatial \
  --num-trials-per-task 50 \
  --driver-mode in_process
```

You can also pass a complete eval config:

```bash
python scripts/infer_libero.py --config configs/libero-eval.yml
```

## RealWorldQA VLM evaluation

Use `eval_realworldqa_vlm.py` to evaluate image question answering on the local
RealWorldQA parquet dataset. This evaluates VLM text generation, not robot
action prediction.

```bash
python scripts/eval_realworldqa_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/benchmarks/RealWorldQA \
  --output-jsonl ./realworldqa_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

The default path sends the dataset question as-is and calls
`model.generate_text` directly so the JSONL keeps the raw generated text. If you
need to compare prompt sensitivity, pass `--prompt-style cot-answer`,
`--prompt-style wallx`, or `--backend wrapper-vqa`, but those paths are mainly
diagnostic.

For a quick smoke test:

```bash
python scripts/eval_realworldqa_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --limit 5
```

## EO-Bench VLM evaluation

Use `eval_eobench_vlm.py` for EO-Bench. It reads image paths from parquet,
supports 1-4 images per sample, scores single-answer and multi-answer
multiple-choice outputs, and reports accuracy by `question_type`.

```bash
python scripts/eval_eobench_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/benchmarks/EO-Bench \
  --output-jsonl ./eobench_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

For a quick smoke test:

```bash
python scripts/eval_eobench_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --limit 5
```

## Where2Place VLM Evaluation

Use `eval_where2place_vlm.py` for Where2Place. The dataset provides an image,
a placement question, and a valid-region mask instead of a text answer. The
script asks the model for pixel-coordinate placement points and scores whether
the points fall inside the mask. The core metrics are Point Acc@1 / Acc@K:
Acc@1 checks whether the first predicted point is inside the mask, and Acc@K
checks whether any of the first K predicted points is inside the mask.

```bash
python scripts/eval_where2place_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/benchmarks/Where2Place \
  --output-jsonl ./where2place_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

For a quick smoke test:

```bash
python scripts/eval_where2place_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --limit 5
```

If the model tends to emit repeated or template-like points, try the stronger
diverse-point prompt:

```bash
python scripts/eval_where2place_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/benchmarks/Where2Place \
  --prompt-style diverse-point-tags \
  --output-jsonl ./where2place_diverse_point_tags_predictions.jsonl \
  --batch-size 1
```

## ERQA VLM Evaluation

Use `eval_erqa_vlm.py` for ERQA. It supports variable-image samples, scores
single-letter multiple-choice answers, and reports accuracy by `question_type`
and image count.

```bash
python scripts/eval_erqa_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/benchmarks/ERQA \
  --output-jsonl ./erqa_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

For a quick smoke test:

```bash
python scripts/eval_erqa_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --limit 5
```

## CV-Bench VLM Evaluation

Use `eval_cvbench_vlm.py` for CV-Bench. It evaluates single-image
multiple-choice 2D/3D vision questions and reports overall accuracy plus the
official CV-Bench combined accuracy:
`0.5 * (((ADE20K accuracy + COCO accuracy) / 2) + Omni3D accuracy)`.

```bash
python scripts/eval_cvbench_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/CV-Bench \
  --output-jsonl ./cvbench_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

For a quick smoke test:

```bash
python scripts/eval_cvbench_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --limit 5
```

## CRPE VLM Evaluation

Use `eval_crpe_vlm.py` for CRPE. CRPE is a single-choice benchmark for object
existence and relation comprehension. The official files are already expanded
for CircularEval, so the script reports both per-query `single.accuracy` and
grouped `circular.circular_accuracy`; the circular metric is the stricter main
score because all rotated versions of a question must be correct.

```bash
python scripts/eval_crpe_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/CRPE \
  --coco-root datasets/public_datasets/MSCOCO \
  --output-jsonl ./crpe_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

For the non-circular meta files:

```bash
python scripts/eval_crpe_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --files meta \
  --output-jsonl ./crpe_meta_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

## MetaVQA-Eval VLM Evaluation

Use `eval_metavqa_eval_vlm.py` for MetaVQA-Eval. It evaluates single-image
embodied/spatial VQA questions from `test.jsonl`. Most samples are
multiple-choice, and the main metric is option-letter accuracy. The small
open-ended subset with no options is skipped by default; pass `--include-open`
to include it with exact and relaxed text matching diagnostics.

```bash
python scripts/eval_metavqa_eval_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/MetaVQA-Eval \
  --output-jsonl ./metavqa_eval_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

Useful filters:

```bash
python scripts/eval_metavqa_eval_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --domain real \
  --type embodied_distance,embodied_collision \
  --output-jsonl ./metavqa_eval_real_subset_predictions.jsonl \
  --batch-size 1
```

## EmbSpatial-Bench VLM Evaluation

Use `eval_embspatial_bench_vlm.py` for EmbSpatial-Bench. It evaluates
single-image embodied spatial multiple-choice questions and reports overall
accuracy plus accuracy by spatial `relation` and `data_source`.

```bash
python scripts/eval_embspatial_bench_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/EmbSpatial-Bench \
  --output-jsonl ./embspatial_bench_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

For a quick smoke test:

```bash
python scripts/eval_embspatial_bench_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --limit 5
```

## SAT VLM Evaluation

Use `eval_sat_vlm.py` for SAT. It evaluates spatial aptitude multiple-choice
questions with one or two images and reports overall accuracy plus accuracy by
`question_type` and image count. The default split is the 150-sample real-image
`test` split.

```bash
python scripts/eval_sat_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/benchmarks/SAT \
  --output-jsonl ./sat_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

The SAT README recommends circular evaluation for the small test split:

```bash
python scripts/eval_sat_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/benchmarks/SAT \
  --circular-eval \
  --output-jsonl ./sat_circular_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

## RoboRefIt VLM Grounding Evaluation

Use `eval_roborefit_vlm.py` for RoboRefIt. The dataset provides one image, one
referring expression, and one target object bounding box per sample. The script
asks the model for a pixel-coordinate box and reports mean IoU plus Acc@IoU
thresholds, with Acc@0.5 as the main grounding metric.

```bash
python scripts/eval_roborefit_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/roborefit \
  --output-jsonl ./roborefit_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

For a quick smoke test:

```bash
python scripts/eval_roborefit_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --limit 5
```

If the model emits repeated tiny boxes or point-like boxes, try the stricter
box prompt:

```bash
python scripts/eval_roborefit_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/roborefit \
  --prompt-style strict-box-tags \
  --output-jsonl ./roborefit_strict_box_tags_predictions.jsonl \
  --batch-size 1
```

## VABench Point-BBox VLM Evaluation

Use `eval_vabench_point_bbox_vlm.py` for VABench point-bbox. Each sample
contains one manipulation prompt and a target-region bbox. The model is asked
to output 2D pixel points, and the main metrics are Point Acc@1 / Acc@K:
Acc@1 checks whether the first predicted point is inside the GT bbox, and
Acc@K checks whether any of the first K predicted points is inside the GT bbox.

```bash
python scripts/eval_vabench_point_bbox_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/vabench-point-bbox \
  --output-jsonl ./vabench_point_bbox_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

For a quick smoke test:

```bash
python scripts/eval_vabench_point_bbox_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --limit 5
```

## Part-Affordance-2K VLM Evaluation

Use `eval_part_affordance_2k_vlm.py` for Part-Affordance-2K. Each sample
contains one image, one grasp instruction, and a binary affordance mask. The
script asks the model for pixel-coordinate grasp points and scores whether the
points fall inside the affordance mask. The main metrics are Point Acc@1 /
Acc@K, reported overall and by `category_type`.

```bash
python scripts/eval_part_affordance_2k_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/Part-Affordance-2K \
  --output-jsonl ./part_affordance_2k_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

For a quick smoke test:

```bash
python scripts/eval_part_affordance_2k_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --limit 5
```

## RoboAfford-Eval VLM Evaluation

Use `eval_roboafford_eval_vlm.py` for RoboAfford-eval. Each sample contains
one image, one point-localization question, and a binary or gray affordance
mask. The dataset question asks for normalized coordinates in `[0, 1]`; the
script accepts normalized or pixel-coordinate outputs and scores Point Acc@1 /
Acc@K against the mask, reported overall and by `category`.

```bash
python scripts/eval_roboafford_eval_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/roboafford-eval \
  --output-jsonl ./roboafford_eval_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

For a stricter normalized-coordinate prompt:

```bash
python scripts/eval_roboafford_eval_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --prompt-style normalized-point-tags \
  --output-jsonl ./roboafford_eval_normalized_point_tags_predictions.jsonl \
  --batch-size 1
```

## RoboSpatial-Home VLM Evaluation

Use `eval_robospatial_home_vlm.py` for RoboSpatial-Home. The dataset is mixed:
`compatibility` and `configuration` are Yes/No spatial QA, while `context`
asks for normalized vacant-space points and provides a mask for scoring. The
script reports QA accuracy for the Yes/No categories and Point Acc@1 / Acc@K
for `context`.

```bash
python scripts/eval_robospatial_home_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/RoboSpatial-Home \
  --output-jsonl ./robospatial_home_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

Useful variants:

```bash
python scripts/eval_robospatial_home_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --category context \
  --prompt-style task-specific \
  --output-jsonl ./robospatial_home_context_predictions.jsonl \
  --batch-size 1
```

By default the evaluator sends only the RGB image to match normal VLM
benchmarking. Pass `--image-source rgb-depth` to provide both RGB and depth
images to the model.

## RefSpatial-Bench VLM Evaluation

Use `eval_refspatial_bench_vlm.py` for RefSpatial-Bench. Each sample contains
one image, one multi-step spatial referring prompt, and one target mask. The
script asks the model for normalized 2D points and reports Point Acc@1 /
Acc@K, sample hit rate, and point precision. The default split selection is
`location,placement`; pass `--split all` only when you also want the official
`unseen` generalization split.

```bash
python scripts/eval_refspatial_bench_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/RefSpatial-Bench \
  --output-jsonl ./refspatial_bench_wall_oss_0_5_predictions.jsonl \
  --batch-size 1
```

Useful variants:

```bash
python scripts/eval_refspatial_bench_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --split location,placement,unseen \
  --prompt-style normalized-point-tags \
  --output-jsonl ./refspatial_bench_all_predictions.jsonl \
  --batch-size 1
```

The default `--prompt-style dataset` uses the benchmark's `prompt + suffix`.
`--coord-mode auto` accepts normalized coordinates and pixel coordinates; use
`--coord-mode percent` or `--coord-mode per-mille` when analyzing outputs from
models that emit 0-100 or 0-1000 normalized coordinates.

## WebSocket serving

`run_serving.sh` launches the Wall-X WebSocket server through the public
vendored serving runtime. Pass paths explicitly; the script has no built-in
checkpoint path.

```bash
bash scripts/run_serving.sh \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --train-config-path model-repos/wall-oss-0.5/config.yml \
  --port 32195
```

By default the script returns raw model action chunks, which is the expected
mode for open-loop plotting. Pass `--serialize-actions` when your client expects
robot-serialized actions.

Useful options:

```bash
CUDA_ID=0
ACTION_HORIZON=32
IMAGE_PASSING_MODE=base64
MAX_BATCH_SIZE=1
```

Additional `launch_serving.py` arguments can be forwarded after `--`:

```bash
bash scripts/run_serving.sh --checkpoint-path model-repos/wall-oss-0.5 -- \
  --model-config.norm-key libero_all
```

## Open-loop WebSocket evaluation

`draw_openloop_plot.py` compares predicted action chunks from a running
WebSocket server against LeRobot dataset ground truth. `--dataset-root` and
`--train-config` are required and have no built-in default.

```bash
python scripts/draw_openloop_plot.py \
  --uri ws://127.0.0.1:32195 \
  --dataset-root datasets/lerobot \
  --train-config configs/train.yml \
  --episode-indices 0,1,2 \
  --save-dir ./openloop_plots
```

## Dataset and checkpoint utilities

- `compute_norm_stats.py`: compute action normalization statistics for a
  local LeRobot v3 dataset. The script reads state/action parquet columns
  directly when available, so image and video columns are not decoded.
- `merge_sharded_weights.py`: merge FSDP sharded checkpoint files into a single
  checkpoint directory.
- `merge_tokenizer.py`: merge FAST action tokens into a Qwen2.5-VL processor
  tokenizer.

```bash
python scripts/merge_tokenizer.py \
  --processor-path model-repos/qwen2.5-vl-3b-instruct \
  --action-tokenizer-path model-repos/fast-tokenizer \
  --output-dir model-repos/wall-oss-0.5/merged-processor
```

Most scripts support `--help` for their command-line options.
