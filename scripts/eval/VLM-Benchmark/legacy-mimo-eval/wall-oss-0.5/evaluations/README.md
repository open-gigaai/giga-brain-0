# Wall-OSS-0.5 Evaluation Entrypoints

This directory contains the Wall-OSS-0.5 evaluators and runtime helpers included in this release. The Wall-X model runtime and its dependencies are external resources and must be installed from the upstream Wall-OSS repository.

Run the commands below from `scripts/eval/VLM-Benchmark`. From the main repository root:

```bash
cd scripts/eval/VLM-Benchmark
```

## Resources

Prepare these local resources before running an evaluation:

```text
model-repos/wall-oss-0.5/        # Wall-OSS-0.5 checkpoint
datasets/public_datasets/VLM/    # MiMo-Embodied datasets
datasets/public_datasets/MSCOCO/ # COCO images used by CRPE
```

The paths are not committed to Git. Pass `--checkpoint-path`, `--dataset-root`, and task-specific path arguments when using another layout.

## VLM Evaluators

The following entrypoints are included:

| Benchmark | Entrypoint |
| --- | --- |
| CRPE | `eval_crpe_vlm.py` |
| CV-Bench | `eval_cvbench_vlm.py` |
| EmbSpatial-Bench | `eval_embspatial_bench_vlm.py` |
| ERQA | `eval_erqa_vlm.py` |
| MetaVQA-Eval | `eval_metavqa_eval_vlm.py` |
| Part-Affordance-2K | `eval_part_affordance_2k_vlm.py` |
| RefSpatial-Bench | `eval_refspatial_bench_vlm.py` |
| RoboAfford-Eval | `eval_roboafford_eval_vlm.py` |
| RoboRefIt | `eval_roborefit_vlm.py` |
| RoboSpatial-Home | `eval_robospatial_home_vlm.py` |
| SAT | `eval_sat_vlm.py` |
| VABench point/bbox | `eval_vabench_point_bbox_vlm.py` |
| Where2Place | `eval_where2place_vlm.py` |

For example, run a RoboSpatial-Home smoke test with:

```bash
python legacy-mimo-eval/wall-oss-0.5/evaluations/eval_robospatial_home_vlm.py \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --dataset-root datasets/public_datasets/VLM/vqa/RoboSpatial-Home \
  --output-jsonl legacy-mimo-eval/wall-oss-0.5/outputs/robospatial_home.jsonl \
  --batch-size 1 \
  --limit 10
```

Remove `--limit 10` for a full run. Each evaluator documents its dataset filters, prompt modes, and scoring options through `--help`:

```bash
python legacy-mimo-eval/wall-oss-0.5/evaluations/eval_robospatial_home_vlm.py --help
```

The shell launchers under `legacy-mimo-eval/wall-oss-0.5/scripts/` are examples with upstream-style resource defaults. Prefer the Python command above when using the repository-relative layout documented here.

## Runtime Helpers

The release also includes the following Wall-X helpers. They require the corresponding Wall-X or simulator dependencies to be importable in the active environment.

Checkpoint loading smoke test:

```bash
python legacy-mimo-eval/wall-oss-0.5/evaluations/fake_inference.py \
  --checkpoint-path model-repos/wall-oss-0.5
```

LIBERO evaluation:

```bash
LIBERO_PATH=/path/to/LIBERO \
bash legacy-mimo-eval/wall-oss-0.5/evaluations/run_libero.sh \
  model-repos/wall-oss-0.5
```

WebSocket serving:

```bash
bash legacy-mimo-eval/wall-oss-0.5/evaluations/run_serving.sh \
  --checkpoint-path model-repos/wall-oss-0.5 \
  --train-config-path model-repos/wall-oss-0.5/config.yml \
  --port 32195
```

Checkpoint and tokenizer utilities:

```bash
python legacy-mimo-eval/wall-oss-0.5/evaluations/merge_sharded_weights.py \
  /path/to/sharded-checkpoint \
  model-repos/wall-oss-0.5/merged

python legacy-mimo-eval/wall-oss-0.5/evaluations/merge_tokenizer.py \
  --processor-path model-repos/qwen2.5-vl-3b-instruct \
  --action-tokenizer-path model-repos/fast-tokenizer \
  --output-dir model-repos/wall-oss-0.5/merged-processor
```

This release does not include the previously documented RealWorldQA, EO-Bench, open-loop plotting, dataset-statistics, or LIBERO configuration files. Those workflows require the corresponding upstream Wall-X files.
