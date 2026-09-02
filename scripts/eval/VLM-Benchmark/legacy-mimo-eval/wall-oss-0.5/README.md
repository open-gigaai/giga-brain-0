# Wall-OSS-0.5 MiMo-Embodied Evaluation

The directory contains evaluation scripts for Wall-OSS-0.5. Install the Wall-X runtime and its dependencies from the upstream repository before running these scripts.

## Resources

Before running, provide:

- `model-repos/wall-oss-0.5/`: Wall-OSS-0.5 checkpoint.
- `datasets/public_datasets/VLM/`: MiMo-Embodied datasets.

Pass `--checkpoint-path` and `--dataset-root` to override these repository-relative locations. Do not commit machine-specific paths.

## Run

Run from the Wall adapter directory because its launchers use paths relative to that directory:

```bash
cd legacy-mimo-eval/wall-oss-0.5
python evaluations/eval_robospatial_home_vlm.py \
  --checkpoint-path ../../model-repos/wall-oss-0.5 \
  --dataset-root ../../datasets/public_datasets/VLM/vqa/RoboSpatial-Home \
  --output-jsonl outputs/robospatial_home_predictions.jsonl \
  --batch-size 1 \
  --limit 10
```

Remove `--limit 10` for a full run. The remaining task entrypoints are the `eval_*_vlm.py` files under `evaluations/`; example shell launchers are under `scripts/`.

See `evaluations/README.md` for task-specific arguments.
