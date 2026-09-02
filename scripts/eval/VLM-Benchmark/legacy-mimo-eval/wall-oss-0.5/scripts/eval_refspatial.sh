python evaluations/eval_refspatial_bench_vlm.py \
  --checkpoint-path x-square-robot/wall-oss-0.5 \
  --dataset-root benchmarks/RefSpatial-Bench \
  --split all \
  --prompt-style normalized-point-tags \
  --output-jsonl ./refspatial_bench_all_predictions.jsonl \
  --batch-size 10 \
  --max-length 4096