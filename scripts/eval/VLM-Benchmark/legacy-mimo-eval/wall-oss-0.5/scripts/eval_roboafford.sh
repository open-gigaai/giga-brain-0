python evaluations/eval_roboafford_eval_vlm.py \
  --checkpoint-path x-square-robot/wall-oss-0.5 \
  --dataset-root benchmarks/roboafford-eval \
  --prompt-style normalized-point-tags \
  --output-jsonl ./roboafford_eval_normalized_point_tags_predictions.jsonl \
  --batch-size 10 \
  --max-length 4096