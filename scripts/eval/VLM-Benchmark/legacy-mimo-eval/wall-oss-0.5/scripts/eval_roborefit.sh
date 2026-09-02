python evaluations/eval_roborefit_vlm.py \
  --checkpoint-path x-square-robot/wall-oss-0.5 \
  --dataset-root benchmarks/roborefit \
  --prompt-style strict-box-tags \
  --output-jsonl ./roborefit_strict_box_tags_predictions.jsonl \
  --batch-size 10
