python evaluations/eval_erqa_vlm.py \
  --checkpoint-path x-square-robot/wall-oss-0.5 \
  --dataset-root benchmarks/ERQA \
  --output-jsonl ./erqa_wall_oss_0_5_predictions.jsonl \
  --batch-size 1 \
  --max-length 4096
