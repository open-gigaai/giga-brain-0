python evaluations/eval_sat_vlm.py \
  --checkpoint-path x-square-robot/wall-oss-0.5 \
  --dataset-root benchmarks/SAT \
  --circular-eval \
  --output-jsonl ./sat_circular_wall_oss_0_5_predictions.jsonl \
  --batch-size 6
