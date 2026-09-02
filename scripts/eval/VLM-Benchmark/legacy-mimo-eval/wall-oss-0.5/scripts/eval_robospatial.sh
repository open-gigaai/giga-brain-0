python evaluations/eval_robospatial_home_vlm.py \
  --checkpoint-path x-square-robot/wall-oss-0.5 \
  --dataset-root benchmarks/RoboSpatial-Home \
  --output-jsonl ./robospatial_home_wall_oss_0_5_predictions.jsonl \
  --batch-size 10 \
  --max-length 4096
