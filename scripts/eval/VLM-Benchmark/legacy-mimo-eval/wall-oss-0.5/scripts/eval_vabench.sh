python evaluations/eval_vabench_point_bbox_vlm.py \
  --checkpoint-path x-square-robot/wall-oss-0.5 \
  --dataset-root benchmarks/vabench-point-bbox \
  --prompt-style point-tags \
  --output-jsonl ./vabench_point_bbox_point_tags_predictions.jsonl \
  --batch-size 1 \
  --max-length 2048
