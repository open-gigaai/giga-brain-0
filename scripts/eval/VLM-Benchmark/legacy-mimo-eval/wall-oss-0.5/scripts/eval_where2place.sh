python evaluations/eval_where2place_vlm.py \
  --checkpoint-path x-square-robot/wall-oss-0.5 \
  --dataset-root benchmarks/Where2Place \
  --prompt-style diverse-point-tags \
  --output-jsonl ./where2place_diverse_point_tags_predictions.jsonl \
  --batch-size 1 \
  --max-length 4096
