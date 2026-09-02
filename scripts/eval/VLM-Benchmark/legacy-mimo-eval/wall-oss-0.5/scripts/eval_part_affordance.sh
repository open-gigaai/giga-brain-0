python evaluations/eval_part_affordance_2k_vlm.py \
  --checkpoint-path x-square-robot/wall-oss-0.5 \
  --dataset-root benchmarks/Part-Affordance-2K \
  --output-jsonl ./part_affordance_2k_wall_oss_0_5_predictions.jsonl \
  --batch-size 10 \
   --max-length 4096
