python evaluations/eval_crpe_vlm.py \
  --checkpoint-path x-square-robot/wall-oss-0.5 \
  --dataset-root benchmarks/CRPE \
  --coco-root MSCOCO \
  --output-jsonl ./crpe_wall_oss_0_5_predictions.jsonl \
  --batch-size 10
