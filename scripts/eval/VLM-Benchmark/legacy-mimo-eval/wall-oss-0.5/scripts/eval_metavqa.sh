python evaluations/eval_metavqa_eval_vlm.py \
  --checkpoint-path x-square-robot/wall-oss-0.5 \
  --dataset-root benchmarks/MetaVQA-Eval \
  --output-jsonl ./metavqa_eval_wall_oss_0_5_predictions.jsonl \
  --batch-size 30 \
  --max-length 4096
