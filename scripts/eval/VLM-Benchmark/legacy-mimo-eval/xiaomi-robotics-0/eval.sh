MODEL=XiaomiRobotics/Xiaomi-Robotics-0-Pretrain
python eval_vlm/where2place.py \
  --model-path ${MODEL} \
  --data-root benchmarks/Where2Place \
  --batch-size 1 \
  --attn-implementation eager \
  --max-length 4096 \
  --output eval_vlm/results/where2place_predictions.jsonl

python eval_vlm/roborefit.py \
  --model-path ${MODEL} \
  --data-root benchmarks/roborefit \
  --batch-size 10 \
  --attn-implementation eager \
  --max-length 4096 \
  --output eval_vlm/results/roborefit_predictions.jsonl

python eval_vlm/vabench_point_bbox.py \
  --model-path ${MODEL} \
  --data-root benchmarks/vabench-point-bbox \
  --batch-size 10 \
  --attn-implementation eager \
  --max-length 4096 \
  --output eval_vlm/results/vabench_point_bbox_predictions.jsonl

python eval_vlm/erqa.py \
  --model-path ${MODEL} \
  --data-root benchmarks/ERQA \
  --batch-size 10 \
  --attn-implementation eager \
  --max-length 4096 \
  --output eval_vlm/results/erqa_predictions.jsonl

python eval_vlm/part_affordance.py \
  --model-path ${MODEL} \
  --data-root benchmarks/Part-Affordance-2K \
  --batch-size 10 \
  --attn-implementation eager \
  --max-length 4096 \
  --output eval_vlm/results/part_affordance_predictions.jsonl

python eval_vlm/roboafford.py \
  --model-path ${MODEL} \
  --data-root benchmarks/roboafford-eval \
  --batch-size 10 \
  --attn-implementation eager \
  --max-length 4096 \
  --output eval_vlm/results/roboafford_predictions.jsonl

python eval_vlm/cvbench.py \
  --model-path ${MODEL} \
  --data-root benchmarks/CV-Bench \
  --split all \
  --batch-size 10 \
  --attn-implementation eager \
  --max-length 4096 \
  --output eval_vlm/results/cvbench_predictions.jsonl

python eval_vlm/embspatial.py \
  --model-path ${MODEL} \
  --data-root EmbSpatial-Bench \
  --batch-size 10 \
  --attn-implementation eager \
  --max-length 4096 \
  --output eval_vlm/results/embspatial_predictions.jsonl

python eval_vlm/sat.py \
  --model-path ${MODEL} \
  --data-root benchmarks/SAT \
  --split test \
  --batch-size 10 \
  --attn-implementation eager \
  --max-length 4096 \
  --output eval_vlm/results/sat_predictions.jsonl

python eval_vlm/robospatial_home.py \
  --model-path ${MODEL} \
  --data-root RoboSpatial-Home \
  --split all \
  --batch-size 10 \
  --attn-implementation eager \
  --max-length 4096 \
  --output eval_vlm/results/robospatial_home_predictions.jsonl

python eval_vlm/refspatial.py \
  --model-path ${MODEL} \
  --data-root benchmarks/RefSpatial-Bench \
  --split all \
  --batch-size 10 \
  --attn-implementation eager \
  --max-length 4096 \
  --output eval_vlm/results/refspatial_predictions.jsonl

python eval_vlm/crpe.py \
  --model-path ${MODEL} \
  --data-root benchmarks/CRPE \
  --image-root MSCOCO \
  --split all \
  --batch-size 10 \
  --attn-implementation eager \
  --max-length 4096 \
  --output eval_vlm/results/crpe_predictions.jsonl

python eval_vlm/metavqa.py \
  --model-path ${MODEL} \
  --data-root benchmarks/MetaVQA-Eval \
  --batch-size 10 \
  --attn-implementation eager \
  --max-length 4096 \
  --output eval_vlm/results/metavqa_predictions.jsonl