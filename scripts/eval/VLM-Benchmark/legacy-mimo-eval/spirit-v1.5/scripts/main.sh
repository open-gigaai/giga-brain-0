export BACKBONE_PATH=model-repos/qwen3-vl-4b-instruct
export PROCESSOR_PATH=model-repos/qwen3-vl-4b-instruct
export BATCH_SIZE=10
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_IMAGES_PER_SAMPLE=1

export GPU_IDS=0,1,2,3

# bash scripts/run_erqa_vlm.sh
# bash scripts/run_crpe_vlm.sh
# bash scripts/run_where2place_vlm.sh
# bash scripts/run_roborefit_vlm.sh
# bash scripts/run_vabench_point_bbox_vlm.sh
# bash scripts/run_part_affordance_vlm.sh
bash scripts/run_roboafford_vlm.sh
bash scripts/run_cvbench_vlm.sh
bash scripts/run_embspatial_vlm.sh
bash scripts/run_sat_vlm.sh
bash scripts/run_robospatial_home_vlm.sh
bash scripts/run_refspatial_vlm.sh
# bash scripts/run_crpe_vlm.sh
bash scripts/run_metavqa_eval_vlm.sh