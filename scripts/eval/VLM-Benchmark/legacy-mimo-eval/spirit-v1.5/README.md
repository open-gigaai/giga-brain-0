# Spirit-v1.5 MiMo-Embodied Evaluation

Run commands from the `VLM-Benchmark` directory.

## Resources

Before running, provide:

- `model-repos/spirit-v1.5/`: Spirit-v1.5 checkpoint containing `config.json` and `model.safetensors`.
- `model-repos/qwen3-vl-4b-instruct/`: optional local Qwen3-VL backbone and processor.
- `datasets/public_datasets/VLM/`: MiMo-Embodied datasets.

The launchers use repository-relative defaults. Override `CKPT_PATH`, `DATASET_PATH`, `BACKBONE_PATH`, `PROCESSOR_PATH`, or `PYTHON_BIN` when resources are stored elsewhere. Do not commit machine-specific paths.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r legacy-mimo-eval/spirit-v1.5/requirements.txt
```

## Run

Smoke-test RoboSpatial-Home:

```bash
CUDA_VISIBLE_DEVICES=0 \
CKPT_PATH=model-repos/spirit-v1.5 \
DATASET_PATH=datasets/public_datasets/VLM/vqa/RoboSpatial-Home \
bash legacy-mimo-eval/spirit-v1.5/scripts/run_robospatial_home_vlm.sh \
  --max-samples 10
```

Remove `--max-samples 10` for a full run. Other tasks use the corresponding launcher under `spirit-v1.5/scripts/`, for example `run_erqa_vlm.sh`, `run_crpe_vlm.sh`, and `run_refspatial_vlm.sh`.

Generated outputs are written under `spirit-v1.5/outputs/` unless `OUTPUT_DIR` is set.
