#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
NUM_SHARDS="${NUM_SHARDS:-${#GPUS[@]}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/erqa_vlm}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "[ERROR] No GPUs specified. Set GPU_IDS, e.g. GPU_IDS=0,1,2,3." >&2
  exit 1
fi
if [[ "${NUM_SHARDS}" -ne "${#GPUS[@]}" ]]; then
  echo "[ERROR] NUM_SHARDS (${NUM_SHARDS}) must match number of GPU_IDS (${#GPUS[@]})." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[INFO] GPU_IDS=${GPU_IDS}"
echo "[INFO] NUM_SHARDS=${NUM_SHARDS}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo "[INFO] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

pids=()
for shard_index in "${!GPUS[@]}"; do
  gpu_id="${GPUS[${shard_index}]}"
  shard_output="${OUTPUT_DIR}/shard_${shard_index}"
  log_file="${LOG_DIR}/shard_${shard_index}.log"
  echo "[INFO] Launching shard ${shard_index}/${NUM_SHARDS} on GPU ${gpu_id}; log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    OUTPUT_DIR="${shard_output}" \
    "${SCRIPT_DIR}/run_erqa_vlm.sh" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard_index}" \
      "$@" > "${log_file}" 2>&1 &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  pid="${pids[${i}]}"
  if wait "${pid}"; then
    echo "[INFO] Shard ${i} finished."
  else
    echo "[ERROR] Shard ${i} failed. See ${LOG_DIR}/shard_${i}.log" >&2
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

MERGE_OUTPUT_DIR="${OUTPUT_DIR}" MERGE_NUM_SHARDS="${NUM_SHARDS}" python - <<'PY'
import json
import os
import time
from pathlib import Path

output_dir = Path(os.environ["MERGE_OUTPUT_DIR"])
num_shards = int(os.environ["MERGE_NUM_SHARDS"])
records_by_id = {}
shard_summaries = []

for shard_index in range(num_shards):
    shard_dir = output_dir / f"shard_{shard_index}"
    predictions_path = shard_dir / "predictions.jsonl"
    summary_path = shard_dir / "summary.json"
    if summary_path.exists():
        shard_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing shard predictions: {predictions_path}")
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records_by_id[int(record["sample_id"])] = record

records = [records_by_id[key] for key in sorted(records_by_id)]
merged_predictions = output_dir / "predictions.jsonl"
with merged_predictions.open("w", encoding="utf-8") as handle:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

def build_type_metrics(items):
    metrics = {}
    for record in items:
        key = record["question_type"]
        bucket = metrics.setdefault(key, {"num_samples": 0, "num_correct": 0, "accuracy": 0.0})
        bucket["num_samples"] += 1
        bucket["num_correct"] += int(record["correct"])
    for bucket in metrics.values():
        bucket["accuracy"] = bucket["num_correct"] / max(bucket["num_samples"], 1)
    return metrics

num_samples = len(records)
num_correct = sum(int(record["correct"]) for record in records)
base_summary = shard_summaries[0] if shard_summaries else {}
summary = {
    "dataset_path": base_summary.get("dataset_path"),
    "ckpt_path": base_summary.get("ckpt_path"),
    "processor_path": base_summary.get("processor_path"),
    "split": base_summary.get("split", "test"),
    "num_total_loaded": base_summary.get("num_total_loaded"),
    "num_shards": num_shards,
    "num_samples": num_samples,
    "num_correct": num_correct,
    "accuracy": num_correct / max(num_samples, 1),
    "by_question_type": build_type_metrics(records),
    "elapsed_seconds": sum(float(summary.get("elapsed_seconds", 0.0)) for summary in shard_summaries),
    "predictions_path": str(merged_predictions),
    "shard_summaries": [str(output_dir / f"shard_{idx}" / "summary.json") for idx in range(num_shards)],
    "merged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "use_visual_indices": base_summary.get("use_visual_indices"),
    "max_images_per_sample": base_summary.get("max_images_per_sample"),
    "generation": base_summary.get("generation", {}),
}
(output_dir / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
