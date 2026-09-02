# Copyright (C) 2026 Xiaomi Corporation.
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, '.')
from eval_vlm.common import (
    add_model_args,
    build_vlm_prompt,
    choice_accuracy,
    exact_metrics,
    generate_predictions,
    load_model_and_processor,
    write_outputs,
)
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XR-0 VLM on MetaVQA-Eval.")
    parser.add_argument("--data-root", default="datasets/public_datasets/VLM/vqa/MetaVQA-Eval")
    add_model_args(parser, "eval_vlm/results/metavqa_predictions.jsonl")
    return parser.parse_args()


def read_records(data_root: Path, limit=None):
    records = []
    with (data_root / "test.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if not row.get("answer"):
                continue
            obs_path = data_root / row["obs"]
            records.append(
                {
                    "id": str(row["question_id"]),
                    "type": row.get("type", ""),
                    "domain": row.get("domain", ""),
                    "answer": row["answer"],
                    "images": [Image.open(obs_path).convert("RGB")],
                    "prompt": build_vlm_prompt(row["question"], answer_instruction="Answer directly with only the option letter."),
                }
            )
            if limit is not None and len(records) >= limit:
                return records
    return records


def main():
    args = parse_args()
    records = read_records(Path(args.data_root), args.limit)
    print(f"Loaded {len(records)} MetaVQA-Eval examples")
    model, processor, device = load_model_and_processor(args)
    outputs = generate_predictions(model, processor, device, records, args, "MetaVQA")
    predictions = []
    for item, pred in zip(records, outputs):
        correct = choice_accuracy(pred, item["answer"], max_letter="D")
        predictions.append({k: item[k] for k in ("id", "type", "domain", "answer")} | {"prediction": pred.strip(), "correct": correct})
    metrics = exact_metrics(predictions, ["type", "domain"])
    metrics.update({"model_path": args.model_path, "data_root": args.data_root})
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['num_examples']})")
    write_outputs(predictions, args.output, metrics)


if __name__ == "__main__":
    main()
