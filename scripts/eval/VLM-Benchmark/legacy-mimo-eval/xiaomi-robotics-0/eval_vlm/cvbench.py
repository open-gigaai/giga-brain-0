# Copyright (C) 2026 Xiaomi Corporation.
import argparse
from pathlib import Path
import sys
sys.path.insert(0, '.')
from eval_vlm.common import (
    add_model_args,
    build_vlm_prompt,
    choice_accuracy,
    exact_metrics,
    generate_predictions,
    image_from_field,
    index_to_letter,
    load_model_and_processor,
    write_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XR-0 VLM on CV-Bench.")
    parser.add_argument("--data-root", default="datasets/public_datasets/VLM/vqa/CV-Bench")
    parser.add_argument("--split", default="all", choices=["all", "2d", "3d"])
    add_model_args(parser, "eval_vlm/results/cvbench_predictions.jsonl")
    return parser.parse_args()


def read_records(data_root: Path, split: str, limit=None):
    import pyarrow.parquet as pq

    files = []
    if split in ("all", "2d"):
        files.append(data_root / "test_2d.parquet")
    if split in ("all", "3d"):
        files.append(data_root / "test_3d.parquet")

    records = []
    for file in files:
        for row in pq.read_table(file).to_pylist():
            answer = row["answer"].strip("()")
            question = row["prompt"] + "\nAnswer directly with only the option letter."
            image = image_from_field(row["image"])
            records.append(
                {
                    "id": str(row["idx"]),
                    "type": row["type"],
                    "task": row["task"],
                    "answer": answer,
                    "images": [image],
                    "prompt": build_vlm_prompt(question, answer_instruction=""),
                }
            )
            if limit is not None and len(records) >= limit:
                return records
    return records


def main():
    args = parse_args()
    records = read_records(Path(args.data_root), args.split, args.limit)
    print(f"Loaded {len(records)} CV-Bench examples")
    model, processor, device = load_model_and_processor(args)
    outputs = generate_predictions(model, processor, device, records, args, "CV-Bench")
    predictions = []
    for item, pred in zip(records, outputs):
        correct = choice_accuracy(pred, item["answer"], max_letter="F")
        predictions.append({k: item[k] for k in ("id", "type", "task", "answer")} | {"prediction": pred.strip(), "correct": correct})
    metrics = exact_metrics(predictions, ["type", "task"])
    metrics.update({"model_path": args.model_path, "data_root": args.data_root, "split": args.split})
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['num_examples']})")
    write_outputs(predictions, args.output, metrics)


if __name__ == "__main__":
    main()
