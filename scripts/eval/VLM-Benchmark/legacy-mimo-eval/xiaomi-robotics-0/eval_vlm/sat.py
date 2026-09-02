# Copyright (C) 2026 Xiaomi Corporation.
import argparse
from pathlib import Path
import sys
sys.path.insert(0, '.')
from eval_vlm.common import (
    add_model_args,
    build_vlm_prompt,
    exact_metrics,
    generate_predictions,
    image_from_bytes,
    load_model_and_processor,
    normalize_answer,
    write_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XR-0 VLM on SAT.")
    parser.add_argument("--data-root", default="datasets/public_datasets/VLM/vqa/benchmarks/SAT")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    add_model_args(parser, "eval_vlm/results/sat_predictions.jsonl")
    return parser.parse_args()


def read_records(data_root: Path, split: str, limit=None):
    import pyarrow.parquet as pq

    file = data_root / f"SAT_{split}.parquet"
    records = []
    table = pq.read_table(file)
    cols = table.column_names
    for idx in range(table.num_rows):
        row = {col: table[col][idx].as_py() for col in cols}
        answers = row["answers"]
        choices = "\n".join(f"({chr(ord('A') + i)}) {ans}" for i, ans in enumerate(answers))
        question = f"{row['question']}\n{choices}\nAnswer directly with the exact choice text."
        records.append(
            {
                "id": str(idx),
                "question_type": row["question_type"],
                "answer": row["correct_answer"],
                "images": [image_from_bytes(data) for data in row["image_bytes"]],
                "prompt": build_vlm_prompt(question, len(row["image_bytes"]), answer_instruction=""),
            }
        )
        if limit is not None and len(records) >= limit:
            return records
    return records


def main():
    args = parse_args()
    records = read_records(Path(args.data_root), args.split, args.limit)
    print(f"Loaded {len(records)} SAT examples")
    model, processor, device = load_model_and_processor(args)
    outputs = generate_predictions(model, processor, device, records, args, "SAT")
    predictions = []
    for item, pred in zip(records, outputs):
        correct = normalize_answer(pred) == normalize_answer(item["answer"])
        predictions.append({k: item[k] for k in ("id", "question_type", "answer")} | {"prediction": pred.strip(), "correct": correct})
    metrics = exact_metrics(predictions, ["question_type"])
    metrics.update({"model_path": args.model_path, "data_root": args.data_root, "split": args.split})
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['num_examples']})")
    write_outputs(predictions, args.output, metrics)


if __name__ == "__main__":
    main()
