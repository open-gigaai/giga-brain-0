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
    parser = argparse.ArgumentParser(description="Evaluate XR-0 VLM on EmbSpatial-Bench.")
    parser.add_argument("--data-root", default="datasets/public_datasets/VLM/vqa/EmbSpatial-Bench")
    add_model_args(parser, "eval_vlm/results/embspatial_predictions.jsonl")
    return parser.parse_args()


def read_records(data_root: Path, limit=None):
    import pyarrow.parquet as pq

    records = []
    for row in pq.read_table(data_root / "data/test-00000-of-00001.parquet").to_pylist():
        choices = "\n".join(f"({index_to_letter(i)}) {choice}" for i, choice in enumerate(row["answer_options"]))
        answer = index_to_letter(row["answer"])
        question = f"{row['question']}\n{choices}\nAnswer directly with only the option letter."
        records.append(
            {
                "id": row["question_id"],
                "data_source": row["data_source"],
                "relation": row["relation"],
                "answer": answer,
                "images": [image_from_field(row["image"])],
                "prompt": build_vlm_prompt(question, answer_instruction=""),
            }
        )
        if limit is not None and len(records) >= limit:
            return records
    return records


def main():
    args = parse_args()
    records = read_records(Path(args.data_root), args.limit)
    print(f"Loaded {len(records)} EmbSpatial-Bench examples")
    model, processor, device = load_model_and_processor(args)
    outputs = generate_predictions(model, processor, device, records, args, "EmbSpatial")
    predictions = []
    for item, pred in zip(records, outputs):
        correct = choice_accuracy(pred, item["answer"], max_letter="D")
        predictions.append({k: item[k] for k in ("id", "data_source", "relation", "answer")} | {"prediction": pred.strip(), "correct": correct})
    metrics = exact_metrics(predictions, ["data_source", "relation"])
    metrics.update({"model_path": args.model_path, "data_root": args.data_root})
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['num_examples']})")
    write_outputs(predictions, args.output, metrics)


if __name__ == "__main__":
    main()
