# Copyright (C) 2026 Xiaomi Corporation.
import argparse
from pathlib import Path
import sys
sys.path.insert(0, '.')
from eval_vlm.common import add_model_args, build_vlm_prompt, generate_predictions, image_from_field, load_model_and_processor, mask_from_field, parse_points, point_mask_score, point_metrics, write_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XR-0 VLM on Part-Affordance-2K.")
    parser.add_argument("--data-root", default="datasets/public_datasets/VLM/vqa/Part-Affordance-2K")
    add_model_args(parser, "eval_vlm/results/part_affordance_predictions.jsonl")
    return parser.parse_args()


def read_records(data_root: Path, limit=None):
    import pyarrow.parquet as pq
    file = data_root / "data/train-00000-of-00001.parquet"
    records = []
    for row in pq.read_table(file).to_pylist():
        image = image_from_field(row["image"])
        question = row["problem"] + "\nPoint to the relevant part."
        records.append({"id": str(row["question_id"]), "category_type": row["category_type"], "mask": mask_from_field(row["mask"]), "image_size": image.size, "images": [image], "prompt": build_vlm_prompt(question, answer_instruction="Answer directly with normalized coordinate tuples, e.g. [(0.5, 0.5)].")})
        if limit is not None and len(records) >= limit:
            return records
    return records


def main():
    args = parse_args()
    records = read_records(Path(args.data_root), args.limit)
    print(f"Loaded {len(records)} Part-Affordance-2K examples")
    model, processor, device = load_model_and_processor(args)
    outputs = generate_predictions(model, processor, device, records, args, "Part-Affordance")
    predictions = []
    for item, pred in zip(records, outputs):
        row = {"id": item["id"], "category_type": item["category_type"], "prediction": pred.strip(), **point_mask_score(parse_points(pred, item["image_size"]), item["mask"])}
        predictions.append(row)
    metrics = point_metrics(predictions, ["category_type"])
    metrics.update({"model_path": args.model_path, "data_root": args.data_root})
    write_outputs(predictions, args.output, metrics)


if __name__ == "__main__":
    main()
