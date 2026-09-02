# Copyright (C) 2026 Xiaomi Corporation.
import argparse
from pathlib import Path

import sys
sys.path.insert(0, '.')
from eval_vlm.common import add_model_args, build_vlm_prompt, generate_predictions, image_from_field, load_model_and_processor, mask_from_field, parse_points, point_mask_score, point_metrics, write_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XR-0 VLM on Where2Place.")
    parser.add_argument("--data-root", default="datasets/public_datasets/VLM/vqa/benchmarks/Where2Place")
    add_model_args(parser, "eval_vlm/results/where2place_predictions.jsonl")
    return parser.parse_args()


def read_records(data_root: Path, limit=None):
    import pyarrow.parquet as pq
    records = []
    for row in pq.read_table(data_root / "data/test-00000-of-00001.parquet").to_pylist():
        image = image_from_field(row["image"])
        records.append({"id": row["question_id"], "question_type": row["question_type"], "mask": mask_from_field(row["mask"]), "image_size": image.size, "images": [image], "prompt": build_vlm_prompt(row["question"], answer_instruction="Answer directly with several normalized coordinate tuples, e.g. [(0.5, 0.5)].")})
        if limit is not None and len(records) >= limit:
            return records
    return records


def main():
    args = parse_args()
    records = read_records(Path(args.data_root), args.limit)
    print(f"Loaded {len(records)} Where2Place examples")
    model, processor, device = load_model_and_processor(args)
    outputs = generate_predictions(model, processor, device, records, args, "Where2Place")
    predictions = []
    for item, pred in zip(records, outputs):
        pts = parse_points(pred, item["image_size"])
        row = {"id": item["id"], "question_type": item["question_type"], "prediction": pred.strip(), **point_mask_score(pts, item["mask"])}
        predictions.append(row)
    metrics = point_metrics(predictions, ["question_type"])
    metrics.update({"model_path": args.model_path, "data_root": args.data_root})
    write_outputs(predictions, args.output, metrics)


if __name__ == "__main__":
    main()
