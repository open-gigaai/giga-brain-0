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
    image_from_field,
    mask_from_field,
    normalize_answer,
    parse_points,
    point_mask_score,
    point_metrics,
    load_model_and_processor,
    write_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XR-0 VLM on RoboSpatial-Home.")
    parser.add_argument("--data-root", default="datasets/public_datasets/VLM/vqa/RoboSpatial-Home")
    parser.add_argument("--split", default="all", choices=["all", "compatibility", "configuration", "context"])
    add_model_args(parser, "eval_vlm/results/robospatial_home_predictions.jsonl")
    return parser.parse_args()


def read_records(data_root: Path, split: str, limit=None):
    import pyarrow.parquet as pq

    names = ["compatibility", "configuration", "context"] if split == "all" else [split]
    records = []
    for name in names:
        file = next((data_root / "data").glob(f"{name}-*.parquet"))
        for idx, row in enumerate(pq.read_table(file).to_pylist()):
            image = image_from_field(row["img"])
            is_point = name == "context"
            instruction = "Answer yes or no." if not is_point else "Answer directly with a list of normalized coordinate tuples."
            records.append(
                {
                    "id": f"{name}:{idx}",
                    "category": name,
                    "answer": row["answer"],
                    "mask": mask_from_field(row.get("mask")),
                    "image_size": image.size,
                    "images": [image],
                    "is_point": is_point,
                    "prompt": build_vlm_prompt(row["question"], answer_instruction=instruction),
                }
            )
            if limit is not None and len(records) >= limit:
                return records
    return records


def main():
    args = parse_args()
    records = read_records(Path(args.data_root), args.split, args.limit)
    print(f"Loaded {len(records)} RoboSpatial-Home examples")
    model, processor, device = load_model_and_processor(args)
    outputs = generate_predictions(model, processor, device, records, args, "RoboSpatial-Home")
    exact_rows, point_rows, predictions = [], [], []
    for item, pred in zip(records, outputs):
        row = {k: item[k] for k in ("id", "category", "answer")} | {"prediction": pred.strip()}
        if item["is_point"]:
            pts = parse_points(pred, item["image_size"])
            score = point_mask_score(pts, item["mask"])
            row.update(score)
            point_rows.append(row)
        else:
            correct = normalize_answer(pred) == normalize_answer(item["answer"])
            row["correct"] = correct
            exact_rows.append(row)
        predictions.append(row)
    metrics = {"exact": exact_metrics(exact_rows, ["category"]) if exact_rows else {}, "point": point_metrics(point_rows, ["category"]) if point_rows else {}}
    metrics.update({"model_path": args.model_path, "data_root": args.data_root})
    write_outputs(predictions, args.output, metrics)


if __name__ == "__main__":
    main()
