# Copyright (C) 2026 Xiaomi Corporation.
import argparse
from pathlib import Path
import sys
sys.path.insert(0, '.')
from eval_vlm.common import add_model_args, build_vlm_prompt, generate_predictions, image_from_field, load_model_and_processor, mask_from_field, parse_points, point_mask_score, point_metrics, write_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XR-0 VLM on RefSpatial-Bench.")
    parser.add_argument("--data-root", default="datasets/public_datasets/VLM/vqa/RefSpatial-Bench")
    parser.add_argument("--split", default="all", choices=["all", "location", "placement", "unseen"])
    add_model_args(parser, "eval_vlm/results/refspatial_predictions.jsonl")
    return parser.parse_args()


def read_records(data_root: Path, split: str, limit=None):
    import pyarrow.parquet as pq
    splits = ["location", "placement", "unseen"] if split == "all" else [split]
    records = []
    for name in splits:
        file = next((data_root / "data").glob(f"{name}-*.parquet"))
        for row in pq.read_table(file).to_pylist():
            image = image_from_field(row["image"])
            question = f"{row['prompt']}\n{row['suffix']}"
            records.append({"id": f"{name}:{row['id']}", "split": name, "object": row["object"], "step": row["step"], "mask": mask_from_field(row["mask"]), "image_size": image.size, "images": [image], "prompt": build_vlm_prompt(question, answer_instruction="Answer directly with normalized coordinate tuple(s).")})
            if limit is not None and len(records) >= limit:
                return records
    return records


def main():
    args = parse_args()
    records = read_records(Path(args.data_root), args.split, args.limit)
    print(f"Loaded {len(records)} RefSpatial-Bench examples")
    model, processor, device = load_model_and_processor(args)
    outputs = generate_predictions(model, processor, device, records, args, "RefSpatial")
    predictions = []
    for item, pred in zip(records, outputs):
        predictions.append({"id": item["id"], "split": item["split"], "object": item["object"], "step": item["step"], "prediction": pred.strip(), **point_mask_score(parse_points(pred, item["image_size"]), item["mask"])})
    metrics = point_metrics(predictions, ["split", "step"])
    metrics.update({"model_path": args.model_path, "data_root": args.data_root, "split": args.split})
    write_outputs(predictions, args.output, metrics)


if __name__ == "__main__":
    main()
