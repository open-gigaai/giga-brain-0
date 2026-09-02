# Copyright (C) 2026 Xiaomi Corporation.
import argparse
from pathlib import Path
import sys
sys.path.insert(0, '.')
from eval_vlm.common import add_model_args, bbox_iou, bbox_metrics, build_vlm_prompt, generate_predictions, image_from_field, load_model_and_processor, parse_bbox, parse_points, point_in_bbox, write_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XR-0 VLM on RoboRefIt.")
    parser.add_argument("--data-root", default="datasets/public_datasets/VLM/vqa/roborefit")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    add_model_args(parser, "eval_vlm/results/roborefit_predictions.jsonl")
    return parser.parse_args()


def read_records(data_root: Path, limit=None):
    import pyarrow.parquet as pq
    records = []
    for file in sorted((data_root / "data").glob("test-*.parquet")):
        for row in pq.read_table(file).to_pylist():
            image = image_from_field(row["image"])
            question = f"{row['ref_exp']}\nReturn the bounding box of the referred object as [x1, y1, x2, y2] in image pixels."
            records.append({"id": str(row["id"]), "target_bbox": list(row["bbox"]), "image_size": image.size, "images": [image], "prompt": build_vlm_prompt(question, answer_instruction="Answer directly with only the bounding box.")})
            if limit is not None and len(records) >= limit:
                return records
    return records


def main():
    args = parse_args()
    records = read_records(Path(args.data_root), args.limit)
    print(f"Loaded {len(records)} RoboRefIt examples")
    model, processor, device = load_model_and_processor(args)
    outputs = generate_predictions(model, processor, device, records, args, "RoboRefIt")
    predictions = []
    for item, pred in zip(records, outputs):
        pred_bbox = parse_bbox(pred, item["image_size"])
        iou = bbox_iou(pred_bbox, item["target_bbox"])
        pts = parse_points(pred, item["image_size"])
        point_hit = any(point_in_bbox(pt, item["target_bbox"], item["image_size"]) for pt in pts)
        predictions.append({"id": item["id"], "answer_bbox": item["target_bbox"], "prediction": pred.strip(), "prediction_bbox": pred_bbox, "iou": iou, "point_hit": point_hit})
    metrics = bbox_metrics(predictions, args.iou_threshold)
    metrics["point_hit_rate"] = sum(int(r["point_hit"]) for r in predictions) / len(predictions) if predictions else 0.0
    metrics.update({"model_path": args.model_path, "data_root": args.data_root})
    write_outputs(predictions, args.output, metrics)


if __name__ == "__main__":
    main()
