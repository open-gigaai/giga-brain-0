# Copyright (C) 2026 Xiaomi Corporation.
import argparse
import json
from pathlib import Path

from PIL import Image
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XR-0 VLM on CRPE.")
    parser.add_argument("--data-root", default="datasets/public_datasets/VLM/vqa/CRPE")
    parser.add_argument("--image-root", default=None, help="Optional root for paths like coco/val2017/*.jpg.")
    parser.add_argument("--split", default="all", choices=["all", "exist", "relation"])
    parser.add_argument("--skip-missing-images", action="store_true", default=True)
    add_model_args(parser, "eval_vlm/results/crpe_predictions.jsonl")
    return parser.parse_args()


def resolve_image(root: Path, image_path: str, image_root: Path = None):
    candidates = []
    if image_root is not None:
        candidates.extend([image_root / image_path, image_root / Path(image_path).name])
    candidates.extend([root / image_path, root / Path(image_path).name, root / "abnormal_images" / Path(image_path).name])
    for path in candidates:
        if path.exists():
            return path
    return None


def read_records(data_root: Path, split: str, limit=None, skip_missing=True, image_root: Path = None):
    files = []
    if split in ("all", "exist"):
        files.append(("exist", data_root / "crpe_exist.jsonl"))
    if split in ("all", "relation"):
        files.append(("relation", data_root / "crpe_relation.jsonl"))
    records = []
    for split_name, file in files:
        with file.open() as f:
            for line in f:
                row = json.loads(line)
                path = resolve_image(data_root, row["image"], image_root)
                if path is None:
                    if skip_missing:
                        continue
                    raise FileNotFoundError(row["image"])
                records.append(
                    {
                        "id": f"{split_name}:{row['question_id']}",
                        "split": split_name,
                        "category": row["category"],
                        "answer": row["correct_option"],
                        "image_path": str(path),
                        "images": [Image.open(path).convert("RGB")],
                        "prompt": build_vlm_prompt(row["text"], answer_instruction="Answer directly with only the option letter."),
                    }
                )
                if limit is not None and len(records) >= limit:
                    return records
    return records


def main():
    args = parse_args()
    image_root = Path(args.image_root) if args.image_root else None
    records = read_records(Path(args.data_root), args.split, args.limit, args.skip_missing_images, image_root)
    print(f"Loaded {len(records)} CRPE examples")
    model, processor, device = load_model_and_processor(args)
    outputs = generate_predictions(model, processor, device, records, args, "CRPE")
    predictions = []
    for item, pred in zip(records, outputs):
        correct = choice_accuracy(pred, item["answer"], max_letter="D")
        predictions.append({k: item[k] for k in ("id", "split", "category", "answer", "image_path")} | {"prediction": pred.strip(), "correct": correct})
    metrics = exact_metrics(predictions, ["split", "category"])
    metrics.update({"model_path": args.model_path, "data_root": args.data_root})
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['num_examples']})")
    write_outputs(predictions, args.output, metrics)


if __name__ == "__main__":
    main()
