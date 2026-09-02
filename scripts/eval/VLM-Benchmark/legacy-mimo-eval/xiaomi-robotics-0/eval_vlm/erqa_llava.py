# Copyright (C) 2026 Xiaomi Corporation.
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

sys.path.insert(0, ".")
from eval_vlm.common import (  # noqa: E402
    add_model_args,
    build_vlm_prompt,
    exact_metrics,
    generate_predictions,
    load_model_and_processor,
    normalize_answer,
    write_outputs,
)


DEFAULT_DATA_PATH = (
    "datasets/public_datasets/VLM/benchmarks/ERQA/llava_json/"
    "erqa_test_llava.jsonl"
)
DEFAULT_MODEL_PATH = "model-repos/xiaomi-robotics-0"

LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate XR-0 on ERQA LLaVA-format JSONL.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="Path to erqa_test_llava.jsonl.")
    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional root used when image paths in the JSONL are relative.",
    )
    add_model_args(parser, "eval_vlm/results/erqa_llava_predictions.jsonl")
    parser.set_defaults(model_path=DEFAULT_MODEL_PATH, max_new_tokens=8)
    return parser.parse_args()


def conversation_value(row: Dict[str, Any], speaker: str) -> str:
    for turn in row.get("conversations", []):
        if turn.get("from") == speaker:
            return str(turn.get("value", ""))
    raise ValueError(f"Missing {speaker!r} conversation in sample id={row.get('id')!r}")


def clean_question(text: str) -> str:
    text = str(text).replace("<image>", "").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def image_paths_from_row(row: Dict[str, Any], data_path: Path, image_root: Optional[Path]) -> List[Path]:
    image_field = row.get("image")
    if not image_field:
        raise ValueError(f"Missing image path in sample id={row.get('id')!r}")

    image_fields = image_field if isinstance(image_field, list) else [image_field]
    image_paths = []
    for field in image_fields:
        path = Path(str(field))
        if not path.is_absolute():
            path = (image_root / path) if image_root is not None else (data_path.parent / path)
        if not path.exists():
            raise FileNotFoundError(f"Missing ERQA image for sample id={row.get('id')!r}: {path}")
        image_paths.append(path)
    return image_paths


def read_records(data_path: Path, image_root: Optional[Path], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            question = clean_question(conversation_value(row, "human"))
            answer = conversation_value(row, "gpt").strip().upper()
            image_paths = image_paths_from_row(row, data_path, image_root)
            images = [Image.open(path).convert("RGB") for path in image_paths]
            records.append(
                {
                    "id": str(row.get("id", line_idx)),
                    "image_paths": [str(path) for path in image_paths],
                    "question": question,
                    "answer": answer,
                    "images": images,
                    "prompt": build_vlm_prompt(
                        question,
                        num_images=len(images),
                        answer_instruction="",
                    ),
                }
            )
            if limit is not None and len(records) >= limit:
                return records
    return records


def first_choice(text: str) -> Optional[str]:
    match = LETTER_RE.search(str(text).strip())
    return match.group(1).upper() if match else None


def score_prediction(prediction: str, answer: str) -> bool:
    return first_choice(prediction) == first_choice(answer)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    image_root = Path(args.image_root) if args.image_root else None
    records = read_records(data_path, image_root, args.limit)
    print(f"Loaded {len(records)} ERQA LLaVA examples from {data_path}")

    model, processor, device = load_model_and_processor(args)
    outputs = generate_predictions(model, processor, device, records, args, "ERQA-LLaVA")

    predictions = []
    for item, pred in zip(records, outputs):
        is_correct = score_prediction(pred, item["answer"])
        predictions.append(
            {
                "id": item["id"],
                "image_paths": item["image_paths"],
                "question": item["question"],
                "answer": item["answer"],
                "prediction": pred.strip(),
                "prediction_choice": first_choice(pred),
                "normalized_prediction": normalize_answer(pred),
                "correct": is_correct,
            }
        )

    metrics = exact_metrics(predictions)
    metrics.update({"model_path": args.model_path, "data_path": args.data_path})
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['num_examples']})")
    write_outputs(predictions, args.output, metrics)


if __name__ == "__main__":
    main()
