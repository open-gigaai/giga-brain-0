#!/usr/bin/env python3
"""Evaluate Wall-OSS-0.5 VLM text generation on ERQA.

ERQA contains single-image and multi-image embodied reasoning questions with
single-letter multiple-choice answers. This evaluator handles the variable
number of images per sample and reports overall and per-category accuracy.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_ROOT = "datasets/public_datasets/VLM/vqa/benchmarks/ERQA"

ROLE_START = "<|im_start|>"
ROLE_END = "<|im_end|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
MAX_IMAGES = 16


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Wall-OSS checkpoint directory, e.g. model-repos/wall-oss-0.5.",
    )
    parser.add_argument(
        "--train-config-path",
        default=None,
        help="Optional train config. Defaults to config.yml/config.yaml in checkpoint.",
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="ERQA root containing data/test-*.parquet.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="erqa_wall_oss_0_5_predictions.jsonl",
        help="Where to write per-sample predictions.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--norm-key", default="x2_normal")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Tokenizer max_length. Multi-image ERQA samples may need a larger value.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["numbered-images", "question-only", "cot-answer"],
        default="numbered-images",
    )
    return parser.parse_args()


def image_key(index: int) -> str:
    return f"image_{index}"


def image_placeholder(index: int, prompt_style: str) -> str:
    placeholder = f"{VISION_START}{IMAGE_PAD}{VISION_END}"
    if prompt_style == "question-only":
        return placeholder
    return f"Image {index + 1}: {placeholder}"


def build_prompt(question: str, num_images: int, prompt_style: str) -> str:
    question = question.strip()
    image_lines = "\n".join(image_placeholder(i, prompt_style) for i in range(num_images))
    prompt_text = question
    if prompt_style == "cot-answer":
        prompt_text = (
            "Answer the following multiple-choice embodied reasoning question. "
            'The last line of your response should be of the form "ANSWER: [ANSWER]" '
            "where [ANSWER] is one option letter.\n\n"
            f"{question}\n\n"
            'Remember to put only the final answer on its own line as "ANSWER: [ANSWER]".'
        )
    return (
        f"{ROLE_START}system\nYou are a helpful assistant.{ROLE_END}\n"
        f"{ROLE_START}user\n"
        f"{image_lines}\n"
        f"{prompt_text}{ROLE_END}\n"
        f"{ROLE_START}assistant\n"
    )


def clean_prediction(text: str) -> str:
    text = str(text)
    text = text.split(ROLE_END, 1)[0]
    for token in (
        ROLE_START,
        ROLE_END,
        VISION_START,
        VISION_END,
        IMAGE_PAD,
        "<|endoftext|>",
    ):
        text = text.replace(token, "")
    text = re.sub(r"^(assistant|answer)\s*:\s*", "", text.strip(), flags=re.I)
    return text.strip()


def extract_final_answer(text: str) -> str:
    cleaned = clean_prediction(text)
    matches = re.findall(r"(?im)^\s*ANSWER\s*:\s*(.+?)\s*$", cleaned)
    if matches:
        return matches[-1].strip()
    inline = re.findall(r"(?i)\bANSWER\s*:\s*([^\n\r]+)", cleaned)
    if inline:
        return inline[-1].strip()
    return cleaned


def normalize_choice(text: str) -> str:
    final = extract_final_answer(text).upper().strip()

    explicit = re.search(
        r"\b(?:ANSWER|ANS|OPTION|CHOICE)\b(?:\s+IS)?\s*[:\-]?\s*([A-D])\b",
        final,
    )
    if explicit:
        return explicit.group(1)

    compact = re.sub(r"[\s,.;:/\\|&()\[\]{}'\"`-]+", "", final)
    if re.fullmatch(r"[A-D]", compact):
        return compact

    match = re.search(r"(?<![A-Z])([A-D])(?![A-Z])", final)
    return match.group(1) if match else ""


def exact_match(prediction: str, answer: str) -> bool:
    return normalize_choice(prediction) == normalize_choice(answer)


def iter_erqa(dataset_root: Path) -> Iterable[dict]:
    import pyarrow.parquet as pq
    from PIL import Image

    parquet_files = sorted((dataset_root / "data").glob("test-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    columns = [
        "question_id",
        "question",
        "question_type",
        "answer",
        "visual_indices",
        "images",
    ]
    for parquet_file in parquet_files:
        pf = pq.ParquetFile(parquet_file)
        for batch in pf.iter_batches(batch_size=32, columns=columns):
            for row in batch.to_pylist():
                images = [load_image(image_obj, Image) for image_obj in row["images"]]
                yield {
                    "index": int(str(row["question_id"]).split("_")[-1]),
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "question_type": row["question_type"],
                    "answer": row["answer"],
                    "visual_indices": list(row["visual_indices"]),
                    "num_images": len(images),
                    "images": images,
                    "image_sizes": [image.size for image in images],
                }


def load_image(image_obj: dict, image_cls):
    if not isinstance(image_obj, dict):
        raise TypeError(f"Unsupported image field type: {type(image_obj).__name__}")
    if image_obj.get("bytes") is None:
        raise ValueError("ERQA image field does not contain bytes")
    return image_cls.open(io.BytesIO(image_obj["bytes"])).convert("RGB")


class ERQAEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        _ensure_repo_on_path()

        from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
        from wall_x._vendor.harrix.serving._wallx_infer.model_wrapper import (
            WallxModelWrapper,
        )
        from wall_x._vendor.harrix.utils.train_config import resolve_max_length

        cam_names = [image_key(i) for i in range(MAX_IMAGES)]
        config = InferConfig(
            checkpoint_path=args.checkpoint_path,
            train_config_path=args.train_config_path,
            model_device=args.device,
            norm_key=args.norm_key,
            cam_names=cam_names,
        )
        if args.max_length is not None:
            config.train_config["max_length"] = args.max_length
        self.wrapper = WallxModelWrapper(config)
        self.max_length = args.max_length or resolve_max_length(config.train_config)

    def predict_batch(self, samples: list[dict], prompt_style: str) -> list[dict]:
        import torch

        prompts = [
            build_prompt(sample["question"], sample["num_images"], prompt_style)
            for sample in samples
        ]
        observations = []
        for sample in samples:
            if sample["num_images"] > MAX_IMAGES:
                raise ValueError(
                    f"Sample {sample['question_id']} has {sample['num_images']} images; "
                    f"MAX_IMAGES={MAX_IMAGES}"
                )
            observations.append(
                {image_key(i): image for i, image in enumerate(sample["images"])}
            )

        model_input = self.wrapper.construct_model_input(
            observations,
            prompts,
            [""] * len(samples),
        )
        if self.max_length is not None:
            seq_len = model_input["input_ids"].shape[1]
            if seq_len > self.max_length:
                raise ValueError(
                    f"Prompt token length {seq_len} exceeds max_length {self.max_length}. "
                    "Increase --max-length or reduce image resolution in the train config."
                )

        with torch.inference_mode():
            model_output = self.wrapper.model.generate_text(**model_input)
        return [
            {
                "prediction": extract_final_answer(text),
                "normalized_prediction": normalize_choice(text),
                "raw_prediction": text,
            }
            for text in model_output["predict_output_text"]
        ]


def batched(items: Iterable[dict], batch_size: int) -> Iterable[list[dict]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.batch_size > 1:
        print("Warning: ERQA variable-image samples are easiest to debug with --batch-size 1.")

    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = ERQAEvaluator(args)

    total = 0
    correct = 0
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    by_num_images: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )

    def filtered_samples():
        seen = 0
        for sample in iter_erqa(dataset_root):
            if sample["index"] < args.start_index:
                continue
            if args.limit is not None and seen >= args.limit:
                break
            seen += 1
            yield sample
        if seen == 0:
            raise ValueError("No samples selected for evaluation")

    with output_path.open("w", encoding="utf-8") as f:
        for batch in batched(filtered_samples(), args.batch_size):
            predictions = evaluator.predict_batch(batch, args.prompt_style)
            for sample, pred in zip(batch, predictions):
                answer_norm = normalize_choice(sample["answer"])
                is_correct = pred["normalized_prediction"] == answer_norm
                total += 1
                correct += int(is_correct)
                by_type[sample["question_type"]]["total"] += 1
                by_type[sample["question_type"]]["correct"] += int(is_correct)
                num_key = str(sample["num_images"])
                by_num_images[num_key]["total"] += 1
                by_num_images[num_key]["correct"] += int(is_correct)

                record = {
                    "index": sample["index"],
                    "question_id": sample["question_id"],
                    "question_type": sample["question_type"],
                    "num_images": sample["num_images"],
                    "visual_indices": sample["visual_indices"],
                    "image_sizes": [list(size) for size in sample["image_sizes"]],
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "normalized_answer": answer_norm,
                    "prediction": pred["prediction"],
                    "normalized_prediction": pred["normalized_prediction"],
                    "raw_prediction": pred["raw_prediction"],
                    "prompt_style": args.prompt_style,
                    "exact": is_correct,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if total % 25 == 0:
                    print(
                        f"evaluated={total} accuracy={correct / total:.4f}",
                        flush=True,
                    )

    by_type_summary = {
        qtype: {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
        }
        for qtype, stats in sorted(by_type.items())
    }
    by_num_images_summary = {
        num_images: {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
        }
        for num_images, stats in sorted(by_num_images.items(), key=lambda kv: int(kv[0]))
    }
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_question_type": by_type_summary,
        "by_num_images": by_num_images_summary,
        "output_jsonl": str(output_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
