#!/usr/bin/env python3
"""Evaluate Wall-OSS-0.5 VLM multiple-choice accuracy on SAT."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_ROOT = "datasets/public_datasets/VLM/vqa/benchmarks/SAT"
MAX_IMAGES = 2

ROLE_START = "<|im_start|>"
ROLE_END = "<|im_end|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"


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
        help="SAT root containing SAT_test.parquet and other splits.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="sat_wall_oss_0_5_predictions.jsonl",
        help="Where to write per-eval-item predictions.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--norm-key", default="x2_normal")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=["test", "val", "train", "static"],
        default="test",
        help="SAT split to evaluate. Default: test.",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--prompt-style",
        choices=["numbered-images", "letter-only", "cot-answer"],
        default="numbered-images",
        help="How to wrap image placeholders and answer guidance.",
    )
    parser.add_argument(
        "--circular-eval",
        action="store_true",
        help=(
            "Evaluate each sample with the correct answer placed at every option "
            "position. Recommended by SAT for the small real-image test split."
        ),
    )
    return parser.parse_args()


def option_letter(index: int) -> str:
    return chr(ord("A") + index)


def option_letters(num_choices: int) -> str:
    return "".join(option_letter(i) for i in range(num_choices))


def image_key(index: int) -> str:
    return f"image_{index}"


def image_placeholder(index: int, num_images: int) -> str:
    placeholder = f"{VISION_START}{IMAGE_PAD}{VISION_END}"
    if num_images == 1:
        return f"Image: {placeholder}"
    if index == 0:
        return f"Image 1 (initial frame): {placeholder}"
    if index == 1:
        return f"Image 2 (second frame): {placeholder}"
    return f"Image {index + 1}: {placeholder}"


def format_options(options: list[str]) -> str:
    return "\n".join(f"({option_letter(i)}) {option}" for i, option in enumerate(options))


def build_prompt(sample: dict, prompt_style: str) -> str:
    image_lines = "\n".join(
        image_placeholder(i, sample["num_images"]) for i in range(sample["num_images"])
    )
    question_block = (
        f"{sample['question'].strip()}\n"
        "Select from the following choices.\n"
        f"{format_options(sample['answer_options'])}"
    )
    if prompt_style == "letter-only":
        prompt_text = f"{question_block}\nAnswer with only the option letter."
    elif prompt_style == "cot-answer":
        prompt_text = (
            "Read the image(s) and answer the following spatial reasoning "
            'multiple-choice question. The last line of your response should be '
            'of the form "ANSWER: [ANSWER]" where [ANSWER] is one option letter.\n\n'
            f"{question_block}\n\n"
            'Remember to put only the final answer on its own line as "ANSWER: [ANSWER]".'
        )
    else:
        prompt_text = question_block

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


def _normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_choice(text: str, options: list[str]) -> str:
    final = extract_final_answer(text).strip()
    upper = final.upper()
    valid = option_letters(len(options))

    explicit = re.search(
        rf"\b(?:ANSWER|ANS|OPTION|CHOICE)\b(?:\s+IS)?\s*[:\-]?\s*([{valid}])\b",
        upper,
    )
    if explicit:
        return explicit.group(1)

    compact = re.sub(r"[\s,.;:/\\|&()\[\]{}'\"`-]+", "", upper)
    if re.fullmatch(rf"[{valid}]", compact):
        return compact

    paren = re.search(rf"\(([{valid}])\)", upper)
    if paren:
        return paren.group(1)

    match = re.search(rf"(?<![A-Z])([{valid}])(?![A-Z])", upper)
    if match:
        return match.group(1)

    normalized_final = _normalize_text(final)
    for i, option in enumerate(options):
        normalized_option = _normalize_text(option)
        if normalized_option and normalized_option == normalized_final:
            return option_letter(i)
    for i, option in enumerate(options):
        normalized_option = _normalize_text(option)
        if normalized_option and re.search(rf"\b{re.escape(normalized_option)}\b", normalized_final):
            return option_letter(i)
    for i, option in enumerate(options):
        normalized_option = _normalize_text(option)
        if normalized_final and len(normalized_final) > 3 and normalized_final in normalized_option:
            return option_letter(i)
    return ""


def correct_answer_index(options: list[str], correct_answer: str) -> int:
    for index, option in enumerate(options):
        if str(option) == str(correct_answer):
            return index
    normalized_correct = _normalize_text(correct_answer)
    for index, option in enumerate(options):
        if _normalize_text(option) == normalized_correct:
            return index
    raise ValueError(f"Correct answer not found in options: {correct_answer!r}")


def make_circular_options(options: list[str], correct_answer: str, position: int) -> list[str]:
    remaining = [option for option in options if _normalize_text(option) != _normalize_text(correct_answer)]
    if len(remaining) != len(options) - 1:
        # Fall back to exact removal once if normalized text is ambiguous.
        remaining = list(options)
        remaining.pop(correct_answer_index(options, correct_answer))
    circular = list(remaining)
    circular.insert(position, correct_answer)
    return circular


def split_path(dataset_root: Path, split: str) -> Path:
    return dataset_root / f"SAT_{split}.parquet"


def iter_sat(dataset_root: Path, split: str) -> Iterable[dict]:
    import pyarrow.parquet as pq
    from PIL import Image

    parquet_file = split_path(dataset_root, split)
    if not parquet_file.exists():
        raise FileNotFoundError(f"SAT parquet not found: {parquet_file}")

    columns = ["image_bytes", "question", "answers", "question_type", "correct_answer"]
    pf = pq.ParquetFile(parquet_file)
    row_index = 0
    for batch in pf.iter_batches(batch_size=16, columns=columns):
        for row in batch.to_pylist():
            images = [
                Image.open(io.BytesIO(image_bytes)).convert("RGB")
                for image_bytes in row["image_bytes"]
            ]
            answers = list(row["answers"])
            answer_index = correct_answer_index(answers, row["correct_answer"])
            yield {
                "index": row_index,
                "question": row["question"],
                "answer_options": answers,
                "original_answer_options": answers,
                "answer": row["correct_answer"],
                "answer_index": answer_index,
                "normalized_answer": option_letter(answer_index),
                "question_type": row["question_type"],
                "images": images,
                "num_images": len(images),
                "image_sizes": [image.size for image in images],
                "circular_position": None,
            }
            row_index += 1


def expand_circular(sample: dict) -> list[dict]:
    variants = []
    for position in range(len(sample["answer_options"])):
        variant = dict(sample)
        variant["answer_options"] = make_circular_options(
            sample["original_answer_options"],
            sample["answer"],
            position,
        )
        variant["answer_index"] = position
        variant["normalized_answer"] = option_letter(position)
        variant["circular_position"] = position
        variants.append(variant)
    return variants


class SATEvaluator:
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

        prompts = [build_prompt(sample, prompt_style) for sample in samples]
        observations = []
        for sample in samples:
            if sample["num_images"] > MAX_IMAGES:
                raise ValueError(
                    f"Sample {sample['index']} has {sample['num_images']} images; "
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
                    "Increase --max-length or reduce the prompt."
                )

        with torch.inference_mode():
            model_output = self.wrapper.model.generate_text(**model_input)
        return [
            {
                "prediction": extract_final_answer(text),
                "normalized_prediction": normalize_choice(text, sample["answer_options"]),
                "raw_prediction": text,
            }
            for sample, text in zip(samples, model_output["predict_output_text"])
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


def accuracy_summary(stats: dict[str, int]) -> dict:
    return {
        "total": stats["total"],
        "correct": stats["correct"],
        "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = SATEvaluator(args)

    total = 0
    original_samples = 0
    correct = 0
    parse_failures = 0
    by_question_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    by_num_images: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )

    def filtered_eval_items():
        nonlocal original_samples
        seen = 0
        for sample in iter_sat(dataset_root, args.split):
            if sample["index"] < args.start_index:
                continue
            if args.limit is not None and seen >= args.limit:
                break
            seen += 1
            original_samples += 1
            if args.circular_eval:
                yield from expand_circular(sample)
            else:
                yield sample
        if seen == 0:
            raise ValueError("No samples selected for evaluation")

    with output_path.open("w", encoding="utf-8") as f:
        for batch in batched(filtered_eval_items(), args.batch_size):
            predictions = evaluator.predict_batch(batch, args.prompt_style)
            for sample, pred in zip(batch, predictions):
                is_correct = pred["normalized_prediction"] == sample["normalized_answer"]
                total += 1
                correct += int(is_correct)
                parse_failures += int(not pred["normalized_prediction"])
                by_question_type[sample["question_type"]]["total"] += 1
                by_question_type[sample["question_type"]]["correct"] += int(is_correct)
                num_key = str(sample["num_images"])
                by_num_images[num_key]["total"] += 1
                by_num_images[num_key]["correct"] += int(is_correct)

                record = {
                    "index": sample["index"],
                    "question_type": sample["question_type"],
                    "num_images": sample["num_images"],
                    "image_sizes": [list(size) for size in sample["image_sizes"]],
                    "question": sample["question"],
                    "answer_options": sample["answer_options"],
                    "original_answer_options": sample["original_answer_options"],
                    "answer": sample["answer"],
                    "answer_index": sample["answer_index"],
                    "normalized_answer": sample["normalized_answer"],
                    "circular_position": sample["circular_position"],
                    "prediction": pred["prediction"],
                    "normalized_prediction": pred["normalized_prediction"],
                    "raw_prediction": pred["raw_prediction"],
                    "prompt_style": args.prompt_style,
                    "correct": is_correct,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if total % 25 == 0:
                    print(
                        f"evaluated={total} accuracy={correct / total:.4f}",
                        flush=True,
                    )

    summary = {
        "total": total,
        "original_samples": original_samples,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / total if total else 0.0,
        "by_question_type": {
            key: accuracy_summary(stats)
            for key, stats in sorted(by_question_type.items())
        },
        "by_num_images": {
            key: accuracy_summary(stats)
            for key, stats in sorted(by_num_images.items(), key=lambda kv: int(kv[0]))
        },
        "split": args.split,
        "circular_eval": args.circular_eval,
        "prompt_style": args.prompt_style,
        "output_jsonl": str(output_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
