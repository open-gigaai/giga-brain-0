#!/usr/bin/env python3
"""Evaluate Wall-OSS-0.5 VLM multiple-choice accuracy on CV-Bench."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_ROOT = "datasets/public_datasets/VLM/vqa/CV-Bench"

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
        help="CV-Bench root containing test_2d.parquet and test_3d.parquet.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="cvbench_wall_oss_0_5_predictions.jsonl",
        help="Where to write per-sample predictions.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--norm-key", default="x2_normal")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=["all", "2d", "3d"],
        default="all",
        help="Evaluate both splits or only test_2d/test_3d.",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--prompt-style",
        choices=["dataset", "letter-only", "cot-answer"],
        default="dataset",
        help="dataset uses the parquet prompt; letter-only adds stricter answer guidance.",
    )
    return parser.parse_args()


def build_prompt(sample: dict, prompt_style: str) -> str:
    prompt = sample["prompt"].strip()
    if prompt_style == "letter-only":
        prompt_text = (
            f"{prompt}\n"
            "Answer with only the option letter, such as A, B, C, D, E, or F."
        )
    elif prompt_style == "cot-answer":
        prompt_text = (
            "Read the image and answer the following multiple-choice question. "
            'The last line of your response should be of the form "ANSWER: [ANSWER]" '
            "where [ANSWER] is one option letter.\n\n"
            f"{prompt}\n\n"
            'Remember to put only the final answer on its own line as "ANSWER: [ANSWER]".'
        )
    else:
        prompt_text = prompt

    return (
        f"{ROLE_START}system\nYou are a helpful assistant.{ROLE_END}\n"
        f"{ROLE_START}user\n"
        f"{VISION_START}{IMAGE_PAD}{VISION_END}\n"
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


def option_letters(num_choices: int) -> str:
    return "".join(chr(ord("A") + i) for i in range(num_choices))


def normalize_choice(text: str, choices: list[str]) -> str:
    final = extract_final_answer(text).upper().strip()
    valid = option_letters(len(choices))

    explicit = re.search(
        rf"\b(?:ANSWER|ANS|OPTION|CHOICE)\b(?:\s+IS)?\s*[:\-]?\s*([{valid}])\b",
        final,
    )
    if explicit:
        return explicit.group(1)

    compact = re.sub(r"[\s,.;:/\\|&()\[\]{}'\"`-]+", "", final)
    if re.fullmatch(rf"[{valid}]", compact):
        return compact

    paren = re.search(rf"\(([{valid}])\)", final)
    if paren:
        return paren.group(1)

    match = re.search(rf"(?<![A-Z])([{valid}])(?![A-Z])", final)
    if match:
        return match.group(1)

    # Fallback: if the model repeats an answer choice exactly, map it back.
    normalized_final = re.sub(r"\s+", " ", final.lower()).strip()
    for i, choice in enumerate(choices):
        normalized_choice = re.sub(r"\s+", " ", str(choice).lower()).strip()
        if normalized_choice and normalized_choice == normalized_final:
            return chr(ord("A") + i)
    for i, choice in enumerate(choices):
        normalized_choice = re.sub(r"\s+", " ", str(choice).lower()).strip()
        if normalized_choice and re.search(rf"\b{re.escape(normalized_choice)}\b", normalized_final):
            return chr(ord("A") + i)
    return ""


def normalize_answer(answer: str) -> str:
    match = re.search(r"\(([A-Z])\)", str(answer).upper())
    if match:
        return match.group(1)
    match = re.search(r"(?<![A-Z])([A-Z])(?![A-Z])", str(answer).upper())
    return match.group(1) if match else ""


def selected_files(dataset_root: Path, split: str) -> list[Path]:
    if split == "2d":
        return [dataset_root / "test_2d.parquet"]
    if split == "3d":
        return [dataset_root / "test_3d.parquet"]
    return [dataset_root / "test_2d.parquet", dataset_root / "test_3d.parquet"]


def iter_cvbench(dataset_root: Path, split: str) -> Iterable[dict]:
    import pyarrow.parquet as pq
    from PIL import Image

    columns = [
        "idx",
        "type",
        "task",
        "image",
        "question",
        "choices",
        "answer",
        "prompt",
        "filename",
        "source",
        "source_dataset",
        "source_filename",
        "target_class",
        "target_size",
        "bbox",
    ]
    for parquet_file in selected_files(dataset_root, split):
        if not parquet_file.exists():
            raise FileNotFoundError(f"CV-Bench parquet not found: {parquet_file}")
        pf = pq.ParquetFile(parquet_file)
        available = set(pf.schema_arrow.names)
        use_columns = [column for column in columns if column in available]
        for batch in pf.iter_batches(batch_size=32, columns=use_columns):
            for row in batch.to_pylist():
                image = load_image(row["image"], Image).convert("RGB")
                yield {
                    "index": int(row["idx"]),
                    "idx": int(row["idx"]),
                    "type": row["type"],
                    "task": row["task"],
                    "image": image,
                    "image_size": image.size,
                    "question": row["question"],
                    "choices": list(row["choices"]),
                    "answer": row["answer"],
                    "normalized_answer": normalize_answer(row["answer"]),
                    "prompt": row["prompt"],
                    "filename": row.get("filename"),
                    "source": row["source"],
                    "source_dataset": row.get("source_dataset"),
                    "source_filename": row.get("source_filename"),
                    "target_class": row.get("target_class"),
                    "target_size": row.get("target_size"),
                    "bbox": row.get("bbox"),
                }


def load_image(image_obj: dict, image_cls):
    if not isinstance(image_obj, dict):
        raise TypeError(f"Unsupported image field type: {type(image_obj).__name__}")
    if image_obj.get("bytes") is None:
        raise ValueError("CV-Bench image field does not contain bytes")
    return image_cls.open(io.BytesIO(image_obj["bytes"]))


class CVBenchEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        _ensure_repo_on_path()

        from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
        from wall_x._vendor.harrix.serving._wallx_infer.model_wrapper import (
            WallxModelWrapper,
        )
        from wall_x._vendor.harrix.utils.train_config import resolve_max_length

        self.image_key = "multi_modal"
        config = InferConfig(
            checkpoint_path=args.checkpoint_path,
            train_config_path=args.train_config_path,
            model_device=args.device,
            norm_key=args.norm_key,
            cam_names=[self.image_key],
        )
        if args.max_length is not None:
            config.train_config["max_length"] = args.max_length
        self.wrapper = WallxModelWrapper(config)
        self.max_length = args.max_length or resolve_max_length(config.train_config)

    def predict_batch(self, samples: list[dict], prompt_style: str) -> list[dict]:
        import torch

        prompts = [build_prompt(sample, prompt_style) for sample in samples]
        observations = [{self.image_key: sample["image"]} for sample in samples]
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
                "normalized_prediction": normalize_choice(text, sample["choices"]),
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


def compute_cvbench_accuracy(by_source: dict[str, dict[str, int]]) -> dict:
    source_acc = {
        source: stats["correct"] / stats["total"] if stats["total"] else 0.0
        for source, stats in by_source.items()
    }
    acc_2d_ade = source_acc.get("ADE20K", 0.0)
    acc_2d_coco = source_acc.get("COCO", 0.0)
    acc_3d_omni = source_acc.get("Omni3D", 0.0)
    has_2d = "ADE20K" in source_acc or "COCO" in source_acc
    has_3d = "Omni3D" in source_acc
    acc_2d = (acc_2d_ade + acc_2d_coco) / 2 if has_2d else 0.0
    acc_3d = acc_3d_omni if has_3d else 0.0
    if has_2d and has_3d:
        combined = (acc_2d + acc_3d) / 2
    elif has_2d:
        combined = acc_2d
    else:
        combined = acc_3d
    return {
        "cv_bench_accuracy": combined,
        "accuracy_2d": acc_2d,
        "accuracy_3d": acc_3d,
        "accuracy_2d_ade": acc_2d_ade,
        "accuracy_2d_coco": acc_2d_coco,
        "accuracy_3d_omni": acc_3d_omni,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = CVBenchEvaluator(args)

    total = 0
    correct = 0
    parse_failures = 0
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    def filtered_samples():
        seen = 0
        for sample in iter_cvbench(dataset_root, args.split):
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
                is_correct = pred["normalized_prediction"] == sample["normalized_answer"]
                total += 1
                correct += int(is_correct)
                parse_failures += int(not pred["normalized_prediction"])

                by_type[sample["type"]]["total"] += 1
                by_type[sample["type"]]["correct"] += int(is_correct)
                by_task[sample["task"]]["total"] += 1
                by_task[sample["task"]]["correct"] += int(is_correct)
                by_source[sample["source"]]["total"] += 1
                by_source[sample["source"]]["correct"] += int(is_correct)

                record = {
                    "index": sample["index"],
                    "idx": sample["idx"],
                    "type": sample["type"],
                    "task": sample["task"],
                    "source": sample["source"],
                    "source_dataset": sample["source_dataset"],
                    "filename": sample["filename"],
                    "source_filename": sample["source_filename"],
                    "image_size": list(sample["image_size"]),
                    "question": sample["question"],
                    "choices": sample["choices"],
                    "answer": sample["answer"],
                    "normalized_answer": sample["normalized_answer"],
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

    by_type_summary = {
        key: accuracy_summary(stats) for key, stats in sorted(by_type.items())
    }
    by_task_summary = {
        key: accuracy_summary(stats) for key, stats in sorted(by_task.items())
    }
    by_source_summary = {
        key: accuracy_summary(stats) for key, stats in sorted(by_source.items())
    }
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / total if total else 0.0,
        **compute_cvbench_accuracy(by_source),
        "by_type": by_type_summary,
        "by_task": by_task_summary,
        "by_source": by_source_summary,
        "split": args.split,
        "prompt_style": args.prompt_style,
        "output_jsonl": str(output_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
