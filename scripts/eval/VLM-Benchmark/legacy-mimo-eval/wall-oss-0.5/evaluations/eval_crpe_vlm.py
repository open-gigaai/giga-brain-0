#!/usr/bin/env python3
"""Evaluate Wall-OSS-0.5 VLM multiple-choice accuracy on CRPE.

CRPE is a circular-evaluation benchmark for object existence and relation
comprehension. The official JSONL files already contain circularly shifted
queries, so this script reports both per-query single accuracy and grouped
circular accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_ROOT = "datasets/public_datasets/VLM/vqa/CRPE"
DEFAULT_COCO_ROOT = "datasets/public_datasets/MSCOCO"
OFFICIAL_FILES = ("crpe_exist.jsonl", "crpe_relation.jsonl")
META_FILES = ("crpe_exist_meta.jsonl", "crpe_relation_meta.jsonl")

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
        help="CRPE root containing crpe_exist.jsonl and crpe_relation.jsonl.",
    )
    parser.add_argument(
        "--coco-root",
        default=DEFAULT_COCO_ROOT,
        help="COCO root used to resolve paths like coco/val2017/*.jpg.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="crpe_wall_oss_0_5_predictions.jsonl",
        help="Where to write per-query predictions.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--norm-key", default="x2_normal")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--files",
        default="official",
        help=(
            "official uses crpe_exist.jsonl,crpe_relation.jsonl; meta uses "
            "non-circular meta files; or pass a comma-separated file list."
        ),
    )
    parser.add_argument(
        "--prompt-style",
        choices=["dataset", "letter-only", "cot-answer"],
        default="dataset",
        help="dataset uses the JSONL text; letter-only adds stricter answer guidance.",
    )
    return parser.parse_args()


def selected_files(value: str) -> list[str]:
    value = value.strip()
    lower = value.lower()
    if lower == "official":
        return list(OFFICIAL_FILES)
    if lower == "meta":
        return list(META_FILES)
    files = [part.strip() for part in value.split(",") if part.strip()]
    if not files:
        raise ValueError("--files must select at least one JSONL file")
    return files


def option_letters(num_choices: int) -> str:
    return "".join(chr(ord("A") + i) for i in range(num_choices))


def parse_options(text: str) -> list[tuple[str, str]]:
    return re.findall(r"(?m)^([A-Z])\.\s*(.+?)\s*$", text)


def correct_answer_from_meta(correct_option: str, choices: list[str]) -> str:
    correct = str(correct_option).strip()
    if re.fullmatch(r"[A-Z]", correct):
        return correct
    normalized_correct = normalize_text(correct)
    for i, choice in enumerate(choices):
        if normalize_text(choice) == normalized_correct:
            return chr(ord("A") + i)
    raise ValueError(f"Correct option not found in choices: {correct_option!r}")


def format_options(choices: list[str]) -> str:
    return "\n".join(f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices))


def build_text_from_meta(row: dict) -> tuple[str, str, list[str]]:
    question = str(row["question"]).strip()
    choices = [str(choice).strip() for choice in row["choices"]]
    correct_answer = correct_answer_from_meta(row["correct_option"], choices)
    text = (
        f"{question}\n"
        f"{format_options(choices)}\n"
        "Answer with the option's letter from the given choices directly."
    )
    return text, correct_answer, choices


def build_prompt(sample: dict, prompt_style: str) -> str:
    text = sample["text"].strip()
    if prompt_style == "letter-only":
        prompt_text = f"{text}\nAnswer with only one letter: A, B, C, or D."
    elif prompt_style == "cot-answer":
        prompt_text = (
            "Read the image and answer the following single-choice question. "
            'The last line of your response should be of the form "ANSWER: [ANSWER]" '
            "where [ANSWER] is one option letter.\n\n"
            f"{text}\n\n"
            'Remember to put only the final answer on its own line as "ANSWER: [ANSWER]".'
        )
    else:
        prompt_text = text

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


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_choice(text: str, choices: list[str]) -> str:
    final = extract_final_answer(text).strip()
    upper = final.upper()
    valid = option_letters(len(choices))

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

    normalized_final = normalize_text(final)
    for i, choice in enumerate(choices):
        normalized_choice = normalize_text(choice)
        if normalized_choice and normalized_choice == normalized_final:
            return chr(ord("A") + i)
    for i, choice in enumerate(choices):
        normalized_choice = normalize_text(choice)
        if normalized_choice and re.search(rf"\b{re.escape(normalized_choice)}\b", normalized_final):
            return chr(ord("A") + i)
    return ""


def resolve_image_path(image_ref: str, dataset_root: Path, coco_root: Path) -> Path:
    ref = Path(image_ref)
    candidates = []
    if ref.is_absolute():
        candidates.append(ref)
    else:
        candidates.extend(
            [
                dataset_root / ref,
                dataset_root / "abnormal_images" / ref.name,
                coco_root / ref,
                coco_root / Path(*ref.parts[1:]) if ref.parts and ref.parts[0].lower() == "coco" else coco_root / ref,
                coco_root.parent / ref,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve CRPE image {image_ref!r}. Tried:\n{tried}")


def circular_group_id(file_name: str, row_number: int, num_choices: int) -> str:
    return f"{file_name}:{row_number // num_choices}"


def iter_crpe(
    dataset_root: Path,
    coco_root: Path,
    files: list[str],
) -> Iterable[dict]:
    from PIL import Image

    index = 0
    for file_name in files:
        path = dataset_root / file_name
        if not path.exists():
            raise FileNotFoundError(f"CRPE JSONL not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            for row_number, line in enumerate(f):
                row = json.loads(line)
                is_meta = "choices" in row and "question" in row
                if is_meta:
                    text, correct_answer, choices = build_text_from_meta(row)
                    question_id = row_number
                    circular_group = f"{file_name}:{row_number}"
                else:
                    text = str(row["text"])
                    options = parse_options(text)
                    choices = [option for _, option in options]
                    correct_answer = str(row["correct_option"]).strip().upper()
                    question_id = int(row.get("question_id", row_number))
                    circular_group = circular_group_id(file_name, row_number, len(choices))
                if not choices:
                    raise ValueError(f"No choices parsed for {file_name}:{row_number}")

                image_path = resolve_image_path(row["image"], dataset_root, coco_root)
                image = Image.open(image_path).convert("RGB")
                yield {
                    "index": index,
                    "file": file_name,
                    "row_number": row_number,
                    "question_id": question_id,
                    "circular_group_id": circular_group,
                    "image_ref": row["image"],
                    "image_path": str(image_path),
                    "image": image,
                    "image_size": image.size,
                    "text": text,
                    "choices": choices,
                    "correct_answer": correct_answer,
                    "category": row["category"],
                    "is_meta": is_meta,
                }
                index += 1


class CRPEEvaluator:
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

        predictions = []
        for sample, text in zip(samples, model_output["predict_output_text"]):
            normalized_prediction = normalize_choice(text, sample["choices"])
            predictions.append(
                {
                    "raw_prediction": text,
                    "prediction": clean_prediction(text),
                    "normalized_prediction": normalized_prediction,
                }
            )
        return predictions


def batched(items: Iterable[dict], batch_size: int) -> Iterable[list[dict]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def empty_stats() -> dict[str, int]:
    return {"total": 0, "correct": 0, "parse_failures": 0}


def update_stats(stats: dict[str, int], correct: bool, parsed: bool) -> None:
    stats["total"] += 1
    stats["correct"] += int(correct)
    stats["parse_failures"] += int(not parsed)


def summarize_stats(stats: dict[str, int]) -> dict:
    total = stats["total"]
    return {
        "total": total,
        "correct": stats["correct"],
        "accuracy": stats["correct"] / total if total else 0.0,
        "parse_failures": stats["parse_failures"],
        "parse_failure_rate": stats["parse_failures"] / total if total else 0.0,
    }


def summarize_circular(groups: dict[str, list[dict]]) -> dict:
    total = len(groups)
    correct = 0
    incomplete = 0
    for records in groups.values():
        if len(records) != len(records[0]["choices"]):
            incomplete += 1
        correct += int(all(record["correct"] for record in records))
    return {
        "total_groups": total,
        "circular_correct": correct,
        "circular_accuracy": correct / total if total else 0.0,
        "incomplete_groups": incomplete,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    files = selected_files(args.files)
    dataset_root = Path(args.dataset_root)
    coco_root = Path(args.coco_root)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = CRPEEvaluator(args)

    overall = empty_stats()
    by_file: dict[str, dict[str, int]] = defaultdict(empty_stats)
    by_category: dict[str, dict[str, int]] = defaultdict(empty_stats)
    by_file_category: dict[str, dict[str, int]] = defaultdict(empty_stats)
    circular_groups: dict[str, list[dict]] = defaultdict(list)

    def filtered_samples():
        seen = 0
        for sample in iter_crpe(dataset_root, coco_root, files):
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
                correct = pred["normalized_prediction"] == sample["correct_answer"]
                parsed = bool(pred["normalized_prediction"])

                update_stats(overall, correct, parsed)
                update_stats(by_file[sample["file"]], correct, parsed)
                update_stats(by_category[sample["category"]], correct, parsed)
                update_stats(
                    by_file_category[f"{sample['file']}/{sample['category']}"],
                    correct,
                    parsed,
                )

                record = {
                    "index": sample["index"],
                    "file": sample["file"],
                    "row_number": sample["row_number"],
                    "question_id": sample["question_id"],
                    "circular_group_id": sample["circular_group_id"],
                    "category": sample["category"],
                    "image_ref": sample["image_ref"],
                    "image_path": sample["image_path"],
                    "image_size": list(sample["image_size"]),
                    "text": sample["text"],
                    "choices": sample["choices"],
                    "correct_answer": sample["correct_answer"],
                    "prediction": pred["prediction"],
                    "raw_prediction": pred["raw_prediction"],
                    "normalized_prediction": pred["normalized_prediction"],
                    "correct": correct,
                    "prompt_style": args.prompt_style,
                    "is_meta": sample["is_meta"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                circular_groups[sample["circular_group_id"]].append(record)

                if overall["total"] % 100 == 0:
                    print(
                        f"evaluated={overall['total']} "
                        f"single_accuracy={overall['correct'] / overall['total']:.4f}",
                        flush=True,
                    )

    summary = {
        "single": summarize_stats(overall),
        "circular": summarize_circular(circular_groups),
        "by_file": {
            key: summarize_stats(stats) for key, stats in sorted(by_file.items())
        },
        "by_category": {
            key: summarize_stats(stats) for key, stats in sorted(by_category.items())
        },
        "by_file_category": {
            key: summarize_stats(stats)
            for key, stats in sorted(by_file_category.items())
        },
        "files": files,
        "prompt_style": args.prompt_style,
        "coco_root": str(coco_root),
        "output_jsonl": str(output_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
