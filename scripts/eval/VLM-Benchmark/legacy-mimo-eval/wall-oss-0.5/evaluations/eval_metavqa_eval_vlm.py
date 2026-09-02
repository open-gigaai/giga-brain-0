#!/usr/bin/env python3
"""Evaluate Wall-OSS-0.5 VLM on MetaVQA-Eval.

MetaVQA-Eval contains single-image embodied/spatial VQA questions. Most samples
are multiple-choice with letter answers; a small open-ended subset has no
choices and can be included with relaxed text matching.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_ROOT = "datasets/public_datasets/VLM/vqa/MetaVQA-Eval"

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
        help="MetaVQA-Eval root containing test.jsonl and obs/*.png.",
    )
    parser.add_argument(
        "--data-file",
        default="test.jsonl",
        help="JSONL or JSON file under dataset root. Default: test.jsonl.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="metavqa_eval_wall_oss_0_5_predictions.jsonl",
        help="Where to write per-sample predictions.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--norm-key", default="x2_normal")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--domain",
        default="all",
        help="Domain filter: all, real, sim, or a comma-separated subset.",
    )
    parser.add_argument(
        "--type",
        default="all",
        help="Question type filter: all or a comma-separated subset.",
    )
    parser.add_argument(
        "--include-open",
        action="store_true",
        help="Include open-ended samples with no options. Default skips them.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["dataset", "letter-only", "cot-answer"],
        default="letter-only",
        help="dataset uses the question; letter-only adds stricter answer guidance.",
    )
    return parser.parse_args()


def split_filter(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    parts = {part.strip() for part in value.split(",") if part.strip()}
    if not parts:
        raise ValueError("Filter must contain at least one value or 'all'")
    return parts


def load_rows(path: Path) -> Iterable[dict]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in sorted(data.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
                row = dict(data[key])
                row.setdefault("question_id", key)
                yield row
        elif isinstance(data, list):
            for index, row in enumerate(data):
                row = dict(row)
                row.setdefault("question_id", str(index))
                yield row
        else:
            raise TypeError(f"Unsupported JSON root type: {type(data).__name__}")


def parse_options(options) -> dict[str, str]:
    if not options:
        return {}
    if isinstance(options, str):
        options = json.loads(options)
    if isinstance(options, dict):
        return {str(k).upper(): str(v) for k, v in options.items()}
    if isinstance(options, list):
        return {chr(ord("A") + i): str(v) for i, v in enumerate(options)}
    raise TypeError(f"Unsupported options type: {type(options).__name__}")


def option_letters(options: dict[str, str]) -> str:
    return "".join(sorted(options.keys()))


def options_list(options: dict[str, str]) -> list[str]:
    return [options[key] for key in sorted(options.keys())]


def resolve_obs(obs, dataset_root: Path) -> Path:
    if isinstance(obs, list):
        if len(obs) != 1:
            raise ValueError(f"Expected one observation image, got {len(obs)}")
        obs = obs[0]
    path = Path(str(obs))
    if not path.is_absolute():
        path = dataset_root / path
    if not path.exists():
        raise FileNotFoundError(f"MetaVQA-Eval image not found: {path}")
    return path


def format_options(options: dict[str, str]) -> str:
    return "; ".join(f"({key}) {value}" for key, value in sorted(options.items()))


def build_prompt(sample: dict, prompt_style: str) -> str:
    question = str(sample["question"]).strip()
    options = sample["options"]
    if options:
        question_has_options = all(f"({key})" in question for key in options.keys())
        if prompt_style == "letter-only":
            if question_has_options:
                prompt_text = (
                    f"{question}\n"
                    f"Answer with only one option letter from {option_letters(options)}."
                )
            else:
                prompt_text = (
                    f"{question}\n"
                    f"Options: {format_options(options)}\n"
                    f"Answer with only one option letter from {option_letters(options)}."
                )
        elif prompt_style == "cot-answer":
            option_block = "" if question_has_options else f"\nOptions: {format_options(options)}"
            prompt_text = (
                "Read the image and answer the following multiple-choice question. "
                'The last line of your response should be of the form "ANSWER: [ANSWER]" '
                "where [ANSWER] is one option letter.\n\n"
                f"{question}{option_block}\n\n"
                'Remember to put only the final answer on its own line as "ANSWER: [ANSWER]".'
            )
        else:
            prompt_text = question if question_has_options else f"{question}\nOptions: {format_options(options)}"
    else:
        prompt_text = question

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
    text = re.sub(r"[^a-z0-9<>]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_choice(text: str, options: dict[str, str]) -> str:
    final = extract_final_answer(text).strip()
    upper = final.upper()
    valid = option_letters(options)
    if not valid:
        return ""

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
    for key, choice in sorted(options.items()):
        normalized_choice = normalize_text(choice)
        if normalized_choice and normalized_choice == normalized_final:
            return key
    for key, choice in sorted(options.items()):
        normalized_choice = normalize_text(choice)
        if normalized_choice and re.search(rf"\b{re.escape(normalized_choice)}\b", normalized_final):
            return key
    return ""


def relaxed_match(prediction: str, answer: str) -> bool:
    pred = normalize_text(extract_final_answer(prediction))
    ans = normalize_text(answer)
    if not ans:
        return False
    return pred == ans or ans in pred


def iter_metavqa_eval(
    dataset_root: Path,
    data_file: str,
    domain_filter: set[str] | None,
    type_filter: set[str] | None,
    include_open: bool,
) -> Iterable[dict]:
    from PIL import Image

    path = dataset_root / data_file
    if not path.exists():
        raise FileNotFoundError(f"MetaVQA-Eval data file not found: {path}")

    for index, row in enumerate(load_rows(path)):
        domain = str(row.get("domain", ""))
        qtype = str(row.get("type", ""))
        if domain_filter is not None and domain not in domain_filter:
            continue
        if type_filter is not None and qtype not in type_filter:
            continue

        options = parse_options(row.get("options"))
        answer = str(row.get("answer", "")).strip().upper()
        is_open = not options or not answer
        if is_open and not include_open:
            continue

        obs_path = resolve_obs(row["obs"], dataset_root)
        image = Image.open(obs_path).convert("RGB")
        yield {
            "index": index,
            "question_id": str(row.get("question_id", index)),
            "question": row["question"],
            "answer": answer,
            "raw_answer": row.get("answer", ""),
            "options": options,
            "type": qtype,
            "domain": domain,
            "obs": row["obs"],
            "image_path": str(obs_path),
            "image": image,
            "image_size": image.size,
            "is_open": is_open,
        }


class MetaVQAEvalEvaluator:
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
            normalized_prediction = normalize_choice(text, sample["options"])
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
    return {
        "total": 0,
        "multiple_choice_total": 0,
        "multiple_choice_correct": 0,
        "parse_failures": 0,
        "open_total": 0,
        "open_exact_correct": 0,
        "open_relaxed_correct": 0,
    }


def update_stats(stats: dict[str, int], sample: dict, pred: dict) -> None:
    stats["total"] += 1
    if sample["is_open"]:
        final = normalize_text(extract_final_answer(pred["raw_prediction"]))
        answer = normalize_text(sample["raw_answer"])
        stats["open_total"] += 1
        stats["open_exact_correct"] += int(bool(answer) and final == answer)
        stats["open_relaxed_correct"] += int(relaxed_match(pred["raw_prediction"], sample["raw_answer"]))
    else:
        correct = pred["normalized_prediction"] == sample["answer"]
        stats["multiple_choice_total"] += 1
        stats["multiple_choice_correct"] += int(correct)
        stats["parse_failures"] += int(not pred["normalized_prediction"])


def summarize_stats(stats: dict[str, int]) -> dict:
    mc_total = stats["multiple_choice_total"]
    open_total = stats["open_total"]
    return {
        "total": stats["total"],
        "multiple_choice_total": mc_total,
        "multiple_choice_correct": stats["multiple_choice_correct"],
        "accuracy": stats["multiple_choice_correct"] / mc_total if mc_total else 0.0,
        "parse_failures": stats["parse_failures"],
        "parse_failure_rate": stats["parse_failures"] / mc_total if mc_total else 0.0,
        "open_total": open_total,
        "open_exact_correct": stats["open_exact_correct"],
        "open_exact_accuracy": (
            stats["open_exact_correct"] / open_total if open_total else 0.0
        ),
        "open_relaxed_correct": stats["open_relaxed_correct"],
        "open_relaxed_accuracy": (
            stats["open_relaxed_correct"] / open_total if open_total else 0.0
        ),
    }


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    domain_filter = split_filter(args.domain)
    type_filter = split_filter(args.type)

    evaluator = MetaVQAEvalEvaluator(args)

    overall = empty_stats()
    by_domain: dict[str, dict[str, int]] = defaultdict(empty_stats)
    by_type: dict[str, dict[str, int]] = defaultdict(empty_stats)
    by_domain_type: dict[str, dict[str, int]] = defaultdict(empty_stats)

    def filtered_samples():
        seen = 0
        for sample in iter_metavqa_eval(
            dataset_root,
            args.data_file,
            domain_filter,
            type_filter,
            args.include_open,
        ):
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
                update_stats(overall, sample, pred)
                update_stats(by_domain[sample["domain"]], sample, pred)
                update_stats(by_type[sample["type"]], sample, pred)
                update_stats(by_domain_type[f"{sample['domain']}/{sample['type']}"], sample, pred)

                correct = (
                    relaxed_match(pred["raw_prediction"], sample["raw_answer"])
                    if sample["is_open"]
                    else pred["normalized_prediction"] == sample["answer"]
                )
                record = {
                    "index": sample["index"],
                    "question_id": sample["question_id"],
                    "domain": sample["domain"],
                    "type": sample["type"],
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "raw_answer": sample["raw_answer"],
                    "options": sample["options"],
                    "obs": sample["obs"],
                    "image_path": sample["image_path"],
                    "image_size": list(sample["image_size"]),
                    "prediction": pred["prediction"],
                    "raw_prediction": pred["raw_prediction"],
                    "normalized_prediction": pred["normalized_prediction"],
                    "correct": correct,
                    "is_open": sample["is_open"],
                    "prompt_style": args.prompt_style,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                mc_total = overall["multiple_choice_total"]
                if overall["total"] % 100 == 0:
                    print(
                        f"evaluated={overall['total']} "
                        f"accuracy={(overall['multiple_choice_correct'] / mc_total) if mc_total else 0.0:.4f}",
                        flush=True,
                    )

    summary = {
        **summarize_stats(overall),
        "by_domain": {
            key: summarize_stats(stats) for key, stats in sorted(by_domain.items())
        },
        "by_type": {
            key: summarize_stats(stats) for key, stats in sorted(by_type.items())
        },
        "by_domain_type": {
            key: summarize_stats(stats)
            for key, stats in sorted(by_domain_type.items())
        },
        "data_file": args.data_file,
        "domain_filter": sorted(domain_filter) if domain_filter else "all",
        "type_filter": sorted(type_filter) if type_filter else "all",
        "include_open": args.include_open,
        "prompt_style": args.prompt_style,
        "output_jsonl": str(output_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
