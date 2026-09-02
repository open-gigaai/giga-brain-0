#!/usr/bin/env python3
"""Evaluate Wall-OSS-0.5 VLM on RoboSpatial-Home.

RoboSpatial-Home is a mixed spatial benchmark:

* compatibility/configuration: yes/no visual QA.
* context: point localization scored against the provided valid-space mask.

The script reports category-specific metrics and a convenience mixed accuracy.
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

import numpy as np


DEFAULT_DATASET_ROOT = "datasets/public_datasets/VLM/vqa/RoboSpatial-Home"
TEXT_CATEGORIES = {"compatibility", "configuration"}
POINT_CATEGORIES = {"context"}
ALL_CATEGORIES = ("compatibility", "configuration", "context")

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
        help="RoboSpatial-Home root containing data/*.parquet.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="robospatial_home_wall_oss_0_5_predictions.jsonl",
        help="Where to write per-sample predictions.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--norm-key", default="x2_normal")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--category",
        default="all",
        help=(
            "Category to evaluate: all, compatibility, configuration, context, "
            "or a comma-separated subset."
        ),
    )
    parser.add_argument(
        "--image-source",
        choices=["rgb", "depth", "rgb-depth"],
        default="rgb",
        help=(
            "Images passed to the VLM. rgb matches the usual VLM benchmark "
            "setting; rgb-depth passes both RGB and depth images."
        ),
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=128,
        help="Mask pixels greater than or equal to this value are valid.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["dataset", "task-specific", "plain"],
        default="task-specific",
        help=(
            "dataset/plain use the dataset question; task-specific adds stricter "
            "Yes/No or normalized-point output constraints."
        ),
    )
    parser.add_argument(
        "--coord-mode",
        choices=["auto", "normalized", "pixel"],
        default="auto",
        help=(
            "How to interpret parsed context points. auto treats values in "
            "[0, 1.5] as normalized and larger values as pixels."
        ),
    )
    parser.add_argument(
        "--acc-k",
        default="1,3,5",
        help="Comma-separated K values for context Point Acc@K. Default: 1,3,5.",
    )
    return parser.parse_args()


def parse_categories(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(ALL_CATEGORIES)
    categories = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = sorted(set(categories) - set(ALL_CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown RoboSpatial-Home categories: {unknown}")
    if not categories:
        raise ValueError("--category must select at least one category")
    return categories


def parse_acc_ks(value: str) -> list[int]:
    ks = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k < 1:
            raise ValueError("--acc-k values must be positive integers")
        ks.append(k)
    if not ks:
        raise ValueError("--acc-k must contain at least one K value")
    return sorted(dict.fromkeys(ks))


def image_placeholder(index: int, image_source: str) -> str:
    placeholder = f"{VISION_START}{IMAGE_PAD}{VISION_END}"
    if image_source == "rgb-depth":
        label = "RGB image" if index == 0 else "Depth image"
        return f"{label}: {placeholder}"
    if image_source == "depth":
        return f"Depth image: {placeholder}"
    return placeholder


def image_block(image_source: str) -> str:
    if image_source == "rgb-depth":
        return "\n".join(image_placeholder(i, image_source) for i in range(2))
    return image_placeholder(0, image_source)


def build_task_prompt(sample: dict, prompt_style: str) -> str:
    question = str(sample["question"]).strip()
    category = sample["category"]

    if prompt_style in {"dataset", "plain"}:
        return question

    if category in TEXT_CATEGORIES:
        return (
            f"{question}\n"
            "Answer only Yes or No. Do not include any other text."
        )

    if category in POINT_CATEGORIES:
        return (
            f"{question}\n"
            "Return 1 to 5 points inside the requested vacant space. Use "
            "normalized coordinates between 0 and 1, where x is horizontal "
            "from left to right and y is vertical from top to bottom. Answer "
            "only as <answer><point>[(x1, y1), (x2, y2), ...]</point></answer>."
        )

    return question


def build_prompt(sample: dict, prompt_style: str, image_source: str) -> str:
    prompt_text = build_task_prompt(sample, prompt_style)
    return (
        f"{ROLE_START}system\nYou are a helpful assistant.{ROLE_END}\n"
        f"{ROLE_START}user\n"
        f"{image_block(image_source)}\n"
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


def normalize_yes_no(text: str) -> str:
    final = extract_final_answer(text).strip().lower()
    compact = re.sub(r"[^a-z]+", "", final)
    if compact in {"yes", "y"}:
        return "yes"
    if compact in {"no", "n"}:
        return "no"

    match = re.search(r"\b(yes|no)\b", final)
    if match:
        return match.group(1)

    short = re.search(r"^\s*([yn])\b", final)
    if short:
        return "yes" if short.group(1) == "y" else "no"
    return ""


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def normalize_point(
    point: tuple[float, float],
    image_size: tuple[int, int],
    coord_mode: str,
) -> tuple[int, int]:
    width, height = image_size
    x, y = point
    if coord_mode == "normalized" or (
        coord_mode == "auto" and max(abs(x), abs(y)) <= 1.5
    ):
        x *= width
        y *= height
    return (round(x), round(y))


def extract_points(
    text: str,
    image_size: tuple[int, int],
    coord_mode: str,
) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    cleaned = clean_prediction(text)
    raw_points: list[tuple[float, float]] = []

    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", cleaned, re.I | re.S)
    search_text = answer_match.group(1) if answer_match else cleaned

    tag_pattern = re.compile(
        r"<(?:point|points)>\s*(.*?)\s*</(?:point|points)>",
        re.I | re.S,
    )
    for match in tag_pattern.finditer(search_text):
        nums = _numbers(match.group(1))
        for i in range(0, len(nums) - 1, 2):
            raw_points.append((nums[i], nums[i + 1]))

    if not raw_points:
        pair_pattern = re.compile(
            r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]\)]"
        )
        for x, y in pair_pattern.findall(search_text):
            raw_points.append((float(x), float(y)))

    if not raw_points:
        xy_pattern = re.compile(
            r"\bx\s*=\s*(-?\d+(?:\.\d+)?)\D{0,20}\by\s*=\s*(-?\d+(?:\.\d+)?)",
            re.I,
        )
        for x, y in xy_pattern.findall(search_text):
            raw_points.append((float(x), float(y)))

    points = [
        normalize_point(point, image_size, coord_mode) for point in raw_points
    ]
    return raw_points, points


def point_hits_mask(point: tuple[int, int], valid_mask: np.ndarray) -> bool:
    x, y = point
    h, w = valid_mask.shape
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return bool(valid_mask[y, x])


def mask_bbox(valid_mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(valid_mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def acc_at_k(point_hits: list[bool], k: int) -> bool:
    return any(point_hits[:k])


def parquet_path(dataset_root: Path, category: str) -> Path:
    return dataset_root / "data" / f"{category}-00000-of-00001.parquet"


def load_image(image_obj: dict | None, image_cls, dataset_root: Path, field_name: str):
    if image_obj is None:
        return None
    if not isinstance(image_obj, dict):
        raise TypeError(f"Unsupported {field_name} field type: {type(image_obj).__name__}")
    if image_obj.get("bytes") is not None:
        return image_cls.open(io.BytesIO(image_obj["bytes"]))

    image_path = image_obj.get("path")
    if image_path:
        path = Path(image_path)
        candidates = [path] if path.is_absolute() else [dataset_root / path, dataset_root / "data" / path]
        for candidate in candidates:
            if candidate.exists():
                return image_cls.open(candidate)
    raise ValueError(f"RoboSpatial-Home {field_name} field does not contain loadable image data")


def image_field_path(image_obj: dict | None) -> str | None:
    if isinstance(image_obj, dict):
        return image_obj.get("path")
    return None


def iter_robospatial_home(
    dataset_root: Path,
    categories: list[str],
    image_source: str,
    mask_threshold: int,
) -> Iterable[dict]:
    import pyarrow.parquet as pq
    from PIL import Image

    row_index = 0
    for category in categories:
        parquet_file = parquet_path(dataset_root, category)
        if not parquet_file.exists():
            raise FileNotFoundError(f"RoboSpatial-Home parquet not found: {parquet_file}")

        columns = ["category", "question", "answer", "img", "depth_image", "mask"]
        pf = pq.ParquetFile(parquet_file)
        available = set(pf.schema_arrow.names)
        use_columns = [column for column in columns if column in available]
        file_row_index = 0
        for batch in pf.iter_batches(batch_size=32, columns=use_columns):
            for row in batch.to_pylist():
                rgb_image = load_image(row["img"], Image, dataset_root, "img").convert("RGB")
                depth_image = None
                if image_source in {"depth", "rgb-depth"}:
                    depth_image = load_image(
                        row.get("depth_image"),
                        Image,
                        dataset_root,
                        "depth_image",
                    ).convert("RGB")

                model_images = [rgb_image]
                if image_source == "depth":
                    model_images = [depth_image]
                elif image_source == "rgb-depth":
                    model_images = [rgb_image, depth_image]

                sample = {
                    "index": row_index,
                    "row_in_file": file_row_index,
                    "source_file": str(parquet_file),
                    "category": str(row.get("category") or category),
                    "question": row["question"],
                    "answer": row["answer"],
                    "image": rgb_image,
                    "depth_image": depth_image,
                    "model_images": model_images,
                    "image_path": image_field_path(row.get("img")),
                    "depth_path": image_field_path(row.get("depth_image")),
                    "image_size": rgb_image.size,
                    "model_image_sizes": [image.size for image in model_images],
                }

                if sample["category"] in POINT_CATEGORIES:
                    mask = load_image(row.get("mask"), Image, dataset_root, "mask").convert("L")
                    if rgb_image.size != mask.size:
                        raise ValueError(
                            f"Image/mask size mismatch for sample {row_index}: "
                            f"{rgb_image.size} vs {mask.size}"
                        )
                    mask_arr = np.array(mask)
                    valid_mask = mask_arr >= mask_threshold
                    sample.update(
                        {
                            "mask_path": image_field_path(row.get("mask")),
                            "mask_positive_pixels": int(valid_mask.sum()),
                            "mask_area_ratio": float(valid_mask.sum() / valid_mask.size),
                            "mask_bbox": mask_bbox(valid_mask),
                            "valid_mask": valid_mask,
                        }
                    )
                else:
                    sample.update(
                        {
                            "mask_path": image_field_path(row.get("mask")),
                            "mask_positive_pixels": None,
                            "mask_area_ratio": None,
                            "mask_bbox": None,
                            "valid_mask": None,
                        }
                    )

                yield sample
                row_index += 1
                file_row_index += 1


class RoboSpatialHomeEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        _ensure_repo_on_path()

        from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
        from wall_x._vendor.harrix.serving._wallx_infer.model_wrapper import (
            WallxModelWrapper,
        )
        from wall_x._vendor.harrix.utils.train_config import resolve_max_length

        if args.image_source == "rgb-depth":
            self.image_keys = ["image_0", "image_1"]
        else:
            self.image_keys = ["multi_modal"]
        self.coord_mode = args.coord_mode
        self.image_source = args.image_source

        config = InferConfig(
            checkpoint_path=args.checkpoint_path,
            train_config_path=args.train_config_path,
            model_device=args.device,
            norm_key=args.norm_key,
            cam_names=self.image_keys,
        )
        if args.max_length is not None:
            config.train_config["max_length"] = args.max_length
        self.wrapper = WallxModelWrapper(config)
        self.max_length = args.max_length or resolve_max_length(config.train_config)

    def predict_batch(self, samples: list[dict], prompt_style: str) -> list[dict]:
        import torch

        prompts = [
            build_prompt(sample, prompt_style, self.image_source) for sample in samples
        ]
        observations = [
            {
                image_key: image
                for image_key, image in zip(self.image_keys, sample["model_images"])
            }
            for sample in samples
        ]
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
            cleaned = clean_prediction(text)
            prediction = {
                "raw_prediction": text,
                "prediction": cleaned,
            }
            if sample["category"] in TEXT_CATEGORIES:
                prediction["normalized_prediction"] = normalize_yes_no(text)
            elif sample["category"] in POINT_CATEGORIES:
                raw_points, points = extract_points(
                    text,
                    sample["image_size"],
                    self.coord_mode,
                )
                prediction["raw_points"] = raw_points
                prediction["points"] = points
            predictions.append(prediction)
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


def update_category_metric(
    by_category: dict[str, dict[str, int]],
    sample: dict,
    pred: dict,
    acc_ks: list[int],
) -> tuple[bool, bool]:
    category = sample["category"]
    stats = by_category[category]
    stats["total"] += 1

    if category in TEXT_CATEGORIES:
        normalized_answer = normalize_yes_no(sample["answer"])
        normalized_prediction = pred.get("normalized_prediction", "")
        is_correct = normalized_prediction == normalized_answer and bool(normalized_answer)
        stats["correct"] += int(is_correct)
        stats["parse_failures"] += int(not normalized_prediction)
        return is_correct, is_correct

    if category in POINT_CATEGORIES:
        points = pred.get("points", [])
        point_hits = [
            point_hits_mask(point, sample["valid_mask"])
            for point in points
        ]
        sample_hit = any(point_hits)
        stats["sample_hits"] += int(sample_hit)
        stats["total_points"] += len(points)
        stats["hit_points"] += sum(point_hits)
        stats["no_point_samples"] += int(len(points) == 0)
        for k in acc_ks:
            stats[f"acc_at_{k}"] += int(acc_at_k(point_hits, k))
        primary_hit = acc_at_k(point_hits, 1)
        return primary_hit, sample_hit

    raise ValueError(f"Unsupported category: {category}")


def build_record(sample: dict, pred: dict, acc_ks: list[int]) -> dict:
    record = {
        "index": sample["index"],
        "row_in_file": sample["row_in_file"],
        "source_file": sample["source_file"],
        "category": sample["category"],
        "question": sample["question"],
        "answer": sample["answer"],
        "image_path": sample["image_path"],
        "depth_path": sample["depth_path"],
        "mask_path": sample["mask_path"],
        "image_size": list(sample["image_size"]),
        "model_image_sizes": [list(size) for size in sample["model_image_sizes"]],
        "prediction": pred["prediction"],
        "raw_prediction": pred["raw_prediction"],
    }

    if sample["category"] in TEXT_CATEGORIES:
        normalized_answer = normalize_yes_no(sample["answer"])
        normalized_prediction = pred.get("normalized_prediction", "")
        record.update(
            {
                "normalized_answer": normalized_answer,
                "normalized_prediction": normalized_prediction,
                "correct": normalized_prediction == normalized_answer and bool(normalized_answer),
            }
        )
    elif sample["category"] in POINT_CATEGORIES:
        point_hits = [
            point_hits_mask(point, sample["valid_mask"])
            for point in pred.get("points", [])
        ]
        record.update(
            {
                "mask_positive_pixels": sample["mask_positive_pixels"],
                "mask_area_ratio": sample["mask_area_ratio"],
                "mask_bbox": sample["mask_bbox"],
                "raw_points": [list(point) for point in pred.get("raw_points", [])],
                "points": [list(point) for point in pred.get("points", [])],
                "point_hits": point_hits,
                "sample_hit": any(point_hits),
                **{f"point_acc_at_{k}": acc_at_k(point_hits, k) for k in acc_ks},
            }
        )

    return record


def summarize_categories(
    by_category: dict[str, dict[str, int]],
    acc_ks: list[int],
) -> dict[str, dict]:
    summary = {}
    for category, stats in sorted(by_category.items()):
        total = stats["total"]
        if category in TEXT_CATEGORIES:
            summary[category] = {
                "total": total,
                "correct": stats["correct"],
                "accuracy": stats["correct"] / total if total else 0.0,
                "parse_failures": stats["parse_failures"],
                "parse_failure_rate": stats["parse_failures"] / total if total else 0.0,
            }
        elif category in POINT_CATEGORIES:
            total_points = stats["total_points"]
            summary[category] = {
                "total": total,
                **{
                    f"point_acc_at_{k}": (
                        stats[f"acc_at_{k}"] / total if total else 0.0
                    )
                    for k in acc_ks
                },
                "sample_hits": stats["sample_hits"],
                "sample_hit_rate": stats["sample_hits"] / total if total else 0.0,
                "total_points": total_points,
                "hit_points": stats["hit_points"],
                "point_precision": (
                    stats["hit_points"] / total_points if total_points else 0.0
                ),
                "no_point_samples": stats["no_point_samples"],
                "avg_points_per_sample": total_points / total if total else 0.0,
            }
    return summary


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    categories = parse_categories(args.category)
    acc_ks = parse_acc_ks(args.acc_k)
    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = RoboSpatialHomeEvaluator(args)

    total = 0
    qa_total = 0
    qa_correct = 0
    context_total = 0
    context_acc_counts = {k: 0 for k in acc_ks}
    context_sample_hits = 0
    context_total_points = 0
    context_hit_points = 0
    context_no_point_samples = 0
    mixed_acc_at_1_correct = 0
    mixed_sample_hit_correct = 0
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def filtered_samples():
        seen = 0
        for sample in iter_robospatial_home(
            dataset_root,
            categories,
            args.image_source,
            args.mask_threshold,
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
                primary_correct, sample_hit_correct = update_category_metric(
                    by_category,
                    sample,
                    pred,
                    acc_ks,
                )
                record = build_record(sample, pred, acc_ks)
                record.update(
                    {
                        "prompt_style": args.prompt_style,
                        "coord_mode": args.coord_mode,
                        "image_source": args.image_source,
                    }
                )
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                total += 1
                mixed_acc_at_1_correct += int(primary_correct)
                mixed_sample_hit_correct += int(sample_hit_correct)

                if sample["category"] in TEXT_CATEGORIES:
                    qa_total += 1
                    qa_correct += int(record["correct"])
                elif sample["category"] in POINT_CATEGORIES:
                    context_total += 1
                    point_hits = record["point_hits"]
                    for k in acc_ks:
                        context_acc_counts[k] += int(record[f"point_acc_at_{k}"])
                    context_sample_hits += int(record["sample_hit"])
                    context_total_points += len(record["points"])
                    context_hit_points += sum(point_hits)
                    context_no_point_samples += int(len(record["points"]) == 0)

                if total % 25 == 0:
                    qa_acc = qa_correct / qa_total if qa_total else 0.0
                    context_acc_1 = (
                        context_acc_counts.get(1, 0) / context_total
                        if context_total
                        else 0.0
                    )
                    print(
                        f"evaluated={total} qa_accuracy={qa_acc:.4f} "
                        f"context_point_acc_at_1={context_acc_1:.4f}",
                        flush=True,
                    )

    by_category_summary = summarize_categories(by_category, acc_ks)
    summary = {
        "total": total,
        "qa_total": qa_total,
        "qa_correct": qa_correct,
        "qa_accuracy": qa_correct / qa_total if qa_total else 0.0,
        "context_total": context_total,
        **{
            f"context_point_acc_at_{k}": (
                context_acc_counts[k] / context_total if context_total else 0.0
            )
            for k in acc_ks
        },
        "context_sample_hits": context_sample_hits,
        "context_sample_hit_rate": (
            context_sample_hits / context_total if context_total else 0.0
        ),
        "context_total_points": context_total_points,
        "context_hit_points": context_hit_points,
        "context_point_precision": (
            context_hit_points / context_total_points if context_total_points else 0.0
        ),
        "context_no_point_samples": context_no_point_samples,
        "context_avg_points_per_sample": (
            context_total_points / context_total if context_total else 0.0
        ),
        "mixed_acc_at_1": mixed_acc_at_1_correct / total if total else 0.0,
        "mixed_sample_hit_accuracy": (
            mixed_sample_hit_correct / total if total else 0.0
        ),
        "by_category": by_category_summary,
        "categories": categories,
        "prompt_style": args.prompt_style,
        "coord_mode": args.coord_mode,
        "image_source": args.image_source,
        "mask_threshold": args.mask_threshold,
        "output_jsonl": str(output_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
