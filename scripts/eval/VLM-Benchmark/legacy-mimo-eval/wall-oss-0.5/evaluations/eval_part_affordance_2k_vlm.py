#!/usr/bin/env python3
"""Evaluate Wall-OSS-0.5 VLM point grounding on Part-Affordance-2K.

Part-Affordance-2K contains one image, one grasp-affordance instruction, and a
binary affordance mask per sample. The evaluator prompts the model to output
one or more grasp points and scores whether predicted points fall inside the
mask.
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


DEFAULT_DATASET_ROOT = "datasets/public_datasets/VLM/vqa/Part-Affordance-2K"

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
        help="Part-Affordance-2K root containing data/train-*.parquet.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="part_affordance_2k_wall_oss_0_5_predictions.jsonl",
        help="Where to write per-sample predictions.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--norm-key", default="x2_normal")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=128,
        help="Mask pixels greater than or equal to this value are valid.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["point-tags", "dataset", "plain"],
        default="point-tags",
        help=(
            "point-tags asks for <point> outputs; dataset keeps the original "
            "problem and adds output guidance; plain sends the problem only."
        ),
    )
    parser.add_argument(
        "--acc-k",
        default="1,3,5",
        help="Comma-separated K values for Point Acc@K. Default: 1,3,5.",
    )
    return parser.parse_args()


def build_prompt(problem: str, image_size: tuple[int, int], prompt_style: str) -> str:
    width, height = image_size
    problem = problem.strip()

    if prompt_style == "plain":
        prompt_text = problem
    elif prompt_style == "dataset":
        prompt_text = (
            f"{problem}\n"
            f"The image size is {width}x{height} pixels. "
            "Return 1 to 5 grasp points in original image pixel coordinates. "
            "Use x for horizontal coordinate from left to right and y for "
            "vertical coordinate from top to bottom. "
            "Answer only with point tags like <point>[[x, y], ...]</point>."
        )
    else:
        prompt_text = (
            f"{problem}\n"
            f"The image size is {width}x{height} pixels. "
            "Identify the part or region that affords grasping. Return 1 to 5 "
            "suitable grasp points in original image pixel coordinates. Use x "
            "for horizontal coordinate from left to right and y for vertical "
            "coordinate from top to bottom. "
            "Answer only as <answer><point>[[x, y], ...]</point></answer>."
        )

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


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


def normalize_point(
    point: tuple[float, float],
    image_size: tuple[int, int],
) -> tuple[int, int]:
    width, height = image_size
    x, y = point
    if max(abs(x), abs(y)) <= 1.5:
        x *= width
        y *= height
    return (round(x), round(y))


def extract_points(text: str, image_size: tuple[int, int]) -> list[tuple[int, int]]:
    cleaned = clean_prediction(text)
    points: list[tuple[float, float]] = []

    tag_pattern = re.compile(
        r"<(?:point|points)>\s*(.*?)\s*</(?:point|points)>",
        re.I | re.S,
    )
    for match in tag_pattern.finditer(cleaned):
        nums = _numbers(match.group(1))
        for i in range(0, len(nums) - 1, 2):
            points.append((nums[i], nums[i + 1]))

    if not points:
        pair_pattern = re.compile(
            r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]\)]"
        )
        for x, y in pair_pattern.findall(cleaned):
            points.append((float(x), float(y)))

    if not points:
        xy_pattern = re.compile(
            r"\bx\s*=\s*(-?\d+(?:\.\d+)?)\D{0,20}\by\s*=\s*(-?\d+(?:\.\d+)?)",
            re.I,
        )
        for x, y in xy_pattern.findall(cleaned):
            points.append((float(x), float(y)))

    return [normalize_point(point, image_size) for point in points]


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


def acc_at_k(point_hits: list[bool], k: int) -> bool:
    return any(point_hits[:k])


def iter_part_affordance_2k(
    dataset_root: Path,
    mask_threshold: int,
) -> Iterable[dict]:
    import pyarrow.parquet as pq
    from PIL import Image

    parquet_files = sorted((dataset_root / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    columns = ["question_id", "problem", "image", "mask", "category_type"]
    row_index = 0
    for parquet_file in parquet_files:
        pf = pq.ParquetFile(parquet_file)
        for batch in pf.iter_batches(batch_size=32, columns=columns):
            for row in batch.to_pylist():
                image = load_image(row["image"], Image).convert("RGB")
                mask = load_image(row["mask"], Image).convert("L")
                if image.size != mask.size:
                    raise ValueError(
                        f"Image/mask size mismatch for {row['question_id']}: "
                        f"{image.size} vs {mask.size}"
                    )
                mask_arr = np.array(mask)
                valid_mask = mask_arr >= mask_threshold
                yield {
                    "index": row_index,
                    "question_id": int(row["question_id"]),
                    "problem": row["problem"],
                    "category_type": row["category_type"],
                    "image": image,
                    "image_size": image.size,
                    "mask_positive_pixels": int(valid_mask.sum()),
                    "mask_area_ratio": float(valid_mask.sum() / valid_mask.size),
                    "mask_bbox": mask_bbox(valid_mask),
                    "valid_mask": valid_mask,
                }
                row_index += 1


def load_image(image_obj: dict, image_cls):
    if not isinstance(image_obj, dict):
        raise TypeError(f"Unsupported image field type: {type(image_obj).__name__}")
    if image_obj.get("bytes") is None:
        raise ValueError("Part-Affordance-2K image field does not contain bytes")
    return image_cls.open(io.BytesIO(image_obj["bytes"]))


class PartAffordance2KEvaluator:
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

        prompts = [
            build_prompt(s["problem"], s["image_size"], prompt_style) for s in samples
        ]
        observations = [{self.image_key: s["image"]} for s in samples]
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
                "raw_prediction": text,
                "prediction": clean_prediction(text),
                "points": extract_points(text, sample["image_size"]),
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


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    acc_ks = parse_acc_ks(args.acc_k)

    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = PartAffordance2KEvaluator(args)

    total = 0
    total_points = 0
    hit_points = 0
    no_point_samples = 0
    sample_hits = 0
    acc_counts = {k: 0 for k in acc_ks}
    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total": 0,
            "sample_hits": 0,
            "total_points": 0,
            "hit_points": 0,
            "no_point_samples": 0,
            **{f"acc_at_{k}": 0 for k in acc_ks},
        }
    )

    def filtered_samples():
        seen = 0
        for sample in iter_part_affordance_2k(dataset_root, args.mask_threshold):
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
                point_hits = [
                    point_hits_mask(point, sample["valid_mask"])
                    for point in pred["points"]
                ]
                sample_acc_at_k = {k: acc_at_k(point_hits, k) for k in acc_ks}
                has_hit = any(point_hits)

                total += 1
                total_points += len(pred["points"])
                hit_points += sum(point_hits)
                no_point_samples += int(len(pred["points"]) == 0)
                sample_hits += int(has_hit)
                for k, hit in sample_acc_at_k.items():
                    acc_counts[k] += int(hit)

                stats = by_category[sample["category_type"]]
                stats["total"] += 1
                stats["sample_hits"] += int(has_hit)
                stats["total_points"] += len(pred["points"])
                stats["hit_points"] += sum(point_hits)
                stats["no_point_samples"] += int(len(pred["points"]) == 0)
                for k, hit in sample_acc_at_k.items():
                    stats[f"acc_at_{k}"] += int(hit)

                record = {
                    "index": sample["index"],
                    "question_id": sample["question_id"],
                    "category_type": sample["category_type"],
                    "problem": sample["problem"],
                    "image_size": list(sample["image_size"]),
                    "mask_positive_pixels": sample["mask_positive_pixels"],
                    "mask_area_ratio": sample["mask_area_ratio"],
                    "mask_bbox": sample["mask_bbox"],
                    "prediction": pred["prediction"],
                    "raw_prediction": pred["raw_prediction"],
                    "points": [list(point) for point in pred["points"]],
                    "point_hits": point_hits,
                    "sample_hit": has_hit,
                    **{f"point_acc_at_{k}": hit for k, hit in sample_acc_at_k.items()},
                    "prompt_style": args.prompt_style,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if total % 25 == 0:
                    print(
                        f"evaluated={total} "
                        f"point_acc_at_{acc_ks[0]}={acc_counts[acc_ks[0]] / total:.4f} "
                        f"point_precision={(hit_points / total_points) if total_points else 0.0:.4f}",
                        flush=True,
                    )

    by_category_summary = {}
    for category, stats in sorted(by_category.items()):
        by_category_summary[category] = {
            "total": stats["total"],
            "sample_hits": stats["sample_hits"],
            "sample_hit_rate": (
                stats["sample_hits"] / stats["total"] if stats["total"] else 0.0
            ),
            "total_points": stats["total_points"],
            "hit_points": stats["hit_points"],
            "point_precision": (
                stats["hit_points"] / stats["total_points"]
                if stats["total_points"]
                else 0.0
            ),
            "no_point_samples": stats["no_point_samples"],
            **{
                f"point_acc_at_{k}": (
                    stats[f"acc_at_{k}"] / stats["total"] if stats["total"] else 0.0
                )
                for k in acc_ks
            },
        }

    summary = {
        "total": total,
        **{f"point_acc_at_{k}": acc_counts[k] / total if total else 0.0 for k in acc_ks},
        "sample_hits": sample_hits,
        "sample_hit_rate": sample_hits / total if total else 0.0,
        "total_points": total_points,
        "hit_points": hit_points,
        "point_precision": hit_points / total_points if total_points else 0.0,
        "no_point_samples": no_point_samples,
        "avg_points_per_sample": total_points / total if total else 0.0,
        "mask_threshold": args.mask_threshold,
        "by_category_type": by_category_summary,
        "prompt_style": args.prompt_style,
        "output_jsonl": str(output_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
