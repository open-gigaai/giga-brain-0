#!/usr/bin/env python3
"""Evaluate Wall-OSS-0.5 VLM point grounding on VABench point-bbox.

VABench point-bbox contains one image, one manipulation instruction prompt, and
one target-region bounding box per sample. The model is asked to output one or
more 2D points, and the main score checks whether the predicted points fall
inside the target box.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_ROOT = "datasets/public_datasets/VLM/vqa/vabench-point-bbox"

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
        help="VABench point-bbox root containing data/test-*.parquet.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="vabench_point_bbox_wall_oss_0_5_predictions.jsonl",
        help="Where to write per-sample predictions.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--norm-key", default="x2_normal")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--prompt-style",
        choices=["dataset", "point-tags", "plain"],
        default="dataset",
        help=(
            "dataset preserves the parquet problem text; point-tags uses a shorter "
            "controlled prompt; plain sends the problem text without extra guidance."
        ),
    )
    parser.add_argument(
        "--acc-k",
        default="1,3,5",
        help="Comma-separated K values for Point Acc@K. Default: 1,3,5.",
    )
    parser.add_argument(
        "--iou-thresholds",
        default="0.25,0.5",
        help="Diagnostic bbox IoU thresholds if the model outputs a box.",
    )
    return parser.parse_args()


def strip_image_token(problem: str) -> str:
    return re.sub(r"\s*<image>\s*", "", problem).strip()


def extract_task_instruction(problem: str) -> str:
    text = strip_image_token(problem)
    match = re.search(
        r"The task instruction is:\s*(.*?)\s*Use 2D points",
        text,
        flags=re.I | re.S,
    )
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return text


def build_prompt(problem: str, image_size: tuple[int, int], prompt_style: str) -> str:
    width, height = image_size
    problem_text = strip_image_token(problem)

    if prompt_style == "plain":
        prompt_text = problem_text
    elif prompt_style == "point-tags":
        instruction = extract_task_instruction(problem)
        prompt_text = (
            "You are currently a robot performing robotic manipulation tasks. "
            f"The task instruction is: {instruction}\n"
            f"The image size is {width}x{height} pixels. "
            "Use 2D points to mark the target location where the object should "
            "ultimately be moved. Use original image pixel coordinates, with x "
            "from left to right and y from top to bottom. "
            "Answer only as <answer><point>[[x, y], ...]</point></answer>."
        )
    else:
        prompt_text = (
            f"{problem_text}\n"
            f"The image size is {width}x{height} pixels. Use original image "
            "pixel coordinates, with x from left to right and y from top to bottom."
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


def extract_points(text: str) -> list[tuple[float, float]]:
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

    if points:
        return points

    pair_pattern = re.compile(
        r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]\)]"
    )
    for x, y in pair_pattern.findall(cleaned):
        points.append((float(x), float(y)))

    if points:
        return points

    xy_pattern = re.compile(
        r"\bx\s*=\s*(-?\d+(?:\.\d+)?)\D{0,20}\by\s*=\s*(-?\d+(?:\.\d+)?)",
        re.I,
    )
    for x, y in xy_pattern.findall(cleaned):
        points.append((float(x), float(y)))
    return points


def extract_box(text: str) -> tuple[float, float, float, float] | None:
    cleaned = clean_prediction(text)

    tag_pattern = re.compile(
        r"<(?:box|bbox|bounding_box)>\s*\[?([^\]<>]+)\]?\s*</(?:box|bbox|bounding_box)>",
        re.I,
    )
    for match in tag_pattern.finditer(cleaned):
        nums = _numbers(match.group(1))
        if len(nums) >= 4:
            return tuple(nums[:4])  # type: ignore[return-value]

    named_pattern = re.compile(
        r"\bx1\s*=\s*(-?\d+(?:\.\d+)?)\D{0,20}"
        r"\by1\s*=\s*(-?\d+(?:\.\d+)?)\D{0,20}"
        r"\bx2\s*=\s*(-?\d+(?:\.\d+)?)\D{0,20}"
        r"\by2\s*=\s*(-?\d+(?:\.\d+)?)",
        re.I,
    )
    named = named_pattern.search(cleaned)
    if named:
        return tuple(float(x) for x in named.groups())  # type: ignore[return-value]

    list_pattern = re.compile(
        r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]\)]"
    )
    listed = list_pattern.search(cleaned)
    if listed:
        return tuple(float(x) for x in listed.groups())  # type: ignore[return-value]

    return None


def normalize_point(
    point: tuple[float, float],
    image_size: tuple[int, int],
) -> tuple[float, float]:
    width, height = image_size
    x, y = point
    if max(abs(x), abs(y)) <= 1.5:
        x *= width
        y *= height
    return (x, y)


def normalize_box(
    box: Iterable[float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    width, height = image_size
    values = [float(x) for x in box]
    if len(values) != 4:
        raise ValueError(f"Expected 4 bbox values, got {len(values)}")

    if max(abs(v) for v in values) <= 1.5:
        values = [
            values[0] * width,
            values[1] * height,
            values[2] * width,
            values[3] * height,
        ]

    x1, y1, x2, y2 = values
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    return (x1, y1, x2, y2)


def bbox_iou(
    pred_box: tuple[float, float, float, float],
    gt_box: tuple[float, float, float, float],
) -> float:
    px1, py1, px2, py2 = pred_box
    gx1, gy1, gx2, gy2 = gt_box

    inter_w = max(0.0, min(px2, gx2) - max(px1, gx1))
    inter_h = max(0.0, min(py2, gy2) - max(py1, gy1))
    inter = inter_w * inter_h

    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pred_area + gt_area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def point_in_box(
    point: tuple[float, float],
    box: tuple[float, float, float, float],
) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


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


def parse_iou_thresholds(value: str) -> list[float]:
    thresholds = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        threshold = float(part)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("--iou-thresholds values must be in [0, 1]")
        thresholds.append(threshold)
    return sorted(dict.fromkeys(thresholds))


def point_acc_at_k(point_hits: list[bool], k: int) -> bool:
    return any(point_hits[:k])


def metric_name_for_threshold(threshold: float) -> str:
    text = f"{threshold:g}".replace(".", "_")
    return f"acc_iou_{text}"


def iter_vabench_point_bbox(dataset_root: Path) -> Iterable[dict]:
    import pyarrow.parquet as pq
    from PIL import Image

    parquet_files = sorted((dataset_root / "data").glob("test-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    columns = ["idx", "problem", "image", "bbox", "normalized_bbox"]
    for parquet_file in parquet_files:
        pf = pq.ParquetFile(parquet_file)
        for batch in pf.iter_batches(batch_size=32, columns=columns):
            for row in batch.to_pylist():
                image = load_image(row["image"], Image).convert("RGB")
                gt_box_original = tuple(float(v) for v in row["bbox"])
                gt_box = normalize_box(gt_box_original, image.size)
                yield {
                    "index": int(row["idx"]),
                    "idx": int(row["idx"]),
                    "problem": row["problem"],
                    "task_instruction": extract_task_instruction(row["problem"]),
                    "image": image,
                    "image_size": image.size,
                    "bbox_original": gt_box_original,
                    "bbox": gt_box,
                    "normalized_bbox": [float(v) for v in row["normalized_bbox"]],
                }


def load_image(image_obj: dict, image_cls):
    if not isinstance(image_obj, dict):
        raise TypeError(f"Unsupported image field type: {type(image_obj).__name__}")
    if image_obj.get("bytes") is None:
        raise ValueError("VABench point-bbox image field does not contain bytes")
    return image_cls.open(io.BytesIO(image_obj["bytes"]))


class VABenchPointBBoxEvaluator:
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

        predictions = []
        for sample, text in zip(samples, model_output["predict_output_text"]):
            raw_points = extract_points(text)
            points = [normalize_point(point, sample["image_size"]) for point in raw_points]
            raw_box = extract_box(text)
            pred_box = normalize_box(raw_box, sample["image_size"]) if raw_box else None
            predictions.append(
                {
                    "raw_prediction": text,
                    "prediction": clean_prediction(text),
                    "raw_points": raw_points,
                    "points": points,
                    "raw_box": raw_box,
                    "box": pred_box,
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


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    acc_ks = parse_acc_ks(args.acc_k)
    iou_thresholds = parse_iou_thresholds(args.iou_thresholds)

    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = VABenchPointBBoxEvaluator(args)

    total = 0
    total_points = 0
    hit_points = 0
    no_point_samples = 0
    point_hit_samples = 0
    acc_counts = {k: 0 for k in acc_ks}
    parsed_boxes = 0
    iou_sum = 0.0
    iou_hits = {threshold: 0 for threshold in iou_thresholds}

    def filtered_samples():
        seen = 0
        for sample in iter_vabench_point_bbox(dataset_root):
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
                point_hits = [point_in_box(point, sample["bbox"]) for point in pred["points"]]
                sample_acc_at_k = {k: point_acc_at_k(point_hits, k) for k in acc_ks}

                total += 1
                total_points += len(pred["points"])
                hit_points += sum(point_hits)
                no_point_samples += int(len(pred["points"]) == 0)
                point_hit = any(point_hits)
                point_hit_samples += int(point_hit)
                for k, hit in sample_acc_at_k.items():
                    acc_counts[k] += int(hit)

                bbox_iou_value = None
                if pred["box"] is not None:
                    parsed_boxes += 1
                    bbox_iou_value = bbox_iou(pred["box"], sample["bbox"])
                    iou_sum += bbox_iou_value
                    for threshold in iou_thresholds:
                        iou_hits[threshold] += int(bbox_iou_value >= threshold)

                record = {
                    "index": sample["index"],
                    "idx": sample["idx"],
                    "image_size": list(sample["image_size"]),
                    "task_instruction": sample["task_instruction"],
                    "problem": sample["problem"],
                    "bbox_original": list(sample["bbox_original"]),
                    "bbox": list(sample["bbox"]),
                    "normalized_bbox": sample["normalized_bbox"],
                    "prediction": pred["prediction"],
                    "raw_prediction": pred["raw_prediction"],
                    "raw_points": [list(point) for point in pred["raw_points"]],
                    "points": [list(point) for point in pred["points"]],
                    "point_hits": point_hits,
                    "point_hit": point_hit,
                    **{f"point_acc_at_{k}": hit for k, hit in sample_acc_at_k.items()},
                    "raw_pred_bbox": list(pred["raw_box"]) if pred["raw_box"] else None,
                    "pred_bbox": list(pred["box"]) if pred["box"] else None,
                    "bbox_iou": bbox_iou_value,
                    **{
                        metric_name_for_threshold(threshold): (
                            bbox_iou_value is not None and bbox_iou_value >= threshold
                        )
                        for threshold in iou_thresholds
                    },
                    "prompt_style": args.prompt_style,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if total % 25 == 0:
                    print(
                        f"evaluated={total} "
                        f"point_acc_at_{acc_ks[0]}={acc_counts[acc_ks[0]] / total:.4f} "
                        f"point_hit_rate={(hit_points / total_points) if total_points else 0.0:.4f}",
                        flush=True,
                    )

    summary = {
        "total": total,
        **{f"point_acc_at_{k}": acc_counts[k] / total if total else 0.0 for k in acc_ks},
        "point_hit_samples": point_hit_samples,
        "point_hit_rate": point_hit_samples / total if total else 0.0,
        "total_points": total_points,
        "hit_points": hit_points,
        "point_precision": hit_points / total_points if total_points else 0.0,
        "no_point_samples": no_point_samples,
        "avg_points_per_sample": total_points / total if total else 0.0,
        "parsed_box_samples": parsed_boxes,
        "parsed_box_rate": parsed_boxes / total if total else 0.0,
        "mean_bbox_iou_on_parsed_boxes": iou_sum / parsed_boxes if parsed_boxes else 0.0,
        **{
            metric_name_for_threshold(threshold): (
                iou_hits[threshold] / parsed_boxes if parsed_boxes else 0.0
            )
            for threshold in iou_thresholds
        },
        "prompt_style": args.prompt_style,
        "output_jsonl": str(output_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
