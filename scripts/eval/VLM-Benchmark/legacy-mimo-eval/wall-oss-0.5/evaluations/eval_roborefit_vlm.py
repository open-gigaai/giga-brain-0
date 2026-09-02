#!/usr/bin/env python3
"""Evaluate Wall-OSS-0.5 VLM grounding on RoboRefIt.

RoboRefIt contains one image, one referring expression, and one target object
box per sample. The evaluator asks the model to return a bounding box in
original image pixel coordinates and reports IoU-based grounding accuracy.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_ROOT = "datasets/public_datasets/VLM/vqa/roborefit"

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
        help="RoboRefIt root containing data/test-*.parquet.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="roborefit_wall_oss_0_5_predictions.jsonl",
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
        choices=["box-tags", "strict-box-tags", "plain-box", "llava"],
        default="box-tags",
        help=(
            "box-tags asks for <box>[x1, y1, x2, y2]</box>; strict-box-tags "
            "adds stronger constraints; plain-box asks for a list only; llava "
            "sends only the referring expression."
        ),
    )
    parser.add_argument(
        "--iou-thresholds",
        default="0.25,0.5",
        help="Comma-separated IoU thresholds for Acc@IoU. Default: 0.25,0.5.",
    )
    return parser.parse_args()


def build_prompt(ref_exp: str, image_size: tuple[int, int], prompt_style: str) -> str:
    width, height = image_size
    ref_exp = ref_exp.strip()

    if prompt_style == "llava":
        prompt_text = ref_exp
    elif prompt_style == "plain-box":
        prompt_text = (
            f"Locate the object described by: {ref_exp}\n"
            f"The image size is {width}x{height} pixels. "
            "Return only one bounding box as [x1, y1, x2, y2] in original "
            "image pixel coordinates."
        )
    elif prompt_style == "strict-box-tags":
        prompt_text = (
            f"Locate the single object described by: {ref_exp}\n"
            f"The image size is {width}x{height} pixels. "
            "Return the tight bounding box around the visible referred object "
            "in original image pixel coordinates. The box must have positive "
            "width and positive height, with x1 < x2 and y1 < y2. Do not output "
            "a point, a center coordinate, a repeated template coordinate, or "
            "the whole image unless the object actually fills the whole image. "
            "Use x for horizontal coordinate from left to right and y for "
            "vertical coordinate from top to bottom. Answer only in this exact "
            "format: <box>[x1, y1, x2, y2]</box>."
        )
    else:
        prompt_text = (
            f"Locate the object described by: {ref_exp}\n"
            f"The image size is {width}x{height} pixels. "
            "Use x for horizontal coordinate from left to right and y for "
            "vertical coordinate from top to bottom. Return only one bounding "
            "box in original image pixel coordinates as "
            "<box>[x1, y1, x2, y2]</box>."
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


def extract_points(text: str) -> list[tuple[float, float]]:
    cleaned = clean_prediction(text)
    points: list[tuple[float, float]] = []

    tag_pattern = re.compile(
        r"<(?:point|points)>\s*\[?([^\]<>]+)\]?\s*</(?:point|points)>",
        re.I,
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


def normalize_box(
    box: Iterable[float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    width, height = image_size
    values = [float(x) for x in box]
    if len(values) != 4:
        raise ValueError(f"Expected 4 bbox values, got {len(values)}")

    # Some models output normalized [0, 1] coordinates despite the prompt.
    if max(abs(v) for v in values) <= 1.5:
        values = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]

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
    if not thresholds:
        raise ValueError("--iou-thresholds must contain at least one threshold")
    return sorted(dict.fromkeys(thresholds))


def metric_name_for_threshold(threshold: float) -> str:
    text = f"{threshold:g}".replace(".", "_")
    return f"acc_iou_{text}"


def iter_roborefit(dataset_root: Path) -> Iterable[dict]:
    import pyarrow.parquet as pq
    from PIL import Image

    parquet_files = sorted((dataset_root / "data").glob("test-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    columns = ["id", "image", "ref_exp", "bbox", "normalized_bbox"]
    for parquet_file in parquet_files:
        pf = pq.ParquetFile(parquet_file)
        for batch in pf.iter_batches(batch_size=32, columns=columns):
            for row in batch.to_pylist():
                image = load_image(row["image"], Image).convert("RGB")
                image_path = row["image"].get("path") if isinstance(row["image"], dict) else None
                gt_box_original = tuple(float(v) for v in row["bbox"])
                gt_box = normalize_box(gt_box_original, image.size)
                yield {
                    "index": int(row["id"]),
                    "id": int(row["id"]),
                    "image": image,
                    "image_path": image_path,
                    "image_size": image.size,
                    "ref_exp": row["ref_exp"],
                    "bbox_original": gt_box_original,
                    "bbox": gt_box,
                    "normalized_bbox": [float(v) for v in row["normalized_bbox"]],
                }


def load_image(image_obj: dict, image_cls):
    if not isinstance(image_obj, dict):
        raise TypeError(f"Unsupported image field type: {type(image_obj).__name__}")
    if image_obj.get("bytes") is None:
        raise ValueError("RoboRefIt image field does not contain bytes")
    return image_cls.open(io.BytesIO(image_obj["bytes"]))


class RoboRefItEvaluator:
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
            build_prompt(s["ref_exp"], s["image_size"], prompt_style) for s in samples
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
                    "Increase --max-length or reduce image resolution in the train config."
                )

        with torch.inference_mode():
            model_output = self.wrapper.model.generate_text(**model_input)

        predictions = []
        for sample, text in zip(samples, model_output["predict_output_text"]):
            raw_box = extract_box(text)
            pred_box = normalize_box(raw_box, sample["image_size"]) if raw_box else None
            raw_points = extract_points(text)
            points = [
                (
                    min(max(float(x), 0.0), float(sample["image_size"][0])),
                    min(max(float(y), 0.0), float(sample["image_size"][1])),
                )
                for x, y in raw_points
            ]
            predictions.append(
                {
                    "raw_prediction": text,
                    "prediction": clean_prediction(text),
                    "raw_box": raw_box,
                    "box": pred_box,
                    "points": points,
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
    iou_thresholds = parse_iou_thresholds(args.iou_thresholds)

    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = RoboRefItEvaluator(args)

    total = 0
    parsed_boxes = 0
    no_box_samples = 0
    parsed_points = 0
    point_hit_samples = 0
    zero_area_box_samples = 0
    pred_box_counter: Counter[tuple[int, int, int, int]] = Counter()
    iou_sum = 0.0
    threshold_hits = {threshold: 0 for threshold in iou_thresholds}

    def filtered_samples():
        seen = 0
        for sample in iter_roborefit(dataset_root):
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
                total += 1

                iou = bbox_iou(pred["box"], sample["bbox"]) if pred["box"] else 0.0
                iou_sum += iou
                parsed_boxes += int(pred["box"] is not None)
                no_box_samples += int(pred["box"] is None)
                if pred["box"] is not None:
                    px1, py1, px2, py2 = pred["box"]
                    zero_area_box_samples += int(px2 <= px1 or py2 <= py1)
                    pred_box_counter[
                        (round(px1), round(py1), round(px2), round(py2))
                    ] += 1
                for threshold in iou_thresholds:
                    threshold_hits[threshold] += int(iou >= threshold)

                point_hits = [point_in_box(point, sample["bbox"]) for point in pred["points"]]
                parsed_points += len(pred["points"])
                point_hit = any(point_hits)
                point_hit_samples += int(point_hit)

                record = {
                    "index": sample["index"],
                    "id": sample["id"],
                    "image_path": sample["image_path"],
                    "image_size": list(sample["image_size"]),
                    "ref_exp": sample["ref_exp"],
                    "bbox_original": list(sample["bbox_original"]),
                    "bbox": list(sample["bbox"]),
                    "normalized_bbox": sample["normalized_bbox"],
                    "prediction": pred["prediction"],
                    "raw_prediction": pred["raw_prediction"],
                    "raw_pred_bbox": list(pred["raw_box"]) if pred["raw_box"] else None,
                    "pred_bbox": list(pred["box"]) if pred["box"] else None,
                    "iou": iou,
                    **{
                        metric_name_for_threshold(threshold): iou >= threshold
                        for threshold in iou_thresholds
                    },
                    "points": [list(point) for point in pred["points"]],
                    "point_hits": point_hits,
                    "point_hit": point_hit,
                    "prompt_style": args.prompt_style,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if total % 25 == 0:
                    main_threshold = 0.5 if 0.5 in threshold_hits else iou_thresholds[-1]
                    print(
                        f"evaluated={total} "
                        f"mean_iou={iou_sum / total:.4f} "
                        f"{metric_name_for_threshold(main_threshold)}="
                        f"{threshold_hits[main_threshold] / total:.4f} "
                        f"parsed_box_rate={parsed_boxes / total:.4f}",
                        flush=True,
                    )

    summary = {
        "total": total,
        "parsed_box_samples": parsed_boxes,
        "parsed_box_rate": parsed_boxes / total if total else 0.0,
        "no_box_samples": no_box_samples,
        "zero_area_box_samples": zero_area_box_samples,
        "unique_pred_boxes": len(pred_box_counter),
        "repeated_box_rate": (
            1.0 - (len(pred_box_counter) / parsed_boxes)
            if parsed_boxes
            else 0.0
        ),
        "most_common_pred_boxes": [
            {"box": list(box), "count": count}
            for box, count in pred_box_counter.most_common(10)
        ],
        "mean_iou": iou_sum / total if total else 0.0,
        **{
            metric_name_for_threshold(threshold): (
                threshold_hits[threshold] / total if total else 0.0
            )
            for threshold in iou_thresholds
        },
        "parsed_points": parsed_points,
        "point_hit_samples": point_hit_samples,
        "point_hit_rate": point_hit_samples / total if total else 0.0,
        "prompt_style": args.prompt_style,
        "output_jsonl": str(output_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
