#!/usr/bin/env python3
"""Evaluate Wall-OSS-0.5 VLM point grounding on RefSpatial-Bench.

RefSpatial-Bench contains Location, Placement, and Unseen splits. Each sample
has one RGB image, one spatial referring prompt, and one binary-ish mask. The
evaluator asks the model to output one or more 2D points and scores whether
predicted points fall inside the mask.
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


DEFAULT_DATASET_ROOT = "datasets/public_datasets/VLM/vqa/RefSpatial-Bench"
ALL_SPLITS = ("location", "placement", "unseen")
DEFAULT_SPLITS = ("location", "placement")

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
        help="RefSpatial-Bench root containing data/location-*.parquet.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="refspatial_bench_wall_oss_0_5_predictions.jsonl",
        help="Where to write per-sample predictions.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--norm-key", default="x2_normal")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--split",
        default="default",
        help=(
            "Split selection: default evaluates location,placement; all also "
            "includes unseen; or pass a comma-separated subset."
        ),
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=128,
        help="Mask pixels greater than or equal to this value are valid.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["dataset", "normalized-point-tags", "object-locate", "plain"],
        default="dataset",
        help=(
            "dataset uses prompt + suffix from the benchmark; normalized-point-tags "
            "adds stricter tag formatting; object-locate uses the object field."
        ),
    )
    parser.add_argument(
        "--coord-mode",
        choices=["auto", "normalized", "pixel", "percent", "per-mille"],
        default="auto",
        help=(
            "How to interpret parsed points. auto treats values in [0, 1.5] as "
            "normalized and larger values as pixels."
        ),
    )
    parser.add_argument(
        "--acc-k",
        default="1,3,5",
        help="Comma-separated K values for Point Acc@K. Default: 1,3,5.",
    )
    return parser.parse_args()


def parse_splits(value: str) -> list[str]:
    value = value.strip().lower()
    if value == "default":
        return list(DEFAULT_SPLITS)
    if value == "all":
        return list(ALL_SPLITS)
    splits = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = sorted(set(splits) - set(ALL_SPLITS))
    if unknown:
        raise ValueError(f"Unknown RefSpatial-Bench splits: {unknown}")
    if not splits:
        raise ValueError("--split must select at least one split")
    return splits


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


def build_prompt(sample: dict, prompt_style: str) -> str:
    prompt = str(sample["prompt"]).strip()
    suffix = str(sample.get("suffix") or "").strip()
    obj = str(sample.get("object") or "").strip()
    width, height = sample["image_size"]

    if prompt_style == "plain":
        prompt_text = prompt
    elif prompt_style == "object-locate":
        target = obj or prompt
        prompt_text = (
            f"Locate the points of {target}.\n"
            "Return 1 to 5 points as normalized coordinates between 0 and 1, "
            "where x is horizontal from left to right and y is vertical from "
            "top to bottom. Answer only as [(x1, y1), (x2, y2), ...]."
        )
    elif prompt_style == "normalized-point-tags":
        prompt_text = (
            f"{prompt}\n"
            "Return 1 to 5 points that satisfy the referring expression. Use "
            "normalized coordinates between 0 and 1, where x is horizontal "
            "from left to right and y is vertical from top to bottom. Answer "
            "only as <answer><point>[(x1, y1), (x2, y2), ...]</point></answer>."
        )
    else:
        prompt_text = f"{prompt} {suffix}".strip()

    if prompt_style != "dataset":
        prompt_text += f"\nThe original image size is {width}x{height} pixels."

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
    coord_mode: str,
) -> tuple[int, int]:
    width, height = image_size
    x, y = point
    max_coord = max(abs(x), abs(y))
    if coord_mode == "normalized" or (coord_mode == "auto" and max_coord <= 1.5):
        x *= width
        y *= height
    elif coord_mode == "percent" or (
        coord_mode == "auto" and max_coord <= 100 and (x > width or y > height)
    ):
        x = x / 100.0 * width
        y = y / 100.0 * height
    elif coord_mode == "per-mille" or (
        coord_mode == "auto" and max_coord <= 1000 and (x > width or y > height)
    ):
        x = x / 1000.0 * width
        y = y / 1000.0 * height
    return (round(x), round(y))


def extract_json_points(text: str) -> list[tuple[float, float]]:
    cleaned = clean_prediction(text)
    candidates = []
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", cleaned, re.I | re.S):
        candidates.append(match.group(1).strip())
    candidates.append(cleaned)

    points: list[tuple[float, float]] = []
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        for item in data:
            if isinstance(item, dict):
                value = item.get("point") or item.get("points") or item.get("coordinate")
                if isinstance(value, list) and len(value) >= 2:
                    points.append((float(value[0]), float(value[1])))
            elif isinstance(item, list) and len(item) >= 2:
                points.append((float(item[0]), float(item[1])))
        if points:
            return points
    return points


def extract_points(
    text: str,
    image_size: tuple[int, int],
    coord_mode: str,
) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    cleaned = clean_prediction(text)
    raw_points: list[tuple[float, float]] = []

    # Molmo-style XML attributes: <points x1="61.5" y1="40.4" ... />
    attr_pattern = re.compile(
        r'x\d*\s*=\s*"(-?\d+(?:\.\d+)?)"\s+y\d*\s*=\s*"(-?\d+(?:\.\d+)?)"',
        re.I,
    )
    for x, y in attr_pattern.findall(cleaned):
        raw_points.append((float(x), float(y)))

    if not raw_points:
        raw_points.extend(extract_json_points(cleaned))

    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", cleaned, re.I | re.S)
    search_text = answer_match.group(1) if answer_match else cleaned

    if not raw_points:
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


def parquet_path(dataset_root: Path, split: str) -> Path:
    return dataset_root / "data" / f"{split}-00000-of-00001.parquet"


def load_image(image_obj: dict, image_cls, field_name: str):
    if not isinstance(image_obj, dict):
        raise TypeError(f"Unsupported {field_name} field type: {type(image_obj).__name__}")
    if image_obj.get("bytes") is None:
        raise ValueError(f"RefSpatial-Bench {field_name} field does not contain bytes")
    return image_cls.open(io.BytesIO(image_obj["bytes"]))


def iter_refspatial_bench(
    dataset_root: Path,
    splits: list[str],
    mask_threshold: int,
) -> Iterable[dict]:
    import pyarrow.parquet as pq
    from PIL import Image

    columns = ["id", "image", "mask", "object", "prompt", "suffix", "step"]
    row_index = 0
    for split in splits:
        parquet_file = parquet_path(dataset_root, split)
        if not parquet_file.exists():
            raise FileNotFoundError(f"RefSpatial-Bench parquet not found: {parquet_file}")
        pf = pq.ParquetFile(parquet_file)
        file_row_index = 0
        for batch in pf.iter_batches(batch_size=32, columns=columns):
            for row in batch.to_pylist():
                image = load_image(row["image"], Image, "image").convert("RGB")
                mask = load_image(row["mask"], Image, "mask").convert("L")
                if image.size != mask.size:
                    raise ValueError(
                        f"Image/mask size mismatch for {split}/{row['id']}: "
                        f"{image.size} vs {mask.size}"
                    )
                mask_arr = np.array(mask)
                valid_mask = mask_arr >= mask_threshold
                yield {
                    "index": row_index,
                    "id": int(row["id"]),
                    "row_in_file": file_row_index,
                    "source_file": str(parquet_file),
                    "split": split,
                    "image": image,
                    "image_path": row["image"].get("path"),
                    "mask_path": row["mask"].get("path"),
                    "image_size": image.size,
                    "object": row["object"],
                    "prompt": row["prompt"],
                    "suffix": row["suffix"],
                    "step": int(row["step"]),
                    "mask_positive_pixels": int(valid_mask.sum()),
                    "mask_area_ratio": float(valid_mask.sum() / valid_mask.size),
                    "mask_bbox": mask_bbox(valid_mask),
                    "valid_mask": valid_mask,
                }
                row_index += 1
                file_row_index += 1


class RefSpatialBenchEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        _ensure_repo_on_path()

        from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
        from wall_x._vendor.harrix.serving._wallx_infer.model_wrapper import (
            WallxModelWrapper,
        )
        from wall_x._vendor.harrix.utils.train_config import resolve_max_length

        self.image_key = "multi_modal"
        self.coord_mode = args.coord_mode
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
            raw_points, points = extract_points(
                text,
                sample["image_size"],
                self.coord_mode,
            )
            predictions.append(
                {
                    "raw_prediction": text,
                    "prediction": clean_prediction(text),
                    "raw_points": raw_points,
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


def empty_group(acc_ks: list[int]) -> dict[str, int]:
    return {
        "total": 0,
        "sample_hits": 0,
        "total_points": 0,
        "hit_points": 0,
        "no_point_samples": 0,
        **{f"acc_at_{k}": 0 for k in acc_ks},
    }


def update_point_stats(stats: dict[str, int], point_hits: list[bool], acc_ks: list[int]) -> None:
    stats["total"] += 1
    stats["sample_hits"] += int(any(point_hits))
    stats["total_points"] += len(point_hits)
    stats["hit_points"] += sum(point_hits)
    stats["no_point_samples"] += int(len(point_hits) == 0)
    for k in acc_ks:
        stats[f"acc_at_{k}"] += int(acc_at_k(point_hits, k))


def summarize_group(stats: dict[str, int], acc_ks: list[int]) -> dict:
    total = stats["total"]
    total_points = stats["total_points"]
    return {
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


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    splits = parse_splits(args.split)
    acc_ks = parse_acc_ks(args.acc_k)
    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = RefSpatialBenchEvaluator(args)

    overall = empty_group(acc_ks)
    by_split: dict[str, dict[str, int]] = defaultdict(lambda: empty_group(acc_ks))
    by_step: dict[str, dict[str, int]] = defaultdict(lambda: empty_group(acc_ks))
    by_split_step: dict[str, dict[str, int]] = defaultdict(lambda: empty_group(acc_ks))

    def filtered_samples():
        seen = 0
        for sample in iter_refspatial_bench(dataset_root, splits, args.mask_threshold):
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
                update_point_stats(overall, point_hits, acc_ks)
                update_point_stats(by_split[sample["split"]], point_hits, acc_ks)
                update_point_stats(by_step[str(sample["step"])], point_hits, acc_ks)
                update_point_stats(
                    by_split_step[f"{sample['split']}/step_{sample['step']}"],
                    point_hits,
                    acc_ks,
                )

                sample_acc_at_k = {k: acc_at_k(point_hits, k) for k in acc_ks}
                record = {
                    "index": sample["index"],
                    "id": sample["id"],
                    "row_in_file": sample["row_in_file"],
                    "source_file": sample["source_file"],
                    "split": sample["split"],
                    "step": sample["step"],
                    "object": sample["object"],
                    "prompt": sample["prompt"],
                    "suffix": sample["suffix"],
                    "image_path": sample["image_path"],
                    "mask_path": sample["mask_path"],
                    "image_size": list(sample["image_size"]),
                    "mask_positive_pixels": sample["mask_positive_pixels"],
                    "mask_area_ratio": sample["mask_area_ratio"],
                    "mask_bbox": sample["mask_bbox"],
                    "prediction": pred["prediction"],
                    "raw_prediction": pred["raw_prediction"],
                    "raw_points": [list(point) for point in pred["raw_points"]],
                    "points": [list(point) for point in pred["points"]],
                    "point_hits": point_hits,
                    "sample_hit": any(point_hits),
                    **{f"point_acc_at_{k}": hit for k, hit in sample_acc_at_k.items()},
                    "prompt_style": args.prompt_style,
                    "coord_mode": args.coord_mode,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if overall["total"] % 25 == 0:
                    first_k = acc_ks[0]
                    print(
                        f"evaluated={overall['total']} "
                        f"point_acc_at_{first_k}="
                        f"{overall[f'acc_at_{first_k}'] / overall['total']:.4f} "
                        f"point_precision="
                        f"{(overall['hit_points'] / overall['total_points']) if overall['total_points'] else 0.0:.4f}",
                        flush=True,
                    )

    summary = {
        **summarize_group(overall, acc_ks),
        "official_success_rate": (
            overall.get("acc_at_1", 0) / overall["total"]
            if 1 in acc_ks and overall["total"]
            else 0.0
        ),
        "mask_threshold": args.mask_threshold,
        "by_split": {
            split: summarize_group(stats, acc_ks)
            for split, stats in sorted(by_split.items())
        },
        "by_step": {
            step: summarize_group(stats, acc_ks)
            for step, stats in sorted(by_step.items(), key=lambda item: int(item[0]))
        },
        "by_split_step": {
            key: summarize_group(stats, acc_ks)
            for key, stats in sorted(by_split_step.items())
        },
        "splits": splits,
        "prompt_style": args.prompt_style,
        "coord_mode": args.coord_mode,
        "output_jsonl": str(output_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
