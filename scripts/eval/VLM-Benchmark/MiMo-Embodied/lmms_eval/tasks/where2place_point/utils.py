# Copyright 2025 Xiaomi Corporation.

# Standard library
import os
from pathlib import Path

# Third-party
import numpy as np
import yaml
from PIL import Image

# Local
from lmms_eval.tasks._task_utils.eval_utils import (
    BoxedFilter,
    PointFilter,
    extract_after_think_content,
    parse_point,
    points_to_mask_coordinates,
    resize_mask,
)

PROMPT_MIMO_EMBODIED = "Based on the description: \"{ref_exp}\", locate points matching the description. Output a JSON in the format [{{\"points\": [...], \"label\": \"{{the_whole_description}}\"}}, ...]."

with open(Path(__file__).parent / "where2place_point.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for line in raw_data:
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

vision_cache_dir = config.get("img_root", "").replace(
    "${MIMO_DATA_ROOT}", os.environ["MIMO_DATA_ROOT"]
)


def _open_local_image(doc, key, subdir):
    image = doc[key]
    if hasattr(image, "convert"):
        return image.convert("RGB")

    image_path = Path(image)
    if not image_path.is_absolute():
        image_path = Path(vision_cache_dir) / subdir / image_path
    return Image.open(image_path).convert("RGB")


def where2place_doc_to_visual(doc):
    return [_open_local_image(doc, "image", "images")]


def where2place_doc_to_text_mimo(doc):
    question = doc.get("question", doc.get("text", ""))
    ANSWER_SUFFIX = " Your answer should be formatted as a list of tuples"
    ref_exp, _, _ = question.partition(ANSWER_SUFFIX)
    return PROMPT_MIMO_EMBODIED.format(ref_exp=ref_exp)


def where2place_process_results(doc, result):
    pred = result[0] if len(result) > 0 else ""
    pred = extract_after_think_content(pred, strict=True)
    if len(pred) == 0:
        return {"accuracy": 0}

    points = parse_point(pred)
    if not points:
        return {"accuracy": 0}
    if "mask" in doc:
        mask = doc["mask"]
    else:
        image_name = Path(doc["image"]).stem
        mask_path = Path(vision_cache_dir) / "masks" / f"{image_name}.png"
        if not mask_path.exists():
            mask_path = Path(vision_cache_dir) / "masks" / f"{image_name}.jpg"
        mask = Image.open(mask_path)
    mask = resize_mask(mask, int(os.getenv("QWEN_RESIZE_MAX_PIXELS", 0)))
    mask = np.array(mask) / 255.0
    mask = (mask > 0).astype(np.uint8)
    points = np.array(points_to_mask_coordinates(points, mask.shape))

    in_range = (
        (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1])
        & (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
    )
    if sum(in_range) > 0:
        acc = np.concatenate([
            mask[points[in_range, 1], points[in_range, 0]],
            np.zeros(points.shape[0] - in_range.sum()),
        ]).mean()
    else:
        acc = 0

    return {"accuracy": acc}


def where2place_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total
