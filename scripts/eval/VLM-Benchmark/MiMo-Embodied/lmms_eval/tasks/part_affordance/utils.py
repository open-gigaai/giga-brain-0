# Copyright 2025 Xiaomi Corporation.

# Standard library
import os

# Third-party
import numpy as np

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

def partaffordance_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def partaffordance_doc_to_text_mimo(doc):
    question = doc["problem"]
    ANSWER_SUFFIX = " Your answer should be formatted as a list of tuples..."
    ref_exp, _, _ = question.partition(ANSWER_SUFFIX)
    return PROMPT_MIMO_EMBODIED.format(ref_exp=ref_exp)


def partaffordance_process_results(doc, result):
    pred = result[0] if len(result) > 0 else ""
    pred = extract_after_think_content(pred, strict=True)
    if len(pred) == 0:
        return {"accuracy": 0}

    points = parse_point(pred)
    if not points:
        return {"accuracy": 0}
    mask = resize_mask(doc["mask"], int(os.getenv("QWEN_RESIZE_MAX_PIXELS", 0)))
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


def partaffordance_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total
