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


_THINK_ANSWER_SUFFIX = (
    "You FIRST think about the reasoning process as an internal monologue and then "
    "provide the final answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags. The answer consists only of several "
    "coordinate points, with the overall format being: <think> reasoning process here "
    "</think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>"
)


def check_points_in_bbox(points, bbox):
    x1, y1, x2, y2 = bbox

    points_in_box = 0
    for point in points:
        if len(point) == 2:
            x, y = point
            if x1 <= x <= x2 and y1 <= y <= y2:
                points_in_box += 1

    return points_in_box, len(points)


def vabench_point_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def vabench_point_doc_to_text_mimo(doc):
    question = doc["problem"]
    ref_exp = question.replace("<image> ", "")
    ref_exp = ref_exp.replace("The task instruction is: ", "")
    ref_exp = ref_exp.replace(_THINK_ANSWER_SUFFIX, "")
    return PROMPT_MIMO_EMBODIED.format(ref_exp=ref_exp)


def vabench_point_process_results(doc, result):
    pred = result[0] if len(result) > 0 else ""
    pred = extract_after_think_content(pred, strict=True)
    if len(pred) == 0:
        return {"accuracy": 0}

    points = parse_point(pred)
    if not points:
        return {"accuracy": 0}

    img = resize_mask(doc["image"], int(os.getenv("QWEN_RESIZE_MAX_PIXELS", 0)))
    img = np.array(img)
    normalized_bbox = doc["normalized_bbox"]

    H, W, _ = img.shape
    points = np.array(points_to_mask_coordinates(points, img.shape))

    norm_x_min, norm_y_min, norm_x_max, norm_y_max = normalized_bbox

    abs_x_min = norm_x_min * W
    abs_y_min = norm_y_min * H
    abs_x_max = norm_x_max * W
    abs_y_max = norm_y_max * H

    unnormalized_bbox = [
        int(abs_x_min),
        int(abs_y_min),
        int(abs_x_max),
        int(abs_y_max),
    ]
    points_in_box, num_points = check_points_in_bbox(points, unnormalized_bbox)

    if num_points == 0:
        return {"accuracy": 0}

    acc = points_in_box / num_points
    return {"accuracy": acc}


def vabench_point_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total
