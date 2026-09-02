# Copyright 2025 Xiaomi Corporation.

# Standard library
import os
import re

# Third-party
import numpy as np
from loguru import logger as eval_logger
from lmms_eval.filters.extraction import RobustYesNoFilter

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

def robospatial_doc_to_visual(doc):
    image = doc.get("image") or doc.get("img")
    return [image.convert("RGB")]


def robospatial_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc["question"] + lmms_eval_specific_kwargs.get("post_prompt")


def robospatial_doc_to_text_point(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    ANSWER_SUFFIX = " Your answer should be formatted as a list of tuples"
    ref_exp, _, _ = question.partition(ANSWER_SUFFIX)
    return PROMPT_MIMO_EMBODIED.format(ref_exp=ref_exp)


def robospatial_process_results_QA(doc, results):
    prediction = results[0]
    final_answer = prediction.lower()
    gt_answer = doc["answer"].lower()

    if final_answer not in ["yes", "no"]:
        eval_logger.warning(f"Not a legal answer: {prediction}")
        return {"accuracy": 0.0}

    return {"accuracy": 1.0 if final_answer == gt_answer else 0.0}


def robospatial_process_results_QA_no_format(doc, results):
    prediction = str(results[0]).strip()
    match = re.match(
        r"^(?:\\boxed\s*\{\s*)?(?:the\s+)?(?:final\s+)?"
        r"(?:answer\s*(?:is|:)\s*)?(yes|no)\b",
        prediction,
        flags=re.IGNORECASE,
    )
    if not match:
        eval_logger.warning(f"Not a legal no-format answer: {prediction}")
        return {"accuracy": 0.0}
    final_answer = match.group(1).lower()
    gt_answer = str(doc["answer"]).strip().lower()
    return {"accuracy": 1.0 if final_answer == gt_answer else 0.0}


def robospatial_process_results_point(doc, result):
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


def robospatial_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total
