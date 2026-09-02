# Copyright 2025 Xiaomi Corporation.

from io import BytesIO

from PIL import Image

from lmms_eval.filters.extraction import RobustChoiceFilter
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter


def _to_rgb_image(image):
    if hasattr(image, "convert"):
        return image.convert("RGB")
    if isinstance(image, dict):
        image = image.get("bytes") or image.get("path")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(BytesIO(image)).convert("RGB")
    return Image.open(image).convert("RGB")


def sat_doc_to_visual(doc):
    if "images" in doc and doc["images"]:
        return [_to_rgb_image(img) for img in doc["images"]]
    return [_to_rgb_image(img) for img in doc["image_bytes"]]


def sat_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    answers = doc.get("answers")
    if answers:
        choices = "\n".join(f"{chr(ord('A') + idx)}. {answer}" for idx, answer in enumerate(answers))
        question = f"{question}\nOptions:\n{choices}"
    post_prompt = (lmms_eval_specific_kwargs or {}).get("post_prompt", "")
    return question + post_prompt


def sat_process_results(doc, results):
    prediction = results[0]
    final_answer = prediction.lower().strip()
    gt_answer = doc.get("answer", doc.get("correct_answer", "")).lower().strip()

    answers = doc.get("answers") or []
    if len(final_answer) == 1 and final_answer.isalpha() and answers:
        option_idx = ord(final_answer.upper()) - ord("A")
        if 0 <= option_idx < len(answers):
            final_answer = answers[option_idx].lower().strip()

    return {"accuracy": 1.0 if final_answer == gt_answer else 0.0}


def sat_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total
