# Copyright 2025 Xiaomi Corporation.

# Standard library
import json
import os
from pathlib import Path

# Third-party
import yaml
from PIL import Image
from lmms_eval.filters.extraction import RobustChoiceFilter
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter

with open(Path(__file__).parent / "metavqa_eval.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

vision_cache_dir = yaml.safe_load("".join(safe_data))["img_root"].replace(
    "${MIMO_DATA_ROOT}", os.environ["MIMO_DATA_ROOT"]
)


def _doc_image_path(doc):
    if "image_path" in doc:
        return doc["image_path"]

    obs = doc["obs"]
    if isinstance(obs, str):
        return obs
    return obs[0]


def _parse_options(options):
    if isinstance(options, str):
        options = options.strip()
        if not options:
            return {}
        try:
            return json.loads(options)
        except json.JSONDecodeError:
            return {}
    return options or {}


def _normalize_answer(answer, options=None):
    options = _parse_options(options)
    text = str(answer).strip().lower()
    if len(text) == 1 and text.isalpha():
        return text
    if len(text) > 1 and text[0].isalpha() and text[1] in ".):":
        return text[0]

    if isinstance(options, dict):
        for key, value in options.items():
            if text == str(key).strip().lower() or text == str(value).strip().lower():
                return str(key).strip().lower()
    elif isinstance(options, list):
        for idx, value in enumerate(options):
            if text == str(value).strip().lower():
                return chr(ord("a") + idx)

    return text


def metavqa_doc_to_visual(doc):
    image_path = os.path.join(vision_cache_dir, _doc_image_path(doc))
    image = Image.open(image_path).convert("RGB")
    return [image]


def metavqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    post_prompt = (lmms_eval_specific_kwargs or {}).get("post_prompt", "")
    return doc["question"] + post_prompt


def metavqa_process_results(doc, results):
    prediction = results[0]
    options = doc.get("options")
    final_answer = _normalize_answer(prediction, options)
    gt_answer = _normalize_answer(doc["answer"], options)

    return {"accuracy": 1.0 if final_answer == gt_answer else 0.0}


def metavqa_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total
