# Copyright 2025 Xiaomi Corporation.

# Standard library
import os
from pathlib import Path

# Third-party
import yaml
from PIL import Image
from lmms_eval.filters.extraction import RobustChoiceFilter
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter

with open(Path(__file__).parent / "crpe_relation.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

vision_cache_dir = yaml.safe_load("".join(safe_data))["img_root"]
if isinstance(vision_cache_dir, str):
    vision_cache_dirs = [
        vision_cache_dir.replace("${MIMO_DATA_ROOT}", os.environ["MIMO_DATA_ROOT"])
    ]
else:
    vision_cache_dirs = [
        path.replace("${MIMO_DATA_ROOT}", os.environ["MIMO_DATA_ROOT"])
        for path in vision_cache_dir
    ]


def _resolve_image_path(image_path):
    for root in vision_cache_dirs:
        candidate = os.path.join(root, image_path)
        if os.path.exists(candidate):
            return candidate
    searched = ", ".join(vision_cache_dirs)
    raise FileNotFoundError(f"CRPE image {image_path!r} not found under: {searched}")


def _doc_value(doc, primary_key, fallback_key):
    if primary_key in doc:
        return doc[primary_key]
    return doc[fallback_key]


def crpe_doc_to_visual(doc):
    image_path = _resolve_image_path(_doc_value(doc, "image_path", "image"))
    image = Image.open(image_path).convert("RGB")
    return [image]


def crpe_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    post_prompt = (lmms_eval_specific_kwargs or {}).get("post_prompt", "")
    return _doc_value(doc, "question", "text") + post_prompt


def crpe_process_results(doc, results):
    prediction = results[0]
    final_answer = prediction.strip().lower()
    gt_answer = _doc_value(doc, "answer", "correct_option").strip().lower()

    return {"accuracy": 1.0 if final_answer == gt_answer else 0.0}


def crpe_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total
