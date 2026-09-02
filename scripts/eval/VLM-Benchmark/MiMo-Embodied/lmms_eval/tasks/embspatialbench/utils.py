# Copyright 2025 Xiaomi Corporation.

from lmms_eval.filters.extraction import RobustChoiceFilter
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter

def embspatial_doc_to_visual(doc):
    return [doc['image'].convert("RGB")]


def _option_letter(idx):
    return chr(ord("A") + idx)


def _doc_answer_to_letter(doc):
    if "answer_letter" in doc:
        return str(doc["answer_letter"]).strip().lower()

    answer = doc["answer"]
    if isinstance(answer, int):
        return _option_letter(answer).lower()

    answer_text = str(answer).strip()
    if answer_text.isdigit():
        return _option_letter(int(answer_text)).lower()

    options = doc.get("answer_options") or []
    for idx, option in enumerate(options):
        if answer_text.lower() == str(option).strip().lower():
            return _option_letter(idx).lower()

    return answer_text.lower()


def _prediction_to_letter(prediction, options):
    final_answer = prediction.strip().lower()
    if len(final_answer) == 1 and final_answer.isalpha():
        return final_answer
    if len(final_answer) > 1 and final_answer[0].isalpha() and final_answer[1] in ".):":
        return final_answer[0]

    for idx, option in enumerate(options):
        if final_answer == str(option).strip().lower():
            return _option_letter(idx).lower()

    return final_answer


def embspatial_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    options = doc.get("answer_options") or doc.get("options") or []
    if options:
        choices = "\n".join(f"{_option_letter(idx)}. {option}" for idx, option in enumerate(options))
        question = f"{question}\nOptions:\n{choices}"
    post_prompt = (lmms_eval_specific_kwargs or {}).get("post_prompt", "")
    return question + post_prompt


def embspatial_process_results(doc, results):
    prediction = results[0]
    options = doc.get("answer_options") or doc.get("options") or []
    final_answer = _prediction_to_letter(prediction, options)
    gt_answer = _doc_answer_to_letter(doc)

    if final_answer == gt_answer:
        return {"accuracy": 1.0}
    else:
        return {"accuracy": 0.0}


def embspatial_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total 

