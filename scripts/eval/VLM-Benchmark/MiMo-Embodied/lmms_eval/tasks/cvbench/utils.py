# Copyright 2025 Xiaomi Corporation.

import os, json
import re 
from typing import List
from lmms_eval.filters.extraction import RobustChoiceFilter
from lmms_eval.tasks._task_utils.eval_utils import BoxedFilter

def cvbench_doc_to_visual(doc):
    return [doc['image'].convert("RGB")]


def cvbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc["prompt"] + lmms_eval_specific_kwargs.get("post_prompt")


def cvbench_process_results(doc, results):
    prediction = results[0]
    answer = doc["answer"] 
    score = 0 

    # More flexible regex patterns to handle various formats
    # Pattern 1: Matches "Answer: (X)" or "Answer: X" or "Answer is (X)" etc.
    # Pattern 2: Matches "(X)" or "X)" or "X." or "X" where X is A-Z
    # Pattern 3: Matches "option X" or "X option" where X is A-Z
    # Pattern 4: Matches "answer is X" or "X is the answer" where X is A-Z
    pred_patterns = [
        r'Answer:?\s*[\(]?([A-Z])[\)]?',  # Pattern 1
        r'[\(]?([A-Z])[\)\.]?',           # Pattern 2
        r'option\s*([A-Z])',              # Pattern 3
        r'([A-Z])\s*option',              # Pattern 3
        r'answer\s*is\s*([A-Z])',         # Pattern 4
        r'([A-Z])\s*is\s*the\s*answer'    # Pattern 4
    ]
    

    pred_letter = None
    for pattern in pred_patterns:
        match = re.search(pattern, prediction, re.IGNORECASE)
        if match:
            pred_letter = match.group(1).upper()
            break
    

    ans_letter = None
    for pattern in pred_patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            ans_letter = match.group(1).upper()
            break
    
    if pred_letter and ans_letter:
        score = 1 if pred_letter == ans_letter else 0
    return {"accuracy": score} 


def cvbench_aggregate_results(results):
    correct = sum(results)
    total = len(results)
    return correct / total 

