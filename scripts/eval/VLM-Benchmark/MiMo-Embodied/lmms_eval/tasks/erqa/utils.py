# Copyright 2025 Xiaomi Corporation.

# Standard library
import re
from pathlib import Path

# Third-party
import yaml

# Local
from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.tasks._task_utils.eval_utils import extract_final_boxed_content

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


def replace_images_tokens(input_string):
    for i in range(1, 16):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def erqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc.get("question_new", doc["question"])
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def erqa_doc_to_visual(doc):
    return [img.convert("RGB") for img in doc["images"]]


def erqa_doc_to_target(doc, model_specific_target_kwargs):
    if "options" not in doc:
        return doc["answer"]

    if model_specific_target_kwargs == "mcq":
        len_choices = len(doc["options"])
        options = [chr(ord("A") + i) for i in range(len_choices)]
        return options[int(doc["answer"])]
    elif model_specific_target_kwargs == "qa":
        return doc["options"][int(doc["answer"])]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            option_letter_regex = re.compile(r"^\s*([A-Z])\.")

            filtered = []
            for resp in r:
                match = option_letter_regex.match(resp)
                if match:
                    filtered.append(match.group(1))
                else:
                    filtered.append(resp)

            filtered_resps.append(filtered[0])

        return filtered_resps


class MultiChoiceBoxedRegexFilter(MultiChoiceRegexFilter):
    def apply(self, resps, docs):
        resps = [[extract_final_boxed_content(r)] for resp in resps for r in resp]
        filtered_resps = super().apply(resps, docs)
        return filtered_resps