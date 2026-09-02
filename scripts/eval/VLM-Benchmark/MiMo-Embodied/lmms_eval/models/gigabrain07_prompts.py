import json
import re


_POINT_JSON_OUTPUT_FORMAT = (
    'Output a JSON in the format [{"points": [...], '
    '"label": "{the_whole_description}"}, ...].'
)
_POINT_PROMPT_PREFIX = 'Based on the description: "'
_POINT_PROMPT_SUFFIX = (
    '", locate points matching the description. ' + _POINT_JSON_OUTPUT_FORMAT
)
_BOXED_OUTPUT_FORMAT_RE = re.compile(
    r"\s*Put your final answer in\s+\\boxed\{\}\.?",
    flags=re.IGNORECASE,
)


def strip_benchmark_output_format(question: str) -> str:
    """Remove MiMo's shared output wrappers while preserving task semantics."""
    question = str(question).replace(_POINT_JSON_OUTPUT_FORMAT, "")
    question = _BOXED_OUTPUT_FORMAT_RE.sub("", question)
    return question.strip()


def format_training_vqa_prompt(question: str) -> str:
    """Build the exact default prompt used by the policy's VQA training transform."""
    return f"Question: {str(question).strip()}\nAnswer:"


def format_strict_point_2dp_prompt(question: str) -> str:
    """Rewrite MiMo's shared point prompt with an explicit coordinate contract."""
    question = str(question).strip()
    if not (
        question.startswith(_POINT_PROMPT_PREFIX)
        and question.endswith(_POINT_PROMPT_SUFFIX)
    ):
        raise ValueError(
            "Strict point prompting requires MiMo's shared point prompt format; "
            f"got: {question[:160]!r}"
        )

    description = question[
        len(_POINT_PROMPT_PREFIX) : -len(_POINT_PROMPT_SUFFIX)
    ]
    quoted_description = json.dumps(description)
    return (
        f"Based on the description: {quoted_description}, locate the matching point "
        "locations in the image. Output ONLY a JSON array in exactly this format: "
        f'[{{"points": [[x, y]], "label": {quoted_description}}}]. '
        "Each point MUST contain exactly two numbers [x, y]: x is horizontal and y "
        "is vertical. Both x and y must be normalized numbers from 0 to 1 relative "
        "to the input image. Every x and y value must contain exactly two digits "
        "after the decimal point, for example [0.25, 0.75]. If several locations "
        'match, put multiple two-number points inside "points". Do NOT output '
        "four-number bounding boxes, coordinate ranges, <loc> tokens, markdown, or "
        "explanations."
    )


def _point_description(question: str, prompt_name: str) -> str:
    question = str(question).strip()
    if not (
        question.startswith(_POINT_PROMPT_PREFIX)
        and question.endswith(_POINT_PROMPT_SUFFIX)
    ):
        raise ValueError(
            f"{prompt_name} requires MiMo's shared point prompt format; "
            f"got: {question[:160]!r}"
        )
    return question[len(_POINT_PROMPT_PREFIX) : -len(_POINT_PROMPT_SUFFIX)]


def _format_single_point_prompt(question: str, task_rule: str, prompt_name: str) -> str:
    description = _point_description(question, prompt_name)
    quoted_description = json.dumps(description)
    return (
        f"Based on the description: {quoted_description}, {task_rule} "
        "Return exactly one normalized point well inside the requested region. "
        "Output ONLY this JSON: "
        f'[{{"points": [[x, y]], "label": {quoted_description}}}]. '
        "The horizontal x and vertical y must each be between 0 and 1 with exactly "
        "two digits after the decimal point. Do not output extra points, bounding "
        "boxes, coordinate ranges, <loc> tokens, markdown, or explanations."
    )


def format_refit_boundary_interior_prompt(question: str) -> str:
    description = _point_description(question, "Refit boundary-interior prompting")
    quoted_description = json.dumps(description)
    return (
        f"Based on the description: {quoted_description}, locate the single referred "
        "object and select exactly one normalized [x, y] point well inside its "
        "visible area, away from its boundary. Output ONLY a JSON array in this "
        "format: "
        f'[{{"points": [[x, y]], "label": {quoted_description}}}]. '
        "Do not output extra points, boxes, or explanations."
    )


def format_single_point_2dp_prompt(question: str) -> str:
    question = str(question).strip()
    _point_description(question, "Single-point prompting")
    return (
        question
        + " Return exactly one normalized [x, y] point that satisfies the "
        "description and lies well inside the requested region. Use exactly two "
        "digits after the decimal point. Do not output extra points, four-number "
        "bounding boxes, coordinate ranges, <loc> tokens, or explanations."
    )


def format_affordance_adaptive_single_point_prompt(question: str) -> str:
    return _format_single_point_prompt(
        question,
        "interpret the request literally: for a functional part, point inside that "
        "part; for a referred object, point inside that object; for vacant or free "
        "space, point inside that region.",
        "Affordance-adaptive single-point prompting",
    )


def ensure_prompt_fits(prompt_length: int, max_length: int) -> None:
    if prompt_length >= max_length:
        raise ValueError(
            f"Prompt length {prompt_length} exceeds max length {max_length}; "
            "refusing to silently truncate the question or answer instruction"
        )
