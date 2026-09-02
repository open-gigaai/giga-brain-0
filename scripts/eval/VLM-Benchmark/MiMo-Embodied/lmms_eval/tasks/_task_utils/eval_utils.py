import re

def extract_final_boxed_content(text, strict=False):
    """
    Extracts the content of the final \\boxed{} command in the given text.
    
    Args:
        text (str): The text containing \\boxed{} commands
        
    Returns:
        str or None: The content of the final \\boxed{} command, or None if no \\boxed{} command is found
    """
    # Find all occurrences of \boxed{...} with regex
    # This handles one level of nested braces by using a non-greedy match
    boxed_matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    
    # Return the last match if any matches were found
    if boxed_matches:
        return boxed_matches[-1]
    else:
        if strict:
            return ""
        else:
            return text


def extract_after_think_content(text, strict=False):
    """
    Extracts the content after the last </think> tag in the given text.
    
    Args:
        text (str): The text containing </think> tags   
    
    Returns:
        str: The content after the last </think> tag, or the original text if no </think> tag is found
    """
    # Find the last occurrence of </think>
    last_think_end = text.rfind("</think>")
    if last_think_end != -1:
        return text[last_think_end + len("</think>"):].strip()
    elif "<think>" not in text:
        return text.strip()
    else:
        if strict:
            return ""
        else:
            return text


_POINT_PAIR_RE = re.compile(
    r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]\)]"
)


def extract_point_content(text, strict=False):
    """Extract complete model-emitted point pairs without changing x/y values."""
    text = extract_after_think_content(str(text), strict=strict)
    pairs = _POINT_PAIR_RE.findall(text)
    if not pairs:
        return "" if strict else text.strip()
    return "[" + ", ".join(f"({x}, {y})" for x, y in pairs) + "]"


def parse_bbox(input_str):
    """
    Extract a sequence of four floating-point numbers from a string in various formats.

    Args:
    input_str (str): A string that may contain a sequence of four floats in different formats.

    Returns:
    list: A list of four floats if any pattern is found, or a list of four zeros if no pattern is found.
    """
    input_str = input_str.lower()

    # Pattern 1: Four floats within square brackets
    pattern1 = r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]"
    match = re.search(pattern1, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    
    # Pattern 2: Four floats within parentheses
    pattern2 = r"\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\)"
    match = re.search(pattern2, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    
    # Pattern 3: Four floats with labels
    top_left_x = re.search(r"top-left x:?\s*(-?\d+(?:\.\d+)?)", input_str)
    top_left_y = re.search(r"top-left y:?\s*(-?\d+(?:\.\d+)?)", input_str)
    bottom_right_x = re.search(r"bottom-right x:?\s*(-?\d+(?:\.\d+)?)", input_str)
    bottom_right_y = re.search(r"bottom-right y:?\s*(-?\d+(?:\.\d+)?)", input_str)
    
    if top_left_x and top_left_y and bottom_right_x and bottom_right_y:
        return [
            float(top_left_x.group(1)),
            float(top_left_y.group(1)),
            float(bottom_right_x.group(1)),
            float(bottom_right_y.group(1))
        ]
    
    # If no pattern matched, return the null float sequence
    return [0, 0, 0, 0]

def parse_bbox_from_point(input_str):
    """
    Extract a sequence of four floating-point numbers from a string in various formats.

    Args:
    input_str (str): A string that may contain a sequence of four floats in different formats.

    Returns:
    list: A list of four floats if any pattern is found, or a list of four zeros if no pattern is found.
    """
    input_str = input_str.lower()

    # Pattern 1: Two floats within square brackets
    pattern1 = r"\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]"
    match = re.search(pattern1, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 3)] * 2
    
    # Pattern 2: Two floats within parentheses
    pattern2 = r"\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\)"
    match = re.search(pattern2, input_str)
    if match:
        return [float(match.group(i)) for i in range(1, 3)] * 2
    
    return [0,0,0,0]


from lmms_eval.models.model_utils.qwen.vision_process import smart_resize
def normalize_bbox(bbox, width, height, resize_max_pixels=0):
    if any(x > 1 for x in bbox):
        if resize_max_pixels > 0:
            height, width = smart_resize(height=height, width=width, max_pixels=resize_max_pixels)
        bbox = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
    return bbox


def parse_point(input_str):
    """
    Extract a sequence of four floating-point numbers from a string in various formats.

    Args:
    input_str (str): A string that may contain a sequence of four floats in different formats.

    Returns:
    list: A list of four floats if any pattern is found, or a list of four zeros if no pattern is found.
    """
    point_content = extract_point_content(input_str, strict=True)
    return [
        [float(x), float(y)]
        for x, y in _POINT_PAIR_RE.findall(point_content)
    ]


def points_to_mask_coordinates(points, mask_shape):
    """Map benchmark-defined normalized points to the scorer image grid."""
    height, width = mask_shape[:2]
    return [
        [int(round(float(x) * width)), int(round(float(y) * height))]
        for x, y in points
    ]


def resize_mask(mask, resize_max_pixels=0):
    if resize_max_pixels <= 0:
        return mask
    width, height = mask.size
    h_bar, w_bar = smart_resize(height=height, width=width, max_pixels=resize_max_pixels)
    mask = mask.resize((w_bar, h_bar))

    return mask



from lmms_eval.filters.extraction import ExtendedRegexFilter
class BoxedFilter(ExtendedRegexFilter):
    def apply(self, resps, docs):
        filtered_resps = [[extract_final_boxed_content(r)][0] for resp in resps for r in resp]
        return filtered_resps


class StrictBoxedFilter(ExtendedRegexFilter):
    def apply(self, resps, docs):
        filtered_resps = [[extract_final_boxed_content(r, strict=True)][0] for resp in resps for r in resp]
        return filtered_resps


class AfterThinkFilter(ExtendedRegexFilter):
    def apply(self, resps, docs):
        filtered_resps = []
        for resp in resps:
            for r in resp:
                filtered_resps.append(extract_after_think_content(r))
        return filtered_resps


class PointFilter(ExtendedRegexFilter):
    """Keep every complete emitted point; do not recover or aggregate answers."""

    def apply(self, resps, docs):
        return [
            extract_point_content(response, strict=True)
            for response_set in resps
            for response in response_set
        ]
