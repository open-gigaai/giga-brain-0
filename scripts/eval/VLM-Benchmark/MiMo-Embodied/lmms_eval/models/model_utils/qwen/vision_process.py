"""Image resize helpers used by MiMo-Embodied point scorers."""

import math


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 16384 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Resize dimensions while preserving aspect ratio and factor alignment."""
    ratio = max(height, width) / min(height, width)
    if ratio > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {ratio}")

    resized_height = max(factor, round_by_factor(height, factor))
    resized_width = max(factor, round_by_factor(width, factor))
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_height = floor_by_factor(height / scale, factor)
        resized_width = floor_by_factor(width / scale, factor)
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = ceil_by_factor(height * scale, factor)
        resized_width = ceil_by_factor(width * scale, factor)
    return resized_height, resized_width
