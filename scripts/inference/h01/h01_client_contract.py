"""Pure data contract shared by the H01 smooth client and offline validators."""

from __future__ import annotations

from typing import Any


CLIENT_TO_CANONICAL_IMAGE_KEYS = {
    "cam_high": "observation.images.cam_high",
    "cam_left_wrist_up": "observation.images.cam_left_wrist",
    "cam_right_wrist_up": "observation.images.cam_right_wrist",
}


def build_unified_inference_request(
    observation: dict[str, Any],
    *,
    inference_seed: int | None = None,
) -> dict[str, Any]:
    """Convert one H01 smooth-client observation to the unified-server schema."""
    if not isinstance(observation, dict):
        raise TypeError(
            f"observation must be a dict, got {type(observation).__name__}"
        )
    for key in ("prompt", "state", "images"):
        if key not in observation:
            raise KeyError(f"observation is missing required field {key!r}")
    if not isinstance(observation["images"], dict):
        raise TypeError("observation['images'] must be a dict")

    request = {
        "task": observation["prompt"],
        "observation.state": observation["state"],
    }
    missing_images = []
    for client_key, canonical_key in CLIENT_TO_CANONICAL_IMAGE_KEYS.items():
        if client_key not in observation["images"]:
            missing_images.append(client_key)
            continue
        request[canonical_key] = observation["images"][client_key]
    if missing_images:
        raise KeyError(
            "observation is missing required H01 image keys: "
            + ", ".join(missing_images)
        )
    if inference_seed is not None:
        request["inference_seed"] = inference_seed
    return request
