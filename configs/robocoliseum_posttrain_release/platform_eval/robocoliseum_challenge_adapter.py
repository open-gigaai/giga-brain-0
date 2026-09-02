"""Protocol conversion helpers for the RoboColiseum challenge agent."""

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np


MODEL_STATE_DIM = 17
MODEL_ACTION_DIM = 16

# The checkpoint was trained with these canonical GigaBrain camera keys.
CAMERA_KEY_MAP = {
    'head': 'observation.images.top_head',
    'hand_left': 'observation.images.hand_left',
    'hand_right': 'observation.images.hand_right',
}


def _as_vector(value: Any, *, name: str, expected_dim: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size != expected_dim:
        raise ValueError(f'{name} must contain {expected_dim} values, got {vector.size}')
    return vector


def build_model_state(params: Mapping[str, Any]) -> np.ndarray:
    """Convert the gateway's 21-D G2 state into the checkpoint's 17-D layout."""
    states = params['states']
    arms = _as_vector(states['arm_joint_states'], name='arm_joint_states', expected_dim=14)
    grippers = _as_vector(states['gripper_states'], name='gripper_states', expected_dim=2)
    waist = _as_vector(states['waist_joint_states'], name='waist_joint_states', expected_dim=5)

    # Training order: left arm, left gripper, right arm, right gripper, waist joint 5.
    model_state = np.concatenate(
        (arms[:7], grippers[:1], arms[7:], grippers[1:], waist[4:5])
    )
    if model_state.size != MODEL_STATE_DIM:
        raise AssertionError(f'internal state layout produced {model_state.size} values')
    return model_state


def decode_model_images(params: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Decode gateway JPEG frames into RGB HWC uint8 arrays keyed for GigaBrain."""
    source_images = params['images']
    decoded: dict[str, np.ndarray] = {}
    for source_key, model_key in CAMERA_KEY_MAP.items():
        frame = source_images[source_key]
        encoding = str(frame.get('encoding', '')).upper()
        if encoding not in {'JPEG', 'JPG'}:
            raise ValueError(f'{source_key} uses unsupported encoding {encoding!r}')

        encoded = np.frombuffer(frame['image_data'], dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f'failed to decode JPEG camera {source_key!r}')
        decoded[model_key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return decoded


def build_model_observation(params: Mapping[str, Any]) -> dict[str, Any]:
    """Build the model-facing observation without depending on the tunnel SDK."""
    return {
        'images': decode_model_images(params),
        'state': build_model_state(params),
        'task': str(params.get('prompt', '')),
    }


def build_action_response(actions: Any) -> dict[str, Any]:
    """Convert a GigaBrain action chunk into the challenge result envelope."""
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] < MODEL_ACTION_DIM:
        raise ValueError(
            f'actions must have shape [H, D] with D >= {MODEL_ACTION_DIM}, got {chunk.shape}'
        )
    if not np.isfinite(chunk[:, :MODEL_ACTION_DIM]).all():
        raise ValueError('actions contain non-finite values')

    # GigaBrain order is L-arm, L-gripper, R-arm, R-gripper, unlike the
    # flat layout used by the reference pi0.5 adapter.
    return {
        'result': {
            'left_arm': {
                'kind': 'JOINT_ABS',
                'values': chunk[:, 0:7].tolist(),
            },
            'right_arm': {
                'kind': 'JOINT_ABS',
                'values': chunk[:, 8:15].tolist(),
            },
            'left_effector': chunk[:, 7:8].tolist(),
            'right_effector': chunk[:, 15:16].tolist(),
        }
    }
