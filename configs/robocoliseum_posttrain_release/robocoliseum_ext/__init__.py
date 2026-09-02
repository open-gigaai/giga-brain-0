"""Public interfaces for using RoboColiseum 3.0 data in GigaBrain-0.7 post-training.

This package exports the raw tensor dimensions, joint indices, repacking helpers,
and training transform. Importing ``robocoliseum_ext`` registers the custom transform
so the configuration system can construct it by name.
"""

from .state_remap import (
    ACTION_GATHER_INDEX,
    ACTION_WAIST_JOINT_5_INDEX,
    ROBOCOLISEUM_ACTION_DIM,
    ROBOCOLISEUM_STATE_DIM,
    ARM_GRIPPER_DIM,
    UNIFIED_ACTION_DIM,
    STATE_GATHER_INDEX,
    STATE_WAIST_JOINT_5_INDEX,
    RoboColiseumJointActionRemapTransform,
    build_action_gather_index,
    build_delta_mask,
    build_state_gather_index,
    repack_action_state,
)


# Keep the public API explicit so downstream configs do not depend on internals.
__all__ = [
    'ACTION_GATHER_INDEX',
    'ACTION_WAIST_JOINT_5_INDEX',
    'ROBOCOLISEUM_ACTION_DIM',
    'ROBOCOLISEUM_STATE_DIM',
    'ARM_GRIPPER_DIM',
    'UNIFIED_ACTION_DIM',
    'STATE_GATHER_INDEX',
    'STATE_WAIST_JOINT_5_INDEX',
    'RoboColiseumJointActionRemapTransform',
    'build_action_gather_index',
    'build_delta_mask',
    'build_state_gather_index',
    'repack_action_state',
]
