"""Repack RoboColiseum 3.0 state and action tensors into GigaBrain joint space.

The ``g2a_sim`` datasets provide 109-D states and 38-D actions. Their
``field_descriptions`` place grippers, end-effector poses, arm joints, head
joints, and waist joints in separate index ranges. GigaBrain-0.7 expects this
AgiBot G1 action order:

    [left arm joints x7, left gripper, right arm joints x7, right gripper,
     waist joint 5]

Every suite is repacked into a unified 17-D tensor. GigaBrain reads
``is_body_moving`` from dataset metadata: false supervises the first 16
dimensions, while true also supervises waist joint 5 in dimension 17.

End-effector poses are deliberately excluded. States contain both ``base_link``
and ``arm_base_link`` poses, but actions use ``arm_base_link`` only. Joint-space
training avoids an unnecessary and error-prone coordinate conversion.
"""

from typing import Any

import torch
from giga_train import TRANSFORMS


ROBOCOLISEUM_STATE_DIM = 109
ROBOCOLISEUM_ACTION_DIM = 38
ARM_GRIPPER_DIM = 16
UNIFIED_ACTION_DIM = 17

# These indices come from field_descriptions in every g2a_sim info.json.
# The target order is left arm, left gripper, right arm, right gripper, waist joint 5.
STATE_GATHER_INDEX = (
    *range(30, 37),
    0,
    *range(37, 44),
    1,
    65,
)
ACTION_GATHER_INDEX = (
    *range(16, 23),
    0,
    *range(23, 30),
    1,
    37,
)
STATE_WAIST_JOINT_5_INDEX = 65
ACTION_WAIST_JOINT_5_INDEX = 37


def build_state_gather_index() -> list[int]:
    """Build indices that extract the unified 17-D layout from a 109-D state."""
    return list(STATE_GATHER_INDEX)


def build_action_gather_index() -> list[int]:
    """Build indices that extract the unified 17-D layout from a 38-D action."""
    return list(ACTION_GATHER_INDEX)


def build_delta_mask() -> list[bool]:
    """Build the delta-action mask for the unified 7+1+7+1+1 layout."""
    return [True] * 7 + [False] + [True] * 7 + [False, True]


def _repack_tensor(
    value: Any,
    *,
    name: str,
    raw_dim: int,
    gather_index: list[int],
    strict: bool,
) -> torch.Tensor:
    """Extract and reorder one state or action tensor along its last axis.

    Inputs may have arbitrary leading dimensions, such as state shape ``[D]``
    or action-chunk shape ``[T, D]``. Tensors that already have the target width
    are returned unchanged to prevent duplicate repacking.

    Args:
        value: State or action data accepted by ``torch.as_tensor``.
        name: Field name used in validation errors.
        raw_dim: Original RoboColiseum width for this field.
        gather_index: Original dimension indices in target order.
        strict: Reject widths other than the raw and repacked dimensions.
    """
    tensor = torch.as_tensor(value)
    # Leading axes may represent time or batches; only the final axis is repacked.
    width = int(tensor.shape[-1])
    target_dim = len(gather_index)
    if width == raw_dim:
        index = torch.tensor(gather_index, dtype=torch.long, device=tensor.device)
        # Keep indices on the input device for both CPU and GPU samples.
        return tensor.index_select(-1, index)
    if width == target_dim:
        # An upstream transform may already have repacked this tensor.
        return tensor
    if strict:
        raise ValueError(
            f'unexpected {name} width {width}; expected raw width {raw_dim} '
            f'or repacked width {target_dim}'
        )
    return tensor


def repack_action_state(
    data_dict: dict[str, Any],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Repack the state and action fields present in a sample in place.

    A sample may contain only ``observation.state`` or only ``action``; missing
    fields are not created. The dictionary identity is preserved while field
    values are replaced with repacked ``torch.Tensor`` objects.

    Args:
        data_dict: Sample dictionary passed through the transform pipeline.
        strict: Reject fields whose width is neither raw nor already repacked.

    Returns:
        The same dictionary object for continued transform composition.
    """
    if 'observation.state' in data_dict:
        data_dict['observation.state'] = _repack_tensor(
            data_dict['observation.state'],
            name='observation.state',
            raw_dim=ROBOCOLISEUM_STATE_DIM,
            gather_index=build_state_gather_index(),
            strict=strict,
        )
    if 'action' in data_dict:
        data_dict['action'] = _repack_tensor(
            data_dict['action'],
            name='action',
            raw_dim=ROBOCOLISEUM_ACTION_DIM,
            gather_index=build_action_gather_index(),
            strict=strict,
        )
    return data_dict


@TRANSFORMS.register
class RoboColiseumJointActionRemapTransform:
    """Repack RoboColiseum joints before applying the standard GigaBrain transform.

    This wrapper adapts the data layout only. It delegates image processing,
    prompts, delta actions, normalization, and padding to the official
    ``GigaBrain07Transform`` configured by ``inner``.
    """

    def __init__(
        self,
        inner: dict[str, Any],
        strict: bool = True,
        state_input_mode: str | None = None,
        observation_memory_size: int | None = None,
        prompt_cfg: dict[str, Any] | None = None,
        agent_pos_config: dict[str, int] | None = None,
    ):
        # Delay construction imports so module loading does not initialize training.
        from giga_train import build_transform

        # Outer configs mirror these values for trainer validation; the wrapped
        # transform owns runtime state handling, so ignoring them here is intentional.
        _ = state_input_mode, observation_memory_size, prompt_cfg, agent_pos_config
        # Build the callable inner transform through the training registry.
        self.inner = build_transform(inner)
        if self.inner is None:
            raise ValueError(f'failed to build inner transform from {inner!r}')
        self.strict = bool(strict)

    def __getattr__(self, name: str) -> Any:
        """Forward attributes not defined by the wrapper to the inner transform.

        The trainer directly reads attributes such as normalization statistics,
        so the wrapper preserves the inner interface. Special-casing ``inner``
        prevents recursion before initialization completes.
        """
        if name == 'inner':
            raise AttributeError(name)
        return getattr(self.__dict__['inner'], name)

    def __call__(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        """Repack joint fields, then apply the standard GigaBrain transform.

        Dataset metadata provides ``is_body_moving``; this wrapper does not
        override it.
        """
        repack_action_state(data_dict, strict=self.strict)
        return self.inner(data_dict)
