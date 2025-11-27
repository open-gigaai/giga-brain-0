import json
from enum import IntEnum, StrEnum
from typing import Any

import torch
from giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils import (
    DeltaActions,
    ImageTransform,
    Normalize,
    PadStatesAndActions,
    PromptTokenizerTransform,
    TrajectoryTransform,
)
from giga_train import TRANSFORMS


class EmbodimentId(IntEnum):
    """Enumeration for robot embodiment IDs."""

    AGILEX = 0
    AGIBOT_G1 = 1
    AGIBOT_WORLD = 2


class RobotType(StrEnum):
    """Enumeration for robot types."""

    AGILEX_COBOT_MAGIC = 'agilex_cobot_magic'
    AGIBOT_G1 = 'agibot_g1'
    AGIBOT_WORLD = 'agibot_world'


robot_type_mapping: dict[RobotType, EmbodimentId] = {
    RobotType.AGILEX_COBOT_MAGIC: EmbodimentId.AGILEX,
    RobotType.AGIBOT_G1: EmbodimentId.AGIBOT_G1,
    RobotType.AGIBOT_WORLD: EmbodimentId.AGIBOT_WORLD,
}


@TRANSFORMS.register
class GigaBrain0Transform:
    """A class to transform raw data into a format suitable for GigaBrain0
    model training."""

    def __init__(
        self,
        delta_action_cfg: dict[str, Any] | None = None,
        norm_cfg: dict[str, Any] | None = None,
        traj_cfg: dict[str, Any] | None = None,
        image_cfg: dict[str, Any] | None = None,
        prompt_cfg: dict[str, Any] | None = None,
        is_train: bool = True,
    ):
        """Initializes the transform pipeline.

        Args:
            delta_action_cfg (dict[str, Any] | None, optional): Configuration for delta actions. Defaults to None.
            norm_cfg (dict[str, Any] | None, optional): Configuration for normalization. Defaults to None.
            traj_cfg (dict[str, Any] | None, optional): Configuration for trajectory transform. Defaults to None.
            image_cfg (dict[str, Any] | None, optional): Configuration for image transform. Defaults to None.
            prompt_cfg (dict[str, Any] | None, optional): Configuration for prompt tokenizer. Defaults to None.
            is_train (bool, optional): Whether the transform is used for training. Defaults to True.
        """
        self.is_train = is_train
        self.use_delta_joint_actions = delta_action_cfg is not None and delta_action_cfg.get('use_delta_joint_actions', False)

        if self.use_delta_joint_actions:
            assert delta_action_cfg is not None
            self.delta_action_transform = DeltaActions(delta_action_cfg['mask'])

        self.pad_transform = PadStatesAndActions(action_dim=32)

        state_norm_stats_data_dict: dict[str, Any] = dict()
        assert norm_cfg is not None, 'norm_cfg is required'
        for key in norm_cfg['norm_stats_path']:
            norm_stats_path = norm_cfg['norm_stats_path'][key]
            with open(norm_stats_path, 'r') as f:
                state_norm_stats_data = json.load(f)['norm_stats']['observation.state']
            state_norm_stats_data_dict[key] = state_norm_stats_data
        self.state_normalize_transform = Normalize(
            state_norm_stats_data_dict,
            use_quantiles=norm_cfg['use_quantiles'],
            enable_clamp=norm_cfg.get('enable_clamp', False),
        )

        action_norm_stats_data_dict: dict[str, Any] = dict()
        for key in norm_cfg['norm_stats_path']:
            norm_stats_path = norm_cfg['norm_stats_path'][key]
            with open(norm_stats_path, 'r') as f:
                action_norm_stats_data = json.load(f)['norm_stats']['action']
            action_norm_stats_data_dict[key] = action_norm_stats_data
        self.action_normalize_transform = Normalize(
            action_norm_stats_data_dict,
            use_quantiles=norm_cfg['use_quantiles'],
            enable_clamp=norm_cfg.get('enable_clamp', False),
        )

        assert prompt_cfg is not None, 'prompt_cfg is required'
        self.prompt_tokenizer_transform = PromptTokenizerTransform(**prompt_cfg, is_train=is_train)

        assert image_cfg is not None, 'image_cfg is required'
        self.image_transform = ImageTransform(**image_cfg, is_train=is_train)

        self.trajectory_transform = None
        if traj_cfg is not None:
            self.trajectory_transform = TrajectoryTransform(**traj_cfg)

    def __call__(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        """Applies the transformation pipeline to a data dictionary.

        Args:
            data_dict (dict[str, Any]): The input data dictionary.

        Returns:
            dict[str, Any]: The transformed data dictionary.
        """
        output_dict: dict[str, Any] = {}

        robot_type = data_dict['meta'].info['robot_type']
        embodiment_id = int(robot_type_mapping[robot_type])
        data_dict['embodiment_id'] = embodiment_id

        if self.use_delta_joint_actions:
            data_dict = self.delta_action_transform(data_dict)

        data_dict['observation.state'] = self.state_normalize_transform(data_dict['observation.state'], embodiment_id=embodiment_id)
        data_dict['action'] = self.action_normalize_transform(data_dict['action'], embodiment_id=embodiment_id)

        (
            output_dict['lang_tokens'],
            output_dict['lang_masks'],
            output_dict['lang_att_masks'],
            output_dict['lang_loss_masks'],
            output_dict['fast_action_indicator'],
            predict_subtask,
        ) = self.prompt_tokenizer_transform(data_dict)

        data_dict = self.pad_transform(data_dict)
        output_dict['observation.state'] = data_dict['observation.state']
        output_dict['action'] = data_dict['action']

        output_dict['images'], output_dict['image_masks'], image_transform_params = self.image_transform(data_dict)

        if self.trajectory_transform is not None:
            traj, traj_is_pad = self.trajectory_transform(
                data_dict, chunk_size=data_dict['action'].shape[0], image_transform_params=image_transform_params
            )
            output_dict['traj'] = traj
            output_dict['traj_loss_mask'] = ~traj_is_pad

        output_dict['action_loss_mask'] = ~data_dict['action_is_pad']
        if predict_subtask:
            # No diffusion loss for subtask prediction
            output_dict['action_loss_mask'] = torch.zeros_like(output_dict['action_loss_mask'])

        output_dict['embodiment_id'] = torch.tensor(embodiment_id, dtype=torch.long)

        return output_dict
