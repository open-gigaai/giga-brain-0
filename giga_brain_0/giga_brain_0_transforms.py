import os
import json
import math
from enum import IntEnum, StrEnum
from typing import Any

import torch
from giga_train import TRANSFORMS

from giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils import (
    ActionStateDimLayout,
    DeltaActions,
    Embodiment3QuaternionTo6D,
    ImageTransform,
    Normalize,
    PadStatesAndActions,
    PromptTokenizerTransform,
    TrajectoryTransform,
    infer_end_effector_type_from_delta_mask,
    reframe_dual_hand_tcp_chunk_to_anchor_camera,
    resolve_action_state_dim_layout,
    resolve_robot_type_mask_key,
    validate_full_action_supervision_config,
)


class EmbodimentId(IntEnum):
    """Enumeration for robot embodiment IDs."""

    AGILEX = 0
    AGIBOT_G1 = 1
    AGIBOT_DEX = 2
    UMI_OMIN = 3
    EGO_DEX = 4
    EGODEX_EEF_HANDBASE = 5
    ROBOCOIN_AGILEX_COBOT_MAGIC = 6
    H01_ROBOT = 7


class RobotType(StrEnum):
    """Enumeration for robot types."""

    AGILEX_COBOT_MAGIC = "aloha"
    AGIBOT_G1 = "agibot_g1"
    AGIBOT_WORLD = "agibot_world"
    UMI_OMIN = "UMI_omin"
    UMI_GIGA = "umi-giga"
    EGODEX_EEF_HANDBASE = "egodex_eef_handbase"
    EGOVERSE_EEF_HANDBASE = "egoverse_eef_handbase"
    WIYH_EEF_HANDBASE = "wiyh_eef_handbase"
    MARKER_U01_EEF_HANDBASE = "marker_u01_eef_handbase"
    ALOHA = "aloha"
    MOBILE_ALOHA = "mobile_aloha"
    ARX5 = "arx5"
    UR5 = "ur5"
    FRANKA = "franka"
    ROBOMIND_UR5 = "robomind_h5_ur_1rgb"
    ROBOMIND_FRANKA = "robomind_h5_franka_3rgb"
    ROBOMIND_FRANKA_DUAL = "robomind_h5_franka_fr3_dual"
    ROBOMIND_AGILEX = "robomind_h5_agilex_3rgb"
    AGIBOT_G1_DEXHAND = "agibot_g1_dexhand"

    # Galaxea Open World Dataset
    GALAXEA_R1LITE = "r1lite"

    # OXE (capital-F variant alongside existing 'franka')
    OXE_FRANKA = "Franka"
    OXE_FRANKA_SINGLE_ARM = "franka-single-arm"
    OXE_WIDOWX_SINGLE_ARM = "widowx-single-arm"
    OXE_FRANKA_SINGLE_ARM_FURTUNE_BENCH = "franka-single-arm-furtune_bench"

    # AgiBot World 2026
    AGIBOT_G2A = "g2a"

    # RoboCOIN
    ROBOCOIN_AIRBOT_MMK2_DISCOVER = "discover_robotics_aitbot_mmk2"
    ROBOCOIN_AIRBOT_MMK2 = "Airbot_MMK2"
    ROBOCOIN_AGILEX_COBOT_MAGIC = "Agilex_Cobot_Magic"
    ROBOCOIN_AGILEX_COBOT_MAGIC_LOWER = "agilex_cobot_magic"
    ROBOCOIN_AGILEX_DECOUPLED = "agilex_cobot_decoupled_magic"
    ROBOCOIN_AGIBOT = "robocoin_agibot"
    ROBOCOIN_RUANTONG_A2D = "ruantong_a2d"
    ROBOCOIN_UNITREE_G1_DEX3 = "Unitree_G1_Dex3_phecda"
    ROBOCOIN_GALAXEA_R1_LITE_TC = "Galaxea_R1_Lite"
    ROBOCOIN_GALAXEA_R1_LITE = "galaxea_r1_lite"
    ROBOCOIN_YINHE = "yinhe"
    ROBOCOIN_REALMAN_RMC_AIDAL = "realman_rmc_aidal"
    ROBOCOIN_REALMAN_RMC_AIDA_L = "Realman_RMC-AIDA-L"
    ROBOCOIN_ALPHA_BOT_2 = "alpha_bot_2"
    ROBOCOIN_LEJU_ROBOT = "leju_robot"

    # RoboMind v2 (placeholder action.names; classified by action dim)
    ROBOMIND2_AGILEX = "agilex"
    ROBOMIND2_AGILEX_MOBILE = "agilex_mobile"
    ROBOMIND2_ARK = "ark"
    ROBOMIND2_ARK_MOBILE = "ark_mobile"
    ROBOMIND2_TIENKUNG = "tienkung"
    ROBOMIND2_TIENKUNG_26D = "tienkung_26d"
    ROBOMIND2_UR = "ur"
    ROBOMIND2_UR_DEX = "ur_dex"

    # H01 / M01 internal robots
    H01_ROBOT = "h01_robot"
    H01_ROBOT_16D = "h01_robot_16d"
    H01_TAIHU_ROBOT = "h01_taihu_robot"
    H01_TAIHU_ROBOT_16D = "h01_taihu_robot_16d"
    M01_ROBOT = "m01_robot"



def _camera_pose7_xyzw_to_mat4(camera_pose: torch.Tensor) -> torch.Tensor:
    """Convert camera pose rows ``xyz + quat_xyzw`` to ``[..., 4, 4]``."""
    pose = camera_pose[..., :7]
    quat = torch.nn.functional.normalize(pose[..., 3:7], dim=-1)
    x, y, z, w = quat.unbind(dim=-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    row0 = torch.stack((1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)), dim=-1)
    row1 = torch.stack((2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)), dim=-1)
    row2 = torch.stack((2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)), dim=-1)
    rot = torch.stack((row0, row1, row2), dim=-2)

    out = torch.zeros(*pose.shape[:-1], 4, 4, dtype=pose.dtype, device=pose.device)
    out[..., :3, :3] = rot
    out[..., :3, 3] = pose[..., :3]
    out[..., 3, 3] = 1.0
    return out


def _camera_chunk_to_mat4(camera_chunk: torch.Tensor) -> torch.Tensor:
    """Normalize camera chunks for chunk-anchor reframe.

    EgoDex stores ``observation.state.camera`` as ``[..., 4, 4]``. WIYH stores
    ``observation.chest_pose`` as 14-D pose rows where the first 7 values are
    the chest camera pose in world coordinates.
    """
    camera_chunk = torch.as_tensor(camera_chunk)
    if camera_chunk.shape[-2:] == (4, 4):
        return camera_chunk
    if camera_chunk.shape[-1] in (7, 14):
        return _camera_pose7_xyzw_to_mat4(camera_chunk)
    raise ValueError(
        f"Unsupported camera chunk shape for reframe: {tuple(camera_chunk.shape)}"
    )


robot_type_mapping: dict[str, EmbodimentId] = {
    # Enum values (from lerobot_converter: 'aloha', 'agibot_g1', 'agibot_world')
    RobotType.AGILEX_COBOT_MAGIC: EmbodimentId.AGILEX,
    RobotType.ALOHA: EmbodimentId.AGILEX,
    RobotType.MOBILE_ALOHA: EmbodimentId.AGILEX,  # 16-dim mobile aloha (pretrained_agilex_data)
    RobotType.AGIBOT_G1: EmbodimentId.AGIBOT_G1,
    RobotType.AGIBOT_WORLD: EmbodimentId.AGIBOT_G1,
    RobotType.UMI_OMIN: EmbodimentId.UMI_OMIN,
    RobotType.UMI_GIGA: EmbodimentId.UMI_OMIN,
    RobotType.EGODEX_EEF_HANDBASE: EmbodimentId.EGODEX_EEF_HANDBASE,
    RobotType.EGOVERSE_EEF_HANDBASE: EmbodimentId.EGODEX_EEF_HANDBASE,
    RobotType.WIYH_EEF_HANDBASE: EmbodimentId.EGODEX_EEF_HANDBASE,
    RobotType.MARKER_U01_EEF_HANDBASE: EmbodimentId.EGODEX_EEF_HANDBASE,
    RobotType.ARX5: EmbodimentId.AGILEX,
    RobotType.UR5: EmbodimentId.AGILEX,
    RobotType.FRANKA: EmbodimentId.AGIBOT_G1,
    RobotType.ROBOMIND_UR5: EmbodimentId.AGILEX,
    RobotType.ROBOMIND_FRANKA: EmbodimentId.AGIBOT_G1,
    RobotType.ROBOMIND_FRANKA_DUAL: EmbodimentId.AGIBOT_G1,
    RobotType.ROBOMIND_AGILEX: EmbodimentId.AGILEX,
    RobotType.AGIBOT_G1_DEXHAND: EmbodimentId.AGIBOT_DEX,
    # Classified by action.names prefix:
    #   first 14 == 6 joints + gripper + 6 joints + gripper -> AGILEX
    #   first 16 == 7 joints + gripper + 7 joints + gripper -> AGIBOT_G1
    #   otherwise -> AGIBOT_DEX
    RobotType.GALAXEA_R1LITE: EmbodimentId.AGILEX,  # 21-dim, prefix 6+1+6+1
    RobotType.OXE_FRANKA: EmbodimentId.AGIBOT_G1,  # 8-dim single-arm
    RobotType.OXE_FRANKA_SINGLE_ARM: EmbodimentId.AGIBOT_G1,  # 8-dim single-arm
    RobotType.OXE_WIDOWX_SINGLE_ARM: EmbodimentId.AGILEX,  # 7-dim single-arm
    RobotType.OXE_FRANKA_SINGLE_ARM_FURTUNE_BENCH: EmbodimentId.AGIBOT_G1,  # 8-dim single-arm
    RobotType.AGIBOT_G2A: EmbodimentId.AGIBOT_G1,  # 26-dim, no names
    RobotType.ROBOCOIN_AIRBOT_MMK2_DISCOVER: EmbodimentId.AGIBOT_DEX,  # 36-dim w/ dex
    RobotType.ROBOCOIN_AIRBOT_MMK2: EmbodimentId.AGIBOT_DEX,  # 36-dim w/ dex
    RobotType.ROBOCOIN_AGILEX_COBOT_MAGIC: EmbodimentId.AGILEX,  # 14-dim, prefix 6+1+6+1
    RobotType.ROBOCOIN_AGILEX_COBOT_MAGIC_LOWER: EmbodimentId.ROBOCOIN_AGILEX_COBOT_MAGIC,  # 16-dim internal AgileX variant
    RobotType.ROBOCOIN_AGILEX_DECOUPLED: EmbodimentId.AGILEX,  # 14-dim, prefix 6+1+6+1
    RobotType.ROBOCOIN_AGIBOT: EmbodimentId.AGIBOT_G1,  # 20-dim, prefix 7+1+7+1
    RobotType.ROBOCOIN_RUANTONG_A2D: EmbodimentId.AGIBOT_G1,  # 16/20-dim, prefix 7+1+7+1
    RobotType.ROBOCOIN_UNITREE_G1_DEX3: EmbodimentId.AGIBOT_DEX,  # 28-dim w/ dex
    RobotType.ROBOCOIN_GALAXEA_R1_LITE_TC: EmbodimentId.AGILEX,  # 14-dim, prefix 6+1+6+1
    RobotType.ROBOCOIN_GALAXEA_R1_LITE: EmbodimentId.AGILEX,  # 14-dim, prefix 6+1+6+1
    RobotType.ROBOCOIN_YINHE: EmbodimentId.AGIBOT_G1,  # 21-dim, prefix 7+1+7+1
    RobotType.ROBOCOIN_REALMAN_RMC_AIDAL: EmbodimentId.AGIBOT_G1,  # 16-dim, prefix 7+1+7+1
    RobotType.ROBOCOIN_REALMAN_RMC_AIDA_L: EmbodimentId.AGIBOT_G1,  # 16-dim, prefix 7+1+7+1
    RobotType.ROBOCOIN_ALPHA_BOT_2: EmbodimentId.AGIBOT_G1,  # 16-dim, prefix 7+1+7+1
    RobotType.ROBOCOIN_LEJU_ROBOT: EmbodimentId.AGIBOT_DEX,  # 40-dim
    RobotType.ROBOMIND2_AGILEX: EmbodimentId.AGILEX,  # 14-dim
    RobotType.ROBOMIND2_AGILEX_MOBILE: EmbodimentId.AGILEX,  # 14-dim
    RobotType.ROBOMIND2_ARK: EmbodimentId.AGILEX,  # 14-dim
    RobotType.ROBOMIND2_ARK_MOBILE: EmbodimentId.AGILEX,  # 14-dim
    RobotType.ROBOMIND2_TIENKUNG: EmbodimentId.AGIBOT_G1,  # 16-dim
    RobotType.ROBOMIND2_TIENKUNG_26D: EmbodimentId.AGIBOT_DEX,  # 26-dim
    RobotType.ROBOMIND2_UR: EmbodimentId.AGILEX,  # 14-dim
    RobotType.ROBOMIND2_UR_DEX: EmbodimentId.AGIBOT_DEX,  # 36-dim w/ dex
    RobotType.H01_ROBOT: EmbodimentId.H01_ROBOT,  # 16/22-dim, prefix 7+1+7+1
    RobotType.H01_ROBOT_16D: EmbodimentId.AGIBOT_G1,  # 16-dim, prefix 7+1+7+1
    RobotType.H01_TAIHU_ROBOT: EmbodimentId.AGIBOT_G1,  # 16/22-dim, prefix 7+1+7+1
    RobotType.H01_TAIHU_ROBOT_16D: EmbodimentId.AGIBOT_G1,  # 16-dim, prefix 7+1+7+1
    RobotType.M01_ROBOT: EmbodimentId.AGIBOT_G1,  # 16-dim, prefix 7+1+7+1
}


def _coerce_embodiment_id(value: int | str | EmbodimentId) -> int:
    if isinstance(value, EmbodimentId):
        return int(value)
    if isinstance(value, str) and value in EmbodimentId.__members__:
        return int(EmbodimentId[value])
    return int(value)


def _build_robot_type_mapping(
    overrides: dict[str, int | str | EmbodimentId] | None = None,
) -> dict[str, int]:
    resolved = {
        str(robot_type): int(embodiment_id)
        for robot_type, embodiment_id in robot_type_mapping.items()
    }
    if overrides is None:
        return resolved
    for robot_type, embodiment_id in overrides.items():
        resolved[str(robot_type)] = _coerce_embodiment_id(embodiment_id)
    return resolved


def _strip_robot_type_dim_suffix(robot_type_key: str) -> str:
    prefix, sep, suffix = robot_type_key.rpartition("_")
    if sep and suffix.endswith("d") and suffix[:-1].isdigit():
        return prefix
    return robot_type_key


@TRANSFORMS.register
class GigaBrain07Transform:
    """A class to transform raw data into a format suitable for GigaBrain07
    model training."""

    @staticmethod
    def _is_vqa_sample(sample: dict) -> bool:
        if bool(sample.get("vqa_language_only", False)):
            return True
        dataset_type = sample.get("dataset_type", None)
        if isinstance(dataset_type, str) and dataset_type.lower() == "vqa":
            return True
        return "question" in sample and ("answer" in sample or "answers" in sample)

    @staticmethod
    def _get_skip_loss_mask(data_dict: dict[str, Any]) -> torch.Tensor:
        return torch.tensor(
            bool(data_dict.get("_skip_loss_for_invalid_images", False)),
            dtype=torch.bool,
        )

    def _get_vqa_tokenizer(self):
        prompt_tokenizer_transform = self.prompt_tokenizer_transform
        tokenizer = getattr(prompt_tokenizer_transform, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        return prompt_tokenizer_transform.paligemma_tokenizer

    def _build_vqa_output(self, data_dict: dict) -> dict:
        tokenizer = self._get_vqa_tokenizer()
        max_length = int(self.prompt_tokenizer_transform.max_length)

        vqa_image_transform = getattr(self, "vqa_image_transform", self.image_transform)
        images, image_masks, image_transform_params = vqa_image_transform(data_dict)

        question = str(data_dict.get("question", "")).strip()
        answer = str(data_dict.get("answer", "")).strip()
        prompt_text = str(
            data_dict.get("vqa_prompt", f"Question: {question}\nAnswer:")
        ).strip()
        full_text = str(data_dict.get("vqa_text", f"{prompt_text} {answer}")).strip()

        prompt_output = tokenizer(
            prompt_text, add_special_tokens=True, return_tensors="pt", truncation=False
        )
        prompt_length = int(prompt_output["input_ids"].shape[-1])

        full_output = tokenizer(
            full_text, add_special_tokens=True, return_tensors="pt", truncation=False
        )
        token_ids = full_output["input_ids"].squeeze(0)
        token_masks = full_output["attention_mask"].squeeze(0)

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if len(eos_token_id) > 0 else None
        has_eos = False
        if eos_token_id is not None and token_ids.numel() > 0:
            eos_token_id = int(eos_token_id)
            has_eos = int(token_ids[-1].item()) == eos_token_id
            if not has_eos:
                token_ids = torch.cat([token_ids, token_ids.new_tensor([eos_token_id])])
                token_masks = torch.cat([token_masks, token_masks.new_tensor([1])])
                has_eos = True

        text_budget = max_length
        if token_ids.shape[0] > text_budget:
            if has_eos and text_budget > 0:
                token_ids = torch.cat([token_ids[: text_budget - 1], token_ids[-1:]])
                token_masks = torch.cat(
                    [token_masks[: text_budget - 1], token_masks[-1:]]
                )
            else:
                token_ids = token_ids[:text_budget]
                token_masks = token_masks[:text_budget]

        text_loss_masks = (
            (token_masks != 0).cumsum(dim=0) > min(prompt_length, text_budget)
        ) & token_masks.bool()
        text_att_masks = text_loss_masks.clone()

        padded = tokenizer.pad(
            {"input_ids": token_ids.tolist(), "attention_mask": token_masks.tolist()},
            padding="max_length",
            padding_side="right",
            max_length=max_length,
            return_tensors="pt",
        )
        lang_tokens = padded["input_ids"].squeeze(0).to(dtype=torch.int32)
        lang_masks = padded["attention_mask"].squeeze(0).to(dtype=torch.bool)
        pad_len = max_length - int(text_loss_masks.shape[0])
        if pad_len > 0:
            lang_loss_masks = torch.nn.functional.pad(
                text_loss_masks, (0, pad_len), value=False
            )
            lang_att_masks = torch.nn.functional.pad(
                text_att_masks, (0, pad_len), value=False
            )
        else:
            lang_loss_masks = text_loss_masks[:max_length]
            lang_att_masks = text_att_masks[:max_length]

        output_dict: dict = {}
        output_dict["lang_tokens"] = lang_tokens
        output_dict["lang_masks"] = lang_masks
        output_dict["lang_att_masks"] = lang_att_masks
        output_dict["lang_loss_masks"] = lang_loss_masks
        output_dict["fast_action_indicator"] = torch.zeros(
            (max_length,), dtype=torch.bool
        )
        output_dict["subtask_indicator"] = torch.zeros((max_length,), dtype=torch.bool)

        state = torch.zeros(32, dtype=torch.float32)
        action = torch.zeros(50, 32, dtype=torch.float32)

        output_dict["observation.state"] = state
        output_dict["action"] = action
        if getattr(self, "state_input_mode", "prompt") == "proprio_anchor":
            output_dict["observation.proprioception"] = torch.zeros(
                1, self.propri_dim, dtype=torch.float32
            )
            output_dict["observation.agent_pos_mask"] = torch.zeros(
                1, self.propri_dim, dtype=torch.float32
            )
            output_dict["observation.proprioception_present"] = torch.tensor(
                False, dtype=torch.bool
            )

        output_dict["images"] = images
        output_dict["image_masks"] = image_masks
        if bool(data_dict.get("vqa_language_only", False)):
            output_dict["skip_loss_mask"] = torch.tensor(False, dtype=torch.bool)
        else:
            output_dict["skip_loss_mask"] = self._get_skip_loss_mask(data_dict)

        action_len = int(action.shape[0])
        output_dict["action_loss_mask"] = torch.zeros((action_len,), dtype=torch.bool)
        if getattr(self, "mask_unsupervised_action_dims_for_noise", False):
            output_dict["action_dim_loss_mask"] = torch.zeros_like(
                action, dtype=torch.bool
            )
        if getattr(self, "_invalid_action_chunk_filter_enabled", False):
            output_dict["invalid_action_chunk"] = torch.tensor(
                False, dtype=torch.bool
            )

        if self.trajectory_transform is not None:
            traj, traj_is_pad = self.trajectory_transform(
                data_dict,
                chunk_size=action_len,
                image_transform_params=image_transform_params,
            )
            output_dict["traj"] = traj
            output_dict["traj_loss_mask"] = torch.zeros_like(
                traj_is_pad, dtype=torch.bool
            )

        output_dict["embodiment_id"] = torch.tensor(
            int(data_dict.get("embodiment_id", 0)), dtype=torch.long
        )
        output_dict["action_fps"] = torch.tensor(30.0, dtype=torch.float32)
        meta = data_dict.get("meta")
        output_dict["debug_repo_id"] = getattr(meta, "repo_id", None)
        episode_index = data_dict.get("episode_index")
        frame_index = data_dict.get("frame_index")
        output_dict["debug_episode_index"] = (
            int(episode_index) if episode_index is not None else -1
        )
        output_dict["debug_frame_index"] = (
            int(frame_index) if frame_index is not None else -1
        )
        return output_dict

    @staticmethod
    def _resolve_norm_stats_paths(
        norm_stats_path_cfg: dict[int | str, str] | list[dict[str, Any]],
    ) -> tuple[
        dict[int | str, str], list[tuple[list[Any] | tuple[Any, ...], int | str]] | None
    ]:
        if isinstance(norm_stats_path_cfg, dict):
            resolved_norm_stats_paths: dict[int | str, str] = dict()
            for key, norm_stats_path in norm_stats_path_cfg.items():
                normalized_key = Normalize._normalize_stats_key(key)
                normalized_path = os.path.normpath(os.fspath(norm_stats_path))
                if (
                    normalized_key in resolved_norm_stats_paths
                    and resolved_norm_stats_paths[normalized_key] != normalized_path
                ):
                    raise ValueError(
                        f"Duplicate normalization key {normalized_key!r} maps to multiple stats files."
                    )
                resolved_norm_stats_paths[normalized_key] = normalized_path
            return resolved_norm_stats_paths, None

        if not isinstance(norm_stats_path_cfg, list):
            raise TypeError(
                'norm_cfg["norm_stats_path"] must be a dict or a list of grouped entries.'
            )

        resolved_norm_stats_paths: dict[int | str, str] = dict()
        selector_groups: list[tuple[list[Any] | tuple[Any, ...], int | str]] = []
        for group_idx, path_group in enumerate(norm_stats_path_cfg):
            if not isinstance(path_group, dict):
                raise TypeError("Each grouped norm_stats_path entry must be a dict.")

            selector_values = path_group.get(
                "selector_values", path_group.get("data_paths", path_group.get("keys"))
            )
            if selector_values is None:
                raise KeyError(
                    'Each grouped norm_stats_path entry must provide one of "selector_values", "data_paths", or "keys".'
                )

            norm_stats_path = path_group.get("norm_stats_path", path_group.get("path"))
            if norm_stats_path is None:
                raise KeyError(
                    'Each grouped norm_stats_path entry must provide "norm_stats_path" or "path".'
                )

            if isinstance(selector_values, (str, os.PathLike)):
                selector_values = [selector_values]

            group_key = path_group.get("group_key", path_group.get("name"))
            if group_key is None:
                if len(selector_values) == 1:
                    group_key = selector_values[0]
                else:
                    group_key = f"__norm_group_{group_idx}"

            normalized_group_key = Normalize._normalize_stats_key(group_key)
            normalized_path = os.path.normpath(os.fspath(norm_stats_path))
            if (
                normalized_group_key in resolved_norm_stats_paths
                and resolved_norm_stats_paths[normalized_group_key] != normalized_path
            ):
                raise ValueError(
                    f"Duplicate normalization group {normalized_group_key!r} maps to multiple stats files."
                )
            resolved_norm_stats_paths[normalized_group_key] = normalized_path
            selector_groups.append((selector_values, normalized_group_key))

        return resolved_norm_stats_paths, selector_groups

    def _map_norm_selector_value(self, selector_value: int | str) -> int | str:
        normalized_selector_value = Normalize._normalize_stats_key(selector_value)

        if self._norm_selector_groups is None:
            return normalized_selector_value

        if normalized_selector_value in self._norm_selector_group_cache:
            return self._norm_selector_group_cache[normalized_selector_value]

        for selector_values, group_key in self._norm_selector_groups:
            for candidate_selector_value in selector_values:
                if (
                    Normalize._normalize_stats_key(candidate_selector_value)
                    == normalized_selector_value
                ):
                    self._norm_selector_group_cache[normalized_selector_value] = (
                        group_key
                    )
                    return group_key

        raise KeyError(
            f"Normalization group not found for selector value {normalized_selector_value!r}"
        )

    def _prepare_delta_mask(self, mask_cfg: dict, dtype: torch.dtype) -> dict:
        """Build a {selector_key: mask_tensor} dict, applying quat->6D mask transform.

        For selector='embodiment_id', the key is int(embodiment_id) and the
        quat->6D transform is left to ``DeltaActions.__init__`` (which only
        knows how to dispatch on embodiment_id) to avoid applying it twice.
        For selector='robot_type', the key is the robot_type string; the
        underlying embodiment_id is resolved via robot_type_mapping here so
        we can call ``transform_mask`` upfront -- ``DeltaActions`` skips the
        transform in robot_type mode.
        """
        prepared: dict = dict()
        for raw_key, mask_list in mask_cfg.items():
            mask_tensor = torch.tensor(mask_list, dtype=dtype)
            if self.delta_mask_selector == "embodiment_id":
                embodiment_id = int(raw_key)
                sel_key: int | str = embodiment_id
                prepared[sel_key] = mask_tensor
                continue

            robot_type_key = str(raw_key)
            base_robot_type_key = _strip_robot_type_dim_suffix(robot_type_key)
            try:
                robot_type_enum = RobotType(base_robot_type_key)
            except ValueError as e:
                raise ValueError(
                    f"delta_action_cfg.mask key {raw_key!r} is not a valid RobotType "
                    f"when selector='robot_type'. Valid values: {[rt.value for rt in RobotType]}"
                ) from e
            if robot_type_enum not in robot_type_mapping:
                raise KeyError(
                    f"RobotType {robot_type_enum.value!r} has no entry in robot_type_mapping."
                )
            embodiment_id = self._resolve_robot_type_embodiment_id(robot_type_enum)
            sel_key = robot_type_key
            if Embodiment3QuaternionTo6D.supports_embodiment(embodiment_id):
                mask_tensor = Embodiment3QuaternionTo6D.transform_mask(
                    mask_tensor, embodiment_id
                )
            prepared[sel_key] = mask_tensor
        return prepared

    def _resolve_robot_type_embodiment_id(self, robot_type: Any) -> int:
        robot_type_key = str(robot_type)
        if robot_type_key in self.robot_type_mapping:
            return int(self.robot_type_mapping[robot_type_key])

        try:
            robot_type_enum = RobotType(robot_type_key)
        except ValueError as e:
            raise KeyError(
                f"RobotType {robot_type_key!r} has no entry in robot_type_mapping. "
                f"Valid values: {sorted(self.robot_type_mapping)}"
            ) from e

        enum_key = str(robot_type_enum)
        if enum_key in self.robot_type_mapping:
            return int(self.robot_type_mapping[enum_key])
        raise KeyError(f"RobotType {enum_key!r} has no entry in robot_type_mapping.")

    def __init__(
        self,
        delta_action_cfg: dict[str, Any] | None = None,
        norm_cfg: dict[str, Any] | None = None,
        traj_cfg: dict[str, Any] | None = None,
        image_cfg: dict[str, Any] | None = None,
        prompt_cfg: dict[str, Any] | None = None,
        state_input_mode: str = "prompt",
        observation_memory_size: int = 1,
        agent_pos_config: dict[str, int] | None = None,
        robot_type_embodiment_id_overrides: dict[str, int | str | EmbodimentId]
        | None = None,
        use_quaternion_to_6d: bool = False,
        eef_dual_hand_prefix_to_6d: bool = False,
        eef_dual_hand_prefix_to_6d_robot_types: list[str]
        | tuple[str, ...]
        | None = None,
        chunk_anchor_camera_reframe_cfg: dict[str, Any] | None = None,
        invalid_action_chunk_filter_cfg: dict[str, Any] | None = None,
        is_train: bool = True,
    ):
        """Initializes the transform pipeline.

        Args:
            delta_action_cfg (dict[str, Any] | None, optional): Configuration for delta actions.
                ``mask_unsupervised_action_dims_for_noise`` enables valid-dimension
                noise and loss masking and defaults to False.
                Defaults to None.
            norm_cfg (dict[str, Any] | None, optional): Configuration for normalization. Defaults to None.
            traj_cfg (dict[str, Any] | None, optional): Configuration for trajectory transform. Defaults to None.
            image_cfg (dict[str, Any] | None, optional): Configuration for image transform. Defaults to None.
            prompt_cfg (dict[str, Any] | None, optional): Configuration for prompt tokenizer. Defaults to None.
            state_input_mode: "prompt" keeps the legacy text-state prompt path;
                "proprio_memory" emits continuous proprio state memory tensors.
            observation_memory_size: Number of state frames expected in proprio_memory mode.
            use_quaternion_to_6d (bool, optional): Whether to convert
                embodiment 3/4/5 pose tensors from quaternion format to 6D rotation format
                before delta-action processing and normalization. Defaults to False.
            eef_dual_hand_prefix_to_6d (bool, optional): When True with ``use_quaternion_to_6d``,
                convert only the leading dual-hand TCP quaternion block (16 floats) to 6D (20 floats)
                and keep any trailing dims unchanged.
            eef_dual_hand_prefix_to_6d_robot_types (list[str] | tuple[str, ...] | None, optional):
                Limit the dual-hand prefix conversion to these robot types. ``None`` preserves the
                legacy behavior of applying it to every non-VQA sample.
            chunk_anchor_camera_reframe_cfg (dict[str, Any] | None, optional): When ``enabled`` is
                True, re-express the action chunk in the anchor camera frame before quat→6D.
                ``camera_key_by_robot_type`` can override ``camera_key`` for mixed datasets.
            invalid_action_chunk_filter_cfg (dict[str, Any] | None, optional): Dynamically mask
                action chunks with invalid camera motion or extreme normalized actions. The
                filter is disabled by default and can be scoped with ``robot_types``.
            is_train (bool, optional): Whether the transform is used for training. Defaults to True.
        """
        self.is_train = is_train
        self._chunk_anchor_camera_reframe_enabled = bool(
            chunk_anchor_camera_reframe_cfg is not None
            and chunk_anchor_camera_reframe_cfg.get("enabled", False)
        )
        self._chunk_anchor_camera_key = (
            str(
                chunk_anchor_camera_reframe_cfg.get(
                    "camera_key", "observation.state.camera"
                )
            )
            if self._chunk_anchor_camera_reframe_enabled
            else "observation.state.camera"
        )
        raw_camera_keys_by_robot_type = (
            chunk_anchor_camera_reframe_cfg.get("camera_key_by_robot_type", {})
            if chunk_anchor_camera_reframe_cfg is not None
            else {}
        )
        if not isinstance(raw_camera_keys_by_robot_type, dict):
            raise TypeError(
                "chunk_anchor_camera_reframe_cfg.camera_key_by_robot_type should be a dict"
            )
        self._chunk_anchor_camera_key_by_robot_type = {
            str(robot_type): str(camera_key)
            for robot_type, camera_key in raw_camera_keys_by_robot_type.items()
        }
        invalid_chunk_cfg = dict(invalid_action_chunk_filter_cfg or {})
        self._invalid_action_chunk_filter_enabled = bool(
            invalid_chunk_cfg.get("enabled", False)
        )
        raw_invalid_chunk_robot_types = invalid_chunk_cfg.get("robot_types")
        if raw_invalid_chunk_robot_types is not None and not isinstance(
            raw_invalid_chunk_robot_types, (list, tuple, set)
        ):
            raise TypeError(
                "invalid_action_chunk_filter_cfg.robot_types should be a list, tuple, or set"
            )
        self._invalid_action_chunk_robot_types = (
            None
            if raw_invalid_chunk_robot_types is None
            else {str(robot_type) for robot_type in raw_invalid_chunk_robot_types}
        )
        self._max_camera_translation_step_m = float(
            invalid_chunk_cfg.get("max_camera_translation_step_m", 1.0)
        )
        self._max_camera_rotation_step_degrees = float(
            invalid_chunk_cfg.get("max_camera_rotation_step_degrees", 120.0)
        )
        self._min_camera_quaternion_norm = float(
            invalid_chunk_cfg.get("min_camera_quaternion_norm", 1e-6)
        )
        self._max_camera_rotation_matrix_error = float(
            invalid_chunk_cfg.get("max_camera_rotation_matrix_error", 0.1)
        )
        self._max_normalized_action_abs = float(
            invalid_chunk_cfg.get("max_normalized_action_abs", 10.0)
        )
        for name, value in (
            ("max_camera_translation_step_m", self._max_camera_translation_step_m),
            (
                "max_camera_rotation_step_degrees",
                self._max_camera_rotation_step_degrees,
            ),
            ("min_camera_quaternion_norm", self._min_camera_quaternion_norm),
            (
                "max_camera_rotation_matrix_error",
                self._max_camera_rotation_matrix_error,
            ),
            ("max_normalized_action_abs", self._max_normalized_action_abs),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"invalid_action_chunk_filter_cfg.{name} must be finite and positive, "
                    f"got {value}"
                )
        self._eef_dual_hand_prefix_to_6d = bool(eef_dual_hand_prefix_to_6d)
        self._eef_dual_hand_prefix_to_6d_robot_types = (
            None
            if eef_dual_hand_prefix_to_6d_robot_types is None
            else {
                str(robot_type) for robot_type in eef_dual_hand_prefix_to_6d_robot_types
            }
        )
        self.use_quaternion_to_6d = use_quaternion_to_6d
        if state_input_mode not in ("prompt", "proprio_memory", "proprio_anchor"):
            raise ValueError(
                "state_input_mode must be 'prompt', 'proprio_memory', or "
                f"'proprio_anchor', got {state_input_mode!r}"
            )
        self.state_input_mode = state_input_mode
        self.agent_pos_config = dict(agent_pos_config or {})
        self.propri_dim = sum(int(dim) for dim in self.agent_pos_config.values())
        if self.state_input_mode == "proprio_anchor" and self.propri_dim <= 0:
            raise ValueError(
                "state_input_mode='proprio_anchor' requires a non-empty agent_pos_config"
            )
        self.observation_memory_size = int(observation_memory_size)
        if self.observation_memory_size < 1:
            raise ValueError(
                f"observation_memory_size must be positive, got {self.observation_memory_size}"
            )
        self.robot_type_mapping = _build_robot_type_mapping(
            robot_type_embodiment_id_overrides
        )
        validate_full_action_supervision_config(delta_action_cfg)
        self.use_delta_joint_actions = (
            delta_action_cfg is not None
            and delta_action_cfg.get("use_delta_joint_actions", False)
        )
        self.mask_unsupervised_action_dims_for_noise = bool(
            delta_action_cfg is not None
            and delta_action_cfg.get(
                "mask_unsupervised_action_dims_for_noise", False
            )
        )
        self.zero_non_action_dims_when_action_mask_short = bool(
            True
            if delta_action_cfg is None
            else delta_action_cfg.get(
                "zero_non_action_dims_when_action_mask_short", True
            )
        )
        self.delta_mask_selector = (
            delta_action_cfg.get("selector", "embodiment_id")
            if delta_action_cfg is not None
            else "embodiment_id"
        )
        if self.delta_mask_selector not in ("embodiment_id", "robot_type"):
            raise ValueError(
                f"Unsupported delta_action_cfg.selector: {self.delta_mask_selector!r}"
            )
        self.delta_action_masks: dict = dict()
        self.end_effector_types: dict[int | str, str] = dict()

        prepared_delta_mask = None
        prepared_delta_action_transform = None
        if delta_action_cfg is not None and "mask" in delta_action_cfg:
            prepared_delta_mask = self._prepare_delta_mask(
                delta_action_cfg["mask"], dtype=torch.bool
            )
            prepared_delta_action_transform = DeltaActions(
                prepared_delta_mask, selector=self.delta_mask_selector
            )
            self.delta_action_masks = prepared_delta_action_transform.mask
            raw_delta_mask = {
                (
                    int(raw_key)
                    if self.delta_mask_selector == "embodiment_id"
                    else str(raw_key)
                ): torch.tensor(mask_list, dtype=torch.bool)
                for raw_key, mask_list in delta_action_cfg["mask"].items()
            }
            for sel_key, mask_tensor in raw_delta_mask.items():
                end_effector_type = infer_end_effector_type_from_delta_mask(mask_tensor)
                if end_effector_type is not None:
                    self.end_effector_types[sel_key] = end_effector_type

        if self.use_delta_joint_actions:
            assert prepared_delta_mask is not None, (
                "delta_action_cfg.mask is required when use_delta_joint_actions is True"
            )
            assert prepared_delta_action_transform is not None
            self.delta_action_transform = prepared_delta_action_transform
        if self.mask_unsupervised_action_dims_for_noise:
            if not prepared_delta_mask:
                raise ValueError(
                    "delta_action_cfg.mask must be non-empty when "
                    "mask_unsupervised_action_dims_for_noise is True"
                )

        self.embodiment3_quaternion_to_6d_transform = Embodiment3QuaternionTo6D()
        self.pad_transform = PadStatesAndActions(action_dim=32)

        assert norm_cfg is not None, "norm_cfg is required"
        self.norm_selector = norm_cfg.get("selector", "embodiment_id")
        resolved_norm_stats_paths, self._norm_selector_groups = (
            self._resolve_norm_stats_paths(norm_cfg["norm_stats_path"])
        )
        self._norm_selector_group_cache: dict[int | str, int | str] = dict()

        loaded_norm_stats_data: dict[str, dict[str, Any]] = dict()
        state_norm_stats_data_dict: dict[int | str, Any] = dict()
        action_norm_stats_data_dict: dict[int | str, Any] = dict()
        for key, norm_stats_path in resolved_norm_stats_paths.items():
            if norm_stats_path not in loaded_norm_stats_data:
                with open(norm_stats_path, "r") as f:
                    loaded_norm_stats_data[norm_stats_path] = json.load(f)["norm_stats"]

            norm_stats_data = loaded_norm_stats_data[norm_stats_path]
            state_norm_stats_data_dict[key] = norm_stats_data["observation.state"]
            action_norm_stats_data_dict[key] = norm_stats_data["action"]

        self.state_normalize_transform = Normalize(
            state_norm_stats_data_dict,
            use_quantiles=norm_cfg["use_quantiles"],
            enable_clamp=norm_cfg.get("enable_clamp", False),
        )

        self.action_normalize_transform = Normalize(
            action_norm_stats_data_dict,
            use_quantiles=norm_cfg["use_quantiles"],
            # Actions are supervision targets; clamping here would alter the GT.
            enable_clamp=False,
        )

        assert image_cfg is not None, "image_cfg is required"
        self.image_transform = ImageTransform(**image_cfg, is_train=is_train)
        vqa_image_cfg = dict(image_cfg)
        vqa_image_cfg["enable_image_aug"] = False
        self.vqa_image_transform = ImageTransform(**vqa_image_cfg, is_train=False)

        assert prompt_cfg is not None, "prompt_cfg is required"
        if self.state_input_mode in ("proprio_memory", "proprio_anchor") and bool(
            prompt_cfg.get("discrete_state_input", True)
        ):
            raise ValueError(
                f"state_input_mode={self.state_input_mode!r} is mutually exclusive with "
                "prompt_cfg.discrete_state_input=True"
            )
        prompt_cfg = dict(prompt_cfg)
        configured_prompt_mode = prompt_cfg.get("state_input_mode", self.state_input_mode)
        if configured_prompt_mode != self.state_input_mode:
            raise ValueError(
                f"prompt_cfg.state_input_mode={configured_prompt_mode!r} does not match "
                f"transform state_input_mode={self.state_input_mode!r}"
            )
        prompt_cfg["state_input_mode"] = self.state_input_mode
        self.prompt_tokenizer_transform = PromptTokenizerTransform(
            **prompt_cfg, is_train=is_train
        )

        self.trajectory_transform = None
        if traj_cfg is not None:
            self.trajectory_transform = TrajectoryTransform(**traj_cfg)

    def _build_state_memory_output(
        self,
        data_dict: dict[str, Any],
        current_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.state_input_mode != "proprio_memory":
            raise RuntimeError(
                "_build_state_memory_output is only valid in proprio_memory mode"
            )

        state_memory = data_dict.get("observation.state_memory")
        state_memory_is_pad = data_dict.get("observation.state_memory_is_pad")

        if state_memory is None:
            raw_state = data_dict.get("observation.state")
            raw_state_is_pad = data_dict.get("observation.state_is_pad")
            if (
                isinstance(raw_state, torch.Tensor)
                and raw_state.ndim >= 2
                and raw_state.shape[0] == self.observation_memory_size
            ):
                state_memory = raw_state
                state_memory_is_pad = raw_state_is_pad
            elif self.observation_memory_size == 1:
                state_memory = current_state.unsqueeze(0)
                state_memory_is_pad = torch.zeros(
                    1, dtype=torch.bool, device=current_state.device
                )
            else:
                raise KeyError(
                    "state_input_mode='proprio_memory' with observation_memory_size > 1 requires "
                    "historical state data. Add delta_info for 'observation.state' or repack it to "
                    "'observation.state_memory'."
                )

        state_memory = torch.as_tensor(
            state_memory, dtype=current_state.dtype, device=current_state.device
        )
        if state_memory.ndim != 2:
            raise ValueError(
                f"observation.state_memory must have shape [K,D], got {tuple(state_memory.shape)}"
            )
        if state_memory.shape[0] != self.observation_memory_size:
            raise ValueError(
                f"observation.state_memory has K={state_memory.shape[0]}, "
                f"but observation_memory_size={self.observation_memory_size}"
            )

        if state_memory_is_pad is None:
            state_memory_mask = torch.ones(
                self.observation_memory_size,
                dtype=torch.bool,
                device=current_state.device,
            )
        else:
            state_memory_mask = ~torch.as_tensor(
                state_memory_is_pad, dtype=torch.bool, device=current_state.device
            )
            if state_memory_mask.ndim == 0:
                state_memory_mask = state_memory_mask.expand(
                    self.observation_memory_size
                )
            state_memory_mask = state_memory_mask.reshape(self.observation_memory_size)

        return state_memory, state_memory_mask

    def _align_proprio_anchor(
        self,
        state: torch.Tensor,
        agent_pos_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.ndim == 2 and state.shape[0] == 1:
            state = state.squeeze(0)
            agent_pos_mask = agent_pos_mask.squeeze(0)
        if state.ndim != 1:
            raise ValueError(
                f"proprio_anchor expects one current state vector [D], got {tuple(state.shape)}"
            )
        if state.shape[-1] > self.propri_dim:
            raise ValueError(
                f"state width {state.shape[-1]} exceeds agent_pos_config total {self.propri_dim}"
            )
        pad_width = self.propri_dim - state.shape[-1]
        if pad_width > 0:
            state = torch.nn.functional.pad(state, (0, pad_width), value=0.0)
            agent_pos_mask = torch.nn.functional.pad(
                agent_pos_mask, (0, pad_width), value=0.0
            )
        return state.unsqueeze(0).clone(), agent_pos_mask.unsqueeze(0).clone()

    @staticmethod
    def _build_temporal_action_loss_mask(action_is_pad: torch.Tensor) -> torch.Tensor:
        if action_is_pad.ndim == 1:
            return ~action_is_pad
        if action_is_pad.ndim == 2:
            return ~action_is_pad.all(dim=-1)
        raise ValueError(
            f"action_is_pad must have shape [T] or [T, D], got {tuple(action_is_pad.shape)}"
        )

    @staticmethod
    def _get_action_fps(data_dict: dict[str, Any]) -> float:
        fps = data_dict.get("fps", data_dict.get("action_fps"))
        if fps is None:
            meta = data_dict.get("meta")
            meta_info = getattr(meta, "info", {}) if meta is not None else {}
            if isinstance(meta_info, dict):
                fps = meta_info.get("fps")
                if fps is None:
                    features = meta_info.get("features")
                    if isinstance(features, dict):
                        for key in (
                            "action",
                            "observation.state",
                            "timestamp",
                            "frame_index",
                        ):
                            feature_info = features.get(key)
                            if isinstance(feature_info, dict):
                                fps = feature_info.get("fps")
                                if fps is not None:
                                    break
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            return 30.0
        if not math.isfinite(fps):
            return 30.0
        return fps if fps > 0 else 30.0

    def _resolve_action_mask_dim(
        self,
        delta_mask_key: int | str,
        raw_action_dim: int,
        raw_state_dim: int,
        *,
        is_robot_moving: bool = False,
        is_body_moving: bool = False,
    ) -> int | None:
        layout = self._resolve_action_state_layout(
            delta_mask_key=delta_mask_key,
            raw_action_dim=raw_action_dim,
            raw_state_dim=raw_state_dim,
            is_robot_moving=is_robot_moving,
            is_body_moving=is_body_moving,
        )
        if layout is None:
            return None
        supervised_dim = layout.action_supervised_dim
        if supervised_dim >= int(self.pad_transform.action_dim):
            return None
        return supervised_dim

    def _resolve_action_state_layout(
        self,
        delta_mask_key: int | str,
        raw_action_dim: int,
        raw_state_dim: int,
        *,
        is_robot_moving: bool = False,
        is_body_moving: bool = False,
    ) -> ActionStateDimLayout | None:
        configured_mask = self.delta_action_masks.get(delta_mask_key)
        if configured_mask is None:
            return None
        return resolve_action_state_dim_layout(
            configured_mask,
            raw_action_dim=raw_action_dim,
            raw_state_dim=raw_state_dim,
            max_action_dim=int(self.pad_transform.action_dim),
            is_robot_moving=is_robot_moving,
            is_body_moving=is_body_moving,
        )

    @staticmethod
    def _get_metadata_bool(
        data_dict: dict[str, Any], key: str, *, default: bool = False
    ) -> bool:
        value = data_dict.get(key)
        if value is None:
            meta = data_dict.get("meta")
            meta_info = getattr(meta, "info", {}) if meta is not None else {}
            if isinstance(meta_info, dict):
                value = meta_info.get(key)
        if value is None:
            return bool(default)

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"{key} must be a dataset/sample-level scalar, got "
                    f"tensor shape {tuple(value.shape)}"
                )
            value = value.item()
        elif hasattr(value, "item") and callable(value.item):
            value = value.item()

        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        raise ValueError(
            f"{key} must be a scalar bool or 0/1 value, got {value!r}"
        )

    def _get_is_robot_moving(self, data_dict: dict[str, Any]) -> bool:
        return self._get_metadata_bool(data_dict, "is_robot_moving", default=False)

    def _get_is_body_moving(self, data_dict: dict[str, Any]) -> bool:
        return self._get_metadata_bool(data_dict, "is_body_moving", default=False)

    def _zero_non_action_dims_outside_action_mask(
        self,
        data_dict: dict[str, Any],
        *,
        delta_mask_key: int | str,
        raw_action_dim: int,
        raw_state_dim: int,
        is_robot_moving: bool = False,
        is_body_moving: bool = False,
        action_mask_dim: int | None = None,
    ) -> None:
        if not self.zero_non_action_dims_when_action_mask_short:
            return
        action = data_dict.get("action")
        if not isinstance(action, torch.Tensor) or action.ndim == 0:
            return

        if action_mask_dim is None:
            action_mask_dim = self._resolve_action_mask_dim(
                delta_mask_key=delta_mask_key,
                raw_action_dim=raw_action_dim,
                raw_state_dim=raw_state_dim,
                is_robot_moving=is_robot_moving,
                is_body_moving=is_body_moving,
            )
        if action_mask_dim is None or action_mask_dim >= int(action.shape[-1]):
            return

        action = action.clone()
        action[..., action_mask_dim:] = 0
        data_dict["action"] = action

    @staticmethod
    def _mask_state_input_dims(
        data_dict: dict[str, Any],
        agent_pos_mask: torch.Tensor | None,
        *,
        state_input_dim: int,
    ) -> torch.Tensor | None:
        for key in ("observation.state", "observation.state_memory"):
            state = data_dict.get(key)
            if not isinstance(state, torch.Tensor) or state.ndim == 0:
                continue
            if state_input_dim >= int(state.shape[-1]):
                continue
            state = state.clone()
            state[..., state_input_dim:] = 0
            data_dict[key] = state

        if agent_pos_mask is None or state_input_dim >= int(agent_pos_mask.shape[-1]):
            return agent_pos_mask
        agent_pos_mask = agent_pos_mask.clone()
        agent_pos_mask[..., state_input_dim:] = 0
        return agent_pos_mask

    def _build_action_dim_loss_mask(
        self,
        action_is_pad: torch.Tensor,
        *,
        action_mask_dim: int | None,
        padded_action_dim: int,
    ) -> torch.Tensor:
        valid_action_dim = (
            int(padded_action_dim)
            if action_mask_dim is None
            else min(int(action_mask_dim), int(padded_action_dim))
        )
        dim_mask = torch.zeros(
            (int(action_is_pad.shape[0]), int(padded_action_dim)),
            dtype=torch.bool,
            device=action_is_pad.device,
        )
        if valid_action_dim <= 0:
            return dim_mask

        if action_is_pad.ndim == 1:
            dim_mask[:, :valid_action_dim] = (~action_is_pad)[:, None]
            return dim_mask
        if action_is_pad.ndim == 2:
            valid_cols = min(valid_action_dim, int(action_is_pad.shape[-1]))
            if valid_cols > 0:
                dim_mask[:, :valid_cols] = ~action_is_pad[:, :valid_cols]
            return dim_mask
        raise ValueError(
            f"action_is_pad must have shape [T] or [T, D], got {tuple(action_is_pad.shape)}"
        )

    def _get_norm_selector(
        self, data_dict: dict[str, Any], embodiment_id: int
    ) -> int | str:
        if self.norm_selector == "embodiment_id":
            return self._map_norm_selector_value(embodiment_id)

        meta = data_dict.get("meta")
        if meta is None:
            raise KeyError(
                f"meta is required when norm_cfg.selector={self.norm_selector!r}"
            )

        if self.norm_selector == "data_path":
            meta_root = getattr(meta, "root", None)
            if meta_root is None:
                raise KeyError(
                    'meta.root is required when norm_cfg.selector="data_path"'
                )
            return self._map_norm_selector_value(os.path.normpath(os.fspath(meta_root)))

        if self.norm_selector == "repo_id":
            repo_id = getattr(meta, "repo_id", None)
            if repo_id is None:
                raise KeyError(
                    'meta.repo_id is required when norm_cfg.selector="repo_id"'
                )
            return self._map_norm_selector_value(repo_id)

        raise ValueError(f"Unsupported norm_cfg.selector: {self.norm_selector}")

    def _get_delta_mask_selector(
        self, data_dict: dict[str, Any], embodiment_id: int
    ) -> int | str:
        if self.delta_mask_selector == "embodiment_id":
            return int(embodiment_id)
        robot_type = data_dict.get("robot_type")
        if robot_type is None:
            meta = data_dict.get("meta")
            if meta is None or "robot_type" not in getattr(meta, "info", {}):
                raise KeyError(
                    "robot_type is required when delta_action_cfg.selector='robot_type'"
                )
            robot_type = meta.info["robot_type"]
        action = data_dict.get("action")
        state = data_dict.get("observation.state")
        action_dim = action.shape[-1] if action is not None else None
        state_dim = state.shape[-1] if state is not None else None
        return resolve_robot_type_mask_key(
            self.delta_action_masks, robot_type, action_dim, state_dim
        )

    def _split_temporal_state_memory(self, data_dict: dict[str, Any]) -> None:
        if self.state_input_mode != "proprio_memory":
            return
        state = data_dict.get("observation.state")
        if not isinstance(state, torch.Tensor) or state.ndim < 2:
            return
        if state.shape[0] != self.observation_memory_size:
            return
        if "observation.state_memory" not in data_dict:
            data_dict["observation.state_memory"] = state
        if (
            "observation.state_memory_is_pad" not in data_dict
            and "observation.state_is_pad" in data_dict
        ):
            data_dict["observation.state_memory_is_pad"] = data_dict[
                "observation.state_is_pad"
            ]
        data_dict["observation.state"] = state[-1]

    def _resolve_chunk_anchor_camera_key(self, data_dict: dict[str, Any]) -> str:
        robot_type = data_dict.get("robot_type")
        return self._chunk_anchor_camera_key_by_robot_type.get(
            str(robot_type),
            self._chunk_anchor_camera_key,
        )

    def _invalid_action_chunk_filter_applies(
        self, data_dict: dict[str, Any]
    ) -> bool:
        if not self._invalid_action_chunk_filter_enabled:
            return False
        if self._invalid_action_chunk_robot_types is None:
            return True
        return (
            str(data_dict.get("robot_type"))
            in self._invalid_action_chunk_robot_types
        )

    def _valid_action_timesteps(
        self,
        action_is_pad: Any,
        *,
        time_steps: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if action_is_pad is None:
            return torch.ones(time_steps, dtype=torch.bool, device=device)
        try:
            valid = self._build_temporal_action_loss_mask(
                torch.as_tensor(action_is_pad, device=device, dtype=torch.bool)
            )
        except (TypeError, ValueError):
            return None
        if valid.ndim != 1 or int(valid.shape[0]) != int(time_steps):
            return None
        return valid

    def _first_invalid_camera_offset(
        self,
        camera_chunk: Any,
        action_is_pad: Any,
    ) -> int | None:
        camera = torch.as_tensor(camera_chunk)
        is_matrix = camera.ndim == 3 and camera.shape[-2:] == (4, 4)
        is_pose = camera.ndim == 2 and camera.shape[-1] in (7, 14)
        if not is_matrix and not is_pose:
            return 0

        time_steps = int(camera.shape[0])
        valid = self._valid_action_timesteps(
            action_is_pad,
            time_steps=time_steps,
            device=camera.device,
        )
        if valid is None:
            return 0
        if not valid.any():
            return None

        finite_rows = torch.isfinite(camera.reshape(time_steps, -1)).all(dim=-1)
        invalid_rows = valid & ~finite_rows
        if is_pose:
            quaternion_norm = torch.linalg.vector_norm(camera[:, 3:7].float(), dim=-1)
            invalid_rows |= valid & (
                quaternion_norm < self._min_camera_quaternion_norm
            )

        camera_to_world = _camera_chunk_to_mat4(camera).float()
        rotation = camera_to_world[:, :3, :3]
        translation = camera_to_world[:, :3, 3]

        rotation_check_rows = valid & finite_rows
        if rotation_check_rows.any():
            checked_rotation = rotation[rotation_check_rows]
            identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
            orthogonality_error = torch.linalg.matrix_norm(
                checked_rotation.transpose(-1, -2) @ checked_rotation - identity,
                ord="fro",
                dim=(-2, -1),
            )
            determinant_error = (torch.linalg.det(checked_rotation) - 1.0).abs()
            invalid_rotation = (
                orthogonality_error > self._max_camera_rotation_matrix_error
            ) | (determinant_error > self._max_camera_rotation_matrix_error)
            checked_indices = torch.nonzero(rotation_check_rows, as_tuple=False).flatten()
            invalid_rows[checked_indices[invalid_rotation]] = True

        first_invalid_offsets: list[int] = []
        if invalid_rows.any():
            first_invalid_offsets.append(
                int(torch.nonzero(invalid_rows, as_tuple=False)[0].item())
            )

        pair_valid = valid[:-1] & valid[1:] & ~invalid_rows[:-1] & ~invalid_rows[1:]
        if not pair_valid.any():
            return min(first_invalid_offsets) if first_invalid_offsets else None

        translation_step = torch.linalg.vector_norm(
            translation[1:] - translation[:-1], dim=-1
        )
        invalid_translation_step = pair_valid & (
            translation_step > self._max_camera_translation_step_m
        )
        if invalid_translation_step.any():
            first_invalid_offsets.append(
                int(
                    torch.nonzero(invalid_translation_step, as_tuple=False)[0].item()
                )
                + 1
            )

        relative_rotation = rotation[:-1].transpose(-1, -2) @ rotation[1:]
        relative_trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        rotation_cosine = ((relative_trace - 1.0) * 0.5).clamp(-1.0, 1.0)
        rotation_step_degrees = torch.rad2deg(torch.acos(rotation_cosine))
        invalid_rotation_step = pair_valid & (
            rotation_step_degrees > self._max_camera_rotation_step_degrees
        )
        if invalid_rotation_step.any():
            first_invalid_offsets.append(
                int(torch.nonzero(invalid_rotation_step, as_tuple=False)[0].item())
                + 1
            )

        return min(first_invalid_offsets) if first_invalid_offsets else None

    def _first_invalid_action_offset(
        self,
        action: Any,
        action_is_pad: Any,
        *,
        max_abs: float | None = None,
    ) -> int | None:
        action = torch.as_tensor(action)
        if action.ndim != 2:
            return 0
        valid = self._valid_action_timesteps(
            action_is_pad,
            time_steps=int(action.shape[0]),
            device=action.device,
        )
        if valid is None:
            return 0
        if not valid.any():
            return None

        invalid_rows = valid & ~torch.isfinite(action).all(dim=-1)
        if max_abs is not None:
            invalid_rows |= valid & (action.abs().amax(dim=-1) > max_abs)
        if not invalid_rows.any():
            return None
        return int(torch.nonzero(invalid_rows, as_tuple=False)[0].item())

    @staticmethod
    def _mask_invalid_action_chunk_supervision(output_dict: dict[str, Any]) -> None:
        output_dict["action_loss_mask"] = torch.zeros_like(
            output_dict["action_loss_mask"], dtype=torch.bool
        )
        if "action_dim_loss_mask" in output_dict:
            output_dict["action_dim_loss_mask"] = torch.zeros_like(
                output_dict["action_dim_loss_mask"], dtype=torch.bool
            )
        fast_action_indicator = output_dict.get("fast_action_indicator")
        if fast_action_indicator is not None:
            output_dict["lang_loss_masks"] = output_dict[
                "lang_loss_masks"
            ].masked_fill(fast_action_indicator.to(dtype=torch.bool), False)

    def _use_eef_dual_hand_prefix_to_6d(self, data_dict: dict[str, Any]) -> bool:
        if not self._eef_dual_hand_prefix_to_6d:
            return False
        if self._eef_dual_hand_prefix_to_6d_robot_types is None:
            return True
        return (
            str(data_dict.get("robot_type"))
            in self._eef_dual_hand_prefix_to_6d_robot_types
        )

    def __call__(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        """Applies the transformation pipeline to a data dictionary.

        Args:
            data_dict (dict[str, Any]): The input data dictionary.

        Returns:
            dict[str, Any]: The transformed data dictionary.
        """
        if self._is_vqa_sample(data_dict):
            return self._build_vqa_output(data_dict)

        output_dict: dict[str, Any] = {}

        if data_dict.get("embodiment_id") is not None:
            embodiment_id = int(data_dict["embodiment_id"])
        else:
            robot_type = data_dict["meta"].info["robot_type"]
            embodiment_id = self._resolve_robot_type_embodiment_id(robot_type)
        data_dict["embodiment_id"] = embodiment_id
        self._split_temporal_state_memory(data_dict)
        if data_dict.get("robot_type") is None:
            meta = data_dict.get("meta")
            meta_robot_type = (
                getattr(meta, "info", {}).get("robot_type")
                if meta is not None
                else None
            )
            if meta_robot_type is not None:
                data_dict["robot_type"] = str(meta_robot_type)
        norm_selector = self._get_norm_selector(data_dict, embodiment_id)
        delta_mask_key = (
            self._get_delta_mask_selector(data_dict, embodiment_id)
            if self.delta_action_masks
            else int(embodiment_id)
        )
        if (
            data_dict.get("end_effector_type") is None
            and delta_mask_key in self.end_effector_types
        ):
            data_dict["end_effector_type"] = self.end_effector_types[delta_mask_key]
        action_fps = self._get_action_fps(data_dict)
        data_dict["action_fps"] = action_fps

        first_invalid_action_offset: int | None = None
        apply_invalid_chunk_filter = self._invalid_action_chunk_filter_applies(
            data_dict
        )
        camera_key = self._resolve_chunk_anchor_camera_key(data_dict)
        if apply_invalid_chunk_filter:
            action = data_dict.get("action")
            camera_chunk = data_dict.get(camera_key)
            action_is_pad = data_dict.get("action_is_pad")
            detected_offsets = [
                0
                if action is None
                else self._first_invalid_action_offset(action, action_is_pad),
                0
                if camera_chunk is None
                else self._first_invalid_camera_offset(camera_chunk, action_is_pad),
            ]
            detected_offsets = [
                offset for offset in detected_offsets if offset is not None
            ]
            if detected_offsets:
                first_invalid_action_offset = min(detected_offsets)

        if self._chunk_anchor_camera_reframe_enabled:
            action = data_dict.get("action")
            camera_chunk = data_dict.get(camera_key)
            if (
                first_invalid_action_offset is None
                and action is not None
                and camera_chunk is not None
                and int(getattr(action, "ndim", 0)) >= 2
            ):
                data_dict["action"] = reframe_dual_hand_tcp_chunk_to_anchor_camera(
                    action,
                    _camera_chunk_to_mat4(camera_chunk),
                )

        if self.use_quaternion_to_6d:
            if self._use_eef_dual_hand_prefix_to_6d(data_dict):
                if "observation.state" in data_dict:
                    data_dict["observation.state"] = (
                        Embodiment3QuaternionTo6D.transform_tensor_dual_hand_quat_prefix(
                            data_dict["observation.state"]
                        )
                    )
                if "action" in data_dict:
                    data_dict["action"] = (
                        Embodiment3QuaternionTo6D.transform_tensor_dual_hand_quat_prefix(
                            data_dict["action"]
                        )
                    )
                if "action_is_pad" in data_dict:
                    ap = data_dict["action_is_pad"]
                    if ap.ndim >= 2 and int(ap.shape[-1]) > 0:
                        data_dict["action_is_pad"] = (
                            Embodiment3QuaternionTo6D.transform_mask_dual_hand_quat_prefix(
                                ap
                            )
                        )
                if (
                    self.state_input_mode == "proprio_memory"
                    and "observation.state_memory" in data_dict
                ):
                    data_dict["observation.state_memory"] = (
                        Embodiment3QuaternionTo6D.transform_tensor_dual_hand_quat_prefix(
                            data_dict["observation.state_memory"]
                        )
                    )
            else:
                data_dict = self.embodiment3_quaternion_to_6d_transform(data_dict)
                if (
                    self.state_input_mode == "proprio_memory"
                    and "observation.state_memory" in data_dict
                ):
                    data_dict["observation.state_memory"] = (
                        Embodiment3QuaternionTo6D._transform_pose_tensor(
                            data_dict["observation.state_memory"],
                            embodiment_id,
                        )
                    )

        if self.use_delta_joint_actions:
            data_dict = self.delta_action_transform(data_dict)

        _model_action_dim = int(self.pad_transform.action_dim)
        if int(data_dict["observation.state"].shape[-1]) > _model_action_dim:
            data_dict["observation.state"] = data_dict["observation.state"][
                ..., :_model_action_dim
            ]
        if (
            "action" in data_dict
            and int(data_dict["action"].shape[-1]) > _model_action_dim
        ):
            data_dict["action"] = data_dict["action"][..., :_model_action_dim]
        if "action_is_pad" in data_dict:
            _ap = data_dict["action_is_pad"]
            if _ap.ndim >= 2 and int(_ap.shape[-1]) > _model_action_dim:
                data_dict["action_is_pad"] = _ap[..., :_model_action_dim]

        agent_pos_mask = None
        if self.state_input_mode == "proprio_anchor":
            raw_state = data_dict["observation.state"]
            agent_pos_mask = (~torch.isnan(raw_state)).to(dtype=torch.float32)
            data_dict["observation.state"] = torch.nan_to_num(raw_state, nan=0.0)

        data_dict["observation.state"] = self.state_normalize_transform(
            data_dict["observation.state"], embodiment_id=norm_selector
        )
        data_dict["action"] = self.action_normalize_transform(
            data_dict["action"], embodiment_id=norm_selector
        )
        if apply_invalid_chunk_filter:
            normalized_invalid_offset = self._first_invalid_action_offset(
                data_dict["action"],
                data_dict.get("action_is_pad"),
                max_abs=self._max_normalized_action_abs,
            )
            if normalized_invalid_offset is not None:
                first_invalid_action_offset = (
                    normalized_invalid_offset
                    if first_invalid_action_offset is None
                    else min(first_invalid_action_offset, normalized_invalid_offset)
                )

        if first_invalid_action_offset is not None:
            data_dict["action"] = torch.zeros_like(data_dict["action"])
        invalid_action_chunk = first_invalid_action_offset is not None
        if (
            self.state_input_mode == "proprio_memory"
            and "observation.state_memory" in data_dict
        ):
            data_dict["observation.state_memory"] = self.state_normalize_transform(
                data_dict["observation.state_memory"],
                embodiment_id=norm_selector,
            )
        raw_action_dim = int(data_dict["action"].shape[-1])
        raw_state_dim = int(data_dict["observation.state"].shape[-1])
        is_robot_moving = self._get_is_robot_moving(data_dict)
        is_body_moving = (
            False if is_robot_moving else self._get_is_body_moving(data_dict)
        )
        action_state_layout = self._resolve_action_state_layout(
            delta_mask_key=delta_mask_key,
            raw_action_dim=raw_action_dim,
            raw_state_dim=raw_state_dim,
            is_robot_moving=is_robot_moving,
            is_body_moving=is_body_moving,
        )
        action_mask_dim = (
            None
            if action_state_layout is None
            else action_state_layout.action_supervised_dim
        )
        state_input_dim = (
            min(raw_state_dim, int(self.pad_transform.action_dim))
            if action_state_layout is None
            else action_state_layout.state_input_dim
        )

        agent_pos_mask = self._mask_state_input_dims(
            data_dict,
            agent_pos_mask,
            state_input_dim=state_input_dim,
        )

        if self.state_input_mode == "proprio_memory":
            state_memory, state_memory_mask = self._build_state_memory_output(
                data_dict,
                data_dict["observation.state"],
            )
            data_dict["observation.state_memory"] = state_memory
            output_dict["observation.state_memory_mask"] = state_memory_mask
        elif self.state_input_mode == "proprio_anchor":
            proprioception, aligned_agent_pos_mask = self._align_proprio_anchor(
                data_dict["observation.state"],
                agent_pos_mask,
            )
            output_dict["observation.proprioception"] = proprioception
            output_dict["observation.agent_pos_mask"] = aligned_agent_pos_mask
            output_dict["observation.proprioception_present"] = torch.tensor(
                True, dtype=torch.bool
            )

        self._zero_non_action_dims_outside_action_mask(
            data_dict,
            delta_mask_key=delta_mask_key,
            raw_action_dim=raw_action_dim,
            raw_state_dim=raw_state_dim,
            is_robot_moving=is_robot_moving,
            is_body_moving=is_body_moving,
            action_mask_dim=action_mask_dim,
        )

        output_dict["images"], output_dict["image_masks"], image_transform_params = (
            self.image_transform(data_dict)
        )

        try:
            (
                output_dict["lang_tokens"],
                output_dict["lang_masks"],
                output_dict["lang_att_masks"],
                output_dict["lang_loss_masks"],
                output_dict["fast_action_indicator"],
                output_dict["subtask_indicator"],
                predict_subtask_only,
            ) = self.prompt_tokenizer_transform(data_dict)
        except Exception as e:
            meta = data_dict.get("meta")
            repo_id = getattr(meta, "repo_id", None)
            root = getattr(meta, "root", None)
            episode_index = data_dict.get("episode_index")
            frame_index = data_dict.get("frame_index")
            task = data_dict.get("task")
            task_index = data_dict.get("task_index")
            raise RuntimeError(
                f"prompt_tokenizer_transform failed: repo_id={repo_id!r}, root={root!r}, "
                f"episode_index={episode_index!r}, frame_index={frame_index!r}, "
                f"task={task!r}, task_type={type(task).__name__}, task_index={task_index!r}"
            ) from e

        data_dict = self.pad_transform(data_dict)
        output_dict["observation.state"] = data_dict["observation.state"]
        if self.state_input_mode == "proprio_memory":
            output_dict["observation.state_memory"] = data_dict[
                "observation.state_memory"
            ]
        output_dict["action"] = data_dict["action"]

        output_dict["skip_loss_mask"] = self._get_skip_loss_mask(data_dict)

        if self.trajectory_transform is not None:
            traj, traj_is_pad = self.trajectory_transform(
                data_dict,
                chunk_size=data_dict["action"].shape[0],
                image_transform_params=image_transform_params,
            )
            output_dict["traj"] = traj
            output_dict["traj_loss_mask"] = ~traj_is_pad

        action_is_pad = data_dict["action_is_pad"]
        output_dict["action_loss_mask"] = self._build_temporal_action_loss_mask(
            action_is_pad
        )
        if self.mask_unsupervised_action_dims_for_noise:
            output_dict["action_dim_loss_mask"] = self._build_action_dim_loss_mask(
                action_is_pad,
                action_mask_dim=action_mask_dim,
                padded_action_dim=int(data_dict["action"].shape[-1]),
            )

        if predict_subtask_only:
            # No diffusion loss when only predicting subtask (no FAST action tokens in suffix)
            output_dict["action_loss_mask"] = torch.zeros_like(
                output_dict["action_loss_mask"]
            )
            if "action_dim_loss_mask" in output_dict:
                output_dict["action_dim_loss_mask"] = torch.zeros_like(
                    output_dict["action_dim_loss_mask"]
                )

        if self._invalid_action_chunk_filter_enabled:
            output_dict["invalid_action_chunk"] = torch.tensor(
                invalid_action_chunk, dtype=torch.bool
            )
        if invalid_action_chunk:
            self._mask_invalid_action_chunk_supervision(output_dict)

        output_dict["embodiment_id"] = torch.tensor(embodiment_id, dtype=torch.long)
        output_dict["action_fps"] = torch.tensor(action_fps, dtype=torch.float32)

        # Lightweight per-sample provenance for NaN-loss debugging. These are
        # plain Python str/int (NOT tensors): DefaultCollator passes them through
        # as a list, so they never reach the GPU or the model forward and add no
        # per-step cost. They let the trainer name the offending sample when a
        # NaN loss is detected.
        meta = data_dict.get("meta")
        output_dict["debug_repo_id"] = getattr(meta, "repo_id", None)
        episode_index = data_dict.get("episode_index")
        frame_index = data_dict.get("frame_index")
        output_dict["debug_episode_index"] = (
            int(episode_index) if episode_index is not None else -1
        )
        output_dict["debug_frame_index"] = (
            int(frame_index) if frame_index is not None else -1
        )

        return output_dict
