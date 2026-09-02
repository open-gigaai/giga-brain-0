"""GigaBrain-0.7 EBench 26-task post-training from the PaliGemma2 200k EMA."""

# EBench metadata uses the legacy "List" feature name.
from pathlib import Path
from datasets import Sequence
from datasets.features.features import _FEATURE_TYPES
_FEATURE_TYPES.setdefault("List", Sequence)

PROJECT_ROOT = "/mnt/pfs/users/wenyao.xue/code/.shared/results/gigabrain1.0/checkpoints"
DATA_ROOT = "/mnt/pfs/users/wenyao.xue/code/.shared/data/EBench-Dataset"


def collect_ebench_lerobot_paths(data_root: Path) -> list[str]:
    """Collect the 9 long-horizon, 10 simple-PnP, and 7 teleop datasets."""
    data_paths = [
        str(info_path.parents[1])
        for info_path in sorted(data_root.glob("*/*/meta/info.json"))
    ]
    if len(data_paths) != 26:
        raise ValueError(
            f"Expected 26 EBench datasets under {data_root}, found {len(data_paths)}"
        )
    return data_paths


DATA_PATHS = collect_ebench_lerobot_paths(Path(DATA_ROOT))
NORM_STATS_PATH = (
    "/mnt/pfs/users/wenyao.xue/code/.shared/results/gigabrain1.0/assets/"
    "ebench_generalist_14d17d_first_gripper_base_delta/norm_stats.json"
)
PRETRAINED_PALIGEMMA_PATH = (
    "/shared_disk/wangyunmo/"
    "gb1_pg2_ruev_0707_coarse_emb_cycle_proprio_anchor_new_data_200k/"
    "model_ema/"
    "diffusion_pytorch_model.bin"
)
PALIGEMMA2_TOKENIZER_PATH = (
    "/shared_disk/models/huggingface/models--google--paligemma2-3b-pt-224"
)
FAST_TOKENIZER_PATH = (
    "/shared_disk/models/huggingface/models--physical-intelligence--fast"
)

ACTION_CHUNK = 50
AGENT_POS_CONFIG = {"state": 32}
CAMERA_KEYS = (
    "video.overlook_camera_view",
    "video.left_camera_view",
    "video.right_camera_view",
)

# gripper_1_left ≈ gripper_1_right。
# gripper_2_left ≈ gripper_2_right。
# state  = arm_1_joints(6) + gripper_1_left(1)
#        + arm_2_joints(6) + gripper_2_left(1) = 14
# action = arm_1_joints(6) + gripper_1_left(1)
#        + arm_2_joints(6) + gripper_2_left(1) + base_delta(3) = 17
EBENCH_REPACK = dict(
    state=[
        dict(key="state.joints", start=0, end=6),
        dict(key="state.gripper", start=0, end=1),
        dict(key="state.joints", start=6, end=12),
        dict(key="state.gripper", start=2, end=3),
    ],
    action=[
        dict(key="action.joints", start=0, end=6),
        dict(key="action.gripper", start=0, end=1),
        dict(key="action.joints", start=6, end=12),
        dict(key="action.gripper", start=2, end=3),
        dict(key="action.base_delta"),
    ],
)


# GigaBrain07Transform expects canonical "observation.state" and "action"
# tensors, while EBench stores joints, grippers, and base deltas in separate
# fields. The embodiment ID only selects robot-specific transform settings; it
# does not repack those fields. This config-local adapter performs that schema
# conversion, and registration lets the config builder resolve it by type name.
import torch
from giga_brain_0 import GigaBrain07Transform
from giga_train import TRANSFORMS
@TRANSFORMS.register
class GigaBrain07EBenchTransform(GigaBrain07Transform):
    """Convert split EBench state/action fields to the GB0.7 canonical schema."""

    # EBench stores joints, grippers and base motion in separate columns.
    @staticmethod
    def _concat_fields(data_dict: dict, fields: list[dict]) -> torch.Tensor:
        values = []
        for field in fields:
            value = torch.as_tensor(data_dict[field["key"]])
            start = field.get("start")
            end = field.get("end")
            values.append(value if start is None else value[..., start:end])
        return torch.cat(values, dim=-1)

    # The base transform requires observation.state/action before delta and norm.
    def __call__(self, data_dict: dict) -> dict:
        data_dict = dict(data_dict)
        data_dict["observation.state"] = self._concat_fields(
            data_dict, EBENCH_REPACK["state"]
        )
        data_dict["action"] = self._concat_fields(
            data_dict, EBENCH_REPACK["action"]
        )

        action_pad_masks = [
            torch.as_tensor(data_dict[f'{field["key"]}_is_pad'], dtype=torch.bool)
            for field in EBENCH_REPACK["action"]
            if f'{field["key"]}_is_pad' in data_dict
        ]
        data_dict["action_is_pad"] = (
            torch.stack(action_pad_masks).any(dim=0)
            if action_pad_masks
            else torch.zeros(data_dict["action"].shape[:-1], dtype=torch.bool)
        )
        data_dict["is_robot_moving"] = True
        return super().__call__(data_dict)


# EBench lift2 use AgileX embodiment id。
EBENCH_LIFT2_EMBODIMENT_ID = 0
EBENCH_LIFT2_DELTA_MASK = [
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    False,
    False,
    False,
]


DATASET_CONFIGS = [
    dict(
        _class_name="LeRobotDataset",
        data_path=data_path,
        delta_info={
            "action.joints": ACTION_CHUNK,
            "action.gripper": ACTION_CHUNK,
            "action.base_delta": ACTION_CHUNK,
        },
        meta_name="meta",
        video_backend="pyav",
        torchcodec_decoder_cache_size=128,
        decode_video_keys=CAMERA_KEYS,
    )
    for data_path in DATA_PATHS
]
delta_mask_by_embodiment_id = {
    str(EBENCH_LIFT2_EMBODIMENT_ID): list(EBENCH_LIFT2_DELTA_MASK),
}
dataset_groups = [
    (
        "ebench_generalist",
        list(DATA_PATHS),
        dict(EBENCH_REPACK),
        NORM_STATS_PATH,
    )
]


config = dict(
    runners=["gigabrain07.GigaBrain07Trainer"],
    project_dir=(
        f"{PROJECT_ROOT}/"
        "gb07_pg2_ebench_generalist_14d17d_first_gripper_base_delta_100k_from_200k"
    ),
    launch=dict(
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        distributed_type="FSDP",
        fsdp_config=dict(
            fsdp_version="2",
            fsdp_auto_wrap_policy="TRANSFORMER_BASED_WRAP",
            fsdp_transformer_layer_cls_to_wrap=(
                "SiglipEncoderLayer,Gemma2DecoderLayerWithExpert"
            ),
            fsdp_cpu_ram_efficient_loading="false",
            fsdp_state_dict_type="FULL_STATE_DICT",
        ),
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=dict(
                _class_name="WeightedConcatDataset",
                datasets=[DATASET_CONFIGS],
                sampling_weights=[1.0],
                group_names=["robot_ebench_generalist"],
            ),
            batch_size_per_gpu=32,
            num_workers=8,
            persistent_workers=False,
            transform=dict(
                type="GigaBrain07EBenchTransform",
                is_train=True,
                state_input_mode="proprio_anchor",
                observation_memory_size=1,
                agent_pos_config=dict(AGENT_POS_CONFIG),
                use_quaternion_to_6d=False,
                robot_type_embodiment_id_overrides={
                    "lift2": EBENCH_LIFT2_EMBODIMENT_ID,
                },
                delta_action_cfg=dict(
                    selector="embodiment_id",
                    use_delta_joint_actions=True,
                    mask_unsupervised_action_dims_for_noise=True,
                    mask={
                        embodiment_id: list(mask)
                        for embodiment_id, mask in delta_mask_by_embodiment_id.items()
                    },
                ),
                norm_cfg=dict(
                    selector="data_path",
                    norm_stats_path=[
                        dict(data_paths=list(DATA_PATHS), path=NORM_STATS_PATH)
                    ],
                    use_quantiles=True,
                    enable_clamp=False,
                ),
                image_cfg=dict(
                    resize_imgs_with_padding=[224, 224],
                    enable_image_aug=True,
                    present_img_keys=list(CAMERA_KEYS),
                    enable_depth_img=False,
                    vlm_type="paligemma2",
                ),
                prompt_cfg=dict(
                    tokenizer_model_path=PALIGEMMA2_TOKENIZER_PATH,
                    fast_tokenizer_path=FAST_TOKENIZER_PATH,
                    enable_control_mode_token=True,
                    enable_end_effector_token=True,
                    max_length=120,
                    state_input_mode="proprio_anchor",
                    discrete_state_input=False,
                    discrete_state_input_for_pose_embodiments=False,
                    encode_action_input=False,
                    encode_sub_task_input=False,
                    encoded_action_horizon=50,
                    sample_ratios=dict(
                        input_task=1,
                        input_subtask=0,
                        input_task_target_subtask=0,
                        input_task_target_action=0,
                        input_subtask_target_action=0,
                        input_task_target_subtask_action=0,
                    ),
                    vlm_type="paligemma2",
                ),
            ),
            sampler=dict(
                type="DefaultSampler",
                shuffle=True,
            ),
        ),
        test=dict(),
    ),
    models=dict(
        vlm_type="paligemma2",
        pretrained_paligemma_path=PRETRAINED_PALIGEMMA_PATH,
        vlm_hidden_size=2304,
        num_embodiments=8,
        has_action_expert=True,
        proj_width=1024,
        n_action_steps=ACTION_CHUNK,
        flow_action_horizon=50,
        max_action_dim=32,
        num_steps=10,
        enable_knowledge_insulation=False,
        action_loss_vlm_gradient_weight=1.0,
        enable_next_token_prediction=False,
        state_input_mode="proprio_anchor",
        observation_memory_size=1,
        agent_pos_config=dict(AGENT_POS_CONFIG),
        state_hidden_size=2048,
        proj_with_mask=True,
        enable_learnable_traj_token=False,
        encode_action_input=True,
        fast_tokenizer_path=FAST_TOKENIZER_PATH,
    ),
    optimizers=dict(
        type="AdamW",
        betas=(0.9, 0.95),
        lr=2.5e-5,
        eps=1e-8,
        weight_decay=1e-10,
    ),
    schedulers=dict(
        type="WarmupCosineScheduler",
        warmup_steps=1000,
        decay_steps=80000,
        end_value=0.1,
    ),
    train=dict(
        resume=True,
        resume_compile_warmup=True,
        max_steps=100000,
        gradient_accumulation_steps=1,
        llm_loss_weight=0.0,
        mixed_precision="no",
        checkpoint_interval=5000,
        checkpoint_total_limit=2,
        dynamo_config=dict(backend="inductor"),
        checkpoint_keeps=[20000, 50000, 80000, 100000],
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with="tensorboard",
        log_interval=50,
        with_ema=True,
        activation_checkpointing=True,
        activation_class_names=[
            "Gemma2DecoderLayerWithExpert__##__15",
        ],
        worker_profile=False,
    ),
)
