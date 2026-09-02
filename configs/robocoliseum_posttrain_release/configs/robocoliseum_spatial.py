"""GigaBrain-0.7 post-training config for spatial tasks."""

import json
import os
from pathlib import Path

# Importing the package registers the RoboColiseum remapping transform.
import robocoliseum_ext  # noqa: F401
from robocoliseum_ext.annotated_dataset import AnnotatedTaskLeRobotDataset  # noqa: F401
from robocoliseum_ext.posttrain_config import (
    ACTION_CHUNK,
    AGENT_POS_CONFIG,
    CAMERA_KEYS,
    DEFAULT_PRETRAINED_CKPT,
    EMBODIMENT_ID,
    EXPERIMENT_ROOT,
    FAST_TOKENIZER_PATH,
    NORM_STATS_ROOT,
    ROBOT_TYPE,
    SPATIAL_TASKS,
    TOKENIZER_PATH,
    _checkpoint_settings,
    _validate_dataset_schema,
)
from robocoliseum_ext.state_remap import UNIFIED_ACTION_DIM


suite = 'spatial'
run_name = 'robocoliseum_spatial'
is_robot_moving = False
# Keep delta supervision fixed to the offline 17-D g2a_sim action layout.
delta_mask = [True] * 7 + [False] + [True] * 7 + [False, True]
data_paths = _validate_dataset_schema(
    suite,
    SPATIAL_TASKS,
    offline_17d=True,
    episode_action_prompts=True,
)


def _validate_robot_moving_flag() -> None:
    """Require every dataset to match the configured mobile-base behavior."""
    for data_path in data_paths:
        info_path = Path(data_path) / 'meta' / 'info.json'
        info = json.loads(info_path.read_text())
        observed = info.get('is_robot_moving', False)
        if not isinstance(observed, bool):
            raise ValueError(f'{info_path}: is_robot_moving must be a bool')
        if observed != is_robot_moving:
            raise ValueError(
                f'{info_path}: expected is_robot_moving={is_robot_moving}, '
                f'got {observed}'
            )


_validate_robot_moving_flag()
data_or_config = []
for data_path in data_paths:
    data_or_config.append(
        dict(
            _class_name='AnnotatedTaskLeRobotDataset',
            data_path=data_path,
            delta_info={'action': ACTION_CHUNK},
            meta_name='meta',
        )
    )

norm_stats_path = str(
    NORM_STATS_ROOT / f'{run_name}_{UNIFIED_ACTION_DIM}d.json'
)
pretrained_ckpt = Path(
    os.environ.get('GIGABRAIN_PRETRAINED_CKPT', str(DEFAULT_PRETRAINED_CKPT))
)
state_input_mode, observation_memory_size = _checkpoint_settings(pretrained_ckpt)

prompt_cfg = dict(
    tokenizer_model_path=TOKENIZER_PATH,
    fast_tokenizer_path=FAST_TOKENIZER_PATH,
    enable_control_mode_token=True,
    enable_end_effector_token=True,
    max_length=120 if state_input_mode == 'proprio_anchor' else 300,
    state_input_mode=state_input_mode,
    discrete_state_input=state_input_mode == 'prompt',
    discrete_state_input_for_pose_embodiments=False,
    encode_action_input=False,
    encode_sub_task_input=False,
    encoded_action_horizon=ACTION_CHUNK,
    sample_ratios=dict(
        input_task=1,
        input_subtask=0,
        input_task_target_subtask=0,
        input_task_target_action=0,
        input_subtask_target_action=0,
        input_task_target_subtask_action=0,
    ),
    vlm_type='paligemma2',
)

inner_transform = dict(
    type='GigaBrain07Transform',
    is_train=True,
    state_input_mode=state_input_mode,
    observation_memory_size=observation_memory_size,
    use_quaternion_to_6d=False,
    robot_type_embodiment_id_overrides={ROBOT_TYPE: EMBODIMENT_ID},
    delta_action_cfg=dict(
        selector='embodiment_id',
        use_delta_joint_actions=True,
        mask_unsupervised_action_dims_for_noise=True,
        mask={str(EMBODIMENT_ID): list(delta_mask)},
    ),
    norm_cfg=dict(
        selector='data_path',
        norm_stats_path=[dict(data_paths=data_paths, path=norm_stats_path)],
        use_quantiles=True,
        enable_clamp=False,
    ),
    image_cfg=dict(
        resize_imgs_with_padding=[224, 224],
        enable_image_aug=True,
        present_img_keys=list(CAMERA_KEYS),
        enable_depth_img=False,
        vlm_type='paligemma2',
    ),
    prompt_cfg=prompt_cfg,
)
if state_input_mode == 'proprio_anchor':
    inner_transform['agent_pos_config'] = dict(AGENT_POS_CONFIG)

config = dict(
    runners=['gigabrain07.GigaBrain07Trainer'],
    project_dir=str(EXPERIMENT_ROOT / 'runs' / run_name),
    launch=dict(
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        distributed_type='FSDP',
        fsdp_config=dict(
            fsdp_version='2',
            fsdp_auto_wrap_policy='TRANSFORMER_BASED_WRAP',
            fsdp_transformer_layer_cls_to_wrap=(
                'SiglipEncoderLayer,Gemma2DecoderLayerWithExpert'
            ),
            fsdp_cpu_ram_efficient_loading='false',
            fsdp_state_dict_type='FULL_STATE_DICT',
        ),
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=data_or_config,
            batch_size_per_gpu=32,
            num_workers=16,
            persistent_workers=False,
            transform=inner_transform,
            sampler=dict(type='DefaultSampler', shuffle=True),
        ),
        test=dict(),
    ),
    models=dict(
        vlm_type='paligemma2',
        pretrained=str(pretrained_ckpt),
        enable_knowledge_insulation=False,
        enable_next_token_prediction=False,
    ),
    optimizers=dict(
        type='AdamW',
        betas=(0.9, 0.95),
        lr=2.5e-5,
        eps=1e-8,
        weight_decay=1e-5,
    ),
    schedulers=dict(
        type='WarmupCosineScheduler',
        warmup_steps=1000,
        decay_steps=30000,
        end_value=0.1,
    ),
    train=dict(
        resume=False,
        resume_compile_warmup=False,
        max_steps=50000,
        gradient_accumulation_steps=1,
        llm_loss_weight=0.0,
        mixed_precision='no',
        max_grad_norm=1.0,
        checkpoint_interval=5000,
        checkpoint_total_limit=10,
        checkpoint_keeps=list(range(5000, 50001, 5000)),
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with='tensorboard',
        log_interval=50,
        with_ema=True,
        dynamo_config=dict(backend='inductor'),
        activation_checkpointing=True,
        activation_class_names=['Gemma2DecoderLayerWithExpert__##__18'],
    ),
)
if state_input_mode == 'proprio_anchor':
    config['models']['state_input_mode'] = state_input_mode
    config['models']['observation_memory_size'] = observation_memory_size
    config['models']['agent_pos_config'] = dict(AGENT_POS_CONFIG)
