"""Shared GigaBrain-0.7 post-training configuration builder for RoboColiseum suites.

This module centralizes resource paths, task lists, state and action layouts, and
training hyperparameters. Configurations use ``build_posttrain_config`` to keep
dataset validation, checkpoint compatibility checks, and transform order aligned.
"""

import json
import os
from pathlib import Path
from typing import Sequence

# Importing the package registers RoboColiseumJointActionRemapTransform with the
# training framework so configuration dictionaries can resolve it by name.
import robocoliseum_ext  # noqa: F401
from robocoliseum_ext.state_remap import UNIFIED_ACTION_DIM, build_delta_mask


# The action chunk must match dataset delta_info, prompt encoding, and model horizon.
ACTION_CHUNK = 50
# Fixed GigaBrain embodiment id for g2a_sim and its delta-action mask.
EMBODIMENT_ID = 1
ROBOT_TYPE = 'g2a_sim'
# Proprio-anchor mode pads states to the model's 32 input slots.
AGENT_POS_CONFIG = {'state': 32}

def _required_path(name: str) -> Path:
    """Return a user-provided path and reject missing or empty values."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable to a valid path.")
    return Path(value).expanduser()


EXPERIMENT_ROOT = _required_path("ROBOCOLISEUM_OUTPUT_ROOT")
NORM_STATS_ROOT = _required_path("ROBOCOLISEUM_NORM_STATS_ROOT")
DEFAULT_PRETRAINED_CKPT = _required_path("GIGABRAIN_PRETRAINED_CKPT")
TOKENIZER_PATH = str(_required_path("PALIGEMMA2_TOKENIZER_PATH"))
FAST_TOKENIZER_PATH = str(_required_path("FAST_TOKENIZER_PATH"))

# Training reads three RGB cameras and does not use depth images.
CAMERA_KEYS = [
    'observation.images.top_head',
    'observation.images.hand_left',
    'observation.images.hand_right',
]

# Instruction and robustness tasks share one data distribution and norm-stats file.
INSTRUCTION_ROBUST_TASKS = (
    'pick_billards_color_500',
    'pick_block_color_500',
    'pick_block_number_500',
    'pick_block_shape_500',
    'pick_block_size_500',
    'pick_common_sense_500',
    'pick_follow_logic_(or)_500',
    'pick_object_type_500',
    'pick_specific_object_500',
    'straighten_object_500',
)
# Spatial tasks use an independent data distribution.
SPATIAL_TASKS = (
    'pick_object_relative_position_absolute',
    'pick_object_relative_position_relative',
    'place_beverage_to_anothers_position',
    'place_object_relative_position',
    'sort_cubes_by_size',
    'sort_number_from_small_to_big',
    'stack_bowls',
    'stack_three_building_blocks',
)
# Manipulation tasks enable waist-joint supervision through is_body_moving.
MANIP_TASKS = (
    'clean_the_desktop',
    'hold_pot',
    'open_door',
    'place_block_into_box',
    'pour_workpiece',
    'scoop_popcorn',
    'sorting_packages',
    'stock_and_straighten_shelf',
    'take_wrong_item_shelf',
)


def _checkpoint_settings(checkpoint: Path) -> tuple[str, int]:
    """Read and validate checkpoint settings that must match training.

    Args:
        checkpoint: Complete ``model_ema`` directory, not an isolated weight file.

    Returns:
        A pair containing ``state_input_mode`` and ``observation_memory_size``.

    Raises:
        FileNotFoundError: The checkpoint does not contain ``config.json``.
        ValueError: Model type, action width, or state settings are incompatible.
    """
    # Validation requires complete model metadata; weights alone are insufficient.
    config_path = checkpoint / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(
            f'GigaBrain checkpoint config not found: {config_path}. '
            'Set GIGABRAIN_PRETRAINED_CKPT to a full model_ema directory.'
        )
    model_config = json.loads(config_path.read_text())
    vlm_type = model_config.get('vlm_type')
    if vlm_type != 'paligemma2':
        raise ValueError(f'expected a paligemma2 GigaBrain-0.7 checkpoint, got {vlm_type!r}')
    max_action_dim = int(model_config.get('max_action_dim', 0))
    if max_action_dim < 17:
        raise ValueError(f'checkpoint max_action_dim={max_action_dim} cannot supervise 17-D manip')
    state_input_mode = str(model_config.get('state_input_mode', 'prompt'))
    if state_input_mode not in ('prompt', 'proprio_anchor'):
        raise ValueError(f'unsupported checkpoint state_input_mode={state_input_mode!r}')
    observation_memory_size = int(model_config.get('observation_memory_size', 1))
    if observation_memory_size != 1:
        raise ValueError(
            'these single-frame configs require observation_memory_size=1, '
            f'got {observation_memory_size}'
        )
    return state_input_mode, observation_memory_size


def _validate_dataset_schema(
    suite: str,
    tasks: Sequence[str],
    *,
    offline_17d: bool,
    episode_action_prompts: bool,
) -> list[str]:
    """Fail before training if field_descriptions no longer match our indices."""
    expected_fields = {
        'observation.state': {
            'state/left_effector/position': [0],
            'state/right_effector/position': [1],
            'state/joint/position': list(range(30, 44)),
            'state/waist/position': list(range(61, 66)),
        },
        'action': {
            'action/left_effector/position': [0],
            'action/right_effector/position': [1],
            'action/joint/position': list(range(16, 30)),
            'action/waist/position': list(range(33, 38)),
        },
    }
    data_root_env = (
        "ROBOCOLISEUM_PROCESSED_DATA_ROOT" if offline_17d else "ROBOCOLISEUM_DATA_ROOT"
    )
    data_root = _required_path(data_root_env)
    if episode_action_prompts:
        from robocoliseum_ext.annotated_dataset import load_episode_action_prompts

    data_paths = []
    for task in tasks:
        data_path = data_root / suite / task
        info_path = data_path / 'meta' / 'info.json'
        if not info_path.is_file():
            raise FileNotFoundError(f'dataset metadata not found: {info_path}')
        info = json.loads(info_path.read_text())
        if info.get('robot_type') != ROBOT_TYPE:
            raise ValueError(f'{info_path}: expected robot_type={ROBOT_TYPE!r}')
        expected_body_moving = suite == 'manipulation'
        observed_body_moving = info.get('is_body_moving', False)
        if not isinstance(observed_body_moving, bool):
            raise ValueError(f'{info_path}: is_body_moving must be a bool')
        if observed_body_moving != expected_body_moving:
            raise ValueError(
                f'{info_path}: expected is_body_moving={expected_body_moving}, '
                f'got {observed_body_moving}'
            )
        for feature_name, fields in expected_fields.items():
            feature = info['features'][feature_name]
            if offline_17d:
                shape = feature.get('shape')
                if shape != [UNIFIED_ACTION_DIM]:
                    raise ValueError(
                        f'{info_path}: expected {feature_name} shape '
                        f'[{UNIFIED_ACTION_DIM}], got {shape}'
                    )
                continue
            descriptions = feature.get('field_descriptions', {})
            for field_name, expected_indices in fields.items():
                observed = descriptions.get(field_name, {}).get('indices')
                if observed != expected_indices:
                    raise ValueError(
                        f'{info_path}: {field_name} indices {observed} '
                        f'do not match expected {expected_indices}'
                    )
        if episode_action_prompts:
            prompts = load_episode_action_prompts(str(data_path))
            expected_episode_indices = set(range(int(info['total_episodes'])))
            if set(prompts) != expected_episode_indices:
                raise ValueError(
                    f'{data_path}: annotation keys do not match episode indices '
                    f'0:{info["total_episodes"]}'
                )
        data_paths.append(str(data_path))
    return data_paths


def build_posttrain_config(
    *,
    suite: str,
    tasks: Sequence[str],
    run_name: str,
    offline_17d: bool = False,
    episode_action_prompts: bool = False,
) -> dict:
    """Build a complete post-training config for one independent task suite.

    Args:
        suite: Suite directory name under the selected dataset root.
        tasks: Task names included in this training run.
        run_name: Experiment name used for outputs and normalization statistics.

    Returns:
        A nested configuration dictionary accepted by ``GigaBrain07Trainer``.

    Notes:
        This function builds configuration only. It reads checkpoint and dataset
        metadata immediately to fail before training when inputs are incompatible.
    """
    # Read the selected complete checkpoint and validate its config.json below.
    pretrained_ckpt = Path(
        os.environ.get('GIGABRAIN_PRETRAINED_CKPT', str(DEFAULT_PRETRAINED_CKPT))
    )
    # The checkpoint determines state input mode; task configs cannot override it.
    state_input_mode, observation_memory_size = _checkpoint_settings(pretrained_ckpt)
    data_paths = _validate_dataset_schema(
        suite,
        tasks,
        offline_17d=offline_17d,
        episode_action_prompts=episode_action_prompts,
    )
    # Every suite uses the same 17-D tensor schema; metadata controls the effective width.
    norm_stats_path = (
        NORM_STATS_ROOT / f'{run_name}_{UNIFIED_ACTION_DIM}d.json'
    )

    dataset_class_name = (
        'EpisodeActionPromptLeRobotDataset'
        if episode_action_prompts
        else 'LeRobotDataset'
    )
    dataset_configs = [
        dict(
            _class_name=dataset_class_name,
            data_path=data_path,
            delta_info={'action': ACTION_CHUNK},
            meta_name='meta',
        )
        for data_path in data_paths
    ]

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
            mask={str(EMBODIMENT_ID): build_delta_mask()},
        ),
        norm_cfg=dict(
            selector='data_path',
            norm_stats_path=[
                dict(data_paths=list(data_paths), path=str(norm_stats_path)),
            ],
            use_quantiles=True,
            enable_clamp=True,
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
                data_or_config=dataset_configs,
                batch_size_per_gpu=32,
                num_workers=16,
                persistent_workers=False,
                transform=(
                    inner_transform
                    if offline_17d
                    else dict(
                        type='RoboColiseumJointActionRemapTransform',
                        inner=inner_transform,
                        state_input_mode=state_input_mode,
                        observation_memory_size=observation_memory_size,
                        prompt_cfg=dict(prompt_cfg),
                    )
                ),
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
            checkpoint_keeps=[5000, 10000, 15000,20000,25000,30000,35000,40000,45000,50000],
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
        config['dataloaders']['train']['transform']['agent_pos_config'] = dict(AGENT_POS_CONFIG)
    return config
