"""Self-contained GigaBrain-0.7 post-training config for RoboColiseum tasks."""

import json
import os
from pathlib import Path

from giga_datasets.datasets.dataset import register_dataset
from giga_datasets.datasets.lerobot_dataset import LeRobotDataset


ACTION_CHUNK = 50
AGENT_POS_CONFIG = {'state': 32}
CAMERA_KEYS = [
    'observation.images.top_head',
    'observation.images.hand_left',
    'observation.images.hand_right',
]
EMBODIMENT_ID = 1
ROBOT_TYPE = 'g2a_sim'
def _required_path(name: str) -> Path:
    """Return a user-provided path and reject missing or empty values."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable to a valid path.")
    return Path(value).expanduser()


EXPERIMENT_ROOT = _required_path("ROBOCOLISEUM_OUTPUT_ROOT")
PROCESSED_DATA_ROOT = _required_path("ROBOCOLISEUM_PROCESSED_DATA_ROOT")
NORM_STATS_ROOT = _required_path("ROBOCOLISEUM_NORM_STATS_ROOT")
DEFAULT_PRETRAINED_CKPT = _required_path("GIGABRAIN_PRETRAINED_CKPT")
TOKENIZER_PATH = str(_required_path("PALIGEMMA2_TOKENIZER_PATH"))
FAST_TOKENIZER_PATH = str(_required_path("FAST_TOKENIZER_PATH"))
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

# Favor the three tasks below the evaluation's overall score while retaining
# one full copy of every stronger task to limit catastrophic forgetting.
TASK_REPEAT_FACTORS = {
    'pick_billards_color_500': 1,
    'pick_block_color_500': 1,
    'pick_block_number_500': 1,
    'pick_block_shape_500': 3,
    'pick_block_size_500': 1,
    'pick_common_sense_500': 2,
    'pick_follow_logic_(or)_500': 1,
    'pick_object_type_500': 1,
    'pick_specific_object_500': 1,
    'straighten_object_500': 2,
}


def _load_annotated_action_steps(
    data_path: str,
) -> dict[int, tuple[tuple[int, int, str], ...]]:
    """Load and validate per-episode action texts from RoboColiseum annotations."""
    annotations_path = Path(data_path) / 'meta' / 'annotations.json'
    if not annotations_path.is_file():
        raise FileNotFoundError(f'annotations not found: {annotations_path}')

    annotations = json.loads(annotations_path.read_text())
    action_steps_by_episode = {}
    for raw_episode_index, annotation in annotations.items():
        episode_index = int(annotation.get('episode_index', raw_episode_index))
        if episode_index in action_steps_by_episode:
            raise ValueError(
                f'{annotations_path}: duplicate episode_index={episode_index}'
            )

        raw_action_steps = annotation.get('action_steps')
        if not isinstance(raw_action_steps, list) or not raw_action_steps:
            raise ValueError(
                f'{annotations_path}: episode {episode_index} has no action_steps'
            )

        action_steps = []
        for raw_step in raw_action_steps:
            action_text = raw_step.get('action_text')
            if not isinstance(action_text, str) or not action_text.strip():
                raise ValueError(
                    f'{annotations_path}: episode {episode_index} has empty action_text'
                )
            start_frame = int(raw_step['start_frame'])
            end_frame = int(raw_step['end_frame'])
            if start_frame < 0 or end_frame < start_frame:
                raise ValueError(
                    f'{annotations_path}: episode {episode_index} has invalid frame '
                    f'range [{start_frame}, {end_frame}]'
                )
            action_steps.append((start_frame, end_frame, action_text.strip()))

        action_steps.sort(key=lambda step: step[0])
        for previous, current in zip(action_steps, action_steps[1:]):
            if current[0] <= previous[1]:
                raise ValueError(
                    f'{annotations_path}: episode {episode_index} has overlapping '
                    f'action step ranges {previous[:2]} and {current[:2]}'
                )
        action_steps_by_episode[episode_index] = tuple(action_steps)

    return action_steps_by_episode


def _combine_action_texts(
    action_steps: tuple[tuple[int, int, str], ...],
) -> str:
    """Combine ordered action steps into one episode-level task prompt."""
    return '; then '.join(action_text for _, _, action_text in action_steps)


@register_dataset
class AnnotatedTaskLeRobotDataset(LeRobotDataset):
    """Use RoboColiseum action_text annotations as the policy task prompt."""

    def __init__(self, data_path: str, **kwargs) -> None:
        super().__init__(data_path=data_path, **kwargs)
        self._action_steps_by_episode = _load_annotated_action_steps(data_path)

    def _get_data(self, index: int) -> dict:
        data_dict = super()._get_data(index)
        episode_index = int(data_dict['episode_index'].item())
        try:
            action_steps = self._action_steps_by_episode[episode_index]
        except KeyError as error:
            raise KeyError(
                f'{self.data_path}: no action_text annotation for '
                f'episode_index={episode_index}'
            ) from error
        data_dict['task'] = _combine_action_texts(action_steps)
        return data_dict


def _checkpoint_settings(checkpoint: Path) -> tuple[str, int]:
    """Read and validate state settings from a complete model_ema checkpoint."""
    config_path = checkpoint / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'GigaBrain checkpoint config not found: {config_path}')
    model_config = json.loads(config_path.read_text())
    vlm_type = model_config.get('vlm_type')
    if vlm_type != 'paligemma2':
        raise ValueError(f'expected a paligemma2 checkpoint, got {vlm_type!r}')
    max_action_dim = int(model_config.get('max_action_dim', 0))
    if max_action_dim < 17:
        raise ValueError(
            f'checkpoint max_action_dim={max_action_dim} cannot supervise 17-D actions'
        )
    state_input_mode = str(model_config.get('state_input_mode', 'prompt'))
    if state_input_mode not in ('prompt', 'proprio_anchor'):
        raise ValueError(f'unsupported checkpoint state_input_mode={state_input_mode!r}')
    observation_memory_size = int(model_config.get('observation_memory_size', 1))
    if observation_memory_size != 1:
        raise ValueError(
            'this single-frame config requires observation_memory_size=1, '
            f'got {observation_memory_size}'
        )
    return state_input_mode, observation_memory_size


suite = 'instruction_and_robust'
run_name = 'robocoliseum_instruction_robust'
is_robot_moving = False
unified_action_dim = 17
delta_mask = [True] * 7 + [False] + [True] * 7 + [False, True]


def _validate_processed_dataset_schema() -> list[str]:
    """Validate the offline 17-D g2a_sim datasets before training starts."""
    data_paths = []
    for task in INSTRUCTION_ROBUST_TASKS:
        data_path = PROCESSED_DATA_ROOT / suite / task
        info_path = data_path / 'meta' / 'info.json'
        if not info_path.is_file():
            raise FileNotFoundError(f'processed dataset metadata not found: {info_path}')
        info = json.loads(info_path.read_text())
        if info.get('robot_type') != ROBOT_TYPE:
            raise ValueError(f'{info_path}: expected robot_type={ROBOT_TYPE!r}')
        observed_robot_moving = info.get('is_robot_moving', False)
        if not isinstance(observed_robot_moving, bool):
            raise ValueError(f'{info_path}: is_robot_moving must be a bool')
        if observed_robot_moving != is_robot_moving:
            raise ValueError(
                f'{info_path}: expected is_robot_moving={is_robot_moving}, '
                f'got {observed_robot_moving}'
            )
        action_steps_by_episode = _load_annotated_action_steps(str(data_path))
        expected_episode_indices = set(range(int(info['total_episodes'])))
        if set(action_steps_by_episode) != expected_episode_indices:
            raise ValueError(
                f'{data_path}: annotations episode indices do not match '
                f'0:{info["total_episodes"]}'
            )
        for feature_name in ('observation.state', 'action'):
            shape = info['features'][feature_name].get('shape')
            if shape != [unified_action_dim]:
                raise ValueError(
                    f'{info_path}: expected {feature_name} shape '
                    f'[{unified_action_dim}], got {shape}'
                )
        data_paths.append(str(data_path))
    return data_paths


data_paths = _validate_processed_dataset_schema()
data_or_config = []
for data_path in data_paths:
    task_name = Path(data_path).name
    for _ in range(TASK_REPEAT_FACTORS[task_name]):
        data_or_config.append(
            dict(
                _class_name='AnnotatedTaskLeRobotDataset',
                data_path=data_path,
                delta_info={'action': ACTION_CHUNK},
                meta_name='meta',
            )
        )

norm_stats_path = str(
    NORM_STATS_ROOT / f'{run_name}_{unified_action_dim}d.json'
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
    # NOTE: is_robot_moving is not a GigaBrain07Transform constructor argument.
    # The transform reads it per sample from the dataset metadata
    # (meta/info.json) via _get_is_robot_moving(), defaulting to False.
    # The module-level is_robot_moving flag above is only used to assert that
    # the offline datasets carry the expected value.
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
        decay_steps=40000,
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
