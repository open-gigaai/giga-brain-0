import gc
import functools
import json
import os
import socket
import time
import warnings
from contextlib import contextmanager
from typing import Any

from accelerate import DistributedType
from accelerate.utils.dataclasses import FP8BackendType
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing as torch_apply_activation_checkpointing,
    checkpoint_wrapper,
    offload_wrapper,
)
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.utils.data import BatchSampler

from giga_train import ParallelBatchSampler, Trainer, build_sampler, build_transform

from giga_models.models.vla.giga_brain_0.modeling_giga_brain_0 import GigaBrain07Policy
from giga_models.utils.action_horizon import downsample_flow_action_tensors
from .giga_brain_0_loss import GigaBrain07Loss
from .lerobot_open_cache import WORKER_PROFILE_KEY, wrap_lerobot_open_cache
from .lightweight_shard_index import wrap_materialized_shard_reader


# Subset of train-time `transform` config that the inference server / rollout script
# need to reconstruct preprocess + postprocess. Anything that's purely a training
# concern (sample_ratios, enable_image_aug=True, is_train=True) is irrelevant, but
# we dump these dicts whole and let the consumer pick fields.
_INFERENCE_CONFIG_KEYS = (
    'image_cfg',
    'prompt_cfg',
    'norm_cfg',
    'delta_action_cfg',
    'state_input_mode',
    'observation_memory_size',
    'agent_pos_config',
)
_PIPELINE_PROFILE_KEY = '__giga_pipeline_profile__'
_PIPELINE_WORKER_PROFILE_KEY = '__giga_pipeline_worker_profile__'
_PIPELINE_PROFILE_TIME_KEYS = (
    'step_total',
    'data_next',
    'data_getitem_total',
    'data_getitem_raw',
    'data_transform',
    'data_collate',
    'forward_step',
    'parse_losses',
    'backward_step',
    'backward',
    'grad_clip',
    'optimizer_step',
    'scheduler_step',
    'zero_grad',
    'ema',
    'print_step',
    'save_checkpoint_step',
)
_PIPELINE_PROFILE_COUNT_KEYS = (
    'steps',
    'micro_steps',
    'data_profiled_samples',
    'data_getitem_calls',
    'data_transform_calls',
    'data_collate_calls',
)
_PIPELINE_PROFILE_SAMPLE_KEYS = ('data_getitem_total', 'data_getitem_raw', 'data_transform')
_PIPELINE_PROFILE_BATCH_KEYS = ('data_collate',)
_WORKER_PROFILE_SUM_KEYS = (
    'getitem_count',
    'touch_count',
    'hit_count',
    'miss_count',
    'eviction_count',
    'gc_count',
)
_WORKER_PROFILE_LATEST_KEYS = (
    'open_cache_size',
    'max_open_per_worker',
)


def _str_to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value_str = str(value).strip().lower()
    if value_str in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if value_str in {'0', 'false', 'no', 'n', 'off'}:
        return False
    return default


def _debug_env_enabled(name: str) -> bool:
    return _str_to_bool(os.environ.get(name), False)


def _drop_disabled_paligemma_lm_head_weight(
    state_dict: dict[str, torch.Tensor],
    *,
    enable_next_token_prediction: bool,
) -> bool:
    """Drop the tied output-head alias when the target model has no LLM head."""
    if enable_next_token_prediction:
        return False
    return (
        state_dict.pop("paligemma_with_expert.lm_head.weight", None) is not None
    )


def _merge_pipeline_profile(target: dict[str, float], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        target[key] = float(target.get(key, 0.0)) + numeric_value


def _attach_pipeline_profile(data: Any, updates: dict[str, Any]) -> None:
    if isinstance(data, dict):
        profile = data.get(_PIPELINE_PROFILE_KEY)
        if not isinstance(profile, dict):
            profile = {}
            data[_PIPELINE_PROFILE_KEY] = profile
        _merge_pipeline_profile(profile, updates)
        return

    if isinstance(data, (list, tuple)):
        items = [item for item in data if isinstance(item, dict)]
        if not items:
            return
        shared_updates = {key: float(value) / len(items) for key, value in updates.items()}
        for item in items:
            _attach_pipeline_profile(item, shared_updates)


def _sync_state_input_config(data_config: Any, model_config: Any) -> None:
    if data_config is None or model_config is None:
        return
    transform_cfg = data_config.get('transform', None)
    if transform_cfg is None:
        return

    state_input_mode = model_config.get('state_input_mode', transform_cfg.get('state_input_mode', 'prompt'))
    if state_input_mode not in ('prompt', 'proprio_memory', 'proprio_anchor'):
        raise ValueError(
            "models.state_input_mode must be 'prompt', 'proprio_memory', or "
            f"'proprio_anchor', got {state_input_mode!r}"
        )
    observation_memory_size = int(
        model_config.get(
            'observation_memory_size',
            transform_cfg.get('observation_memory_size', 1),
        )
    )
    transform_state_input_mode = transform_cfg.get('state_input_mode', 'prompt')
    if transform_state_input_mode != state_input_mode:
        raise ValueError(
            f'dataloaders.train.transform.state_input_mode={transform_state_input_mode!r} '
            f'does not match models.state_input_mode={state_input_mode!r}'
        )
    transform_observation_memory_size = int(transform_cfg.get('observation_memory_size', observation_memory_size))
    if transform_observation_memory_size != observation_memory_size:
        raise ValueError(
            f'dataloaders.train.transform.observation_memory_size={transform_observation_memory_size} '
            f'does not match models.observation_memory_size={observation_memory_size}'
        )

    prompt_cfg = transform_cfg.get('prompt_cfg', {})
    if state_input_mode in ('proprio_memory', 'proprio_anchor') and bool(
        prompt_cfg.get('discrete_state_input', True)
    ):
        raise ValueError(
            f"models.state_input_mode={state_input_mode!r} is mutually exclusive with "
            "dataloaders.train.transform.prompt_cfg.discrete_state_input=True"
        )
    prompt_state_input_mode = prompt_cfg.get('state_input_mode', state_input_mode)
    if prompt_state_input_mode != state_input_mode:
        raise ValueError(
            f'prompt_cfg.state_input_mode={prompt_state_input_mode!r} does not match '
            f'models.state_input_mode={state_input_mode!r}'
        )
    prompt_cfg['state_input_mode'] = state_input_mode

    if state_input_mode == 'proprio_anchor':
        if model_config.get('vlm_type', 'paligemma2') != 'paligemma2':
            raise ValueError("proprio_anchor currently requires models.vlm_type='paligemma2'")
        # Visual memory may span multiple frames; the anchor still uses current state.
        model_agent_pos_config = dict(model_config.get('agent_pos_config', {}) or {})
        transform_agent_pos_config = dict(transform_cfg.get('agent_pos_config', {}) or {})
        if model_agent_pos_config and transform_agent_pos_config:
            if model_agent_pos_config != transform_agent_pos_config:
                raise ValueError(
                    'models.agent_pos_config does not match '
                    'dataloaders.train.transform.agent_pos_config'
                )
        agent_pos_config = model_agent_pos_config or transform_agent_pos_config
        if not agent_pos_config:
            raise ValueError('proprio_anchor requires agent_pos_config in model or transform config')
        model_config['agent_pos_config'] = agent_pos_config
        transform_cfg['agent_pos_config'] = agent_pos_config


def _collect_pipeline_profiles(data: Any) -> list[dict[str, float]]:
    if isinstance(data, dict):
        profile = data.get(_PIPELINE_PROFILE_KEY)
        return [profile] if isinstance(profile, dict) else []
    if isinstance(data, (list, tuple)):
        profiles: list[dict[str, float]] = []
        for item in data:
            profiles.extend(_collect_pipeline_profiles(item))
        return profiles
    return []


def _pop_pipeline_profiles(data: Any) -> list[dict[str, float]]:
    if isinstance(data, dict):
        profile = data.pop(_PIPELINE_PROFILE_KEY, None)
        return [profile] if isinstance(profile, dict) else []
    if isinstance(data, (list, tuple)):
        profiles: list[dict[str, float]] = []
        for item in data:
            profiles.extend(_pop_pipeline_profiles(item))
        return profiles
    return []


def _pop_worker_profiles(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        profile = data.pop(WORKER_PROFILE_KEY, None)
        return [profile] if isinstance(profile, dict) else []
    if isinstance(data, (list, tuple)):
        profiles: list[dict[str, Any]] = []
        for item in data:
            profiles.extend(_pop_worker_profiles(item))
        return profiles
    return []


def _sum_pipeline_profiles(profiles: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for profile in profiles:
        _merge_pipeline_profile(summary, profile)
    return summary


def _worker_profile_id(profile: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(profile.get('host', 'unknown')),
        int(profile.get('rank', -1)),
        int(profile.get('pid', -1)),
        int(profile.get('worker_id', -1)),
    )


def _summarize_worker_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    workers: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for profile in profiles:
        worker_id = _worker_profile_id(profile)
        worker = workers.setdefault(
            worker_id,
            {
                'host': str(profile.get('host', 'unknown')),
                'rank': int(profile.get('rank', -1)),
                'pid': int(profile.get('pid', -1)),
                'worker_id': int(profile.get('worker_id', -1)),
                'worker_num': int(profile.get('worker_num', 0)),
                **{key: 0.0 for key in _WORKER_PROFILE_SUM_KEYS},
                **{key: 0.0 for key in _WORKER_PROFILE_LATEST_KEYS},
            },
        )
        for key in _WORKER_PROFILE_SUM_KEYS:
            worker[key] = float(worker.get(key, 0.0)) + float(profile.get(key, 0.0))
        for key in _WORKER_PROFILE_LATEST_KEYS:
            worker[key] = float(profile.get(key, worker.get(key, 0.0)))
    return {'workers': list(workers.values())}


def _merge_worker_profile_window(target: dict[str, Any], updates: dict[str, Any]) -> None:
    target_workers = target.setdefault('workers', {})
    for worker in updates.get('workers', []) or []:
        worker_id = _worker_profile_id(worker)
        bucket = target_workers.setdefault(
            worker_id,
            {
                'host': str(worker.get('host', 'unknown')),
                'rank': int(worker.get('rank', -1)),
                'pid': int(worker.get('pid', -1)),
                'worker_id': int(worker.get('worker_id', -1)),
                'worker_num': int(worker.get('worker_num', 0)),
                **{key: 0.0 for key in _WORKER_PROFILE_SUM_KEYS},
                **{key: 0.0 for key in _WORKER_PROFILE_LATEST_KEYS},
            },
        )
        for key in _WORKER_PROFILE_SUM_KEYS:
            bucket[key] = float(bucket.get(key, 0.0)) + float(worker.get(key, 0.0))
        for key in _WORKER_PROFILE_LATEST_KEYS:
            bucket[key] = float(worker.get(key, bucket.get(key, 0.0)))


class _PipelineProfiledTransform:
    def __init__(self, transform: Any) -> None:
        self.transform = transform

    def __getattr__(self, name: str) -> Any:
        return getattr(self.transform, name)

    def __call__(self, data: Any) -> Any:
        tic = time.perf_counter()
        output = self.transform(data)
        _attach_pipeline_profile(
            output,
            {
                'data_transform': time.perf_counter() - tic,
                'data_transform_calls': 1.0,
            },
        )
        return output


class _PipelineProfiledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        self.dataset = dataset

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: Any) -> Any:
        tic = time.perf_counter()
        data = self.dataset[index]
        total_time = time.perf_counter() - tic
        transform_time = sum(float(profile.get('data_transform', 0.0)) for profile in _collect_pipeline_profiles(data))
        _attach_pipeline_profile(
            data,
            {
                'data_getitem_total': total_time,
                'data_getitem_raw': max(0.0, total_time - transform_time),
                'data_getitem_calls': 1.0,
            },
        )
        return data

    def __getitems__(self, indices: list[Any]) -> list[Any]:
        tic = time.perf_counter()
        data = [self.dataset[index] for index in indices]
        total_time = time.perf_counter() - tic
        transform_time = sum(float(profile.get('data_transform', 0.0)) for profile in _collect_pipeline_profiles(data))
        _attach_pipeline_profile(
            data,
            {
                'data_getitem_total': total_time,
                'data_getitem_raw': max(0.0, total_time - transform_time),
                'data_getitem_calls': 1.0,
            },
        )
        return data

    def set_transform(self, transform: Any) -> None:
        self.dataset.set_transform(transform)


class _PipelineProfiledCollator:
    def __init__(self, collator: Any, *, pipeline_profile_enabled: bool = True, worker_profile_enabled: bool = True) -> None:
        self.collator = collator
        self.pipeline_profile_enabled = bool(pipeline_profile_enabled)
        self.worker_profile_enabled = bool(worker_profile_enabled)

    def __call__(self, batch: Any) -> Any:
        profiles = _pop_pipeline_profiles(batch) if self.pipeline_profile_enabled else []
        worker_profiles = _pop_worker_profiles(batch) if self.worker_profile_enabled else []
        tic = time.perf_counter()
        if _debug_env_enabled('GIGA_DEBUG_TRAIN_BOOT'):
            worker_info = torch.utils.data.get_worker_info()
            worker_id = None if worker_info is None else worker_info.id
            worker_count = None if worker_info is None else worker_info.num_workers
            logging_msg = (
                f'[GIGA_DEBUG_TRAIN_BOOT host={socket.gethostname()} pid={os.getpid()} '
                f'rank={os.environ.get("RANK", "?")} local_rank={os.environ.get("LOCAL_RANK", "?")} '
                f'worker={worker_id}/{worker_count}] collate start batch_len={len(batch) if hasattr(batch, "__len__") else "?"}'
            )
            print(logging_msg, flush=True)
        output = self.collator(batch)
        collate_time = time.perf_counter() - tic
        if _debug_env_enabled('GIGA_DEBUG_TRAIN_BOOT'):
            logging_msg = (
                f'[GIGA_DEBUG_TRAIN_BOOT host={socket.gethostname()} pid={os.getpid()} '
                f'rank={os.environ.get("RANK", "?")} local_rank={os.environ.get("LOCAL_RANK", "?")}] '
                f'collate finish elapsed={collate_time:.3f}s'
            )
            print(logging_msg, flush=True)
        if isinstance(output, dict):
            if self.pipeline_profile_enabled:
                summary = _sum_pipeline_profiles(profiles)
                summary['data_collate'] = collate_time
                summary['data_collate_calls'] = 1.0
                summary['data_profiled_samples'] = float(len(profiles))
                output[_PIPELINE_PROFILE_KEY] = summary
            if worker_profiles:
                output[_PIPELINE_WORKER_PROFILE_KEY] = _summarize_worker_profiles(worker_profiles)
        return output


def _wrap_worker_range_dataset(dataset: Any, worker_range_cfg: Any, sampler: Any, log_fn: Any | None = None) -> Any:
    if not worker_range_cfg:
        return dataset
    if worker_range_cfg is True:
        worker_range_cfg = {}
    elif not isinstance(worker_range_cfg, dict):
        raise TypeError(
            'dataloaders.train.worker_range should be a bool or dict, '
            f'got {type(worker_range_cfg).__name__}'
        )

    from giga_datasets import WorkerRangeDataset

    kwargs = dict(worker_range_cfg)
    mode = kwargs.get('mode', 'whole')
    if 'range_start' not in kwargs and 'range_end' not in kwargs:
        shard_start = getattr(sampler, 'shard_start', None)
        shard_end = getattr(sampler, 'shard_end', None)
        if shard_start is not None and shard_end is not None:
            kwargs['range_start'] = int(shard_start)
            kwargs['range_end'] = int(shard_end)
    if mode == 'per_child' and 'sub_dataset_ranges' not in kwargs:
        sub_dataset_ranges = getattr(sampler, 'sub_dataset_ranges', None)
        if sub_dataset_ranges is not None:
            kwargs['sub_dataset_ranges'] = [(int(start), int(end)) for start, end in sub_dataset_ranges]

    if log_fn is not None:
        range_text = (
            f"[{kwargs['range_start']}, {kwargs['range_end']})"
            if 'range_start' in kwargs and 'range_end' in kwargs
            else 'inferred'
        )
        log_fn(f'Enable WorkerRangeDataset: mode={mode}, range={range_text}')
    return WorkerRangeDataset(dataset=dataset, **kwargs)


def _apply_skip_loss_mask(
    skip_loss_mask: torch.Tensor | None,
    lang_loss_masks: torch.Tensor,
    action_loss_mask: torch.Tensor,
    action_dim_loss_mask: torch.Tensor | None,
    traj_loss_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if skip_loss_mask is None:
        return lang_loss_masks, action_loss_mask, action_dim_loss_mask, traj_loss_mask

    skip_loss_mask = skip_loss_mask.to(device=lang_loss_masks.device, dtype=torch.bool).reshape(-1)
    if not skip_loss_mask.any():
        return lang_loss_masks, action_loss_mask, action_dim_loss_mask, traj_loss_mask

    keep_loss_mask = ~skip_loss_mask
    lang_loss_masks = lang_loss_masks & keep_loss_mask[:, None]
    action_loss_mask = action_loss_mask & keep_loss_mask[:, None]

    if action_dim_loss_mask is not None:
        action_dim_loss_mask = action_dim_loss_mask & keep_loss_mask[:, None, None].to(
            device=action_dim_loss_mask.device
        )

    if traj_loss_mask is not None:
        expand_shape = [skip_loss_mask.shape[0]] + [1] * (traj_loss_mask.ndim - 1)
        traj_keep_loss_mask = keep_loss_mask.reshape(expand_shape).to(device=traj_loss_mask.device)
        traj_loss_mask = traj_loss_mask & traj_keep_loss_mask

    return lang_loss_masks, action_loss_mask, action_dim_loss_mask, traj_loss_mask


def _get_flow_action_horizon(model: Any) -> int | None:
    for candidate in (model, getattr(model, 'module', None), getattr(model, '_orig_mod', None)):
        if candidate is None:
            continue
        flow_action_horizon = getattr(candidate, 'flow_action_horizon', None)
        if flow_action_horizon is not None:
            return int(flow_action_horizon)
        config = getattr(candidate, 'config', None)
        if isinstance(config, dict):
            flow_action_horizon = config.get('flow_action_horizon')
        else:
            flow_action_horizon = getattr(config, 'flow_action_horizon', None)
        if flow_action_horizon is not None:
            return int(flow_action_horizon)
    return None


def _normalize_activation_checkpoint_skip_layer_indices(value: Any) -> dict[str, set[int]]:
    if value is None or value is False:
        return {}
    if not isinstance(value, dict):
        raise TypeError(
            'train.activation_checkpoint_skip_layer_indices must map layer class names '
            'to index lists'
        )

    normalized: dict[str, set[int]] = {}
    for class_name, indices in value.items():
        if indices is None or indices is False:
            continue
        if isinstance(indices, (int, float)):
            raw_indices = [indices]
        elif isinstance(indices, str):
            raw_indices = [part.strip() for part in indices.split(',') if part.strip()]
        elif isinstance(indices, (list, tuple, set)):
            raw_indices = list(indices)
        else:
            raise TypeError(
                'train.activation_checkpoint_skip_layer_indices values must be an int, '
                f'string, or sequence of ints; got {type(indices).__name__} for {class_name!r}'
            )

        idx_set: set[int] = set()
        for idx in raw_indices:
            idx_int = int(idx)
            if idx_int < 0:
                raise ValueError('activation checkpoint skip layer indices must be >= 0')
            idx_set.add(idx_int)
        if idx_set:
            normalized[str(class_name)] = idx_set
    return normalized


def _module_auto_wrap_policy_with_layer_skips(
    module_names: list[str],
    skip_layer_indices: dict[str, set[int]],
    sep: str = '__##__',
):
    module_info_dict = {}
    for module_name in module_names:
        parts = module_name.split(sep)
        if len(parts) == 1:
            module_info_dict[parts[0]] = dict(total=-1, count=0)
        elif len(parts) == 2:
            if not parts[1].isdigit():
                raise ValueError(f'Invalid activation checkpoint rule: {module_name!r}')
            module_info_dict[parts[0]] = dict(total=int(parts[1]), count=0)
        else:
            raise ValueError(f'Invalid activation checkpoint rule: {module_name!r}')

    def _wrap(module: torch.nn.Module, recurse: bool, nonwrapped_numel: int) -> bool:
        if recurse:
            return True
        module_name = module.__class__.__name__
        module_info = module_info_dict.get(module_name)
        if module_info is None:
            return False

        if module_info['total'] >= 0:
            module_info['count'] += 1
            if module_info['count'] > module_info['total']:
                return False

        layer_idx = getattr(module, 'layer_idx', None)
        if layer_idx is not None and int(layer_idx) in skip_layer_indices.get(module_name, set()):
            return False
        return True

    return _wrap


def _activation_checkpoint_impl_from_kwargs(kwargs: dict[str, Any]) -> CheckpointImpl | None:
    value = kwargs.get('activation_checkpoint_impl', None)
    if value is None and 'activation_checkpoint_reentrant' in kwargs:
        return CheckpointImpl.REENTRANT if _str_to_bool(kwargs.get('activation_checkpoint_reentrant')) else CheckpointImpl.NO_REENTRANT
    if value is None:
        return None
    if isinstance(value, CheckpointImpl):
        return value

    value_str = str(value).strip().lower().replace('-', '_')
    if value_str in {'reentrant', 'true', '1', 'yes'}:
        return CheckpointImpl.REENTRANT
    if value_str in {'no_reentrant', 'non_reentrant', 'false', '0', 'no'}:
        return CheckpointImpl.NO_REENTRANT
    raise ValueError(f'Unsupported activation_checkpoint_impl={value!r}')


class GigaBrain07Trainer(Trainer):
    def _debug_train_boot(self, message: str, *args: Any, **kwargs: Any) -> None:
        if not _debug_env_enabled('GIGA_DEBUG_TRAIN_BOOT'):
            return

        rank = getattr(self, 'process_index', '?')
        local_rank = getattr(self, 'local_process_index', '?')
        prefix = f'[GIGA_DEBUG_TRAIN_BOOT host={socket.gethostname()} pid={os.getpid()} rank={rank} local_rank={local_rank}] '
        logger = getattr(self, 'logger', None)
        if logger is not None:
            logger.info(prefix + message, *args, **kwargs)
        else:
            if args:
                message = message % args
            print(prefix + message, flush=True)

    def _pipeline_profile_cfg(self) -> dict[str, Any]:
        cfg = self.kwargs.get('pipeline_profile', {})
        if cfg is True:
            cfg = {}
        elif cfg is False or cfg is None:
            env_enabled = _str_to_bool(os.environ.get('GIGA_PIPELINE_PROFILE'), False)
            if not env_enabled:
                return {}
            cfg = {}
        elif not isinstance(cfg, dict):
            raise TypeError(f'train.pipeline_profile should be a bool or dict, got {type(cfg).__name__}')

        cfg = dict(cfg)
        cfg['enabled'] = _str_to_bool(cfg.get('enabled', True), True)
        if not cfg['enabled']:
            return {}
        cfg['epoch_interval'] = int(cfg.get('epoch_interval', os.environ.get('GIGA_PIPELINE_PROFILE_EPOCH_INTERVAL', 10)))
        cfg['step_interval'] = int(cfg.get('step_interval', os.environ.get('GIGA_PIPELINE_PROFILE_STEP_INTERVAL', 0)))
        cfg['sync_cuda'] = _str_to_bool(cfg.get('sync_cuda', os.environ.get('GIGA_PIPELINE_PROFILE_SYNC_CUDA')), False)
        cfg['include_step_interval'] = _str_to_bool(cfg.get('include_step_interval', True), True)
        return cfg

    def _pipeline_profile_enabled(self) -> bool:
        return bool(self._pipeline_profile_cfg())

    def apply_activation_checkpointing(self, models: list[torch.nn.Module] | None = None) -> None:
        offload_class_names = self.kwargs.get('activation_offload_class_names', None)
        offload_skip_layer_indices = _normalize_activation_checkpoint_skip_layer_indices(
            self.kwargs.get('activation_offload_skip_layer_indices', None)
        )
        skip_layer_indices = _normalize_activation_checkpoint_skip_layer_indices(
            self.kwargs.get('activation_checkpoint_skip_layer_indices', None)
        )
        checkpoint_impl = _activation_checkpoint_impl_from_kwargs(self.kwargs)
        if offload_class_names is None and not skip_layer_indices and checkpoint_impl is None:
            super().apply_activation_checkpointing(models)
            return

        models = models or self.models
        if offload_class_names is not None:
            for model in models:
                offload_policy = _module_auto_wrap_policy_with_layer_skips(
                    offload_class_names,
                    offload_skip_layer_indices,
                )
                torch_apply_activation_checkpointing(
                    model,
                    checkpoint_wrapper_fn=offload_wrapper,
                    auto_wrap_policy=offload_policy,
                )

        if not self.activation_checkpointing or self.activation_class_names is None:
            return

        for model in models:
            auto_wrap_policy = _module_auto_wrap_policy_with_layer_skips(
                self.activation_class_names,
                skip_layer_indices,
            )
            checkpoint_wrapper_kwargs = {}
            if checkpoint_impl is not None:
                checkpoint_wrapper_kwargs['checkpoint_impl'] = checkpoint_impl
            if self.mixed_precision == 'fp8' and self.accelerator.fp8_backend == FP8BackendType.TE:
                from transformer_engine.pytorch.distributed import checkpoint as te_checkpoint

                checkpoint_wrapper_fn = functools.partial(
                    checkpoint_wrapper,
                    checkpoint_fn=te_checkpoint,
                    **checkpoint_wrapper_kwargs,
                )
                torch_apply_activation_checkpointing(
                    model,
                    checkpoint_wrapper_fn=checkpoint_wrapper_fn,
                    auto_wrap_policy=auto_wrap_policy,
                )
            else:
                checkpoint_wrapper_fn = (
                    functools.partial(checkpoint_wrapper, **checkpoint_wrapper_kwargs)
                    if checkpoint_wrapper_kwargs
                    else checkpoint_wrapper
                )
                torch_apply_activation_checkpointing(
                    model,
                    checkpoint_wrapper_fn=checkpoint_wrapper_fn,
                    auto_wrap_policy=auto_wrap_policy,
                )

    def _pipeline_profile_sync(self) -> None:
        cfg = getattr(self, '_active_pipeline_profile_cfg', None) or self._pipeline_profile_cfg()
        if bool(cfg.get('sync_cuda', False)) and torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextmanager
    def _pipeline_profile_stage(self, name: str):
        if not getattr(self, '_active_pipeline_profile_enabled', False):
            yield
            return
        self._pipeline_profile_sync()
        tic = time.perf_counter()
        try:
            yield
        finally:
            self._pipeline_profile_sync()
            self._pipeline_profile_add(name, time.perf_counter() - tic)

    def _pipeline_profile_add(self, key: str, value: float) -> None:
        if not getattr(self, '_active_pipeline_profile_enabled', False):
            return
        profile = getattr(self, '_pipeline_profile_window', None)
        if profile is None:
            return
        profile[key] = float(profile.get(key, 0.0)) + float(value)

    def _pipeline_profile_add_many(self, values: dict[str, Any]) -> None:
        if not getattr(self, '_active_pipeline_profile_enabled', False):
            return
        for key, value in values.items():
            self._pipeline_profile_add(key, float(value))

    def _pipeline_profile_pop_batch(self, batch_dict: Any) -> dict[str, float]:
        if not isinstance(batch_dict, dict):
            return {}
        profile = batch_dict.pop(_PIPELINE_PROFILE_KEY, None)
        if isinstance(profile, dict):
            return {key: float(value) for key, value in profile.items() if isinstance(value, (int, float))}
        return {}

    def _pipeline_worker_profile_pop_batch(self, batch_dict: Any) -> dict[str, Any]:
        if not isinstance(batch_dict, dict):
            return {}
        profile = batch_dict.pop(_PIPELINE_WORKER_PROFILE_KEY, None)
        return profile if isinstance(profile, dict) else {}

    def _worker_profile_enabled(self) -> bool:
        cfg = self.kwargs.get('worker_profile', os.environ.get('GIGA_WORKER_PROFILE', True))
        if isinstance(cfg, dict):
            cfg = cfg.get('enabled', True)
        return _str_to_bool(cfg, True)

    def _lerobot_open_cache_worker_profile_enabled(self, cache_cfg: Any) -> bool:
        if not self._worker_profile_enabled():
            return False
        if not isinstance(cache_cfg, dict):
            return False
        open_cache_cfg = cache_cfg.get('lerobot_open_cache', True)
        return open_cache_cfg is not False and open_cache_cfg is not None

    def _worker_profile_reset_window(self) -> None:
        self._pipeline_worker_profile_window = {'workers': {}}

    def _worker_profile_add_many(self, values: dict[str, Any]) -> None:
        if not values:
            return
        profile = getattr(self, '_pipeline_worker_profile_window', None)
        if profile is None:
            return
        _merge_worker_profile_window(profile, values)

    def _pipeline_profile_should_emit(self, epoch_completed: bool) -> bool:
        cfg = getattr(self, '_active_pipeline_profile_cfg', {}) or {}
        epoch_interval = int(cfg.get('epoch_interval', 0))
        if epoch_completed and epoch_interval > 0 and self.cur_epoch > 0 and self.cur_epoch % epoch_interval == 0:
            return True

        if bool(cfg.get('include_step_interval', True)):
            step_interval = int(cfg.get('step_interval', 0))
            if step_interval > 0 and self.cur_step % step_interval == 0:
                return True
        return self.cur_step == self.max_steps

    def _pipeline_profile_reset_window(self) -> None:
        self._pipeline_profile_window = {key: 0.0 for key in (*_PIPELINE_PROFILE_TIME_KEYS, *_PIPELINE_PROFILE_COUNT_KEYS)}
        self._pipeline_profile_window_start_step = int(self.cur_step)
        self._pipeline_profile_window_start_epoch = int(self.cur_epoch)

    def _pipeline_profile_gather_window(self) -> dict[str, dict[str, float]] | None:
        local = getattr(self, '_pipeline_profile_window', None)
        if not local:
            return None

        keys = list(_PIPELINE_PROFILE_TIME_KEYS) + list(_PIPELINE_PROFILE_COUNT_KEYS)
        tensor = torch.tensor([float(local.get(key, 0.0)) for key in keys], device=self.device, dtype=torch.float64)
        if self.num_processes > 1:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rows = [tensor.clone() for _ in range(self.num_processes)]
                torch.distributed.all_gather(rows, tensor)
                gathered = torch.stack(rows)
            else:
                gathered = self.accelerator.gather(tensor).reshape(-1, len(keys))
        else:
            gathered = tensor.reshape(1, -1)

        if not self.is_main_process:
            return None

        stats: dict[str, dict[str, float]] = {}
        for idx, key in enumerate(keys):
            values = gathered[:, idx]
            stats[key] = {
                'max': float(values.max().item()),
                'mean': float(values.mean().item()),
                'min': float(values.min().item()),
            }
        return stats

    def _pipeline_worker_profile_gather_window(self) -> dict[str, dict[str, float]] | None:
        local = getattr(self, '_pipeline_worker_profile_window', None) or {'workers': {}}
        local_workers = list((local.get('workers') or {}).values())

        if self.num_processes > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: list[Any] = [None] * self.num_processes
            torch.distributed.all_gather_object(gathered, local_workers)
        else:
            gathered = [local_workers]

        if not self.is_main_process:
            return None

        by_worker: dict[tuple[str, int, int, int], dict[str, Any]] = {}
        for worker_list in gathered:
            if not worker_list:
                continue
            for worker in worker_list:
                if not isinstance(worker, dict):
                    continue
                worker_id = _worker_profile_id(worker)
                bucket = by_worker.setdefault(
                    worker_id,
                    {
                        'host': str(worker.get('host', 'unknown')),
                        'rank': int(worker.get('rank', -1)),
                        'pid': int(worker.get('pid', -1)),
                        'worker_id': int(worker.get('worker_id', -1)),
                        **{key: 0.0 for key in _WORKER_PROFILE_SUM_KEYS},
                        **{key: 0.0 for key in _WORKER_PROFILE_LATEST_KEYS},
                    },
                )
                for key in _WORKER_PROFILE_SUM_KEYS:
                    bucket[key] = float(bucket.get(key, 0.0)) + float(worker.get(key, 0.0))
                for key in _WORKER_PROFILE_LATEST_KEYS:
                    bucket[key] = float(worker.get(key, bucket.get(key, 0.0)))

        by_node: dict[str, list[dict[str, Any]]] = {}
        for worker in by_worker.values():
            by_node.setdefault(str(worker.get('host', 'unknown')), []).append(worker)

        node_stats: dict[str, dict[str, float]] = {}
        for node in sorted(by_node):
            workers = by_node[node]
            worker_count = max(len(workers), 1)
            open_cache_size_avg = sum(float(worker.get('open_cache_size', 0.0)) for worker in workers) / worker_count
            max_open_per_worker_avg = (
                sum(float(worker.get('max_open_per_worker', 0.0)) for worker in workers) / worker_count
            )
            hit_rate_sum = 0.0
            touch_per_getitem_sum = 0.0
            miss_per_getitem_sum = 0.0
            eviction_per_getitem_sum = 0.0
            gc_per_getitem_sum = 0.0
            for worker in workers:
                getitem_count = float(worker.get('getitem_count', 0.0))
                touch_count = float(worker.get('touch_count', 0.0))
                hit_count = float(worker.get('hit_count', 0.0))
                miss_count = float(worker.get('miss_count', 0.0))
                eviction_count = float(worker.get('eviction_count', 0.0))
                gc_count = float(worker.get('gc_count', 0.0))
                touch_denom = max(touch_count, 1.0)
                getitem_denom = max(getitem_count, 1.0)
                hit_rate_sum += hit_count / touch_denom
                touch_per_getitem_sum += touch_count / getitem_denom
                miss_per_getitem_sum += miss_count / getitem_denom
                eviction_per_getitem_sum += eviction_count / getitem_denom
                gc_per_getitem_sum += gc_count / getitem_denom
            node_stats[node] = {
                'worker_count': float(worker_count),
                'open_cache_size_avg': open_cache_size_avg,
                'max_open_per_worker_avg': max_open_per_worker_avg,
                'hit_rate_avg': hit_rate_sum / worker_count,
                'touch_per_getitem_avg': touch_per_getitem_sum / worker_count,
                'miss_per_getitem_avg': miss_per_getitem_sum / worker_count,
                'eviction_per_getitem_avg': eviction_per_getitem_sum / worker_count,
                'gc_per_getitem_avg': gc_per_getitem_sum / worker_count,
            }
        return node_stats

    def _worker_profile_outputs_and_message(
        self,
        *,
        metric_prefix: str = 'worker_profile',
        message_prefix: str = 'Worker profile',
    ) -> tuple[dict[str, float], str | None]:
        worker_profile_by_node = self._pipeline_worker_profile_gather_window()
        if not worker_profile_by_node:
            self._worker_profile_reset_window()
            return {}, None

        outputs: dict[str, float] = {}
        msg_parts = [message_prefix]
        for node, node_stats in worker_profile_by_node.items():
            node_metric_prefix = f'{metric_prefix}/{node}'
            for key, value in node_stats.items():
                outputs[f'{node_metric_prefix}/{key}'] = float(value)
            msg_parts.append(
                f'{node}: workers={int(node_stats["worker_count"])}, '
                f'open_cache={node_stats["open_cache_size_avg"]:.2f}/'
                f'{node_stats["max_open_per_worker_avg"]:.2f}, '
                f'hit_rate_avg={node_stats["hit_rate_avg"]:.3f}, '
                f'miss_per_getitem_avg={node_stats["miss_per_getitem_avg"]:.3f}, '
                f'evict_per_getitem_avg={node_stats["eviction_per_getitem_avg"]:.3f}, '
                f'gc_per_getitem_avg={node_stats["gc_per_getitem_avg"]:.5f}'
            )
        self._worker_profile_reset_window()
        return outputs, ', '.join(msg_parts)

    def _pipeline_profile_log_window(self) -> None:
        stats = self._pipeline_profile_gather_window()
        if not self.is_main_process or stats is None:
            self._pipeline_profile_reset_window()
            return

        steps = max(stats['steps']['max'], 1.0)
        total = max(stats['step_total']['max'], 1e-12)
        outputs: dict[str, float] = {}
        msg_parts = [
            (
                'Pipeline profile '
                f'window_step[{self._pipeline_profile_window_start_step + 1}-{self.cur_step}] '
                f'epoch[{self._pipeline_profile_window_start_epoch}-{self.cur_epoch}], '
                f'step_total: {total / steps:.4f}s/step, '
                f'samples_s: {self.batch_size / max(total / steps, 1e-12):.3f}'
            )
        ]

        for key in _PIPELINE_PROFILE_TIME_KEYS:
            key_stats = stats[key]
            per_step = key_stats['max'] / steps
            pct = key_stats['max'] / total * 100.0
            outputs[f'pipeline_time/{key}_max_s_per_step'] = per_step
            outputs[f'pipeline_time/{key}_mean_s_per_step'] = key_stats['mean'] / steps
            outputs[f'pipeline_time/{key}_pct'] = pct
            if key != 'step_total':
                msg_parts.append(f'{key}: {per_step:.4f}s/step {pct:.1f}%')

        profiled_samples = max(stats['data_profiled_samples']['max'], 1.0)
        collate_calls = max(stats['data_collate_calls']['max'], 1.0)
        for key in _PIPELINE_PROFILE_SAMPLE_KEYS:
            outputs[f'pipeline_data/{key}_max_s_per_sample'] = stats[key]['max'] / profiled_samples
        for key in _PIPELINE_PROFILE_BATCH_KEYS:
            outputs[f'pipeline_data/{key}_max_s_per_batch'] = stats[key]['max'] / collate_calls

        outputs['pipeline_count/steps'] = stats['steps']['max']
        outputs['pipeline_count/micro_steps'] = stats['micro_steps']['max']
        outputs['pipeline_count/data_profiled_samples'] = stats['data_profiled_samples']['max']
        self.accelerator.log(outputs, self.cur_step)
        self.logger.info(', '.join(msg_parts))
        self._pipeline_profile_reset_window()

    def prepare(self, dataloaders: Any, models: Any, optimizers: Any, schedulers: Any) -> None:
        _sync_state_input_config(dataloaders, models)
        # Stash the transform config so save_model_hook can dump inference_config.json
        # alongside the diffusers config.json on every checkpoint.
        self._train_transform_cfg = dict(dataloaders.get('transform', {})) if dataloaders else {}
        tic = time.perf_counter()
        self._debug_train_boot('prepare start')
        super().prepare(dataloaders, models, optimizers, schedulers)
        self._debug_train_boot('prepare finish elapsed=%.3fs', time.perf_counter() - tic)

    def save_model_hook(self, models, weights, output_dir: str) -> None:
        super().save_model_hook(models, weights, output_dir)
        if not self.is_main_process:
            return
        cfg = getattr(self, '_train_transform_cfg', None)
        if not cfg:
            return
        inference_cfg = {k: cfg[k] for k in _INFERENCE_CONFIG_KEYS if k in cfg}
        if not inference_cfg:
            return
        # Write into every subdir that holds a diffusers config.json (model/, model_ema/).
        for name in os.listdir(output_dir):
            sub = os.path.join(output_dir, name)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, 'config.json')):
                with open(os.path.join(sub, 'inference_config.json'), 'w') as f:
                    json.dump(inference_cfg, f, indent=2, default=lambda o: dict(o) if isinstance(o, dict) else str(o))

    def get_dataloaders(self, data_config: Any):
        from giga_datasets import DefaultCollator, load_dataset

        total_tic = time.perf_counter()
        self._debug_train_boot('get_dataloaders start')
        batch_size_per_gpu = data_config.get('batch_size_per_gpu', 1)
        batch_size = batch_size_per_gpu * self.num_processes * self.gradient_accumulation_steps
        self._debug_train_boot(
            'get_dataloaders config batch_size_per_gpu=%s global_batch_size=%s num_workers=%s cache=%s sampler=%s',
            batch_size_per_gpu,
            batch_size,
            data_config.get('num_workers', None),
            data_config.get('cache', None),
            data_config.get('sampler', None),
        )
        tic = time.perf_counter()
        dataset = load_dataset(data_config.data_or_config)
        self._debug_train_boot(
            'load_dataset finish elapsed=%.3fs dataset=%s len=%s',
            time.perf_counter() - tic,
            dataset.__class__.__name__,
            len(dataset),
        )

        filter_cfg = data_config.get('filter', None)
        if filter_cfg is not None:
            tic = time.perf_counter()
            self._debug_train_boot('dataset.filter start cfg=%s', filter_cfg)
            dataset.filter(**filter_cfg)
            self._debug_train_boot('dataset.filter finish elapsed=%.3fs len=%s', time.perf_counter() - tic, len(dataset))
        tic = time.perf_counter()
        self._debug_train_boot('HF Arrow cache warmup/wait start')
        self._warmup_hf_arrow_cache(
            dataset,
            data_config.get('cache', None),
            data_config=data_config,
            batch_size=batch_size,
            batch_size_per_gpu=batch_size_per_gpu,
        )
        self._debug_train_boot('HF Arrow cache warmup/wait finish elapsed=%.3fs', time.perf_counter() - tic)
        tic = time.perf_counter()
        self._debug_train_boot('build_transform start')
        transform = build_transform(data_config.transform)
        self._debug_train_boot(
            'build_transform finish elapsed=%.3fs transform=%s',
            time.perf_counter() - tic,
            transform.__class__.__name__ if transform is not None else None,
        )
        if self._pipeline_profile_enabled() and transform is not None:
            transform = _PipelineProfiledTransform(transform)
        tic = time.perf_counter()
        dataset.set_transform(transform)
        self._debug_train_boot('dataset.set_transform finish elapsed=%.3fs', time.perf_counter() - tic)
        if 'batch_sampler' in data_config:
            batch_sampler_cfg = data_config.batch_sampler
            tic = time.perf_counter()
            self._debug_train_boot('build batch_sampler start cfg=%s', batch_sampler_cfg)
            batch_sampler = build_sampler(
                batch_sampler_cfg,
                dataset=dataset,
                batch_size_per_gpu=batch_size_per_gpu,
                batch_size=batch_size,
            )
            sampler = getattr(batch_sampler, 'sampler', None)
            self._debug_train_boot(
                'build batch_sampler finish elapsed=%.3fs batch_sampler=%s sampler=%s',
                time.perf_counter() - tic,
                batch_sampler.__class__.__name__,
                sampler.__class__.__name__ if sampler is not None else None,
            )
        else:
            sampler_cfg = data_config.get('sampler', {'type': 'DefaultSampler'})
            sampler_kwargs = self._build_train_sampler_kwargs(
                sampler_cfg,
                dataset=dataset,
                batch_size=batch_size,
                batch_size_per_gpu=batch_size_per_gpu,
            )
            tic = time.perf_counter()
            self._debug_train_boot('build sampler start cfg=%s kwargs_keys=%s', sampler_cfg, list(sampler_kwargs))
            sampler = build_sampler(sampler_cfg, **sampler_kwargs)
            self._debug_train_boot(
                'build sampler finish elapsed=%.3fs sampler=%s len=%s shard=%s/%s range=[%s,%s)',
                time.perf_counter() - tic,
                sampler.__class__.__name__,
                len(sampler) if hasattr(sampler, '__len__') else '?',
                getattr(sampler, 'shard_rank', '?'),
                getattr(sampler, 'shard_world_size', '?'),
                getattr(sampler, 'shard_start', '?'),
                getattr(sampler, 'shard_end', '?'),
            )
            batch_sampler = BatchSampler(sampler, batch_size=batch_size_per_gpu, drop_last=False)
        if self.data_parallel_size > 1:
            batch_sampler = ParallelBatchSampler(batch_sampler, data_parallel_size=self.data_parallel_size)
        tic = time.perf_counter()
        self._debug_train_boot('build collator start')
        collator = data_config.get('collator', {})
        collator = DefaultCollator(**collator)
        pipeline_profile_enabled = self._pipeline_profile_enabled()
        worker_profile_enabled = self._lerobot_open_cache_worker_profile_enabled(data_config.get('cache', None))
        self._active_worker_profile_enabled = worker_profile_enabled
        if pipeline_profile_enabled or worker_profile_enabled:
            collator = _PipelineProfiledCollator(
                collator,
                pipeline_profile_enabled=pipeline_profile_enabled,
                worker_profile_enabled=worker_profile_enabled,
            )
        self._debug_train_boot('build collator finish elapsed=%.3fs collator=%s', time.perf_counter() - tic, collator.__class__.__name__)
        tic = time.perf_counter()
        self._debug_train_boot('wrap dataset_for_loader start')
        dataset_for_loader = wrap_materialized_shard_reader(
            dataset,
            data_config.get('materialized_shard_reader', None),
            log_fn=self.print,
            profile_enabled=pipeline_profile_enabled,
        )
        dataset_for_loader = wrap_lerobot_open_cache(
            dataset_for_loader,
            data_config.get('cache', None),
            log_fn=self.print,
            profile_enabled=worker_profile_enabled,
        )
        dataset_for_loader = _wrap_worker_range_dataset(
            dataset_for_loader,
            data_config.get('worker_range', None),
            sampler,
            log_fn=self.print,
        )
        if pipeline_profile_enabled:
            dataset_for_loader = _PipelineProfiledDataset(dataset_for_loader)
        self._debug_train_boot(
            'wrap dataset_for_loader finish elapsed=%.3fs dataset_for_loader=%s len=%s',
            time.perf_counter() - tic,
            dataset_for_loader.__class__.__name__,
            len(dataset_for_loader),
        )
        num_workers = int(data_config.num_workers)
        dataloader_kwargs = dict(
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=bool(data_config.get('pin_memory', True)),
        )
        if num_workers > 0:
            dataloader_kwargs.update(
                persistent_workers=bool(data_config.get('persistent_workers', True)),
                prefetch_factor=int(data_config.get('prefetch_factor', 4)),
            )
        self._debug_train_boot('DataLoader construct start kwargs=%s', dataloader_kwargs)
        tic = time.perf_counter()
        dataloader = torch.utils.data.DataLoader(
            dataset_for_loader,
            **dataloader_kwargs,
        )
        self._debug_train_boot(
            'DataLoader construct finish elapsed=%.3fs len=%s',
            time.perf_counter() - tic,
            len(dataloader),
        )
        if self.distributed_type == DistributedType.DEEPSPEED:
            if getattr(batch_sampler, 'batch_size', None) is not None:
                batch_size = batch_sampler.batch_size
            elif getattr(batch_sampler, 'batch_sizes', None) is not None:
                batch_size = min(batch_sampler.batch_sizes)
            else:
                assert False
            self.accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = batch_size
        if getattr(batch_sampler, 'batch_size', None) is None:
            self.accelerator.even_batches = False
        self._debug_train_boot('get_dataloaders finish elapsed=%.3fs', time.perf_counter() - total_tic)
        return dataloader

    def _build_train_sampler_kwargs(
        self,
        sampler_cfg: Any,
        *,
        dataset: Any,
        batch_size: int,
        batch_size_per_gpu: int,
    ) -> dict[str, Any]:
        sampler_kwargs = dict(dataset=dataset, batch_size=batch_size)
        if isinstance(sampler_cfg, dict) and sampler_cfg.get('type') == 'ShardedListWeightedSampler':
            inferred_process_shard_size = int(os.environ.get('NPROC_PER_NODE', os.environ.get('LOCAL_WORLD_SIZE', self.num_processes)))
            extra_sampler_kwargs = dict(
                num_processes=self.num_processes,
                process_index=self.process_index,
                process_shard_size=inferred_process_shard_size,
                process_batch_size=batch_size_per_gpu,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
            )
            sampler_kwargs.update(
                {key: value for key, value in extra_sampler_kwargs.items() if key not in sampler_cfg}
            )
        return sampler_kwargs

    def _warmup_hf_arrow_cache(
        self,
        dataset: Any,
        cache_cfg: Any,
        *,
        data_config: Any | None = None,
        batch_size: int | None = None,
        batch_size_per_gpu: int | None = None,
    ) -> None:
        if not cache_cfg:
            return
        if cache_cfg is True:
            cache_cfg = {}
        elif isinstance(cache_cfg, str):
            cache_cfg = {'type': cache_cfg}
        elif not isinstance(cache_cfg, dict):
            raise TypeError(f'dataloaders.train.cache should be a bool, str, or dict, got {type(cache_cfg).__name__}')

        cache_type = cache_cfg.get('type', 'hf_arrow')
        if cache_type != 'hf_arrow':
            raise ValueError(f'Unsupported dataloaders.train.cache.type={cache_type!r}; only "hf_arrow" is supported')

        process_scope = cache_cfg.get('process_scope', 'main')
        if process_scope in {'main', 'shared'}:
            should_warmup = self.is_main_process
        elif process_scope == 'local_main':
            should_warmup = self.is_local_main_process
        elif process_scope == 'all':
            should_warmup = True
        else:
            raise ValueError(
                'Unsupported dataloaders.train.cache.process_scope='
                f'{process_scope!r}; expected "shared", "main", "local_main", or "all"'
            )

        marker_path = self._hf_arrow_cache_marker_path(cache_cfg)
        warmup_mode = cache_cfg.get('warmup_mode', 'full')
        if warmup_mode not in {'full', 'sampler_shard'}:
            raise ValueError(f'Unsupported dataloaders.train.cache.warmup_mode={warmup_mode!r}')
        self._debug_train_boot(
            'HF Arrow cache config type=%s process_scope=%s should_warmup=%s warmup_mode=%s marker_path=%s',
            cache_type,
            process_scope,
            should_warmup,
            warmup_mode,
            marker_path,
        )
        sampler_shard_plan = None
        if warmup_mode == 'sampler_shard':
            if data_config is None or batch_size is None or batch_size_per_gpu is None:
                raise ValueError('sampler_shard HF Arrow cache warmup requires data_config and batch size context')
            tic = time.perf_counter()
            self._debug_train_boot('prepare sampler_shard warmup plan start')
            sampler_shard_plan = self._prepare_hf_arrow_cache_sampler_shard(
                dataset,
                data_config=data_config,
                cache_cfg=cache_cfg,
                batch_size=batch_size,
                batch_size_per_gpu=batch_size_per_gpu,
            )
            self._debug_train_boot(
                'prepare sampler_shard warmup plan finish elapsed=%.3fs warmup_items=%s skipped_non_lerobot=%s fallback_full=%s shard=%s/%s range=[%s,%s)',
                time.perf_counter() - tic,
                len(sampler_shard_plan['warmup_items']),
                sampler_shard_plan['skipped_non_lerobot'],
                sampler_shard_plan['fallback_full'],
                sampler_shard_plan['shard_rank'],
                sampler_shard_plan['shard_world_size'],
                sampler_shard_plan['shard_start'],
                sampler_shard_plan['shard_end'],
            )

        if should_warmup:
            log_interval = int(cache_cfg.get('log_interval', 10))
            release_after_open = bool(cache_cfg.get('release_after_open', True))
            start = time.time()
            if warmup_mode == 'full':
                total = self._warmup_hf_arrow_cache_full(
                    dataset,
                    log_interval=log_interval,
                    release_after_open=release_after_open,
                    process_scope=process_scope,
                )
            elif warmup_mode == 'sampler_shard':
                total = self._warmup_hf_arrow_cache_sampler_shard(
                    sampler_shard_plan,
                    log_interval=log_interval,
                    release_after_open=release_after_open,
                    process_scope=process_scope,
                )
            if release_after_open:
                gc.collect()
            self._write_hf_arrow_cache_marker(marker_path, total)
            self.print(f'HF Arrow cache warmup finished in {time.time() - start:.1f}s')
            self._debug_train_boot('HF Arrow cache marker written path=%s total=%s', marker_path, total)
        else:
            self._debug_train_boot('wait HF Arrow cache marker start path=%s', marker_path)
            self._wait_hf_arrow_cache_marker(marker_path, cache_cfg)
            self._debug_train_boot('wait HF Arrow cache marker finish path=%s', marker_path)
        self._debug_train_boot('accelerator.wait_for_everyone after cache start')
        self.accelerator.wait_for_everyone()
        self._debug_train_boot('accelerator.wait_for_everyone after cache finish')

    def _warmup_hf_arrow_cache_full(
        self,
        dataset: Any,
        *,
        log_interval: int,
        release_after_open: bool,
        process_scope: str,
    ) -> int:
        datasets = list(self._iter_lerobot_datasets(dataset))
        total = len(datasets)
        self.print(f'Warm up HF Arrow cache for {total} LeRobot datasets (process_scope={process_scope}, warmup_mode=full)')
        for i, child in enumerate(datasets, start=1):
            data_path = getattr(child, 'data_path', '<unknown>')
            tic = time.time()
            try:
                self._warmup_lerobot_child_hf_arrow_cache(child, release_after_open=release_after_open)
            except Exception as error:
                raise RuntimeError(f'Failed to warm up HF Arrow cache for LeRobotDataset: data_path={data_path}') from error
            if log_interval > 0 and (i == 1 or i == total or i % log_interval == 0):
                self.print(f'HF Arrow cache warmup [{i}/{total}] {time.time() - tic:.1f}s {data_path}')
        return total

    def _prepare_hf_arrow_cache_sampler_shard(
        self,
        dataset: Any,
        *,
        data_config: Any,
        cache_cfg: dict[str, Any],
        batch_size: int,
        batch_size_per_gpu: int,
    ) -> dict[str, Any]:
        if 'batch_sampler' in data_config:
            raise ValueError('sampler_shard HF Arrow cache warmup does not support dataloaders.train.batch_sampler')

        sampler_cfg = data_config.get('sampler', {'type': 'DefaultSampler'})
        if not isinstance(sampler_cfg, dict) or sampler_cfg.get('type') != 'ShardedListWeightedSampler':
            raise ValueError('sampler_shard HF Arrow cache warmup requires sampler.type="ShardedListWeightedSampler"')

        sampler = build_sampler(
            sampler_cfg,
            **self._build_train_sampler_kwargs(
                sampler_cfg,
                dataset=dataset,
                batch_size=batch_size,
                batch_size_per_gpu=batch_size_per_gpu,
            ),
        )
        sub_dataset_ranges = getattr(sampler, 'sub_dataset_ranges', None)
        if sub_dataset_ranges is None:
            raise ValueError('ShardedListWeightedSampler should expose sub_dataset_ranges for sampler_shard warmup')

        warmup_items, skipped_non_lerobot, fallback_full = self._prepare_hf_arrow_cache_sampler_shard_ranges(
            dataset,
            sub_dataset_ranges,
            cache_cfg=cache_cfg,
        )
        return dict(
            warmup_items=warmup_items,
            skipped_non_lerobot=skipped_non_lerobot,
            fallback_full=fallback_full,
            shard_rank=getattr(sampler, 'shard_rank', '?'),
            shard_world_size=getattr(sampler, 'shard_world_size', '?'),
            shard_start=getattr(sampler, 'shard_start', '?'),
            shard_end=getattr(sampler, 'shard_end', '?'),
        )

    def _prepare_hf_arrow_cache_sampler_shard_ranges(
        self,
        dataset: Any,
        sub_dataset_ranges: list[tuple[int, int]],
        *,
        cache_cfg: dict[str, Any],
    ) -> tuple[list[tuple[Any, int, int, str]], int, int]:
        fallback = str(cache_cfg.get('sampler_shard_fallback', 'full')).lower()
        if fallback not in {'full', 'skip'}:
            raise ValueError('dataloaders.train.cache.sampler_shard_fallback should be "full" or "skip"')

        top_children = getattr(dataset, 'datasets', None)
        if top_children is None:
            raise TypeError('sampler_shard HF Arrow cache warmup requires a concat-style dataset with child datasets')
        if len(sub_dataset_ranges) != len(top_children):
            raise ValueError(
                'sampler shard range count should match top-level dataset count, '
                f'got {len(sub_dataset_ranges)} and {len(top_children)}'
            )

        warmup_items: list[tuple[Any, int, int, str]] = []
        skipped_non_lerobot = 0
        fallback_full = 0
        for top_child, (range_start, range_end) in zip(top_children, sub_dataset_ranges):
            if range_end <= range_start:
                continue
            for child, child_start, child_end in self._iter_dataset_range_leaves(top_child, int(range_start), int(range_end)):
                if child.__class__.__name__ != 'LeRobotDataset':
                    skipped_non_lerobot += 1
                    continue
                data_path = getattr(child, 'data_path', '<unknown>')
                child_len = len(child)
                if child_start == 0 and child_end == child_len:
                    if hasattr(child, 'clear_hf_cache_frame_range'):
                        child.clear_hf_cache_frame_range()
                    warmup_items.append((child, 0, child_len, 'full'))
                    continue
                try:
                    child.set_hf_cache_frame_range(child_start, child_end)
                except Exception as error:
                    if fallback == 'skip':
                        self.print(
                            'Skip shard HF Arrow cache warmup because frame range could not be mapped: '
                            f'data_path={data_path}, range=[{child_start}, {child_end}), error={error}'
                        )
                        continue
                    fallback_full += 1
                    self.print(
                        'Fallback to full HF Arrow cache warmup because frame range could not be mapped: '
                        f'data_path={data_path}, range=[{child_start}, {child_end}), error={error}'
                    )
                    if hasattr(child, 'clear_hf_cache_frame_range'):
                        child.clear_hf_cache_frame_range()
                    warmup_items.append((child, 0, child_len, 'full'))
                    continue
                warmup_items.append((child, child_start, child_end, 'shard'))
        return warmup_items, skipped_non_lerobot, fallback_full

    def _warmup_hf_arrow_cache_sampler_shard(
        self,
        sampler_shard_plan: dict[str, Any],
        *,
        log_interval: int,
        release_after_open: bool,
        process_scope: str,
    ) -> int:
        warmup_items = sampler_shard_plan['warmup_items']
        skipped_non_lerobot = sampler_shard_plan['skipped_non_lerobot']
        fallback_full = sampler_shard_plan['fallback_full']
        self.print(
            'Warm up HF Arrow cache for sampler shard '
            f'{sampler_shard_plan["shard_rank"]}/{sampler_shard_plan["shard_world_size"]} '
            f'global_range=[{sampler_shard_plan["shard_start"]}, {sampler_shard_plan["shard_end"]}) '
            f'process_index={getattr(self, "process_index", "?")}, '
            f'local_process_index={getattr(self, "local_process_index", "?")} '
            f'(process_scope={process_scope}, warmup_mode=sampler_shard)'
        )

        total = len(warmup_items)
        self.print(
            'Sampler-shard HF Arrow cache warmup selected '
            f'{total} LeRobot ranges; skipped_non_lerobot={skipped_non_lerobot}, fallback_full={fallback_full}'
        )
        for i, (child, child_start, child_end, mode) in enumerate(warmup_items, start=1):
            data_path = getattr(child, 'data_path', '<unknown>')
            episodes = getattr(child, '_hf_cache_episode_ids', None)
            tic = time.time()
            self._debug_train_boot(
                'HF Arrow shard warmup item start [%s/%s] mode=%s range=[%s,%s) episodes=%s data_path=%s',
                i,
                total,
                mode,
                child_start,
                child_end,
                len(episodes) if episodes is not None else None,
                data_path,
            )
            try:
                self._warmup_lerobot_child_hf_arrow_cache(child, release_after_open=release_after_open)
            except Exception as error:
                raise RuntimeError(
                    'Failed to warm up HF Arrow cache for LeRobotDataset: '
                    f'data_path={data_path}, range=[{child_start}, {child_end}), mode={mode}'
                ) from error
            self._debug_train_boot(
                'HF Arrow shard warmup item finish [%s/%s] elapsed=%.3fs mode=%s range=[%s,%s) data_path=%s',
                i,
                total,
                time.time() - tic,
                mode,
                child_start,
                child_end,
                data_path,
            )
            if log_interval > 0 and (i == 1 or i == total or i % log_interval == 0):
                episode_text = '' if episodes is None else f', episodes={len(episodes)}'
                self.print(
                    f'HF Arrow shard warmup [{i}/{total}] {time.time() - tic:.1f}s '
                    f'mode={mode}, range=[{child_start}, {child_end}){episode_text} {data_path}'
                )
        return total

    def _warmup_lerobot_child_hf_arrow_cache(self, child: Any, *, release_after_open: bool) -> None:
        if release_after_open and hasattr(child, 'warmup_hf_arrow_cache'):
            child.warmup_hf_arrow_cache()
            if hasattr(child, 'dataset'):
                child.dataset = None
            return

        child.open()
        if release_after_open and hasattr(child, 'dataset'):
            child.dataset = None

    def _iter_dataset_range_leaves(self, dataset: Any, range_start: int, range_end: int):
        if range_end <= range_start:
            return
        children = getattr(dataset, 'datasets', None)
        if not children:
            yield dataset, range_start, range_end
            return

        if not hasattr(dataset, '_get_cumulative_sizes'):
            for child in children:
                yield from self._iter_dataset_range_leaves(child, range_start, range_end)
            return

        previous_size = 0
        for child, cumulative_size in zip(children, dataset._get_cumulative_sizes()):
            child_start = max(range_start, previous_size)
            child_end = min(range_end, cumulative_size)
            if child_start < child_end:
                yield from self._iter_dataset_range_leaves(
                    child,
                    child_start - previous_size,
                    child_end - previous_size,
                )
            previous_size = cumulative_size

    def _iter_lerobot_datasets(self, dataset: Any):
        if dataset.__class__.__name__ == 'LeRobotDataset':
            yield dataset
            return
        for child in getattr(dataset, 'datasets', []) or []:
            yield from self._iter_lerobot_datasets(child)

    def _hf_arrow_cache_marker_path(self, cache_cfg: dict[str, Any]) -> str:
        marker_path = cache_cfg.get('marker_path', None)
        if marker_path is not None:
            return os.fspath(marker_path)
        cache_root = os.environ.get('HF_DATASETS_CACHE', None)
        if cache_root is None:
            cache_root = os.path.join(self.project_dir, 'cache')
        job_id = '_'.join(
            str(os.environ.get(name, 'unknown')).replace('/', '_').replace(':', '_')
            for name in ('MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE')
        )
        return os.path.join(cache_root, '_warmup_markers', f'hf_arrow_warmup_{job_id}.done')

    def _write_hf_arrow_cache_marker(self, marker_path: str, total: int) -> None:
        os.makedirs(os.path.dirname(marker_path), exist_ok=True)
        tmp_path = f'{marker_path}.{os.getpid()}.tmp'
        with open(tmp_path, 'w') as f:
            f.write(f'pid={os.getpid()}\n')
            f.write(f'process_index={self.process_index}\n')
            f.write(f'num_lerobot_datasets={total}\n')
            f.write(f'time={time.time()}\n')
        os.replace(tmp_path, marker_path)

    def _wait_hf_arrow_cache_marker(self, marker_path: str, cache_cfg: dict[str, Any]) -> None:
        poll_interval_s = float(cache_cfg.get('poll_interval_s', 10.0))
        wait_log_interval_s = float(cache_cfg.get('wait_log_interval_s', 300.0))
        timeout_s = cache_cfg.get('wait_timeout_s', 0)
        timeout_s = float(timeout_s) if timeout_s is not None else 0.0

        start = time.time()
        last_log = start
        while not os.path.exists(marker_path):
            elapsed = time.time() - start
            if timeout_s > 0 and elapsed > timeout_s:
                raise TimeoutError(f'Timed out waiting for HF Arrow cache marker after {elapsed:.1f}s: {marker_path}')
            if self.local_process_index == 0 and time.time() - last_log >= wait_log_interval_s:
                print(
                    f'[rank{self.process_index}] waiting for HF Arrow cache marker '
                    f'for {elapsed:.1f}s: {marker_path}',
                    flush=True,
                )
                last_log = time.time()
            time.sleep(poll_interval_s)

    def get_models(self, model_config: dict[str, Any]) -> GigaBrain07Policy:
        """Initializes and returns the GigaBrain07Policy model."""
        llm_loss_weight = self.kwargs.get('llm_loss_weight')
        if llm_loss_weight is None and 'llm_loss_weight' in model_config:
            llm_loss_weight = model_config.pop('llm_loss_weight')

        flow_action_dim_loss_weight_cfg = self.kwargs.get(
            'flow_action_dim_loss_weight_cfg'
        )
        if (
            flow_action_dim_loss_weight_cfg is None
            and 'flow_action_dim_loss_weight_cfg' in model_config
        ):
            flow_action_dim_loss_weight_cfg = model_config.pop(
                'flow_action_dim_loss_weight_cfg'
            )

        vlm_type = model_config.get('vlm_type', 'paligemma2')
        if vlm_type != 'paligemma2':
            raise ValueError(
                f'GigaBrain07Trainer requires PaliGemma2, got '
                f'vlm_type={vlm_type!r}'
            )

        prompt_cfg = (
            getattr(self, '_train_transform_cfg', {})
            .get('prompt_cfg', {})
        )
        if model_config.get('state_input_mode') == 'proprio_anchor':
            tokenizer_model_path = prompt_cfg.get('tokenizer_model_path')
            if not tokenizer_model_path:
                raise ValueError('proprio_anchor requires prompt_cfg.tokenizer_model_path')
            from giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils import (
                resolve_propri_token_id,
            )

            propri_token_id, tokenizer_size, base_vocab_size = resolve_propri_token_id(
                tokenizer_model_path
            )
            if propri_token_id >= 257216:
                raise ValueError(
                    f'<|propri|> token id {propri_token_id} exceeds the fixed '
                    'PaliGemma2 embedding table (257216 rows)'
                )
            fast_max_id = (
                base_vocab_size
                - 1
                - int(model_config.get('fast_token_tail_skip_tokens', 128))
            )
            fast_vocab_size = int(model_config.get('fast_vocab_size', 0))
            fast_min_id = fast_max_id - fast_vocab_size + 1
            if fast_vocab_size > 0 and fast_min_id <= propri_token_id <= fast_max_id:
                raise ValueError(
                    f'<|propri|> token id {propri_token_id} overlaps the FAST tail range '
                    f'[{fast_min_id}, {fast_max_id}]'
                )
            if tokenizer_size > 257216:
                raise ValueError(
                    f'tokenizer size {tokenizer_size} exceeds the fixed PaliGemma2 vocabulary'
                )
            configured_id = model_config.get('propri_token_id')
            if configured_id is not None and int(configured_id) != propri_token_id:
                raise ValueError(
                    f'configured propri_token_id={configured_id} does not match tokenizer '
                    f'id {propri_token_id}'
                )
            model_config['propri_token_id'] = propri_token_id

        if hasattr(model_config, 'pretrained'):
            pretrained = model_config.pop('pretrained')
            gigabrain07 = GigaBrain07Policy.from_pretrained(pretrained)
            loaded_vlm_type = getattr(gigabrain07, 'vlm_type', vlm_type)
            if loaded_vlm_type != 'paligemma2':
                raise ValueError(
                    f'GigaBrain07Trainer only supports PaliGemma2 checkpoints, '
                    f'got vlm_type={loaded_vlm_type!r}'
                )
            if len(model_config.keys()) > 0:
                gigabrain07 = process_model(gigabrain07, model_config)
        else:
            gigabrain07 = GigaBrain07Policy(**model_config)
            if hasattr(model_config, 'pretrained_paligemma_path'):
                pretrained_paligemma_state_dict = torch.load(
                    model_config.pretrained_paligemma_path,
                    map_location='cpu',
                )

                dropped_lm_head = _drop_disabled_paligemma_lm_head_weight(
                    pretrained_paligemma_state_dict,
                    enable_next_token_prediction=gigabrain07.enable_next_token_prediction,
                )
                if dropped_lm_head:
                    self.print(
                        'Skipped paligemma_with_expert.lm_head.weight because '
                        'enable_next_token_prediction=False.'
                    )

                patch_embedding_key = (
                    'paligemma_with_expert.vision_tower.embeddings.'
                    'patch_embedding.weight'
                )
                weight = pretrained_paligemma_state_dict[patch_embedding_key]
                pretrained_paligemma_state_dict[patch_embedding_key] = (
                    _resize_patch_embedding_weight(
                        weight,
                        gigabrain07.vision_in_channels,
                    )
                )
                _resize_embodiment_specific_params_for_load(
                    pretrained_paligemma_state_dict,
                    gigabrain07,
                    source='PaliGemma pretrained checkpoint',
                )

                _, unexpected_keys = gigabrain07.load_state_dict(
                    pretrained_paligemma_state_dict,
                    strict=False,
                )
                if unexpected_keys:
                    raise ValueError(f'Unexpected keys: {unexpected_keys}')

        if gigabrain07.enable_next_token_prediction:
            assert (
                gigabrain07.paligemma_with_expert.lm_head.weight.data_ptr()
                == gigabrain07.paligemma_with_expert.embed_tokens.weight.data_ptr()
            ), 'PaliGemma lm_head and embed_tokens weights are not tied'

        if (
            self.distributed_type == DistributedType.DEEPSPEED
            and getattr(self, 'is_deepspeed_zero3', False)
        ):
            self.print(
                'DeepSpeed ZeRO-3 is enabled; leaving the model on its '
                'construction device until accelerator.prepare().'
            )
        else:
            gigabrain07.to(self.device)
        gigabrain07.train()

        self.loss_func = GigaBrain07Loss(
            llm_loss_weight=(
                float(llm_loss_weight)
                if llm_loss_weight is not None
                else 1.0
            ),
            flow_action_dim_loss_weight_cfg=(
                dict(flow_action_dim_loss_weight_cfg)
                if flow_action_dim_loss_weight_cfg is not None
                else None
            ),
        )
        self.num_embodiments = int(
            getattr(gigabrain07, 'config', {}).get('num_embodiments', 0)
        )
        return gigabrain07

    def _log_forward_crash_provenance(self, batch_dict: dict, err: BaseException) -> None:
        """Name the offending sample when the model forward raises (e.g. a CUDA
        illegal memory access). CPU-side only: after a device fault the CUDA
        context is poisoned, so this must NOT touch any GPU tensor's data."""
        def _safe(v):
            try:
                if isinstance(v, torch.Tensor):
                    if v.is_cuda:
                        return f'<cuda tensor {tuple(v.shape)}>'  # avoid D2H after poison
                    return v.tolist()
                return v
            except Exception:
                return '<unrepr>'

        rank = getattr(self, 'process_index', '?')
        lines = [
            f'[GIGA_IMA host={socket.gethostname()} rank={rank}] '
            f'model forward raised {type(err).__name__}: {err}',
            f'  debug_repo_id={_safe(batch_dict.get("debug_repo_id"))}',
            f'  debug_episode_index={_safe(batch_dict.get("debug_episode_index"))}',
            f'  debug_frame_index={_safe(batch_dict.get("debug_frame_index"))}',
            f'  embodiment_id={_safe(batch_dict.get("embodiment_id"))}',
        ]
        imgs = batch_dict.get('images')
        if isinstance(imgs, (list, tuple)):
            shapes = [tuple(i.shape) for i in imgs if isinstance(i, torch.Tensor)]
            lines.append(f'  n_cameras={len(imgs)} image_shapes={shapes}')
        lt = batch_dict.get('lang_tokens')
        if isinstance(lt, torch.Tensor):
            lines.append(f'  lang_tokens.shape={tuple(lt.shape)}')
        msg = '\n'.join(str(line) for line in lines)
        logger = getattr(self, 'logger', None)
        try:
            if logger is not None:
                logger.error(msg)
        except Exception:
            pass
        print(msg, flush=True)

    def forward_step(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Performs a forward pass and calculates the loss.

        Args:
            batch_dict (dict[str, torch.Tensor]): A dictionary containing the batch data.

        Returns:
            dict[str, torch.Tensor]: A dictionary of losses.
        """
        if getattr(self, '_active_worker_profile_enabled', False):
            self._worker_profile_add_many(self._pipeline_worker_profile_pop_batch(batch_dict))

        images = batch_dict['images']
        img_masks = batch_dict['image_masks']
        lang_tokens = batch_dict['lang_tokens']
        lang_masks = batch_dict['lang_masks']

        lang_att_masks = batch_dict['lang_att_masks']
        lang_loss_masks = batch_dict['lang_loss_masks']
        fast_action_indicator = batch_dict['fast_action_indicator']
        subtask_indicator = batch_dict['subtask_indicator']

        actions = batch_dict['action']
        action_loss_mask = batch_dict['action_loss_mask']
        action_dim_loss_mask = batch_dict.get('action_dim_loss_mask')
        action_fps = batch_dict.get('action_fps')
        skip_loss_mask = batch_dict.get('skip_loss_mask')

        traj = None
        traj_loss_mask = None
        if 'traj' in batch_dict:
            traj = batch_dict['traj']
            traj_loss_mask = batch_dict['traj_loss_mask']

        lang_loss_masks, action_loss_mask, action_dim_loss_mask, traj_loss_mask = _apply_skip_loss_mask(
            skip_loss_mask,
            lang_loss_masks,
            action_loss_mask,
            action_dim_loss_mask,
            traj_loss_mask,
        )

        emb_ids = batch_dict['embodiment_id']
        # print(f'emb_ids: {emb_ids}')
        flow_action_horizon = _get_flow_action_horizon(self.model)
        if action_dim_loss_mask is None:
            actions, action_loss_mask = downsample_flow_action_tensors(
                actions,
                action_loss_mask,
                flow_action_horizon,
                action_fps=action_fps,
            )
        else:
            actions, action_loss_mask, action_dim_loss_mask = downsample_flow_action_tensors(
                actions,
                action_loss_mask,
                flow_action_horizon,
                action_fps=action_fps,
                action_dim_loss_mask=action_dim_loss_mask,
            )
        if actions.ndim != 3 or actions.shape[-1] != 32:
            raise ValueError(f'Flow actions must have shape [B, T, 32], got {tuple(actions.shape)}')

        enable_multi_flow_samples = bool(self.kwargs.get('enable_multi_flow_samples', False))
        flow_matching_batch_mul = int(self.kwargs.get('flow_matching_batch_mul', 4)) if enable_multi_flow_samples else 1
        if flow_matching_batch_mul < 1:
            raise ValueError(f'flow_matching_batch_mul must be positive, but got {flow_matching_batch_mul}')

        noisy_model_input, timesteps = self.loss_func.add_noise(
            actions,
            action_dim_loss_mask,
            batch_mul=flow_matching_batch_mul,
        )
        action_emb_ids = emb_ids
        if flow_matching_batch_mul > 1:
            action_emb_ids = emb_ids.repeat_interleave(flow_matching_batch_mul, dim=0)
            action_loss_mask = action_loss_mask.repeat_interleave(flow_matching_batch_mul, dim=0)
            if action_dim_loss_mask is not None:
                action_dim_loss_mask = action_dim_loss_mask.repeat_interleave(flow_matching_batch_mul, dim=0)
        # --- GIGA_DEBUG_IMA: name the offending sample on a CUDA illegal-memory-access ---
        # With env GIGA_DEBUG_IMA=1 we synchronize right after the forward so an ASYNC
        # device fault surfaces HERE (attributed to THIS batch) instead of at a later
        # collective; on any forward error we log the batch's repo_id/episode/frame
        # (CPU-side only) and re-raise. Zero extra cost for normal runs (env unset).
        try:
            model_pred = self.model(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                noisy_model_input,
                timesteps,
                emb_ids,
                lang_att_masks=lang_att_masks,
                fast_action_indicator=fast_action_indicator,
                subtask_indicator=subtask_indicator,
                lang_loss_masks=lang_loss_masks,
                state_memory=batch_dict.get('observation.state_memory', None),
                state_memory_masks=batch_dict.get('observation.state_memory_mask', None),
                proprioception=batch_dict.get('observation.proprioception', None),
                agent_pos_mask=batch_dict.get('observation.agent_pos_mask', None),
                proprioception_present=batch_dict.get(
                    'observation.proprioception_present', None
                ),
                state=batch_dict.get('observation.state'),
                action_emb_ids=action_emb_ids,
                flow_matching_batch_mul=flow_matching_batch_mul,
            )
            if _str_to_bool(os.environ.get('GIGA_DEBUG_IMA'), False) and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
        except Exception as _giga_fwd_err:
            self._log_forward_crash_provenance(batch_dict, _giga_fwd_err)
            raise

        # Cast compact predictions to float32 for loss computation (u_t from
        # add_noise is float32). Keep full-vocab language logits in their
        # forward dtype; GigaBrain07Loss chunks and casts them before CE.
        model_pred = {
            k: v
            if k in {'lang_logits', 'lang_logits_mask', 'lang_token_loss'} or not isinstance(v, torch.Tensor)
            else v.float()
            for k, v in model_pred.items()
        }
        loss = self.loss_func(
            model_pred,
            lang_tokens,
            lang_loss_masks,
            action_loss_mask,
            traj,
            traj_loss_mask,
            action_dim_loss_mask=action_dim_loss_mask,
            fast_action_indicator=fast_action_indicator,
            subtask_indicator=subtask_indicator,
            embodiment_id=action_emb_ids,
            num_embodiments=getattr(self, 'num_embodiments', 0),
        )
        invalid_action_chunk = batch_dict.get('invalid_action_chunk')
        if invalid_action_chunk is not None:
            loss['metric/invalid_action_chunk_rate'] = invalid_action_chunk.float().mean()
        loss['metric/flow_matching_batch_mul'] = model_pred['v_t'].new_tensor(
            float(flow_matching_batch_mul)
        )
        loss['metric/data_batch_size'] = model_pred['v_t'].new_tensor(float(actions.shape[0]))
        loss['metric/action_batch_size'] = model_pred['v_t'].new_tensor(
            float(model_pred['v_t'].shape[0])
        )
        return loss

    def backward_step(self, loss: torch.Tensor) -> None:
        if not torch.isfinite(loss.detach()).all().item():
            raise FloatingPointError(
                f'Non-finite loss on rank {self.process_index}: {loss.detach()}'
            )
        with self._pipeline_profile_stage('backward'):
            self.accelerator.backward(loss)
        max_grad_norm = self.kwargs.get('max_grad_norm', None)
        grad_norm_type = self.kwargs.get('grad_norm_type', 2)
        if self.accelerator.sync_gradients and max_grad_norm is not None:
            with self._pipeline_profile_stage('grad_clip'):
                params = []
                for model in self.models:
                    params += list(model.parameters())
                self.accelerator.clip_grad_norm_(params, max_grad_norm, grad_norm_type)
        with self._pipeline_profile_stage('optimizer_step'):
            for optimizer in self.optimizers:
                optimizer.step()
        with self._pipeline_profile_stage('scheduler_step'):
            for scheduler in self.schedulers:
                scheduler.step()
        with self._pipeline_profile_stage('zero_grad'):
            for optimizer in self.optimizers:
                optimizer.zero_grad()
        if self.accelerator.sync_gradients and self.with_ema:
            with self._pipeline_profile_stage('ema'):
                for model, ema_model in zip(self.models, self.ema_models):
                    if self.distributed_type == DistributedType.DEEPSPEED:
                        if self.accelerator.deepspeed_config['zero_optimization']['stage'] == 3:
                            state_dict = self.accelerator.get_state_dict(model)
                        else:
                            state_dict = self.accelerator.unwrap_model(model, keep_torch_compile=False).state_dict()
                    elif self.accelerator.is_fsdp2:
                        options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, cpu_offload=False)
                        state_dict = get_model_state_dict(model, options=options)
                    else:
                        state_dict = self.accelerator.unwrap_model(model, keep_torch_compile=False).state_dict()
                    ema_model.step(state_dict)

    def _record_output(
        self,
        key: str,
        value: torch.Tensor | float,
        count: torch.Tensor | float | int = 1,
    ) -> None:
        """Record a scalar across both current and legacy GigaTrain APIs."""
        parent_record = getattr(super(), '_record_output', None)
        if parent_record is not None:
            parent_record(key, value, count=count)
            return

        value_scalar = value.detach().item() if isinstance(value, torch.Tensor) else float(value)
        count_scalar = count.detach().item() if isinstance(count, torch.Tensor) else float(count)
        bucket = self._outputs.setdefault(key, {'sum': 0.0, 'num': 0.0})
        bucket['sum'] += value_scalar
        bucket['num'] += count_scalar

    def _record_outputs(self, outputs: dict[str, torch.Tensor]) -> None:
        for key, value in outputs.items():
            self._record_output(key, value)

    def parse_losses(self, losses):
        """Same as Trainer.parse_losses, but skips the per-step global NaN
        all-reduce and treats keys starting with
        ``metric/`` as diagnostics: they are accumulated locally (NaN-aware,
        a per-rank NaN means the rank had no sample of that embodiment) and
        handed to the base trainer's logging accumulator, but NOT added to
        ``total_loss`` or used in backward.
        """
        if not isinstance(losses, dict):
            loss = losses.mean()
            self._record_output('total_loss', loss.detach())
            return loss

        metric_losses = {key: losses.pop(key) for key in list(losses.keys()) if key.startswith('metric/')}

        assert 'total_loss' not in losses
        for key, val in losses.items():
            losses[key] = val.mean()
        loss = sum(losses.values())
        outputs = {key: val.detach() for key, val in losses.items()}
        outputs['total_loss'] = loss.detach()
        self._record_outputs(outputs)

        if metric_losses:
            for key, val in metric_losses.items():
                metric = val.detach().mean()
                finite = torch.isfinite(metric).to(dtype=torch.float64)
                self._record_output(key, torch.nan_to_num(metric, nan=0.0, posinf=0.0, neginf=0.0), count=finite)

        return loss

    def print_before_train(self) -> None:
        self._worker_profile_reset_window()
        super().print_before_train()

    def print_step(self) -> None:
        should_log = self.cur_step % self.log_interval == 0
        worker_outputs: dict[str, float] = {}
        worker_msg = None
        if should_log and getattr(self, '_active_worker_profile_enabled', False):
            worker_outputs, worker_msg = self._worker_profile_outputs_and_message()
        super().print_step()
        if should_log and self.is_main_process:
            if worker_outputs:
                self.accelerator.log(worker_outputs, self.cur_step)
            if worker_msg:
                self.logger.info(worker_msg)

    def resume(self, checkpoint: str | list[str] | None = None) -> None:
        # FSDP2 guard: reconcile the live trainable-parameter set with what the
        # checkpoint optimizer actually stored, before the base resume loads it.
        resolved_checkpoint = self.get_checkpoint(checkpoint)
        if resolved_checkpoint is None:
            return
        self._freeze_optimizer_state_orphans(resolved_checkpoint)
        compile_warmed = bool(self.kwargs.get('resume_compile_warmup', False))
        if compile_warmed:
            self._compile_backward_before_resume()
        super().resume(resolved_checkpoint)
        gc.collect()
        if not compile_warmed:
            torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

    def _compile_backward_before_resume(self) -> None:
        """Compile forward/backward before restored Adam states occupy GPU memory."""
        self.print(
            '[resume] running one compile-only forward/backward before loading checkpoint state'
        )
        dataloader_iter = iter(self.dataloader)
        batch_dict = next(dataloader_iter)
        losses = self.forward_step(batch_dict)
        loss = self.parse_losses(losses)
        self.accelerator.backward(loss)
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        self._outputs.clear()
        self.accelerator.wait_for_everyone()
        del loss, losses, batch_dict, dataloader_iter
        gc.collect()
        self.print(
            '[resume] compile-only backward finished; loading checkpoint state'
        )

    def _freeze_optimizer_state_orphans(self, checkpoint: str | list[str] | None = None) -> None:
        """Freeze trainable params the checkpoint optimizer has no state for.

        torch's ``set_optimizer_state_dict`` (FSDP2 / FULL_STATE_DICT resume path)
        does a hard lookup ``state[fqn] = saved[_STATE][fqn]`` guarded only by
        ``if param.requires_grad`` (torch/distributed/checkpoint/state_dict.py).
        Adam allocates per-parameter state lazily, so any ``requires_grad=True``
        parameter that never received a gradient (e.g. ``lm_head`` and the final
        VLM decoder layer when ``enable_next_token_prediction=False``) is absent
        from the saved optimizer and makes resume raise ``KeyError``. Those params
        never updated anyway, so we freeze exactly the orphans to make the live
        trainable set match the checkpoint; training is unaffected.
        """
        if self.distributed_type != DistributedType.FSDP:
            return
        fsdp_plugin = getattr(self.accelerator.state, 'fsdp_plugin', None)
        if fsdp_plugin is None or getattr(fsdp_plugin, 'fsdp_version', 1) != 2:
            return
        ckpt = self.get_checkpoint(checkpoint)
        if ckpt is None:
            return
        ckpt_dir = ckpt[0] if isinstance(ckpt, list) else ckpt

        # Read the saved optimizer state FQNs on the main process only, then
        # broadcast so every rank freezes an identical set (FSDP needs agreement).
        from accelerate.utils import broadcast_object_list

        payload: list = [None]
        if self.is_main_process:
            saved_keys = []
            for i in range(len(self.optimizers)):
                opt_file = os.path.join(ckpt_dir, 'optimizer.bin' if i == 0 else f'optimizer_{i}.bin')
                if not os.path.exists(opt_file):
                    saved_keys.append(None)  # sharded/missing optimizer -> skip
                    continue
                opt_state = torch.load(opt_file, map_location='cpu', mmap=True, weights_only=True)
                state = opt_state.get('state', {}) if isinstance(opt_state, dict) else {}
                saved_keys.append(set(state.keys()))
            payload = [saved_keys]
        broadcast_object_list(payload, from_process=0)
        saved_keys = payload[0]
        if not saved_keys:
            return

        # named_parameters() may carry FSDP / activation-checkpoint / torch.compile
        # wrapper path components, while the saved optimizer FQNs are torch-normalized
        # (clean). Strip those components before comparing, mirroring torch's _get_fqns.
        _wrap = ('_fsdp_wrapped_module', '_checkpoint_wrapped_module', '_orig_mod')

        def _fqn(name: str) -> str:
            return '.'.join(c for c in name.split('.') if c not in _wrap)

        total_frozen = 0
        for i, model in enumerate(self.models):
            if i >= len(saved_keys) or saved_keys[i] is None:
                continue
            saved = saved_keys[i]
            trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
            orphans = [(name, p) for name, p in trainable if _fqn(name) not in saved]
            # Safety valve: a healthy resume orphans only a handful of params (e.g.
            # lm_head + the final VLM layer). If most params look orphaned the FQN
            # normalization is wrong -> skip freezing rather than cripple the model.
            if trainable and len(orphans) > 0.4 * len(trainable):
                self.logger.info(
                    f'[resume] WARNING: {len(orphans)}/{len(trainable)} trainable params appear absent '
                    f'from optimizer[{i}] state in {ckpt_dir} - likely an FQN-normalization mismatch; '
                    f'skipping freeze (resume may still raise). First: {orphans[0][0]}'
                )
                continue
            for name, p in orphans:
                p.requires_grad_(False)
            total_frozen += len(orphans)
            if orphans:
                self.logger.info(
                    f'[resume] froze {len(orphans)} trainable param(s) with no optimizer state in '
                    f'{ckpt_dir} (optimizer[{i}]); first: {orphans[0][0]}. These never received a '
                    f'gradient, so training is unaffected.'
                )
        if total_frozen == 0:
            self.logger.info(f'[resume] trainable set matches checkpoint optimizer in {ckpt_dir}; nothing frozen.')

    def train(self) -> None:
        cfg = self._pipeline_profile_cfg()
        if not cfg:
            return super().train()

        self._active_pipeline_profile_enabled = True
        self._active_pipeline_profile_cfg = cfg
        if int(cfg.get('step_interval', 0)) <= 0:
            cfg['step_interval'] = max(1, int(self.log_interval) * 10)
        self.print_before_train()
        self._pipeline_profile_reset_window()
        if self.is_main_process:
            self.logger.info(
                'Pipeline profile enabled: epoch_interval=%s, step_interval=%s, sync_cuda=%s',
                cfg.get('epoch_interval'),
                cfg.get('step_interval'),
                cfg.get('sync_cuda'),
            )

        tic = time.perf_counter()
        self._debug_train_boot('train iter(self.dataloader) start')
        dataloader_iter = iter(self.dataloader)
        self._debug_train_boot('train iter(self.dataloader) finish elapsed=%.3fs', time.perf_counter() - tic)
        for self._cur_step in range(self._cur_step, self.max_steps):
            self._cur_step += 1
            step_tic = time.perf_counter()
            for _ in range(self.gradient_accumulation_steps):
                data_tic = time.perf_counter()
                if self.cur_step == 1:
                    self._debug_train_boot('train first next(dataloader_iter) start')
                batch_dict = next(dataloader_iter)
                if self.cur_step == 1:
                    self._debug_train_boot('train first next(dataloader_iter) finish elapsed=%.3fs', time.perf_counter() - data_tic)
                data_time = time.perf_counter() - data_tic
                self._pipeline_profile_add('data_next', data_time)
                self._pipeline_profile_add('micro_steps', 1.0)
                if getattr(self, 'log_data_time', False):
                    self._data_time_sum = getattr(self, '_data_time_sum', 0.0) + data_time
                    self._data_time_num = getattr(self, '_data_time_num', 0) + 1
                self._pipeline_profile_add_many(self._pipeline_profile_pop_batch(batch_dict))
                with self.accelerator.accumulate(*self.models):
                    with self._pipeline_profile_stage('forward_step'):
                        losses = self.forward_step(batch_dict)
                    with self._pipeline_profile_stage('parse_losses'):
                        loss = self.parse_losses(losses)
                    with self._pipeline_profile_stage('backward_step'):
                        self.backward_step(loss)

            with self._pipeline_profile_stage('print_step'):
                self.print_step()
            with self._pipeline_profile_stage('save_checkpoint_step'):
                self.save_checkpoint_step()
            self._pipeline_profile_add('step_total', time.perf_counter() - step_tic)
            self._pipeline_profile_add('steps', 1.0)
            epoch_completed = self.cur_step % self.epoch_size == 0
            if self._pipeline_profile_should_emit(epoch_completed):
                self._pipeline_profile_log_window()

        self.print_after_train()
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()




def process_model(model: GigaBrain07Policy, model_config: dict[str, Any]) -> GigaBrain07Policy:
    """Rebuild a pretrained model under an updated config and carry its weights over.

    Two use cases:
      - PaliGemma input-channel resize (RGB -> RGB+depth) of the vision patch
        embedding, plus embodiment-category count changes.
      - PaliGemma config-only overrides that do not change any parameter shape
        (for example, disabling knowledge insulation or next-token prediction).

    Args:
        model (GigaBrain07Policy): The pre-trained GigaBrain07Policy model.
        model_config (dict[str, Any]): Config keys to override on top of model.config.

    Returns:
        GigaBrain07Policy: The processed model with updated weights.
    """
    state_dict = model.state_dict()

    updated_model_config = dict(model.config)
    for key in model_config:
        updated_model_config[key] = model_config[key]

    new_model = GigaBrain07Policy(**updated_model_config)
    new_state_dict = new_model.state_dict()
    resize_warnings = []

    # PaliGemma / PaliGemma2 support input-channel resize (RGB -> RGB+depth)
    # through their Conv2d patch embedding.
    patch_embedding_key = 'paligemma_with_expert.vision_tower.embeddings.patch_embedding.weight'
    if patch_embedding_key in state_dict and 'vision_in_channels' in updated_model_config:
        weight = state_dict[patch_embedding_key]
        new_weight = _resize_patch_embedding_weight(weight, updated_model_config['vision_in_channels'])
        if tuple(weight.shape) != tuple(new_weight.shape):
            resize_warnings.append(
                f'{patch_embedding_key}: checkpoint{tuple(weight.shape)} -> model{tuple(new_weight.shape)} (resized in_channels)'
            )
        state_dict[patch_embedding_key] = new_weight

    # Copy checkpoint params into the resized model.
    # For embodiment-specific projections, allow category-count changes
    # (e.g. checkpoint has 3 categories, target model uses 4).
    for key, value in state_dict.items():
        if key not in new_state_dict:
            continue
        target = new_state_dict[key]
        if value.shape == target.shape:
            new_state_dict[key] = value
            continue
        if _is_embodiment_specific_proj_param(key):
            resized_tensor, n_copy, copied = _resize_embodiment_param(value, target)
            new_state_dict[key] = resized_tensor
            if copied:
                resize_warnings.append(
                    f'{key}: checkpoint{tuple(value.shape)} -> model{tuple(target.shape)} (copied first {n_copy} embodiment categories)'
                )
            else:
                resize_warnings.append(
                    f'{key}: checkpoint{tuple(value.shape)} -> model{tuple(target.shape)} (shape incompatible, kept model initialization)'
                )
            continue
        resize_warnings.append(
            f'{key}: checkpoint{tuple(value.shape)} -> model{tuple(target.shape)} (shape mismatch, kept model initialization)'
        )

    if resize_warnings:
        warning_msg = 'Detected parameter shape mismatch while loading pretrained weights:\n' + '\n'.join(
            f'- {msg}' for msg in resize_warnings
        )
        warnings.warn(warning_msg, stacklevel=2)
    new_model.load_state_dict(new_state_dict, strict=False)
    del model
    model = new_model

    return model




def _resize_patch_embedding_weight(weight: torch.Tensor, target_in_channels: int) -> torch.Tensor:
    """Resizes the patch embedding weights to match the target number of input
    channels.

    Args:
        weight (torch.Tensor): The original patch embedding weights.
        target_in_channels (int): The target number of input channels.

    Returns:
        torch.Tensor: The resized patch embedding weights.
    """
    current_in_channels = weight.shape[1]
    if current_in_channels == target_in_channels:
        return weight
    if current_in_channels > target_in_channels:
        return weight[:, :target_in_channels, :, :]
    new_shape = list(weight.shape)
    new_shape[1] = target_in_channels
    new_weight = weight.new_zeros(new_shape)
    new_weight[:, :current_in_channels, :, :] = weight
    return new_weight


def _is_embodiment_specific_proj_param(key: str) -> bool:
    return key in {
        'action_in_proj.weight',
        'action_in_proj.bias',
        'action_out_proj.weight',
        'action_out_proj.bias',
        'proprio_state_proj.weight',
        'proprio_state_proj.bias',
    }


def _resize_embodiment_param(src: torch.Tensor, dst_template: torch.Tensor) -> tuple[torch.Tensor, int, bool]:
    """Resize embodiment-specific params along category dimension (dim=0).

    Copy the shared prefix categories and keep remaining target categories as
    freshly initialized values from dst_template.
    """
    if src.ndim != dst_template.ndim:
        return dst_template, 0, False
    if src.shape[1:] != dst_template.shape[1:]:
        return dst_template, 0, False

    dst = dst_template.clone()
    n_copy = min(src.shape[0], dst.shape[0])
    dst[:n_copy] = src[:n_copy]
    return dst, n_copy, True


def _resize_embodiment_specific_params_for_load(
    state_dict: dict,
    model: torch.nn.Module,
    source: str,
) -> None:
    model_state_dict = model.state_dict()
    resize_warnings = []

    for key in tuple(state_dict):
        value = state_dict[key]
        if key not in model_state_dict or not isinstance(value, torch.Tensor):
            continue
        target = model_state_dict[key]
        if value.shape == target.shape or not _is_embodiment_specific_proj_param(key):
            continue

        resized_tensor, n_copy, copied = _resize_embodiment_param(value, target)
        if not copied:
            continue

        state_dict[key] = resized_tensor
        resize_warnings.append(
            f'{key}: checkpoint{tuple(value.shape)} -> model{tuple(target.shape)} '
            f'(copied first {n_copy} embodiment categories)'
        )

    if resize_warnings:
        warning_msg = f'Detected embodiment-count mismatch while loading {source}:\n' + '\n'.join(
            f'- {msg}' for msg in resize_warnings
        )
        warnings.warn(warning_msg, stacklevel=2)
