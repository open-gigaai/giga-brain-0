"""PaliGemma2 flow-matching open-loop rollout inference.

The script evaluates a trained checkpoint on full LeRobot episodes. It uses the
checkpoint/training config to rebuild the same inference transform path used by
training, including norm stats, robot-type delta masks, control/end-effector
tokens, expanded FAST vocab, and proprio-memory state input.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
GIGABRAIN07_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = SCRIPT_DIR.parents[4]
CODE_ROOT = REPO_ROOT.parent


def _resolve_dependency_root(env_name: str, *sibling_names: str) -> Path:
    configured = os.environ.get(env_name)
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f'{env_name} does not exist: {root}')
        return root
    for sibling_name in sibling_names:
        root = CODE_ROOT / sibling_name
        if root.is_dir():
            return root
    expected = ', '.join(os.fspath(CODE_ROOT / name) for name in sibling_names)
    raise FileNotFoundError(f'Set {env_name}; no dependency checkout found at: {expected}')


GIGA_TRAIN_ROOT = _resolve_dependency_root('GIGA_TRAIN_ROOT', 'giga-train')
GIGA_DATASETS_ROOT = _resolve_dependency_root(
    'GIGA_DATASETS_ROOT', 'giga-datasets', 'giga-datasets-v3.0'
)
os.environ['GIGA_TRAIN_ROOT'] = os.fspath(GIGA_TRAIN_ROOT)
os.environ['GIGA_DATASETS_ROOT'] = os.fspath(GIGA_DATASETS_ROOT)
BOOTSTRAP_PATHS = (
    REPO_ROOT,
    GIGABRAIN07_ROOT,
    GIGA_TRAIN_ROOT,
    GIGA_DATASETS_ROOT,
)
for _path in reversed(BOOTSTRAP_PATHS):
    _path_text = os.fspath(_path)
    if _path_text not in sys.path:
        sys.path.insert(0, _path_text)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# Disable torch.compile completely to avoid long JIT compilation overhead.
torch.compile = lambda f=None, *args, **kwargs: f if f is not None else (lambda x: x)
torch._dynamo.config.disable = True

# Import giga_models BEFORE matplotlib: matplotlib loads native libs that conflict with
# torchao's cpp-extension probe, segfaulting if matplotlib is imported first. Pulling in
# the giga modeling module here (after the torch.compile patch) settles the load order.
from giga_models import GigaBrain07Policy  # noqa: F401,E402


def parse_args():
    parser = argparse.ArgumentParser(description='PaliGemma2 flow rollout inference')
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--data-path', type=str, default=None,
                        help='LeRobot dataset root. If omitted, use the first dataset in the training config.')
    parser.add_argument('--norm-stats-path', type=str, default=None,
                        help='Override norm stats path. If omitted, use the training/inference config.')
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--config-path', type=str, default=None,
                        help='Training config.json, experiment dir, or Python config. If omitted, search upward.')
    parser.add_argument('--checkpoint-subdir', type=str, default='model_ema',
                        choices=['model_ema', 'model', 'auto'],
                        help='Diffusers subdir under checkpoint root. Default evaluates EMA weights.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float32'])
    parser.add_argument('--sampler-backend', type=str, default='sample_actions',
                        choices=['sample_actions', 'train_forward'],
                        help='sample_actions uses policy.sample_actions(); train_forward denoises via the training forward path.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set torch/NumPy RNG seed before rollout so sampled flow noise is reproducible.')
    parser.add_argument(
        '--prompt-mode',
        type=str,
        default='main-task',
        choices=['main-task', 'subtask'],
        help=(
            'Condition action prediction on the main task or on the current subtask. '
            'Subtask mode requires dataset task strings containing " Subtask:".'
        ),
    )
    parser.add_argument('--action-chunk', type=int, default=None,
                        help='Frames to advance per chunk. Defaults to policy.n_action_steps.')
    parser.add_argument('--num-rollouts', type=int, default=-1,
                        help='Number of chunks per episode. Use -1 for all chunks.')
    parser.add_argument('--episode-idx', type=int, default=0)
    parser.add_argument('--num-episodes', type=int, default=3,
                        help='Number of episodes. Use -1 for all episodes from episode-idx.')
    parser.add_argument('--embodiment-id', type=int, default=None,
                        help='Override embodiment id. By default it is inferred from robot_type.')
    parser.add_argument('--robot-type', type=str, default=None,
                        help='Override robot_type for robot-type delta masks.')
    parser.add_argument('--original-action-dim', type=int, default=None,
                        help=(
                            'Optional leading action-dimension cap. By default the dimension is inferred '
                            'from the training action supervision mask. This never changes state input width.'
                        ))
    parser.add_argument('--plot-action-dim', type=int, default=None,
                        help='Leading action dims to plot/score. Defaults to original-action-dim.')
    parser.add_argument(
        '--zero-noise-outside-action-dim',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Keep the denoising latent zero after original-action-dim on every flow step '
            '(default: follow delta_action_cfg.mask_unsupervised_action_dims_for_noise; '
            'use this option or its --no- form to override the config).'
        ),
    )
    parser.add_argument('--plot-backend', type=str, default='auto', choices=['auto', 'matplotlib', 'pillow'],
                        help='Use pillow to avoid matplotlib/NumPy ABI issues.')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip rollout PNG generation and write metrics only.')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip episodes with an existing result and rollout image.')
    parser.add_argument('--use-predicted-action', action='store_true',
                        help='Use predicted action as next state instead of GT state.')
    parser.add_argument('--save-replay', action='store_true',
                        help='Dump predicted/GT trajectories under episode_XXX_replay/.')
    return parser.parse_args()


def _read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def _to_plain_config(value: Any) -> Any:
    if hasattr(value, 'to_dict'):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _to_plain_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_plain_config(item) for item in value)
    return value


def _load_train_config_file(path: Path) -> dict[str, Any]:
    if path.suffix == '.py':
        path_parent = os.fspath(path.parent)
        if path_parent not in sys.path:
            sys.path.insert(0, path_parent)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Failed to load train config from {path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, 'config'):
            raise AttributeError(f'{path} does not define a top-level config variable.')
        return _to_plain_config(module.config)

    if path.suffix == '.json':
        return _read_json(path)

    from giga_train import load_config

    return _to_plain_config(load_config(os.fspath(path)))


def find_project_config(checkpoint_path: str, config_path: str | None = None) -> tuple[dict[str, Any], str | None]:
    """Find full training config: explicit path first, then search upward."""
    candidates: list[Path] = []
    if config_path:
        explicit = Path(config_path).expanduser().resolve()
        candidates.append(explicit if explicit.is_file() else explicit / 'config.json')

    p = Path(checkpoint_path).expanduser().resolve()
    if p.is_file():
        p = p.parent
    for current in (p, *p.parents):
        candidates.append(current / 'config.json')

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists():
            continue
        cfg = _load_train_config_file(candidate)
        if 'dataloaders' in cfg or 'models' in cfg:
            print(f'  Loaded project config: {candidate}')
            return cfg, os.fspath(candidate)

    print('  WARNING: No full project config found; falling back to checkpoint-only config')
    return {}, None


def _resolve_ckpt_dir(checkpoint_path: str, checkpoint_subdir: str) -> str:
    root = Path(checkpoint_path).expanduser().resolve()
    if (root / 'config.json').exists() and (root / 'diffusion_pytorch_model.bin').exists():
        return os.fspath(root)

    subdirs = [checkpoint_subdir] if checkpoint_subdir != 'auto' else []
    for subdir in ('model_ema', 'model', ''):
        if subdir not in subdirs:
            subdirs.append(subdir)

    tried = []
    for subdir in subdirs:
        cand = root / subdir if subdir else root
        tried.append(os.fspath(cand))
        if (cand / 'config.json').exists() and (cand / 'diffusion_pytorch_model.bin').exists():
            return os.fspath(cand)
    raise FileNotFoundError(f'No diffusers checkpoint found under {checkpoint_path}. Tried: {tried}')


def _iter_lerobot_dataset_configs(data_or_config: Any):
    data_or_config = _to_plain_config(data_or_config)
    if isinstance(data_or_config, dict) and data_or_config.get('_class_name') == 'WeightedConcatDataset':
        groups = data_or_config.get('datasets', data_or_config.get('data_or_config_list'))
        if not isinstance(groups, list):
            return
        for group in groups:
            group_items = group if isinstance(group, list) else [group]
            for item in group_items:
                yield from _iter_lerobot_dataset_configs(item)
        return

    if isinstance(data_or_config, list):
        for item in data_or_config:
            yield from _iter_lerobot_dataset_configs(item)
        return

    if isinstance(data_or_config, dict) and data_or_config.get('data_path') is not None:
        yield copy.deepcopy(data_or_config)


def _normalize_path(value: str | os.PathLike[str]) -> str:
    return os.path.normpath(os.fspath(value))


def _resolve_dataset_cfg(
    train_cfg: dict[str, Any],
    data_path: str | None,
    action_chunk: int,
) -> tuple[dict[str, Any], str]:
    train_data_cfg = train_cfg.get('dataloaders', {}).get('train', {}).get('data_or_config')
    candidates = list(_iter_lerobot_dataset_configs(train_data_cfg))
    if data_path is None:
        if not candidates:
            raise ValueError('--data-path is required because no LeRobotDataset was found in the training config.')
        dataset_cfg = copy.deepcopy(candidates[0])
        data_path = dataset_cfg['data_path']
    else:
        matches = [
            cfg for cfg in candidates
            if _normalize_path(cfg['data_path']) == _normalize_path(data_path)
        ]
        dataset_cfg = copy.deepcopy(matches[0]) if matches else dict(
            _class_name='LeRobotDataset',
            data_path=data_path,
            meta_name='meta',
        )

    dataset_cfg.setdefault('_class_name', 'LeRobotDataset')
    dataset_cfg.setdefault('meta_name', 'meta')
    delta_info = dict(dataset_cfg.get('delta_info') or {})
    if 'action' in delta_info or not delta_info:
        delta_info['action'] = int(action_chunk)
    elif len(delta_info) == 1:
        only_key = next(iter(delta_info))
        delta_info[only_key] = int(action_chunk)
    dataset_cfg['delta_info'] = delta_info
    dataset_cfg['data_path'] = data_path
    return dataset_cfg, data_path


def _override_norm_stats_path(transform_cfg: dict[str, Any], data_path: str, norm_stats_path: str | None) -> None:
    if not norm_stats_path:
        return
    norm_cfg = transform_cfg.setdefault('norm_cfg', {})
    selector = norm_cfg.get('selector', 'embodiment_id')
    if selector == 'data_path':
        norm_cfg['norm_stats_path'] = [dict(data_paths=[data_path], path=norm_stats_path)]
    elif selector == 'embodiment_id':
        norm_cfg['norm_stats_path'] = {0: norm_stats_path}
    else:
        norm_cfg['norm_stats_path'] = norm_stats_path


def _resolve_norm_stats_path(transform_cfg: dict[str, Any], data_path: str, norm_stats_path: str | None) -> str | None:
    if norm_stats_path:
        return norm_stats_path
    norm_cfg = transform_cfg.get('norm_cfg') or {}
    path_cfg = norm_cfg.get('norm_stats_path')
    if isinstance(path_cfg, str):
        return path_cfg
    if isinstance(path_cfg, dict):
        if len(path_cfg) == 1:
            return os.fspath(next(iter(path_cfg.values())))
        return None
    if isinstance(path_cfg, list):
        for entry in path_cfg:
            path = entry.get('path', entry.get('norm_stats_path'))
            data_paths = entry.get('data_paths', entry.get('selector_values', entry.get('keys')))
            if path is None:
                continue
            if isinstance(data_paths, (str, os.PathLike)):
                data_paths = [data_paths]
            if data_paths is None or any(_normalize_path(p) == _normalize_path(data_path) for p in data_paths):
                return os.fspath(path)
        if len(path_cfg) == 1:
            return os.fspath(path_cfg[0].get('path', path_cfg[0].get('norm_stats_path')))
    return None


def build_transform(
    train_cfg: dict[str, Any],
    data_path: str,
    norm_stats_path: str | None,
    prompt_mode: str = 'main-task',
):
    from gigabrain07 import GigaBrain07Transform

    transform_cfg = copy.deepcopy(train_cfg['dataloaders']['train']['transform'])
    transform_cfg.pop('type', None)
    transform_cfg.pop('is_train', None)
    delta_action_cfg = transform_cfg.get('delta_action_cfg')
    if isinstance(delta_action_cfg, dict):
        # Keep mask_unsupervised_action_dims_for_noise so the inference transform
        # exposes the exact action_dim_loss_mask used by training.
        delta_action_cfg.pop('use_action_result_dim_supervision_mask', None)
    image_cfg = transform_cfg.setdefault('image_cfg', {})
    image_cfg['enable_image_aug'] = False
    prompt_cfg = transform_cfg.setdefault('prompt_cfg', {})
    # With is_train=False, encode_sub_task_input selects the observable prompt:
    # ``Task: <main task>`` when false and ``Subtask: <subtask>`` when true.
    # Action targets remain disabled in both modes.
    prompt_cfg['encode_sub_task_input'] = prompt_mode == 'subtask'
    prompt_cfg['encode_action_input'] = False
    prompt_cfg['sample_ratios'] = None
    _override_norm_stats_path(transform_cfg, data_path, norm_stats_path)
    return GigaBrain07Transform(**transform_cfg, is_train=False), transform_cfg


def load_model(ckpt_dir: str, device: str, dtype: torch.dtype):
    with open(os.path.join(ckpt_dir, 'config.json'), 'r') as f:
        cfg = json.load(f)
    if cfg.get('vlm_type') != 'paligemma2':
        raise ValueError(f'This rollout script expects vlm_type=paligemma2, got {cfg.get("vlm_type")!r}')

    print(f'  Loading checkpoint: {ckpt_dir}')
    policy = GigaBrain07Policy.from_pretrained(ckpt_dir)
    policy.to(dtype=dtype)
    nn.Module.to(policy, device)
    policy.eval()
    param_count = sum(p.numel() for p in policy.parameters()) / 1e9
    print(
        f'  Model loaded: {param_count:.2f}B params on {device}; '
        f'n_action_steps={policy.n_action_steps}, state_input_mode={policy.state_input_mode}'
    )
    return policy, cfg


def _clone_for_transform(data: dict[str, Any]) -> dict[str, Any]:
    cloned = {}
    for key, value in data.items():
        cloned[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return cloned


def _trim_tensor_last_dim(value: Any, dim: int) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    if value.ndim == 0 or int(value.shape[-1]) <= dim:
        return value
    return value[..., :dim].clone()


def _resize_tensor_last_dim(value: Any, dim: int, *, pad_value: float = 0.0) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    if value.ndim == 0:
        return value
    current_dim = int(value.shape[-1])
    if current_dim == dim:
        return value
    if current_dim > dim:
        return value[..., :dim].clone()
    pad_shape = (*value.shape[:-1], dim - current_dim)
    pad = value.new_full(pad_shape, pad_value)
    return torch.cat([value, pad], dim=-1)


def _prepare_sample_dims(data: dict[str, Any], action_dim: int, state_dim: int | None = None) -> dict[str, Any]:
    prepared = _clone_for_transform(data)
    if state_dim is not None:
        state_dim = int(state_dim)
        for key in (
            'observation.state',
            'observation.state_memory',
        ):
            if key in prepared:
                prepared[key] = _resize_tensor_last_dim(
                    prepared[key], state_dim, pad_value=0.0
                )
    if 'action' in prepared:
        prepared['action'] = _resize_tensor_last_dim(prepared['action'], action_dim, pad_value=0.0)
    if (
        'action_is_pad' in prepared
        and isinstance(prepared['action_is_pad'], torch.Tensor)
        and prepared['action_is_pad'].ndim >= 2
    ):
        prepared['action_is_pad'] = _trim_tensor_last_dim(
            prepared['action_is_pad'], action_dim
        )
    if state_dim is not None:
        for key in ('observation.state_is_pad', 'observation.state_memory_is_pad'):
            if key in prepared and isinstance(prepared[key], torch.Tensor) and prepared[key].ndim >= 2:
                prepared[key] = _trim_tensor_last_dim(prepared[key], state_dim)
    return prepared


def _prepare_raw_sample_dims(
    data: dict[str, Any],
    *,
    action_dim: int,
    representation_changes_action_dim: bool,
) -> dict[str, Any]:
    """Resize only when the raw and training action representations share a width.

    Some transforms expand the raw action representation before normalization. In
    particular, embodiment-3 quaternion actions are 16-D on disk and 20-D after
    quaternion-to-Rotation-6D conversion. Padding those raw actions to the inferred
    20-D training width before the transform corrupts the pose layout.
    """
    if representation_changes_action_dim:
        return _clone_for_transform(data)
    return _prepare_sample_dims(data, action_dim)


def _infer_transformed_raw_action_dim(
    transform: Any,
    data: dict[str, Any],
    *,
    embodiment_id: int,
    raw_action_dim: int,
) -> int:
    """Return the action width immediately after the configured pose conversion."""
    raw_action_dim = int(raw_action_dim)
    if not bool(getattr(transform, 'use_quaternion_to_6d', False)):
        return raw_action_dim

    converter = transform.embodiment3_quaternion_to_6d_transform
    if transform._use_eef_dual_hand_prefix_to_6d(data):
        return int(converter.transform_output_dim_dual_hand_prefix(raw_action_dim))
    if converter.supports_embodiment(embodiment_id):
        return int(converter.transform_dim(raw_action_dim, embodiment_id))
    return raw_action_dim


def _trim_sample_dims(data: dict[str, Any], dim: int) -> dict[str, Any]:
    trimmed = _clone_for_transform(data)
    for key in (
        'observation.state',
        'observation.state_memory',
        'action',
    ):
        if key in trimmed:
            trimmed[key] = _trim_tensor_last_dim(trimmed[key], dim)
    for key in ('action_is_pad', 'observation.state_is_pad', 'observation.state_memory_is_pad'):
        if key in trimmed and isinstance(trimmed[key], torch.Tensor) and trimmed[key].ndim >= 2:
            trimmed[key] = _trim_tensor_last_dim(trimmed[key], dim)
    return trimmed


def _replace_current_state(data: dict[str, Any], state: torch.Tensor) -> None:
    raw_state = data['observation.state']
    state = state.to(dtype=raw_state.dtype, device=raw_state.device)
    copy_dim = min(int(raw_state.shape[-1]), int(state.shape[-1]))
    if raw_state.ndim == 1:
        raw_state = raw_state.clone()
        raw_state[:copy_dim] = state[..., :copy_dim]
        data['observation.state'] = raw_state
    elif raw_state.ndim == 2:
        raw_state = raw_state.clone()
        raw_state[-1, :copy_dim] = state[..., :copy_dim]
        data['observation.state'] = raw_state
    else:
        raise ValueError(f'Unsupported observation.state shape: {tuple(raw_state.shape)}')


def _current_state_tensor(data: dict[str, Any]) -> torch.Tensor:
    state = torch.as_tensor(data['observation.state'])
    if state.ndim == 1:
        return state
    if state.ndim == 2:
        return state[-1]
    raise ValueError(f'Unsupported observation.state shape: {tuple(state.shape)}')


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    item = getattr(value, 'item', None)
    return int(item() if callable(item) else value)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalizer_key(normalizer, key: int | str) -> int | str:
    return normalizer._normalize_stats_key(key)


def _unnormalize_value(normalizer, x: torch.Tensor, norm_key: int | str) -> torch.Tensor:
    stats_key = _normalizer_key(normalizer, norm_key)
    x_dim = x.shape[-1]
    if normalizer.use_quantiles:
        q01 = normalizer.q01[stats_key][..., :x_dim].to(device=x.device, dtype=x.dtype)
        q99 = normalizer.q99[stats_key][..., :x_dim].to(device=x.device, dtype=x.dtype)
        return (x + 1.0) / 2.0 * (q99 - q01 + normalizer.EPSILON) + q01
    mean = normalizer.mean[stats_key][..., :x_dim].to(device=x.device, dtype=x.dtype)
    std = normalizer.std[stats_key][..., :x_dim].to(device=x.device, dtype=x.dtype)
    return x * (std + normalizer.EPSILON) + mean


def _unnormalize_action(transform, x: torch.Tensor, norm_key: int | str) -> torch.Tensor:
    return _unnormalize_value(
        transform.action_normalize_transform,
        x,
        norm_key,
    )


def _unnormalize_state(transform, x: torch.Tensor, norm_key: int | str) -> torch.Tensor:
    return _unnormalize_value(
        transform.state_normalize_transform,
        x,
        norm_key,
    )


def _reconstruct_absolute_action(
    transform,
    normalized_action: torch.Tensor,
    normalized_state: torch.Tensor,
    norm_key: int | str,
    delta_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Undo the training normalization and componentwise delta transform."""
    action_delta = _unnormalize_action(transform, normalized_action, norm_key)
    reference_state = _unnormalize_state(transform, normalized_state, norm_key)
    action_absolute = action_delta.clone()
    mask = torch.as_tensor(
        delta_mask,
        dtype=action_absolute.dtype,
        device=action_absolute.device,
    ).reshape(-1)
    shared_dim = min(
        int(action_absolute.shape[-1]),
        int(reference_state.shape[-1]),
        int(mask.numel()),
    )
    action_absolute[..., :shared_dim] += (
        reference_state[..., :shared_dim] * mask[:shared_dim]
    )
    return action_absolute, action_delta, reference_state


def _resolve_delta_mask(transform, raw_data: dict[str, Any], embodiment_id: int) -> tuple[torch.Tensor, str, str]:
    selector = getattr(transform, 'delta_mask_selector', 'embodiment_id')
    if getattr(transform, 'use_delta_joint_actions', False):
        mask_key = transform._get_delta_mask_selector(raw_data, embodiment_id)
        mask = transform.delta_action_transform.mask[mask_key]
    else:
        mask_key = embodiment_id
        mask = torch.zeros(raw_data['action'].shape[-1], dtype=torch.bool)
    effective = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
    return effective, str(mask_key), str(selector)


def _active_prefix_dim(mask: torch.Tensor) -> int:
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if mask.ndim == 0:
        return int(bool(mask.item()))
    active_by_dim = mask.reshape(-1, mask.shape[-1]).any(dim=0)
    active_indices = active_by_dim.nonzero(as_tuple=False).flatten()
    return 0 if active_indices.numel() == 0 else int(active_indices[-1].item()) + 1


def _resolve_rollout_action_dim(
    first_batch: dict[str, Any],
    *,
    configured_action_dim: int,
    requested_action_dim: int | None,
) -> tuple[int, int]:
    action_dim_loss_mask = first_batch.get('action_dim_loss_mask')
    supervised_action_dim = (
        0
        if action_dim_loss_mask is None
        else _active_prefix_dim(action_dim_loss_mask)
    )
    if supervised_action_dim <= 0:
        supervised_action_dim = int(configured_action_dim)

    supervised_action_dim = min(
        int(supervised_action_dim), int(configured_action_dim)
    )
    if supervised_action_dim <= 0:
        raise ValueError('Could not infer a positive training action dimension.')

    if requested_action_dim is None:
        return supervised_action_dim, supervised_action_dim
    if int(requested_action_dim) <= 0:
        raise ValueError(
            f'--original-action-dim must be positive, got {requested_action_dim}'
        )
    return min(int(requested_action_dim), supervised_action_dim), supervised_action_dim


def _resolve_zero_noise_outside_action_dim(
    requested: bool | None,
    transform_cfg: dict[str, Any],
) -> tuple[bool, bool]:
    delta_action_cfg = transform_cfg.get('delta_action_cfg') or {}
    configured = bool(
        delta_action_cfg.get('mask_unsupervised_action_dims_for_noise', False)
    )
    resolved = configured if requested is None else bool(requested)
    return resolved, configured


def _batch_optional_tensor(batch: dict[str, Any], key: str, device: str) -> torch.Tensor | None:
    value = batch.get(key)
    if value is None:
        return None
    return value.unsqueeze(0).to(device)


@torch.no_grad()
def run_single_inference(
    policy,
    batch: dict[str, Any],
    device: str,
    sampler_backend: str = 'sample_actions',
    zero_noise_after_dim: int | None = None,
) -> torch.Tensor:
    images = [img.unsqueeze(0).to(device) for img in batch['images']]
    img_masks = [mask.unsqueeze(0).to(device) for mask in batch['image_masks']]
    lang_tokens = batch['lang_tokens'].unsqueeze(0).to(device)
    lang_masks = batch['lang_masks'].unsqueeze(0).to(device)
    emb_ids = batch['embodiment_id'].reshape(1).to(device)
    lang_att_masks = _batch_optional_tensor(batch, 'lang_att_masks', device)
    state_memory = _batch_optional_tensor(batch, 'observation.state_memory', device)
    state_memory_masks = _batch_optional_tensor(batch, 'observation.state_memory_mask', device)
    proprioception = _batch_optional_tensor(batch, 'observation.proprioception', device)
    agent_pos_mask = _batch_optional_tensor(batch, 'observation.agent_pos_mask', device)
    proprioception_present = _batch_optional_tensor(
        batch, 'observation.proprioception_present', device
    )
    noise = None
    if zero_noise_after_dim is not None:
        zero_noise_after_dim = int(zero_noise_after_dim)
        if not 0 < zero_noise_after_dim <= int(policy.max_action_dim):
            raise ValueError(
                f'zero_noise_after_dim must be in [1, {policy.max_action_dim}], got {zero_noise_after_dim}'
            )
        noise = policy._prepare_action_noise(None, images[0].shape[0], torch.device(device))
        noise[..., zero_noise_after_dim:] = 0

    if sampler_backend == 'sample_actions':
        original_denoise_step = None
        if zero_noise_after_dim is not None:
            original_denoise_step = policy.denoise_step

            def masked_denoise_step(prefix_pad_masks, past_key_values, x_t, timestep, step_emb_ids):
                x_t = x_t.clone()
                x_t[..., zero_noise_after_dim:] = 0
                v_t = original_denoise_step(
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    timestep,
                    step_emb_ids,
                )
                v_t[..., zero_noise_after_dim:] = 0
                return v_t

            policy.denoise_step = masked_denoise_step
        try:
            actions = policy.sample_actions(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                emb_ids=emb_ids,
                noise=noise,
                lang_att_masks=lang_att_masks,
                state_memory=state_memory,
                state_memory_masks=state_memory_masks,
                proprioception=proprioception,
                agent_pos_mask=agent_pos_mask,
                proprioception_present=proprioception_present,
            )
        finally:
            if original_denoise_step is not None:
                policy.denoise_step = original_denoise_step
        return actions[0].detach().cpu().float()

    if sampler_backend != 'train_forward':
        raise ValueError(f'Unsupported sampler_backend: {sampler_backend!r}')

    bsize = images[0].shape[0]
    fast_action_indicator = _batch_optional_tensor(batch, 'fast_action_indicator', device)
    subtask_indicator = _batch_optional_tensor(batch, 'subtask_indicator', device)
    lang_loss_masks = _batch_optional_tensor(batch, 'lang_loss_masks', device)
    state = _batch_optional_tensor(batch, 'observation.state', device)
    image_grid_thw = batch.get('image_grid_thw')
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)

    if noise is None:
        noise = policy._prepare_action_noise(None, bsize, torch.device(device))
    x_t = noise.to(dtype=policy.action_in_proj.weight.dtype)
    dt = -1.0 / policy.num_steps
    timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)
    for timestep in timesteps:
        model_pred = policy(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            x_t,
            timestep.expand(bsize),
            emb_ids,
            lang_att_masks=lang_att_masks,
            fast_action_indicator=fast_action_indicator,
            subtask_indicator=subtask_indicator,
            image_grid_thw=image_grid_thw,
            lang_loss_masks=lang_loss_masks,
            state_memory=state_memory,
            state_memory_masks=state_memory_masks,
            proprioception=proprioception,
            agent_pos_mask=agent_pos_mask,
            proprioception_present=proprioception_present,
            state=state,
        )
        x_t = x_t + dt * model_pred['v_t'].to(dtype=x_t.dtype)
        if zero_noise_after_dim is not None:
            x_t[..., zero_noise_after_dim:] = 0
    return x_t[0].detach().cpu().float()


def build_episode_index_map(data_path: str):
    episodes_file = os.path.join(data_path, 'meta', 'episodes.jsonl')
    if not os.path.exists(episodes_file):
        return build_episode_index_map_from_parquet(data_path)
    episode_map = {}
    cumsum = 0
    with open(episodes_file, 'r') as f:
        for ep_idx, line in enumerate(f):
            ep = json.loads(line.strip())
            length = int(ep['length'])
            episode_map[ep_idx] = list(range(cumsum, cumsum + length))
            cumsum += length
    return episode_map


def build_episode_index_map_from_parquet(data_path: str):
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        print(f'  WARNING: pyarrow unavailable for parquet episode map ({exc}); falling back to dataset scan.')
        return None

    root = Path(data_path)
    parquet_files = sorted((root / 'data').glob('chunk-*/*.parquet'))
    if not parquet_files:
        parquet_files = sorted((root / 'data').glob('*.parquet'))
    if not parquet_files:
        return None

    episode_map: dict[int, list[int]] = {}
    row_offset = 0
    try:
        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file, columns=['episode_index'])
            episode_values = table.column('episode_index').to_pylist()
            for local_idx, episode_idx in enumerate(episode_values):
                episode_map.setdefault(int(episode_idx), []).append(row_offset + local_idx)
            row_offset += len(episode_values)
    except Exception as exc:
        print(f'  WARNING: failed to build parquet episode map ({exc}); falling back to dataset scan.')
        return None
    return episode_map


def find_episode_frames(dataset, episode_idx: int, episode_map=None):
    if episode_map is not None and episode_idx in episode_map:
        return episode_map[episode_idx]
    indices = []
    for i in range(len(dataset)):
        d = dataset[i]
        sample_ep = _as_int(d['episode_index'])
        if sample_ep == episode_idx:
            indices.append(i)
        elif sample_ep > episode_idx:
            break
    return indices


def load_existing_results(output_path: str) -> dict[int, dict[str, Any]]:
    results_by_episode: dict[int, dict[str, Any]] = {}
    jsonl_path = os.path.join(output_path, 'results.jsonl')
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    results_by_episode[int(item['episode_idx'])] = item
        return results_by_episode

    json_path = os.path.join(output_path, 'results.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            for item in json.load(f):
                results_by_episode[int(item['episode_idx'])] = item
    return results_by_episode


def append_episode_result(output_path: str, result: dict[str, Any]) -> None:
    with open(os.path.join(output_path, 'results.jsonl'), 'a') as f:
        f.write(json.dumps(result) + '\n')


def visualize_rollout(
    gt_actions,
    pred_actions,
    task_text,
    episode_idx,
    out_path,
    action_dim,
    action_chunk,
    closed_loop=False,
    dim_names=None,
    backend='auto',
):
    min_steps = min(gt_actions.shape[0], pred_actions.shape[0])
    gt = gt_actions[:min_steps, :action_dim]
    pred = pred_actions[:min_steps, :action_dim]

    if backend != 'pillow':
        try:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(action_dim, 1, figsize=(16, 2 * action_dim))
            if action_dim == 1:
                axs = [axs]
            time_axis = np.arange(min_steps)
            for dim in range(action_dim):
                ax = axs[dim]
                ax.plot(time_axis, gt[:, dim], label='GT', linewidth=2, color='blue')
                ax.plot(time_axis, pred[:, dim], label='Pred', linewidth=2, color='red', linestyle='--')
                for c in range(action_chunk, min_steps, action_chunk):
                    ax.axvline(x=c, color='gray', linestyle=':', alpha=0.5)
                label = dim_names[dim] if dim_names and dim < len(dim_names) else f'J{dim}'
                ax.set_ylabel(label)
                if dim == 0:
                    ax.legend(loc='upper right', fontsize=7)
                ax.grid(True, alpha=0.3)
            mode_tag = ' [PREDICTED-ACTION]' if closed_loop else ' [OPEN-LOOP]'
            axs[0].set_title(f'Episode {episode_idx}: {task_text}{mode_tag} (steps={min_steps}, chunk={action_chunk})')
            axs[-1].set_xlabel('Timestep')
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f'  Saved: {out_path}')
            return
        except Exception as exc:
            if backend == 'matplotlib':
                raise
            print(f'  WARNING: matplotlib visualization failed ({exc}); using Pillow fallback.')

    visualize_rollout_with_pillow(gt, pred, task_text, episode_idx, out_path, action_chunk, closed_loop, dim_names)
    print(f'  Saved: {out_path}')


def visualize_rollout_with_pillow(gt, pred, task_text, episode_idx, out_path, action_chunk, closed_loop, dim_names=None):
    from PIL import Image, ImageDraw, ImageFont

    action_dim = pred.shape[1]
    min_steps = pred.shape[0]
    width = 1800
    row_h = 120
    left = 190
    right = 30
    top = 70
    bottom = 40
    height = top + action_dim * row_h + bottom
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    mode_tag = 'PREDICTED-ACTION' if closed_loop else 'OPEN-LOOP'
    title = f'Episode {episode_idx}: {task_text} [{mode_tag}] steps={min_steps}, chunk={action_chunk}'
    draw.text((left, 15), title[:220], fill=(0, 0, 0), font=font)
    draw.line((width - 250, 30, width - 210, 30), fill=(35, 90, 210), width=3)
    draw.text((width - 205, 24), 'GT', fill=(0, 0, 0), font=font)
    draw.line((width - 150, 30, width - 110, 30), fill=(220, 40, 40), width=3)
    draw.text((width - 105, 24), 'Pred', fill=(0, 0, 0), font=font)

    plot_w = width - left - right
    for dim in range(action_dim):
        y0 = top + dim * row_h
        y1 = y0 + row_h - 24
        label = dim_names[dim] if dim_names and dim < len(dim_names) else f'J{dim}'
        draw.text((10, y0 + 35), label[:24], fill=(0, 0, 0), font=font)

        values = np.concatenate([gt[:, dim], pred[:, dim]])
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = float(finite.min()), float(finite.max())
            if abs(vmax - vmin) < 1e-6:
                vmin -= 1.0
                vmax += 1.0
        pad = 0.05 * (vmax - vmin)
        vmin -= pad
        vmax += pad

        draw.rectangle((left, y0, width - right, y1), outline=(210, 210, 210))
        for frac in (0.25, 0.5, 0.75):
            yy = int(y0 + frac * (y1 - y0))
            draw.line((left, yy, width - right, yy), fill=(235, 235, 235))
        for c in range(action_chunk, min_steps, action_chunk):
            x = left + int(c / max(min_steps - 1, 1) * plot_w)
            draw.line((x, y0, x, y1), fill=(190, 190, 190))

        def to_points(series):
            pts = []
            for i, value in enumerate(series):
                x = left + int(i / max(min_steps - 1, 1) * plot_w)
                y = y1 - int((float(value) - vmin) / max(vmax - vmin, 1e-6) * (y1 - y0))
                pts.append((x, y))
            return pts

        gt_pts = to_points(gt[:, dim])
        pred_pts = to_points(pred[:, dim])
        if len(gt_pts) > 1:
            draw.line(gt_pts, fill=(35, 90, 210), width=3)
        if len(pred_pts) > 1:
            draw.line(pred_pts, fill=(220, 40, 40), width=3)
        draw.text((left, y1 + 4), f'{vmin:.3f}', fill=(80, 80, 80), font=font)
        draw.text((width - right - 80, y1 + 4), f'{vmax:.3f}', fill=(80, 80, 80), font=font)

    img.save(out_path)


def _feature_names(sample: dict[str, Any], key: str) -> list[str] | None:
    meta = sample.get('meta')
    info = getattr(meta, 'info', {}) if meta is not None else {}
    feature = info.get('features', {}).get(key, {}) if isinstance(info, dict) else {}
    names = feature.get('names')
    return None if not names else [str(name) for name in names]


def main():
    args = parse_args()
    device = args.device
    if device.startswith('cuda'):
        torch.cuda.set_device(torch.device(device))
    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed) % (2**32 - 1))
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32
    os.makedirs(args.output_path, exist_ok=True)

    print('=' * 60)
    mode_str = 'PREDICTED-ACTION (predicted state)' if args.use_predicted_action else 'OPEN-LOOP (GT state)'
    print(f'PaliGemma2 Flow Matching Rollout Inference [{mode_str}]')
    print('=' * 60)

    ckpt_dir = _resolve_ckpt_dir(args.checkpoint_path, args.checkpoint_subdir)
    train_cfg, train_cfg_path = find_project_config(args.checkpoint_path, args.config_path)

    print('\n[1/4] Loading model...')
    policy, model_cfg = load_model(ckpt_dir, device, dtype=dtype)
    policy_steps = int(policy.n_action_steps)
    if args.action_chunk is None:
        args.action_chunk = policy_steps
    elif int(args.action_chunk) != policy_steps:
        print(
            f'  WARNING: --action-chunk={args.action_chunk} differs from '
            f'policy.n_action_steps={policy_steps}; predictions are truncated for metrics.'
        )

    if not train_cfg:
        raise ValueError('A full training config is required to rebuild the rollout transform.')

    dataset_cfg, data_path = _resolve_dataset_cfg(train_cfg, args.data_path, args.action_chunk)

    print('\n[2/4] Building transform and dataset...')
    transform, transform_cfg = build_transform(
        train_cfg,
        data_path,
        args.norm_stats_path,
        prompt_mode=args.prompt_mode,
    )
    zero_noise_outside_action_dim, configured_noise_mask = (
        _resolve_zero_noise_outside_action_dim(
            args.zero_noise_outside_action_dim,
            transform_cfg,
        )
    )
    resolved_norm_stats = _resolve_norm_stats_path(transform_cfg, data_path, args.norm_stats_path)
    print(f'  data_path: {data_path}')
    print(f'  norm_stats_path: {resolved_norm_stats}')
    print(f'  state_input_mode: {transform.state_input_mode}')
    print(f'  prompt_mode: {args.prompt_mode}')
    print(f'  action_chunk={args.action_chunk}, num_rollouts={args.num_rollouts}')

    from giga_datasets import load_dataset

    dataset = load_dataset([dataset_cfg])
    print(f'  Dataset size: {len(dataset)}')
    first_sample_raw = dataset[0]
    raw_state_dim = int(torch.as_tensor(first_sample_raw['observation.state']).shape[-1])
    raw_action_dim = int(torch.as_tensor(first_sample_raw['action']).shape[-1])
    first_sample = _clone_for_transform(first_sample_raw)
    if args.robot_type is not None:
        first_sample = dict(first_sample)
        first_sample['robot_type'] = args.robot_type

    first_for_transform = _clone_for_transform(first_sample)
    if args.embodiment_id is not None:
        first_for_transform['embodiment_id'] = args.embodiment_id
    if args.robot_type is not None:
        first_for_transform['robot_type'] = args.robot_type
    first_batch = transform(first_for_transform)
    inferred_embodiment_id = _as_int(first_batch['embodiment_id'])
    robot_type = args.robot_type
    if robot_type is None:
        meta = first_sample.get('meta')
        robot_type = first_sample.get('robot_type') or getattr(meta, 'info', {}).get('robot_type')
    raw_selector_sample = dict(first_sample)
    raw_selector_sample['embodiment_id'] = inferred_embodiment_id
    if robot_type is not None:
        raw_selector_sample['robot_type'] = str(robot_type)
    norm_selector = transform._get_norm_selector(raw_selector_sample, inferred_embodiment_id)
    delta_mask, delta_mask_key, delta_mask_selector = _resolve_delta_mask(transform, raw_selector_sample, inferred_embodiment_id)

    configured_action_dim = min(
        int(delta_mask.shape[-1]), int(policy.max_action_dim)
    )
    original_action_dim, supervised_action_dim = _resolve_rollout_action_dim(
        first_batch,
        configured_action_dim=configured_action_dim,
        requested_action_dim=args.original_action_dim,
    )
    if (
        args.original_action_dim is not None
        and int(args.original_action_dim) > supervised_action_dim
    ):
        print(
            f'  WARNING: --original-action-dim={args.original_action_dim} exceeds '
            f'the training-supervised width {supervised_action_dim}; clipping to '
            f'{original_action_dim}.'
        )
    plot_action_dim = args.plot_action_dim if args.plot_action_dim is not None else original_action_dim
    plot_action_dim = min(int(plot_action_dim), original_action_dim)
    delta_mask = delta_mask[:original_action_dim]
    dim_names = _feature_names(first_sample, 'action') or _feature_names(first_sample, 'observation.state')
    transformed_raw_action_dim = _infer_transformed_raw_action_dim(
        transform,
        raw_selector_sample,
        embodiment_id=inferred_embodiment_id,
        raw_action_dim=raw_action_dim,
    )
    representation_changes_action_dim = raw_action_dim != transformed_raw_action_dim
    if representation_changes_action_dim:
        dim_names = None
    if args.use_predicted_action and representation_changes_action_dim:
        raise ValueError(
            '--use-predicted-action is not supported when the training transform '
            f'changes action width ({raw_action_dim}D raw -> '
            f'{transformed_raw_action_dim}D transformed).'
        )
    is_robot_moving = transform._get_is_robot_moving(raw_selector_sample)
    is_body_moving = (
        False
        if is_robot_moving
        else transform._get_is_body_moving(raw_selector_sample)
    )

    print(f'  robot_type: {robot_type}')
    print(
        f'  is_robot_moving={is_robot_moving}, '
        f'is_body_moving={is_body_moving}'
    )
    print(f'  embodiment_id: {inferred_embodiment_id}')
    print(f'  norm selector/key: {transform.norm_selector}/{norm_selector}')
    print(f'  delta_mask selector/key: {delta_mask_selector}/{delta_mask_key}')
    print(f'  raw_state_dim={raw_state_dim}, raw_action_dim={raw_action_dim}')
    print(
        f'  state_input_dim={raw_state_dim}, '
        f'training_action_dim={supervised_action_dim}, '
        f'original_action_dim={original_action_dim}, '
        f'plot_action_dim={plot_action_dim}'
    )
    print(
        '  action representation width: '
        f'raw={raw_action_dim}, transformed={transformed_raw_action_dim}, '
        f'supervised={supervised_action_dim}, '
        f'changes_dim={representation_changes_action_dim}'
    )
    denoise_action_dim = original_action_dim if zero_noise_outside_action_dim else int(policy.max_action_dim)
    noise_mask_source = 'config' if args.zero_noise_outside_action_dim is None else 'cli'
    print(
        f'  mask_unsupervised_action_dims_for_noise={configured_noise_mask}, '
        f'zero_noise_outside_action_dim={zero_noise_outside_action_dim} '
        f'(source={noise_mask_source}), '
        f'denoise_action_dim={denoise_action_dim}'
    )
    print('\n[3/4] Building episode map...')
    episode_map = build_episode_index_map(data_path)
    if episode_map is not None:
        print(f'  Built fast episode index map for {len(episode_map)} episodes')
    if args.num_episodes == -1:
        if episode_map is None:
            raise ValueError('--num-episodes=-1 requires a metadata or parquet episode map.')
        episode_indices = [idx for idx in sorted(episode_map) if idx >= args.episode_idx]
    else:
        episode_indices = [args.episode_idx + offset for offset in range(args.num_episodes)]

    # Model construction can consume a config-dependent amount of RNG state.
    # Reset here so --seed controls rollout noise consistently across checkpoints.
    if args.seed is not None:
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed) % (2**32 - 1))
        print(f'  Reset rollout RNG seed to {args.seed}')

    existing_results = load_existing_results(args.output_path) if args.skip_existing else {}
    all_results = [existing_results[ep] for ep in sorted(existing_results) if ep in episode_indices]

    print(f'\n[4/4] Running rollout on {len(episode_indices)} episodes...')
    if args.skip_existing and existing_results:
        print(f'  skip_existing enabled: {len(existing_results)} existing result rows')

    for ep_progress, ep_idx in enumerate(episode_indices, start=1):
        rollout_png = os.path.join(args.output_path, f'episode_{ep_idx:03d}_rollout.png')
        if args.skip_existing and ep_idx in existing_results and (args.no_plot or os.path.exists(rollout_png)):
            print(f'\n  Episode {ep_idx} ({ep_progress}/{len(episode_indices)}): already complete, skipping')
            continue

        print(f'\n  Episode {ep_idx} ({ep_progress}/{len(episode_indices)}):')
        ep_frames = find_episode_frames(dataset, ep_idx, episode_map=episode_map)
        if not ep_frames:
            print('    No frames found, skipping')
            continue

        max_rollouts = int(math.ceil(len(ep_frames) / args.action_chunk))
        actual_rollouts = max_rollouts if args.num_rollouts < 0 else min(int(args.num_rollouts), max_rollouts)
        pred_chunks = []
        gt_chunks = []
        chunk_mses = []
        prev_state = None
        initial_state = None
        first_task = None
        last_task = None

        for r in range(actual_rollouts):
            start_frame = r * args.action_chunk
            remaining = len(ep_frames) - start_frame
            if remaining <= 0:
                break
            compare_len = min(args.action_chunk, remaining, policy_steps)
            raw_data = _prepare_raw_sample_dims(
                dataset[ep_frames[start_frame]],
                action_dim=original_action_dim,
                representation_changes_action_dim=representation_changes_action_dim,
            )
            if args.prompt_mode == 'subtask' and ' subtask:' not in str(raw_data.get('task', '')).lower():
                raise ValueError(
                    f'--prompt-mode=subtask requires a subtask label, but episode {ep_idx} '
                    f'frame {start_frame} has task={raw_data.get("task")!r}'
                )
            if args.robot_type is not None:
                raw_data['robot_type'] = args.robot_type
            if args.embodiment_id is not None:
                raw_data['embodiment_id'] = args.embodiment_id

            model_data = _clone_for_transform(raw_data)
            if args.use_predicted_action and prev_state is not None:
                _replace_current_state(model_data, prev_state)
            current_state_raw = _current_state_tensor(model_data)
            if r == 0:
                initial_state = _tensor_to_numpy(current_state_raw)
            batch = transform(model_data)
            normalized_reference_state = batch['observation.state'][:original_action_dim]
            pred_normed = run_single_inference(
                policy,
                batch,
                device,
                sampler_backend=args.sampler_backend,
                zero_noise_after_dim=(original_action_dim if zero_noise_outside_action_dim else None),
            )
            pred_normed = pred_normed[:compare_len, :original_action_dim]
            pred_abs, pred_delta, action_reference_state = _reconstruct_absolute_action(
                transform,
                pred_normed,
                normalized_reference_state,
                norm_selector,
                delta_mask,
            )

            gt_normed = batch['action'][:compare_len, :original_action_dim]
            gt_abs, _, _ = _reconstruct_absolute_action(
                transform,
                gt_normed,
                normalized_reference_state,
                norm_selector,
                delta_mask,
            )

            if args.use_predicted_action:
                prev_state = pred_abs[-1].detach()

            gt_action = _tensor_to_numpy(gt_abs)
            pred_np = pred_abs.numpy()
            pred_chunks.append(pred_np)
            gt_chunks.append(gt_action)
            mse = ((pred_np[:, :plot_action_dim] - gt_action[:, :plot_action_dim]) ** 2).mean()
            chunk_mses.append(float(mse))
            task = str(raw_data.get('task', ''))
            last_task = task
            if first_task is None:
                first_task = task

        if not pred_chunks:
            print('    No chunks produced, skipping')
            continue

        pred_full = np.concatenate(pred_chunks, axis=0)
        gt_full = np.concatenate(gt_chunks, axis=0)
        total_steps = pred_full.shape[0]
        pred_eval = pred_full[:, :plot_action_dim]
        gt_eval = gt_full[:total_steps, :plot_action_dim]
        overall_mse = float(((pred_eval - gt_eval) ** 2).mean())
        overall_mae = float(np.abs(pred_eval - gt_eval).mean())

        print(f'    Episode frames: {len(ep_frames)}, rollout steps: {total_steps}, chunks: {len(pred_chunks)}')
        print(f'    Per-chunk MSE@{plot_action_dim}d: {[f"{m:.4f}" for m in chunk_mses]}')
        print(f'    Overall MAE@{plot_action_dim}d: {overall_mae:.4f}')
        print(f'    Overall MSE@{plot_action_dim}d: {overall_mse:.4f}')

        if not args.no_plot:
            visualize_rollout(
                gt_full,
                pred_full,
                first_task or last_task or '',
                ep_idx,
                rollout_png,
                action_dim=plot_action_dim,
                action_chunk=args.action_chunk,
                closed_loop=args.use_predicted_action,
                dim_names=dim_names,
                backend=args.plot_backend,
            )

        if args.save_replay:
            replay_dir = os.path.join(args.output_path, f'episode_{ep_idx:03d}_replay')
            os.makedirs(replay_dir, exist_ok=True)
            np.save(os.path.join(replay_dir, 'action_trajectory.npy'), pred_full.astype(np.float64))
            np.save(os.path.join(replay_dir, 'gt_trajectory.npy'), gt_full[:total_steps].astype(np.float64))
            if initial_state is not None:
                np.save(os.path.join(replay_dir, 'initial_state.npy'), initial_state.astype(np.float64))
            with open(os.path.join(replay_dir, 'episode_meta.json'), 'w') as f:
                json.dump({
                    'episode_idx': int(ep_idx),
                    'task_first': first_task,
                    'task_last': last_task,
                    'action_chunk': int(args.action_chunk),
                    'num_rollouts': int(len(pred_chunks)),
                    'episode_frames': int(len(ep_frames)),
                    'total_steps': int(total_steps),
                    'action_dim': int(pred_full.shape[1]),
                    'raw_action_dim': int(raw_action_dim),
                    'transformed_raw_action_dim': int(transformed_raw_action_dim),
                    'training_action_dim': int(supervised_action_dim),
                    'representation_changes_action_dim': representation_changes_action_dim,
                    'plot_action_dim': int(plot_action_dim),
                    'zero_noise_outside_action_dim': zero_noise_outside_action_dim,
                    'zero_noise_outside_action_dim_source': noise_mask_source,
                    'denoise_action_dim': int(denoise_action_dim),
                    'use_predicted_action': bool(args.use_predicted_action),
                    'checkpoint_path': args.checkpoint_path,
                    'checkpoint_dir': ckpt_dir,
                    'train_config_path': train_cfg_path,
                    'data_path': data_path,
                    'norm_stats_path': resolved_norm_stats,
                    'vlm_type': model_cfg.get('vlm_type'),
                    'state_input_mode': model_cfg.get('state_input_mode'),
                    'sampler_backend': args.sampler_backend,
                    'seed': args.seed,
                    'prompt_mode': args.prompt_mode,
                    'robot_type': robot_type,
                    'embodiment_id': int(inferred_embodiment_id),
                    'delta_mask_selector': delta_mask_selector,
                    'delta_mask_key': delta_mask_key,
                    'delta_mask': [bool(x) for x in delta_mask.tolist()],
                }, f, indent=2)
            print(f'    [replay] Saved trajectory to {replay_dir}')

        result = {
            'episode_idx': int(ep_idx),
            'num_rollouts': int(len(pred_chunks)),
            'episode_frames': int(len(ep_frames)),
            'total_steps': int(total_steps),
            'per_chunk_mse': chunk_mses,
            'overall_mae': overall_mae,
            'overall_mse': overall_mse,
            'metric_action_dim': int(plot_action_dim),
            'raw_action_dim': int(raw_action_dim),
            'transformed_raw_action_dim': int(transformed_raw_action_dim),
            'training_action_dim': int(supervised_action_dim),
            'representation_changes_action_dim': representation_changes_action_dim,
            'zero_noise_outside_action_dim': zero_noise_outside_action_dim,
            'zero_noise_outside_action_dim_source': noise_mask_source,
            'denoise_action_dim': int(denoise_action_dim),
            'checkpoint_dir': ckpt_dir,
            'state_input_mode': model_cfg.get('state_input_mode'),
            'sampler_backend': args.sampler_backend,
            'seed': args.seed,
            'prompt_mode': args.prompt_mode,
        }
        all_results.append(result)
        append_episode_result(args.output_path, result)
        with open(os.path.join(args.output_path, 'results.json'), 'w') as f:
            json.dump(sorted(all_results, key=lambda item: int(item['episode_idx'])), f, indent=2)

    print(f'\n{"=" * 60}')
    if all_results:
        avg_mse = float(np.mean([r['overall_mse'] for r in all_results]))
        avg_mae = float(np.mean([r['overall_mae'] for r in all_results]))
        metric_dim = all_results[0].get('metric_action_dim')
        print(f'Avg overall MAE@{metric_dim}d across episodes: {avg_mae:.4f}')
        print(f'Avg overall MSE@{metric_dim}d across episodes: {avg_mse:.4f}')
    print(f'{len(all_results)} episodes completed')
    print(f'Results saved to: {args.output_path}')
    with open(os.path.join(args.output_path, 'results.json'), 'w') as f:
        json.dump(sorted(all_results, key=lambda item: int(item['episode_idx'])), f, indent=2)
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
