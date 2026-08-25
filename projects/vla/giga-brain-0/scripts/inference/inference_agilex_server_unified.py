"""PaliGemma2-only VLA Phase 3 flow-matching inference server.

The repository's ``gemma2`` VLA implementation is the PaliGemma2 path
(``vlm_type='paligemma2'``). This server intentionally accepts only that path.

必填参数:
    --model-path        ckpt 目录 (含 config.json + diffusion_pytorch_model.bin,
                        或其上一级含 model_ema/ 或 model/ 子目录)

可选参数:
    --pretrained-path   PaliGemma2 HuggingFace/tokenizer 目录. 不传时使用
                        PALIGEMMA2_DEFAULT_PRETRAINED_PATH.

用法:
    # 默认 bf16, PaliGemma2 pretrained_path 走默认
    python inference_agilex_server_unified.py --model-path /path/to/ckpt

    # 覆盖默认 pretrained_path
    python inference_agilex_server_unified.py \
        --model-path /path/to/ckpt \
        --pretrained-path /path/to/custom_hf_dir

    # fp32 推理 (对齐 fp32 训练 ckpt; 显存翻倍)
    python inference_agilex_server_unified.py --model-path /path/to/ckpt --no-use-bf16

    # 保存每步推理 IO
    python inference_agilex_server_unified.py --model-path /path/to/ckpt --save-dir /path/to/io_dump

    # gb1_pg2_ruev_0707_coarse_emb_cycle.py 的 h01_robot_16d 完整参数
    bash h01_client/run_h01_server_16d.sh

要求 ckpt 旁边有 inference_config.json sidecar (老 PG2 ckpt 可能没有, 需要手写一份).
"""

import json
import os
import time
import types
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import tyro

from giga_models.sockets import RobotInferenceServer

_SAVE_COUNTER = 0

PALIGEMMA2_VLM_TYPE = 'paligemma2'
PALIGEMMA2_DEFAULT_PRETRAINED_PATH = (
    '/home/agilex-home/agilex/models/huggingface/models--google--paligemma2-3b-mix-224'
)


def save_inference_inputs(
    save_dir,
    images,
    state,
    task,
    lang_tokens,
    lang_masks,
    images_batched,
    img_masks_batched,
    image_params,
    intermediates=None,
):
    global _SAVE_COUNTER
    step_dir = Path(save_dir) / f'step_{_SAVE_COUNTER:06d}'
    step_dir.mkdir(parents=True, exist_ok=True)

    torch.save(state.cpu(), step_dir / 'state_raw.pt')
    with open(step_dir / 'task.txt', 'w') as f:
        f.write(task)
    for k, v in images.items():
        name = k.replace('.', '_').replace('/', '_')
        if isinstance(v, torch.Tensor):
            torch.save(v.cpu(), step_dir / f'image_raw_{name}.pt')

    torch.save(lang_tokens.cpu(), step_dir / 'lang_tokens.pt')
    torch.save(lang_masks.cpu(), step_dir / 'lang_masks.pt')
    for i, (img, mask) in enumerate(zip(images_batched, img_masks_batched)):
        torch.save(img.cpu(), step_dir / f'image_preprocessed_{i}.pt')
        torch.save(mask.cpu(), step_dir / f'image_mask_{i}.pt')
    for name, value in (intermediates or {}).items():
        if isinstance(value, torch.Tensor):
            torch.save(value.detach().cpu(), step_dir / f'{name}.pt')

    params_to_save = {}
    for k, v in image_params.items():
        params_to_save[k] = v.cpu().tolist() if isinstance(v, torch.Tensor) else v
    with open(step_dir / 'image_params.json', 'w') as f:
        json.dump(params_to_save, f, indent=2)

    print(f'[save] Inputs saved to {step_dir}')


def save_inference_outputs(
    save_dir,
    pred_action_raw,
    pred_action_final,
    intermediates=None,
):
    global _SAVE_COUNTER
    step_dir = Path(save_dir) / f'step_{_SAVE_COUNTER:06d}'
    step_dir.mkdir(parents=True, exist_ok=True)

    torch.save(pred_action_raw.cpu(), step_dir / 'action_raw.pt')
    torch.save(pred_action_final.cpu(), step_dir / 'action_final.pt')
    np.save(step_dir / 'action_final.npy', pred_action_final.cpu().numpy())
    for name, value in (intermediates or {}).items():
        if isinstance(value, torch.Tensor):
            torch.save(value.detach().cpu(), step_dir / f'{name}.pt')

    meta = {
        'action_raw_shape': list(pred_action_raw.shape),
        'action_final_shape': list(pred_action_final.shape),
        'timestamp': datetime.now().isoformat(),
    }
    with open(step_dir / 'output_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'[save] Outputs saved to {step_dir}')
    _SAVE_COUNTER += 1


def _resolve_model_dir(model_path: str) -> str:
    """Locate the directory holding config.json + diffusion_pytorch_model.bin.

    Order: model_path itself, then model_path/model_ema, then model_path/model.
    """
    for candidate in [model_path, os.path.join(model_path, 'model_ema'), os.path.join(model_path, 'model')]:
        if os.path.exists(os.path.join(candidate, 'config.json')) and os.path.exists(
            os.path.join(candidate, 'diffusion_pytorch_model.bin')
        ):
            return candidate
    raise FileNotFoundError(
        f'No diffusers-style ckpt (config.json + diffusion_pytorch_model.bin) under {model_path}'
    )


def _validate_vlm_type(saved_config: dict) -> None:
    """Reject checkpoints from removed VLM backends early and explicitly."""
    saved_vlm_type = saved_config.get('vlm_type')
    if saved_vlm_type not in (None, PALIGEMMA2_VLM_TYPE):
        raise ValueError(
            f'Only vlm_type={PALIGEMMA2_VLM_TYPE!r} is supported by this server; '
            f'checkpoint declares {saved_vlm_type!r}.'
        )


def _resolve_delta_mask(
    delta_cfg: dict[str, Any],
    embodiment_id: int,
    robot_type: str | None,
) -> tuple[list[bool], str | None]:
    mask_cfg = delta_cfg.get('mask')
    if not isinstance(mask_cfg, dict) or not mask_cfg:
        raise ValueError('inference_config.json delta_action_cfg.mask must be a non-empty mapping')

    selector = str(delta_cfg.get('selector', 'embodiment_id'))
    if selector == 'robot_type':
        if robot_type is None:
            if len(mask_cfg) != 1:
                raise ValueError(
                    '--robot-type is required when delta_action_cfg.selector="robot_type" '
                    f'and multiple masks are available: {sorted(mask_cfg)}'
                )
            robot_type = str(next(iter(mask_cfg)))
            print(f'Inferred --robot-type from the only delta mask: {robot_type}')
        if robot_type not in mask_cfg:
            raise KeyError(
                f'No delta mask for robot_type={robot_type!r}; available keys: {sorted(mask_cfg)}'
            )
        return [bool(value) for value in mask_cfg[robot_type]], robot_type

    if selector != 'embodiment_id':
        raise ValueError(
            f'AgileX unified server does not support delta mask selector {selector!r}; '
            "supported selectors are 'embodiment_id' and 'robot_type'."
        )
    key = str(embodiment_id)
    if key not in mask_cfg and embodiment_id in mask_cfg:
        key = embodiment_id
    if key not in mask_cfg:
        raise KeyError(
            f'No delta mask for embodiment_id={embodiment_id}; available keys: {sorted(mask_cfg)}'
        )
    return [bool(value) for value in mask_cfg[key]], robot_type


def _resolve_deployment_layout(
    delta_mask: list[bool],
    original_action_dim: int,
    state_dim: int,
    max_action_dim: int,
    is_robot_moving: bool,
    is_body_moving: bool,
):
    from giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils import (
        resolve_action_state_dim_layout,
    )

    return resolve_action_state_dim_layout(
        delta_mask,
        raw_action_dim=original_action_dim,
        raw_state_dim=state_dim,
        max_action_dim=max_action_dim,
        is_robot_moving=is_robot_moving,
        is_body_moving=is_body_moving,
    )


def _zero_dims_after(value: torch.Tensor, active_dim: int) -> torch.Tensor:
    """Return a copy with all columns after the active prefix set to zero."""
    active_dim = max(0, min(int(active_dim), int(value.shape[-1])))
    if active_dim == int(value.shape[-1]):
        return value
    result = value.clone()
    result[..., active_dim:] = 0
    return result


def _prepare_action_for_unnormalize(
    pred_action: torch.Tensor,
    active_dim: int,
) -> torch.Tensor:
    """Select one batch and match rollout's FP32 deployed-action postprocessing."""
    if pred_action.ndim != 3 or int(pred_action.shape[0]) != 1:
        raise ValueError(
            f'pred_action must have shape [1,T,D], got {tuple(pred_action.shape)}'
        )
    if not 0 < int(active_dim) <= int(pred_action.shape[-1]):
        raise ValueError(
            f'active_dim must be in [1, {pred_action.shape[-1]}], got {active_dim}'
        )
    return pred_action[0, :, :active_dim].float()


def _resolve_effective_action_dim(
    layout,
    mask_unsupervised_action_dims_for_noise: bool,
) -> int:
    """Match rollout action width to the train-time noise-mask setting."""
    if mask_unsupervised_action_dims_for_noise:
        return int(layout.action_supervised_dim)
    return int(layout.full_action_dim)


def _resolve_server_state_dim(
    expected_state_dim: int | None,
    configured_action_dim: int,
    max_state_dim: int,
) -> tuple[int, str]:
    """Resolve a fixed server-owned state schema before accepting requests."""
    state_dim = configured_action_dim if expected_state_dim is None else expected_state_dim
    source = 'configured_action_dim' if expected_state_dim is None else 'cli'
    if not 0 < int(state_dim) <= int(max_state_dim):
        raise ValueError(
            f'server state dimension must be in [1, {max_state_dim}], got {state_dim} '
            f'(source={source})'
        )
    return int(state_dim), source


def _validate_configured_action_dim(
    configured_action_dim: int,
    max_action_dim: int,
) -> int:
    """Validate the raw training action width against the model capacity."""
    configured_action_dim = int(configured_action_dim)
    max_action_dim = int(max_action_dim)
    if not 0 < configured_action_dim <= max_action_dim:
        raise ValueError(
            f'original_action_dim must be in [1, {max_action_dim}], '
            f'got {configured_action_dim}'
        )
    return configured_action_dim


def _normalize_request_label(field: str, value: Any) -> Any:
    if value is None or field not in {
        'control_mode',
        'control_mode_override',
        'end_effector_type',
        'end_effector_override',
    }:
        return value
    if not isinstance(value, str):
        raise TypeError(f'{field} must be a string, got {type(value).__name__}')

    normalized = ' '.join(value.strip().lower().replace('_', ' ').replace('-', ' ').split())
    if field in {'control_mode', 'control_mode_override'}:
        if normalized in {'ee', 'endeffector', 'end effector'}:
            return 'end effector'
        if normalized == 'joint':
            return 'joint'
    else:
        if normalized in {'dexhand', 'dex hand', 'dexterous hand', 'dexterous hands'}:
            return 'dex hand'
        if normalized == 'gripper':
            return 'gripper'
    raise ValueError(f'Unsupported {field}: {value!r}')


def _server_owned_value_matches(field: str, actual: Any, expected: Any) -> bool:
    actual = _normalize_request_label(field, actual)
    expected = _normalize_request_label(field, expected)
    if isinstance(expected, bool):
        return isinstance(actual, (bool, np.bool_)) and bool(actual) == expected
    if isinstance(expected, int):
        return (
            isinstance(actual, (int, np.integer))
            and not isinstance(actual, (bool, np.bool_))
            and int(actual) == expected
        )
    return actual == expected


def _strip_server_owned_request_fields(
    data: dict[str, Any],
    server_values: dict[str, Any],
) -> dict[str, Any]:
    """Reject deployment overrides and remove compatible client declarations."""
    if not isinstance(data, dict):
        raise TypeError(f'inference request must be a dict, got {type(data).__name__}')

    sanitized = dict(data)
    for field, expected in server_values.items():
        if field not in sanitized:
            continue
        actual = sanitized.pop(field)
        if not _server_owned_value_matches(field, actual, expected):
            raise ValueError(
                f'Client may not override server deployment field {field!r}: '
                f'client={actual!r}, server={expected!r}'
            )
    return sanitized


def _canonicalize_task(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if not isinstance(value, str):
        raise TypeError(f'task must be a string, got {type(value).__name__}')
    value = value.strip()
    if not value:
        raise ValueError('task must be a non-empty string')
    return value


def _canonicalize_inference_seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f'inference_seed must be an integer, got {type(value).__name__}'
        )
    value = int(value)
    if not 0 <= value < 2**63:
        raise ValueError('inference_seed must be in [0, 2**63)')
    return value


def _canonicalize_state(
    value: Any,
    *,
    expected_state_dim: int,
    observation_memory_size: int,
    allow_nan: bool,
) -> torch.Tensor:
    if isinstance(value, np.ndarray) and not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    try:
        state = torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError('observation.state must be numeric tensor-like data') from exc
    if state.dtype == torch.bool or state.is_complex():
        raise TypeError(f'observation.state must be real-valued, got dtype={state.dtype}')
    if state.ndim not in (1, 2):
        raise ValueError(f'observation.state must have shape [D] or [K,D], got {tuple(state.shape)}')
    if int(state.shape[-1]) != expected_state_dim:
        raise ValueError(
            f'observation.state has {int(state.shape[-1])} columns, but this server is configured '
            f'for {expected_state_dim}. Fix the client robot schema or restart the server with '
            '--expected-state-dim set to the deployment state width.'
        )
    if state.ndim == 2:
        if int(state.shape[0]) != observation_memory_size:
            raise ValueError(
                f'Pre-stacked observation.state has {int(state.shape[0])} frames, but the server '
                f'requires observation_memory_size={observation_memory_size}'
            )
        if observation_memory_size == 1:
            state = state[0]

    state = state.to(dtype=torch.float32)
    if bool(torch.isinf(state).any()):
        raise ValueError('observation.state contains +/-inf')
    if not allow_nan and bool(torch.isnan(state).any()):
        raise ValueError(
            'observation.state contains NaN, but NaN state masking is only supported by '
            "state_input_mode='proprio_anchor'"
        )
    return state.contiguous()


def _canonicalize_image(
    value: Any,
    *,
    key: str,
    observation_memory_size: int,
) -> torch.Tensor:
    if isinstance(value, np.ndarray) and not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    try:
        image = torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f'{key} must be numeric tensor-like data') from exc
    if image.ndim == 3:
        if int(image.shape[0]) != 3:
            if int(image.shape[-1]) != 3:
                raise ValueError(f'{key} must have 3 RGB channels, got shape {tuple(image.shape)}')
            image = image.permute(2, 0, 1)
        if observation_memory_size != 1:
            raise ValueError(
                f'{key} provides one frame, but this server requires '
                f'observation_memory_size={observation_memory_size} pre-stacked frames'
            )
    elif image.ndim == 4:
        if int(image.shape[1]) != 3:
            if int(image.shape[-1]) != 3:
                raise ValueError(
                    f'{key} must have shape [K,3,H,W] or [K,H,W,3], got {tuple(image.shape)}'
                )
            image = image.permute(0, 3, 1, 2)
        if int(image.shape[0]) != observation_memory_size:
            raise ValueError(
                f'{key} has {int(image.shape[0])} frames, but the server requires '
                f'observation_memory_size={observation_memory_size}'
            )
        if observation_memory_size == 1:
            image = image[0]
    else:
        raise ValueError(
            f'{key} must have shape [3,H,W], [H,W,3], [K,3,H,W], or [K,H,W,3], '
            f'got {tuple(image.shape)}'
        )
    if any(int(dim) <= 0 for dim in image.shape):
        raise ValueError(f'{key} has an empty dimension: {tuple(image.shape)}')
    if image.dtype == torch.bool or image.is_complex():
        raise TypeError(f'{key} must contain real-valued pixels, got dtype={image.dtype}')

    if image.dtype == torch.uint8:
        return image.contiguous().clone()
    if image.is_floating_point():
        image = image.to(dtype=torch.float32)
        if not bool(torch.isfinite(image).all()):
            raise ValueError(f'{key} contains NaN or +/-inf pixels')
        min_value = float(image.min().item())
        max_value = float(image.max().item())
        if min_value < 0.0 or max_value > 255.0:
            raise ValueError(
                f'{key} floating pixels must be in [0,1] or [0,255], got '
                f'[{min_value}, {max_value}]'
            )
        if max_value > 1.0:
            image = image / 255.0
        return image.contiguous().clone()

    integer_image = image.to(dtype=torch.int64)
    min_value = int(integer_image.min().item())
    max_value = int(integer_image.max().item())
    if min_value < 0 or max_value > 255:
        raise ValueError(
            f'{key} integer pixels must be in [0,255], got [{min_value}, {max_value}]'
        )
    return image.to(dtype=torch.uint8).contiguous()


def _canonicalize_inference_request(
    data: dict[str, Any],
    *,
    server_values: dict[str, Any],
    present_img_keys: list[str],
    expected_state_dim: int,
    observation_memory_size: int,
    state_input_mode: str,
) -> dict[str, Any]:
    data = _strip_server_owned_request_fields(data, server_values)
    if 'task' not in data:
        raise KeyError("inference request is missing required field 'task'")
    if 'observation.state' not in data:
        raise KeyError("inference request is missing required field 'observation.state'")

    data['task'] = _canonicalize_task(data['task'])
    if 'inference_seed' in data:
        data['inference_seed'] = _canonicalize_inference_seed(
            data['inference_seed']
        )
    data['observation.state'] = _canonicalize_state(
        data['observation.state'],
        expected_state_dim=expected_state_dim,
        observation_memory_size=observation_memory_size,
        allow_nan=state_input_mode == 'proprio_anchor',
    )

    image_count = 0
    for key in present_img_keys:
        if key not in data:
            continue
        data[key] = _canonicalize_image(
            data[key], key=key, observation_memory_size=observation_memory_size
        )
        image_count += 1
    if image_count == 0:
        raise ValueError(
            'inference request has none of the server-configured image keys: '
            f'{present_img_keys}'
        )

    for reset_key in ('reset_observation_memory', 'reset'):
        if reset_key in data and not isinstance(data[reset_key], (bool, np.bool_)):
            raise TypeError(f'{reset_key} must be a boolean, got {type(data[reset_key]).__name__}')
    if (
        'reset_observation_memory' in data
        and 'reset' in data
        and bool(data['reset_observation_memory']) != bool(data['reset'])
    ):
        raise ValueError('reset_observation_memory and reset disagree')
    explicit_pad_masks = data.get('image_pad_masks', data.get('images_is_pad', {}))
    if explicit_pad_masks is not None and not isinstance(explicit_pad_masks, dict):
        raise TypeError('image_pad_masks/images_is_pad must be a mapping keyed by camera name')
    return data


def get_policy(
    model_path: str,
    pretrained_path: str | None,
    fast_tokenizer_path: str,
    embodiment_id: int,
    norm_stats_path: str,
    save_dir: str | None = None,
    use_bf16: bool = True,
    mask_unsupervised_action_dims_for_noise: bool | None = None,
    robot_type: str | None = None,
    original_action_dim: int | None = None,
    expected_state_dim: int | None = None,
    is_robot_moving: bool = False,
    is_body_moving: bool = False,
):
    from giga_models import GigaBrain07Policy
    from giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils import (
        AbsoluteActions,
        ImageTransform,
        Normalize,
        PadStatesAndActions,
        PromptTokenizerTransform,
        Unnormalize,
        infer_end_effector_type_from_delta_mask,
        load_inference_config,
    )

    # 1. Locate ckpt + read config.json + inference_config.json sidecar.
    ckpt_dir = _resolve_model_dir(model_path)
    with open(os.path.join(ckpt_dir, 'config.json')) as f:
        saved_config = json.load(f)
    inference_cfg = load_inference_config(ckpt_dir)
    image_cfg = inference_cfg['image_cfg']
    prompt_cfg = inference_cfg['prompt_cfg']
    norm_cfg = inference_cfg['norm_cfg']
    delta_cfg = inference_cfg['delta_action_cfg']

    delta_mask, robot_type = _resolve_delta_mask(delta_cfg, embodiment_id, robot_type)
    if original_action_dim is None:
        original_action_dim = len(delta_mask)
    original_action_dim = _validate_configured_action_dim(
        original_action_dim,
        saved_config.get('max_action_dim', len(delta_mask)),
    )
    if expected_state_dim is not None and expected_state_dim <= 0:
        raise ValueError(f'expected_state_dim must be positive, got {expected_state_dim}')
    if is_robot_moving and is_body_moving:
        print('WARNING: is_robot_moving=True takes precedence; ignoring is_body_moving=True.')
        is_body_moving = False
    configured_noise_mask = bool(
        delta_cfg.get('mask_unsupervised_action_dims_for_noise', False)
    )
    noise_mask_source = (
        'config'
        if mask_unsupervised_action_dims_for_noise is None
        else 'cli'
    )
    if mask_unsupervised_action_dims_for_noise is None:
        mask_unsupervised_action_dims_for_noise = configured_noise_mask

    _validate_vlm_type(saved_config)
    vlm_type = PALIGEMMA2_VLM_TYPE
    with_expert_attr = 'paligemma_with_expert'

    # Fill in pretrained_path from the PaliGemma2 default if the user didn't pass one.
    if pretrained_path is None:
        pretrained_path = PALIGEMMA2_DEFAULT_PRETRAINED_PATH
        if pretrained_path is None:
            raise ValueError(
                f'No --pretrained-path supplied for vlm_type={vlm_type!r}. '
                'Pass --pretrained-path explicitly.'
            )
        print(f'Using default --pretrained-path for {vlm_type}: {pretrained_path}')

    print(f'Loading from: {ckpt_dir} (vlm_type={PALIGEMMA2_VLM_TYPE})')

    # 2. Build the one legacy override still needed by PaliGemma2 checkpoints:
    #    older checkpoints may omit has_action_expert, so infer it from the weights.
    overrides: dict = {}
    if 'has_action_expert' not in saved_config:
        state_dict = torch.load(
            os.path.join(ckpt_dir, 'diffusion_pytorch_model.bin'),
            map_location='cpu', weights_only=False, mmap=True,
        )
        inferred = any(k.startswith(f'{with_expert_attr}.') for k in state_dict.keys())
        print(f'WARNING: config.json missing has_action_expert; inferred from weights -> {inferred}')
        overrides['has_action_expert'] = inferred

    policy = GigaBrain07Policy.from_pretrained(ckpt_dir, **overrides)
    if not getattr(policy, 'has_action_expert', False):
        raise ValueError(
            'The PaliGemma2 unified flow server requires a checkpoint with '
            'has_action_expert=True; pure language/FAST checkpoints are not supported.'
        )
    original_action_dim = _validate_configured_action_dim(
        original_action_dim,
        policy.max_action_dim,
    )

    state_input_mode = getattr(policy.config, 'state_input_mode', 'prompt')
    if state_input_mode not in ('prompt', 'proprio_memory', 'proprio_anchor'):
        raise ValueError(
            f'AgileX unified server does not support state_input_mode={state_input_mode!r}; '
            "supported modes are 'prompt', 'proprio_memory', and 'proprio_anchor'."
        )
    observation_memory_size = int(getattr(policy.config, 'observation_memory_size', 1))
    if observation_memory_size < 1:
        raise ValueError(f'observation_memory_size must be positive, got {observation_memory_size}')

    deployment_state_dim, state_dim_source = _resolve_server_state_dim(
        expected_state_dim,
        original_action_dim,
        policy.max_action_dim,
    )
    deployment_layout = _resolve_deployment_layout(
        delta_mask,
        original_action_dim,
        deployment_state_dim,
        policy.max_action_dim,
        is_robot_moving,
        is_body_moving,
    )
    deployment_action_dim = _resolve_effective_action_dim(
        deployment_layout,
        mask_unsupervised_action_dims_for_noise,
    )
    if mask_unsupervised_action_dims_for_noise:
        action_denoise_mask = torch.zeros(policy.max_action_dim, dtype=torch.bool)
        action_denoise_mask[:deployment_action_dim] = True
        policy.set_action_denoise_mask(action_denoise_mask)
        print(
            'Action denoise mask enabled: '
            f'active dims [0, {deployment_action_dim}), '
            f'padded dims [{deployment_action_dim}, {policy.max_action_dim}) stay zero'
        )
    else:
        policy.set_action_denoise_mask(None)
    print(
        'Deployment layout: '
        f'input_state_dim={deployment_state_dim}, configured_action_dim={original_action_dim}, '
        f'output_action_dim={deployment_action_dim}, '
        f'state_input_dim={deployment_layout.state_input_dim}, '
        f'action_supervised_dim={deployment_layout.action_supervised_dim}, '
        f'mask_unsupervised_action_dims_for_noise={mask_unsupervised_action_dims_for_noise} '
        f'(source={noise_mask_source}), '
        f'is_robot_moving={is_robot_moving}, is_body_moving={is_body_moving}'
    )

    # 3. Cast and move to GPU.
    if use_bf16:
        policy.to(dtype=torch.bfloat16)
    nn.Module.to(policy, 'cuda')
    policy.eval()

    inference_dtype = next(policy.parameters()).dtype
    param_count = sum(p.numel() for p in policy.parameters()) / 1e9
    print(f'Model loaded: {param_count:.2f}B params, inference dtype = {inference_dtype}')

    # 5. Build transforms — preprocess values come from the inference_config.json
    # sidecar (tokenizer / norm-stats *paths* are CLI args because the training
    # paths usually don't exist on the robot machine).
    with open(norm_stats_path, 'r') as f:
        norm_stats_data = json.load(f)['norm_stats']

    # PaliGemma2 uses the PaliGemma SentencePiece tokenizer/processor.
    tokenizer_model_path = pretrained_path
    if tokenizer_model_path is None:
        raise ValueError('--pretrained-path is required for tokenizer loading.')
    end_effector_type = infer_end_effector_type_from_delta_mask(delta_mask)
    server_control_mode = 'end effector' if embodiment_id in (3, 4, 5) else 'joint'

    present_img_keys = image_cfg['present_img_keys']
    image_transform = ImageTransform(
        is_train=False,
        resize_imgs_with_padding=image_cfg['resize_imgs_with_padding'],
        present_img_keys=present_img_keys,
        enable_image_aug=False,
        vlm_type=PALIGEMMA2_VLM_TYPE,
        high_res_cam=image_cfg.get('high_res_cam'),
    )
    prompt_transform = PromptTokenizerTransform(
        is_train=False,
        tokenizer_model_path=tokenizer_model_path,
        fast_tokenizer_path=fast_tokenizer_path,
        max_length=prompt_cfg['max_length'],
        discrete_state_input=prompt_cfg['discrete_state_input'],
        discrete_state_input_for_pose_embodiments=prompt_cfg.get(
            'discrete_state_input_for_pose_embodiments', False
        ),
        encode_action_input=prompt_cfg['encode_action_input'],
        fast_token_vocab_mode=getattr(policy, 'fast_token_vocab_mode', None),
        fast_token_tail_skip_tokens=getattr(policy, 'fast_token_tail_skip_tokens', 128),
        fast_token_tail_vocab_size=getattr(policy, 'fast_token_tail_vocab_size', None),
        encoded_action_horizon=prompt_cfg.get('encoded_action_horizon'),
        encode_sub_task_input=prompt_cfg['encode_sub_task_input'],
        enable_control_mode_token=prompt_cfg.get('enable_control_mode_token', False),
        control_mode_override=server_control_mode,
        enable_end_effector_token=prompt_cfg.get('enable_end_effector_token', False),
        end_effector_override=end_effector_type,
        vlm_type=PALIGEMMA2_VLM_TYPE,
        prefix_lm_text=prompt_cfg.get('prefix_lm_text', False),
        image_attn=prompt_cfg.get('image_attn', 'bidirectional'),
        use_chat_template=prompt_cfg.get('use_chat_template', False),
        chat_system_prompt=prompt_cfg.get('chat_system_prompt'),
        prompt_filler_text=prompt_cfg.get('prompt_filler_text'),
        resize_imgs_with_padding=image_cfg['resize_imgs_with_padding'],
        state_input_mode=state_input_mode,
    )
    if (
        state_input_mode == 'proprio_anchor'
        and prompt_transform.propri_token_id != policy.propri_token_id
    ):
        raise ValueError(
            'Tokenizer/model <|propri|> token id mismatch: '
            f'{prompt_transform.propri_token_id} vs {policy.propri_token_id}'
        )
    use_quantiles = norm_cfg['use_quantiles']
    state_normalize = Normalize(
        {embodiment_id: norm_stats_data['observation.state']},
        use_quantiles=use_quantiles,
        enable_clamp=bool(norm_cfg.get('enable_clamp', False)),
    )
    action_unnormalize = Unnormalize({embodiment_id: norm_stats_data['action']}, use_quantiles=use_quantiles)
    state_unnormalize = Unnormalize({embodiment_id: norm_stats_data['observation.state']}, use_quantiles=use_quantiles)
    absolute_actions = AbsoluteActions({embodiment_id: delta_mask})
    pad_states_and_actions = PadStatesAndActions(action_dim=policy.max_action_dim)

    state_normalize.to('cuda')
    action_unnormalize.to('cuda')
    state_unnormalize.to('cuda')
    absolute_actions.to('cuda')
    prompt_transform.to('cuda')

    # 6. Bind inference method.
    _save_dir = save_dir
    if _save_dir:
        Path(_save_dir).mkdir(parents=True, exist_ok=True)
        print(f'[save] Will save inference I/O to: {_save_dir}')

    state_memory_buffer: deque[torch.Tensor] = deque(maxlen=observation_memory_size)
    server_owned_request_values = {
        'embodiment_id': int(embodiment_id),
        'robot_type': robot_type,
        'control_mode': server_control_mode,
        'control_mode_override': server_control_mode,
        'end_effector_type': end_effector_type,
        'end_effector_override': end_effector_type,
        'expected_state_dim': deployment_state_dim,
        'state_input_dim': deployment_state_dim,
        'configured_action_dim': int(original_action_dim),
        'original_action_dim': deployment_action_dim,
        'action_output_dim': deployment_action_dim,
        'training_action_dim': deployment_action_dim,
        'observation_memory_size': observation_memory_size,
        'mask_unsupervised_action_dims_for_noise': bool(
            mask_unsupervised_action_dims_for_noise
        ),
        'zero_noise_outside_action_dim': bool(mask_unsupervised_action_dims_for_noise),
        'is_robot_moving': bool(is_robot_moving),
        'is_body_moving': bool(is_body_moving),
        'n_action_steps': int(policy.n_action_steps),
    }

    def prepare_state_memory(
        state_normed: torch.Tensor,
        data: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        current_state = state_normed[-1] if state_normed.ndim == 2 else state_normed
        if state_input_mode != 'proprio_memory':
            return current_state, None, None

        if state_normed.ndim == 2:
            if state_normed.shape[0] != observation_memory_size:
                raise ValueError(
                    f'Pre-stacked state memory has {state_normed.shape[0]} frames but '
                    f'observation_memory_size={observation_memory_size}'
                )
            explicit_mask = data.get(
                'observation.state_memory_mask', data.get('state_memory_mask')
            )
            if explicit_mask is None:
                memory_mask = torch.ones(
                    observation_memory_size, dtype=torch.bool, device=state_normed.device
                )
            else:
                memory_mask = torch.as_tensor(
                    explicit_mask, dtype=torch.bool, device=state_normed.device
                )
                if memory_mask.shape != (observation_memory_size,):
                    raise ValueError(
                        f'state memory mask must have shape ({observation_memory_size},), '
                        f'got {tuple(memory_mask.shape)}'
                    )
            return current_state, state_normed, memory_mask

        if state_normed.ndim != 1:
            raise ValueError(f'state must have shape [D] or [K,D], got {tuple(state_normed.shape)}')

        state_memory_buffer.append(state_normed.detach().clone())
        num_real = len(state_memory_buffer)
        num_pad = observation_memory_size - num_real
        frames = list(state_memory_buffer)
        if num_pad > 0:
            frames = [frames[0]] * num_pad + frames
        memory = torch.stack(frames, dim=0)
        memory_mask = torch.ones(observation_memory_size, dtype=torch.bool, device=state_normed.device)
        if num_pad > 0:
            memory_mask[:num_pad] = False
        return current_state, memory, memory_mask

    @torch.no_grad()
    def inference(self, data: dict[str, Any]) -> torch.Tensor:
        t0 = time.time()
        data = _canonicalize_inference_request(
            data,
            server_values=server_owned_request_values,
            present_img_keys=present_img_keys,
            expected_state_dim=deployment_state_dim,
            observation_memory_size=observation_memory_size,
            state_input_mode=state_input_mode,
        )
        inference_seed = data.pop('inference_seed', None)
        if data.get('reset_observation_memory', data.get('reset', False)):
            state_memory_buffer.clear()
        images = {k: data[k] for k in present_img_keys if k in data}
        explicit_pad_masks = data.get('image_pad_masks', data.get('images_is_pad', {})) or {}
        for key in present_img_keys:
            pad_key = f'{key}_is_pad'
            if pad_key in data:
                images[pad_key] = data[pad_key]
            elif key in explicit_pad_masks:
                images[pad_key] = explicit_pad_masks[key]
            elif pad_key in explicit_pad_masks:
                images[pad_key] = explicit_pad_masks[pad_key]

        task = data['task']
        state = data['observation.state'].to('cuda')

        imgs, img_masks, image_params = image_transform(images)
        if images.get('_skip_loss_for_invalid_images', False):
            raise ValueError(
                'inference request has no valid current RGB image after applying camera pad masks'
            )
        agent_pos_mask = (~torch.isnan(state)).to(dtype=torch.float32)
        if state_input_mode == 'proprio_anchor':
            state = torch.nan_to_num(state, nan=0.0)
        state_normed = state_normalize(state, embodiment_id=embodiment_id)
        action_reference_state = state_normed[-1] if state_normed.ndim == 2 else state_normed
        current_state, state_memory, state_memory_masks = prepare_state_memory(
            state_normed, data
        )

        prompt_data = {'task': task, 'embodiment_id': embodiment_id}
        if state_input_mode == 'prompt':
            prompt_data['observation.state'] = current_state
        lang_tokens, lang_masks, lang_att_masks, _, _, _, _ = prompt_transform(prompt_data)

        state_padded = pad_states_and_actions({'observation.state': current_state})['observation.state']
        proprioception = None
        anchor_agent_pos_mask = None
        if state_input_mode == 'proprio_anchor':
            current_agent_pos_mask = (
                agent_pos_mask[-1]
                if agent_pos_mask.ndim == 2
                else agent_pos_mask
            )
            mask_padded = pad_states_and_actions(
                {'observation.state': current_agent_pos_mask}
            )['observation.state']
            proprioception = state_padded.unsqueeze(0).unsqueeze(0)
            anchor_agent_pos_mask = mask_padded.unsqueeze(0).unsqueeze(0)

        emb_ids = torch.tensor(embodiment_id, dtype=torch.long, device='cuda').unsqueeze(0)
        images_batched = [img.unsqueeze(0).to('cuda') for img in imgs]
        img_masks_batched = [m.unsqueeze(0).to('cuda') for m in img_masks]
        lang_tokens_b = lang_tokens.unsqueeze(0).to('cuda')
        lang_masks_b = lang_masks.unsqueeze(0).to('cuda')
        lang_att_masks_b = lang_att_masks.unsqueeze(0).to('cuda')

        if _save_dir:
            save_inference_inputs(
                _save_dir, images, state, task, lang_tokens_b, lang_masks_b,
                images_batched, img_masks_batched, image_params,
                intermediates={
                    'lang_att_masks': lang_att_masks_b,
                    'state_normalized': state_normed,
                    'action_reference_state_normalized': action_reference_state,
                    'proprioception': proprioception,
                    'agent_pos_mask': anchor_agent_pos_mask,
                },
            )

        if inference_seed is not None:
            torch.manual_seed(inference_seed)
            torch.cuda.manual_seed_all(inference_seed)

        pred_action = policy.sample_actions(
            images=images_batched,
            img_masks=img_masks_batched,
            lang_tokens=lang_tokens_b,
            lang_masks=lang_masks_b,
            emb_ids=emb_ids,
            lang_att_masks=lang_att_masks_b,
            state_memory=state_memory.unsqueeze(0) if state_memory is not None else None,
            state_memory_masks=(
                state_memory_masks.unsqueeze(0) if state_memory_masks is not None else None
            ),
            proprioception=proprioception,
            agent_pos_mask=anchor_agent_pos_mask,
            state=(state_padded.unsqueeze(0) if state_padded.dim() == 1 else state_padded).to('cuda'),
        )

        pred_action = _zero_dims_after(pred_action, deployment_action_dim)
        pred_action_raw = pred_action.clone()
        output_dict = {
            # Match the rollout path: trim to the deployed action schema and cast
            # BF16 model output to FP32 before quantile unnormalization. Performing
            # ``(x + 1) / 2`` in BF16 introduces milliscale deployment drift.
            'action': _prepare_action_for_unnormalize(
                pred_action, deployment_action_dim
            ),
            'observation.state': action_reference_state,
            'embodiment_id': embodiment_id,
        }
        output_dict['observation.state'] = state_unnormalize(output_dict['observation.state'], embodiment_id=embodiment_id)
        output_dict['action'] = action_unnormalize(output_dict['action'], embodiment_id=embodiment_id)
        pred_action_delta = output_dict['action'].clone()
        action_reference_state_raw = output_dict['observation.state'].clone()
        output_dict = absolute_actions(output_dict)
        pred_action = _zero_dims_after(
            output_dict['action'], deployment_action_dim
        )[:, :deployment_action_dim]

        if _save_dir:
            save_inference_outputs(
                _save_dir,
                pred_action_raw,
                pred_action,
                intermediates={
                    'action_delta_unnormalized': pred_action_delta,
                    'action_reference_state_raw': action_reference_state_raw,
                },
            )

        elapsed = time.time() - t0
        print(f'Inference took {elapsed:.3f}s')

        return pred_action

    policy.inference = types.MethodType(inference, policy)
    policy.server_info = {
        'policy_family': 'gigabrain07',
        'model_path': os.path.abspath(ckpt_dir),
        'norm_stats_path': os.path.abspath(norm_stats_path),
        'vlm_type': vlm_type,
        'state_input_mode': state_input_mode,
        'observation_memory_size': observation_memory_size,
        'mask_unsupervised_action_dims_for_noise': bool(
            mask_unsupervised_action_dims_for_noise
        ),
        'mask_unsupervised_action_dims_for_noise_config': configured_noise_mask,
        'zero_noise_outside_action_dim': bool(
            mask_unsupervised_action_dims_for_noise
        ),
        'zero_noise_outside_action_dim_source': noise_mask_source,
        'configured_action_dim': int(original_action_dim),
        'original_action_dim': deployment_action_dim,
        'action_output_dim': deployment_action_dim,
        'training_action_dim': deployment_action_dim,
        'n_action_steps': int(policy.n_action_steps),
        'robot_type': robot_type,
        'embodiment_id': int(embodiment_id),
        'expected_state_dim': deployment_state_dim,
        'state_input_dim': deployment_state_dim,
        'state_input_dim_source': state_dim_source,
        'action_supervised_dim': int(deployment_layout.action_supervised_dim),
        'action_output_dim_state_dependent': False,
        'is_robot_moving': bool(is_robot_moving),
        'is_body_moving': bool(is_body_moving),
        'control_mode': server_control_mode,
        'end_effector_type': end_effector_type,
        'present_img_keys': list(present_img_keys),
        'request_schema_policy': 'server_authoritative',
        'supports_inference_seed': True,
    }

    return policy


def run_server(
    model_path: str,
    pretrained_path: str | None = None,
    fast_tokenizer_path: str = '/home/agilex-home/agilex/models/huggingface/models--physical-intelligence--fast',
    embodiment_id: int = 0,
    norm_stats_path: str = '/home/agilex-home/agilex/wangyumeng/data/clean_desk/meta/norm_stats_wy.json',
    host: str = '0.0.0.0',
    port: int = 8080,
    save_dir: str | None = None,
    use_bf16: bool = True,
    mask_unsupervised_action_dims_for_noise: bool | None = None,
    robot_type: str | None = None,
    original_action_dim: int | None = None,
    expected_state_dim: int | None = None,
    is_robot_moving: bool = False,
    is_body_moving: bool = False,
) -> None:
    """Launch the unified inference server.

    Args:
        model_path: Path to the diffusers-style checkpoint (or its parent that contains `model/` or `model_ema/`).
        pretrained_path: PaliGemma2 HF tokenizer / processor directory.
        fast_tokenizer_path: Path to the FAST action tokenizer.
        embodiment_id: Robot embodiment id (per-embodiment norm-stats / delta-mask).
        norm_stats_path: Path to norm_stats.json produced during training.
        host/port: ZMQ server bind address.
        save_dir: If set, dump inference inputs/outputs under this dir for offline analysis.
        use_bf16: Cast policy weights to bf16 (default). Pass --no-use-bf16 for fp32.
        mask_unsupervised_action_dims_for_noise: None follows
            inference_config.json delta_action_cfg.mask_unsupervised_action_dims_for_noise;
            an explicit CLI value overrides the checkpoint sidecar.
        robot_type: Delta-mask key when the sidecar uses selector='robot_type'. If the
            sidecar has only one robot mask, it is inferred automatically.
        original_action_dim: Optional action-schema cap. Defaults to the selected
            delta-mask width. The actual returned width follows the train-time noise-mask
            config and movement layout, matching the rollout script.
        expected_state_dim: Fixed deployment state width owned by the server. Defaults to
            original_action_dim. Set this explicitly when state/action widths differ (for
            example AgileX mobile state=14/action=16). Requests with another width fail.
        is_robot_moving: Enable the full configured action prefix for mobile tasks.
            This does not change the state input dimensions.
        is_body_moving: Enable state-aligned body action outputs while the mobile base is
            stationary. This does not change the state input dimensions.
    """
    policy = get_policy(
        model_path=model_path,
        pretrained_path=pretrained_path,
        fast_tokenizer_path=fast_tokenizer_path,
        embodiment_id=embodiment_id,
        norm_stats_path=norm_stats_path,
        save_dir=save_dir,
        use_bf16=use_bf16,
        mask_unsupervised_action_dims_for_noise=mask_unsupervised_action_dims_for_noise,
        robot_type=robot_type,
        original_action_dim=original_action_dim,
        expected_state_dim=expected_state_dim,
        is_robot_moving=is_robot_moving,
        is_body_moving=is_body_moving,
    )

    server = RobotInferenceServer(policy, host=host, port=port)
    server.register_endpoint('server_info', lambda: policy.server_info, requires_input=False)
    server.register_endpoint(
        'ping',
        lambda: (
            print('Client connected')
            or {'status': 'ok', 'message': 'Server is running', 'server_info': policy.server_info}
        ),
        requires_input=False,
    )
    server.run()


if __name__ == '__main__':
    tyro.cli(run_server)
