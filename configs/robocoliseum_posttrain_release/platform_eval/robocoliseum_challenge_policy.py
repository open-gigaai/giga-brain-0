"""GigaBrain-0.7 runtime used by the Simulation Challenge tunnel handler."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from robocoliseum_challenge_adapter import (
    CAMERA_KEY_MAP,
    MODEL_ACTION_DIM,
    MODEL_STATE_DIM,
    build_action_response,
    build_model_observation,
)


MODEL_ACTION_HORIZON = 50
EMBODIMENT_ID = 1
DELTA_MASK = [True] * 7 + [False] + [True] * 7 + [False, True]


class GigaBrainChallengePolicy:
    """Load the fine-tuned checkpoint and serve one challenge inference stream."""

    def __init__(
        self,
        *,
        model_path: str,
        norm_stats_path: str,
        tokenizer_model_path: str,
        fast_tokenizer_path: str,
        device: str = 'cuda',
    ) -> None:
        # Heavy training-image dependencies stay lazy so protocol tests remain CPU-only.
        import torch
        from giga_models import GigaBrain07Pipeline
        from giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils import (
            ImageTransform,
        )

        model_dir = Path(model_path)
        if not (model_dir / 'config.json').is_file():
            raise FileNotFoundError(f'model config not found under {model_dir}')
        if not (model_dir / 'diffusion_pytorch_model.bin').is_file():
            raise FileNotFoundError(f'model weights not found under {model_dir}')

        with open(norm_stats_path, 'r') as file:
            norm_stats = json.load(file)['norm_stats']

        self._torch = torch
        with open(model_dir / 'config.json', 'r') as file:
            model_config = json.load(file)
        state_input_mode = str(model_config.get('state_input_mode', 'prompt'))
        if state_input_mode not in {'prompt', 'proprio_anchor'}:
            raise ValueError(f'unsupported state_input_mode={state_input_mode!r}')

        self.pipeline = GigaBrain07Pipeline(
            model_path=str(model_dir),
            tokenizer_model_path=tokenizer_model_path,
            fast_tokenizer_path=fast_tokenizer_path,
            embodiment_id=EMBODIMENT_ID,
            state_norm_stats=norm_stats['observation.state'],
            action_norm_stats=norm_stats['action'],
            delta_mask=DELTA_MASK,
            original_action_dim=MODEL_STATE_DIM,
            discrete_state_input=state_input_mode == 'prompt',
            encode_sub_task_input=False,
            enable_control_mode_token=True,
            control_mode_override='joint',
            enable_end_effector_token=True,
            end_effector_override='gripper',
            resize_imgs_with_padding=(224, 224),
            prompt_max_length=120 if state_input_mode == 'proprio_anchor' else 300,
        )

        # GigaBrain07Pipeline builds ImageTransform with the default cam_high /
        # cam_left_wrist / cam_right_wrist keys, but this checkpoint was trained on
        # the RoboColiseum keys. Without this override every camera misses and the
        # transform substitutes CPU zero placeholders, so the model would see three
        # blank frames. Mirrors image_cfg in robocoliseum_ext/posttrain_config.py with
        # augmentation disabled for inference.
        self.pipeline.image_transform = ImageTransform(
            is_train=False,
            resize_imgs_with_padding=self.pipeline.resize_imgs_with_padding,
            present_img_keys=list(CAMERA_KEY_MAP.values()),
            enable_image_aug=False,
            enable_depth_img=self.pipeline.enable_depth_img,
            vlm_type=self.pipeline.policy.vlm_type,
        )

        # Training masked the stationary waist dimension during flow matching.
        denoise_mask = torch.zeros(
            self.pipeline.policy.max_action_dim, dtype=torch.bool
        )
        denoise_mask[:MODEL_ACTION_DIM] = True
        self.pipeline.policy.set_action_denoise_mask(denoise_mask)
        self.pipeline.set_action_output_dim_mask(denoise_mask)
        self._device = torch.device(device)
        if (
            str(device).startswith('cuda')
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
        ):
            # Spread the heavy policy across visible GPUs while keeping the
            # lightweight transforms and input tensors on the first device.
            from accelerate import dispatch_model

            self.pipeline.device = 'cuda:0'
            for transform in (
                self.pipeline.state_normalize_transform,
                self.pipeline.image_transform,
                self.pipeline.prompt_tokenizer_transform,
                self.pipeline.pad_states_and_actions_transform,
                self.pipeline.state_unnormalize_transform,
                self.pipeline.action_unnormalize_transform,
                self.pipeline.absolute_actions_transform,
            ):
                if hasattr(transform, 'to'):
                    transform.to('cuda:0')
            if self.pipeline.action_output_dim_mask is not None:
                self.pipeline.action_output_dim_mask = (
                    self.pipeline.action_output_dim_mask.to('cuda:0')
                )
            policy = self.pipeline.policy
            # Keep each decoder layer intact and distribute layers round-robin.
            # The vision tower uses the last visible device to reduce pressure
            # on the input device.
            n_layers = len(policy.paligemma_with_expert.layers)
            num_devices = torch.cuda.device_count()
            visible_devices = [f'cuda:{i}' for i in range(num_devices)]
            device_map = {'': visible_devices[0]}
            for i in range(n_layers):
                device_map[f'paligemma_with_expert.layers.{i}'] = visible_devices[i % num_devices]
            device_map['paligemma_with_expert.vision_tower'] = visible_devices[-1]
            self.pipeline.policy = dispatch_model(policy, device_map=device_map)
            self._device = torch.device('cuda:0')
        else:
            self.pipeline.to(device)
        # Inference tensors must be built on the same device as the weights.
        self._episode_idx: int | None = None

    def reset(self) -> None:
        """Clear temporal state before the next simulator session starts."""
        self.pipeline.reset_observation_memory()
        self._episode_idx = None

    def infer(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Run one gateway observation through GigaBrain and return native lists."""
        observation = build_model_observation(params)
        episode_idx = int(params.get('episode_idx', 0))
        if self._episode_idx is not None and episode_idx != self._episode_idx:
            self.pipeline.reset_observation_memory()
        self._episode_idx = episode_idx

        images = {
            key: self._torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .to(device=self._device, dtype=self._torch.float32)
            .div_(255.0)
            for key, image in observation['images'].items()
        }
        state = self._torch.from_numpy(observation['state']).to(self._device)
        actions = self.pipeline(
            images,
            observation['task'],
            state,
            is_robot_moving=False,
            is_body_moving=False,
        )

        if actions.ndim != 2 or actions.shape[0] != MODEL_ACTION_HORIZON:
            raise ValueError(
                f'model returned {tuple(actions.shape)}, expected '
                f'[{MODEL_ACTION_HORIZON}, D]'
            )
        return build_action_response(actions.detach().cpu().numpy())
