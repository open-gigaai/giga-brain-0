import json
import types
from typing import Any

import torch
import tyro
from giga_models import GigaBrain0Pipeline
from giga_models.sockets import RobotInferenceServer


def get_policy(
    model_path: str,
    tokenizer_model_path: str,
    fast_tokenizer_path: str,
    embodiment_id: int,
    norm_stats_path: str,
    delta_mask: list[bool],
    original_action_dim: int,
    autoregressive_mode_only: bool = False,
    depth_img_prefix_name: str | None = None,
) -> GigaBrain0Pipeline:
    """Build and initialize a GigaBrain0 policy for inference.

    Args:
        model_path: Path to the model checkpoint directory.
        tokenizer_model_path: Path to the tokenizer model.
        fast_tokenizer_path: Path to the fast tokenizer model.
        embodiment_id: Embodiment identifier of the robot/task.
        norm_stats_path: Path to the JSON file containing normalization stats.
        delta_mask: Boolean mask indicating which action dimensions are delta-controlled.
        original_action_dim: Expected original action vector dimension.
        autoregressive_mode_only: If True, only use autoregressive mode to predict actions.
        depth_img_prefix_name: Optional prefix for depth image keys when depth is enabled.

    Returns:
        Initialized GigaBrain0Pipeline with CUDA device and compiled graph. Also binds
        a convenience `inference` method to the returned instance.
    """
    with open(norm_stats_path, 'r') as f:
        norm_stats_data = json.load(f)['norm_stats']

    pipe = GigaBrain0Pipeline(
        model_path=model_path,
        tokenizer_model_path=tokenizer_model_path,
        fast_tokenizer_path=fast_tokenizer_path,
        embodiment_id=embodiment_id,
        state_norm_stats=norm_stats_data['observation.state'],
        action_norm_stats=norm_stats_data['action'],
        delta_mask=delta_mask,
        original_action_dim=original_action_dim,
        autoregressive_inference_mode=autoregressive_mode_only,
        depth_img_prefix_name=depth_img_prefix_name,
    )
    pipe.to('cuda')
    if not autoregressive_mode_only:
        pipe.compile()

    def inference(self, data: dict[str, Any]) -> torch.Tensor:
        """Run policy inference to get the predicted action.

        Args:
            data: Input dictionary containing observation images, optional depth images,
                a task string under key 'task', and a state tensor under
                'observation.state'.

        Returns:
            Predicted action tensor produced by the policy.
        """
        images = {
            'observation.images.cam_high': data['observation.images.cam_high'],
            'observation.images.cam_left_wrist': data['observation.images.cam_left_wrist'],
            'observation.images.cam_right_wrist': data['observation.images.cam_right_wrist'],
        }
        if pipe.enable_depth_img and 'observation.depth_images.cam_high' in data:
            images['observation.depth_images.cam_high'] = data['observation.depth_images.cam_high']
        if pipe.enable_depth_img and 'observation.depth_images.cam_left_wrist' in data:
            images['observation.depth_images.cam_left_wrist'] = data['observation.depth_images.cam_left_wrist']
        if pipe.enable_depth_img and 'observation.depth_images.cam_right_wrist' in data:
            images['observation.depth_images.cam_right_wrist'] = data['observation.depth_images.cam_right_wrist']

        task = data['task']
        state = data['observation.state']

        pred_action = pipe(images, task, state, autoregressive_mode_only=autoregressive_mode_only)

        return pred_action

    pipe.inference = types.MethodType(inference, pipe)

    return pipe


def run_server(
    model_path: str,
    tokenizer_model_path: str,
    fast_tokenizer_path: str,
    embodiment_id: int,
    norm_stats_path: str,
    delta_mask: list[bool],
    original_action_dim: int,
    autoregressive_mode_only: bool,
    depth_img_prefix_name: str | None = None,
    host: str = '127.0.0.1',
    port: int = 8080,
) -> None:
    """Start a RobotInferenceServer with a configured GigaBrain0 policy.

    Args:
        model_path: Path to the model checkpoint directory.
        tokenizer_model_path: Path to the tokenizer model.
        fast_tokenizer_path: Path to the fast tokenizer model.
        embodiment_id: Embodiment identifier of the robot/task.
        norm_stats_path: Path to the JSON file containing normalization stats.
        delta_mask: Boolean mask indicating which action dimensions are delta-controlled.
        original_action_dim: Expected original action vector dimension.
        autoregressive_mode_only: If True, only use autoregressive mode to predict actions.
        depth_img_prefix_name: Optional prefix for depth image keys when depth is enabled.
        host: Host address to bind the inference server.
        port: TCP port to bind the inference server.
    """
    policy = get_policy(
        model_path=model_path,
        tokenizer_model_path=tokenizer_model_path,
        fast_tokenizer_path=fast_tokenizer_path,
        embodiment_id=embodiment_id,
        norm_stats_path=norm_stats_path,
        delta_mask=delta_mask,
        original_action_dim=original_action_dim,
        autoregressive_mode_only=autoregressive_mode_only,
        depth_img_prefix_name=depth_img_prefix_name,
    )

    server = RobotInferenceServer(policy, host=host, port=port)
    server.run()


if __name__ == '__main__':
    tyro.cli(run_server)
