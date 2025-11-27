import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from giga_datasets import load_dataset
from giga_models import GigaBrain0Pipeline


def inference_discrete_action(
    model_path: str,
    data_path: str,
    output_path: str,
    norm_stats_path: str,
    delta_mask: list[bool],
    embodiment_id: int,
    original_action_dim: int,
    action_chunk: int = 50,
    tokenizer_model_path: str = 'google/paligemma-3b-pt-224',
    fast_tokenizer_path: str = 'physical-intelligence/fast',
    depth_img_prefix_name: str | None = None,
    device: str = 'cuda:0',
):
    """Run discrete action inference with GigaBrain0 VLM pipeline.

    Args:
        model_path: Base directory containing model checkpoints and artifacts.
        data_path: Path to the LeRobot dataset to evaluate on.
        output_path: Path to save the prediction results.
        norm_stats_path: Path to JSON file containing normalization statistics ('norm_stats').
        delta_mask: Boolean mask indicating which action dimensions use delta representation.
        embodiment_id: Integer ID specifying the robot embodiment/type. Currently 0 for AgileX and 1 for Agibot G1.
        original_action_dim: Dimension of the original action space used for state truncation.
        action_chunk: Temporal chunk size for delta computation in dataset loading.
        tokenizer_model_path: Path to the tokenizer model.
        fast_tokenizer_path: Path to the fast tokenizer.
        depth_img_prefix_name: Dataset key prefix for depth images; used when depth is enabled.
        device: Compute device to run inference on, e.g. 'cuda:0' or 'cpu'.
    """
    if device.startswith('cuda'):
        torch.cuda.set_device(device)
    os.makedirs(output_path, exist_ok=True)

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
        autoregressive_inference_mode=True,
        depth_img_prefix_name=depth_img_prefix_name,
    )
    pipe.to(device)

    data_or_config = [
        dict(
            _class_name='LeRobotDataset',
            data_path=data_path,
            delta_info={
                'action': action_chunk,
            },
            meta_name='meta',
        )
    ]
    dataset = load_dataset(data_or_config)

    # Create observation
    for idx in range(0, 10000, 1000):
        try:
            data = dataset[idx]
        except (IndexError, KeyError) as e:
            print(f'Warning: Failed to access dataset at index {idx}: {e}')
            continue

        images = {
            'observation.images.cam_high': data['observation.images.cam_high'],
            'observation.images.cam_left_wrist': data['observation.images.cam_left_wrist'],
            'observation.images.cam_right_wrist': data['observation.images.cam_right_wrist'],
        }

        task = data['task']
        state = data['observation.state'][:original_action_dim]

        actions = pipe(images, task, state, autoregressive_mode_only=True)

        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = actions

        gt_action_np = data['action'].numpy() if isinstance(data['action'], torch.Tensor) else data['action']

        visualize_result(gt_action_np, actions_np, os.path.join(output_path, f'{idx}.png'))


def visualize_result(gt_action: np.ndarray, pred_action: np.ndarray, out_path: str, action_names: list[str] | None = None) -> None:
    """Visualize and compare ground-truth and predicted action trajectories.

    Args:
        gt_action: Ground-truth action tensor.
        pred_action: Predicted action tensor.
        out_path: File path to save the visualization.
        action_names: Optional list of names for each action dimension.
    """
    min_dims = min(gt_action.shape[1], pred_action.shape[1])
    pred_action = pred_action[:, :min_dims]
    gt_action = gt_action[:, :min_dims]

    num_ts, num_dim = gt_action.shape

    if num_dim == 0:
        print(f'Warning: No dimensions to visualize for {out_path}')
        return

    if num_ts == 0:
        print(f'Warning: No time steps to visualize for {out_path}')
        return

    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(10, 2 * num_dim))

    time_axis = np.arange(num_ts) / 30.0

    colors = plt.cm.viridis(np.linspace(0, 1, num_dim))

    if action_names is None or len(action_names) == 0:
        action_names = [str(i) for i in range(num_dim)]

    dim_list = range(num_dim)
    for ax_idx, dim_idx in enumerate(dim_list):
        ax = axs[ax_idx]

        ax.plot(time_axis, gt_action[:, dim_idx], label='GT', color=colors[ax_idx], linewidth=2, linestyle='-')
        ax.plot(time_axis, pred_action[:, dim_idx], label='Pred', color=colors[ax_idx], linewidth=2, linestyle='--')

        ax.set_title(f'Joint {ax_idx}: {action_names[ax_idx]}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')

        ax.scatter(time_axis[-1], gt_action[-1, dim_idx], color='red', s=50, zorder=5)
        ax.text(time_axis[-1], gt_action[-1, dim_idx], f' {gt_action[-1, dim_idx]:.3f}', verticalalignment='bottom', horizontalalignment='left')

        ax.scatter(time_axis[-1], pred_action[-1, dim_idx], color='blue', s=50, zorder=5)
        ax.text(time_axis[-1], pred_action[-1, dim_idx], f' {pred_action[-1, dim_idx]:.3f}', verticalalignment='top', horizontalalignment='left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    tyro.cli(inference_discrete_action)
