import json
import os

import matplotlib
import numpy as np
import torch
import tyro

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from giga_datasets import load_dataset
from giga_models import GigaBrain0Pipeline


def inference_giga_brain_0(
    model_path: str,
    data_path: str,
    output_path: str,
    norm_stats_path: str,
    delta_mask: list[bool],
    embodiment_id: int,
    original_action_dim: int,
    action_chunk: int = 50,
    enable_2d_traj_output: bool = False,
    tokenizer_model_path: str = 'google/paligemma-3b-pt-224',
    fast_tokenizer_path: str = 'physical-intelligence/fast',
    depth_img_prefix_name: str | None = None,
    device: str = 'cuda:0',
):
    """Run action prediction inference with GigaBrain0 pipeline.

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
        depth_img_prefix_name=depth_img_prefix_name,
    )
    pipe.to(device)
    pipe.compile()

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
    indexes = range(0, 1000, 100)
    for idx in indexes:
        data = dataset[idx]

        images = {
            'observation.images.cam_high': data['observation.images.cam_high'],
            'observation.images.cam_left_wrist': data['observation.images.cam_left_wrist'],
            'observation.images.cam_right_wrist': data['observation.images.cam_right_wrist'],
        }
        if pipe.enable_depth_img:
            images[f'{depth_img_prefix_name}.cam_high'] = data[f'{depth_img_prefix_name}.cam_high']
            images[f'{depth_img_prefix_name}.cam_left_wrist'] = data[f'{depth_img_prefix_name}.cam_left_wrist']
            images[f'{depth_img_prefix_name}.cam_right_wrist'] = data[f'{depth_img_prefix_name}.cam_right_wrist']

        task = data['task']
        state = data['observation.state']

        if enable_2d_traj_output:
            pred_action, traj_pred = pipe(images, task, state, enable_2d_traj_output=enable_2d_traj_output)
        else:
            pred_action = pipe(images, task, state)

        if not output_path:
            continue

        action_names = None
        if 'meta' in data and 'names' in data['meta'].info['features']['action']:
            action_names = data['meta'].info['features']['action']['names']
        visualize_result(data['action'].numpy(), pred_action.numpy(), os.path.join(output_path, f'{idx}.png'), action_names)
        if enable_2d_traj_output:
            visualize_traj(images['observation.images.cam_high'], traj_pred.numpy(), os.path.join(output_path, f'{idx}_traj.png'))


def visualize_result(gt_action: np.ndarray, pred_action: np.ndarray, out_path: str, action_names: list[str] | None = None) -> None:
    """Visualize and compare ground-truth and predicted action trajectories.

    Args:
        gt_action: Ground-truth action tensor.
        pred_action: Predicted action tensor.
        out_path: File path to save the visualization.
        action_names: Optional list of names for each action dimension.
    """
    pred_action = pred_action[:, :14]
    gt_action = gt_action[:, :14]

    num_ts, num_dim = gt_action.shape
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

        if num_ts > 0:
            ax.scatter(time_axis[-1], gt_action[-1, dim_idx], color='red', s=50, zorder=5)
            ax.text(time_axis[-1], gt_action[-1, dim_idx], f' {gt_action[-1, dim_idx]:.3f}', verticalalignment='bottom', horizontalalignment='left')

            ax.scatter(time_axis[-1], pred_action[-1, dim_idx], color='blue', s=50, zorder=5)
            ax.text(time_axis[-1], pred_action[-1, dim_idx], f' {pred_action[-1, dim_idx]:.3f}', verticalalignment='top', horizontalalignment='left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_traj(images: np.ndarray, traj_pred: np.ndarray, out_path: str) -> None:
    """Visualize a 2D trajectory overlaid on an image.

    Args:
        images: The background image for the plot.
        traj_pred: The 2D trajectory prediction data.
        out_path: File path to save the visualization.
    """
    # Prepare background image (H, W, C) uint8
    img = images
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255.0).clip(0, 255).astype(np.uint8)

    H, W = img.shape[:2]

    # Prepare trajectory points
    traj = traj_pred.detach().cpu().numpy() if torch.is_tensor(traj_pred) else np.asarray(traj_pred)
    if traj.ndim == 1:
        traj = traj.reshape(1, 4)

    x1, y1, x2, y2 = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
    mask1 = np.isfinite(x1) & np.isfinite(y1)
    mask2 = np.isfinite(x2) & np.isfinite(y2)

    # Plot
    fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0), dpi=100)
    ax.imshow(img)
    ax.scatter(x1[mask1], y1[mask1], c='red', s=10)
    ax.scatter(x2[mask2], y2[mask2], c='red', s=10)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # keep origin at top-left to match image coordinates
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == '__main__':
    tyro.cli(inference_giga_brain_0)
