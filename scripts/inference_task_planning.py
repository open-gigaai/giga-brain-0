import json

import torch
import tyro
from giga_datasets import load_dataset
from giga_models import GigaBrain0Pipeline


def inference_task_planning(
    model_path: str,
    data_path: str,
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
    """Run task-planning inference with GigaBrain0 VLM pipeline.

    Args:
        model_path: Base directory containing model checkpoints and artifacts.
        data_path: Path to the LeRobot dataset to evaluate on.
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
        discrete_state_input=False,
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
        data = dataset[idx]

        images = {
            'observation.images.cam_high': data['observation.images.cam_high'],
            'observation.images.cam_left_wrist': data['observation.images.cam_left_wrist'],
            'observation.images.cam_right_wrist': data['observation.images.cam_right_wrist'],
        }

        task = data['task']

        subtask = pipe.predict_current_subtask(images, task)[0]

        pairs = task.lower().strip().split(' subtask: ')
        main_task = pairs[0]
        sub_task = pairs[1] if len(pairs) > 1 else 'None'
        print('#' * 40)
        print('Main task:', main_task)
        print('Subtask[GT]:', sub_task)
        print('Subtask[Pred]:', subtask)


if __name__ == '__main__':
    tyro.cli(inference_task_planning)
