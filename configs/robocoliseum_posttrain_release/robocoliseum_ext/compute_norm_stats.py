"""Compute normalization statistics for the unified 17-D RoboColiseum joint layout.

The preprocessing order matches training: repack the joint dimensions, convert
selected joint actions into deltas from the current state, and pad tensors to the
32-D model width used by GigaBrain-0.7. Instruction/robustness, spatial, and
manipulation suites have different distributions and require separate statistics.
"""

import os
import pathlib
import sys
from typing import Any

import numpy as np
import torch
import tyro
from giga_datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils import (
    DeltaActions,
    PadStatesAndActions,
)
from robocoliseum_ext.state_remap import (
    UNIFIED_ACTION_DIM,
    build_delta_mask,
    repack_action_state,
)

# Reuse the upstream statistics implementation from a user-selected checkout.
project_root = os.environ.get("GIGABRAIN_PROJECT_ROOT")
if not project_root:
    raise RuntimeError(
        "Set the GIGABRAIN_PROJECT_ROOT environment variable to the "
        "GigaBrain-0 project directory."
    )
UPSTREAM_SCRIPTS = pathlib.Path(project_root).expanduser() / "scripts"
sys.path.insert(0, str(UPSTREAM_SCRIPTS))
from compute_norm_stats import RunningStats, TransformDataset, serialize_json  # noqa: E402


MODEL_ACTION_DIM = 32


class RoboColiseumJointRepack:
    """Standalone joint repack for the norm-stats pipeline."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return repack_action_state(data, strict=True)


class SetEmbodimentId:
    """Assign a fixed embodiment id without relying on robot_type lookup."""

    def __init__(self, embodiment_id: int):
        self.embodiment_id = int(embodiment_id)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        data['embodiment_id'] = self.embodiment_id
        return data


def compute_norm_stats(
    data_paths: list[str],
    output_path: str,
    embodiment_id: int = 1,
    sample_rate: float = 1.0,
    action_chunk: int = 50,
    num_workers: int = 16,
    seed: int = 0,
) -> None:
    """Accumulate joint-space state/action statistics.

    Args:
        data_paths: LeRobot dataset roots belonging to exactly one suite.
        output_path: Destination JSON path unique to that suite and dimension.
        embodiment_id: GigaBrain embodiment id. RoboColiseum g2a_sim uses id 1.
        sample_rate: Fraction of frames to visit, in (0, 1].
        action_chunk: Action horizon; must match the training config.
        num_workers: DataLoader worker count.
        seed: Shuffle seed used when sample_rate is below 1.
    """
    if not 0.0 < sample_rate <= 1.0:
        raise ValueError(f'sample_rate must be in (0, 1], got {sample_rate}')

    schema_dim = UNIFIED_ACTION_DIM
    delta_masks = {
        int(embodiment_id): build_delta_mask(),
    }
    data_or_config = [
        dict(
            _class_name='LeRobotDataset',
            data_path=data_path,
            delta_info={'action': action_chunk},
            meta_name='meta',
            skip_video_decoding=True,
        )
        for data_path in data_paths
    ]
    dataset = load_dataset(data_or_config)

    data_transforms = [
        SetEmbodimentId(embodiment_id),
        RoboColiseumJointRepack(),
        DeltaActions(mask=delta_masks),
        PadStatesAndActions(action_dim=MODEL_ACTION_DIM),
    ]
    keys = ['observation.state', 'action']
    stats = {key: RunningStats() for key in keys}
    num_frames = int(sample_rate * len(dataset))
    print(
        'purpose=RoboColiseum GigaBrain-0.7 joint-space norm stats '
        f'schema_dim={schema_dim} model_dim={MODEL_ACTION_DIM} '
        f'datasets={len(data_paths)} total_frames={len(dataset)} '
        f'visited_frames={num_frames} seed={seed}'
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = DataLoader(
        TransformDataset(dataset, data_transforms, keys),
        batch_size=1,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    for batch_idx, batch_data in tqdm(enumerate(dataloader), total=num_frames):
        if batch_idx >= num_frames:
            break
        for key in keys:
            stats[key].update(batch_data[key][0].numpy())

    norm_stats = {key: accumulator.get_statistics() for key, accumulator in stats.items()}
    for key, value in norm_stats.items():
        print(
            f'{key}: mean[:4]={np.asarray(value.mean)[:4]} '
            f'q01[:4]={np.asarray(value.q01)[:4]} '
            f'q99[:4]={np.asarray(value.q99)[:4]}'
        )

    destination = pathlib.Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialize_json(norm_stats))
    print(f'wrote={destination}')


if __name__ == '__main__':
    tyro.cli(compute_norm_stats)
