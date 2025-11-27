"""Script to convert HDF5 data to the LeRobot dataset v2.1 format."""

import dataclasses
from pathlib import Path
from typing import Dict, List, Literal

import h5py
import numpy as np
import psutil
import torch
import tqdm
import tyro
from giga_datasets.datasets.lerobot_dataset import FastLeRobotDataset


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """Configuration for LeRobot dataset creation.

    Attributes:
        use_videos: Whether to use video format for images.
        tolerance_s: Time tolerance in seconds for frame synchronization.
        image_writer_processes: Number of processes for writing images.
        image_writer_threads: Number of threads per image writer process.
        video_backend: Video backend to use (None for default).
    """

    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def get_cpu_memory(unit: str = 'GB') -> str:
    """Get current CPU memory usage as a formatted string.

    Args:
        unit: Unit for memory measurement ('GB', 'MB', 'KB', or 'B').

    Returns:
        Formatted string showing used/total memory in the specified unit.
    """
    factors = {
        'GB': 1024 * 1024 * 1024,
        'MB': 1024 * 1024,
        'KB': 1024,
        'B': 1,
    }
    factor = factors[unit]
    mem_info = psutil.virtual_memory()
    mem_total = mem_info.total / factor
    mem_used = mem_info.used / factor
    msg = f'{mem_used:.2f}/{mem_total:.2f} {unit}'
    return msg


def create_empty_dataset(
    out_dir: Path,
    repo_id: str,
    robot_type: str,
    mode: Literal['video', 'image'] = 'image',
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> FastLeRobotDataset:
    """Create an empty LeRobot dataset with predefined features.

    Args:
        out_dir: Output directory path for the dataset.
        repo_id: Repository ID for the dataset.
        robot_type: Type of robot (e.g., 'agilex_cobot_magic').
        mode: Data storage mode, either 'video' or 'image'.
        dataset_config: Configuration for dataset creation.

    Returns:
        Newly created empty FastLeRobotDataset instance.
    """
    motors = [
        'left_waist',
        'left_shoulder',
        'left_elbow',
        'left_forearm_roll',
        'left_wrist_angle',
        'left_wrist_rotate',
        'left_gripper',
        'right_waist',
        'right_shoulder',
        'right_elbow',
        'right_forearm_roll',
        'right_wrist_angle',
        'right_wrist_rotate',
        'right_gripper',
    ]
    cameras = [
        'cam_high',
        'cam_left_wrist',
        'cam_right_wrist',
    ]

    features = {
        'observation.state': {
            'dtype': 'float32',
            'shape': (len(motors),),
            'names': motors,
        },
        'action': {
            'dtype': 'float32',
            'shape': (len(motors) + 2,),
            'names': motors + ['linear_x', 'angular_z'],
        },
    }

    for cam in cameras:
        features[f'observation.images.{cam}'] = {
            'dtype': mode,
            'shape': (3, 480, 640),
            'names': [
                'channels',
                'height',
                'width',
            ],
        }

    lerobot_dataset = FastLeRobotDataset.create(
        root=out_dir,
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )
    return lerobot_dataset


def get_cameras(hdf5_files: List[Path]) -> List[str]:
    """Extract camera names from the first HDF5 file.

    Args:
        hdf5_files: List of HDF5 file paths.

    Returns:
        List of camera names excluding depth cameras.
    """
    with h5py.File(hdf5_files[0], 'r') as ep:
        return [key for key in ep['/observations/images'].keys() if 'depth' not in key]  # noqa: SIM118


def has_velocity(hdf5_files: List[Path]) -> bool:
    """Check if velocity data exists in the HDF5 files.

    Args:
        hdf5_files: List of HDF5 file paths.

    Returns:
        True if velocity observations exist, False otherwise.
    """
    with h5py.File(hdf5_files[0], 'r') as ep:
        return '/observations/qvel' in ep


def has_effort(hdf5_files: List[Path]) -> bool:
    """Check if effort data exists in the HDF5 files.

    Args:
        hdf5_files: List of HDF5 file paths.

    Returns:
        True if effort observations exist, False otherwise.
    """
    with h5py.File(hdf5_files[0], 'r') as ep:
        return '/observations/effort' in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: List[str]) -> Dict[str, np.ndarray]:
    """Load image data for all specified cameras from an HDF5 episode file.

    Args:
        ep: Opened HDF5 file object containing episode data.
        cameras: List of camera names to load images from.

    Returns:
        Dictionary mapping camera names to numpy arrays of images.
    """
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f'/observations/images/{camera}'].ndim == 4

        if uncompressed:
            imgs_array = ep[f'/observations/images/{camera}'][:]
        else:
            import cv2

            imgs_array = []
            for data in ep[f'/observations/images/{camera}']:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Load all data from a single episode HDF5 file.

    Args:
        ep_path: Path to the episode HDF5 file.

    Returns:
        Tuple containing:
            - Dictionary of camera names to image arrays
            - State tensor (joint positions)
            - Action tensor (including base actions)
            - Velocity tensor (None if not available)
            - Effort tensor (None if not available)
    """
    with h5py.File(ep_path, 'r') as ep:
        qpos_data = ep['/observations/qpos'][:]
        start_idx = 0
        end_idx = qpos_data.shape[0]
        state = torch.from_numpy(qpos_data[start_idx:end_idx])
        action = torch.from_numpy(ep['/action'][start_idx:end_idx])
        if '/base_action' in ep:
            base_action = torch.from_numpy(ep['/base_action'][start_idx:end_idx])
        else:
            base_action = torch.zeros((end_idx - start_idx, 2))
        action = torch.cat((action, base_action), dim=1)
        velocity = None
        effort = None

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                'cam_high',
                'cam_left_wrist',
                'cam_right_wrist',
            ],
        )
        for cam in imgs_per_cam:
            imgs_per_cam[cam] = imgs_per_cam[cam][start_idx:end_idx]

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: FastLeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> FastLeRobotDataset:
    """Populate a LeRobot dataset with data from HDF5 episode files.

    Args:
        dataset: Empty FastLeRobotDataset to populate.
        hdf5_files: List of HDF5 episode file paths.
        task: Task name to assign to all episodes.
        episodes: Optional list of episode indices to process. If None, processes all episodes.

    Returns:
        Populated FastLeRobotDataset instance.
    """
    if episodes is None:
        episodes = list(range(len(hdf5_files)))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        try:
            imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)

            num_frames = state.shape[0]

            for i in range(num_frames):
                frame = {
                    'observation.state': state[i],
                    'action': action[i],
                }

                for camera, img_array in imgs_per_cam.items():
                    frame[f'observation.images.{camera}'] = img_array[i]

                final_task = task

                dataset.add_frame(frame, final_task)

            dataset.save_episode()

            print(f'{ep_idx} Done, get_cpu_memory: {get_cpu_memory()}')
        except Exception as e:
            print(f'{ep_idx} Error: {e}')
            continue

    return dataset


def convert_lerobot(
    data_path: Path,
    out_dir: Path,
    task: str,
) -> None:
    """Convert raw HDF5 data to LeRobot dataset.

    Args:
        data_path: Path to directory containing HDF5 files.
        out_dir: Output directory for the dataset.
        task: Task name for the dataset.
    """

    data_files = sorted(
        [p for p in data_path.glob('*.hdf5') if p.is_file()], key=lambda x: int(x.stem.split('_')[-1].split('.')[0])  # e.g. episode_1.hdf5
    )

    dataset = create_empty_dataset(
        out_dir,
        'giga-brain/agilex_example',
        robot_type='agilex_cobot_magic',
        mode='video',
    )

    dataset = populate_dataset(
        dataset,
        data_files,
        task=task,
    )


if __name__ == '__main__':
    tyro.cli(convert_lerobot)
