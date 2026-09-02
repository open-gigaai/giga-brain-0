# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# Released by Spirit AI Team.
# ==============================================================================

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"


class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    IDENTITY = "IDENTITY"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple


def create_stats_buffers(
    features: Dict[str, PolicyFeature],
    norm_map: Dict[str, NormalizationMode],
    stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> Dict[str, Dict[str, nn.ParameterDict]]:
    stats_buffers = {}
    for key, ft in features.items():
        norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if norm_mode is NormalizationMode.IDENTITY:
            continue
        assert isinstance(norm_mode, NormalizationMode)
        if norm_mode is not NormalizationMode.MIN_MAX:
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")
        shape = tuple(ft.shape)
        if ft.type is FeatureType.VISUAL:
            assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            shape = (c, 1, 1)
        min_v = torch.ones(shape, dtype=torch.float32) * torch.inf
        max_v = torch.ones(shape, dtype=torch.float32) * torch.inf
        buffer = nn.ParameterDict(
            {"min": nn.Parameter(min_v, requires_grad=False), "max": nn.Parameter(max_v, requires_grad=False)}
        )
        if stats is not None:
            if key not in stats:
                raise ValueError(f"Missing stats for feature `{key}` (expected `min`/`max`).")
            if "min" not in stats[key] or "max" not in stats[key]:
                raise ValueError(f"Stats for `{key}` must contain `min` and `max` for MIN_MAX normalization.")
            min_src, max_src = stats[key]["min"], stats[key]["max"]
            if isinstance(min_src, np.ndarray) and isinstance(max_src, np.ndarray):
                buffer["min"].data = torch.from_numpy(min_src).to(dtype=torch.float32)
                buffer["max"].data = torch.from_numpy(max_src).to(dtype=torch.float32)
            elif isinstance(min_src, torch.Tensor) and isinstance(max_src, torch.Tensor):
                buffer["min"].data = min_src.clone().to(dtype=torch.float32)
                buffer["max"].data = max_src.clone().to(dtype=torch.float32)
            else:
                raise ValueError(f"Unexpected stats type for `{key}`: min={type(min_src)}, max={type(max_src)}")
        stats_buffers[key] = buffer
    return stats_buffers


def no_stats_error_str(name: str) -> str:
    return f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a pretrained model."


def build_norm_state(
    features: Dict[str, PolicyFeature],
    norm_map: Dict[str, NormalizationMode],
    stats: Optional[Dict[str, dict[str, torch.Tensor]]] = None,
) -> Tuple[Dict[FeatureType, NormalizationMode], Dict[str, nn.ParameterDict]]:
    norm_mode_map: dict[FeatureType, NormalizationMode] = {}
    for k, v in (norm_map or {}).items():
        ft = k if isinstance(k, FeatureType) else FeatureType(k)
        mode = v if isinstance(v, NormalizationMode) else NormalizationMode(v)
        if mode not in (NormalizationMode.IDENTITY, NormalizationMode.MIN_MAX):
            raise ValueError(f"Unsupported normalization mode: {mode}")
        norm_mode_map[ft] = mode
    stats_buffers = create_stats_buffers(features, norm_mode_map, stats)
    return norm_mode_map, stats_buffers


class _LowDimDatasetView(Dataset):
    """Wrap a dataset to expose only state/actions for norm-stats sampling."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.get_lowdim_item(idx)


def compute_norm_stats(
    dataset,
    num_samples: int = 20000,
    batch_size: int = 32,
    num_workers: int = 2,
) -> dict:
    """
    Returns:
        {
            "state_min": Tensor[14],
            "state_max": Tensor[14],
            "action_min": Tensor[14],
            "action_max": Tensor[14],
        }
    """
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    stats = None

    if rank == 0:
        print(f"Computing norm stats from {num_samples} samples...")

        dataset_size = len(dataset)
        num_samples = min(num_samples, dataset_size)
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(dataset_size, size=num_samples, replace=False).tolist()
        if hasattr(dataset, "get_lowdim_item") and hasattr(dataset, "collate_lowdim_fn"):
            source_dataset = _LowDimDatasetView(dataset)
            collate_fn = dataset.collate_lowdim_fn
        else:
            source_dataset = dataset
            collate_fn = dataset.collate_fn

        subset = Subset(source_dataset, indices)
        dataloader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        all_states = []
        all_actions = []
        for batch in tqdm(dataloader, desc="Loading samples"):
            all_states.append(batch["observation.state"])
            all_actions.append(batch["action"])
        all_states = torch.cat(all_states, dim=0)
        all_actions = torch.cat(all_actions, dim=0)
        state_values = all_states.numpy().reshape(-1, all_states.shape[-1]).astype(np.float64)
        action_values = all_actions.numpy().reshape(-1, all_actions.shape[-1]).astype(np.float64)

        stats = {
            "state_min": torch.from_numpy(np.percentile(state_values, 1, axis=0).astype(np.float32)),
            "state_max": torch.from_numpy(np.percentile(state_values, 99, axis=0).astype(np.float32)),
            "action_min": torch.from_numpy(np.percentile(action_values, 1, axis=0).astype(np.float32)),
            "action_max": torch.from_numpy(np.percentile(action_values, 99, axis=0).astype(np.float32)),
        }
        print(
            f"State percentile range [p1, p99]: "
            f"[{stats['state_min'].min():.4f}, {stats['state_max'].max():.4f}]"
        )
        print(
            f"Action percentile range [p1, p99]: "
            f"[{stats['action_min'].min():.4f}, {stats['action_max'].max():.4f}]"
        )
    if is_dist:
        object_list = [stats]
        dist.broadcast_object_list(
            object_list,
            src=0,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        stats = object_list[0]
        
    return stats
