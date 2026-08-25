import bisect
import gc
import os
import socket
from collections import OrderedDict
from typing import Any, Callable

import torch


DEFAULT_MAX_OPEN_PER_WORKER = 4
DEFAULT_GC_INTERVAL = 32
WORKER_PROFILE_KEY = '__giga_lerobot_open_cache_worker_profile__'
_WORKER_PROFILE_SUM_KEYS = (
    'getitem_count',
    'touch_count',
    'hit_count',
    'miss_count',
    'eviction_count',
    'gc_count',
)


def _attach_worker_profile(data: Any, profile: dict[str, Any]) -> None:
    if not profile:
        return
    if isinstance(data, dict):
        data[WORKER_PROFILE_KEY] = profile
        return
    if isinstance(data, (list, tuple)):
        if not data:
            return
        shared_profile = dict(profile)
        divisor = float(len(data))
        for key in _WORKER_PROFILE_SUM_KEYS:
            if isinstance(shared_profile.get(key), (int, float)):
                shared_profile[key] = float(shared_profile[key]) / divisor
        for item in data:
            _attach_worker_profile(item, shared_profile)


class LeRobotOpenCacheDataset(torch.utils.data.Dataset):
    """Bound the number of opened LeRobotDataset handles kept by a worker.

    giga_datasets.LeRobotDataset.close() currently clears data_path through
    BaseDataset.close(), so runtime eviction must release only the heavy opened
    FastLeRobotDataset object by setting child.dataset = None.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        max_open_per_worker: int = DEFAULT_MAX_OPEN_PER_WORKER,
        gc_interval: int = DEFAULT_GC_INTERVAL,
        profile_enabled: bool = False,
    ) -> None:
        self.dataset = dataset
        self.max_open_per_worker = int(max_open_per_worker)
        self.gc_interval = int(gc_interval)
        self.profile_enabled = bool(profile_enabled)
        if self.max_open_per_worker < 1:
            raise ValueError(f'max_open_per_worker must be positive, got {self.max_open_per_worker}')
        self._open_lerobot_children: OrderedDict[int, Any] = OrderedDict()
        self._evictions_since_gc = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int | list[int] | tuple[int, ...]) -> Any:
        if not self.profile_enabled:
            data = self.dataset[index]
            if isinstance(index, (list, tuple)):
                for idx in index:
                    self._touch_index(int(idx))
            else:
                self._touch_index(int(index))
            return data

        children = self._children_for_index(index)
        miss_count = 0
        hit_count = 0
        for child in children:
            if getattr(child, 'dataset', None) is None:
                miss_count += 1
            else:
                hit_count += 1

        data = self.dataset[index]

        evictions = 0
        gc_collects = 0
        for child in children:
            evicted, did_gc = self._touch_child(child)
            evictions += evicted
            gc_collects += int(did_gc)

        _attach_worker_profile(
            data,
            self._worker_profile(
                getitem_count=len(index) if isinstance(index, (list, tuple)) else 1,
                touch_count=len(children),
                hit_count=hit_count,
                miss_count=miss_count,
                eviction_count=evictions,
                gc_count=gc_collects,
            ),
        )
        return data

    def __getitems__(self, indices: list[int]) -> list[Any]:
        return self.__getitem__(indices)

    def _children_for_index(self, index: int | list[int] | tuple[int, ...]) -> list[Any]:
        if isinstance(index, (list, tuple)):
            children = []
            for idx in index:
                children.extend(self._iter_lerobot_children_for_index(self.dataset, int(idx)))
            return children
        else:
            return list(self._iter_lerobot_children_for_index(self.dataset, int(index)))

    def set_transform(self, transform: Any) -> None:
        self.dataset.set_transform(transform)

    def _touch_index(self, index: int) -> None:
        for child in self._iter_lerobot_children_for_index(self.dataset, index):
            self._touch_child(child)

    def _touch_child(self, child: Any) -> tuple[int, bool]:
        if getattr(child, 'dataset', None) is None:
            return 0, False

        child_id = id(child)
        self._open_lerobot_children.pop(child_id, None)
        self._open_lerobot_children[child_id] = child

        evictions = 0
        while len(self._open_lerobot_children) > self.max_open_per_worker:
            _, evicted_child = self._open_lerobot_children.popitem(last=False)
            if getattr(evicted_child, 'dataset', None) is not None:
                evicted_child.dataset = None
                evictions += 1
                self._evictions_since_gc += 1

        did_gc = False
        if self.gc_interval > 0 and self._evictions_since_gc >= self.gc_interval:
            gc.collect()
            self._evictions_since_gc = 0
            did_gc = True
        return evictions, did_gc

    def _worker_profile(
        self,
        *,
        getitem_count: int,
        touch_count: int,
        hit_count: int,
        miss_count: int,
        eviction_count: int,
        gc_count: int,
    ) -> dict[str, Any]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = -1 if worker_info is None else int(worker_info.id)
        worker_num = 0 if worker_info is None else int(worker_info.num_workers)
        return {
            'host': socket.gethostname(),
            'pid': int(os.getpid()),
            'rank': int(os.environ.get('RANK', '-1')),
            'local_rank': int(os.environ.get('LOCAL_RANK', '-1')),
            'worker_id': worker_id,
            'worker_num': worker_num,
            'open_cache_size': float(len(self._open_lerobot_children)),
            'max_open_per_worker': float(self.max_open_per_worker),
            'getitem_count': float(getitem_count),
            'touch_count': float(touch_count),
            'hit_count': float(hit_count),
            'miss_count': float(miss_count),
            'eviction_count': float(eviction_count),
            'gc_count': float(gc_count),
        }

    def _iter_lerobot_children_for_index(self, dataset: Any, index: int):
        if dataset.__class__.__name__ == 'LeRobotDataset':
            yield dataset
            return

        children = getattr(dataset, 'datasets', None)
        if not children:
            return

        if not hasattr(dataset, '_get_cumulative_sizes'):
            for child in children:
                yield from self._iter_lerobot_children_for_index(child, index)
            return

        cumulative_sizes = self._get_cumulative_sizes(dataset, children)
        child_index = bisect.bisect_right(cumulative_sizes, index)
        if child_index >= len(cumulative_sizes):
            return
        previous_size = 0 if child_index == 0 else cumulative_sizes[child_index - 1]
        yield from self._iter_lerobot_children_for_index(children[child_index], index - previous_size)

    @staticmethod
    def _get_cumulative_sizes(dataset: Any, children: list[Any]) -> list[int]:
        if hasattr(dataset, '_get_cumulative_sizes'):
            return dataset._get_cumulative_sizes()
        cumulative_sizes: list[int] = []
        total_size = 0
        for child in children:
            total_size += len(child)
            cumulative_sizes.append(total_size)
        return cumulative_sizes


def wrap_lerobot_open_cache(
    dataset: Any,
    cache_cfg: Any,
    log_fn: Callable[[str], None] | None = None,
    profile_enabled: bool = False,
) -> Any:
    if not isinstance(cache_cfg, dict):
        return dataset

    open_cache_cfg = cache_cfg.get('lerobot_open_cache', True)
    if open_cache_cfg is False or open_cache_cfg is None:
        return dataset

    if open_cache_cfg is True:
        max_open_per_worker = DEFAULT_MAX_OPEN_PER_WORKER
        gc_interval = DEFAULT_GC_INTERVAL
    elif isinstance(open_cache_cfg, int):
        max_open_per_worker = open_cache_cfg
        gc_interval = DEFAULT_GC_INTERVAL
    elif isinstance(open_cache_cfg, dict):
        max_open_per_worker = int(open_cache_cfg.get('max_open_per_worker', DEFAULT_MAX_OPEN_PER_WORKER))
        gc_interval = int(open_cache_cfg.get('gc_interval', DEFAULT_GC_INTERVAL))
    else:
        raise TypeError(
            'dataloaders.train.cache.lerobot_open_cache should be a bool, int, or dict, '
            f'got {type(open_cache_cfg).__name__}'
        )

    if log_fn is not None:
        log_fn(
            'Enable LeRobot open dataset LRU: '
            f'max_open_per_worker={max_open_per_worker}, gc_interval={gc_interval}'
        )
    return LeRobotOpenCacheDataset(
        dataset,
        max_open_per_worker=max_open_per_worker,
        gc_interval=gc_interval,
        profile_enabled=profile_enabled,
    )
