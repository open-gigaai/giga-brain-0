import bisect
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch


@dataclass
class _OpenedMaterializedShard:
    shard_id: int
    dataset: Any
    sources: dict[int, Any]
    source_row_offsets: dict[int, int]
    logical_index_maps: dict[int, '_SparseLogicalIndexMap']


@dataclass(frozen=True)
class _SparseLogicalIndexMap:
    """Map source-local logical indexes to physical rows using contiguous runs."""

    logical_starts: np.ndarray
    logical_ends: np.ndarray
    row_starts: np.ndarray

    @classmethod
    def from_values(cls, values: Any, *, row_start: int = 0) -> '_SparseLogicalIndexMap':
        logical_values = np.asarray(values)
        if logical_values.ndim != 1:
            raise ValueError(f'logical index values should be one-dimensional, got shape={logical_values.shape}')
        logical_values = logical_values.astype(np.int64, copy=False)
        if logical_values.size == 0:
            empty = np.empty(0, dtype=np.int64)
            return cls(empty, empty.copy(), empty.copy())

        deltas = np.diff(logical_values)
        non_increasing = np.flatnonzero(deltas <= 0)
        if non_increasing.size:
            position = int(non_increasing[0] + 1)
            raise ValueError(
                'logical index values should be strictly increasing; '
                f'values[{position - 1}:{position + 1}]={logical_values[position - 1:position + 1].tolist()}'
            )

        run_start_positions = np.concatenate(
            (np.array([0], dtype=np.int64), np.flatnonzero(deltas != 1).astype(np.int64) + 1)
        )
        run_end_positions = np.concatenate(
            (run_start_positions[1:], np.array([logical_values.size], dtype=np.int64))
        )
        return cls(
            logical_starts=logical_values[run_start_positions].copy(),
            logical_ends=(logical_values[run_end_positions - 1] + 1).copy(),
            row_starts=(run_start_positions + int(row_start)).copy(),
        )

    def row_for(self, logical_index: int) -> int | None:
        run_index = int(np.searchsorted(self.logical_starts, int(logical_index), side='right')) - 1
        if run_index < 0 or logical_index >= int(self.logical_ends[run_index]):
            return None
        return int(self.row_starts[run_index]) + int(logical_index) - int(self.logical_starts[run_index])


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {'0', 'false', 'no', 'off'}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == '':
        return default
    return int(value)


@dataclass(frozen=True)
class _MaterializedSource:
    source_id: int
    data_path: str
    local_start: int
    local_end: int
    global_start: int
    global_end: int


@dataclass(frozen=True)
class _MaterializedShard:
    shard_id: int
    group_index: int
    start: int
    end: int
    path: str
    sources: tuple[_MaterializedSource, ...]


def _require_lerobot_helpers():
    from giga_datasets.datasets.lerobot_dataset import (  # type: ignore
        BoundedVideoDecoderCache,
        _SampleLeRobotMetadata,
        _giga_attach_pipeline_profile,
        _giga_pipeline_profile_enabled,
        _giga_profile_add,
        _build_delta_timestamps,
        _load_lerobot_episode_frame_ranges_no_hf_cache,
        _load_lerobot_hf_parquet_dataset,
        _load_lerobot_info_no_hf_cache,
        _normalize_lerobot_episode_frame_ranges,
        _resolve_lerobot_data_columns,
        decode_video_frames_torchcodec,
        _LeRobotDatasetMetadata,
    )

    return dict(
        SampleLeRobotMetadata=_SampleLeRobotMetadata,
        BoundedVideoDecoderCache=BoundedVideoDecoderCache,
        attach_profile=_giga_attach_pipeline_profile,
        pipeline_profile_enabled=_giga_pipeline_profile_enabled,
        profile_add=_giga_profile_add,
        build_delta_timestamps=_build_delta_timestamps,
        load_episode_ranges=_load_lerobot_episode_frame_ranges_no_hf_cache,
        load_hf_parquet=_load_lerobot_hf_parquet_dataset,
        load_info=_load_lerobot_info_no_hf_cache,
        normalize_episode_ranges=_normalize_lerobot_episode_frame_ranges,
        resolve_columns=_resolve_lerobot_data_columns,
        decode_video_frames_torchcodec=decode_video_frames_torchcodec,
        LeRobotDatasetMetadata=_LeRobotDatasetMetadata,
    )


def _is_lerobot_dataset(dataset: Any) -> bool:
    return dataset.__class__.__name__ == 'LeRobotDataset' and hasattr(dataset, 'data_path')


def _iter_materialized_group_shards(dataset: Any, children_per_shard: int) -> Iterator[_MaterializedShard]:
    shard_id = 0
    global_group_start = 0
    for group_index, group in enumerate(getattr(dataset, 'datasets', []) or []):
        children = list(getattr(group, 'datasets', []) or [])
        if not children:
            global_group_start += len(group)
            continue
        child_lengths = [len(child) for child in children]
        child_starts = [0]
        for length in child_lengths[:-1]:
            child_starts.append(child_starts[-1] + length)
        for child_start in range(0, len(children), children_per_shard):
            child_end = min(child_start + children_per_shard, len(children))
            shard_local_start = child_starts[child_start]
            shard_local_end = child_starts[child_end - 1] + child_lengths[child_end - 1]
            sources = []
            for source_id, child_index in enumerate(range(child_start, child_end)):
                child = children[child_index]
                if not _is_lerobot_dataset(child):
                    raise TypeError(
                        'materialized_shard_reader currently supports LeRobotDataset children only; '
                        f'got {child.__class__.__name__}'
                    )
                local_start = child_starts[child_index]
                local_end = local_start + child_lengths[child_index]
                sources.append(
                    _MaterializedSource(
                        source_id=source_id,
                        data_path=os.fspath(child.data_path),
                        local_start=local_start,
                        local_end=local_end,
                        global_start=global_group_start + local_start,
                        global_end=global_group_start + local_end,
                    )
                )
            yield _MaterializedShard(
                shard_id=shard_id,
                group_index=group_index,
                start=global_group_start + shard_local_start,
                end=global_group_start + shard_local_end,
                path=f'shards/shard_{shard_id:06d}.parquet',
                sources=tuple(sources),
            )
            shard_id += 1
        global_group_start += len(group)


def _materialize_one_shard(
    shard: _MaterializedShard,
    *,
    output_dir: Path,
    source_datasets: list[Any],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    def normalize_fixed_size_list_columns(table: pa.Table) -> pa.Table:
        columns = []
        changed = False
        for column in table.itercolumns():
            if not pa.types.is_fixed_size_list(column.type):
                columns.append(column)
                continue
            changed = True
            chunks = []
            for chunk in column.chunks:
                list_size = chunk.type.list_size
                offsets = pa.array([i * list_size for i in range(len(chunk) + 1)], type=pa.int32())
                chunks.append(pa.ListArray.from_arrays(offsets, chunk.values))
            columns.append(pa.chunked_array(chunks))
        if not changed:
            return table
        return pa.table(columns, names=table.column_names)

    helpers = _require_lerobot_helpers()
    tables = []
    for source in shard.sources:
        child = source_datasets[source.source_id]
        root = Path(source.data_path)
        info = helpers['load_info'](root)
        delta_timestamps = helpers['build_delta_timestamps'](child.delta_info, info['fps'])
        hf_dataset = helpers['load_hf_parquet'](
            root=root,
            repo_id=os.path.basename(os.path.normpath(os.fspath(root))),
            features_info=info['features'],
            episodes=child.kwargs.get('episodes'),
            delta_timestamps=delta_timestamps,
            repack_transform=child.repack_transform,
            decode_video_keys=child.decode_video_keys,
            data_columns=child.data_columns,
        )
        table = hf_dataset.data.table
        table = table.append_column('__source_id', pa.array([source.source_id] * len(table), type=pa.int16()))
        table = normalize_fixed_size_list_columns(table)
        tables.append(table)
    shard_path = output_dir / shard.path
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables(tables, promote_options='default'), shard_path)


def materialize_lerobot_shards(
    dataset: Any,
    *,
    output_dir: str | os.PathLike[str],
    children_per_shard: int = 8,
    overwrite: bool = False,
    max_shards: int | None = None,
    log_fn: Any | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    manifest_path = output_path / 'manifest.json'
    if manifest_path.exists() and not overwrite:
        return json.loads(manifest_path.read_text())

    if children_per_shard <= 0:
        raise ValueError('children_per_shard should be positive')
    transform = getattr(dataset, 'transform', None)
    if transform is not None and hasattr(dataset, 'set_transform'):
        dataset.set_transform(None)
    shards = list(_iter_materialized_group_shards(dataset, children_per_shard))
    if max_shards is not None:
        shards = shards[: int(max_shards)]

    children_by_group = [list(getattr(group, 'datasets', []) or []) for group in getattr(dataset, 'datasets', []) or []]
    manifest = {
        'version': 1,
        'kind': 'materialized_lerobot_shard_v1',
        'length': len(dataset),
        'children_per_shard': children_per_shard,
        'shard_count': len(shards),
        'shards': [],
    }
    output_path.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        if log_fn is not None:
            log_fn(f'materialize shard {shard.shard_id}/{len(shards)} range=[{shard.start},{shard.end})')
        group_children = children_by_group[shard.group_index]
        source_children = []
        for source in shard.sources:
            child = next(
                child for child in group_children
                if os.fspath(child.data_path) == source.data_path and len(child) == source.local_end - source.local_start
            )
            source_children.append(child)
        _materialize_one_shard(shard, output_dir=output_path, source_datasets=source_children)
        manifest['shards'].append(
            {
                'shard_id': shard.shard_id,
                'group_index': shard.group_index,
                'start': shard.start,
                'end': shard.end,
                'path': shard.path,
                'sources': [source.__dict__ for source in shard.sources],
            }
        )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    if transform is not None and hasattr(dataset, 'set_transform'):
        dataset.set_transform(transform)
    return manifest


class _ShardSourceState:
    def __init__(self, source: _MaterializedSource, template_child: Any) -> None:
        helpers = _require_lerobot_helpers()
        self.source = source
        self.root = Path(source.data_path)
        self.repo_id = os.path.basename(os.path.normpath(source.data_path))
        self.meta = helpers['LeRobotDatasetMetadata'](self.repo_id, root=self.root)
        self.sample_meta = helpers['SampleLeRobotMetadata'](self.meta)
        self.info = helpers['load_info'](self.root)
        self.fps = self.info['fps']
        self.delta_info = template_child.delta_info
        self.delta_offsets = self._build_delta_offsets(template_child.delta_info)
        self.decode_video_keys = template_child.decode_video_keys
        self.repack_transform = template_child.repack_transform
        self.repack_pairs = self._build_repack_pairs(self.repack_transform)
        self.episode_ranges = helpers['normalize_episode_ranges'](helpers['load_episode_ranges'](self.root))
        self.video_key_set = self._resolve_video_key_set()

    @staticmethod
    def _build_delta_offsets(delta_info: Mapping[str, Any] | None) -> dict[str, list[int]]:
        if not delta_info:
            return {}
        offsets = {}
        for key, spec in delta_info.items():
            if isinstance(spec, int):
                offsets[key] = list(range(spec))
            elif isinstance(spec, Mapping) and 'offsets' in spec:
                offsets[key] = [int(v) for v in spec['offsets']]
            elif isinstance(spec, Mapping):
                offsets[key] = list(range(int(spec['start']), int(spec['stop']), int(spec.get('stride', 1))))
            else:
                offsets[key] = [int(v) for v in spec]
        return offsets

    @staticmethod
    def _build_repack_pairs(repack_transform: Mapping | None) -> tuple[tuple[str, str], ...]:
        if not repack_transform or any(isinstance(value, Mapping) for value in repack_transform.values()):
            return ()
        pairs = [(str(src), str(dst)) for src, dst in repack_transform.items() if src != dst]
        for src, dst in tuple(pairs):
            if src.endswith('_is_pad') or dst.endswith('_is_pad'):
                continue
            if (f'{src}_is_pad', f'{dst}_is_pad') not in pairs:
                pairs.append((f'{src}_is_pad', f'{dst}_is_pad'))
        return tuple(pairs)

    def _resolve_video_key_set(self) -> set[str] | None:
        if self.decode_video_keys is None:
            return None
        available = set(self.meta.video_keys)
        resolved = set()
        for key in self.decode_video_keys:
            candidates = [key]
            for left, right in self.repack_pairs:
                if left == key:
                    candidates.append(right)
                if right == key:
                    candidates.append(left)
            match = next((candidate for candidate in candidates if candidate in available), None)
            if match is not None:
                resolved.add(match)
        return resolved


class MaterializedShardLeRobotDataset:
    def __init__(
        self,
        dataset: Any,
        manifest_dir: str | os.PathLike[str],
        *,
        profile_enabled: bool | None = None,
        shard_cache_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.manifest_dir = Path(manifest_dir)
        manifest = json.loads((self.manifest_dir / 'manifest.json').read_text())
        self.length = int(manifest['length'])
        self.shards = manifest['shards']
        self._shard_ends = [int(shard['end']) for shard in self.shards]
        self.transform = getattr(dataset, 'transform', None)
        self.profile_enabled = profile_enabled
        self.shard_cache_size = max(1, int(shard_cache_size))
        self._opened_shards: OrderedDict[int, _OpenedMaterializedShard] = OrderedDict()
        self._source_templates = self._build_source_templates()
        decoder_cache_size = _env_int('GIGA_MATERIALIZED_TORCHCODEC_DECODER_CACHE_SIZE', 128)
        self._torchcodec_decoder_cache = _require_lerobot_helpers()['BoundedVideoDecoderCache'](
            max_size=decoder_cache_size
        )

    def _build_source_templates(self) -> dict[str, Any]:
        templates = {}
        for group in getattr(self.dataset, 'datasets', []) or []:
            for child in getattr(group, 'datasets', []) or []:
                if _is_lerobot_dataset(child):
                    templates[os.fspath(child.data_path)] = child
        return templates

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __len__(self) -> int:
        return self.length

    def set_transform(self, transform: Any) -> None:
        self.transform = transform

    def _find_shard(self, index: int) -> dict[str, Any]:
        shard_idx = bisect.bisect_right(self._shard_ends, index)
        if shard_idx >= len(self.shards):
            raise IndexError(index)
        shard = self.shards[shard_idx]
        if not (int(shard['start']) <= index < int(shard['end'])):
            raise IndexError(index)
        return shard

    def _open_shard(self, shard: dict[str, Any]) -> tuple[_OpenedMaterializedShard, float, bool, bool]:
        import datasets
        from lerobot.datasets.utils import hf_transform_to_torch

        shard_id = int(shard['shard_id'])
        cached = self._opened_shards.get(shard_id)
        if cached is not None:
            self._opened_shards.move_to_end(shard_id)
            return cached, 0.0, True, False
        tic = time.perf_counter()
        opened_dataset = datasets.Dataset.from_parquet(str(self.manifest_dir / shard['path']))
        opened_dataset.set_transform(hf_transform_to_torch)
        opened_sources = {}
        source_row_offsets = {}
        row_offset = 0
        for source_cfg in shard['sources']:
            source = _MaterializedSource(**source_cfg)
            template = self._source_templates[source.data_path]
            opened_sources[source.source_id] = _ShardSourceState(source, template)
            source_row_offsets[source.source_id] = row_offset
            row_offset += source.local_end - source.local_start
        state = _OpenedMaterializedShard(
            shard_id=shard_id,
            dataset=opened_dataset,
            sources=opened_sources,
            source_row_offsets=source_row_offsets,
            logical_index_maps={},
        )
        self._opened_shards[shard_id] = state
        evicted = False
        while len(self._opened_shards) > self.shard_cache_size:
            self._opened_shards.popitem(last=False)
            evicted = True
        return state, time.perf_counter() - tic, False, evicted

    @staticmethod
    def _scalar(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.item()
        if hasattr(value, 'item'):
            return value.item()
        return value

    @staticmethod
    def _get_logical_index_map(
        opened_shard: _OpenedMaterializedShard,
        source_id: int,
    ) -> _SparseLogicalIndexMap:
        cached = opened_shard.logical_index_maps.get(source_id)
        if cached is not None:
            return cached

        state = opened_shard.sources[source_id]
        row_start = opened_shard.source_row_offsets[source_id]
        row_count = state.source.local_end - state.source.local_start
        index_column = opened_shard.dataset.data.table.column('index')
        logical_values = index_column.slice(row_start, row_count).combine_chunks().to_numpy(zero_copy_only=False)
        if len(logical_values) != row_count:
            raise ValueError(
                f'Materialized shard {opened_shard.shard_id} source_id={source_id} declares {row_count} rows, '
                f'but only {len(logical_values)} rows are available from offset {row_start}'
            )
        try:
            index_map = _SparseLogicalIndexMap.from_values(logical_values, row_start=row_start)
        except ValueError as error:
            raise ValueError(
                f'Invalid logical index column in materialized shard {opened_shard.shard_id}, '
                f'source_id={source_id}, data_path={state.source.data_path!r}'
            ) from error
        opened_shard.logical_index_maps[source_id] = index_map
        return index_map

    def _query_delta(self, opened_shard: _OpenedMaterializedShard, item: dict[str, Any], source_id: int, abs_idx: int, ep_idx: int) -> None:
        state = opened_shard.sources[source_id]
        if not state.delta_offsets:
            return
        index_map = self._get_logical_index_map(opened_shard, source_id)
        fallback_row_idx = index_map.row_for(abs_idx)
        if fallback_row_idx is None:
            raise IndexError(
                f'Current logical index {abs_idx} is missing from materialized shard {opened_shard.shard_id}, '
                f'source_id={source_id}, data_path={state.source.data_path!r}'
            )
        episode_start, episode_end = state.episode_ranges[ep_idx]
        for key, offsets in state.delta_offsets.items():
            values = []
            pads = []
            for offset in offsets:
                query_idx = abs_idx + int(offset)
                is_pad = query_idx < episode_start or query_idx >= episode_end
                query_idx = min(max(query_idx, episode_start), episode_end - 1)
                row_idx = index_map.row_for(query_idx)
                if row_idx is None:
                    row_idx = fallback_row_idx
                    is_pad = True
                values.append(opened_shard.dataset[row_idx][key])
                pads.append(is_pad)
            item[key] = torch.stack([value if torch.is_tensor(value) else torch.as_tensor(value) for value in values])
            item[f'{key}_is_pad'] = torch.as_tensor(pads, dtype=torch.bool)

    def _query_videos(self, opened_shard: _OpenedMaterializedShard, item: dict[str, Any], source_id: int, ep_idx: int) -> None:
        helpers = _require_lerobot_helpers()
        state = opened_shard.sources[source_id]
        video_keys = state.meta.video_keys if state.video_key_set is None else [key for key in state.meta.video_keys if key in state.video_key_set]
        if not video_keys:
            return
        timestamp = float(self._scalar(item['timestamp']))
        ep = state.meta.episodes[ep_idx]
        for video_key in video_keys:
            from_timestamp = ep[f'videos/{video_key}/from_timestamp']
            shifted = [from_timestamp + timestamp]
            video_path = state.root / state.meta.get_video_file_path(ep_idx, video_key)
            try:
                frames = helpers['decode_video_frames_torchcodec'](
                    video_path,
                    shifted,
                    2e-2,
                    decoder_cache=self._torchcodec_decoder_cache,
                )
                item[video_key] = frames.squeeze(0)
            except Exception:
                shape = state.sample_meta.shapes.get(video_key, (3, 224, 224))
                c, h, w = (shape if len(shape) == 3 and shape[0] in (1, 3, 4) else (3, shape[0], shape[1]))
                item[video_key] = torch.zeros((c, h, w), dtype=torch.float32)

    @staticmethod
    def _apply_repack(item: dict[str, Any], pairs: tuple[tuple[str, str], ...]) -> dict[str, Any]:
        for left, right in pairs:
            if left in item and right not in item:
                item[right] = item.pop(left)
            elif right in item and left not in item:
                item[left] = item.pop(right)
            elif left in item and right in item:
                item[right] = item.pop(left)
        return item

    def __getitem__(self, index: int | list[int] | tuple[int, ...]) -> Any:
        if isinstance(index, (list, tuple)):
            return [self[int(idx)] for idx in index]
        helpers = _require_lerobot_helpers()
        profile_enabled = helpers['pipeline_profile_enabled']() if self.profile_enabled is None else bool(self.profile_enabled)
        profile = {} if profile_enabled else None
        total_tic = time.perf_counter() if profile_enabled else None
        index = int(index)
        if index < 0:
            index += len(self)
        if profile_enabled:
            tic_stage = time.perf_counter()
        shard = self._find_shard(index)
        if profile_enabled:
            helpers['profile_add'](profile, 'data_materialized_find_shard', time.perf_counter() - tic_stage)
        opened_shard, open_time, cache_hit, evicted = self._open_shard(shard)
        if profile_enabled:
            helpers['profile_add'](profile, 'data_materialized_open_shard', open_time)
            helpers['profile_add'](profile, 'data_materialized_open_shard_cache_hit', 1.0 if cache_hit else 0.0)
            helpers['profile_add'](profile, 'data_materialized_open_shard_cache_miss', 0.0 if cache_hit else 1.0)
            helpers['profile_add'](profile, 'data_materialized_open_shard_cache_evict', 1.0 if evicted else 0.0)
            helpers['profile_add'](profile, 'data_materialized_shard_cache_size', float(len(self._opened_shards)))
            helpers['profile_add'](profile, 'data_materialized_shard_cache_capacity', float(self.shard_cache_size))
            helpers['profile_add'](profile, 'data_lerobot_open', open_time)
            tic_stage = time.perf_counter()
        row = opened_shard.dataset[index - int(shard['start'])]
        if profile_enabled:
            helpers['profile_add'](profile, 'data_materialized_parquet_get', time.perf_counter() - tic_stage)
            helpers['profile_add'](profile, 'data_lerobot_hf_get', time.perf_counter() - tic_stage)
            tic_stage = time.perf_counter()
        item = dict(row)
        if profile_enabled:
            helpers['profile_add'](profile, 'data_materialized_row_to_dict', time.perf_counter() - tic_stage)
            tic_stage = time.perf_counter()
        source_id = int(self._scalar(item.pop('__source_id')))
        state = opened_shard.sources[source_id]
        abs_idx = int(self._scalar(item['index']))
        ep_idx = int(self._scalar(item['episode_index']))
        if profile_enabled:
            helpers['profile_add'](profile, 'data_materialized_source_lookup', time.perf_counter() - tic_stage)
            tic_stage = time.perf_counter()
        self._query_delta(opened_shard, item, source_id, abs_idx, ep_idx)
        if profile_enabled:
            helpers['profile_add'](profile, 'data_materialized_delta_query', time.perf_counter() - tic_stage)
            helpers['profile_add'](profile, 'data_lerobot_delta_query', time.perf_counter() - tic_stage)
            tic_stage = time.perf_counter()
        self._query_videos(opened_shard, item, source_id, ep_idx)
        if profile_enabled:
            helpers['profile_add'](profile, 'data_materialized_video_decode', time.perf_counter() - tic_stage)
            helpers['profile_add'](profile, 'data_lerobot_video_decode', time.perf_counter() - tic_stage)
            tic_stage = time.perf_counter()
        task_idx = int(self._scalar(item['task_index']))
        item['task'] = state.meta.tasks.iloc[task_idx].name
        item['meta'] = state.sample_meta
        if profile_enabled:
            helpers['profile_add'](profile, 'data_materialized_task_lookup', time.perf_counter() - tic_stage)
            helpers['profile_add'](profile, 'data_lerobot_task_lookup', time.perf_counter() - tic_stage)
            tic_stage = time.perf_counter()
        item = self._apply_repack(item, state.repack_pairs)
        if profile_enabled:
            helpers['profile_add'](profile, 'data_materialized_repack', time.perf_counter() - tic_stage)
            helpers['profile_add'](profile, 'data_materialized_total', time.perf_counter() - total_tic)
            helpers['profile_add'](profile, 'data_lerobot_repack', time.perf_counter() - tic_stage)
            helpers['profile_add'](profile, 'data_lerobot_fast_total', time.perf_counter() - total_tic)
            helpers['profile_add'](profile, 'data_lerobot_total', time.perf_counter() - total_tic)
        if self.transform is not None:
            item = self.transform(item)
        if profile_enabled:
            helpers['attach_profile'](item, profile)
        return item

    def __getitems__(self, indices: list[int]) -> list[Any]:
        return [self[index] for index in indices]


def wrap_materialized_shard_reader(
    dataset: Any,
    shard_cfg: Any,
    log_fn: Any | None = None,
    *,
    profile_enabled: bool | None = None,
) -> Any:
    if not shard_cfg:
        return dataset
    if shard_cfg is True:
        shard_cfg = {}
    elif not isinstance(shard_cfg, dict):
        raise TypeError(
            'dataloaders.train.materialized_shard_reader should be a bool or dict, '
            f'got {type(shard_cfg).__name__}'
        )
    index_dir = shard_cfg.get('index_dir')
    if not index_dir:
        raise ValueError('materialized_shard_reader.index_dir is required')
    materialize = _as_bool(shard_cfg.get('materialize', False), False)
    children_per_shard = int(shard_cfg.get('children_per_shard', 8))
    if materialize:
        materialize_lerobot_shards(
            dataset,
            output_dir=index_dir,
            children_per_shard=children_per_shard,
            overwrite=_as_bool(shard_cfg.get('overwrite', False), False),
            max_shards=shard_cfg.get('max_shards'),
            log_fn=log_fn,
        )
    shard_cache_size = int(shard_cfg.get('shard_cache_size', 1))
    wrapped = MaterializedShardLeRobotDataset(
        dataset,
        index_dir,
        profile_enabled=profile_enabled,
        shard_cache_size=shard_cache_size,
    )
    if log_fn is not None:
        log_fn(
            f'Enable materialized LeRobot shard reader: index_dir={index_dir}, '
            f'len={len(wrapped)}, shard_cache_size={shard_cache_size}'
        )
    return wrapped
