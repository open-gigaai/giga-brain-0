"""Create offline 17-D g2a_sim datasets for GigaBrain post-training."""

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _required_path(name: str) -> Path:
    """Return a user-provided path and reject missing or empty values."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable to a valid path.")
    return Path(value).expanduser()


SOURCE_SUITE_ROOT = (
    _required_path("ROBOCOLISEUM_DATA_ROOT") / "instruction_and_robust"
)
OUTPUT_DATASET_ROOT = (
    _required_path("ROBOCOLISEUM_PROCESSED_DATA_ROOT") / "instruction_and_robust"
)
TASKS = (
    'pick_billards_color_500',
    'pick_block_color_500',
    'pick_block_number_500',
    'pick_block_shape_500',
    'pick_block_size_500',
    'pick_common_sense_500',
    'pick_follow_logic_(or)_500',
    'pick_object_type_500',
    'pick_specific_object_500',
    'straighten_object_500',
)

# Preserve the exact ordering previously applied by RoboColiseumJointActionRemapTransform.
STATE_GATHER_INDEX = (*range(30, 37), 0, *range(37, 44), 1, 65)
ACTION_GATHER_INDEX = (*range(16, 23), 0, *range(23, 30), 1, 37)
TARGET_DIM = 17
MAX_WORKERS = 4


def _target_list_type(source_type: pa.DataType) -> pa.DataType:
    """Return the corresponding Arrow list type with a 17-value payload."""
    if pa.types.is_fixed_size_list(source_type):
        return pa.list_(source_type.value_type, TARGET_DIM)
    if pa.types.is_large_list(source_type):
        return pa.large_list(source_type.value_type)
    if pa.types.is_list(source_type):
        return pa.list_(source_type.value_type)
    raise TypeError(f'expected an Arrow list column, got {source_type}')


def _repack_column(
    table: pa.Table,
    feature_name: str,
    gather_index: tuple[int, ...],
) -> pa.Table:
    """Gather one nested vector column while preserving its floating-point type."""
    column_index = table.schema.get_field_index(feature_name)
    if column_index < 0:
        raise KeyError(f'missing Parquet column {feature_name!r}')
    source_type = table.schema.field(column_index).type
    rows = table.column(column_index).to_pylist()
    expected_width = max(gather_index) + 1
    for row_index, row in enumerate(rows):
        if row is None or len(row) < expected_width:
            width = None if row is None else len(row)
            raise ValueError(
                f'{feature_name} row {row_index} has width {width}; '
                f'expected at least {expected_width}'
            )
    repacked = [[row[index] for index in gather_index] for row in rows]
    array = pa.array(repacked, type=_target_list_type(source_type))
    return table.set_column(column_index, feature_name, array)


def _repack_parquet(path: Path) -> None:
    """Atomically replace one hard-linked raw Parquet file with its 17-D copy."""
    parquet_file = pq.ParquetFile(path)
    table = parquet_file.read()
    compression = parquet_file.metadata.row_group(0).column(0).compression.lower()
    table = _repack_column(table, 'observation.state', STATE_GATHER_INDEX)
    table = _repack_column(table, 'action', ACTION_GATHER_INDEX)
    temporary_path = path.with_name(f'.{path.name}.tmp')
    pq.write_table(table, temporary_path, compression=compression)
    temporary_path.chmod(path.stat().st_mode)
    os.replace(temporary_path, path)


def _repack_stats(path: Path) -> None:
    """Apply the same gather operation to per-episode feature statistics."""
    temporary_path = path.with_name(f'.{path.name}.tmp')
    with path.open() as source, temporary_path.open('w') as target:
        for line in source:
            record = json.loads(line)
            stats = record['stats']
            for feature_name, gather_index in (
                ('observation.state', STATE_GATHER_INDEX),
                ('action', ACTION_GATHER_INDEX),
            ):
                feature_stats = stats[feature_name]
                for statistic in ('min', 'max', 'mean', 'std'):
                    values = feature_stats[statistic]
                    feature_stats[statistic] = [values[index] for index in gather_index]
            target.write(json.dumps(record, separators=(',', ':')) + '\n')
    temporary_path.chmod(path.stat().st_mode)
    os.replace(temporary_path, path)


def _feature_description(prefix: str) -> dict[str, dict]:
    """Describe the 7+1+7+1+1 joint layout stored in the processed dataset."""
    target = 'target ' if prefix == 'action' else ''
    return {
        f'{prefix}/joint/position': {
            'description': f'Dual-arm {target}joint angles (left arm then right arm).',
            'dimensions': 14,
            'indices': [*range(0, 7), *range(8, 15)],
        },
        f'{prefix}/left_effector/position': {
            'description': f'Left end-effector {target}open/close position.',
            'dimensions': 1,
            'indices': [7],
        },
        f'{prefix}/right_effector/position': {
            'description': f'Right end-effector {target}open/close position.',
            'dimensions': 1,
            'indices': [15],
        },
        f'{prefix}/waist/position': {
            'description': f'Fifth waist joint {target}position.',
            'dimensions': 1,
            'indices': [16],
        },
    }


def _rewrite_info(path: Path) -> None:
    """Update feature widths and semantic indices without changing robot identity."""
    info = json.loads(path.read_text())
    if info.get('robot_type') != 'g2a_sim':
        raise ValueError(f'{path}: expected robot_type="g2a_sim"')
    expected_source_shapes = {'observation.state': [109], 'action': [38]}
    for feature_name, expected_shape in expected_source_shapes.items():
        feature = info['features'][feature_name]
        if feature.get('shape') != expected_shape:
            raise ValueError(
                f'{path}: expected {feature_name} shape {expected_shape}, '
                f'got {feature.get("shape")}'
            )
        feature['shape'] = [TARGET_DIM]
        prefix = 'state' if feature_name == 'observation.state' else 'action'
        feature['field_descriptions'] = _feature_description(prefix)
    temporary_path = path.with_name(f'.{path.name}.tmp')
    temporary_path.write_text(json.dumps(info, indent=4) + '\n')
    temporary_path.chmod(path.stat().st_mode)
    os.replace(temporary_path, path)


def _process_task(source_task: Path, target_task: Path) -> int:
    """Hard-link one dataset, then replace every layout-dependent file."""
    shutil.copytree(source_task, target_task, copy_function=os.link)
    parquet_paths = sorted(target_task.glob('data/**/*.parquet'))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(_repack_parquet, parquet_paths))
    _repack_stats(target_task / 'meta' / 'episodes_stats.jsonl')
    _rewrite_info(target_task / 'meta' / 'info.json')
    return len(parquet_paths)


def main() -> None:
    """Build all datasets in a staging tree and publish the completed result."""
    if OUTPUT_DATASET_ROOT.exists():
        raise FileExistsError(f'output already exists: {OUTPUT_DATASET_ROOT}')
    staging_root = OUTPUT_DATASET_ROOT.with_name(f'.{OUTPUT_DATASET_ROOT.name}.incomplete')
    if staging_root.exists():
        raise FileExistsError(f'staging output already exists: {staging_root}')
    target_suite_root = staging_root / 'task_suite' / 'instruction_and_robust'
    target_suite_root.mkdir(parents=True)

    total_parquet = 0
    for task in TASKS:
        source_task = SOURCE_SUITE_ROOT / task
        if not source_task.is_dir():
            raise FileNotFoundError(f'source dataset not found: {source_task}')
        count = _process_task(source_task, target_suite_root / task)
        total_parquet += count
        print(f'processed {task}: {count} Parquet files', flush=True)

    os.replace(staging_root, OUTPUT_DATASET_ROOT)
    print(
        f'created {OUTPUT_DATASET_ROOT} with {total_parquet} processed Parquet files',
        flush=True,
    )


if __name__ == '__main__':
    main()
