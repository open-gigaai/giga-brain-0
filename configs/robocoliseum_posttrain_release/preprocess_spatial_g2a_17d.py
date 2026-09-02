"""Create offline 17-D g2a_sim datasets for the spatial task suite."""

import os
from pathlib import Path

from preprocess_instruction_robust_g2a_17d import _process_task, _required_path


SOURCE_SUITE_ROOT = _required_path("ROBOCOLISEUM_DATA_ROOT") / "spatial"
OUTPUT_TASK_SUITE_ROOT = _required_path("ROBOCOLISEUM_PROCESSED_DATA_ROOT")
SPATIAL_TASKS = (
    'pick_object_relative_position_absolute',
    'pick_object_relative_position_relative',
    'place_beverage_to_anothers_position',
    'place_object_relative_position',
    'sort_cubes_by_size',
    'sort_number_from_small_to_big',
    'stack_bowls',
    'stack_three_building_blocks',
)


def main() -> None:
    """Build the spatial suite in a staging directory and publish it atomically."""
    output_suite_root = OUTPUT_TASK_SUITE_ROOT / 'spatial'
    staging_suite_root = OUTPUT_TASK_SUITE_ROOT / '.spatial.incomplete'
    if output_suite_root.exists():
        raise FileExistsError(f'output already exists: {output_suite_root}')
    if staging_suite_root.exists():
        raise FileExistsError(f'staging output already exists: {staging_suite_root}')
    staging_suite_root.mkdir(parents=True)

    total_parquet = 0
    for task in SPATIAL_TASKS:
        source_task = SOURCE_SUITE_ROOT / task
        if not source_task.is_dir():
            raise FileNotFoundError(f'source dataset not found: {source_task}')
        count = _process_task(source_task, staging_suite_root / task)
        total_parquet += count
        print(f'processed {task}: {count} Parquet files', flush=True)

    os.replace(staging_suite_root, output_suite_root)
    print(
        f'created {output_suite_root} with {total_parquet} processed Parquet files',
        flush=True,
    )


if __name__ == '__main__':
    main()
