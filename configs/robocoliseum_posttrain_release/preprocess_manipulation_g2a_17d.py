"""Create offline 17-D g2a_sim datasets for the manipulation task suite."""

import json
import os
from pathlib import Path

from preprocess_instruction_robust_g2a_17d import _process_task, _required_path


SOURCE_SUITE_ROOT = _required_path("ROBOCOLISEUM_DATA_ROOT") / "manipulation"
OUTPUT_TASK_SUITE_ROOT = _required_path("ROBOCOLISEUM_PROCESSED_DATA_ROOT")
MANIPULATION_TASKS = (
    'clean_the_desktop',
    'hold_pot',
    'open_door',
    'place_block_into_box',
    'pour_workpiece',
    'scoop_popcorn',
    'sorting_packages',
    'stock_and_straighten_shelf',
    'take_wrong_item_shelf',
)


def _write_annotations(source_task: Path, target_task: Path) -> None:
    """Copy episode-level prompts into meta/annotations.json."""
    info_path = source_task / 'meta' / 'info.json'
    if not info_path.is_file():
        raise FileNotFoundError(f'source info not found: {info_path}')

    info = json.loads(info_path.read_text())
    high_level_instruction = info.get('high_level_instruction')
    if not isinstance(high_level_instruction, dict) or not high_level_instruction:
        raise ValueError(f'{info_path}: missing high_level_instruction records')

    annotations = {}
    for raw_episode_index, record in high_level_instruction.items():
        if not isinstance(record, dict):
            raise ValueError(
                f'{info_path}: episode {raw_episode_index} must be a JSON object'
            )
        episode_index = int(raw_episode_index)
        prompt = record.get('high_level_instruction')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                f'{info_path}: episode {episode_index} has empty high_level_instruction'
            )
        annotations[str(episode_index)] = {
            'episode_index': episode_index,
            'high_level_instruction': prompt.strip(),
        }

    annotations_path = target_task / 'meta' / 'annotations.json'
    temporary_path = annotations_path.with_name('.annotations.json.tmp')
    temporary_path.write_text(json.dumps(annotations, indent=2, ensure_ascii=False) + '\n')
    temporary_path.chmod(info_path.stat().st_mode)
    os.replace(temporary_path, annotations_path)


def main() -> None:
    """Build the manipulation suite in a staging directory and publish it atomically."""
    output_suite_root = OUTPUT_TASK_SUITE_ROOT / 'manipulation'
    staging_suite_root = OUTPUT_TASK_SUITE_ROOT / '.manipulation.incomplete'
    if output_suite_root.exists():
        raise FileExistsError(f'output already exists: {output_suite_root}')
    if staging_suite_root.exists():
        raise FileExistsError(f'staging output already exists: {staging_suite_root}')
    staging_suite_root.mkdir(parents=True)

    total_parquet = 0
    for task in MANIPULATION_TASKS:
        source_task = SOURCE_SUITE_ROOT / task
        if not source_task.is_dir():
            raise FileNotFoundError(f'source dataset not found: {source_task}')
        target_task = staging_suite_root / task
        count = _process_task(source_task, target_task)
        _write_annotations(source_task, target_task)
        total_parquet += count
        print(f'processed {task}: {count} Parquet files', flush=True)

    os.replace(staging_suite_root, output_suite_root)
    print(
        f'created {output_suite_root} with {total_parquet} processed Parquet files',
        flush=True,
    )


if __name__ == '__main__':
    main()
