"""Generate suite-specific normalization statistics for RoboColiseum tasks.

The command accepts a suite name, resolves its task list and output file, and
always computes statistics with the unified 17-D input schema.
"""

from typing import Literal

import tyro

from robocoliseum_ext.compute_norm_stats import compute_norm_stats
from robocoliseum_ext.posttrain_config import (
    NORM_STATS_ROOT,
    _required_path,
    INSTRUCTION_ROBUST_TASKS,
    MANIP_TASKS,
    SPATIAL_TASKS,
)


Suite = Literal['instruction_and_robust', 'spatial', 'manipulation']
# Each suite keeps an independent task list and norm-stats file.
# All files use the unified 17-D tensor schema.
SUITE_SETTINGS = {
    'instruction_and_robust': (
        INSTRUCTION_ROBUST_TASKS,
        'robocoliseum_instruction_robust_17d.json',
    ),
    'spatial': (
        SPATIAL_TASKS,
        'robocoliseum_spatial_17d.json',
    ),
    'manipulation': (
        MANIP_TASKS,
        'robocoliseum_manip_17d.json',
    ),
}


def compute_suite_norm_stats(
    suite: Suite,
    sample_rate: float = 1.0,
    num_workers: int = 16,
    seed: int = 0,
) -> None:
    """Generate reproducible normalization statistics for one training suite.

    Args:
        suite: Suite name that selects the task list and output file.
        sample_rate: Fraction of frames to include, in the interval (0, 1].
        num_workers: Number of parallel data-loading workers.
        seed: Dataset traversal seed; identical inputs and seeds visit the same frames.
    """
    tasks, output_name = SUITE_SETTINGS[suite]
    # Resolve user-provided roots only when the utility is executed.
    data_root = _required_path("ROBOCOLISEUM_DATA_ROOT")
    data_paths = [str(data_root / suite / task) for task in tasks]
    output_path = str(NORM_STATS_ROOT / output_name)
    compute_norm_stats(
        data_paths=data_paths,
        output_path=output_path,
        sample_rate=sample_rate,
        num_workers=num_workers,
        seed=seed,
    )


if __name__ == '__main__':
    tyro.cli(compute_suite_norm_stats)
