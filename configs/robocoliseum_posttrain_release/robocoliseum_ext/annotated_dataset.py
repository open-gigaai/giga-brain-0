"""LeRobot dataset variant using episode-level RoboColiseum prompts."""

import json
from pathlib import Path

from giga_datasets.datasets.dataset import register_dataset
from giga_datasets.datasets.lerobot_dataset import LeRobotDataset


def _combine_action_steps(
    raw_action_steps: list[dict],
    *,
    annotations_path: Path,
    episode_index: int,
) -> str:
    """Normalize action-step annotations into one episode-level prompt."""
    action_steps = []
    for raw_step in raw_action_steps:
        action_text = raw_step.get('action_text')
        if not isinstance(action_text, str) or not action_text.strip():
            raise ValueError(
                f'{annotations_path}: episode {episode_index} has empty action_text'
            )
        start_frame = int(raw_step['start_frame'])
        end_frame = int(raw_step['end_frame'])
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError(
                f'{annotations_path}: episode {episode_index} has invalid frame '
                f'range [{start_frame}, {end_frame}]'
            )
        action_steps.append((start_frame, end_frame, action_text.strip()))

    action_steps.sort(key=lambda step: step[0])
    for previous, current in zip(action_steps, action_steps[1:]):
        if current[0] <= previous[1]:
            raise ValueError(
                f'{annotations_path}: episode {episode_index} has overlapping '
                f'action step ranges {previous[:2]} and {current[:2]}'
            )
    return '; then '.join(action_text for _, _, action_text in action_steps)


def _annotation_to_prompt(
    annotation: dict,
    *,
    annotations_path: Path,
    episode_index: int,
) -> str:
    """Extract the prompt string from episode-level annotation records."""
    high_level_instruction = annotation.get('high_level_instruction')
    if isinstance(high_level_instruction, str) and high_level_instruction.strip():
        return high_level_instruction.strip()

    task = annotation.get('task')
    if isinstance(task, str) and task.strip():
        return task.strip()

    tasks = annotation.get('tasks')
    if isinstance(tasks, list) and tasks:
        normalized_tasks = []
        for raw_task in tasks:
            if not isinstance(raw_task, str) or not raw_task.strip():
                raise ValueError(
                    f'{annotations_path}: episode {episode_index} has empty task text'
                )
            normalized_tasks.append(raw_task.strip())
        return '; then '.join(normalized_tasks)

    raw_action_steps = annotation.get('action_steps')
    if not isinstance(raw_action_steps, list) or not raw_action_steps:
        raise ValueError(f'{annotations_path}: episode {episode_index} has no prompt text')
    return _combine_action_steps(
        raw_action_steps,
        annotations_path=annotations_path,
        episode_index=episode_index,
    )


def _load_prompt_records(data_path: str) -> tuple[Path, dict[str, dict]]:
    """Load prompt records from info.json, annotations.json, or episode metadata."""
    meta_dir = Path(data_path) / 'meta'
    info_path = meta_dir / 'info.json'
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        high_level_instruction = info.get('high_level_instruction')
        if isinstance(high_level_instruction, dict) and high_level_instruction:
            return info_path, high_level_instruction

    annotations_path = meta_dir / 'annotations.json'
    if annotations_path.is_file():
        return annotations_path, json.loads(annotations_path.read_text())

    episodes_path = meta_dir / 'episodes.jsonl'
    if not episodes_path.is_file():
        raise FileNotFoundError(f'annotations not found: {annotations_path}')

    annotations = {}
    with episodes_path.open() as source:
        for line in source:
            record = json.loads(line)
            episode_index = int(record['episode_index'])
            tasks = record.get('tasks')
            if not isinstance(tasks, list) or not tasks:
                raise ValueError(
                    f'{episodes_path}: episode {episode_index} has no tasks'
                )
            annotations[str(episode_index)] = {
                'episode_index': episode_index,
                'tasks': tasks,
            }
    return episodes_path, annotations


def load_episode_action_prompts(data_path: str) -> dict[int, str]:
    """Load one ordered prompt for every annotation JSON object key."""
    annotations_path, annotations = _load_prompt_records(data_path)

    prompts = {}
    for raw_episode_index, annotation in annotations.items():
        if not isinstance(annotation, dict):
            raise ValueError(
                f'{annotations_path}: episode {raw_episode_index} must be a JSON object'
            )
        episode_index = int(annotation.get('episode_index', raw_episode_index))
        if episode_index in prompts:
            raise ValueError(
                f'{annotations_path}: duplicate episode_index={episode_index}'
            )
        prompts[episode_index] = _annotation_to_prompt(
            annotation,
            annotations_path=annotations_path,
            episode_index=episode_index,
        )
    return prompts


def load_first_instruction_segments(
    data_path: str,
) -> dict[int, tuple[int, int, str]]:
    """Load the first annotated instruction and its inclusive frame range."""
    info_path = Path(data_path) / 'meta' / 'info.json'
    if not info_path.is_file():
        raise FileNotFoundError(f'dataset metadata not found: {info_path}')

    info = json.loads(info_path.read_text())
    raw_segments = info.get('instruction_segments')
    if not isinstance(raw_segments, dict):
        raise ValueError(f'{info_path}: missing instruction_segments')

    total_episodes = int(info['total_episodes'])
    first_segments = {}
    for episode_index in range(total_episodes):
        episode_segments = raw_segments.get(str(episode_index))
        if not isinstance(episode_segments, list) or not episode_segments:
            raise ValueError(
                f'{info_path}: episode {episode_index} has no instruction segments'
            )
        first_segment = episode_segments[0]
        instruction = first_segment.get('instruction')
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                f'{info_path}: episode {episode_index} has an empty first instruction'
            )
        start_frame = int(first_segment['start_frame_index'])
        end_frame = int(first_segment['end_frame_index'])
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError(
                f'{info_path}: episode {episode_index} has invalid first-segment '
                f'range [{start_frame}, {end_frame}]'
            )
        first_segments[episode_index] = (
            start_frame,
            end_frame,
            instruction.strip(),
        )
    return first_segments


def load_concatenated_instruction_segments(
    data_path: str,
) -> dict[int, tuple[int, int, str]]:
    """Join all episode instructions and cover their complete frame range."""
    info_path = Path(data_path) / 'meta' / 'info.json'
    if not info_path.is_file():
        raise FileNotFoundError(f'dataset metadata not found: {info_path}')

    info = json.loads(info_path.read_text())
    raw_episode_segments = info.get('instruction_segments')
    if not isinstance(raw_episode_segments, dict):
        raise ValueError(f'{info_path}: missing instruction_segments')

    training_segments = {}
    for episode_index in range(int(info['total_episodes'])):
        raw_segments = raw_episode_segments.get(str(episode_index))
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError(
                f'{info_path}: episode {episode_index} has no instruction segments'
            )
        normalized_segments = []
        for segment_index, raw_segment in enumerate(raw_segments):
            instruction = raw_segment.get('instruction')
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(
                    f'{info_path}: episode {episode_index} segment '
                    f'{segment_index} has an empty instruction'
                )
            start_frame = int(raw_segment['start_frame_index'])
            end_frame = int(raw_segment['end_frame_index'])
            if start_frame < 0 or end_frame < start_frame:
                raise ValueError(
                    f'{info_path}: episode {episode_index} segment {segment_index} '
                    f'has invalid range [{start_frame}, {end_frame}]'
                )
            if normalized_segments and start_frame < normalized_segments[-1][1]:
                raise ValueError(
                    f'{info_path}: episode {episode_index} segment {segment_index} '
                    'starts before the preceding segment ends'
                )
            normalized_segments.append(
                (start_frame, end_frame, instruction.strip())
            )
        training_segments[episode_index] = (
            normalized_segments[0][0],
            normalized_segments[-1][1],
            '; then '.join(segment[2] for segment in normalized_segments),
        )
    return training_segments


@register_dataset
class EpisodeActionPromptLeRobotDataset(LeRobotDataset):
    """Replace each sample task with its episode's complete prompt."""

    def __init__(self, data_path: str, **kwargs) -> None:
        super().__init__(data_path=data_path, **kwargs)
        self._episode_action_prompts = load_episode_action_prompts(data_path)

    def _get_data(self, index: int) -> dict:
        data_dict = super()._get_data(index)
        episode_index = int(data_dict['episode_index'].item())
        try:
            data_dict['task'] = self._episode_action_prompts[episode_index]
        except KeyError as error:
            raise KeyError(
                f'{self.data_path}: no action prompt for episode_index={episode_index}'
            ) from error
        return data_dict


@register_dataset
class AnnotatedTaskLeRobotDataset(EpisodeActionPromptLeRobotDataset):
    """Backward-compatible alias for episode-level action prompts."""


@register_dataset
class FirstInstructionSegmentLeRobotDataset(LeRobotDataset):
    """Train only action chunks fully contained in each episode's first step."""

    segment_loader = staticmethod(load_first_instruction_segments)

    def __init__(self, data_path: str, **kwargs) -> None:
        super().__init__(data_path=data_path, **kwargs)
        self._training_segments = self.segment_loader(data_path)
        self._sample_indices = None

    def open(self) -> None:
        if self.dataset is not None:
            return
        super().open()

        action_horizon = 1
        if self.delta_info is not None:
            action_horizon = int(self.delta_info.get('action', action_horizon))
        if action_horizon < 1:
            raise ValueError(f'action horizon must be positive, got {action_horizon}')

        sample_indices = []
        episode_offset = 0
        for episode_index in range(len(self.dataset.meta.episodes)):
            episode = self.dataset.meta.episodes[episode_index]
            episode_length = int(episode['length'])
            try:
                start_frame, end_frame, _ = self._training_segments[episode_index]
            except KeyError as error:
                raise KeyError(
                    f'{self.data_path}: no training instruction for '
                    f'episode_index={episode_index}'
                ) from error
            if end_frame >= episode_length:
                raise ValueError(
                    f'{self.data_path}: episode {episode_index} training-range end '
                    f'{end_frame} exceeds final frame {episode_length - 1}'
                )

            # The final retained anchor must keep the complete future action
            # chunk inside the selected training range.
            final_anchor = end_frame - action_horizon + 1
            if final_anchor < start_frame:
                raise ValueError(
                    f'{self.data_path}: episode {episode_index} training range is '
                    f'shorter than action horizon {action_horizon}'
                )
            sample_indices.extend(
                range(episode_offset + start_frame, episode_offset + final_anchor + 1)
            )
            episode_offset += episode_length

        if episode_offset != len(self.dataset):
            raise ValueError(
                f'{self.data_path}: episode lengths sum to {episode_offset}, '
                f'but dataset contains {len(self.dataset)} frames'
            )
        self._sample_indices = sample_indices
        self.data_size = len(sample_indices)

    def _get_data(self, index: int) -> dict:
        source_index = self._sample_indices[index]
        data_dict = super()._get_data(source_index)
        episode_index = int(data_dict['episode_index'].item())
        data_dict['task'] = self._training_segments[episode_index][2]
        return data_dict


@register_dataset
class ConcatenatedInstructionSegmentsLeRobotDataset(
    FirstInstructionSegmentLeRobotDataset
):
    """Train all annotated steps under their ordered concatenated prompt."""

    segment_loader = staticmethod(load_concatenated_instruction_segments)
