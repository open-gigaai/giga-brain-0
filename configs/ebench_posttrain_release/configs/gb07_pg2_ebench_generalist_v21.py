"""GigaBrain-0.7 EBench training configuration with a local LeRobot v2.1 reader.

The reader is intentionally kept in this configuration module so the regular
GigaBrain configuration and the installed v3 dataset reader remain unchanged.
"""

from bisect import bisect_right
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
import torch

from giga_datasets.datasets.base_dataset import BaseDataset
from giga_datasets.datasets.dataset import DATASETS

from configs.gb07_pg2_ebench_generalist_100k import config as _base_config


class LeRobotV21Dataset(BaseDataset):
    """Read an on-disk LeRobot v2.1 dataset without using the v3 metadata API."""

    def __init__(
        self,
        data_path: str,
        data_size: int | None = None,
        delta_info: dict[str, int] | None = None,
        meta_name: str | None = None,
        episodes: list[int] | None = None,
        **_: Any,
    ) -> None:
        super().__init__(data_path=data_path)
        self.root = Path(data_path)
        self.data_size = data_size
        self.delta_info = delta_info or {}
        self.meta_name = meta_name
        self.requested_episodes = None if episodes is None else {int(x) for x in episodes}
        self._opened = False
        self._episodes: list[dict[str, Any]] = []
        self._starts: list[int] = []
        self._tables: dict[int, Any] = {}
        self._videos: dict[str, av.container.InputContainer] = {}
        self._tasks: list[str] = []
        self._info: dict[str, Any] = {}

    @classmethod
    def load(cls, data_or_config: str | dict[str, Any]) -> "LeRobotV21Dataset":
        if isinstance(data_or_config, str):
            with open(data_or_config, encoding="utf-8") as f:
                cfg = json.load(f)
            cfg.setdefault("data_path", str(Path(data_or_config).parent))
        else:
            cfg = dict(data_or_config)
        cfg.pop("_class_name", None)
        cfg.pop("config_path", None)
        return cls(**cfg)

    def open(self) -> None:
        if self._opened:
            return
        with (self.root / "meta" / "info.json").open(encoding="utf-8") as f:
            self._info = json.load(f)
        with (self.root / "meta" / "episodes.jsonl").open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        if self.requested_episodes is not None:
            records = [r for r in records if int(r["episode_index"]) in self.requested_episodes]
        records.sort(key=lambda r: int(r["episode_index"]))
        self._episodes = records
        total = 0
        self._starts = []
        for record in records:
            self._starts.append(total)
            total += int(record["length"])
        self.data_size = total if self.data_size is None else int(self.data_size)
        with (self.root / "meta" / "tasks.jsonl").open(encoding="utf-8") as f:
            self._tasks = [json.loads(line).get("task", "") for line in f if line.strip()]
        self._opened = True

    def close(self) -> None:
        for container in self._videos.values():
            try:
                container.close()
            except Exception:
                pass
        self._videos.clear()
        self._tables.clear()
        self._opened = False
        super().close()

    def __len__(self) -> int:
        self.open()
        return int(self.data_size or 0)

    def _episode_and_row(self, index: int) -> tuple[dict[str, Any], int]:
        self.open()
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_pos = bisect_right(self._starts, index) - 1
        return self._episodes[episode_pos], index - self._starts[episode_pos]

    def _episode_path(self, episode_index: int) -> Path:
        chunk_size = int(self._info.get("chunks_size", 1000))
        return self.root / "data" / f"chunk-{episode_index // chunk_size:03d}" / f"episode_{episode_index:06d}.parquet"

    def _table(self, episode_index: int):
        table = self._tables.get(episode_index)
        if table is None:
            table = pq.read_table(self._episode_path(episode_index))
            self._tables[episode_index] = table
        return table

    @staticmethod
    def _value(table: Any, key: str, row: int) -> torch.Tensor:
        column = table[key][row].as_py()
        return torch.as_tensor(column, dtype=torch.float32)

    def _video_key(self, requested_key: str) -> str | None:
        features = self._info.get("features", {})
        if requested_key in features:
            return requested_key
        # EBench uses both "top" and "overlook" for the overhead camera.
        if requested_key.endswith("overlook_camera_view"):
            candidate = requested_key.replace("overlook_camera_view", "top_camera_view")
            if candidate in features:
                return candidate
        return None

    def _video(self, key: str, episode_index: int, frame_index: int) -> torch.Tensor:
        actual_key = self._video_key(key)
        shape = self._info.get("features", {}).get(actual_key or key, {}).get("shape", [3, 224, 224])
        if actual_key is None:
            return torch.zeros(tuple(int(x) for x in shape), dtype=torch.uint8)
        chunk_size = int(self._info.get("chunks_size", 1000))
        path = self.root / "videos" / f"chunk-{episode_index // chunk_size:03d}" / actual_key / f"episode_{episode_index:06d}.mp4"
        try:
            # Open per request; this avoids leaking decoder state across workers.
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                fps = float(self._info.get("fps", 15))
                timestamp = max(0.0, float(frame_index) / fps)
                container.seek(int(timestamp / float(stream.time_base)), stream=stream, backward=True)
                best = None
                for frame in container.decode(stream):
                    best = frame
                    if frame.time >= timestamp:
                        break
                if best is not None:
                    array = best.to_ndarray(format="rgb24")
                    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        except Exception:
            pass
        return torch.zeros(tuple(int(x) for x in shape), dtype=torch.uint8)

    def _get_data(self, index: int) -> dict[str, Any]:
        record, row = self._episode_and_row(index)
        episode_index = int(record["episode_index"])
        table = self._table(episode_index)
        item: dict[str, Any] = {
            "episode_index": torch.tensor(episode_index, dtype=torch.int64),
            "frame_index": torch.tensor(row, dtype=torch.int64),
            "timestamp": torch.tensor(float(table["timestamp"][row].as_py()), dtype=torch.float32),
            "task_index": torch.tensor(int(table["task_index"][row].as_py()), dtype=torch.int64),
            "task": self._tasks[int(table["task_index"][row].as_py())],
        }
        for key in ("state.joints", "state.gripper"):
            item[key] = self._value(table, key, row)

        for key, horizon in self.delta_info.items():
            values = []
            pads = []
            length = int(record["length"])
            for offset in range(int(horizon)):
                query_row = row + offset
                padded = query_row >= length
                query_row = min(query_row, length - 1)
                values.append(self._value(table, key, query_row))
                pads.append(padded)
            item[key] = torch.stack(values)
            item[f"{key}_is_pad"] = torch.tensor(pads, dtype=torch.bool)

        # Return all cameras declared by the GB config. Missing aliases become zeros.
        for key in (
            "video.overlook_camera_view",
            "video.left_camera_view",
            "video.right_camera_view",
        ):
            item[key] = self._video(key, episode_index, row)

        if self.meta_name is not None:
            item[self.meta_name] = SimpleNamespace(
                root=self.root,
                repo_id=self.root.name,
                info=self._info,
                tasks=self._tasks,
                fps=float(self._info.get("fps", 15)),
            )
        return item


# The registry is populated when this config is imported by the config loader.
DATASETS[LeRobotV21Dataset.__name__] = LeRobotV21Dataset


config = deepcopy(_base_config)
config["project_dir"] = f'{config["project_dir"]}_v21'
dataset_groups = config["dataloaders"]["train"]["data_or_config"]["datasets"]
for group in dataset_groups:
    for dataset_config in group:
        dataset_config["_class_name"] = "LeRobotV21Dataset"

