import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpydantic
import numpy as np
import pandas as pd
import pydantic
import torch
import tyro
from tqdm import tqdm

from giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils import (
    Embodiment3QuaternionTo6D,
    reframe_dual_hand_tcp_chunk_to_anchor_camera_numpy,
)


DEFAULT_MIN_QUANTILE_RANGE = 1e-6
DEFAULT_MIN_QUANTILE_STD_RATIO = 0.05


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None
    q99: numpydantic.NDArray | None = None
    min: numpydantic.NDArray | None = None
    max: numpydantic.NDArray | None = None


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000

    def update(self, batch: np.ndarray) -> None:
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1) for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(
            mean=self._mean,
            std=stddev,
            q01=q01,
            q99=q99,
            min=self._min,
            max=self._max,
        )

    def _adjust_histograms(self):
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])
            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def _is_lerobot_frame_parquet(file_path: str) -> bool:
    """LeRobot frame parquets live under ``.../data/chunk-*/file-*.parquet``."""
    normalized = file_path.replace(os.sep, "/")
    if "_backup_" in normalized:
        return False
    return "/data/" in normalized


def _collect_parquet_files(data_paths: list[str], *, data_subdir_only: bool = True) -> list[str]:
    parquet_files: list[str] = []
    for data_path in data_paths:
        for root, dirs, files in os.walk(data_path):
            # Skip meta tables and conversion backups during traversal (not frame data).
            dirs[:] = [d for d in dirs if d != "meta" and not d.startswith("_backup_")]
            for file_name in files:
                if not file_name.endswith(".parquet"):
                    continue
                file_path = os.path.join(root, file_name)
                if data_subdir_only and not _is_lerobot_frame_parquet(file_path):
                    continue
                parquet_files.append(file_path)
    parquet_files.sort()
    return parquet_files


def _window_actions(actions_2d: np.ndarray, horizon: int) -> np.ndarray:
    """Rebuild LeRobot delta_info action windows.

    For timestep t, window is actions[t:t+horizon], and if short at the tail,
    repeat the last action to fill the horizon.
    """
    num_steps, action_dim = actions_2d.shape
    windows = np.empty((num_steps, horizon, action_dim), dtype=actions_2d.dtype)
    last = actions_2d[-1]
    for t in range(num_steps):
        end = min(t + horizon, num_steps)
        valid = actions_2d[t:end]
        windows[t, : len(valid)] = valid
        if len(valid) < horizon:
            windows[t, len(valid) :] = last
    return windows


def _window_values_by_episode(
    values: np.ndarray,
    episode_indices: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Build clamped future windows without crossing episode boundaries."""
    episode_indices = np.asarray(episode_indices).reshape(-1)
    if len(values) != len(episode_indices):
        raise ValueError(
            "values and episode_indices must have the same length, got "
            f"{len(values)} and {len(episode_indices)}"
        )
    if len(values) == 0:
        raise ValueError("Cannot window an empty array")

    windows = np.empty(
        (len(values), horizon, *values.shape[1:]),
        dtype=values.dtype,
    )
    starts = np.flatnonzero(
        np.concatenate(([True], episode_indices[1:] != episode_indices[:-1]))
    )
    ends = np.concatenate((starts[1:], [len(values)]))
    for start, end in zip(starts, ends, strict=True):
        episode_values = values[start:end]
        episode_length = len(episode_values)
        offsets = (
            np.arange(episode_length, dtype=np.int64)[:, None]
            + np.arange(horizon, dtype=np.int64)[None, :]
        )
        offsets = np.minimum(offsets, episode_length - 1)
        windows[start:end] = episode_values[offsets]
    return windows


def _build_delta_mask(
    embodiment_id: int,
    delta_mask: list[bool],
    *,
    dual_hand_quat_prefix_to_6d: bool = False,
    delta_mask_post_6d: bool = False,
) -> np.ndarray:
    mask_t = torch.tensor(delta_mask, dtype=torch.bool)
    if delta_mask_post_6d:
        return mask_t.numpy()
    layout_id = Embodiment3QuaternionTo6D.DUAL_HAND_QUAT_LAYOUT_EMBODIMENT_ID if dual_hand_quat_prefix_to_6d else embodiment_id
    if Embodiment3QuaternionTo6D.supports_embodiment(layout_id):
        mask_t = Embodiment3QuaternionTo6D.transform_mask(mask_t, layout_id)
    return mask_t.numpy()


def _infer_repeated_front_pair_dim(delta_mask: np.ndarray) -> int | None:
    mask = np.asarray(delta_mask, dtype=bool).reshape(-1)
    mask_dim = int(mask.shape[-1])
    for single_side_dim in range(mask_dim // 2, 0, -1):
        side_mask = mask[:single_side_dim]
        if side_mask.any() and (~side_mask).any() and np.array_equal(side_mask, mask[single_side_dim : 2 * single_side_dim]):
            return single_side_dim * 2
    return None


def _use_minmax_quantiles_after_front_pair(norm_stats: dict[str, NormStats], delta_mask: np.ndarray) -> int | None:
    front_pair_dim = _infer_repeated_front_pair_dim(delta_mask)
    if front_pair_dim is None:
        return None

    for stats in norm_stats.values():
        if stats.min is None or stats.max is None or stats.q01 is None or stats.q99 is None:
            continue
        start = min(front_pair_dim, stats.q01.shape[-1], stats.q99.shape[-1], stats.min.shape[-1], stats.max.shape[-1])
        if start < stats.q01.shape[-1]:
            stats.q01[start:] = stats.min[start:]
            stats.q99[start:] = stats.max[start:]

    return front_pair_dim


def _stabilize_degenerate_quantiles(
    norm_stats: dict[str, NormStats],
    delta_mask: np.ndarray,
    *,
    min_quantile_range: float,
    min_quantile_std_ratio: float,
) -> dict[str, dict[str, list[int]]]:
    fallback_dims_by_key: dict[str, dict[str, list[int]]] = {}
    mask_dim = int(np.asarray(delta_mask).reshape(-1).shape[-1])
    for key, stats in norm_stats.items():
        if stats.min is None or stats.max is None or stats.q01 is None or stats.q99 is None:
            continue

        dim = min(
            mask_dim,
            stats.min.shape[-1],
            stats.max.shape[-1],
            stats.std.shape[-1],
            stats.q01.shape[-1],
            stats.q99.shape[-1],
        )
        min_values = np.asarray(stats.min[:dim])
        max_values = np.asarray(stats.max[:dim])
        std_values = np.asarray(stats.std[:dim])
        quantile_ranges = np.asarray(stats.q99[:dim] - stats.q01[:dim])
        constant_mask = min_values == max_values
        small_absolute_range = quantile_ranges <= float(min_quantile_range)
        small_relative_range = quantile_ranges <= (
            np.maximum(std_values, 0.0) * float(min_quantile_std_ratio)
        )
        fallback_mask = small_absolute_range | small_relative_range
        fallback_dims = np.flatnonzero(fallback_mask).astype(int)
        if fallback_dims.size == 0:
            continue

        nonconstant_dims = np.flatnonzero(fallback_mask & ~constant_mask).astype(int)
        if nonconstant_dims.size:
            stats.q01[nonconstant_dims] = stats.min[nonconstant_dims]
            stats.q99[nonconstant_dims] = stats.max[nonconstant_dims]

        constant_dims = np.flatnonzero(fallback_mask & constant_mask).astype(int)
        stats.q01[constant_dims] = stats.min[constant_dims] - 1.0
        stats.q99[constant_dims] = stats.max[constant_dims] + 1.0

        fallback_dims_by_key[key] = {
            "minmax": nonconstant_dims.tolist(),
            "constant": constant_dims.tolist(),
        }
        print(
            f"[compute_norm] stabilized degenerate quantiles for {key}: "
            f"minmax_dims={nonconstant_dims.tolist()} "
            f"constant_dims={constant_dims.tolist()}",
            flush=True,
        )

    return fallback_dims_by_key


def _pad_last_dim(x: np.ndarray, target_dim: int) -> np.ndarray:
    if x.shape[-1] >= target_dim:
        return x[..., :target_dim]
    pad = np.zeros((*x.shape[:-1], target_dim - x.shape[-1]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=-1)


def _find_series(df: pd.DataFrame, keys: list[str]) -> pd.Series | None:
    for key in keys:
        if key in df.columns:
            return df[key]
    return None


def _window_cameras(cameras_3d: np.ndarray, horizon: int) -> np.ndarray:
    """Rebuild camera pose windows aligned with :func:`_window_actions`."""
    num_steps = cameras_3d.shape[0]
    windows = np.empty((num_steps, horizon, 4, 4), dtype=cameras_3d.dtype)
    last = cameras_3d[-1]
    for t in range(num_steps):
        end = min(t + horizon, num_steps)
        valid = cameras_3d[t:end]
        windows[t, : len(valid)] = valid
        if len(valid) < horizon:
            windows[t, len(valid) :] = last
    return windows


def _quaternion_xyzw_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion ``[4]`` (xyzw) to rotation matrix ``[3, 3]``."""
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError("Zero-norm quaternion")
    x, y, z, w = quat / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _pose7_xyzw_to_mat4(pose7: np.ndarray) -> np.ndarray:
    """Build camera local->world ``[4, 4]`` from ``[x, y, z, qx, qy, qz, qw]``."""
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(-1)
    if pose7.size == 14:
        pose7 = pose7[:7]
    if pose7.size != 7:
        raise ValueError(f"Expected pose length 7 or 14, got {pose7.size}")
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _quaternion_xyzw_to_matrix_np(pose7[3:7])
    out[:3, 3] = pose7[:3]
    return out


def _camera_value_to_mat4(value) -> np.ndarray:
    """Parse parquet camera cells to ``[4, 4]`` local->world.

    Supports EgoDex ``observation.state.camera`` (4 row vectors / flat 16) and ego
    datasets' 7-D xyz+quat poses (e.g. ``observation.state.head_pose``,
    ``observation.chest_pose`` first block).
    """
    if isinstance(value, np.ndarray):
        if value.dtype == object or (value.ndim == 1 and value.shape[0] == 4 and hasattr(value[0], '__len__')):
            return np.stack([np.asarray(row, dtype=np.float64) for row in value], axis=0)
        flat = np.asarray(value, dtype=np.float64)
        if flat.shape == (4, 4):
            return flat
        if flat.size == 16:
            return flat.reshape(4, 4)
        if flat.size in (7, 14):
            return _pose7_xyzw_to_mat4(flat)
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and hasattr(value[0], '__len__'):
            return np.stack([np.asarray(row, dtype=np.float64) for row in value], axis=0)
        flat = np.asarray(value, dtype=np.float64)
        if flat.size in (7, 14):
            return _pose7_xyzw_to_mat4(flat)
    raise ValueError(f'Unsupported camera pose shape/type: {type(value)!r}')


def _parquet_columns_to_read(
    state_keys: list[str],
    action_keys: list[str],
    camera_key: str | None = None,
) -> list[str]:
    cols = list(dict.fromkeys(state_keys + action_keys))
    cols.append("episode_index")
    if camera_key is not None:
        cols.append(camera_key)
    return cols


def _read_parquet_for_norm_stats(
    parquet_file: str,
    state_keys: list[str],
    action_keys: list[str],
    camera_key: str | None = None,
) -> pd.DataFrame:
    """Read only columns needed for norm stats; fall back to full read on schema mismatch."""
    read_columns = _parquet_columns_to_read(state_keys, action_keys, camera_key)
    try:
        return pd.read_parquet(parquet_file, columns=read_columns)
    except Exception:
        return pd.read_parquet(parquet_file)


def _process_single_parquet(
    parquet_file: str,
    sample_rate: float,
    action_chunk: int,
    action_dim: int,
    embodiment_id: int,
    mask: np.ndarray,
    state_keys: list[str],
    action_keys: list[str],
    state_parquet_truncate: int | None,
    action_parquet_truncate: int | None,
    dual_hand_quat_prefix_to_6d: bool,
    chunk_anchor_camera_reframe: bool = False,
    camera_parquet_column: str = "observation.state.camera",
) -> tuple[np.ndarray, np.ndarray, int] | None:
    try:
        camera_key = camera_parquet_column if chunk_anchor_camera_reframe else None
        df = _read_parquet_for_norm_stats(parquet_file, state_keys, action_keys, camera_key)
    except Exception as exc:  # noqa: BLE001
        print(f"Skip unreadable parquet file {parquet_file}: {exc}")
        return None

    state_series = _find_series(df, state_keys)
    action_series = _find_series(df, action_keys)
    if state_series is None or action_series is None:
        return None

    try:
        states = np.stack([np.asarray(v, dtype=np.float64) for v in state_series.to_list()], axis=0)
        actions = np.stack([np.asarray(v, dtype=np.float64) for v in action_series.to_list()], axis=0)
        episode_series = df.get("episode_index")
        episode_indices = (
            None
            if episode_series is None
            else np.asarray(episode_series.to_numpy()).reshape(-1)
        )

        if state_parquet_truncate is not None and states.shape[-1] > state_parquet_truncate:
            states = states[..., :state_parquet_truncate]
        if action_parquet_truncate is not None and actions.shape[-1] > action_parquet_truncate:
            actions = actions[..., :action_parquet_truncate]

        if sample_rate < 1.0:
            take = max(1, int(len(states) * sample_rate))
            states = states[:take]
            actions = actions[:take]
            if episode_indices is not None:
                episode_indices = episode_indices[:take]

        if states.ndim != 2:
            raise ValueError(f"Unsupported state ndim={states.ndim}, expected 2")
        if actions.ndim == 2:
            actions = (
                _window_actions(actions, action_chunk)
                if episode_indices is None
                else _window_values_by_episode(
                    actions,
                    episode_indices,
                    action_chunk,
                )
            )
        elif actions.ndim != 3:
            raise ValueError(f"Unsupported action ndim={actions.ndim}, expected 2 or 3")

        if chunk_anchor_camera_reframe:
            camera_series = df.get(camera_parquet_column)
            if camera_series is None:
                raise KeyError(f"Missing camera column {camera_parquet_column!r} in {parquet_file}")
            cameras = np.stack(
                [_camera_value_to_mat4(v) for v in camera_series.to_list()],
                axis=0,
            )
            if sample_rate < 1.0:
                cameras = cameras[: len(states)]
            camera_windows = (
                _window_cameras(cameras, action_chunk)
                if episode_indices is None
                else _window_values_by_episode(
                    cameras,
                    episode_indices,
                    action_chunk,
                )
            )
            # Batched SE(3) reframe: [N, T, D] + [N, T, 4, 4] in one torch matmul pass.
            actions = reframe_dual_hand_tcp_chunk_to_anchor_camera_numpy(actions, camera_windows)

        if dual_hand_quat_prefix_to_6d:
            states = (
                Embodiment3QuaternionTo6D.transform_tensor_dual_hand_quat_prefix(torch.from_numpy(states).float())
                .numpy()
            )
            actions = (
                Embodiment3QuaternionTo6D.transform_tensor_dual_hand_quat_prefix(torch.from_numpy(actions).float())
                .numpy()
            )
        elif Embodiment3QuaternionTo6D.supports_embodiment(embodiment_id):
            states = Embodiment3QuaternionTo6D._transform_pose_tensor(  # pylint: disable=protected-access
                torch.from_numpy(states), embodiment_id
            ).numpy()
            actions = Embodiment3QuaternionTo6D._transform_pose_tensor(  # pylint: disable=protected-access
                torch.from_numpy(actions), embodiment_id
            ).numpy()

        # Prefix-align with training ``resolve_delta_mask_prefix``: state/action may be shorter
        # than the configured mask (e.g. AgiBot G1 state 20-dim vs 22-dim mask).
        eff = min(int(mask.shape[-1]), int(states.shape[-1]), int(actions.shape[-1]))
        if eff < 1:
            raise ValueError(
                f"Empty overlap between mask ({mask.shape[-1]}), state ({states.shape[-1]}), "
                f"action ({actions.shape[-1]}) in {parquet_file}"
            )
        mask_eff = mask[:eff]
        base = np.where(mask_eff[None, :], states[:, :eff], 0.0)
        actions[:, :, :eff] -= base[:, None, :]

        states = _pad_last_dim(states, action_dim)
        actions = _pad_last_dim(actions, action_dim)

        return states, actions.reshape(-1, actions.shape[-1]), len(states)
    except Exception as exc:  # noqa: BLE001
        print(f"Skip malformed parquet file {parquet_file}: {exc}")
        return None


def compute_norm_stats_fast_v2(
    data_paths: list[str],
    output_path: str | pathlib.Path,
    embodiment_id: int,
    delta_mask: list[bool],
    sample_rate: float = 1.0,
    action_chunk: int = 50,
    action_dim: int = 32,
    num_workers: int = 64,
    state_parquet_column: str | None = None,
    action_parquet_column: str | None = None,
    state_parquet_truncate: int | None = None,
    action_parquet_truncate: int | None = None,
    dual_hand_quat_prefix_to_6d: bool = False,
    delta_mask_post_6d: bool = False,
    data_subdir_only: bool = True,
    chunk_anchor_camera_reframe: bool = False,
    camera_parquet_column: str = "observation.state.camera",
    min_quantile_range: float = DEFAULT_MIN_QUANTILE_RANGE,
    min_quantile_std_ratio: float = DEFAULT_MIN_QUANTILE_STD_RATIO,
) -> None:
    """Fast stats computation via direct parquet processing.

    CLI is kept compatible with compute_norm_stats.py.
    num_workers controls parquet read/preprocess threading.

    ``state_parquet_column`` / ``action_parquet_column``: if set, only that parquet column is used
    (e.g. EgoDex after ``conver_egodex_action``: ``observation.state_action_hands_episode_first`` and
    ``action_hands_episode_first``). Output JSON keys remain ``observation.state`` and ``action`` so
    training configs need not rename norm buckets.

    ``state_parquet_truncate`` / ``action_parquet_truncate``: if set, slice each row to ``[..., :N]``
    when the parquet vector is longer than ``N`` (e.g. keep only the first 16 endpose dims and
    ignore trailing fields such as ``linear_x`` / ``angular_z``).

    ``dual_hand_quat_prefix_to_6d``: convert only the leading 16-D dual-hand TCP quaternion block
    to 6D (20-D) and keep any trailing dimensions, then compute norms on the full vector. When True,
    ``--embodiment-id`` is still used for logging, but delta-mask expansion uses the same dual-hand
    layout as embodiment **5**. Prefer leaving truncate unset so suffix dims are included in stats.

    If unset, falls back to legacy column priority lists (joint state / tcp / generic action columns).

    ``data_subdir_only``: when True (default), only collect LeRobot frame parquets under
    ``**/data/**/*.parquet`` and skip ``meta/`` / ``_backup_*`` trees during traversal.

    ``chunk_anchor_camera_reframe``: when True, re-express each action window in the anchor
    (index 0) camera frame using ``camera_parquet_column`` before quat→6D and delta subtraction.
    """
    if not data_paths:
        raise ValueError("data_paths is empty")
    if not (0.0 < sample_rate <= 1.0):
        raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
    if action_chunk < 1:
        raise ValueError(f"action_chunk must be >= 1, got {action_chunk}")
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if min_quantile_range < 0:
        raise ValueError(
            f"min_quantile_range must be non-negative, got {min_quantile_range}"
        )
    if min_quantile_std_ratio < 0:
        raise ValueError(
            "min_quantile_std_ratio must be non-negative, got "
            f"{min_quantile_std_ratio}"
        )
    if action_parquet_truncate is not None and action_parquet_truncate < 1:
        raise ValueError(f"action_parquet_truncate must be >= 1 when set, got {action_parquet_truncate}")
    if state_parquet_truncate is not None and state_parquet_truncate < 1:
        raise ValueError(f"state_parquet_truncate must be >= 1 when set, got {state_parquet_truncate}")

    keys = ["observation.state", "action"]
    stats = {key: RunningStats() for key in keys}
    mask = _build_delta_mask(
        embodiment_id,
        delta_mask,
        dual_hand_quat_prefix_to_6d=dual_hand_quat_prefix_to_6d,
        delta_mask_post_6d=delta_mask_post_6d,
    )

    if state_parquet_column is not None:
        state_keys = [state_parquet_column]
    else:
        state_keys = [
            "observation.state",
            "observation.state_tcp_endpose_quat",
            "observation.state_action",
        ]
    if action_parquet_column is not None:
        action_keys = [action_parquet_column]
    else:
        action_keys = ["action", "action_tcp_endpose_quat"]

    parquet_files = _collect_parquet_files(data_paths, data_subdir_only=data_subdir_only)
    if not parquet_files:
        raise FileNotFoundError("No parquet files found under data_paths")

    print(f"Found {len(parquet_files)} parquet files (data_subdir_only={data_subdir_only})")
    print(f"Using parquet columns: state_keys={state_keys!r} action_keys={action_keys!r}")
    if dual_hand_quat_prefix_to_6d:
        if delta_mask_post_6d:
            print(
                f"dual_hand_quat_prefix_to_6d=True: prefix 16 quat -> 20 6D + suffix unchanged; "
                f"delta_mask_post_6d=True: use mask as-is (len={len(delta_mask)})"
            )
        else:
            print(
                "dual_hand_quat_prefix_to_6d=True: prefix 16 quat -> 20 6D + suffix; "
                "delta mask expanded via embodiment-5 layout (transform_mask on prefix)"
            )
    if state_parquet_truncate is not None:
        print(f"state_parquet_truncate={state_parquet_truncate} (slice longer state rows to this width)")
    if action_parquet_truncate is not None:
        print(f"action_parquet_truncate={action_parquet_truncate} (slice longer action rows to this width)")
    if chunk_anchor_camera_reframe:
        print(
            f"chunk_anchor_camera_reframe=True camera_parquet_column={camera_parquet_column!r} "
            "(action windows expressed in anchor camera frame before delta)"
        )

    total_steps = 0
    processed_files = 0

    if num_workers == 1:
        result_iter = (
            _process_single_parquet(
                parquet_file=parquet_file,
                sample_rate=sample_rate,
                action_chunk=action_chunk,
                action_dim=action_dim,
                embodiment_id=embodiment_id,
                mask=mask,
                state_keys=state_keys,
                action_keys=action_keys,
                state_parquet_truncate=state_parquet_truncate,
                action_parquet_truncate=action_parquet_truncate,
                dual_hand_quat_prefix_to_6d=dual_hand_quat_prefix_to_6d,
                chunk_anchor_camera_reframe=chunk_anchor_camera_reframe,
                camera_parquet_column=camera_parquet_column,
            )
            for parquet_file in parquet_files
        )
    else:
        executor = ThreadPoolExecutor(max_workers=num_workers)
        result_iter = executor.map(
            _process_single_parquet,
            parquet_files,
            [sample_rate] * len(parquet_files),
            [action_chunk] * len(parquet_files),
            [action_dim] * len(parquet_files),
            [embodiment_id] * len(parquet_files),
            [mask] * len(parquet_files),
            [state_keys] * len(parquet_files),
            [action_keys] * len(parquet_files),
            [state_parquet_truncate] * len(parquet_files),
            [action_parquet_truncate] * len(parquet_files),
            [dual_hand_quat_prefix_to_6d] * len(parquet_files),
            [chunk_anchor_camera_reframe] * len(parquet_files),
            [camera_parquet_column] * len(parquet_files),
        )

    try:
        for result in tqdm(result_iter, total=len(parquet_files), desc="Processing parquet files"):
            if result is None:
                continue
            states, action_flat, step_count = result
            stats["observation.state"].update(states)
            stats["action"].update(action_flat)
            total_steps += step_count
            processed_files += 1
    finally:
        if num_workers > 1:
            executor.shutdown(wait=True)

    if total_steps < 2:
        raise ValueError("Insufficient valid samples (<2) to compute normalization stats")

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    minmax_quantile_start = _use_minmax_quantiles_after_front_pair(norm_stats, mask)
    quantile_fallback_dims = _stabilize_degenerate_quantiles(
        norm_stats,
        mask,
        min_quantile_range=min_quantile_range,
        min_quantile_std_ratio=min_quantile_std_ratio,
    )

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialize_json(norm_stats))

    print(
        json.dumps(
            {
                "processed_files": processed_files,
                "total_steps": total_steps,
                "minmax_quantile_start": minmax_quantile_start,
                "min_quantile_range": min_quantile_range,
                "min_quantile_std_ratio": min_quantile_std_ratio,
                "quantile_fallback_dims": quantile_fallback_dims,
                "output_path": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    tyro.cli(compute_norm_stats_fast_v2)
