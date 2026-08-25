"""Check GigaBrain robot norm-stat files for suspicious dimensions.

The training transform uses quantile normalization:

    normalized = (x - q01) / (q99 - q01 + 1e-6) * 2 - 1

Very small ``q99 - q01`` ranges make otherwise finite actions explode after
normalization and can later crash the FAST action tokenizer. This script audits
the norm-stat files referenced by a training config and reports problematic
groups/dimensions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT.parent
DEFAULT_CONFIG = PROJECT_ROOT / 'configs' / 'gb07_pg2_pick_and_place_piper_30k.py'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'temp' / 'norm_stats_anomalies'


def _resolve_dependency_root(env_name: str, *sibling_names: str) -> Path:
    configured = os.environ.get(env_name)
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f'{env_name} does not exist: {root}')
        return root
    for sibling_name in sibling_names:
        root = CODE_ROOT / sibling_name
        if root.is_dir():
            return root
    expected = ', '.join(os.fspath(CODE_ROOT / name) for name in sibling_names)
    raise FileNotFoundError(f'Set {env_name}; no dependency checkout found at: {expected}')


GIGA_TRAIN_ROOT = _resolve_dependency_root('GIGA_TRAIN_ROOT', 'giga-train')
GIGA_DATASETS_ROOT = _resolve_dependency_root(
    'GIGA_DATASETS_ROOT', 'giga-datasets', 'giga-datasets-v3.0'
)
os.environ['GIGA_TRAIN_ROOT'] = os.fspath(GIGA_TRAIN_ROOT)
os.environ['GIGA_DATASETS_ROOT'] = os.fspath(GIGA_DATASETS_ROOT)
for _path in reversed(
    (REPO_ROOT, PROJECT_ROOT, GIGA_TRAIN_ROOT, GIGA_DATASETS_ROOT)
):
    if os.fspath(_path) not in sys.path:
        sys.path.insert(0, os.fspath(_path))


@dataclass(frozen=True)
class NormGroup:
    name: str
    norm_path: str
    selector_values: list[str]


def _load_python_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_plain(value: Any) -> Any:
    if hasattr(value, 'to_dict'):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_plain(v) for v in value)
    return value


def _load_train_config(config_path: Path) -> tuple[Any, dict[str, Any]]:
    if config_path.suffix == '.py':
        module = _load_python_module(config_path)
        return module, _to_plain(module.config)

    from giga_train import load_config

    return None, _to_plain(load_config(str(config_path)))


def _norm_groups_from_config(module: Any, train_config: dict[str, Any]) -> list[NormGroup]:
    train = train_config['dataloaders']['train']
    norm_cfg = train['transform']['norm_cfg']
    norm_stats_path = norm_cfg['norm_stats_path']

    group_names = getattr(module, 'group_names', None) if module is not None else None

    if isinstance(norm_stats_path, list):
        groups = []
        for idx, entry in enumerate(norm_stats_path):
            selector_values = entry.get('selector_values', entry.get('data_paths', entry.get('keys')))
            if selector_values is None:
                selector_values = []
            if isinstance(selector_values, (str, os.PathLike)):
                selector_values = [selector_values]
            norm_path = entry.get('norm_stats_path', entry.get('path'))
            if norm_path is None:
                raise KeyError(f'Norm stats entry {idx} has no path/norm_stats_path.')
            name = entry.get('name', entry.get('group_key'))
            if name is None and group_names and idx < len(group_names):
                name = group_names[idx]
            if name is None:
                name = f'__norm_group_{idx}'
            groups.append(NormGroup(str(name), os.fspath(norm_path), [os.fspath(v) for v in selector_values]))
        return groups

    if isinstance(norm_stats_path, dict):
        return [
            NormGroup(str(key), os.fspath(path), [str(key)])
            for key, path in norm_stats_path.items()
        ]

    raise TypeError('Unsupported norm_cfg.norm_stats_path format.')


def _load_stats(path: str) -> dict[str, Any]:
    with open(path, 'r') as f:
        data = json.load(f)
    return data['norm_stats']


def _array(stats: dict[str, Any], key: str, field: str) -> np.ndarray:
    if key not in stats:
        raise KeyError(f'{key!r} missing from norm_stats.')
    if field not in stats[key]:
        raise KeyError(f'{key}.{field!r} missing from norm_stats.')
    return np.asarray(stats[key][field], dtype=np.float64)


def _finite_min(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.min())


def _finite_max(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.max())


def _stats_record(
    *,
    group: NormGroup,
    tensor_key: str,
    dim: int,
    issue: str,
    severity: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    return {
        'group': group.name,
        'norm_path': group.norm_path,
        'tensor': tensor_key,
        'dim': int(dim),
        'issue': issue,
        'severity': severity,
        **values,
        'selector_value_count': len(group.selector_values),
        'selector_values_sample': group.selector_values[:3],
    }


def _audit_tensor(
    *,
    group: NormGroup,
    stats: dict[str, Any],
    tensor_key: str,
    min_quantile_range: float,
    tiny_std: float,
    large_mean_abs: float,
    qrange_to_std_ratio: float,
    ignore_zero_padding: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    q01 = _array(stats, tensor_key, 'q01')
    q99 = _array(stats, tensor_key, 'q99')
    mean = _array(stats, tensor_key, 'mean')
    std = _array(stats, tensor_key, 'std')
    lengths = {len(q01), len(q99), len(mean), len(std)}
    records: list[dict[str, Any]] = []

    if len(lengths) != 1:
        max_len = max(lengths)
        records.append(
            _stats_record(
                group=group,
                tensor_key=tensor_key,
                dim=-1,
                issue='length_mismatch',
                severity='error',
                values={'lengths': {'q01': len(q01), 'q99': len(q99), 'mean': len(mean), 'std': len(std)}},
            )
        )
        q01 = np.resize(q01, max_len)
        q99 = np.resize(q99, max_len)
        mean = np.resize(mean, max_len)
        std = np.resize(std, max_len)

    qrange = q99 - q01
    for dim in range(len(qrange)):
        dim_values = {
            'q01': float(q01[dim]) if np.isfinite(q01[dim]) else str(q01[dim]),
            'q99': float(q99[dim]) if np.isfinite(q99[dim]) else str(q99[dim]),
            'qrange': float(qrange[dim]) if np.isfinite(qrange[dim]) else str(qrange[dim]),
            'mean': float(mean[dim]) if np.isfinite(mean[dim]) else str(mean[dim]),
            'std': float(std[dim]) if np.isfinite(std[dim]) else str(std[dim]),
        }

        if not np.isfinite([q01[dim], q99[dim], qrange[dim], mean[dim], std[dim]]).all():
            records.append(
                _stats_record(
                    group=group,
                    tensor_key=tensor_key,
                    dim=dim,
                    issue='non_finite_stat',
                    severity='error',
                    values=dim_values,
                )
            )
            continue

        if ignore_zero_padding and q01[dim] == 0.0 and q99[dim] == 0.0 and mean[dim] == 0.0 and std[dim] == 0.0:
            continue

        if q99[dim] < q01[dim]:
            records.append(
                _stats_record(
                    group=group,
                    tensor_key=tensor_key,
                    dim=dim,
                    issue='q99_less_than_q01',
                    severity='error',
                    values=dim_values,
                )
            )
        elif qrange[dim] <= min_quantile_range:
            records.append(
                _stats_record(
                    group=group,
                    tensor_key=tensor_key,
                    dim=dim,
                    issue='tiny_quantile_range',
                    severity='error',
                    values=dim_values,
                )
            )

        if std[dim] <= tiny_std:
            records.append(
                _stats_record(
                    group=group,
                    tensor_key=tensor_key,
                    dim=dim,
                    issue='tiny_std',
                    severity='warn',
                    values=dim_values,
                )
            )

        if abs(mean[dim]) >= large_mean_abs and qrange[dim] <= 1.0:
            records.append(
                _stats_record(
                    group=group,
                    tensor_key=tensor_key,
                    dim=dim,
                    issue='large_mean_with_small_quantile_range',
                    severity='warn',
                    values=dim_values,
                )
            )

        ratio = qrange[dim] / (std[dim] + 1e-12)
        if ratio <= qrange_to_std_ratio:
            values = {**dim_values, 'qrange_to_std': float(ratio)}
            records.append(
                _stats_record(
                    group=group,
                    tensor_key=tensor_key,
                    dim=dim,
                    issue='quantile_range_too_small_vs_std',
                    severity='warn',
                    values=values,
                )
            )

    summary = {
        'dim': int(len(qrange)),
        'qrange_min': _finite_min(qrange),
        'qrange_max': _finite_max(qrange),
        'std_min': _finite_min(std),
        'std_max': _finite_max(std),
        'mean_min': _finite_min(mean),
        'mean_max': _finite_max(mean),
    }
    return records, summary


def _print_top_records(records: list[dict[str, Any]], limit: int) -> None:
    if not records:
        print('[norm] no anomalies found')
        return

    severity_rank = {'error': 0, 'warn': 1}
    records = sorted(
        records,
        key=lambda r: (
            severity_rank.get(r['severity'], 9),
            r['group'],
            r['tensor'],
            r['dim'],
            r['issue'],
        ),
    )
    print(f'[norm] anomalies={len(records)} showing={min(limit, len(records))}')
    for record in records[:limit]:
        print(
            f"[{record['severity']}] group={record['group']} tensor={record['tensor']} "
            f"dim={record['dim']} issue={record['issue']} "
            f"q01={record.get('q01')} q99={record.get('q99')} "
            f"qrange={record.get('qrange')} std={record.get('std')} "
            f"path={record['norm_path']}",
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Check GigaBrain norm stats for suspicious dimensions.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--group-filter', default=None)
    parser.add_argument('--path-filter', default=None)
    parser.add_argument('--tensor', choices=('action', 'observation.state', 'both'), default='both')
    parser.add_argument('--min-quantile-range', type=float, default=1e-3)
    parser.add_argument('--tiny-std', type=float, default=1e-6)
    parser.add_argument('--large-mean-abs', type=float, default=10.0)
    parser.add_argument('--qrange-to-std-ratio', type=float, default=0.05)
    parser.add_argument(
        '--ignore-zero-padding',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Ignore dims whose q01/q99/mean/std are all zero. These are usually padded action tail dims.',
    )
    parser.add_argument('--print-limit', type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    module, train_config = _load_train_config(args.config.resolve())
    groups = _norm_groups_from_config(module, train_config)
    if args.group_filter:
        groups = [group for group in groups if args.group_filter in group.name]
    if args.path_filter:
        groups = [group for group in groups if args.path_filter in group.norm_path]
    if not groups:
        raise ValueError('No norm-stat groups matched the filters.')

    tensors = ['action', 'observation.state'] if args.tensor == 'both' else [args.tensor]
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / 'anomalies.jsonl'
    summary_path = output_dir / 'summary.json'

    all_records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        'config': str(args.config.resolve()),
        'groups_checked': 0,
        'records': 0,
        'by_group': {},
        'records_path': str(records_path),
        'summary_path': str(summary_path),
        'thresholds': {
            'min_quantile_range': args.min_quantile_range,
            'tiny_std': args.tiny_std,
            'large_mean_abs': args.large_mean_abs,
            'qrange_to_std_ratio': args.qrange_to_std_ratio,
            'ignore_zero_padding': args.ignore_zero_padding,
        },
    }

    for group in groups:
        if not Path(group.norm_path).exists():
            record = {
                'group': group.name,
                'norm_path': group.norm_path,
                'issue': 'missing_norm_stats_file',
                'severity': 'error',
            }
            all_records.append(record)
            continue
        stats = _load_stats(group.norm_path)
        summary['groups_checked'] += 1
        group_summary = summary['by_group'].setdefault(group.name, {'norm_path': group.norm_path, 'tensors': {}, 'anomalies': 0})
        for tensor_key in tensors:
            records, tensor_summary = _audit_tensor(
                group=group,
                stats=stats,
                tensor_key=tensor_key,
                min_quantile_range=args.min_quantile_range,
                tiny_std=args.tiny_std,
                large_mean_abs=args.large_mean_abs,
                qrange_to_std_ratio=args.qrange_to_std_ratio,
                ignore_zero_padding=args.ignore_zero_padding,
            )
            group_summary['tensors'][tensor_key] = tensor_summary
            group_summary['anomalies'] += len(records)
            all_records.extend(records)

    with records_path.open('w') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    summary['records'] = len(all_records)
    with summary_path.open('w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _print_top_records(all_records, args.print_limit)
    print(f'[output] summary={summary_path} anomalies={records_path}')
    return 1 if any(r.get('severity') == 'error' for r in all_records) else 0


if __name__ == '__main__':
    raise SystemExit(main())
