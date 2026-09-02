#!/usr/bin/env python3
"""Run LIBERO evaluation through harrix.

The script accepts either a full harrix EvalConfig YAML or a checkpoint path
plus common command-line overrides. It intentionally bypasses the legacy
Wall-X inference stack.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

LIBERO_DEFAULT_MAX_INFER_TIMES = {
    "libero_spatial": 22,
    "libero_object": 28,
    "libero_goal": 30,
    "libero_10": 52,
    "libero_90": 40,
}


def _ensure_local_harrix_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    harrix_python = repo_root / "third_party" / "harrix" / "python"
    if harrix_python.is_dir():
        sys.path.insert(0, str(harrix_python))


def _parse_task_indices(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(x) for x in value.split(",") if x.strip()]


def _resolve_max_infer_times(
    task_suite_name: str | None, max_infer_times: int | None
) -> int:
    if max_infer_times is not None:
        return max_infer_times
    suite = task_suite_name or "libero_spatial"
    return LIBERO_DEFAULT_MAX_INFER_TIMES.get(suite, 22)


def _load_or_build_raw_config(args: argparse.Namespace) -> dict:
    if args.config is not None:
        with open(args.config, "r") as f:
            raw = yaml.safe_load(f) or {}
        model = raw.setdefault("model", {})
        env = raw.setdefault("env", {})
        libero = env.setdefault("libero", {})
        runtime = raw.setdefault("runtime", {})
        debug = raw.setdefault("debug", {})
        if args.checkpoint_path is not None:
            model["checkpoint_path"] = args.checkpoint_path
        if args.train_config_path is not None:
            model["train_config_path"] = args.train_config_path
        task_indices = _parse_task_indices(args.task_indices)
        if task_indices is not None:
            libero["task_indices"] = task_indices
        if args.max_infer_times is not None or libero.get("max_infer_times") is None:
            libero["max_infer_times"] = _resolve_max_infer_times(
                libero.get("task_suite_name", args.task_suite_name),
                args.max_infer_times,
            )
        if args.smoke:
            libero["task_indices"] = [0]
            libero["num_trials_per_task"] = 5
            runtime["num_workers"] = 1
            runtime["max_batch_size"] = 1
        if args.deterministic_model:
            debug["deterministic_model"] = True
        return raw
    else:
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint-path is required when --config is not set")
        max_infer_times = _resolve_max_infer_times(
            args.task_suite_name, args.max_infer_times
        )
        raw = {
            "model": {
                "checkpoint_path": args.checkpoint_path,
                "norm_key": args.norm_key,
                "cam_names": args.cam_names,
                "architecture": args.architecture,
                "action_mode": args.action_mode,
            },
            "env": {
                "type": "libero",
                "seed": args.seed,
                "libero": {
                    "task_suite_name": args.task_suite_name,
                    "initial_states_path": args.initial_states_path,
                    "num_trials_per_task": args.num_trials_per_task,
                    "max_infer_times": max_infer_times,
                    "skip_intermediate_render": args.skip_intermediate_render,
                },
            },
            "runtime": {
                "num_workers": args.num_workers,
                "max_batch_size": args.max_batch_size,
                "ws_port": args.ws_port,
                "log_dir": args.log_dir,
                "driver_mode": args.driver_mode,
            },
            "debug": {"deterministic_model": args.deterministic_model},
        }

    model = raw.setdefault("model", {})
    env = raw.setdefault("env", {})
    libero = env.setdefault("libero", {})
    runtime = raw.setdefault("runtime", {})
    debug = raw.setdefault("debug", {})

    if args.checkpoint_path is not None:
        model["checkpoint_path"] = args.checkpoint_path
    if args.train_config_path is not None:
        model["train_config_path"] = args.train_config_path
    if args.norm_key is not None:
        model["norm_key"] = args.norm_key
    if args.cam_names is not None:
        model["cam_names"] = args.cam_names
    if args.action_horizon is not None:
        model["action_horizon"] = args.action_horizon
    if args.architecture is not None:
        model["architecture"] = args.architecture
    if args.action_mode is not None:
        model["action_mode"] = args.action_mode

    env["type"] = "libero"
    env["seed"] = args.seed
    libero["task_suite_name"] = args.task_suite_name
    libero["initial_states_path"] = args.initial_states_path
    libero["num_trials_per_task"] = args.num_trials_per_task
    libero["max_infer_times"] = _resolve_max_infer_times(
        args.task_suite_name, args.max_infer_times
    )
    libero["skip_intermediate_render"] = args.skip_intermediate_render
    task_indices = _parse_task_indices(args.task_indices)
    if task_indices is not None:
        libero["task_indices"] = task_indices
    if args.smoke:
        libero["task_indices"] = [0]
        libero["num_trials_per_task"] = 5
        runtime["num_workers"] = 1
        runtime["max_batch_size"] = 1

    runtime["num_workers"] = (
        args.num_workers if not args.smoke else runtime["num_workers"]
    )
    runtime["max_batch_size"] = (
        args.max_batch_size if not args.smoke else runtime["max_batch_size"]
    )
    runtime["ws_port"] = args.ws_port
    runtime["log_dir"] = args.log_dir
    runtime["driver_mode"] = args.driver_mode
    debug["deterministic_model"] = args.deterministic_model
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=None, help="Optional harrix EvalConfig YAML."
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--train-config-path", default=None)
    parser.add_argument("--norm-key", default="libero_all")
    parser.add_argument("--architecture", default="qwen2_5")
    parser.add_argument("--action-mode", default="flow")
    parser.add_argument(
        "--cam-names", nargs="+", default=["face_view", "right_wrist_view"]
    )
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--initial-states-path", default="DEFAULT")
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument(
        "--task-indices", default=None, help="Comma-separated task ids."
    )
    parser.add_argument(
        "--max-infer-times",
        type=int,
        default=None,
        help=(
            "Number of model action chunks per episode. Defaults are suite-specific "
            "and match the internal LIBERO evaluator."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--log-dir", default="outputs/libero/logs")
    parser.add_argument("--driver-mode", choices=["in_process"], default="in_process")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--deterministic-model", action="store_true")
    parser.add_argument(
        "--skip-intermediate-render",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _ensure_local_harrix_on_path()

    from wall_x._vendor.harrix.eval_config import (
        autofill_from_checkpoint,
        load_eval_config,
    )

    raw = _load_or_build_raw_config(args)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(raw, f, sort_keys=False)
        tmp_config = f.name

    cfg = autofill_from_checkpoint(load_eval_config(tmp_config))
    if cfg.runtime.driver_mode != "in_process":
        raise ValueError("Only driver_mode='in_process' is supported")
    from wall_x._vendor.harrix.drivers.inproc import run

    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
