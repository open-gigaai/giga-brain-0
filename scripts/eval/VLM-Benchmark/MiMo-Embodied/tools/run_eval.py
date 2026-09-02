#!/usr/bin/env python3
"""Run a model profile across MiMo-Embodied tasks and summarize actual scores."""

import argparse
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path


SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SAFE_TASK = re.compile(r"^[A-Za-z0-9._-]+$")


def _is_safe_name(value):
    return bool(SAFE_NAME.fullmatch(value)) and ".." not in value


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_path(mimo_root, model):
    if not _is_safe_name(model):
        raise ValueError(
            "model must start with a letter or digit and contain only letters, "
            "digits, dot, underscore, or dash"
        )
    return mimo_root / "runners" / "model_configs" / (model + ".sh")


def load_profile(mimo_root, model):
    path = _profile_path(mimo_root, model)
    if not path.is_file():
        raise FileNotFoundError("unsupported model or missing profile: {}".format(path))
    completed = subprocess.run(
        ["bash", str(path), "--dump-json"],
        check=True,
        capture_output=True,
        text=True,
    )
    profile = json.loads(completed.stdout)
    required = {
        "schema_version",
        "model_id",
        "display_name",
        "adapter",
        "defaults",
        "model_args",
        "model_arg_profiles",
        "runtime_env_profiles",
        "tasks",
        "aggregates",
    }
    if set(profile) != required or profile["schema_version"] != 1:
        raise ValueError("unsupported or incomplete model profile")
    if profile["model_id"] != model or not _is_safe_name(profile["adapter"]):
        raise ValueError("model profile identity does not match --model")
    _validate_profile(profile)
    return {"path": path, "sha256": _sha256(path), "raw": profile}


def _validate_profile(profile):
    tasks = profile["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("profile tasks must be a non-empty list")
    names = []
    for task in tasks:
        name = task.get("task")
        if not isinstance(name, str) or not SAFE_TASK.fullmatch(name):
            raise ValueError("invalid task name: {!r}".format(name))
        if "recovery" in name.casefold():
            raise ValueError("recovery tasks are forbidden: {}".format(name))
        names.append(name)
        if task.get("kind") not in {"point", "nonpoint"}:
            raise ValueError("invalid task kind for {}".format(name))
        if task.get("model_arg_profile") not in profile["model_arg_profiles"]:
            raise ValueError("unknown model_arg_profile for {}".format(name))
        if task.get("runtime_env_profile") not in profile["runtime_env_profiles"]:
            raise ValueError("unknown runtime_env_profile for {}".format(name))
        seeds = task.get("framework_seed")
        if not isinstance(seeds, list) or len(seeds) != 4 or not all(
            type(value) is int for value in seeds
        ):
            raise ValueError("framework_seed for {} must contain four integers".format(name))
        if task["kind"] == "point":
            generation = task.get("generation_kwargs")
            if not isinstance(generation, dict):
                raise ValueError("point task {} requires generation_kwargs".format(name))
            if generation.get("point_stop_after_first") is not True:
                raise ValueError("point task {} must stop after one complete point".format(name))
            if generation.get("do_sample"):
                if type(generation.get("sampling_seed")) is not int:
                    raise ValueError("sampled task {} requires sampling_seed".format(name))
                if float(generation.get("temperature", 0)) <= 0:
                    raise ValueError("sampled task {} requires temperature > 0".format(name))
            elif "sampling_seed" in generation:
                raise ValueError("greedy task {} must not set sampling_seed".format(name))
        else:
            for key in ("metric_key", "score_field"):
                if not isinstance(task.get(key), str) or not task[key]:
                    raise ValueError("non-point task {} requires {}".format(name, key))
    if len(names) != len(set(names)):
        raise ValueError("profile contains duplicate tasks")

    covered = []
    for aggregate in profile["aggregates"]:
        if set(aggregate) != {"name", "tasks"} or not aggregate["tasks"]:
            raise ValueError("invalid aggregate entry")
        if any(task not in names for task in aggregate["tasks"]):
            raise ValueError("aggregate {} references an unknown task".format(aggregate["name"]))
        covered.extend(aggregate["tasks"])
    if len(covered) != len(set(covered)) or set(covered) != set(names):
        raise ValueError("aggregates must cover every task exactly once")


def _resolve(repo_root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _validate_resources(paths):
    required = {
        "model config": paths["model_path"] / "config.json",
        "model inference config": paths["model_path"] / "inference_config.json",
        "model weights": paths["model_path"] / "diffusion_pytorch_model.bin",
        "GigaModels package": paths["giga_models_dir"] / "giga_models",
        "PaliGemma2 tokenizer": paths["tokenizer_model_path"] / "tokenizer.json",
        "FAST tokenizer": paths["fast_tokenizer_path"] / "tokenizer.json",
        "dataset root": paths["data_root"],
    }
    missing = ["{}: {}".format(label, path) for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("required evaluation resources are missing:\n- " + "\n- ".join(missing))


def _format_model_args(values, paths):
    resolved = {}
    for key, value in values.items():
        if isinstance(value, str):
            value = value.format(**{name: str(path) for name, path in paths.items()})
        resolved[key] = value

    def encode(value):
        if isinstance(value, bool):
            return str(value).lower()
        if value is None:
            return ""
        return str(value)

    for key, value in resolved.items():
        if "," in encode(value):
            raise ValueError("model arg {} contains a comma".format(key))
    return ",".join("{}={}".format(key, encode(value)) for key, value in resolved.items())


def _format_generation_kwargs(values):
    def encode(value):
        return str(value).lower() if isinstance(value, bool) else str(value)

    return ",".join("{}={}".format(key, encode(value)) for key, value in values.items())


def build_plan(profile, selected_tasks=None):
    selected = None if selected_tasks is None else set(selected_tasks)
    plan = []
    for raw in profile["raw"]["tasks"]:
        if selected is not None and raw["task"] not in selected:
            continue
        task = dict(raw)
        model_args = dict(profile["raw"]["model_args"])
        model_args.update(profile["raw"]["model_arg_profiles"][task["model_arg_profile"]])
        model_args.update(task.get("model_args", {}))
        task["resolved_model_args"] = model_args
        task["runtime_env"] = dict(
            profile["raw"]["runtime_env_profiles"][task["runtime_env_profile"]]
        )
        plan.append(task)
    missing = set(selected_tasks or ()) - {task["task"] for task in plan}
    if missing:
        raise ValueError("unknown tasks: {}".format(", ".join(sorted(missing))))
    return sorted(plan, key=lambda item: (-int(item.get("estimated_rows", 0)), item["task"]))


def _find_one(root, pattern):
    matches = [path for path in root.rglob(pattern) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError("expected one {}, found {} under {}".format(pattern, len(matches), root))
    return matches[0]


def _audit_task(mimo_root, entry, sample_path):
    sys.path.insert(0, str(mimo_root))
    if entry["kind"] == "point":
        from tools.audit_single_point_run import audit_sample

        return audit_sample(sample_path)
    from tools.audit_nonpoint_run import audit_sample

    audit = audit_sample(
        sample_path,
        expected_task=entry["task"],
        metric_key=entry["metric_key"],
        score_field=entry["score_field"],
    )
    contains = entry.get("filter_contains")
    filters = audit["filter_functions"]
    if contains is None and filters:
        raise RuntimeError("unexpected parser filter for {}".format(entry["task"]))
    if contains is not None and not any(contains in value for value in filters):
        raise RuntimeError("required parser filter missing for {}".format(entry["task"]))
    return audit


def _run_task(args, profile, entry, gpu, paths):
    task_name = entry["task"]
    task_dir = args.output_root / task_name
    task_dir.mkdir(parents=True)
    raw_dir = task_dir / ".raw"
    raw_dir.mkdir()
    log_path = task_dir / "run.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(args.mimo_root), str(paths["giga_models_dir"]), env.get("PYTHONPATH", "")]
    )
    env["MIMO_DATA_ROOT"] = str(paths["data_root"])
    for key, value in entry["runtime_env"].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)

    command = [
        str(args.python_bin),
        "-m",
        "accelerate.commands.launch",
        "--num_processes=1",
        "-m",
        "lmms_eval",
        "--model",
        profile["raw"]["adapter"],
        "--model_args",
        _format_model_args(entry["resolved_model_args"], paths),
        "--tasks",
        task_name,
        "--batch_size",
        "1",
        "--seed",
        ",".join(str(seed) for seed in entry["framework_seed"]),
        "--log_samples",
        "--log_samples_suffix",
        task_name,
        "--output_path",
        str(raw_dir),
    ]
    if entry["kind"] == "point":
        command.extend(["--gen_kwargs", _format_generation_kwargs(entry["generation_kwargs"])])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])

    run_config = {
        "schema_version": 1,
        "model": profile["raw"]["model_id"],
        "profile_sha256": profile["sha256"],
        "task": task_name,
        "kind": entry["kind"],
        "gpu": int(gpu),
        "framework_seed": entry["framework_seed"],
        "generation_kwargs": entry.get("generation_kwargs"),
        "model_args": _format_model_args(entry["resolved_model_args"], paths),
        "limit": args.limit,
    }
    _atomic_json(task_dir / "run_config.json", run_config)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=args.mimo_root,
            env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            print("[GPU {} {}] {}".format(gpu, task_name, line), end="", flush=True)
        returncode = process.wait()
    if returncode:
        raise RuntimeError("{} failed; see {}".format(task_name, log_path))

    sample_path = _find_one(raw_dir, "*_samples_{}.jsonl".format(task_name))
    result_path = _find_one(raw_dir, "*_results.json")
    audit = _audit_task(args.mimo_root, entry, sample_path)
    stable_sample = task_dir / "samples.jsonl"
    stable_result = task_dir / "results.json"
    shutil.move(str(sample_path), stable_sample)
    shutil.move(str(result_path), stable_result)
    shutil.rmtree(raw_dir)
    audit.update(
        {
            "sample_path": str(stable_sample.relative_to(args.output_root)),
            "result_path": str(stable_result.relative_to(args.output_root)),
            "sample_sha256": _sha256(stable_sample),
            "result_sha256": _sha256(stable_result),
        }
    )
    _atomic_json(task_dir / "audit.json", audit)
    return {
        "task": task_name,
        "kind": entry["kind"],
        "rows": audit["rows"],
        "accuracy": audit["accuracy"],
        "gpu": int(gpu),
        "result_path": str(stable_result.relative_to(args.output_root)),
        "sample_path": str(stable_sample.relative_to(args.output_root)),
    }


def aggregate_results(profile, task_results):
    scores = {result["task"]: result["accuracy"] for result in task_results}
    metrics = []
    for aggregate in profile["raw"]["aggregates"]:
        if not all(task in scores for task in aggregate["tasks"]):
            continue
        accuracy = math.fsum(scores[task] for task in aggregate["tasks"]) / len(
            aggregate["tasks"]
        )
        metrics.append(
            {"name": aggregate["name"], "tasks": aggregate["tasks"], "accuracy": accuracy}
        )
    complete = len(task_results) == len(profile["raw"]["tasks"])
    macro = math.fsum(item["accuracy"] for item in metrics) / len(metrics) if complete else None
    return metrics, macro


def _write_summary_md(path, summary):
    lines = ["# 测评结果", "", "| 指标 | 精度 |", "|---|---:|"]
    for item in summary["metrics_13"]:
        lines.append("| {} | {:.6f} |".format(item["name"], item["accuracy"]))
    if summary["macro_accuracy"] is not None:
        lines.extend(["", "13 项宏平均：`{:.12f}`".format(summary["macro_accuracy"])])
    lines.extend(["", "## 底层任务", "", "| 任务 | 样本数 | 精度 |", "|---|---:|---:|"])
    for item in summary["underlying_tasks"]:
        lines.append("| {} | {} | {:.6f} |".format(item["task"], item["rows"], item["accuracy"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    mimo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--giga-models", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--fast-tokenizer", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--tasks", help="comma-separated subset for smoke testing")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.mimo_root = mimo_root
    args.repo_root = mimo_root.parent
    # Keep the virtualenv entry path. Resolving its symlink would launch the base
    # interpreter and lose packages installed only in the evaluation environment.
    args.python_bin = Path(sys.executable).absolute()
    return args


def main():
    args = parse_args()
    profile = load_profile(args.mimo_root, args.model)
    defaults = profile["raw"]["defaults"]
    paths = {
        "model_path": _resolve(args.repo_root, args.model_path or defaults["model_path"]),
        "giga_models_dir": _resolve(
            args.repo_root, args.giga_models or defaults["giga_models_dir"]
        ),
        "tokenizer_model_path": _resolve(
            args.repo_root, args.tokenizer or defaults["tokenizer_model_path"]
        ),
        "fast_tokenizer_path": _resolve(
            args.repo_root, args.fast_tokenizer or defaults["fast_tokenizer_path"]
        ),
        "data_root": _resolve(args.repo_root, args.data_root or defaults["data_root"]),
    }
    selected = args.tasks.split(",") if args.tasks else None
    plan = build_plan(profile, selected)
    gpu_values = [int(value) for value in args.gpus.split(",")]
    if not gpu_values or len(gpu_values) != len(set(gpu_values)) or any(gpu < 0 for gpu in gpu_values):
        raise ValueError("--gpus must be a non-empty list of unique non-negative IDs")

    public_plan = {
        "model": profile["raw"]["model_id"],
        "profile": str(profile["path"].relative_to(args.repo_root)),
        "profile_sha256": profile["sha256"],
        "gpus": gpu_values,
        "tasks": [
            {
                "task": entry["task"],
                "kind": entry["kind"],
                "estimated_rows": entry.get("estimated_rows"),
                "framework_seed": entry["framework_seed"],
                "generation_kwargs": entry.get("generation_kwargs"),
            }
            for entry in plan
        ],
    }
    if args.dry_run:
        print(json.dumps(public_plan, indent=2, ensure_ascii=False))
        return 0

    _validate_resources(paths)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.output_root is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
        args.output_root = _resolve(
            args.repo_root, Path(defaults["output_root"]) / timestamp
        )
    else:
        args.output_root = _resolve(args.repo_root, args.output_root)
    if args.output_root.exists():
        raise FileExistsError("refusing to overwrite output root: {}".format(args.output_root))
    args.output_root.mkdir(parents=True)
    _atomic_json(args.output_root / "run_plan.json", public_plan)

    jobs = queue.Queue()
    for entry in plan:
        jobs.put(entry)
    completed = []
    failures = []
    lock = threading.Lock()
    stop_event = threading.Event()

    def worker(gpu):
        while not stop_event.is_set():
            try:
                entry = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                result = _run_task(args, profile, entry, gpu, paths)
                with lock:
                    completed.append(result)
            except Exception as exc:
                with lock:
                    failures.append({"task": entry["task"], "gpu": gpu, "error": str(exc)})
                stop_event.set()
            finally:
                jobs.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpu_values]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    metrics, macro = aggregate_results(profile, completed)
    summary = {
        "schema_version": 1,
        "model": profile["raw"]["model_id"],
        "status": "complete" if not failures and len(completed) == len(plan) else "failed",
        "underlying_tasks": sorted(completed, key=lambda item: item["task"]),
        "metrics_13": metrics,
        "macro_accuracy": macro,
        "failures": sorted(failures, key=lambda item: item["task"]),
    }
    _atomic_json(args.output_root / "summary.json", summary)
    _write_summary_md(args.output_root / "summary.md", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
