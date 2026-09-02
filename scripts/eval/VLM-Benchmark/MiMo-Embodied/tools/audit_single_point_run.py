#!/usr/bin/env python3
"""Audit saved point-task results for answer-preserving single-point scoring."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from lmms_eval.tasks._task_utils.eval_utils import (
    _POINT_PAIR_RE,
    extract_after_think_content,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scalar_response(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str):
        raise ValueError(f"response is not a scalar string: {value!r}")
    return value


def _pairs(text: str):
    final = extract_after_think_content(str(text), strict=True)
    return [tuple(map(float, pair)) for pair in _POINT_PAIR_RE.findall(final)]


def _find_result_file(sample_path: Path) -> Path:
    timestamp = sample_path.name.split("_samples_", 1)[0]
    candidate = sample_path.with_name(f"{timestamp}_results.json")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"result JSON not found for {sample_path}")


def _task_from_sample_name(sample_path: Path) -> str:
    name = sample_path.name
    marker = "_samples_"
    if marker not in name or not name.endswith(".jsonl"):
        raise ValueError(f"not an lmms sample JSONL: {sample_path}")
    return name.split(marker, 1)[1][:-len(".jsonl")]


def audit_sample(sample_path: Path):
    task = _task_from_sample_name(sample_path)
    if "recovery" in task.casefold():
        raise ValueError(f"recovery task is forbidden: {task}")

    rows = [json.loads(line) for line in sample_path.read_text().splitlines()]
    if not rows:
        raise ValueError(f"empty sample file: {sample_path}")

    raw_histogram = Counter()
    filtered_histogram = Counter()
    mismatches = []
    doc_ids = []
    scores = []
    for row in rows:
        doc_id = int(row["doc_id"])
        doc_ids.append(doc_id)
        raw = _scalar_response(row["resps"])
        filtered = _scalar_response(row["filtered_resps"])
        raw_pairs = _pairs(raw)
        filtered_pairs = _pairs(filtered)
        raw_histogram[len(raw_pairs)] += 1
        filtered_histogram[len(filtered_pairs)] += 1
        if len(raw_pairs) > 1:
            mismatches.append({"doc_id": doc_id, "reason": "multiple raw points"})
        if raw_pairs != filtered_pairs:
            mismatches.append(
                {
                    "doc_id": doc_id,
                    "reason": "raw/filtered coordinates differ",
                    "raw": raw_pairs,
                    "filtered": filtered_pairs,
                }
            )
        scores.append(float(row["accuracy"]))

    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError(f"duplicate doc_id in {sample_path}")
    if mismatches:
        preview = json.dumps(mismatches[:10], ensure_ascii=True)
        raise ValueError(f"single-point answer-preservation audit failed: {preview}")

    result_path = _find_result_file(sample_path)
    result = json.loads(result_path.read_text())
    task_result = result["results"][task]
    metric_keys = [
        key
        for key in task_result
        if key.startswith("accuracy,") and not key.startswith("accuracy_stderr,")
    ]
    if len(metric_keys) != 1:
        raise ValueError(f"expected one accuracy metric for {task}: {metric_keys}")
    official_accuracy = float(task_result[metric_keys[0]])
    sample_accuracy = math.fsum(scores) / len(scores)
    if not math.isclose(official_accuracy, sample_accuracy, abs_tol=1e-12):
        raise ValueError(
            f"official/sample accuracy mismatch for {task}: "
            f"{official_accuracy} != {sample_accuracy}"
        )

    config = result["configs"][task]
    filters = config.get("filter_list") or []
    filter_function = filters[0]["filter"][0].get("function", "") if filters else ""
    if not str(filter_function).endswith("eval_utils.PointFilter'>"):
        raise ValueError(f"non-primary point filter for {task}: {filter_function}")
    generation_kwargs = config.get("generation_kwargs") or {}
    if generation_kwargs.get("point_stop_after_first") is not True:
        raise ValueError(f"point_stop_after_first is not true for {task}")
    effective = int(result["n-samples"][task]["effective"])
    if effective != len(rows):
        raise ValueError(f"effective row count mismatch for {task}: {effective}")

    return {
        "task": task,
        "rows": len(rows),
        "accuracy": official_accuracy,
        "raw_pair_histogram": dict(sorted(raw_histogram.items())),
        "filtered_pair_histogram": dict(sorted(filtered_histogram.items())),
        "raw_filtered_coordinates_identical": True,
        "filter": filter_function,
        "sample_sha256": _sha256(sample_path),
        "result_sha256": _sha256(result_path),
        "sample_path": str(sample_path.resolve()),
        "result_path": str(result_path.resolve()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Sample JSONL files or directories containing *_samples_*.jsonl",
    )
    args = parser.parse_args()

    sample_paths = []
    for path in args.paths:
        if path.is_dir():
            sample_paths.extend(sorted(path.rglob("*_samples_*.jsonl")))
        else:
            sample_paths.append(path)
    if not sample_paths:
        raise SystemExit("no sample JSONL files found")

    audits = [audit_sample(path) for path in sample_paths]
    print(json.dumps({"audits": audits}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
