#!/usr/bin/env python3
"""Audit one saved non-point lmms-eval run without rewriting responses."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sample_task(path):
    name = Path(path).name
    marker = "_samples_"
    if marker not in name or not name.endswith(".jsonl"):
        raise ValueError("not an lmms sample JSONL: {}".format(path))
    return name.split(marker, 1)[1][:-len(".jsonl")]


def _result_path(sample_path):
    timestamp = sample_path.name.split("_samples_", 1)[0]
    path = sample_path.with_name("{}_results.json".format(timestamp))
    if not path.is_file():
        raise FileNotFoundError("result JSON not found for {}".format(sample_path))
    return path


def _filter_functions(config):
    functions = []
    for filter_group in config.get("filter_list") or []:
        for filter_config in filter_group.get("filter") or []:
            functions.append(str(filter_config.get("function", "")))
    return functions


def audit_sample(sample_path, expected_task, metric_key, score_field):
    sample_path = Path(sample_path).resolve()
    task = _sample_task(sample_path)
    if task != expected_task:
        raise ValueError("unexpected task {} != {}".format(task, expected_task))
    if "recovery" in task.casefold():
        raise ValueError("recovery task is forbidden: {}".format(task))

    rows = [json.loads(line) for line in sample_path.read_text().splitlines()]
    if not rows:
        raise ValueError("empty sample file: {}".format(sample_path))
    doc_ids = [int(row["doc_id"]) for row in rows]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("duplicate doc_id in {}".format(sample_path))
    if any(score_field not in row for row in rows):
        raise ValueError("sample score field {} is missing".format(score_field))

    result_path = _result_path(sample_path)
    result = json.loads(result_path.read_text())
    if set(result.get("results", {})) != {task}:
        raise ValueError("result file must contain exactly task {}".format(task))
    task_result = result["results"][task]
    if metric_key not in task_result:
        raise ValueError("metric {} is missing for {}".format(metric_key, task))
    official_accuracy = float(task_result[metric_key])
    effective = int(result["n-samples"][task]["effective"])
    if effective != len(rows):
        raise ValueError("effective row count mismatch for {}".format(task))

    # For these tasks the official aggregation is an exact dataset mean.
    sample_accuracy = math.fsum(float(row[score_field]) for row in rows) / len(rows)
    if not math.isclose(official_accuracy, sample_accuracy, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "official/sample accuracy mismatch for {}: {} != {}".format(
                task, official_accuracy, sample_accuracy
            )
        )

    config = result["configs"][task]
    return {
        "task": task,
        "rows": len(rows),
        "accuracy": official_accuracy,
        "metric_key": metric_key,
        "score_field": score_field,
        "filter_functions": _filter_functions(config),
        "sample_sha256": _sha256(sample_path),
        "result_sha256": _sha256(result_path),
        "sample_path": str(sample_path),
        "result_path": str(result_path.resolve()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--metric-key", required=True)
    parser.add_argument("--score-field", required=True)
    args = parser.parse_args()

    if args.path.is_dir():
        paths = sorted(args.path.rglob("*_samples_*.jsonl"))
        if len(paths) != 1:
            raise SystemExit("expected exactly one sample JSONL, found {}".format(len(paths)))
        sample_path = paths[0]
    else:
        sample_path = args.path
    audit = audit_sample(
        sample_path,
        expected_task=args.task,
        metric_key=args.metric_key,
        score_field=args.score_field,
    )
    print(json.dumps({"audits": [audit]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
