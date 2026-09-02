# Copyright (C) 2026 Xiaomi Corporation.
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize eval_vlm metrics into one dataset-level table.")
    parser.add_argument("--results-dir", default="eval_vlm/results")
    parser.add_argument("--output-json", default="eval_vlm/summary_metrics.json")
    parser.add_argument("--output-csv", default="eval_vlm/summary_metrics.csv")
    return parser.parse_args()


def dataset_name(path: Path) -> str:
    name = path.name
    suffix = "_predictions.metrics.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name.replace(".metrics.json", "")


def weighted_average(parts: List[Dict[str, Any]]) -> Optional[float]:
    total = sum(int(part.get("num_examples", 0) or 0) for part in parts)
    if total == 0:
        return None
    accum = 0.0
    for part in parts:
        n = int(part.get("num_examples", 0) or 0)
        if "accuracy" in part:
            score = part["accuracy"]
        elif "success_rate" in part:
            score = part["success_rate"]
        else:
            continue
        accum += float(score) * n
    return accum / total


def extract_summary(path: Path) -> Dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    row: Dict[str, Any] = {
        "dataset": dataset_name(path),
        "metrics_file": str(path),
        "num_examples": metrics.get("num_examples"),
        "primary_metric": "",
        "overall_score": None,
        "accuracy": metrics.get("accuracy"),
        "success_rate": metrics.get("success_rate"),
        "mean_point_hit_rate": metrics.get("mean_point_hit_rate"),
        "mean_iou": metrics.get("mean_iou"),
        "point_hit_rate": metrics.get("point_hit_rate"),
        "correct": metrics.get("correct"),
        "success_count": metrics.get("success_count"),
        "notes": "",
    }

    if "accuracy" in metrics:
        row["primary_metric"] = "accuracy"
        row["overall_score"] = metrics["accuracy"]
    elif "success_rate" in metrics:
        row["primary_metric"] = "success_rate"
        row["overall_score"] = metrics["success_rate"]
    elif "exact" in metrics or "point" in metrics:
        parts = [metrics[key] for key in ("exact", "point") if isinstance(metrics.get(key), dict)]
        row["num_examples"] = sum(int(part.get("num_examples", 0) or 0) for part in parts)
        row["overall_score"] = weighted_average(parts)
        row["primary_metric"] = "weighted_accuracy_success_rate"
        if isinstance(metrics.get("exact"), dict):
            row["accuracy"] = metrics["exact"].get("accuracy")
        if isinstance(metrics.get("point"), dict):
            row["success_rate"] = metrics["point"].get("success_rate")
            row["mean_point_hit_rate"] = metrics["point"].get("mean_point_hit_rate")
        row["notes"] = "mixed exact and point tasks; overall is sample-weighted"
    else:
        row["notes"] = "no recognized aggregate metric"

    if row["mean_iou"] is not None:
        row["notes"] = (row["notes"] + "; " if row["notes"] else "") + "bbox task: accuracy is IoU-threshold accuracy"

    return row


def main() -> None:
    args = parse_args()
    paths = sorted(Path(args.results_dir).glob("*.metrics.json"))
    rows = [extract_summary(path) for path in paths]

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "dataset",
        "num_examples",
        "primary_metric",
        "overall_score",
        "accuracy",
        "success_rate",
        "mean_point_hit_rate",
        "mean_iou",
        "point_hit_rate",
        "correct",
        "success_count",
        "notes",
        "metrics_file",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    print("\nDataset summary:")
    for row in rows:
        score = row["overall_score"]
        score_text = "n/a" if score is None else f"{score:.4f}"
        print(f"{row['dataset']}: {row['primary_metric']}={score_text} n={row['num_examples']}")


if __name__ == "__main__":
    main()
