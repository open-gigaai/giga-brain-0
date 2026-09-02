#!/usr/bin/env python3
"""
get_ebench_val.py
将 EBench 评估结果 JSON 整理为榜单格式，输出4列：
  任务名 | 测评总数 | 成功率(100%, 2位小数) | 分数(归一化到100分, 2位小数)
"""

import argparse
import csv
import json
import os


def parse_task_name(raw_key: str) -> str:
    """从形如 '(60/60)ebench/mobile_manip/task_name' 的 key 中提取最后的任务名"""
    # 去掉括号部分 (XX/XX)，再取最后一段
    key = raw_key
    if key.startswith("("):
        key = key[key.index(")") + 1:]  # -> ebench/mobile_manip/task_name
    return key.strip("/")


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0, out_min: float = 1.0, out_max: float = 100.0) -> float:
    """将 [min_val, max_val] 线性映射到 [out_min, out_max]"""
    if max_val == min_val:
        return out_min
    return out_min + (score - min_val) / (max_val - min_val) * (out_max - out_min)


# 解析 result.json 输入路径
def resolve_result_path(path: str) -> str:
    result_path = os.path.abspath(path)
    if os.path.isdir(result_path):
        result_path = os.path.join(result_path, "result.json")
    return result_path


# 统计任务目录下的 episode 文件夹数量
def count_episode_dirs(result_dir: str, task_path: str) -> int:
    task_dir = os.path.join(result_dir, task_path)
    if not os.path.isdir(task_dir):
        return 0
    return sum(
        1
        for entry in os.scandir(task_dir)
        if entry.is_dir() and not entry.name.startswith(".")
    )


def main():
    parser = argparse.ArgumentParser(description="将 EBench result.json 整理为榜单格式（4列表格）")
    parser.add_argument("--dir", required=True, help="result.json 文件路径，或包含 result.json 的结果目录")
    parser.add_argument("--sort-by", choices=["name", "sr", "score", "none"], default="name", 
                            help="排序方式：name / sr / score / none（保持原始顺序），默认按任务名字母排序")
    args = parser.parse_args()

    result_path = resolve_result_path(args.dir)
    result_dir = os.path.dirname(result_path)

    # ---------- 读取 JSON ----------
    with open(result_path, "r", encoding="utf-8") as f:
        data: dict = json.load(f)

    # 输出 CSV 路径：与 result.json 同目录，同名 .csv
    csv_path = os.path.splitext(result_path)[0] + ".csv"

    # ---------- 解析每个任务 ----------
    rows = []
    for raw_key, metrics in data.items():
        task_name = parse_task_name(raw_key)
        episode_count = count_episode_dirs(result_dir, task_name)
        sr = float(metrics.get("sr", 0.0))
        score_raw = float(metrics.get("score", 0.0))
        score_norm = normalize_score(score_raw)  # 1 ~ 100
        rows.append((task_name, episode_count, sr, score_norm))

    # ---------- 排序 ----------
    if args.sort_by == "name":
        rows.sort(key=lambda x: x[0])
    elif args.sort_by == "sr":
        rows.sort(key=lambda x: x[2], reverse=True)
    elif args.sort_by == "score":
        rows.sort(key=lambda x: x[3], reverse=True)

    # ---------- 计算摘要 ----------
    all_count = [r[1] for r in rows]
    all_sr = [r[2] for r in rows]
    all_score = [r[3] for r in rows]
    total_count = sum(all_count)
    avg_sr = sum(all_sr) / len(all_sr) if all_sr else 0.0
    avg_score = sum(all_score) / len(all_score) if all_score else 0.0

    # ---------- 格式化输出 ----------
    col_w = max(len(r[0]) for r in rows) + 2  # 任务名列宽
    count_w = 12
    sr_w = 18
    score_w = 18

    header_task = "任务名"
    header_count = "测评总数"
    header_sr = "成功率 (SR%)"
    header_score = "分数 (1-100)"

    sep = "-" * (col_w + count_w + sr_w + score_w + 4)

    print(sep)
    print(f"{header_task:<{col_w}}{header_count:<{count_w}}{header_sr:<{sr_w}}{header_score:<{score_w}}")
    print(sep)

    for task_name, episode_count, sr, score_norm in rows:
        sr_str = f"{sr * 100:.2f}%"
        score_str = f"{score_norm:.2f}"
        print(f"{task_name:<{col_w}}{episode_count:<{count_w}}{sr_str:<{sr_w}}{score_str:<{score_w}}")

    print(sep)
    # 汇总行
    avg_sr_str = f"{avg_sr * 100:.2f}%"
    avg_score_str = f"{avg_score:.2f}"
    summary_label = f"Average ({len(rows)} tasks)"
    print(f"{summary_label:<{col_w}}{total_count:<{count_w}}{avg_sr_str:<{sr_w}}{avg_score_str:<{score_w}}")
    print(sep)

    # ---------- 写 CSV ----------
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([header_task, header_count, header_sr, header_score])
            for task_name, episode_count, sr, score_norm in rows:
                writer.writerow([task_name, episode_count, f"{sr * 100:.2f}%", f"{score_norm:.2f}"])
            writer.writerow([summary_label, total_count, avg_sr_str, avg_score_str])
        print(f"CSV saved to: {csv_path}")
    except OSError as exc:
        print(f"[WARN] CSV not saved: {exc}")


if __name__ == "__main__":
    main()
