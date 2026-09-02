#!/usr/bin/env python3
"""汇总 RoboTwin 并行评测日志，输出每任务成功率和 clean / randomized 平均值。

用法:
    python scripts/parse_eval_results.py <eval_result_dir> [--accepted-totals 100]

<eval_result_dir> 就是 run_robotwin_parallel.sh 打印的结果目录 (其下有 logs/)。
汇总同时写到 <eval_result_dir>/logs/summary.txt。
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

TASKS = [
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller",
    "handover_block", "handover_mic", "hanging_mug", "lift_pot",
    "move_can_pot", "move_pillbottle_pad", "move_playingcard_away", "move_stapler_pad",
    "open_laptop", "open_microwave", "pick_diverse_bottles", "pick_dual_bottles",
    "place_a2b_left", "place_a2b_right", "place_bread_basket", "place_bread_skillet",
    "place_burger_fries", "place_can_basket", "place_cans_plasticbox", "place_container_plate",
    "place_dual_shoes", "place_empty_cup", "place_fan", "place_mouse_pad",
    "place_object_basket", "place_object_scale", "place_object_stand", "place_phone_stand",
    "place_shoe", "press_stapler", "put_bottles_dustbin", "put_object_cabinet",
    "rotate_qrcode", "scan_object", "shake_bottle", "shake_bottle_horizontally",
    "stack_blocks_three", "stack_blocks_two", "stack_bowls_three", "stack_bowls_two",
    "stamp_seal", "turn_switch",
]

CONFIGS = ["demo_clean", "demo_randomized"]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_log(log_path: Path, accepted_totals: set[int]) -> tuple[str, str]:
    """解析单个任务日志。

    返回 (status, result_str)，status ∈
    {'completed', 'partial', 'running', 'error', 'not_started'}。
    """
    if not log_path.exists():
        return "not_started", "未开始"

    try:
        text = strip_ansi(log_path.read_text(errors="replace"))
    except OSError as exc:
        return "error", f"读取失败: {exc}"

    # 「已启动」：外层 launcher 写的 "开始: <日期>"，或客户端自己的启动横幅
    has_start = (
        bool(re.search(r"开始.*\d{4}-\d{2}-\d{2}", text))
        or "RoboTwin evaluation client" in text
        or "Test episodes" in text
    )
    # 「正常跑完」：外层的 "结束: <日期>"，或客户端最后的 "Results saved to"
    # 两者都只在评估循环走完后出现；崩溃/中断都不会有。
    has_end = (
        bool(re.search(r"结束.*\d{4}-\d{2}-\d{2}", text))
        or "Results saved to" in text
    )
    has_error = "Traceback" in text or ("Error" in text and "ConnectionClosed" not in text)

    matches = re.findall(r"Success rate:\s*(\d+)/(\d+)", text)

    if not has_start:
        return "not_started", "未开始"

    if has_end:
        if not matches:
            # 有结束标记但 0 条成功率记录：多为运动规划全程失败
            return "partial", "部分完成 0集(无成功率记录)"
        suc, total = int(matches[-1][0]), int(matches[-1][1])
        pct = round(suc / total * 100, 1) if total else 0.0
        if total in accepted_totals:
            return "completed", f"{suc}/{total} = {pct}%"
        return "partial", f"部分完成 {suc}/{total} = {pct}% (仅{total}集)"

    if has_error and not matches:
        err_lines = [ln for ln in text.splitlines() if "rror" in ln]
        err_hint = err_lines[-1].strip()[:60] if err_lines else ""
        return "error", f"异常退出: {err_hint}"

    if matches:
        suc, total = int(matches[-1][0]), int(matches[-1][1])
        pct = round(suc / total * 100, 1) if total else 0.0
        return "running", f"正在评估 ({total} episodes, 当前 {suc}/{total}={pct}%)"
    return "running", "正在评估 (初始化中)"


def detect_log_suffix(log_dir: Path, task: str, config: str) -> str:
    """探测日志文件名在 "<task>_<config>" 之后的额外后缀。"""
    base = f"{task}_{config}"
    for path in sorted(log_dir.glob(f"{base}*.log")):
        return path.stem[len(base):]
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize RoboTwin parallel evaluation logs."
    )
    parser.add_argument(
        "eval_dir", type=Path,
        help="Evaluation result directory printed by run_robotwin_parallel.sh",
    )
    parser.add_argument(
        "--accepted-totals", default="50,100",
        help="Episode counts treated as a finished run (comma separated)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    accepted_totals = {int(x) for x in args.accepted_totals.split(",")}

    log_dir = args.eval_dir / "logs"
    if not log_dir.is_dir():
        raise SystemExit(f"日志目录不存在: {log_dir}")

    log_suffix = detect_log_suffix(log_dir, TASKS[0], CONFIGS[0])
    accepted_str = "/".join(str(x) for x in sorted(accepted_totals))

    col_task, col_config, col_result = 32, 16, 45
    separator = "-" * (col_task + col_config + col_result + 2)

    lines = [
        f"评估目录: {args.eval_dir}",
        f"日志后缀: '{log_suffix}' (自动探测)",
        f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        separator,
        f"{'任务':<{col_task}} {'配置':<{col_config}} {'结果':<{col_result}}",
        separator,
    ]

    counts = {k: 0 for k in ("completed", "partial", "running", "error", "not_started")}
    pct_sums: dict[str, list[float]] = {cfg: [] for cfg in CONFIGS}

    for task in TASKS:
        for i, config in enumerate(CONFIGS):
            log_path = log_dir / f"{task}_{config}{log_suffix}.log"
            status, result_str = parse_log(log_path, accepted_totals)
            counts[status] += 1

            if status == "completed":
                match = re.search(r"=\s*([\d.]+)%", result_str)
                if match:
                    pct_sums[config].append(float(match.group(1)))

            task_label = task if i == 0 else ""
            lines.append(
                f"{task_label:<{col_task}} {config:<{col_config}} {result_str:<{col_result}}"
            )
        lines.append("")

    lines += [
        separator,
        f"汇总: 共 {len(TASKS) * len(CONFIGS)} 项",
        f"  已完成(满{accepted_str}集): {counts['completed']}",
        f"  部分完成(集数不符): {counts['partial']}",
        f"  正在评估: {counts['running']}",
        f"  异常退出: {counts['error']}",
        f"  未开始:   {counts['not_started']}",
        separator,
        "",
        f"平均成功率(仅统计跑满 {accepted_str} 集的已完成任务):",
    ]
    for config in CONFIGS:
        vals = pct_sums[config]
        if vals:
            lines.append(
                f"  {config:<20} {len(vals):>2} 个任务  平均 {sum(vals) / len(vals):.1f}%"
            )
        else:
            lines.append(f"  {config:<20} 暂无有效数据")
    lines.append(separator)

    output = "\n".join(lines)
    print(output)

    out_path = log_dir / "summary.txt"
    out_path.write_text(output, encoding="utf-8")
    print(f"\n结果已保存至: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
