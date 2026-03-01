#!/usr/bin/env python3
"""
Decode 长度 Sweep 分析: 对比不同 output_len 下 BF16 vs W8A8D 的吞吐与加速比.

输入: run_sweep.sh 产生的 results 目录 (含 {quant}_olen{N}.json 文件)
输出: 对比表格 (txt / markdown / csv)

用法:
    python3 analyze_sweep.py results/                  # 文本表格 (stdout)
    python3 analyze_sweep.py results/ --markdown       # Markdown 格式
    python3 analyze_sweep.py results/ --csv            # CSV 格式
    python3 analyze_sweep.py results/ --input-len 1    # 指定 input_len (用于 output tput 计算)
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

# 结果文件命名: {quant}_olen{output_len}.json
FILENAME_RE = re.compile(r'^(bf16|w8a8)_olen(\d+)\.json$')

# BF16 为基线
BASELINE_QUANT = "bf16"


def _to_float(val) -> float | None:
    """安全转换为 float."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isfinite(f):
            return f
    except (TypeError, ValueError):
        pass
    return None


def load_result(path: Path) -> dict | None:
    """加载并解析 JSON 结果文件."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"警告: 无法加载 {path}: {e}", file=sys.stderr)
        return None


def extract_metrics(
    data: dict,
    *,
    input_len: int,
    output_len: int,
) -> dict:
    """
    从 vllm bench throughput JSON 中提取关键指标.

    兼容新旧两种输出格式:
      旧: elapsed_time, tokens_per_second, requests_per_second
      新: output_throughput, total_token_throughput, request_throughput

    返回: {output_tput, total_tput, elapsed_time, num_requests, source}
    """
    elapsed = _to_float(data.get("elapsed_time"))
    num_req = _to_float(data.get("num_requests"))

    # output throughput: 优先用 JSON 原生字段
    output_tput = _to_float(data.get("output_throughput"))
    source = "measured"

    # 回退: 从 total_output_tokens / elapsed_time 推算
    if output_tput is None:
        total_output = _to_float(data.get("total_output_tokens"))
        if total_output and elapsed and elapsed > 0:
            output_tput = total_output / elapsed
            source = "derived_total_output_elapsed"

    # 再回退: 从 num_requests * output_len / elapsed_time 推算
    if output_tput is None and num_req and elapsed and elapsed > 0:
        # n (num_samples) 已经乘进 total tokens 了, 但 num_requests 是原始 prompt 数
        # 需要从 total_num_tokens 反推
        total_tokens = _to_float(data.get("total_num_tokens"))
        if total_tokens and elapsed > 0:
            total_tput_derived = total_tokens / elapsed
            # output 占比 = output_len / (input_len + output_len)
            ratio = output_len / (input_len + output_len)
            output_tput = total_tput_derived * ratio
            source = "derived_ratio_from_total"

    # total throughput
    total_tput = (
        _to_float(data.get("total_token_throughput"))
        or _to_float(data.get("tokens_per_second"))
    )
    if total_tput is None and elapsed and elapsed > 0:
        total_tokens = _to_float(data.get("total_num_tokens"))
        if total_tokens:
            total_tput = total_tokens / elapsed

    return {
        "output_tput": output_tput,
        "total_tput": total_tput,
        "elapsed_time": elapsed,
        "num_requests": num_req,
        "source": source,
    }


def scan_results(results_dir: Path) -> dict[int, dict[str, dict]]:
    """
    扫描结果目录, 按 output_len 分组.

    返回: {output_len: {quant: metrics_dict}}
    """
    grouped: dict[int, dict[str, dict]] = {}

    for f in sorted(results_dir.iterdir()):
        m = FILENAME_RE.match(f.name)
        if not m:
            continue

        quant = m.group(1)
        output_len = int(m.group(2))

        data = load_result(f)
        if data is None:
            continue

        if output_len not in grouped:
            grouped[output_len] = {}
        grouped[output_len][quant] = data

    return grouped


def compute_speedups(
    grouped: dict[int, dict[str, dict]],
    *,
    input_len: int,
) -> list[dict]:
    """
    计算每个 output_len 下的吞吐和加速比.

    返回按 output_len 排序的列表:
    [{output_len, bf16_tput, w8a8_tput, speedup, bf16_elapsed, w8a8_elapsed, ...}]
    """
    rows = []
    for output_len in sorted(grouped.keys()):
        quants = grouped[output_len]
        row = {"output_len": output_len}

        for quant, data in quants.items():
            metrics = extract_metrics(data, input_len=input_len, output_len=output_len)
            row[f"{quant}_output_tput"] = metrics["output_tput"]
            row[f"{quant}_total_tput"] = metrics["total_tput"]
            row[f"{quant}_elapsed"] = metrics["elapsed_time"]
            row[f"{quant}_source"] = metrics["source"]

        # 加速比 = w8a8 / bf16
        bf16_tput = row.get("bf16_output_tput")
        w8a8_tput = row.get("w8a8_output_tput")
        if bf16_tput and w8a8_tput and bf16_tput > 0:
            row["speedup"] = w8a8_tput / bf16_tput
        else:
            row["speedup"] = None

        # elapsed 加速比 = bf16_elapsed / w8a8_elapsed (反过来, elapsed 越短越好)
        bf16_elapsed = row.get("bf16_elapsed")
        w8a8_elapsed = row.get("w8a8_elapsed")
        if bf16_elapsed and w8a8_elapsed and w8a8_elapsed > 0:
            row["elapsed_speedup"] = bf16_elapsed / w8a8_elapsed
        else:
            row["elapsed_speedup"] = None

        rows.append(row)

    return rows


def format_speedup(val: float | None) -> str:
    """格式化加速比, 带方向标记."""
    if val is None:
        return "N/A"
    arrow = "▲" if val >= 1.0 else "▼"
    return f"{val:.3f}x {arrow}"


def format_tput(val: float | None) -> str:
    """格式化吞吐."""
    if val is None:
        return "N/A"
    return f"{val:.1f}"


def format_elapsed(val: float | None) -> str:
    """格式化耗时."""
    if val is None:
        return "N/A"
    return f"{val:.1f}"


def print_text_table(rows: list[dict]):
    """输出文本表格."""
    print("=" * 100)
    print("  Decode 长度 Sweep: BF16 vs W8A8D 吞吐对比 (input_len=1, 纯 decode)")
    print("=" * 100)
    print()

    header = f"{'output_len':>12}  {'BF16 (tok/s)':>14}  {'W8A8 (tok/s)':>14}  {'加速比':>12}  {'BF16 耗时(s)':>14}  {'W8A8 耗时(s)':>14}"
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['output_len']:>12}  "
            f"{format_tput(row.get('bf16_output_tput')):>14}  "
            f"{format_tput(row.get('w8a8_output_tput')):>14}  "
            f"{format_speedup(row.get('speedup')):>12}  "
            f"{format_elapsed(row.get('bf16_elapsed')):>14}  "
            f"{format_elapsed(row.get('w8a8_elapsed')):>14}"
        )

    print()

    # 趋势总结
    speedups = [r["speedup"] for r in rows if r["speedup"] is not None]
    if len(speedups) >= 2:
        trend = speedups[-1] - speedups[0]
        direction = "上升" if trend > 0 else "下降"
        print(f"趋势: 加速比从 {speedups[0]:.3f}x → {speedups[-1]:.3f}x ({direction} {abs(trend):.3f})")
        if all(s < 1.0 for s in speedups):
            print("结论: W8A8D 在所有 decode 长度下均有回退, 长尾 decode 放大了性能损失" if trend < 0
                  else "结论: W8A8D 在所有 decode 长度下均有回退, 但长序列下回退减小")
        elif all(s >= 1.0 for s in speedups):
            print("结论: W8A8D 在所有 decode 长度下均有加速")


def print_markdown_table(rows: list[dict]):
    """输出 Markdown 表格."""
    print("## Decode 长度 Sweep: BF16 vs W8A8D")
    print()
    print(f"- 模型: Qwen3-30B-A3B")
    print(f"- input_len: 1 (纯 decode)")
    print(f"- n: 8 (GRPO 多采样)")
    print(f"- ignore_eos: True (输出长度确定)")
    print()
    print("| output_len | BF16 (tok/s) | W8A8 (tok/s) | 加速比 | BF16 耗时(s) | W8A8 耗时(s) |")
    print("|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        speedup = row.get("speedup")
        speedup_str = f"{speedup:.3f}x" if speedup else "N/A"
        print(
            f"| {row['output_len']} "
            f"| {format_tput(row.get('bf16_output_tput'))} "
            f"| {format_tput(row.get('w8a8_output_tput'))} "
            f"| {speedup_str} "
            f"| {format_elapsed(row.get('bf16_elapsed'))} "
            f"| {format_elapsed(row.get('w8a8_elapsed'))} |"
        )


def print_csv(rows: list[dict]):
    """输出 CSV."""
    print("output_len,bf16_output_tput,w8a8_output_tput,speedup,bf16_elapsed,w8a8_elapsed")
    for row in rows:
        vals = [
            str(row["output_len"]),
            f"{row.get('bf16_output_tput', '')}" if row.get("bf16_output_tput") else "",
            f"{row.get('w8a8_output_tput', '')}" if row.get("w8a8_output_tput") else "",
            f"{row.get('speedup', '')}" if row.get("speedup") else "",
            f"{row.get('bf16_elapsed', '')}" if row.get("bf16_elapsed") else "",
            f"{row.get('w8a8_elapsed', '')}" if row.get("w8a8_elapsed") else "",
        ]
        print(",".join(vals))


def main():
    parser = argparse.ArgumentParser(
        description="Decode 长度 Sweep 分析: BF16 vs W8A8D 吞吐对比",
    )
    parser.add_argument("results_dir", help="结果目录 (含 {quant}_olen{N}.json)")
    parser.add_argument("--markdown", action="store_true", help="Markdown 格式输出")
    parser.add_argument("--csv", action="store_true", help="CSV 格式输出")
    parser.add_argument(
        "--input-len", type=int, default=1,
        help="输入长度, 用于推算 output throughput (默认: 1)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"错误: 目录不存在: {results_dir}", file=sys.stderr)
        sys.exit(1)

    grouped = scan_results(results_dir)
    if not grouped:
        print(f"错误: 未找到结果文件 ({results_dir}/*.json)", file=sys.stderr)
        sys.exit(1)

    rows = compute_speedups(grouped, input_len=args.input_len)

    if args.csv:
        print_csv(rows)
    elif args.markdown:
        print_markdown_table(rows)
    else:
        print_text_table(rows)


if __name__ == "__main__":
    main()
