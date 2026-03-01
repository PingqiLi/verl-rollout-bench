#!/usr/bin/env python3
"""
Profiling 结果分析: 从 torch_npu profiler trace 中提取算子耗时, 对比 BF16 vs W8A8D.

支持两种输入:
  1. 离线 profiling 目录 (torch_npu profiler trace)
  2. 在线 profiling 目录 (msserviceprofiler 输出)

用法:
    # 分析离线 profiling 结果
    python3 analyze_profile.py outputs/<timestamp>_graph/

    # 只看 MoE 相关算子
    python3 analyze_profile.py outputs/<timestamp>_graph/ --filter moe

    # 只看 top-20 算子
    python3 analyze_profile.py outputs/<timestamp>_graph/ --top 20

    # Markdown 输出
    python3 analyze_profile.py outputs/<timestamp>_graph/ --markdown
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# MoE 相关算子关键字
MOE_KEYWORDS = [
    "grouped_matmul", "npu_grouped_matmul",
    "npu_grouped_matmul_swiglu_quant",
    "npu_swiglu", "topk", "moe",
]

# 量化相关算子关键字
QUANT_KEYWORDS = [
    "npu_dynamic_quant", "npu_quant_matmul",
    "npu_dequant", "dequant", "quant",
]

# Attention 相关算子关键字
ATTN_KEYWORDS = [
    "flash_attention", "paged_attention", "incre_flash_attention",
    "reshape_and_cache", "rotary_embedding",
]


def find_trace_files(profile_dir: Path) -> list[Path]:
    """在 profiling 输出目录中查找 trace 文件."""
    traces = []
    for pattern in ["*.json", "**/*.json"]:
        for f in profile_dir.glob(pattern):
            if f.name.startswith("trace") or "chrome_tracing" in f.name:
                traces.append(f)
    # torch_npu profiler 输出的 tensorboard trace
    for f in profile_dir.glob("**/*.pt.trace.json"):
        traces.append(f)
    return sorted(set(traces))


def parse_torch_trace(trace_path: Path) -> list[dict]:
    """
    解析 torch_npu profiler 输出的 chrome trace JSON.

    提取 NPU kernel 事件, 返回:
    [{name, cat, dur, ts, args}, ...]
    """
    with open(trace_path) as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", [])

    npu_ops = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        # NPU kernel 事件通常有 cat="kernel" 或在 NPU stream 上
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        dur = ev.get("dur", 0)  # 微秒

        # 过滤: 只要有实际耗时的事件
        if dur <= 0:
            continue

        # 识别 NPU 算子 (启发式: kernel 类型 或 包含 npu/acl 关键字)
        is_npu_op = (
            cat in ("kernel", "gpu_memcpy", "Kernel", "NPU")
            or "npu" in name.lower()
            or "acl" in name.lower()
            or "aclnn" in name.lower()
            or cat == "cpu_op"  # CPU 侧的算子调度
        )

        if is_npu_op or cat in ("cpu_op", "Runtime", "Operator"):
            npu_ops.append({
                "name": name,
                "cat": cat,
                "dur_us": dur,
                "dur_ms": dur / 1000.0,
                "ts": ev.get("ts", 0),
                "args": ev.get("args", {}),
            })

    return npu_ops


def aggregate_ops(ops: list[dict]) -> dict[str, dict]:
    """
    按算子名聚合: 计算总耗时, 调用次数, 平均耗时.

    返回: {op_name: {total_us, count, avg_us, min_us, max_us}}
    """
    agg: dict[str, dict] = defaultdict(lambda: {
        "total_us": 0, "count": 0, "min_us": float("inf"), "max_us": 0,
    })

    for op in ops:
        name = op["name"]
        dur = op["dur_us"]
        entry = agg[name]
        entry["total_us"] += dur
        entry["count"] += 1
        entry["min_us"] = min(entry["min_us"], dur)
        entry["max_us"] = max(entry["max_us"], dur)

    for name, entry in agg.items():
        entry["avg_us"] = entry["total_us"] / entry["count"] if entry["count"] > 0 else 0

    return dict(agg)


def classify_op(name: str) -> str:
    """将算子分类: MoE / Quant / Attention / Other."""
    name_lower = name.lower()
    for kw in MOE_KEYWORDS:
        if kw in name_lower:
            return "MoE"
    for kw in QUANT_KEYWORDS:
        if kw in name_lower:
            return "Quant"
    for kw in ATTN_KEYWORDS:
        if kw in name_lower:
            return "Attention"
    return "Other"


def filter_ops(agg: dict[str, dict], *, keyword: str = "") -> dict[str, dict]:
    """按关键字过滤算子."""
    if not keyword:
        return agg
    kw_lower = keyword.lower()
    return {k: v for k, v in agg.items() if kw_lower in k.lower()}


def analyze_quant_pair(
    run_dir: Path,
) -> tuple[dict[str, dict] | None, dict[str, dict] | None]:
    """
    分析一次运行的 BF16 和 W8A8 profiling 数据.

    返回: (bf16_agg, w8a8_agg) — 聚合后的算子统计
    """
    results = {}
    for quant in ["bf16", "w8a8"]:
        prof_dir = run_dir / f"profile_{quant}"
        if not prof_dir.is_dir():
            continue

        traces = find_trace_files(prof_dir)
        if not traces:
            print(f"警告: {prof_dir} 中未找到 trace 文件", file=sys.stderr)
            continue

        # 用最大的 trace 文件 (通常是最完整的)
        trace_file = max(traces, key=lambda f: f.stat().st_size)
        print(f"解析: {trace_file} ({trace_file.stat().st_size / 1024 / 1024:.1f} MB)",
              file=sys.stderr)

        ops = parse_torch_trace(trace_file)
        if not ops:
            print(f"警告: {trace_file} 中未找到 NPU 算子事件", file=sys.stderr)
            continue

        agg = aggregate_ops(ops)
        results[quant] = agg
        print(f"  {quant}: {len(ops)} 个事件, {len(agg)} 种算子", file=sys.stderr)

    return results.get("bf16"), results.get("w8a8")


def format_us(val: float) -> str:
    """格式化微秒."""
    if val >= 1_000_000:
        return f"{val / 1_000_000:.2f}s"
    if val >= 1000:
        return f"{val / 1000:.2f}ms"
    return f"{val:.1f}μs"


def print_single_table(agg: dict[str, dict], *, quant: str, top: int = 30):
    """打印单个 quant 的算子统计."""
    print(f"\n{'=' * 90}")
    print(f"  算子统计: {quant.upper()}")
    print(f"{'=' * 90}")

    sorted_ops = sorted(agg.items(), key=lambda x: x[1]["total_us"], reverse=True)
    total_time = sum(v["total_us"] for v in agg.values())

    header = f"{'算子名':50s}  {'分类':8s}  {'总耗时':>12s}  {'次数':>6s}  {'平均':>10s}  {'占比':>6s}"
    print(header)
    print("-" * len(header))

    for i, (name, stats) in enumerate(sorted_ops[:top]):
        cat = classify_op(name)
        pct = stats["total_us"] / total_time * 100 if total_time > 0 else 0
        print(
            f"{name[:50]:50s}  {cat:8s}  "
            f"{format_us(stats['total_us']):>12s}  "
            f"{stats['count']:>6d}  "
            f"{format_us(stats['avg_us']):>10s}  "
            f"{pct:>5.1f}%"
        )

    if len(sorted_ops) > top:
        print(f"  ... 省略 {len(sorted_ops) - top} 个算子")

    print(f"\n总计: {len(agg)} 种算子, 总耗时 {format_us(total_time)}")

    # 按分类汇总
    cat_totals: dict[str, float] = defaultdict(float)
    for name, stats in agg.items():
        cat_totals[classify_op(name)] += stats["total_us"]

    print("\n分类汇总:")
    for cat, total in sorted(cat_totals.items(), key=lambda x: x[1], reverse=True):
        pct = total / total_time * 100 if total_time > 0 else 0
        print(f"  {cat:12s}  {format_us(total):>12s}  ({pct:.1f}%)")


def print_comparison_table(
    bf16_agg: dict[str, dict],
    w8a8_agg: dict[str, dict],
    *,
    top: int = 30,
    keyword: str = "",
    markdown: bool = False,
):
    """打印 BF16 vs W8A8 对比表."""
    # 取并集
    all_ops = set(bf16_agg.keys()) | set(w8a8_agg.keys())

    # 按 BF16 总耗时排序
    rows = []
    for name in all_ops:
        bf16 = bf16_agg.get(name, {})
        w8a8 = w8a8_agg.get(name, {})
        bf16_total = bf16.get("total_us", 0)
        w8a8_total = w8a8.get("total_us", 0)

        # 加速比
        speedup = None
        if bf16_total > 0 and w8a8_total > 0:
            speedup = bf16_total / w8a8_total  # >1 = W8A8 更快

        rows.append({
            "name": name,
            "cat": classify_op(name),
            "bf16_total": bf16_total,
            "w8a8_total": w8a8_total,
            "bf16_count": bf16.get("count", 0),
            "w8a8_count": w8a8.get("count", 0),
            "speedup": speedup,
        })

    if keyword:
        rows = [r for r in rows if keyword.lower() in r["name"].lower()]

    rows.sort(key=lambda r: r["bf16_total"] + r["w8a8_total"], reverse=True)
    rows = rows[:top]

    if markdown:
        print("## BF16 vs W8A8D 算子耗时对比\n")
        print("| 算子 | 分类 | BF16 耗时 | W8A8 耗时 | 加速比 |")
        print("|:---|:---|---:|---:|---:|")
        for r in rows:
            sp = f"{r['speedup']:.2f}x" if r["speedup"] else "N/A"
            print(
                f"| {r['name'][:60]} "
                f"| {r['cat']} "
                f"| {format_us(r['bf16_total'])} "
                f"| {format_us(r['w8a8_total'])} "
                f"| {sp} |"
            )
    else:
        print(f"\n{'=' * 110}")
        print(f"  BF16 vs W8A8D 算子耗时对比")
        print(f"{'=' * 110}")

        header = (
            f"{'算子名':45s}  {'分类':8s}  "
            f"{'BF16 耗时':>12s}  {'W8A8 耗时':>12s}  "
            f"{'加速比':>8s}"
        )
        print(header)
        print("-" * len(header))

        for r in rows:
            sp = f"{r['speedup']:.2f}x" if r["speedup"] else "N/A"
            print(
                f"{r['name'][:45]:45s}  {r['cat']:8s}  "
                f"{format_us(r['bf16_total']):>12s}  "
                f"{format_us(r['w8a8_total']):>12s}  "
                f"{sp:>8s}"
            )

    # 分类汇总对比
    cat_bf16: dict[str, float] = defaultdict(float)
    cat_w8a8: dict[str, float] = defaultdict(float)
    for r in rows:
        cat_bf16[r["cat"]] += r["bf16_total"]
        cat_w8a8[r["cat"]] += r["w8a8_total"]

    all_cats = set(cat_bf16.keys()) | set(cat_w8a8.keys())
    if markdown:
        print("\n### 分类汇总\n")
        print("| 分类 | BF16 | W8A8 | 加速比 |")
        print("|:---|---:|---:|---:|")
    else:
        print(f"\n分类汇总:")

    for cat in sorted(all_cats, key=lambda c: cat_bf16.get(c, 0), reverse=True):
        b = cat_bf16.get(cat, 0)
        w = cat_w8a8.get(cat, 0)
        sp = f"{b / w:.2f}x" if w > 0 else "N/A"
        if markdown:
            print(f"| {cat} | {format_us(b)} | {format_us(w)} | {sp} |")
        else:
            print(f"  {cat:12s}  BF16={format_us(b):>12s}  W8A8={format_us(w):>12s}  {sp}")


def main():
    parser = argparse.ArgumentParser(
        description="Profiling 分析: 从 trace 中提取算子耗时, 对比 BF16 vs W8A8D",
    )
    parser.add_argument("run_dir", help="profiling 输出目录 (含 profile_bf16/ 和 profile_w8a8/)")
    parser.add_argument("--filter", default="", help="按算子名过滤 (如 'moe', 'matmul', 'quant')")
    parser.add_argument("--top", type=int, default=30, help="显示 top-N 算子 (默认: 30)")
    parser.add_argument("--markdown", action="store_true", help="Markdown 格式输出")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"错误: 目录不存在: {run_dir}", file=sys.stderr)
        sys.exit(1)

    bf16_agg, w8a8_agg = analyze_quant_pair(run_dir)

    if bf16_agg is None and w8a8_agg is None:
        print("错误: 未找到可解析的 profiling 数据", file=sys.stderr)
        print("确认目录下有 profile_bf16/ 或 profile_w8a8/ 子目录, 且含 trace JSON", file=sys.stderr)
        sys.exit(1)

    # 应用过滤
    if args.filter:
        if bf16_agg:
            bf16_agg = filter_ops(bf16_agg, keyword=args.filter)
        if w8a8_agg:
            w8a8_agg = filter_ops(w8a8_agg, keyword=args.filter)

    # 输出
    if bf16_agg and w8a8_agg:
        print_comparison_table(
            bf16_agg, w8a8_agg,
            top=args.top,
            keyword=args.filter,
            markdown=args.markdown,
        )
    elif bf16_agg:
        print_single_table(bf16_agg, quant="bf16", top=args.top)
    elif w8a8_agg:
        print_single_table(w8a8_agg, quant="w8a8", top=args.top)


if __name__ == "__main__":
    main()
