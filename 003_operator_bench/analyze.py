#!/usr/bin/env python3
"""
结果分析脚本: 读取 bench_ops.py 输出的 JSON, 生成对比表格

使用方式:
    python analyze.py results.json
    python analyze.py results.json --markdown
    python analyze.py results.json --validate 0.89   # 和实测值对比
"""

import argparse
import json
import sys


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_model_table(model_name: str, data: dict, fmt: str = "text"):
    """打印单模型结果表"""
    shapes = data.get("shapes", [])
    prediction = data.get("prediction", {})

    if fmt == "markdown":
        print(f"\n### {model_name}\n")
        print("| 算子 | 类别 | M | K | N | BF16 (ms) | W8A8D (ms) | "
              "DQuant (ms) | Speedup | DQuant% |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in shapes:
            arrow = "↑" if r["speedup"] >= 1.0 else "↓"
            print(f"| {r['name']} | {r['category']} | {r['M']} | "
                  f"{r['K']} | {r['N']} | {r['bf16_ms']:.4f} | "
                  f"{r['w8a8d_total_ms']:.4f} | {r['w8a8d_dquant_ms']:.4f} | "
                  f"{r['speedup']:.3f}x {arrow} | {r['dquant_overhead_pct']:.0f}% |")
    else:
        print(f"\n{'='*80}")
        print(f"  {model_name}")
        print(f"{'='*80}")
        print(f"  {'算子':<25s} {'类别':<10s} {'M':>4s} {'K':>6s} {'N':>6s}"
              f"  {'BF16':>8s}  {'W8A8D':>8s}  {'DQuant':>8s}"
              f"  {'Speedup':>8s}  {'DQ%':>5s}")
        print(f"  {'-'*95}")
        for r in shapes:
            arrow = "↑" if r["speedup"] >= 1.0 else "↓"
            print(f"  {r['name']:<25s} {r['category']:<10s}"
                  f" {r['M']:>4d} {r['K']:>6d} {r['N']:>6d}"
                  f"  {r['bf16_ms']:>8.4f}  {r['w8a8d_total_ms']:>8.4f}"
                  f"  {r['w8a8d_dquant_ms']:>8.4f}"
                  f"  {r['speedup']:>7.3f}x{arrow}"
                  f"  {r['dquant_overhead_pct']:>4.0f}%")

    if prediction:
        pred = prediction["predicted_speedup"]
        arrow = "↑" if pred >= 1.0 else "↓"
        print(f"\n  >>> 预测整模型 decode speedup: {pred:.3f}x {arrow}")


def print_sweep_table(data: dict, fmt: str = "text"):
    """打印 M-sweep 结果"""
    K, N = data["K"], data["N"]
    results = data["results"]

    if fmt == "markdown":
        print(f"\n### Sweep M (K={K}, N={N})\n")
        print("| M | BF16 (ms) | W8A8D (ms) | Speedup | DQuant% |")
        print("|---|---|---|---|---|")
        for r in results:
            arrow = "↑" if r["speedup"] >= 1.0 else "↓"
            print(f"| {r['M']} | {r['bf16_ms']:.4f} | "
                  f"{r['w8a8d_total_ms']:.4f} | "
                  f"{r['speedup']:.3f}x {arrow} | "
                  f"{r['dquant_overhead_pct']:.0f}% |")
    else:
        print(f"\n  Sweep M  (K={K}, N={N})")
        print(f"  {'M':>5s}  {'BF16(ms)':>10s}  {'W8A8D(ms)':>10s}  "
              f"{'Speedup':>8s}  {'DQuant%':>8s}")
        print(f"  {'-'*50}")
        for r in results:
            arrow = "↑" if r["speedup"] >= 1.0 else "↓"
            print(f"  {r['M']:>5d}  {r['bf16_ms']:>10.4f}"
                  f"  {r['w8a8d_total_ms']:>10.4f}"
                  f"  {r['speedup']:>7.3f}x{arrow}"
                  f"  {r['dquant_overhead_pct']:>7.1f}%")

    # 找 breakeven point
    for i, r in enumerate(results):
        if r["speedup"] >= 1.0:
            if i > 0:
                prev = results[i - 1]
                print(f"\n  >>> Breakeven point: M 在 {prev['M']} "
                      f"和 {r['M']} 之间")
            else:
                print(f"\n  >>> M={r['M']} 时已经有正收益")
            break
    else:
        print(f"\n  >>> 在测试范围内 W8A8D 均无正收益")


def validate(results: dict, actual_speedup: float):
    """对比预测值和实测值"""
    print(f"\n{'='*60}")
    print(f"  验证: 预测 vs 实测")
    print(f"{'='*60}")

    for model_name, data in results.items():
        if model_name in ("sweep_m", "custom"):
            continue
        prediction = data.get("prediction", {})
        if not prediction:
            continue
        pred = prediction["predicted_speedup"]
        diff = abs(pred - actual_speedup) / actual_speedup * 100
        status = "✓ 偏差 < 5%" if diff < 5 else (
            "△ 偏差 5-15%" if diff < 15 else "✗ 偏差 > 15%")
        print(f"  {model_name}:")
        print(f"    预测: {pred:.3f}x")
        print(f"    实测: {actual_speedup:.3f}x")
        print(f"    偏差: {diff:.1f}%  {status}")


def main():
    parser = argparse.ArgumentParser(description="分析单算子 benchmark 结果")
    parser.add_argument("input", help="bench_ops.py 输出的 JSON 文件")
    parser.add_argument("--markdown", action="store_true",
                        help="输出 markdown 格式")
    parser.add_argument("--validate", type=float, default=None,
                        help="和实测 speedup 对比 (如 0.89)")
    args = parser.parse_args()

    results = load_results(args.input)
    fmt = "markdown" if args.markdown else "text"

    for key, data in results.items():
        if key == "sweep_m":
            print_sweep_table(data, fmt)
        elif key == "custom":
            arrow = "↑" if data["speedup"] >= 1.0 else "↓"
            print(f"\n  Custom: M={data['M']} K={data['K']} N={data['N']}")
            print(f"  Speedup: {data['speedup']:.3f}x {arrow}")
        else:
            print_model_table(key, data, fmt)

    if args.validate is not None:
        validate(results, args.validate)


if __name__ == "__main__":
    main()
