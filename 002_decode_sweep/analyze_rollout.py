#!/usr/bin/env python3
"""
分析 rollout benchmark 结果: 对比 BF16 vs W8A8D

使用方式:
    python3 analyze_rollout.py results/
    python3 analyze_rollout.py results/ --markdown
    python3 analyze_rollout.py bf16.json w8a8.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_results(paths: list[str]) -> dict[str, dict]:
    """加载结果, 返回 {quant_name: result_dict}"""
    results = {}
    for path_str in paths:
        p = Path(path_str)
        if p.is_dir():
            for f in sorted(p.glob("*.json")):
                with open(f) as fh:
                    results[f.stem] = json.load(fh)
        elif p.is_file() and p.suffix == ".json":
            with open(p) as fh:
                results[p.stem] = json.load(fh)
    return results


def print_comparison(results: dict[str, dict], *, fmt: str = "text"):
    """打印 BF16 vs W8A8D 对比"""
    if not results:
        print("无结果", file=sys.stderr)
        return

    # 打印负载信息 (取第一个结果的配置)
    first = next(iter(results.values()))
    cfg = first.get("config", {})
    output_lens = cfg.get("output_lens", [])

    if output_lens:
        ols = np.array(output_lens)
        if fmt == "markdown":
            print(f"\n## 负载配置\n")
            print(f"- input_len: {cfg.get('input_len', '?')}")
            print(f"- num_prompts: {cfg.get('num_prompts', '?')}")
            print(f"- num_samples: {cfg.get('num_samples', '?')} "
                  f"(总序列: {cfg.get('num_prompts', 0) * cfg.get('num_samples', 0)})")
            print(f"- output_len 分布: `{cfg.get('dist', '?')}` "
                  f"[{cfg.get('min_tokens', '?')}, {cfg.get('max_tokens', '?')}]")
            print(f"- output_len 统计: 均值={ols.mean():.0f}, "
                  f"中位={np.median(ols):.0f}, "
                  f"P90={np.percentile(ols, 90):.0f}")
        else:
            print(f"\n  负载配置:")
            print(f"    input_len: {cfg.get('input_len', '?')}")
            print(f"    num_prompts: {cfg.get('num_prompts', '?')}, "
                  f"num_samples: {cfg.get('num_samples', '?')}")
            print(f"    output_len 分布: {cfg.get('dist', '?')} "
                  f"[{cfg.get('min_tokens', '?')}, {cfg.get('max_tokens', '?')}]")
            print(f"    均值={ols.mean():.0f}, 中位={np.median(ols):.0f}, "
                  f"P90={np.percentile(ols, 90):.0f}")

    # 确定 baseline
    baseline_key = "bf16" if "bf16" in results else None

    # 对比表
    if fmt == "markdown":
        print(f"\n## 吞吐对比\n")
        print("| 精度 | Output 吞吐 (tok/s) | 耗时 (s) | 总输出 tokens | 加速比 |")
        print("|---|---:|---:|---:|---:|")
        for name, r in results.items():
            tput = r["output_tput_tok_s"]
            elapsed = r["elapsed_s"]
            total = r["total_output_tokens"]
            if baseline_key and name != baseline_key:
                speedup = tput / results[baseline_key]["output_tput_tok_s"]
                arrow = "↑" if speedup >= 1.0 else "↓"
                print(f"| {name} | {tput:.1f} | {elapsed:.1f} | "
                      f"{total:,} | {speedup:.3f}x {arrow} |")
            else:
                print(f"| {name} | {tput:.1f} | {elapsed:.1f} | "
                      f"{total:,} | baseline |")
    else:
        print(f"\n  {'精度':<10s}  {'Output(tok/s)':>14s}  {'耗时(s)':>8s}  "
              f"{'总输出':>12s}  {'加速比':>10s}")
        print(f"  {'-'*60}")
        for name, r in results.items():
            tput = r["output_tput_tok_s"]
            elapsed = r["elapsed_s"]
            total = r["total_output_tokens"]
            if baseline_key and name != baseline_key:
                speedup = tput / results[baseline_key]["output_tput_tok_s"]
                arrow = "↑" if speedup >= 1.0 else "↓"
                print(f"  {name:<10s}  {tput:>14.1f}  {elapsed:>8.1f}  "
                      f"{total:>12,}  {speedup:>8.3f}x {arrow}")
            else:
                print(f"  {name:<10s}  {tput:>14.1f}  {elapsed:>8.1f}  "
                      f"{total:>12,}  {'baseline':>10s}")


def main():
    parser = argparse.ArgumentParser(
        description="分析 rollout benchmark 结果")
    parser.add_argument("inputs", nargs="+",
                        help="结果 JSON 文件或目录")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    results = load_results(args.inputs)
    fmt = "markdown" if args.markdown else "text"
    print_comparison(results, fmt=fmt)


if __name__ == "__main__":
    main()
