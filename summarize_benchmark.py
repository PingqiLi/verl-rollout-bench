#!/usr/bin/env python3
"""
汇总 vLLM benchmark 结果, 生成 verl RL rollout 性能对比表格.

可独立运行:
    python3 summarize_benchmark.py ./benchmark_results
    python3 summarize_benchmark.py ./benchmark_results --csv --markdown

也可被 run_vllm_benchmark.sh 自动调用.

输出:
    summary.txt  - 终端友好的对比表格
    summary.csv  - 供 Excel / pandas 后续分析
    summary.md   - Markdown 格式 (--markdown)
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

# ============================================================
# 关键指标说明 (为什么选这些指标来反映 verl RL rollout 性能):
#
# verl rollout 的工作模式:
#   1. 一个 batch 的 prompt (如 256 条) 同时发给 vLLM
#   2. 每条 prompt 生成 n 个 completion (如 n=8)
#   3. 等待全部完成后, 才进入下一步 (actor update)
#   => rollout 耗时 = 最慢一条请求的完成时间
#
# 因此最核心的指标:
#   ★ Output Throughput (tok/s): 生成 token 的速率, 直接决定 rollout 吞吐
#   ★ P99 E2EL (ms): 第 99 百分位端到端延迟, 反映 "最慢请求" 的耗时
#     (verl 要等全部请求完成, 所以尾部延迟决定实际 rollout step 时间)
#   - Mean TTFT (ms): 首 token 时间, 反映 prefill 阶段效率
#   - Mean TPOT (ms): 每个输出 token 的平均时间, 反映 decode 阶段效率
#
# 对比维度:
#   - 同模型不同量化: 量化的加速比 & 精度代价
#   - 同量化不同模型: 模型规模的性能 scaling
# ============================================================

# ============================================================
# Offline (vllm bench throughput) 输出的 JSON key:
#   elapsed_time, num_requests, total_num_tokens,
#   requests_per_second, tokens_per_second
#
# Online (vllm bench serve) 输出的 JSON key:
#   output_throughput, total_token_throughput, request_throughput,
#   mean_ttft_ms, p99_ttft_ms, mean_tpot_ms, p99_tpot_ms, ...
#
# 为了兼容两种格式, 加载时统一 key 名 (见 normalize_result)
# ============================================================

def normalize_result(data: dict) -> dict:
    """统一 offline 和 online 两种 JSON 格式的 key 名."""
    d = dict(data)

    # offline → 统一命名
    if "tokens_per_second" in d and "total_token_throughput" not in d:
        d["total_token_throughput"] = d["tokens_per_second"]
    if "requests_per_second" in d and "request_throughput" not in d:
        d["request_throughput"] = d["requests_per_second"]

    # offline 没有 output_throughput, 用 total 近似
    # (offline 用 ignore_eos=True, output_len 固定, 所以 total ≈ input + output)
    if "output_throughput" not in d and "tokens_per_second" in d:
        # 尝试计算: output_tokens / elapsed_time
        elapsed = d.get("elapsed_time", 0)
        total_tokens = d.get("total_num_tokens", 0)
        num_requests = d.get("num_requests", 0)
        if elapsed > 0 and total_tokens > 0:
            d["output_throughput"] = d["tokens_per_second"]  # 近似值

    return d


# (json_key, display_name, format_str, higher_is_better)
METRICS = [
    ("total_token_throughput", "Total Tput (tok/s)",  "{:.1f}",  True),
    ("request_throughput",     "Req Tput (req/s)",    "{:.2f}",  True),
    ("elapsed_time",           "Elapsed (s)",         "{:.1f}",  False),
    ("num_requests",           "Requests",            "{}",      None),
    ("total_num_tokens",       "Total Tokens",        "{}",      None),
    ("mean_ttft_ms",           "TTFT Mean (ms)",      "{:.2f}",  False),
    ("mean_tpot_ms",           "TPOT Mean (ms)",      "{:.2f}",  False),
    ("mean_e2el_ms",           "E2EL Mean (ms)",      "{:.1f}",  False),
    ("p99_e2el_ms",            "E2EL P99 (ms)",       "{:.1f}",  False),
]

# 用于加速比计算的核心子集
SPEEDUP_METRICS = [
    ("total_token_throughput", "Total Tput", True),
    ("request_throughput",     "Req Tput",   True),
    ("elapsed_time",           "Elapsed",    False),
]


# ======================== 数据加载 ========================

def load_results(result_dir: str) -> list[dict]:
    """从 result_dir 下读取所有 benchmark JSON 文件."""
    results = []
    skip_files = {"summary.csv", "summary.txt", "summary.md"}

    result_path = Path(result_dir)
    if not result_path.is_dir():
        print(f"[ERROR] 结果目录不存在: {result_dir}", file=sys.stderr)
        sys.exit(1)

    for fpath in sorted(result_path.glob("*.json")):
        if fpath.name in skip_files:
            continue
        # 文件名格式: <model_key>_<quant>.json  e.g. qwen3-1.7b_bf16.json
        stem = fpath.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            print(f"  [WARN] 跳过无法解析的文件: {fpath.name}")
            continue

        model_key, quant = parts

        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [WARN] 读取失败: {fpath.name}: {e}")
            continue

        results.append({
            "model": model_key,
            "quant": quant.upper(),
            "file": fpath.name,
            "data": normalize_result(data),
        })

    return results


# ======================== 纯文本表格 ========================

def fmt_val(val, fmt_str: str) -> str:
    if val is None:
        return "N/A"
    try:
        return fmt_str.format(val)
    except (ValueError, TypeError):
        return str(val)


def build_main_table(results: list[dict]) -> list[str]:
    """表格 1: 所有实验的指标一览 (自动隐藏全 N/A 的列)."""
    # 过滤掉所有实验都没有的指标列
    all_keys = set()
    for r in results:
        all_keys.update(r["data"].keys())
    active_metrics = [(k, d, f, h) for k, d, f, h in METRICS if k in all_keys]

    col_w = 15
    model_w = 20
    quant_w = 8

    header = f"{'Model':<{model_w}} {'Quant':<{quant_w}}"
    for _, display, _, _ in active_metrics:
        header += f" {display:>{col_w}}"

    sep = "-" * len(header)
    lines = [
        "",
        "=" * 80,
        "  verl RL Rollout 性能对比表",
        "=" * 80,
        "",
        sep,
        header,
        sep,
    ]

    for r in results:
        d = r["data"]
        row = f"{r['model']:<{model_w}} {r['quant']:<{quant_w}}"
        for key, _, fmt_str, _ in active_metrics:
            row += f" {fmt_val(d.get(key), fmt_str):>{col_w}}"
        lines.append(row)

    lines.append(sep)
    return lines


def build_speedup_table(results: list[dict]) -> list[str]:
    """表格 2: 以 BF16 为基线的量化加速比."""
    # 按模型分组
    models: dict[str, dict[str, dict]] = OrderedDict()
    for r in results:
        models.setdefault(r["model"], OrderedDict())[r["quant"]] = r["data"]

    col_w = 16
    model_w = 20
    quant_w = 8

    header = f"{'Model':<{model_w}} {'Quant':<{quant_w}}"
    for _, display, _ in SPEEDUP_METRICS:
        header += f" {display:>{col_w}}"

    lines = [
        "",
        "=" * 80,
        "  量化加速比 (相对 BF16 基线)",
        "  >1.00x = 量化后更优  <1.00x = 量化后更差",
        "=" * 80,
        "",
        header,
        "-" * len(header),
    ]

    for model_name, quant_data in models.items():
        bf16_data = quant_data.get("BF16")
        if not bf16_data:
            lines.append(f"{model_name:<{model_w}} (无 BF16 基线, 跳过加速比计算)")
            lines.append("")
            continue

        for quant_name, qdata in quant_data.items():
            if quant_name == "BF16":
                row = f"{model_name:<{model_w}} {'BF16':<{quant_w}}"
                for _ in SPEEDUP_METRICS:
                    row += f" {'(baseline)':>{col_w}}"
                lines.append(row)
            else:
                row = f"{model_name:<{model_w}} {quant_name:<{quant_w}}"
                for key, _, higher_better in SPEEDUP_METRICS:
                    bf16_val = bf16_data.get(key)
                    q_val = qdata.get(key)
                    if bf16_val and q_val and bf16_val != 0:
                        if higher_better:
                            ratio = q_val / bf16_val
                        else:
                            ratio = bf16_val / q_val
                        arrow = "^" if ratio > 1.0 else "v"
                        cell = f"{ratio:.2f}x {arrow}"
                        row += f" {cell:>{col_w}}"
                    else:
                        row += f" {'N/A':>{col_w}}"
                lines.append(row)
        lines.append("")  # 模型间空行

    return lines


# ======================== CSV ========================

def write_csv(results: list[dict], out_path: str):
    header = ["model", "quant"]
    for key, _, _, _ in METRICS:
        header.append(key)

    with open(out_path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in results:
            d = r["data"]
            vals = [r["model"], r["quant"]]
            for key, _, _, _ in METRICS:
                v = d.get(key)
                vals.append(str(v) if v is not None else "")
            f.write(",".join(vals) + "\n")


# ======================== Markdown ========================

def build_markdown(results: list[dict]) -> list[str]:
    """生成 Markdown 格式的对比表格, 方便粘贴到文档/Issue."""
    lines = [
        "## verl RL Rollout 性能对比",
        "",
        "> 核心指标: **Total Tput** (越高越好) 和 **Elapsed** (越低越好)",
        "",
    ]

    # 根据数据中实际存在的 key 动态选择列
    # offline 有: total_token_throughput, request_throughput, elapsed_time
    # online 还有: mean_ttft_ms, mean_tpot_ms, mean_e2el_ms, p99_e2el_ms
    md_metrics = [
        ("total_token_throughput", "Total Tput<br>(tok/s)", "{:.1f}"),
        ("request_throughput",     "Req Tput<br>(req/s)",   "{:.2f}"),
        ("elapsed_time",           "Elapsed<br>(s)",        "{:.1f}"),
        ("mean_ttft_ms",           "TTFT Mean<br>(ms)",     "{:.2f}"),
        ("mean_tpot_ms",           "TPOT Mean<br>(ms)",     "{:.2f}"),
        ("mean_e2el_ms",           "E2EL Mean<br>(ms)",     "{:.1f}"),
        ("p99_e2el_ms",            "E2EL P99<br>(ms)",      "{:.1f}"),
    ]

    # 只保留至少有一个实验包含该 key 的列
    all_keys = set()
    for r in results:
        all_keys.update(r["data"].keys())
    md_metrics = [(k, d, f) for k, d, f in md_metrics if k in all_keys]

    header = "| Model | Quant |"
    align  = "|:------|:------|"
    for _, display, _ in md_metrics:
        header += f" {display} |"
        align  += " ---:|"

    lines.append(header)
    lines.append(align)

    for r in results:
        d = r["data"]
        row = f"| {r['model']} | {r['quant']} |"
        for key, _, fmt_str in md_metrics:
            row += f" {fmt_val(d.get(key), fmt_str)} |"
        lines.append(row)

    lines.append("")

    # 加速比表
    models: dict[str, dict[str, dict]] = OrderedDict()
    for r in results:
        models.setdefault(r["model"], OrderedDict())[r["quant"]] = r["data"]

    has_speedup = any(
        "BF16" in quant_data and len(quant_data) > 1
        for quant_data in models.values()
    )

    if has_speedup:
        lines.append("### 量化加速比 (vs BF16)")
        lines.append("")
        lines.append("> \\>1.00x = 量化后更优, <1.00x = 量化后更差")
        lines.append("")

        sp_header = "| Model | Quant |"
        sp_align  = "|:------|:------|"
        for _, display, _ in SPEEDUP_METRICS:
            sp_header += f" {display} |"
            sp_align  += " ---:|"

        lines.append(sp_header)
        lines.append(sp_align)

        for model_name, quant_data in models.items():
            bf16_data = quant_data.get("BF16")
            if not bf16_data:
                continue
            for quant_name, qdata in quant_data.items():
                if quant_name == "BF16":
                    row = f"| {model_name} | BF16 |"
                    for _ in SPEEDUP_METRICS:
                        row += " baseline |"
                    lines.append(row)
                else:
                    row = f"| {model_name} | {quant_name} |"
                    for key, _, higher_better in SPEEDUP_METRICS:
                        bf16_val = bf16_data.get(key)
                        q_val = qdata.get(key)
                        if bf16_val and q_val and bf16_val != 0:
                            ratio = (q_val / bf16_val) if higher_better else (bf16_val / q_val)
                            emoji = "+" if ratio > 1.0 else "-"
                            row += f" {ratio:.2f}x ({emoji}) |"
                        else:
                            row += " N/A |"
                    lines.append(row)

        lines.append("")

    return lines


# ======================== 主函数 ========================

def main():
    parser = argparse.ArgumentParser(
        description="汇总 vLLM benchmark 结果, 生成 verl RL rollout 性能对比表格",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 summarize_benchmark.py ./benchmark_results
  python3 summarize_benchmark.py ./benchmark_results --markdown
  python3 summarize_benchmark.py ./benchmark_results --csv --no-print
        """,
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        default="./benchmark_results",
        help="benchmark 结果 JSON 文件所在目录 (默认: ./benchmark_results)",
    )
    parser.add_argument(
        "--csv", action="store_true", default=True,
        help="输出 CSV 文件 (默认开启)",
    )
    parser.add_argument(
        "--markdown", action="store_true", default=False,
        help="同时输出 Markdown 格式的对比表格",
    )
    parser.add_argument(
        "--no-print", action="store_true", default=False,
        help="不输出到终端, 仅写入文件",
    )
    args = parser.parse_args()

    result_dir = args.result_dir

    # 加载数据
    results = load_results(result_dir)
    if not results:
        print("[WARN] 未找到有效的 benchmark 结果文件, 跳过汇总")
        sys.exit(0)

    print(f"  找到 {len(results)} 组实验结果")

    # 构建纯文本表格
    txt_lines = []
    txt_lines.extend(build_main_table(results))
    txt_lines.extend(build_speedup_table(results))
    txt_lines.append("")

    # 输出到终端
    if not args.no_print:
        for line in txt_lines:
            print(line)

    # 写入 summary.txt
    summary_txt = os.path.join(result_dir, "summary.txt")
    with open(summary_txt, "w") as f:
        for line in txt_lines:
            f.write(line + "\n")
    print(f"  汇总表格已保存: {summary_txt}")

    # 写入 CSV
    if args.csv:
        summary_csv = os.path.join(result_dir, "summary.csv")
        write_csv(results, summary_csv)
        print(f"  CSV 数据已保存:  {summary_csv}")

    # 写入 Markdown
    if args.markdown:
        summary_md = os.path.join(result_dir, "summary.md")
        md_lines = build_markdown(results)
        with open(summary_md, "w") as f:
            for line in md_lines:
                f.write(line + "\n")
        print(f"  Markdown 已保存: {summary_md}")


if __name__ == "__main__":
    main()
