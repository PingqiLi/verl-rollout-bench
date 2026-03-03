#!/usr/bin/env python3
"""
Benchmark 散点图: 多模型 × 多精度吞吐对比可视化.

从 benchmark JSON 结果文件中读取 output throughput, 绘制散点图.
支持多次重复实验, 每次实验显示为独立散点, 均值以横线标注.

用法:
    python3 plot_benchmark.py results/
    python3 plot_benchmark.py results/ -o benchmark.pdf
    python3 plot_benchmark.py results/ -o benchmark.png --table
    python3 plot_benchmark.py results/ --model-order qwen3-1.7b,pangu-7b
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 默认配置
# ============================================================

# 模型显示顺序
DEFAULT_MODEL_ORDER = [
    "qwen3-1.7b",
    "qwen3-30b-a3b",
    "pangu-7b",
    "qwen3-32b",
]

# 模型显示名
DISPLAY_NAMES = {
    "qwen3-1.7b": "Qwen3-1.7B",
    "qwen3-30b-a3b": "Qwen3-30B-A3B",
    "pangu-7b": "Pangu-7B",
    "qwen3-32b": "Qwen3-32B",
}

# 精度配色 (tab10 风格, 对比鲜明)
QUANT_STYLE = {
    "BF16": {"color": "#4878D0", "marker": "o", "label": "BF16"},
    "W8A8": {"color": "#EE854A", "marker": "D", "label": "W8A8D"},
    "W8A16": {"color": "#6ACC64", "marker": "^", "label": "W8A16"},
}

# 图表尺寸参数
FIGURE_SIZE = (9, 5.5)
MARKER_SIZE = 70
JITTER_HALF = 0.04          # 同组散点的水平抖动半宽
GROUP_OFFSET = 0.18         # BF16 / W8A8D 两组之间的中心偏移
MEAN_BAR_HALF = 0.08        # 均值横线半宽


# ============================================================
# 数据加载
# ============================================================

def parse_result_filename(stem: str) -> tuple[str, str, int]:
    """
    解析结果文件名, 返回 (model_key, quant, run_idx).

    支持格式:
        qwen3-1.7b_bf16.json       → ("qwen3-1.7b", "BF16", 1)
        qwen3-1.7b_bf16_run2.json  → ("qwen3-1.7b", "BF16", 2)
    """
    m = re.match(r'^(.+)_run(\d+)$', stem)
    if m:
        base, run_idx = m.group(1), int(m.group(2))
    else:
        base, run_idx = stem, 1

    parts = base.rsplit("_", 1)
    if len(parts) != 2:
        return "", "", 0
    return parts[0], parts[1].upper(), run_idx


def infer_output_throughput(data: dict) -> float | None:
    """从 benchmark JSON 推断 output throughput (tok/s)."""
    # 直接字段
    v = data.get("output_throughput")
    if v is not None and float(v) > 0:
        return float(v)

    # offline: total_output_tokens / elapsed_time
    elapsed = data.get("elapsed_time")
    total_out = data.get("total_output_tokens")
    if elapsed and total_out and float(elapsed) > 0:
        return float(total_out) / float(elapsed)

    # offline: tokens_per_second * output/(input+output)
    tps = data.get("tokens_per_second")
    if tps and float(tps) > 0:
        il = float(data.get("input_len", 512))
        ol = float(data.get("output_len", 256))
        total = il + ol
        if total > 0:
            return float(tps) * ol / total

    return None


def load_results(result_dir: Path) -> list[dict]:
    """加载目录下所有 benchmark JSON 结果."""
    results = []
    skip_prefixes = ("summary",)

    for fpath in sorted(result_dir.glob("*.json")):
        if any(fpath.name.startswith(p) for p in skip_prefixes):
            continue

        model, quant, run = parse_result_filename(fpath.stem)
        if not model or not quant:
            print(f"  跳过: {fpath.name} (无法解析文件名)", file=sys.stderr)
            continue

        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  跳过: {fpath.name} ({e})", file=sys.stderr)
            continue

        tput = infer_output_throughput(data)
        if tput is None:
            print(f"  跳过: {fpath.name} (无法推断 throughput)", file=sys.stderr)
            continue

        results.append({
            "model": model,
            "quant": quant,
            "run": run,
            "throughput": tput,
        })

    return results


# ============================================================
# 散点图
# ============================================================

def plot_throughput(
    results: list[dict],
    *,
    output: str = "benchmark_scatter.pdf",
    model_order: list[str] | None = None,
    title: str = "",
):
    """绘制多模型 × 多精度吞吐散点图."""
    if model_order is None:
        model_order = DEFAULT_MODEL_ORDER

    # 只保留有数据的模型, 维持指定顺序
    available = {r["model"] for r in results}
    models = [m for m in model_order if m in available]
    # 未在 order 中的模型追加到末尾
    for r in results:
        if r["model"] not in models:
            models.append(r["model"])

    # 精度列表 (按出现顺序, BF16 优先)
    quants = []
    for r in results:
        if r["quant"] not in quants:
            quants.append(r["quant"])
    if "BF16" in quants:
        quants.remove("BF16")
        quants.insert(0, "BF16")

    n_quants = len(quants)
    offsets = np.linspace(-GROUP_OFFSET, GROUP_OFFSET, n_quants)

    # ---- 样式 ----
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(axis="y", linestyle="--", alpha=0.25, color="#888888", zorder=0)

    # ---- 绘制 ----
    speedup_annotations = []  # (x, y, text, color) 汇总加速比标注

    for m_idx, model in enumerate(models):
        model_means = {}  # quant → mean

        for q_idx, quant in enumerate(quants):
            style = QUANT_STYLE.get(quant, {
                "color": "#999999", "marker": "o", "label": quant,
            })
            runs = [r["throughput"] for r in results
                    if r["model"] == model and r["quant"] == quant]
            if not runs:
                continue

            x_center = m_idx + offsets[q_idx]

            # 散点: 少量点用均匀分布, 多点用随机抖动
            n = len(runs)
            if n <= 5:
                jitter = np.linspace(-JITTER_HALF, JITTER_HALF, n)
            else:
                rng = np.random.RandomState(42 + m_idx * 10 + q_idx)
                jitter = rng.uniform(-JITTER_HALF, JITTER_HALF, n)
            x_pts = x_center + jitter

            ax.scatter(
                x_pts, runs,
                c=style["color"],
                marker=style["marker"],
                s=MARKER_SIZE,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.8,
                label=style["label"] if m_idx == 0 else None,
                zorder=4,
            )

            # 均值横线
            mean_val = float(np.mean(runs))
            model_means[quant] = mean_val
            ax.hlines(
                mean_val,
                x_center - MEAN_BAR_HALF,
                x_center + MEAN_BAR_HALF,
                colors=style["color"],
                linewidths=2.5,
                alpha=0.9,
                zorder=5,
            )

        # 加速比标注 (W8A8D vs BF16)
        if "BF16" in model_means:
            bf16_mean = model_means["BF16"]
            for quant, q_mean in model_means.items():
                if quant == "BF16" or bf16_mean == 0:
                    continue
                speedup = q_mean / bf16_mean
                all_vals = [r["throughput"] for r in results if r["model"] == model]
                y_top = max(all_vals) if all_vals else q_mean
                color = "#2E8B57" if speedup >= 1.0 else "#CD3333"
                speedup_annotations.append((m_idx, y_top, f"{speedup:.2f}×", color))

    # 绘制加速比标注
    for x, y, text, color in speedup_annotations:
        ax.annotate(
            text, xy=(x, y),
            xytext=(0, 10), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=color,
        )

    # ---- 坐标轴 ----
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(
        [DISPLAY_NAMES.get(m, m) for m in models],
        fontsize=11, fontweight="medium",
    )
    ax.set_ylabel("Output Throughput (tok/s)", fontsize=12, labelpad=8)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(bottom=0)

    # y 轴上边界留出标注空间
    y_max = ax.get_ylim()[1]
    ax.set_ylim(top=y_max * 1.08)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    # ---- 图例 ----
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels,
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            edgecolor="#CCCCCC",
            fontsize=10,
            markerscale=1.2,
        )

    # ---- 保存 ----
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"散点图已保存: {output}", file=sys.stderr)


# ============================================================
# 加速比表格
# ============================================================

def build_speedup_table(
    results: list[dict],
    *,
    model_order: list[str] | None = None,
    baseline: str = "BF16",
    markdown: bool = True,  # noqa: ARG001
) -> list[str]:
    """生成加速比汇总表格."""
    if model_order is None:
        model_order = DEFAULT_MODEL_ORDER

    available = {r["model"] for r in results}
    models = [m for m in model_order if m in available]
    quants = sorted({r["quant"] for r in results} - {baseline})

    if not quants:
        return ["(无非基线精度数据)"]

    # 表头
    header = f"| 模型 | {baseline} (tok/s) |"
    align = "|:---|---:|"
    for q in quants:
        header += f" {QUANT_STYLE.get(q, {}).get('label', q)} (tok/s) | 加速比 |"
        align += " ---:| ---:|"

    lines = [header, align]

    for model in models:
        display = DISPLAY_NAMES.get(model, model)

        # 基线
        base_runs = [r["throughput"] for r in results
                     if r["model"] == model and r["quant"] == baseline]
        if not base_runs:
            continue

        base_mean = float(np.mean(base_runs))
        base_std = float(np.std(base_runs))

        if len(base_runs) > 1:
            row = f"| {display} | {base_mean:.0f} ± {base_std:.0f} |"
        else:
            row = f"| {display} | {base_mean:.0f} |"

        for q in quants:
            q_runs = [r["throughput"] for r in results
                      if r["model"] == model and r["quant"] == q]
            if not q_runs:
                row += " — | — |"
                continue

            q_mean = float(np.mean(q_runs))
            q_std = float(np.std(q_runs))
            speedup = q_mean / base_mean if base_mean > 0 else 0

            if len(q_runs) > 1:
                row += f" {q_mean:.0f} ± {q_std:.0f} | {speedup:.2f}× |"
            else:
                row += f" {q_mean:.0f} | {speedup:.2f}× |"

        lines.append(row)

    return lines


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 散点图: 多模型 × 多精度吞吐对比可视化",
    )
    parser.add_argument(
        "result_dir",
        help="benchmark 结果 JSON 目录",
    )
    parser.add_argument(
        "-o", "--output",
        default="benchmark_scatter.pdf",
        help="散点图输出路径 (默认: benchmark_scatter.pdf)",
    )
    parser.add_argument(
        "--model-order",
        default=None,
        help="模型显示顺序, 逗号分隔",
    )
    parser.add_argument(
        "--title",
        default="vLLM Offline Throughput (Ascend NPU)",
        help="图表标题 (空字符串则不显示)",
    )
    parser.add_argument(
        "--table", action="store_true",
        help="同时输出加速比表格 (Markdown)",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.is_dir():
        print(f"错误: 目录不存在: {result_dir}", file=sys.stderr)
        sys.exit(1)

    results = load_results(result_dir)
    if not results:
        print("错误: 未找到有效的 benchmark 结果", file=sys.stderr)
        sys.exit(1)

    n_models = len({r["model"] for r in results})
    n_quants = len({r["quant"] for r in results})
    print(f"加载 {len(results)} 组结果 ({n_models} 模型, {n_quants} 精度)",
          file=sys.stderr)

    model_order = None
    if args.model_order:
        model_order = [m.strip() for m in args.model_order.split(",")]

    # 散点图
    plot_throughput(
        results,
        output=args.output,
        model_order=model_order,
        title=args.title,
    )

    # 加速比表格
    if args.table:
        table = build_speedup_table(results, model_order=model_order)
        print()
        for line in table:
            print(line)


if __name__ == "__main__":
    main()
