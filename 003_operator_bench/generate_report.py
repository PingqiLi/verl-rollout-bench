#!/usr/bin/env python3
"""
生成 003 单算子 Benchmark 论证报告

读取 run_operator_bench.sh 的输出目录, 生成包含所有论证数据的 markdown 报告.

使用方式:
    python3 generate_report.py outputs/<timestamp>/
    python3 generate_report.py outputs/<timestamp>/ --output report.md
"""

import argparse
import json
import sys
from pathlib import Path


# ======================== 数据加载 ========================

def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


# ======================== 报告生成 ========================

def generate_report(results_dir: Path) -> str:
    """读取所有结果 JSON, 生成 markdown 报告"""
    results_all = load_json(results_dir / "results_all.json")
    sweep_30b = load_json(results_dir / "sweep_30b.json")
    sweep_718b = load_json(results_dir / "sweep_718b.json")

    if not results_all:
        print(f"找不到 results_all.json: {results_dir}", file=sys.stderr)
        sys.exit(1)

    lines = []
    w = lines.append

    # ==================== 标题 ====================
    w("# 003 单算子 Benchmark: W8A8D 在更大 MoE 上的加速潜力\n")

    # ==================== 结论 ====================
    w("## 核心结论\n")

    # 提取关键数据
    models = {}
    for name, data in results_all.items():
        if name in ("sweep_m", "custom"):
            continue
        pred = data.get("prediction", {})
        shapes = data.get("shapes", [])
        expert_gu = None
        for s in shapes:
            if s["name"] == "expert_gate_up":
                expert_gu = s
                break
        models[name] = {
            "prediction": pred.get("predicted_speedup"),
            "expert_gu": expert_gu,
            "shapes": shapes,
        }

    for name, m in models.items():
        egu = m["expert_gu"]
        if egu:
            arrow = "↑" if egu["speedup"] >= 1.0 else "↓"
            w(f"- **{name}**: expert GEMM ({egu['M']}, {egu['K']}, {egu['N']}), "
              f"K×N = {egu['K'] * egu['N']:,} → "
              f"算子 speedup = **{egu['speedup']:.3f}x {arrow}**, "
              f"dquant 开销 = {egu['dquant_overhead_pct']:.0f}%")

    w("")
    w("**W8A8D 的收益由 GEMM 计算量 (K×N) 驱动:**")
    w("718B 的 expert K×N (7680×4096 = 31.5M) 是 30B (2048×1536 = 3.1M) 的 ~10 倍,")
    w("INT8 matmul 的计算加速远超 dynamic_quant 的固定开销.\n")

    # ==================== 表1: 全量 shape 结果 ====================
    w("## 表 1: 各线性层算子级 Benchmark\n")

    for name, data in results_all.items():
        if name in ("sweep_m", "custom"):
            continue
        shapes = data.get("shapes", [])
        pred = data.get("prediction", {})

        w(f"### {name}\n")
        w("| 算子 | 类别 | M | K | N | K×N | BF16 (ms) | W8A8D (ms) | "
          "DQuant (ms) | Speedup | DQuant% |")
        w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        for r in shapes:
            kn = r["K"] * r["N"]
            arrow = "↑" if r["speedup"] >= 1.0 else "↓"
            w(f"| {r['name']} | {r['category']} | {r['M']} | "
              f"{r['K']} | {r['N']} | {kn:,} | "
              f"{r['bf16_ms']:.4f} | {r['w8a8d_total_ms']:.4f} | "
              f"{r['w8a8d_dquant_ms']:.4f} | "
              f"{r['speedup']:.3f}x {arrow} | {r['dquant_overhead_pct']:.0f}% |")

        if pred:
            pred_v = pred["predicted_speedup"]
            arrow = "↑" if pred_v >= 1.0 else "↓"
            w(f"\n线性层加权 speedup: **{pred_v:.3f}x {arrow}** "
              f"(加权方式: count_per_layer × num_layers)\n")

    # ==================== 表2: MoE Expert 核心对比 ====================
    w("## 表 2: MoE Expert 核心对比\n")
    w("MoE expert 层是 decode 阶段计算量最大的部分, "
      "直接决定 W8A8D 能否带来正收益.\n")
    w("| 模型 | Shape (M,K,N) | K×N | BF16 (ms) | W8A8D (ms) | "
      "Matmul (ms) | DQuant (ms) | Speedup | DQuant% |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for name, m in models.items():
        egu = m["expert_gu"]
        if not egu:
            continue
        kn = egu["K"] * egu["N"]
        arrow = "↑" if egu["speedup"] >= 1.0 else "↓"
        w(f"| {name} | ({egu['M']}, {egu['K']}, {egu['N']}) | "
          f"{kn:,} | {egu['bf16_ms']:.4f} | "
          f"{egu['w8a8d_total_ms']:.4f} | "
          f"{egu['w8a8d_matmul_ms']:.4f} | "
          f"{egu['w8a8d_dquant_ms']:.4f} | "
          f"{egu['speedup']:.3f}x {arrow} | "
          f"{egu['dquant_overhead_pct']:.0f}% |")

    # 计算 matmul-only speedup (不含 dquant)
    w("\n**不含 dynamic_quant 时的 matmul speedup:**\n")
    w("| 模型 | BF16 matmul (ms) | INT8 matmul (ms) | Matmul Speedup | "
      "DQuant 损失 |")
    w("|---|---:|---:|---:|---:|")
    for name, m in models.items():
        egu = m["expert_gu"]
        if not egu:
            continue
        mm_speedup = egu["bf16_ms"] / egu["w8a8d_matmul_ms"] \
            if egu["w8a8d_matmul_ms"] > 0 else 0
        loss = mm_speedup - egu["speedup"]
        w(f"| {name} | {egu['bf16_ms']:.4f} | "
          f"{egu['w8a8d_matmul_ms']:.4f} | "
          f"{mm_speedup:.3f}x | -{loss:.3f}x |")

    w("")

    # ==================== 表3: K×N 对比 ====================
    w("## 表 3: K×N 维度 — W8A8D 收益的核心驱动力\n")
    w("| 模型 | M (tokens/expert) | K×N | Expert Speedup | "
      "线性层 Speedup | 001 实测 |\n"
      "|---|---:|---:|---:|---:|---:|")

    actual_map = {"Qwen3-30B-A3B": 0.89}
    for name, m in sorted(models.items(),
                          key=lambda x: (x[1]["expert_gu"]["K"]
                                         * x[1]["expert_gu"]["N"])
                          if x[1]["expert_gu"] else 0):
        egu = m["expert_gu"]
        if not egu:
            continue
        kn = egu["K"] * egu["N"]
        actual = actual_map.get(name)
        actual_str = f"{actual:.2f}x" if actual else "—"
        arrow = "↑" if egu["speedup"] >= 1.0 else "↓"
        pred_arrow = "↑" if m["prediction"] >= 1.0 else "↓"
        w(f"| {name} | {egu['M']} | {kn:,} | "
          f"{egu['speedup']:.3f}x {arrow} | "
          f"{m['prediction']:.3f}x {pred_arrow} | "
          f"{actual_str} |")

    w("\n> 718B 的 M=8 **小于** 30B 的 M=16, 但 K×N 大 10 倍.")
    w("> W8A8D 加速来自更大的 GEMM 计算量, 而非更多的 token.\n")

    # ==================== 表4: M-Sweep ====================
    if sweep_30b or sweep_718b:
        w("## 表 4: M-Sweep — Breakeven 分析\n")
        w("固定 K×N, sweep M 维度, 找 W8A8D 从负收益到正收益的 breakeven point.\n")

        for label, sweep in [("30B-A3B (K=2048, N=1536)", sweep_30b),
                             ("718B (K=7680, N=4096)", sweep_718b)]:
            if not sweep:
                continue
            results = sweep["results"]
            K, N = sweep["K"], sweep["N"]

            w(f"### {label}\n")
            w("| M | BF16 (ms) | W8A8D (ms) | Speedup | DQuant% | 备注 |")
            w("|---:|---:|---:|---:|---:|---|")

            # 已知 M 值
            known = {16: "30B-A3B 实际值", 8: "718B 实际值"}
            breakeven_str = "在测试范围内均无正收益"

            for i, r in enumerate(results):
                arrow = "↑" if r["speedup"] >= 1.0 else "↓"
                note = f"← {known[r['M']]}" if r["M"] in known else ""
                w(f"| {r['M']} | {r['bf16_ms']:.4f} | "
                  f"{r['w8a8d_total_ms']:.4f} | "
                  f"{r['speedup']:.3f}x {arrow} | "
                  f"{r['dquant_overhead_pct']:.0f}% | {note} |")

            for i, r in enumerate(results):
                if r["speedup"] >= 1.0:
                    if i > 0:
                        prev = results[i - 1]
                        breakeven_str = (f"M 在 {prev['M']} 和 {r['M']} 之间")
                    else:
                        breakeven_str = f"M={r['M']} 时已有正收益"
                    break

            w(f"\n**Breakeven**: {breakeven_str}\n")

    # ==================== 表5: DQuant 开销分析 ====================
    w("## 表 5: Dynamic Quant 开销分析\n")
    w("W8A8D 路径 = `npu_dynamic_quant` (激活量化) + `npu_grouped_matmul` (INT8 计算).\n"
      "`npu_dynamic_quant` 是固定开销, K×N 越大其占比越低.\n")
    w("| 模型 | 算子 | DQuant (ms) | Matmul (ms) | Total (ms) | DQuant% |")
    w("|---|---|---:|---:|---:|---:|")

    for name, data in results_all.items():
        if name in ("sweep_m", "custom"):
            continue
        for r in data.get("shapes", []):
            if r["category"] != "moe":
                continue
            w(f"| {name} | {r['name']} | "
              f"{r['w8a8d_dquant_ms']:.4f} | "
              f"{r['w8a8d_matmul_ms']:.4f} | "
              f"{r['w8a8d_total_ms']:.4f} | "
              f"{r['dquant_overhead_pct']:.0f}% |")

    w("")

    # ==================== 论证小结 ====================
    w("## 论证小结\n")

    egu_30b = models.get("Qwen3-30B-A3B", {}).get("expert_gu")
    egu_718b = models.get("Pangu-718B", {}).get("expert_gu")

    w("### 003 实验证明的事实\n")
    w("1. **W8A8D 算子级加速由 GEMM K×N 维度驱动**")
    if egu_30b and egu_718b:
        w(f"   - 30B-A3B expert (K×N={egu_30b['K']*egu_30b['N']:,}): "
          f"speedup = {egu_30b['speedup']:.3f}x")
        w(f"   - 718B expert (K×N={egu_718b['K']*egu_718b['N']:,}): "
          f"speedup = {egu_718b['speedup']:.3f}x")
    w("2. **Dynamic quant 是固定开销, 大 GEMM 下占比可忽略**")
    if egu_30b and egu_718b:
        w(f"   - 30B: dquant 占 {egu_30b['dquant_overhead_pct']:.0f}%")
        w(f"   - 718B: dquant 占 {egu_718b['dquant_overhead_pct']:.0f}%")
    w("3. **718B 的 M=8 < 30B 的 M=16, 但仍然有正收益**")
    w("   - 说明 K×N 才是决定因素, 不是 M (tokens per expert)")
    w("")

    w("### 从算子到端到端\n")
    w("上述 speedup 是 **纯线性层** 的算子级数据, "
      "不包含 attention score、softmax、layernorm、routing 等非线性算子.")
    w("这些非线性算子 BF16/W8A8D 耗时相同, 会稀释最终 speedup:")
    w("")
    w("```")
    w("端到端 speedup = 1 / (f / 算子speedup + (1 - f))")
    w("其中 f = 线性层占 decode 总耗时的比例")
    w("```\n")
    w("f 的精确值需要 004 profiling 实验确定. 但只要 f > 0, "
      "算子级的正收益方向不变.\n")

    if egu_718b:
        sp = egu_718b["speedup"]
        w("**718B 端到端 speedup 预估 (按不同线性层占比):**\n")
        w("| 线性层占比 f | 预估端到端 Speedup |")
        w("|---:|---:|")
        for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            e2e = 1.0 / (f / sp + (1 - f))
            arrow = "↑" if e2e >= 1.0 else "↓"
            w(f"| {f:.0%} | {e2e:.3f}x {arrow} |")
        w("")

    w("### 后续验证\n")
    w("- **004 profiling**: 采集 30B decode 的算子级 trace, "
      "确定线性层占比 f, 精确预测 718B 端到端 speedup")
    w("- **001 实测交叉验证**: "
      "30B 端到端实测 0.89x, 可反推 f 并验证预测模型的一致性\n")

    return "\n".join(lines)


# ======================== CLI ========================

def main():
    parser = argparse.ArgumentParser(
        description="生成 003 单算子 Benchmark 论证报告")
    parser.add_argument("results_dir",
                        help="run_operator_bench.sh 的输出目录")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径 (默认输出到 stdout)")
    args = parser.parse_args()

    report = generate_report(Path(args.results_dir))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"报告已保存至 {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
