#!/usr/bin/env python3
"""
单算子 Benchmark: BF16 vs W8A8D (grouped_matmul / quant_matmul)

在 Ascend NPU 上对比 decode 阶段线性层的算子级性能:
  - BF16 路径: grouped_matmul(BF16) + swiglu + grouped_matmul(BF16)
  - W8A8D 路径: dynamic_quant + grouped_matmul(INT8) + swiglu + grouped_matmul(INT8)

使用方式:
    # 测试所有模型所有 shape
    python bench_ops.py --all

    # 只测 MoE expert 层, 指定模型
    python bench_ops.py --model Qwen3-30B-A3B --category moe

    # 自定义 shape
    python bench_ops.py --custom --M 16 --K 7680 --N 4096

    # sweep M 维度 (关键实验: 找 breakeven point)
    python bench_ops.py --sweep-m --K 2048 --N 1536   # 30B shape
    python bench_ops.py --sweep-m --K 7680 --N 4096   # 718B shape

    # 输出 JSON
    python bench_ops.py --all --output results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from shapes import MODELS, get_all_shapes


# ============================================================
# 算子 Benchmark 核心函数
# ============================================================

def _sync():
    """等待 NPU 计算完成"""
    torch.npu.synchronize()


def bench_grouped_matmul_bf16(M: int, K: int, N: int,
                              num_experts: int = 8,
                              warmup: int = 20,
                              repeats: int = 100) -> float:
    """
    BF16 MoE 路径 (单次 grouped_matmul)

    模拟 unquant_apply_mlp 中的一次 npu_grouped_matmul:
      x: [total_tokens, K] (BF16)
      weight: [num_experts, K, N] (BF16, 实际按 expert 分组)

    Args:
        M: 每个 expert 的 token 数
        K: 输入维度
        N: 输出维度
        num_experts: 活跃 expert 数

    Returns:
        平均耗时 (ms)
    """
    total_tokens = M * num_experts
    x = torch.randn(total_tokens, K, dtype=torch.bfloat16, device="npu")
    w = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device="npu")

    # group_list: 每个 expert 分到的 token 数 (均匀分布)
    group_list = torch.full((num_experts,), M,
                            dtype=torch.int64, device="npu")

    # warmup
    for _ in range(warmup):
        _ = torch_npu.npu_grouped_matmul(
            x=[x], weight=[w],
            split_item=2, group_list_type=1,
            group_type=0, group_list=group_list,
        )[0]
    _sync()

    # 计时
    start = time.perf_counter()
    for _ in range(repeats):
        _ = torch_npu.npu_grouped_matmul(
            x=[x], weight=[w],
            split_item=2, group_list_type=1,
            group_type=0, group_list=group_list,
        )[0]
    _sync()
    elapsed = (time.perf_counter() - start) / repeats * 1000
    return elapsed


def bench_grouped_matmul_w8a8(M: int, K: int, N: int,
                              num_experts: int = 8,
                              warmup: int = 20,
                              repeats: int = 100) -> dict:
    """
    W8A8D MoE 路径 (dynamic_quant + grouped_matmul INT8)

    模拟 quant_apply_mlp 中的一次量化 matmul:
      1. npu_dynamic_quant(x) → quantized_x, pertoken_scale
      2. npu_grouped_matmul(quantized_x, w_int8, scale=w_scale,
                            per_token_scale=[pertoken_scale])

    Returns:
        {"total_ms": float, "dynamic_quant_ms": float, "matmul_ms": float}
    """
    total_tokens = M * num_experts
    x = torch.randn(total_tokens, K, dtype=torch.bfloat16, device="npu")
    w_int8 = torch.randint(-128, 127, (num_experts, K, N),
                           dtype=torch.int8, device="npu")
    w_scale = torch.randn(num_experts, N, dtype=torch.bfloat16, device="npu")

    group_list = torch.full((num_experts,), M,
                            dtype=torch.int64, device="npu")

    # warmup
    for _ in range(warmup):
        qx, pscale = torch_npu.npu_dynamic_quant(x)
        _ = torch_npu.npu_grouped_matmul(
            x=[qx], weight=[w_int8],
            scale=[w_scale], per_token_scale=[pscale],
            split_item=2, group_list_type=1,
            group_type=0, group_list=group_list,
            output_dtype=torch.bfloat16,
        )[0]
    _sync()

    # 测 dynamic_quant 单独耗时
    start = time.perf_counter()
    for _ in range(repeats):
        qx, pscale = torch_npu.npu_dynamic_quant(x)
    _sync()
    dq_ms = (time.perf_counter() - start) / repeats * 1000

    # 测 grouped_matmul(INT8) 单独耗时
    qx, pscale = torch_npu.npu_dynamic_quant(x)
    _sync()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = torch_npu.npu_grouped_matmul(
            x=[qx], weight=[w_int8],
            scale=[w_scale], per_token_scale=[pscale],
            split_item=2, group_list_type=1,
            group_type=0, group_list=group_list,
            output_dtype=torch.bfloat16,
        )[0]
    _sync()
    gmm_ms = (time.perf_counter() - start) / repeats * 1000

    # 测总耗时 (dynamic_quant + grouped_matmul 串行)
    _sync()
    start = time.perf_counter()
    for _ in range(repeats):
        qx, pscale = torch_npu.npu_dynamic_quant(x)
        _ = torch_npu.npu_grouped_matmul(
            x=[qx], weight=[w_int8],
            scale=[w_scale], per_token_scale=[pscale],
            split_item=2, group_list_type=1,
            group_type=0, group_list=group_list,
            output_dtype=torch.bfloat16,
        )[0]
    _sync()
    total_ms = (time.perf_counter() - start) / repeats * 1000

    return {"total_ms": total_ms, "dynamic_quant_ms": dq_ms, "matmul_ms": gmm_ms}


def bench_dense_matmul_bf16(M: int, K: int, N: int,
                            warmup: int = 20,
                            repeats: int = 100) -> float:
    """BF16 Dense 路径 (标准 matmul, 用于 Attention 投影)"""
    x = torch.randn(M, K, dtype=torch.bfloat16, device="npu")
    w = torch.randn(K, N, dtype=torch.bfloat16, device="npu")

    for _ in range(warmup):
        _ = torch.matmul(x, w)
    _sync()

    start = time.perf_counter()
    for _ in range(repeats):
        _ = torch.matmul(x, w)
    _sync()
    return (time.perf_counter() - start) / repeats * 1000


def bench_dense_matmul_w8a8(M: int, K: int, N: int,
                            warmup: int = 20,
                            repeats: int = 100) -> dict:
    """
    W8A8D Dense 路径 (dynamic_quant + quant_matmul, 用于 Attention 投影)

    对应 AscendW8A8DynamicLinearMethod.apply()
    """
    x = torch.randn(M, K, dtype=torch.bfloat16, device="npu")
    w_int8 = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="npu")
    # 转置 + NZ format (模拟 process_weights_after_loading)
    w_int8 = w_int8.transpose(0, 1).contiguous()
    w_scale = torch.randn(N, dtype=torch.bfloat16, device="npu")

    for _ in range(warmup):
        qx, pscale = torch_npu.npu_dynamic_quant(x)
        _ = torch_npu.npu_quant_matmul(
            qx, w_int8, w_scale,
            pertoken_scale=pscale, output_dtype=torch.bfloat16,
        )
    _sync()

    # dynamic_quant 耗时
    start = time.perf_counter()
    for _ in range(repeats):
        qx, pscale = torch_npu.npu_dynamic_quant(x)
    _sync()
    dq_ms = (time.perf_counter() - start) / repeats * 1000

    # quant_matmul 耗时
    qx, pscale = torch_npu.npu_dynamic_quant(x)
    _sync()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = torch_npu.npu_quant_matmul(
            qx, w_int8, w_scale,
            pertoken_scale=pscale, output_dtype=torch.bfloat16,
        )
    _sync()
    mm_ms = (time.perf_counter() - start) / repeats * 1000

    # 总耗时
    _sync()
    start = time.perf_counter()
    for _ in range(repeats):
        qx, pscale = torch_npu.npu_dynamic_quant(x)
        _ = torch_npu.npu_quant_matmul(
            qx, w_int8, w_scale,
            pertoken_scale=pscale, output_dtype=torch.bfloat16,
        )
    _sync()
    total_ms = (time.perf_counter() - start) / repeats * 1000

    return {"total_ms": total_ms, "dynamic_quant_ms": dq_ms, "matmul_ms": mm_ms}


# ============================================================
# 高层封装: 按模型/类别跑 benchmark
# ============================================================

def run_shape(shape: dict, warmup: int, repeats: int) -> dict:
    """对单个 shape 跑 BF16 和 W8A8D 对比"""
    M, K, N = shape["M"], shape["K"], shape["N"]
    name = shape["name"]
    is_moe = "expert" in name

    if is_moe:
        num_experts = shape.get("count_per_layer", 8)
        bf16_ms = bench_grouped_matmul_bf16(
            M, K, N, num_experts=num_experts,
            warmup=warmup, repeats=repeats)
        w8a8 = bench_grouped_matmul_w8a8(
            M, K, N, num_experts=num_experts,
            warmup=warmup, repeats=repeats)
    else:
        bf16_ms = bench_dense_matmul_bf16(
            M, K, N, warmup=warmup, repeats=repeats)
        w8a8 = bench_dense_matmul_w8a8(
            M, K, N, warmup=warmup, repeats=repeats)

    speedup = bf16_ms / w8a8["total_ms"] if w8a8["total_ms"] > 0 else 0
    result = {
        "name": name,
        "M": M, "K": K, "N": N,
        "bf16_ms": round(bf16_ms, 4),
        "w8a8d_total_ms": round(w8a8["total_ms"], 4),
        "w8a8d_dquant_ms": round(w8a8["dynamic_quant_ms"], 4),
        "w8a8d_matmul_ms": round(w8a8["matmul_ms"], 4),
        "speedup": round(speedup, 4),
        "dquant_overhead_pct": round(
            w8a8["dynamic_quant_ms"] / w8a8["total_ms"] * 100, 1)
        if w8a8["total_ms"] > 0 else 0,
    }
    return result


def run_model(model_name: str, category: str = "all",
              decode_batch: int = 256,
              warmup: int = 20, repeats: int = 100) -> list[dict]:
    """对指定模型跑全量或指定类别的 benchmark"""
    all_shapes = get_all_shapes(model_name, decode_batch)
    results = []

    categories = (
        [category] if category != "all"
        else ["attention", "moe", "dense_ffn"]
    )

    for cat in categories:
        shapes = all_shapes.get(cat, [])
        for shape in shapes:
            print(f"  [{model_name}] {shape['name']:<25s} "
                  f"M={shape['M']:<4d} K={shape['K']:<6d} N={shape['N']:<6d} ...",
                  end="", flush=True)
            result = run_shape(shape, warmup, repeats)
            result["model"] = model_name
            result["category"] = cat
            result["num_layers"] = shape["num_layers"]
            result["count_per_layer"] = shape["count_per_layer"]

            arrow = "↑" if result["speedup"] >= 1.0 else "↓"
            print(f"  speedup={result['speedup']:.3f}x {arrow}  "
                  f"(dquant={result['dquant_overhead_pct']:.0f}%)")
            results.append(result)

    return results


def run_sweep_m(K: int, N: int,
                m_values: list[int] | None = None,
                num_experts: int = 8,
                warmup: int = 20, repeats: int = 100) -> list[dict]:
    """Sweep M 维度, 找 BF16 vs W8A8D 的 breakeven point"""
    if m_values is None:
        m_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    results = []
    print(f"\n  Sweep M for grouped_matmul  K={K}, N={N}, "
          f"num_experts={num_experts}")
    print(f"  {'M':>5s}  {'BF16(ms)':>10s}  {'W8A8D(ms)':>10s}  "
          f"{'Speedup':>8s}  {'DQuant%':>8s}")
    print(f"  {'-'*50}")

    for M in m_values:
        bf16_ms = bench_grouped_matmul_bf16(
            M, K, N, num_experts=num_experts,
            warmup=warmup, repeats=repeats)
        w8a8 = bench_grouped_matmul_w8a8(
            M, K, N, num_experts=num_experts,
            warmup=warmup, repeats=repeats)
        speedup = bf16_ms / w8a8["total_ms"] if w8a8["total_ms"] > 0 else 0
        dq_pct = (w8a8["dynamic_quant_ms"] / w8a8["total_ms"] * 100
                  if w8a8["total_ms"] > 0 else 0)

        arrow = "↑" if speedup >= 1.0 else "↓"
        print(f"  {M:>5d}  {bf16_ms:>10.4f}  {w8a8['total_ms']:>10.4f}  "
              f"{speedup:>7.3f}x{arrow} {dq_pct:>7.1f}%")

        results.append({
            "M": M, "K": K, "N": N,
            "bf16_ms": round(bf16_ms, 4),
            "w8a8d_total_ms": round(w8a8["total_ms"], 4),
            "w8a8d_dquant_ms": round(w8a8["dynamic_quant_ms"], 4),
            "w8a8d_matmul_ms": round(w8a8["matmul_ms"], 4),
            "speedup": round(speedup, 4),
            "dquant_overhead_pct": round(dq_pct, 1),
        })

    return results


# ============================================================
# 预测: 加权汇总整模型 decode speedup
# ============================================================

def predict_model_speedup(results: list[dict]) -> dict:
    """
    根据单算子结果预测整模型 decode 阶段 speedup

    加权方式: 每个算子的权重 = count_per_layer × num_layers × bf16_ms
    """
    total_bf16 = 0.0
    total_w8a8d = 0.0

    for r in results:
        weight = r.get("count_per_layer", 1) * r.get("num_layers", 1)
        total_bf16 += r["bf16_ms"] * weight
        total_w8a8d += r["w8a8d_total_ms"] * weight

    predicted = total_bf16 / total_w8a8d if total_w8a8d > 0 else 0

    return {
        "predicted_speedup": round(predicted, 4),
        "total_bf16_weighted_ms": round(total_bf16, 4),
        "total_w8a8d_weighted_ms": round(total_w8a8d, 4),
    }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="单算子 Benchmark: BF16 vs W8A8D on Ascend NPU")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                       help="测试所有模型所有 shape")
    group.add_argument("--model", type=str, choices=list(MODELS.keys()),
                       help="指定模型")
    group.add_argument("--sweep-m", action="store_true",
                       help="Sweep M 维度 (需配合 --K, --N)")
    group.add_argument("--custom", action="store_true",
                       help="自定义 shape (需配合 --M, --K, --N)")

    parser.add_argument("--category", type=str, default="all",
                        choices=["all", "attention", "moe", "dense_ffn"],
                        help="测试类别 (默认 all)")
    parser.add_argument("--decode-batch", type=int, default=256,
                        help="decode 并发序列数 (默认 256)")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--K", type=int, default=2048)
    parser.add_argument("--N", type=int, default=1536)
    parser.add_argument("--num-experts", type=int, default=8,
                        help="活跃 expert 数 (用于 sweep/custom)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=str, default=None,
                        help="输出 JSON 路径")
    args = parser.parse_args()

    all_results = {}

    if args.all:
        for model_name in MODELS:
            print(f"\n{'='*60}")
            print(f"  Model: {model_name}")
            print(f"{'='*60}")
            results = run_model(
                model_name, args.category, args.decode_batch,
                args.warmup, args.repeats)
            prediction = predict_model_speedup(results)
            print(f"\n  >>> 预测整模型 decode speedup: "
                  f"{prediction['predicted_speedup']:.3f}x")
            all_results[model_name] = {
                "shapes": results,
                "prediction": prediction,
            }

    elif args.model:
        results = run_model(
            args.model, args.category, args.decode_batch,
            args.warmup, args.repeats)
        prediction = predict_model_speedup(results)
        print(f"\n  >>> 预测整模型 decode speedup: "
              f"{prediction['predicted_speedup']:.3f}x")
        all_results[args.model] = {
            "shapes": results,
            "prediction": prediction,
        }

    elif args.sweep_m:
        results = run_sweep_m(
            args.K, args.N,
            num_experts=args.num_experts,
            warmup=args.warmup, repeats=args.repeats)
        all_results["sweep_m"] = {
            "K": args.K, "N": args.N,
            "num_experts": args.num_experts,
            "results": results,
        }

    elif args.custom:
        shape = {"name": "custom", "M": args.M, "K": args.K, "N": args.N,
                 "count_per_layer": args.num_experts, "num_layers": 1}
        result = run_shape(shape, args.warmup, args.repeats)
        arrow = "↑" if result["speedup"] >= 1.0 else "↓"
        print(f"\n  M={args.M} K={args.K} N={args.N}")
        print(f"  BF16:  {result['bf16_ms']:.4f} ms")
        print(f"  W8A8D: {result['w8a8d_total_ms']:.4f} ms "
              f"(dquant={result['w8a8d_dquant_ms']:.4f} "
              f"+ matmul={result['w8a8d_matmul_ms']:.4f})")
        print(f"  Speedup: {result['speedup']:.3f}x {arrow}")
        all_results["custom"] = result

    # 输出 JSON
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
