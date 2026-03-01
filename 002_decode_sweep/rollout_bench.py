#!/usr/bin/env python3
"""
模拟 GRPO Rollout 负载: 混合 output_len 分布

每个 prompt 的 max_tokens 从指定分布中采样, 模拟实际 rollout 中
不同 prompt 产生不同长度回复的场景.

使用方式:
    python3 rollout_bench.py --model /path/to/model --tp 4
    python3 rollout_bench.py --model /path/to/model --tp 4 \
        --dist uniform --min-tokens 128 --max-tokens 4096
    python3 rollout_bench.py --model /path/to/model --tp 4 --output result.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ======================== 负载生成 ========================

def generate_output_lens(num_prompts: int, *, dist: str,
                         min_tokens: int, max_tokens: int,
                         seed: int) -> list[int]:
    """
    生成 output_len 分布

    zipf: 大部分短序列, 少量长序列 (最接近真实 rollout)
    uniform: 均匀分布
    bimodal: 70% 短 + 30% 长
    """
    rng = np.random.RandomState(seed)

    if dist == "zipf":
        # Truncated exponential 近似真实 rollout 分布
        scale = (max_tokens - min_tokens) / 3.0
        raw = rng.exponential(scale=scale, size=num_prompts)
        lens = np.clip(raw + min_tokens, min_tokens, max_tokens).astype(int)
    elif dist == "uniform":
        lens = rng.randint(min_tokens, max_tokens + 1, size=num_prompts)
    elif dist == "bimodal":
        mid = (min_tokens + max_tokens) // 2
        n_short = int(num_prompts * 0.7)
        n_long = num_prompts - n_short
        short = rng.randint(min_tokens, mid + 1, size=n_short)
        long_ = rng.randint(mid, max_tokens + 1, size=n_long)
        lens = np.concatenate([short, long_])
        rng.shuffle(lens)
    else:
        print(f"未知分布: {dist}", file=sys.stderr)
        sys.exit(1)

    return lens.tolist()


# ======================== 配置解析 ========================

def resolve_from_config(
    *,
    model_key: str,
    quant: str,
    config_file: str | None = None,
) -> tuple[str, int, float]:
    """通过 config_parser.py 解析模型路径、TP 数、gpu_mem_util."""
    repo_root = Path(__file__).resolve().parent.parent
    parser_script = repo_root / "config_parser.py"
    config_path = Path(config_file) if config_file else repo_root / "config.yaml"

    if not parser_script.is_file():
        print(f"找不到 config_parser.py: {parser_script}", file=sys.stderr)
        sys.exit(1)
    if not config_path.is_file():
        print(f"找不到配置文件: {config_path}", file=sys.stderr)
        sys.exit(1)

    def cfg(*args: str) -> str:
        result = subprocess.run(
            ["python3", str(parser_script), str(config_path)] + list(args),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"配置解析失败: {' '.join(args)}", file=sys.stderr)
            print(f"  {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)
        return result.stdout.strip()

    model_path = cfg("get-path", model_key, quant)
    tp = int(cfg("get", model_key, "tp"))
    gpu_mem_util = float(cfg("get", model_key, "gpu_mem_util"))

    return model_path, tp, gpu_mem_util


# ======================== Benchmark ========================

def run_benchmark(model_path: str, *, tp_size: int, gpu_mem_util: float,
                  input_len: int, output_lens: list[int],
                  num_samples: int) -> dict:
    """加载模型, 运行混合长度 benchmark"""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=False,
        trust_remote_code=True,
    )

    tokenizer = llm.get_tokenizer()

    # 构造 dummy prompts (固定 input_len)
    dummy_ids = tokenizer.encode("benchmark", add_special_tokens=False)
    if not dummy_ids:
        dummy_ids = [1]
    prompt_ids = (dummy_ids * ((input_len // len(dummy_ids)) + 1))[:input_len]

    # 每个 prompt 独立的 SamplingParams (不同 max_tokens)
    prompts = [{"prompt_token_ids": prompt_ids} for _ in output_lens]
    sp_list = [
        SamplingParams(
            n=num_samples,
            max_tokens=olen,
            ignore_eos=True,
            temperature=1.0,
        )
        for olen in output_lens
    ]

    # Warmup (触发 graph capture + JIT)
    _ = llm.generate(
        [{"prompt_token_ids": prompt_ids[:min(16, input_len)]}],
        [SamplingParams(n=1, max_tokens=1, ignore_eos=True)],
    )

    # Benchmark
    start = time.perf_counter()
    outputs = llm.generate(prompts, sp_list)
    elapsed = time.perf_counter() - start

    total_input = sum(len(o.prompt_token_ids) for o in outputs)
    total_output = sum(
        sum(len(s.token_ids) for s in o.outputs)
        for o in outputs
    )

    return {
        "elapsed_s": round(elapsed, 2),
        "num_prompts": len(output_lens),
        "num_samples_per_prompt": num_samples,
        "total_sequences": len(output_lens) * num_samples,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "input_tput_tok_s": round(total_input / elapsed, 1),
        "output_tput_tok_s": round(total_output / elapsed, 1),
    }


# ======================== CLI ========================

def main():
    parser = argparse.ArgumentParser(
        description="模拟 GRPO Rollout: 混合 output_len 分布")

    # 模型来源: --model 直接指定 或 --model-key + --quant 从 config.yaml 读
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", type=str, help="模型路径")
    src.add_argument("--model-key", type=str,
                     help="config.yaml 中的模型 key (配合 --quant)")

    parser.add_argument("--quant", type=str, default="bf16",
                        help="量化类型 (配合 --model-key)")
    parser.add_argument("--config", type=str, default=None,
                        help="config.yaml 路径")
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--gpu-mem-util", type=float, default=None)

    # 负载参数
    parser.add_argument("--input-len", type=int, default=512,
                        help="输入长度 (模拟 prompt)")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=8,
                        help="GRPO 采样数 (n=8)")
    parser.add_argument("--dist", default="zipf",
                        choices=["zipf", "uniform", "bimodal"],
                        help="output_len 分布")
    parser.add_argument("--min-tokens", type=int, default=64,
                        help="最短输出长度")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="最长输出长度")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output", type=str, help="输出 JSON 路径")
    args = parser.parse_args()

    # 解析模型配置
    if args.model_key:
        model_path, cfg_tp, cfg_mem = resolve_from_config(
            model_key=args.model_key, quant=args.quant,
            config_file=args.config)
        tp_size = args.tp if args.tp is not None else cfg_tp
        gpu_mem_util = args.gpu_mem_util if args.gpu_mem_util is not None \
            else cfg_mem
    else:
        model_path = args.model
        tp_size = args.tp or 4
        gpu_mem_util = args.gpu_mem_util or 0.8

    # 生成 output_len 分布
    output_lens = generate_output_lens(
        args.num_prompts, dist=args.dist,
        min_tokens=args.min_tokens, max_tokens=args.max_tokens,
        seed=args.seed)

    ols = np.array(output_lens)
    print(f"\n  负载:")
    print(f"    model: {model_path}")
    print(f"    input_len={args.input_len}, prompts={args.num_prompts}, "
          f"n={args.num_samples}")
    print(f"    output_len: {args.dist} [{args.min_tokens}, {args.max_tokens}]")
    print(f"    均值={ols.mean():.0f}, 中位={np.median(ols):.0f}, "
          f"P90={np.percentile(ols, 90):.0f}, "
          f"max={ols.max()}")

    # 运行
    result = run_benchmark(
        model_path, tp_size=tp_size, gpu_mem_util=gpu_mem_util,
        input_len=args.input_len, output_lens=output_lens,
        num_samples=args.num_samples)

    # 附加配置
    result["config"] = {
        "model": model_path,
        "tp": tp_size,
        "gpu_mem_util": gpu_mem_util,
        "input_len": args.input_len,
        "num_prompts": args.num_prompts,
        "num_samples": args.num_samples,
        "dist": args.dist,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "output_lens": output_lens,
    }

    print(f"\n  结果:")
    print(f"    耗时: {result['elapsed_s']:.1f}s")
    print(f"    Output 吞吐: {result['output_tput_tok_s']:.1f} tok/s")
    print(f"    总输出 tokens: {result['total_output_tokens']:,}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"    保存: {args.output}")


if __name__ == "__main__":
    main()
