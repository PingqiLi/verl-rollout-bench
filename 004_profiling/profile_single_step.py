#!/usr/bin/env python3
"""
单步 Profiling: 精确采集少量 decode step 的算子级 trace.

取代 vllm bench throughput --profile 的粗粒度方式 (采集全部 step, 产生 GB 级 trace),
本脚本仅采集极少量 step (默认 1 prefill + 1 decode), 数据量约 5-20 MB.

工作原理:
  1. 设置 VLLM_TORCH_PROFILER_DIR 环境变量 (必须在 import vllm 之前)
  2. LLM() 加载模型, worker 中初始化 torch_npu.profiler (不开始采集)
  3. generate() warmup 若干步 (graph capture + JIT 编译, 不采集)
  4. start_profile() → generate(max_tokens=N) → stop_profile()
  5. trace 自动保存到 output_dir

图模式:
  V1 Engine 默认启用 ACLGraph. torch_npu.profiler 在 kernel 级别采集,
  不受图模式影响, 每个 NPU kernel (npu_grouped_matmul 等) 独立可见.

输出:
  Chrome Trace JSON (*.pt.trace.json), 可在 Perfetto UI 或 MindStudio Insight 中查看,
  也可用 analyze_profile.py 自动解析对比.

用法:
    # 直接指定模型路径 (BF16)
    python3 profile_single_step.py --model /path/to/model --quant bf16

    # 通过 config.yaml 指定
    python3 profile_single_step.py --model-key qwen3-30b-a3b --quant bf16

    # W8A8D 量化
    python3 profile_single_step.py --model-key qwen3-30b-a3b --quant w8a8

    # 自定义参数
    python3 profile_single_step.py --model /path/to/model --quant bf16 \\
        --batch-size 64 --warmup-steps 5 --tp 4

    # Eager 模式 (对照实验)
    python3 profile_single_step.py --model-key qwen3-30b-a3b --quant bf16 --enforce-eager
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ======================== 默认值 ========================

DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_TOKENS = 2       # 1 prefill + 1 decode step
DEFAULT_WARMUP_STEPS = 3     # graph capture + JIT 需要 2-3 步
DEFAULT_MAX_MODEL_LEN = 16   # 仅需容纳 input + max_tokens
DEFAULT_TP = 4
DEFAULT_GPU_MEM_UTIL = 0.8


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


# ======================== Profiling 主逻辑 ========================

def run_profiling(
    *,
    model_path: str,
    output_dir: str,
    quant: str = "bf16",
    tp: int = DEFAULT_TP,
    gpu_mem_util: float = DEFAULT_GPU_MEM_UTIL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    enforce_eager: bool = False,
):
    """
    加载模型, warmup, 精确采集指定 step 数的 profiling trace.

    流程:
      1. 设 VLLM_TORCH_PROFILER_DIR (worker 初始化时读取)
      2. LLM() 加载模型 → worker._init_profiler() 创建 torch_npu.profiler
      3. warmup: generate() 若干步, 触发 ACLGraph capture + JIT
      4. start_profile() → profiler.start() (开始采集)
      5. generate(max_tokens) → 触发 1+ execute_model()
      6. stop_profile() → profiler.stop() (保存 trace)

    max_tokens=2 时: 1 次 prefill (input→token1) + 1 次 decode (token1→token2)
    """
    # 必须在 import vllm 之前设置, worker 初始化时检查此变量
    os.environ["VLLM_TORCH_PROFILER_DIR"] = output_dir
    os.environ.setdefault("VLLM_USE_V1", "1")

    # 延迟 import: 确保环境变量已生效
    from vllm import LLM, SamplingParams

    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  单步 Profiling: {quant.upper()}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"模型:     {model_path}", file=sys.stderr)
    print(f"精度:     {quant}", file=sys.stderr)
    print(f"TP:       {tp}", file=sys.stderr)
    print(f"显存:     {gpu_mem_util}", file=sys.stderr)
    print(f"Batch:    {batch_size}", file=sys.stderr)
    print(f"Tokens:   {max_tokens} (1 prefill + {max_tokens - 1} decode)", file=sys.stderr)
    print(f"模式:     {'eager' if enforce_eager else 'graph (ACLGraph)'}", file=sys.stderr)
    print(f"Warmup:   {warmup_steps} 步", file=sys.stderr)
    print(f"输出:     {output_dir}", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)

    # ---- 构建 LLM ----
    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        # NZ 格式: Ascend 生产环境标配, 让 npu_grouped_matmul 等走 FRACTAL_NZ 路径
        additional_config={"enable_weight_nz_layout": True},
    )
    if quant not in ("bf16", "bfloat16"):
        llm_kwargs["quantization"] = "ascend"

    print("加载模型中...", file=sys.stderr)
    t0 = time.time()
    llm = LLM(**llm_kwargs)
    print(f"模型加载完成 ({time.time() - t0:.1f}s)\n", file=sys.stderr)

    # ---- Warmup ----
    # warmup 触发 ACLGraph capture + JIT 编译, 确保后续采集的是稳态算子.
    # warmup 期间不开启 profiler, 不产生 trace 数据.
    warmup_prompts = ["hello"] * batch_size
    warmup_params = SamplingParams(max_tokens=max_tokens, temperature=0)

    print(f"Warmup ({warmup_steps} 步)...", file=sys.stderr)
    for i in range(warmup_steps):
        llm.generate(warmup_prompts, warmup_params)
        print(f"  step {i + 1}/{warmup_steps} 完成", file=sys.stderr)
    print("Warmup 完成.\n", file=sys.stderr)

    # ---- Profile ----
    # start_profile() → worker.profiler.start() → torch_npu.profiler 开始采集
    # generate() 触发 execute_model(), profiler 记录所有 NPU kernel
    # stop_profile() → worker.profiler.stop() → on_trace_ready 保存 trace
    profile_prompts = ["hello"] * batch_size
    profile_params = SamplingParams(max_tokens=max_tokens, temperature=0)

    print(f"开始 profiling: {batch_size} prompts × {max_tokens} tokens ...",
          file=sys.stderr)

    llm.start_profile()
    outputs = llm.generate(profile_prompts, profile_params)
    llm.stop_profile()

    print(f"\nProfiling 完成!", file=sys.stderr)
    print(f"  采集 {len(outputs)} 个请求", file=sys.stderr)

    # 等待 MP 后台进程写完 trace
    print("  等待 trace 写入...", file=sys.stderr)
    time.sleep(5)

    # ---- 报告输出文件 ----
    _report_trace_files(output_dir)


def _report_trace_files(output_dir: str):
    """扫描并报告 trace 文件."""
    trace_dir = Path(output_dir)
    trace_files = sorted(trace_dir.rglob("*.pt.trace.json"))

    if trace_files:
        total_size = sum(f.stat().st_size for f in trace_files)
        print(f"\n  trace 文件 ({len(trace_files)} 个, "
              f"共 {total_size / 1024 / 1024:.1f} MB):", file=sys.stderr)
        for tf in trace_files[:5]:
            rel = tf.relative_to(trace_dir)
            size_mb = tf.stat().st_size / 1024 / 1024
            print(f"    {rel} ({size_mb:.1f} MB)", file=sys.stderr)
        if len(trace_files) > 5:
            print(f"    ... 共 {len(trace_files)} 个文件", file=sys.stderr)
    else:
        print(f"\n  警告: 未找到 *.pt.trace.json 文件", file=sys.stderr)
        # 列出目录内容供排查
        all_files = sorted(trace_dir.rglob("*"))
        if all_files:
            print(f"  目录下有 {len(all_files)} 个文件:", file=sys.stderr)
            for f in list(all_files)[:10]:
                print(f"    {f.relative_to(trace_dir)}", file=sys.stderr)
        else:
            print(f"  目录为空: {output_dir}", file=sys.stderr)

    print(f"\n  输出目录: {output_dir}", file=sys.stderr)
    print(f"\n查看方式:", file=sys.stderr)
    print(f"  1. Perfetto UI:  https://ui.perfetto.dev/", file=sys.stderr)
    print(f"  2. MindStudio Insight", file=sys.stderr)
    print(f"  3. python3 analyze_profile.py <run_dir>", file=sys.stderr)


# ======================== 参数解析 ========================

def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(
        description="单步 Profiling: 精确采集少量 decode step 的算子级 trace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # BF16 profiling (使用 config.yaml)
  python3 profile_single_step.py --model-key qwen3-30b-a3b --quant bf16

  # W8A8D profiling
  python3 profile_single_step.py --model-key qwen3-30b-a3b --quant w8a8

  # 直接指定模型路径
  python3 profile_single_step.py --model /data/models/Qwen3-30B --quant bf16

  # 大 batch + eager 模式
  python3 profile_single_step.py --model-key qwen3-30b-a3b --quant bf16 \\
      --batch-size 128 --enforce-eager
""",
    )

    # 模型指定方式 (二选一)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        help="模型路径 (直接指定)",
    )
    model_group.add_argument(
        "--model-key",
        help="config.yaml 中的模型 key (如 qwen3-30b-a3b)",
    )

    parser.add_argument(
        "--quant", default="bf16",
        help="精度: bf16, w8a8, w8a16 (默认: bf16)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="trace 输出目录 (默认: 自动生成 outputs/<timestamp>/profile_<quant>)",
    )
    parser.add_argument(
        "--config", default=None,
        help="配置文件路径 (默认: ../config.yaml, 仅 --model-key 时使用)",
    )

    # 模型参数
    parser.add_argument(
        "--tp", type=int, default=None,
        help=f"tensor parallel 数 (默认: 从 config 读取, 或 {DEFAULT_TP})",
    )
    parser.add_argument(
        "--gpu-mem-util", type=float, default=None,
        help=f"GPU 显存利用率 (默认: 从 config 读取, 或 {DEFAULT_GPU_MEM_UTIL})",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN,
        help=f"最大序列长度 (默认: {DEFAULT_MAX_MODEL_LEN})",
    )
    parser.add_argument(
        "--enforce-eager", action="store_true",
        help="禁用图模式 (ACLGraph), 使用 eager 模式",
    )

    # Profiling 参数
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"同时推理的 prompt 数 (默认: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"每 prompt 生成 token 数, 2=1 prefill+1 decode (默认: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS,
        help=f"warmup 步数, 用于 graph capture + JIT (默认: {DEFAULT_WARMUP_STEPS})",
    )

    return parser.parse_args()


# ======================== 入口 ========================

def main():
    args = parse_args()

    # 解析模型路径
    if args.model:
        model_path = args.model
        tp = args.tp if args.tp is not None else DEFAULT_TP
        gpu_mem_util = (args.gpu_mem_util if args.gpu_mem_util is not None
                        else DEFAULT_GPU_MEM_UTIL)
    else:
        model_path, tp_cfg, mem_cfg = resolve_from_config(
            model_key=args.model_key,
            quant=args.quant,
            config_file=args.config,
        )
        tp = args.tp if args.tp is not None else tp_cfg
        gpu_mem_util = (args.gpu_mem_util if args.gpu_mem_util is not None
                        else mem_cfg)

    # 检查模型路径
    # 注: 远程模型 (HuggingFace) 不检查本地路径
    model_p = Path(model_path)
    if not model_p.is_dir() and "/" not in model_path:
        print(f"模型路径不存在: {model_path}", file=sys.stderr)
        sys.exit(1)

    # 自动生成输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        from datetime import datetime
        script_dir = Path(__file__).resolve().parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_tag = "eager" if args.enforce_eager else "graph"
        output_dir = str(
            script_dir / "outputs" / f"{timestamp}_{mode_tag}" / f"profile_{args.quant}"
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    run_profiling(
        model_path=model_path,
        output_dir=output_dir,
        quant=args.quant,
        tp=tp,
        gpu_mem_util=gpu_mem_util,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        warmup_steps=args.warmup_steps,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )


if __name__ == "__main__":
    main()
