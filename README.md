# verl-rollout-bench

Benchmark toolkit for measuring vLLM inference performance under verl RL rollout workloads on Ascend NPU.

Simulates the exact per-GPU workload of verl GRPO training's rollout phase, supporting BF16 / W8A16 / W8A8 quantization comparison across multiple models.

## Architecture

```
verl_rollout_bench/
├── run_vllm_benchmark.sh      # Main script: experiment orchestration
├── summarize_benchmark.py     # Result analysis: JSON → tables / CSV / Markdown
└── README.md
```

## How verl rollout works (what we're simulating)

```
verl GRPO training step:
  train_batch_size=256 prompts, n=8 completions each
  8 NPUs, TP=1, DP=8
                    ↓
  Each NPU's vLLM engine receives:
    32 prompts × n=8 = 256 sequences
    gpu_memory_utilization = 0.4 (shares NPU with actor model)
    input ≤ 512 tokens, output ≤ 256 tokens
                    ↓
  All 256 sequences submitted at once
  Engine schedules based on KV cache capacity
  Wait for ALL completions → actor update
```

The benchmark's **offline mode** replicates this exactly: calling `LLM.generate()` with 32 prompts, `n=8`, matching memory and sequence length settings.

## Quick start

### 1. Configure model paths

Edit the top of `run_vllm_benchmark.sh`:

```bash
MODEL_BASE="/data/l50044498/models"  # Your model root directory
```

And the `declare_model_config()` function for each model's paths:

```bash
QWEN3_1_7B_PATH_BF16="${MODEL_BASE}/qwen3-1.7b"
QWEN3_1_7B_PATH_W8A16="${MODEL_BASE}/qwen3-1.7b-W8A16"
QWEN3_1_7B_PATH_W8A8="${MODEL_BASE}/qwen3-1.7b-W8A8D"
```

> If a model doesn't have a certain quantization variant (e.g., Qwen3-30B-A3B has no W8A16), simply don't define that variable. The script auto-skips unconfigured combos.

### 2. Diagnostic run (verify environment)

```bash
bash run_vllm_benchmark.sh --diagnostic
```

Runs BF16 + Qwen3-1.7B only (4 prompts), checks: model path exists → vllm CLI works → benchmark runs → results saved.

### 3. Full experiment matrix

```bash
bash run_vllm_benchmark.sh
```

Runs all configured model × quantization combinations (default: 8 experiments).

## Two benchmark modes

### Offline mode (default, recommended)

```bash
bash run_vllm_benchmark.sh --offline
```

| Aspect | Detail |
|:---|:---|
| How it works | Calls `LLM.generate()` directly via `vllm bench throughput` |
| Concurrency | 32 prompts × n=8 = 256 sequences per GPU (matches verl) |
| GPU memory | 0.4 (matches verl's co-located deployment) |
| Profiling | Supported via `VLLM_TORCH_PROFILER_DIR` + `--profile` |
| Output metrics | `tokens/s`, `requests/s`, `elapsed_time` |
| Accuracy | **High** — same code path as verl, same batch size, same GEMM dimensions |

**When to use**: Performance comparison (BF16 vs W8A16 vs W8A8). This is the mode whose numbers you should report.

### Online mode

```bash
bash run_vllm_benchmark.sh --online
```

| Aspect | Detail |
|:---|:---|
| How it works | Starts vLLM HTTP server, sends requests via `vllm bench serve` |
| Concurrency | Limited by `--max-concurrency` (default 128), no `n` param |
| GPU memory | 0.4 (configurable) |
| Profiling | Supported via HTTP `/start_profile` / `/stop_profile` |
| Output metrics | TTFT, TPOT, ITL, E2EL (latency breakdown) |
| Accuracy | **Low for rollout simulation** — HTTP overhead, can't do n=8, lower concurrency |

**When to use**: When you need TTFT/TPOT/E2EL latency breakdown, or profiling via HTTP API. Not for reporting rollout throughput.

### Mode comparison

```
Offline (default):                       Online:
  LLM.generate()                           HTTP API Server
  32 prompts, n=8                          256 requests, n=1
  = 256 concurrent sequences               ≤ 128 concurrent (max-concurrency)
  decode GEMM M=256                        decode GEMM M=128
  ✓ matches verl                           ✗ underestimates concurrency
  ✓ quantization benefits accurate         ✗ may underestimate quant benefits
```

## Default parameters (aligned with verl)

| Parameter | Default | verl source | CLI flag |
|:---|:---|:---|:---|
| `NUM_PROMPTS` | 32 | `train_batch_size=256 / 8 GPUs` | `--num-prompts` |
| `NUM_SAMPLES_PER_PROMPT` | 8 | `rollout.n=8` | `-n` |
| `INPUT_LEN` | 512 | `max_prompt_length=512` | `--input-len` |
| `OUTPUT_LEN` | 256 | `max_response_length=256` | `--output-len` |
| `GPU_MEMORY_UTILIZATION` | 0.4 | `rollout.gpu_memory_utilization=0.4` | `--gpu-mem-util` |
| `max_num_seqs` | 1024 | verl RolloutConfig default | (in model config) |
| `max_model_len` | 768 | 512 + 256 | (in model config) |
| `TP` | per-model | `rollout.tensor_model_parallel_size` | (in model config) |
| `BENCH_MODE` | offline | — | `--offline` / `--online` |

## CLI reference

```
bash run_vllm_benchmark.sh [OPTIONS]

Mode:
  --offline             [Default] Offline throughput mode (matches verl rollout)
  --online              Online serving mode (HTTP, latency metrics)
  --diagnostic          Quick validation (BF16 + Qwen3-1.7B, 4 prompts)

Experiment selection:
  --models MODELS       Comma-separated model list (qwen3-1.7b,pangu-7b,qwen3-30b-a3b)
  --quants QUANTS       Comma-separated quant list (bf16,w8a16,w8a8)

Workload parameters:
  --num-prompts N       Prompts per experiment (default: 32, per-GPU in verl)
  -n N                  Samples per prompt (default: 8, offline only)
  --input-len N         Input token length (default: 512)
  --output-len N        Output token length (default: 256)
  --gpu-mem-util F      GPU memory fraction (default: 0.4)

Online mode only:
  --max-concurrency N   Max concurrent HTTP requests (default: 128)
  --port PORT           Server port (default: 8080)

Other:
  --no-profile          Disable torch profiling
  --model-base DIR      Model root directory
  -h, --help            Show help
```

## Common recipes

```bash
# Match verl exactly (default settings)
bash run_vllm_benchmark.sh

# Only Qwen3-1.7B, BF16 vs W8A16
bash run_vllm_benchmark.sh --models qwen3-1.7b --quants bf16,w8a16

# Standalone peak performance (more memory, no verl constraint)
bash run_vllm_benchmark.sh --gpu-mem-util 0.9

# Quick test with small workload
bash run_vllm_benchmark.sh --num-prompts 4 -n 2 --no-profile

# Online mode for latency analysis + profiling
bash run_vllm_benchmark.sh --online --num-prompts 64

# Re-generate summary tables from existing results
python3 summarize_benchmark.py ./benchmark_results --markdown
```

## Output

After experiments complete, the script auto-generates:

```
benchmark_results/
├── qwen3-1.7b_bf16.json       # Raw vLLM benchmark output
├── qwen3-1.7b_w8a16.json
├── ...
├── summary.txt                 # Terminal-friendly comparison table
├── summary.csv                 # For Excel / pandas
└── summary.md                  # For docs / GitHub Issues

benchmark_logs/                 # Execution logs
benchmark_profiles/             # Torch profiling traces (if enabled)
```

### Key metrics to look at

| Metric | Meaning | Why it matters for rollout |
|:---|:---|:---|
| **Output Throughput (tok/s)** | Tokens generated per second | Directly determines rollout speed |
| **Elapsed Time (s)** | Total wall-clock time | rollout step time ≈ elapsed_time |
| TTFT (ms) | Time to first token | Prefill efficiency (online mode only) |
| TPOT (ms) | Time per output token | Decode efficiency (online mode only) |
| E2EL P99 (ms) | 99th percentile end-to-end latency | Tail latency bottleneck (online mode only) |

### Quantization speedup table

The summary automatically computes speedup ratios relative to BF16:

```
Model                Quant         Output Tput     E2EL Mean      E2EL P99
---------------------------------------------------------------------------
qwen3-1.7b          BF16           (baseline)     (baseline)     (baseline)
qwen3-1.7b          W8A16             1.19x ^        1.19x ^        1.20x ^
qwen3-1.7b          W8A8              1.28x ^        1.30x ^        1.30x ^
```

`^` = quantized is faster, `v` = quantized is slower.

## Profiling analysis

### Ascend NPU (recommended)

```python
from torch_npu.profiler.profiler import analyse
analyse(profiler_path="./benchmark_profiles/qwen3-1.7b_bf16")
```

### Perfetto UI (universal)

Upload trace files from `benchmark_profiles/<model>_<quant>/` to https://ui.perfetto.dev/

> Profiling significantly slows inference. Profiled numbers are NOT valid performance measurements. Use `--no-profile` for performance data.

## Troubleshooting

| Symptom | Cause | Fix |
|:---|:---|:---|
| `No available memory for cache blocks` | Previous server not cleaned up, or gpu_memory_utilization too high | Script auto-kills process tree; reduce `--gpu-mem-util` if still OOM |
| Server start timeout | Model too large for available memory | Check `benchmark_logs/server_*.log` |
| Empty profiling directory | Server exited before flush | Increase workload (`--num-prompts`, `--output-len`) |
| `set -u` unbound variable | Typo in model config variable name | Check `declare_model_config()` spelling |
| Summary shows N/A | Experiment failed or JSON filename format wrong | Check logs; ensure `<model>_<quant>.json` naming |
