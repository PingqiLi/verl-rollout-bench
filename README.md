# verl-rollout-bench

Benchmark toolkit for comparing BF16 / W8A16 / W8A8 inference throughput on Ascend NPU, targeting verl RL rollout workloads.

**Core question: does low-precision quantization speed up RL rollout?**

## Files

```
run_vllm_benchmark.sh      # Experiment orchestration (start/stop/profile/cleanup)
summarize_benchmark.py     # Result analysis (JSON → comparison tables)
```

## Quick start

### 1. Edit model paths

Open `run_vllm_benchmark.sh`, update the `declare_model_config()` section:

```bash
MODEL_BASE="/data/l50044498/models"

# Each model: display name, paths for each precision, TP size, memory fraction
QWEN3_1_7B_PATH_BF16="${MODEL_BASE}/qwen3-1.7b"
QWEN3_1_7B_PATH_W8A16="${MODEL_BASE}/qwen3-1.7b-W8A16"
QWEN3_1_7B_PATH_W8A8="${MODEL_BASE}/qwen3-1.7b-W8A8D"
QWEN3_1_7B_TP=1
QWEN3_1_7B_GPU_MEM_UTIL=0.4
```

> If a model doesn't have a certain precision variant, don't define that path variable. The script auto-skips it.

### 2. Diagnostic run

```bash
bash run_vllm_benchmark.sh --diagnostic
```

Runs BF16 + Qwen3-1.7B only (4 prompts) to verify the environment works.

### 3. Run all experiments

```bash
bash run_vllm_benchmark.sh
```

Default: 8 experiments (3 models × 3 precisions - 1 missing combo).

### 4. View results

```bash
# Auto-generated after experiments; or re-generate manually:
python3 summarize_benchmark.py ./benchmark_results --markdown
```

## Two modes

| | Offline (default) | Online |
|:---|:---|:---|
| Command | `--offline` | `--online` |
| How | `LLM.generate()` directly | HTTP server + client |
| Supports n=8 | Yes | No |
| Metrics | throughput (tok/s) | TTFT, TPOT, ITL, E2EL |
| Use for | **Performance comparison** | Latency breakdown |

## CLI reference

```
bash run_vllm_benchmark.sh [OPTIONS]

Mode:
  --offline             [Default] Offline throughput benchmark
  --online              Online serving benchmark (latency metrics)
  --diagnostic          Quick validation run

Experiments:
  --models M1,M2,...    Models to test (qwen3-1.7b, pangu-7b, qwen3-30b-a3b)
  --quants Q1,Q2,...    Precisions to test (bf16, w8a16, w8a8)

Workload:
  --num-prompts N       Number of prompts (default: 32)
  -n N                  Samples per prompt (default: 8, offline only)
  --input-len N         Input length (default: 512)
  --output-len N        Output length (default: 256)
  --gpu-mem-util F      Override GPU memory fraction for all models

Other:
  --no-profile          Skip torch profiling
  --model-base DIR      Model root directory
```

## Examples

```bash
# Default: all models, all precisions, offline mode
bash run_vllm_benchmark.sh

# Just Qwen3-1.7B, BF16 vs W8A16
bash run_vllm_benchmark.sh --models qwen3-1.7b --quants bf16,w8a16

# Quick small run
bash run_vllm_benchmark.sh --num-prompts 4 -n 2 --no-profile

# Online mode for latency analysis
bash run_vllm_benchmark.sh --online --num-prompts 64
```

## Output

```
benchmark_results/
├── qwen3-1.7b_bf16.json       # Raw results
├── qwen3-1.7b_w8a16.json
├── summary.txt                 # Comparison table (terminal)
├── summary.csv                 # For Excel / pandas
└── summary.md                  # For docs / Issues

benchmark_logs/                 # Execution logs
benchmark_profiles/             # Torch profiling traces (if enabled)
```

## Model configs

| Model | TP | gpu_memory_utilization |
|:---|:---|:---|
| Qwen3-1.7B | 1 | 0.4 |
| Pangu-7B | 1 | 0.6 |
| Qwen3-30B-A3B | 4 | 0.8 |

`gpu_memory_utilization` is per-model (reflecting different verl co-location constraints). Override all with `--gpu-mem-util 0.9` for standalone peak testing.

## Profiling

```python
# Ascend NPU
from torch_npu.profiler.profiler import analyse
analyse(profiler_path="./benchmark_profiles/qwen3-1.7b_bf16")
```

Or upload traces to https://ui.perfetto.dev/

> Profiling slows inference significantly. Use `--no-profile` when you only need throughput numbers.
