# verl-rollout-bench

在 Ascend NPU 上对比 BF16 / W8A16 / W8A8 低精度量化对 vLLM 推理吞吐的加速效果，面向 verl RL rollout 场景。

## 文件

```
run_vllm_benchmark.sh      # 主控脚本: 实验编排、server 管理、profiling
summarize_benchmark.py     # 结果分析: JSON → 对比表格 (txt/csv/markdown)
```

## 环境变量

使用前通过环境变量配置路径，避免修改脚本：

```bash
# 必须: 模型根目录
export MODEL_BASE="/data/l50044498/models"

# 可选: 结果/日志/profiling 输出的根目录 (默认: 当前目录)
export BENCH_BASE_DIR="/home/l00861652/exps"
```

设好后，结果会输出到：
```
$BENCH_BASE_DIR/benchmark_results/   # JSON 结果 + 汇总表格
$BENCH_BASE_DIR/benchmark_logs/      # 运行日志
$BENCH_BASE_DIR/benchmark_profiles/  # Torch profiling 数据
```

## 快速开始

### 1. 诊断 (首次必跑, 验证环境)

```bash
export MODEL_BASE="/data/l50044498/models"
bash run_vllm_benchmark.sh --diagnostic
```

只跑 BF16 + Qwen3-1.7B，4 个 prompt，验证环境能跑通。

### 2. 跑全量实验

```bash
bash run_vllm_benchmark.sh
```

默认跑 8 组 (3 模型 × 3 精度 - 1 个缺失组合)，自动生成对比表格。

### 3. 查看结果

```bash
# 跑完自动生成; 也可手动重新生成:
python3 summarize_benchmark.py $BENCH_BASE_DIR/benchmark_results --markdown
```

## 两种 Benchmark 模式

### Offline 模式 (默认, 推荐)

```bash
bash run_vllm_benchmark.sh --offline
```

直接调用 `LLM.generate()`，支持 `n=8` (每 prompt 多次采样)，最接近 verl rollout 的真实调用路径。**性能对比数据用这个模式。**

### Online 模式

```bash
bash run_vllm_benchmark.sh --online
```

走 HTTP 服务，输出 TTFT / TPOT / ITL / E2EL 延迟指标。适合做延迟分析，但并发受限、不支持 `n` 参数，**不建议用于吞吐对比**。

### 模式对比

| | Offline (默认) | Online |
|:---|:---|:---|
| 原理 | `LLM.generate()` 直接推理 | HTTP server + client |
| 支持 n=8 | 支持 | 不支持 |
| 输出指标 | throughput (tok/s) | TTFT, TPOT, ITL, E2EL |
| 用途 | **吞吐对比** | 延迟分析 |

## 命令行参数

```
bash run_vllm_benchmark.sh [选项]

模式:
  --offline             [默认] 离线吞吐 benchmark
  --online              在线服务 benchmark (延迟指标)
  --diagnostic          快速诊断 (BF16 + Qwen3-1.7B, 4 prompts)

实验选择:
  --models M1,M2,...    模型列表 (qwen3-1.7b, pangu-7b, qwen3-30b-a3b)
  --quants Q1,Q2,...    精度列表 (bf16, w8a16, w8a8)

负载参数:
  --num-prompts N       prompt 数量 (默认: 32)
  -n N                  每 prompt 采样数 (默认: 8, 仅 offline)
  --input-len N         输入长度 (默认: 512)
  --output-len N        输出长度 (默认: 256)
  --gpu-mem-util F      覆盖所有模型的 GPU 显存比例

其他:
  --no-profile          不采集 torch profiling
  --model-base DIR      覆盖模型根目录 (也可用环境变量 MODEL_BASE)
```

## 常用命令

```bash
# 默认: 全部模型 × 全部精度
bash run_vllm_benchmark.sh

# 只对比 Qwen3-1.7B 的 BF16 和 W8A16
bash run_vllm_benchmark.sh --models qwen3-1.7b --quants bf16,w8a16

# 快速小规模测试
bash run_vllm_benchmark.sh --num-prompts 4 -n 2 --no-profile

# 独立峰值性能 (不模拟 verl 显存约束)
bash run_vllm_benchmark.sh --gpu-mem-util 0.9

# Online 模式做延迟分析
bash run_vllm_benchmark.sh --online --num-prompts 64
```

## 模型配置

| 模型 | TP | gpu_memory_utilization | 备注 |
|:---|:---|:---|:---|
| Qwen3-1.7B | 1 | 0.4 | 小模型, verl 中 actor 占 60% |
| Pangu-7B | 1 | 0.6 | 中模型 |
| Qwen3-30B-A3B | 4 | 0.8 | MoE 大模型, 无 W8A16 |

`gpu_memory_utilization` 按模型大小设置，模拟 verl 中 rollout 与 actor 共享显存的约束。用 `--gpu-mem-util 0.9` 可覆盖所有模型，测独立峰值性能。

模型路径在 `run_vllm_benchmark.sh` 的 `declare_model_config()` 中配置。如果某个模型没有某种精度的权重 (如 Qwen3-30B-A3B 无 W8A16)，不定义对应路径变量即可，脚本自动跳过。

## 输出文件

```
$BENCH_BASE_DIR/
├── benchmark_results/
│   ├── qwen3-1.7b_bf16.json       # 原始结果
│   ├── qwen3-1.7b_w8a16.json
│   ├── summary.txt                 # 终端对比表格
│   ├── summary.csv                 # Excel / pandas 用
│   └── summary.md                  # 文档 / Issue 用
├── benchmark_logs/                 # 运行日志
└── benchmark_profiles/             # Torch profiling 数据
```

## Profiling 分析

```python
# Ascend NPU
from torch_npu.profiler.profiler import analyse
analyse(profiler_path="./benchmark_profiles/qwen3-1.7b_bf16")
```

或上传 trace 文件到 https://ui.perfetto.dev/ 可视化。

> Profiling 会显著拖慢推理。采集 profiling 时的数字**不能作为性能参考**。只需要吞吐数据时用 `--no-profile`。

## 故障排查

| 问题 | 原因 | 解决 |
|:---|:---|:---|
| `No available memory for cache blocks` | 上一个实验进程未清理干净 | 脚本有进程树清理，也可手动 `kill` 残留 vllm 进程 |
| `--gpu-memory-utilization: expected one argument` | bash 间接展开失败 | 已修复 (改用 eval) |
| Server 启动超时 | 模型太大 / 显存不足 | 查看 `benchmark_logs/server_*.log` |
| Profiling 目录为空 | 推理太快, 未来得及 flush | 增大 `--num-prompts` 或 `--output-len` |
