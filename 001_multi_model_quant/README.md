# 多模型 × 多精度吞吐对比

## 文件

```
run_vllm_benchmark.sh      # 主控脚本: 实验编排、server 管理、profiling
summarize_benchmark.py     # 结果分析: JSON → 对比表格 (txt/csv/markdown)
```

## 快速开始

```bash
export MODEL_BASE="/data/l50044498/models"

# 诊断 (首次验证环境)
bash run_vllm_benchmark.sh --diagnostic

# 全量实验
bash run_vllm_benchmark.sh

# 查看结果
python3 summarize_benchmark.py outputs/<timestamp>/results --markdown
```

## 命令行参数

```
bash run_vllm_benchmark.sh [选项]

模式:
  --offline             [默认] 离线吞吐 benchmark (LLM.generate(), 支持 n=8)
  --online              在线服务 benchmark (延迟指标: TTFT/TPOT/ITL/E2EL)
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
| Qwen3-1.7B | 1 | 0.4 | 小模型 |
| Pangu-7B | 1 | 0.6 | 中模型 |
| Qwen3-30B-A3B | 4 | 0.8 | MoE 大模型, 无 W8A16 |

模型路径在 `config.yaml` 中配置, 缺失精度的权重路径自动跳过.

## 输出目录

```
outputs/
└── 20260209_201500/
    ├── results/
    │   ├── qwen3-1.7b_bf16.json
    │   ├── qwen3-1.7b_w8a16.json
    │   ├── summary.txt
    │   ├── summary.csv
    │   └── summary.md
    ├── logs/
    └── profiles/
```

## 故障排查

| 问题 | 解决 |
|:---|:---|
| `No available memory for cache blocks` | 手动 `kill` 残留 vllm 进程 |
| Server 启动超时 | 查看 `logs/server_*.log` |
| Profiling 目录为空 | 增大 `--num-prompts` 或 `--output-len` |
