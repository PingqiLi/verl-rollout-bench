# 多模型 × 多精度吞吐对比

## 文件

```
run_vllm_benchmark.sh      # 主控脚本: 实验编排、server 管理、profiling
summarize_benchmark.py     # 结果分析: JSON → 对比表格 (txt/csv/markdown)
plot_benchmark.py          # 散点图 + 加速比表格 (matplotlib)
```

## 快速开始

```bash
export MODEL_BASE="/data/l50044498/models"

# 诊断 (首次验证环境)
bash run_vllm_benchmark.sh --diagnostic

# W8A8D 加速比实验 (4 模型 × 2 精度 × 3 轮, 一键完成)
bash run_vllm_benchmark.sh \
    --models qwen3-1.7b,qwen3-30b-a3b,pangu-7b,qwen3-32b \
    --quants bf16,w8a8 \
    --repeats 3

# 全量实验 (所有模型 × 所有精度)
bash run_vllm_benchmark.sh
```

实验结束后自动生成:
- `outputs/<timestamp>/results/summary.md` — 汇总表格
- `outputs/<timestamp>/results/speedup_table.md` — 加速比表格 (mean ± std)
- `outputs/<timestamp>/benchmark_scatter.pdf` — 散点图

## 命令行参数

```
bash run_vllm_benchmark.sh [选项]

模式:
  --offline             [默认] 离线吞吐 benchmark (LLM.generate(), 支持 n=8)
  --online              在线服务 benchmark (延迟指标: TTFT/TPOT/ITL/E2EL)
  --diagnostic          快速诊断 (BF16 + Qwen3-1.7B, 4 prompts)

实验选择:
  --models M1,M2,...    模型列表 (qwen3-1.7b, pangu-7b, qwen3-30b-a3b, qwen3-32b)
  --quants Q1,Q2,...    精度列表 (bf16, w8a16, w8a8)
  --repeats N           每组实验重复次数 (默认: 1)

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
# W8A8D 加速比实验 (推荐, 一键完成: 实验 + 表格 + 散点图)
bash run_vllm_benchmark.sh \
    --models qwen3-1.7b,qwen3-30b-a3b,pangu-7b,qwen3-32b \
    --quants bf16,w8a8 \
    --repeats 3

# 只对比 Qwen3-1.7B 的 BF16 和 W8A16
bash run_vllm_benchmark.sh --models qwen3-1.7b --quants bf16,w8a16

# 快速小规模测试
bash run_vllm_benchmark.sh --num-prompts 4 -n 2 --no-profile

# 独立峰值性能 (不模拟 verl 显存约束)
bash run_vllm_benchmark.sh --gpu-mem-util 0.9

# 单独生成散点图 (用已有结果)
python3 plot_benchmark.py outputs/<timestamp>/results/ \
    -o benchmark.pdf --table
```

## 模型配置

| 模型 | TP | gpu_memory_utilization | 备注 |
|:---|:---|:---|:---|
| Qwen3-1.7B | 1 | 0.4 | 小模型 |
| Pangu-7B | 1 | 0.6 | 中模型, Dense |
| Qwen3-30B-A3B | 4 | 0.8 | MoE 大模型, 无 W8A16 |
| Qwen3-32B | 4 | 0.9 | Dense 大模型 |

模型路径在 `config.yaml` 中配置, 缺失精度的权重路径自动跳过.

## 输出目录

```
outputs/
└── 20260209_201500/
    ├── benchmark_scatter.pdf          # 散点图
    ├── results/
    │   ├── qwen3-1.7b_bf16_run1.json  # 重复实验结果
    │   ├── qwen3-1.7b_bf16_run2.json
    │   ├── qwen3-1.7b_bf16_run3.json
    │   ├── qwen3-1.7b_w8a8_run1.json
    │   ├── ...
    │   ├── speedup_table.md           # 加速比表格
    │   ├── summary.txt
    │   ├── summary.csv
    │   └── summary.md
    └── logs/
```

## 故障排查

| 问题 | 解决 |
|:---|:---|
| `No available memory for cache blocks` | 手动 `kill` 残留 vllm 进程 |
| Server 启动超时 | 查看 `logs/server_*.log` |
| Profiling 目录为空 | 增大 `--num-prompts` 或 `--output-len` |
| 散点图生成失败 | `pip install matplotlib numpy` |
