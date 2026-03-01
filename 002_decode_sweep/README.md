# 模拟 GRPO Rollout 负载

在 Qwen3-30B-A3B 上对比 **BF16** 与 **W8A8D** 在混合长度 rollout 负载下的吞吐.
每个 prompt 生成不同长度的回复, 模拟真实 rollout 场景.

## 核心参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| input_len | 512 | 模拟真实 prompt 长度 |
| num_prompts | 32 | 并发 prompt 数 |
| n | 8 | GRPO 多采样 |
| dist | zipf | output_len 分布 (大部分短, 少量长) |
| min_tokens | 64 | 最短输出 |
| max_tokens | 2048 | 最长输出 |
| ignore_eos | True | 输出长度严格确定 |

## 使用

### 快速开始

```bash
export MODEL_BASE="/data/l50044498/models"
bash run_rollout_bench.sh
```

自动检测 NPU 数量: 若设备数 ≥ 2×TP, 不同精度并行跑在不同设备组上.

### 自定义参数

```bash
# 均匀分布
bash run_rollout_bench.sh --dist uniform

# 双峰分布 (70% 短 + 30% 长)
bash run_rollout_bench.sh --dist bimodal

# 更长序列范围
bash run_rollout_bench.sh --min-tokens 128 --max-tokens 4096

# 减少并发, 放宽显存
bash run_rollout_bench.sh --num-prompts 8 --gpu-mem-util 0.9

# 手动指定 4 卡
bash run_rollout_bench.sh --devices 0,1,2,3
```

### 直接用 Python 脚本

```bash
# 单次 benchmark (不走 shell 编排)
python3 rollout_bench.py --model /path/to/model --tp 4 --output result.json

# 从 config.yaml 读配置
python3 rollout_bench.py --model-key qwen3-30b-a3b --quant bf16 --output bf16.json
python3 rollout_bench.py --model-key qwen3-30b-a3b --quant w8a8 --output w8a8.json
```

### 独立分析

```bash
python3 analyze_rollout.py outputs/<timestamp>/results
python3 analyze_rollout.py outputs/<timestamp>/results --markdown
```

## 输出目录

```
002_decode_sweep/
└── outputs/
    └── 20260301_120000/
        ├── experiment.json
        ├── results/
        │   ├── bf16.json
        │   ├── w8a8.json
        │   └── summary.md
        └── logs/
            ├── bf16.log
            └── w8a8.log
```

## 分布说明

| 分布 | 特点 |
|:---|:---|
| zipf | 大部分短回复, 少量长回复 (最接近真实 rollout) |
| uniform | 均匀分布 (基线对比) |
| bimodal | 70% 短 + 30% 长 (模拟混合任务) |

## 注意事项

- 所有精度使用相同 seed → 相同 output_len 分布, 保证公平对比
- max_tokens ≥ 4096 时, 如遇 OOM, 用 `--num-prompts 8` 或 `--gpu-mem-util 0.9`
- 每次跑完自动清理残留 vllm worker 进程
