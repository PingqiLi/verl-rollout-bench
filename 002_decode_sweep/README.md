# Decode 长度 Sweep 实验

在 Qwen3-30B-A3B 上对比 **BF16** 与 **W8A8D** 在不同 decode 长度下的纯 decode 吞吐.

## 核心参数

| 参数 | 值 | 说明 |
|:---|:---|:---|
| input_len | 1 | 消除 prefill, 近似纯 decode |
| output_len | 256, 512, 1024, 2048, 4096, 8192, 16384 | 覆盖短回复到长尾 |
| n | 8 | GRPO 多采样 |
| num_prompts | 32 | 总并发 = 32 × 8 = 256 序列 |
| ignore_eos | True | vllm 内部硬编码, 输出长度严格确定 |

## 使用

### 快速开始

```bash
export MODEL_BASE="/data/l50044498/models"
bash run_sweep.sh
```

### 自定义参数

```bash
# 长序列友好 (减少并发, 放宽显存)
bash run_sweep.sh --num-prompts 8 --gpu-mem-util 0.9

# 只跑部分数据点
bash run_sweep.sh --output-lens "256 1024 4096 16384"

# 指定模型目录
bash run_sweep.sh --model-base /path/to/models
```

### 独立分析

```bash
# 文本表格
python3 analyze_sweep.py outputs/<timestamp>/results

# Markdown
python3 analyze_sweep.py outputs/<timestamp>/results --markdown

# CSV
python3 analyze_sweep.py outputs/<timestamp>/results --csv
```

## 输出目录

```
002_decode_sweep/
└── outputs/
    └── 20260301_120000/
        ├── experiment.json
        ├── results/
        │   ├── bf16_olen256.json
        │   ├── w8a8_olen256.json
        │   ├── ...
        │   ├── summary.md
        │   └── summary.csv
        └── logs/
            ├── bf16_olen256.log
            └── ...
```

## 注意事项

- output_len ≥ 4096 时, 如遇 OOM, 用 `--num-prompts 8` 或 `--gpu-mem-util 0.9`
- 每个数据点约需 1-5 分钟, 全量 sweep (7 × 2 = 14 次) 约需 30-60 分钟
