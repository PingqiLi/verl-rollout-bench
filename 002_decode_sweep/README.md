# Decode 长度 Sweep 实验

## 目的

在 Qwen3-30B-A3B 上对比 **BF16** 与 **W8A8 Dynamic** 在不同 decode 长度下的纯 decode 吞吐, 验证:

1. W8A8D 的加速比 (或回退) 随 decode 长度如何变化
2. GRPO rollout 中长尾 decode (困难问题 → 极长回复) 是否放大了 W8A8D 的性能差异

## 实验设计

### 核心参数

| 参数 | 值 | 说明 |
|:---|:---|:---|
| input_len | 1 | 消除 prefill, 近似纯 decode |
| output_len | 256, 512, 1024, 2048, 4096, 8192, 16384 | 覆盖短回复到长尾 |
| n | 8 | GRPO 多采样 (每 prompt 8 条回复) |
| num_prompts | 32 | 总并发 = 32 × 8 = 256 序列 |
| ignore_eos | True | vllm 内部硬编码, 输出长度严格确定 |

### 为什么 input_len=1

`input_len=1` 使 prefill 仅处理 1 个 token, 耗时可忽略. 整个 benchmark 时间 ≈ 纯 decode 时间.

### 为什么 sweep output_len

GRPO 训练中, 模型对不同难度的问题生成不同长度的回复:
- 简单问题 → 短回复 (~256 token)
- 困难问题 → 长回复 (数千 token)
- **一个 batch 的完成时间取决于最慢的那条序列 (长尾)**

如果 W8A8D 在长 decode 上回退更严重, 长尾序列就是拖慢整体训练的元凶.

### 与 GRPO Rollout 的对应关系

| | 真实 GRPO | 本实验 |
|:---|:---|:---|
| 调用方式 | `LLM.generate()` | ✅ `vllm bench throughput` (相同路径) |
| 多采样 | n=8 | ✅ `--n 8` |
| GPU 显存 | actor + rollout 共享 | ✅ `gpu_mem_util=0.8` |
| 输入长度 | 可变 (真实 prompt) | 固定 1 (隔离 decode) |
| 输出长度 | 可变 (到 EOS 或 max_tokens) | 固定 (逐点扫描, 覆盖分布) |

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

# CSV (可导入 Excel / 画图)
python3 analyze_sweep.py outputs/<timestamp>/results --csv
```

## 输出示例

```
======================================================================
  Decode 长度 Sweep: BF16 vs W8A8D 吞吐对比 (input_len=1, 纯 decode)
======================================================================

  output_len  BF16 (tok/s)  W8A8 (tok/s)        加速比  BF16 耗时(s)  W8A8 耗时(s)
--------------------------------------------------------------------------------
         256         593.4         529.7     0.893x ▼          13.8          15.5
         512         ...           ...       ...               ...           ...
        1024         ...           ...       ...               ...           ...
        ...

趋势: 加速比从 0.893x → 0.xxx (下降/上升 0.xxx)
```

## 输出目录

```
decode_sweep/
└── outputs/
    └── 20260301_120000/
        ├── experiment.json          # 实验配置记录
        ├── results/
        │   ├── bf16_olen256.json    # 各数据点的原始 JSON
        │   ├── w8a8_olen256.json
        │   ├── bf16_olen512.json
        │   ├── ...
        │   ├── summary.md           # 自动生成的 Markdown 报告
        │   └── summary.csv          # 自动生成的 CSV
        └── logs/
            ├── bf16_olen256.log     # benchmark 运行日志
            └── ...
```

## 注意事项

### KV Cache 内存

output_len 越大, KV cache 需求越高:

| output_len | 单序列 KV cache (估算, tp=4) | 256 序列总量 |
|---:|---:|---:|
| 256 | ~6 MB | ~1.5 GB |
| 1024 | ~24 MB | ~6 GB |
| 4096 | ~96 MB | ~24 GB |
| 16384 | ~384 MB | ~96 GB |

> output_len ≥ 4096 时, 如遇 OOM, 用 `--num-prompts 8` 或 `--gpu-mem-util 0.9`.
> vLLM 不会直接崩溃, 而是自动降低并发 (分批调度), 但实际 batch size 变小会影响量化加速效果的体现.

### 实验耗时

每个数据点 (一次 benchmark) 约需 1-5 分钟. 全量 sweep (7 × 2 = 14 次) 约需 30-60 分钟.
