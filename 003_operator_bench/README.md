# 单算子 Benchmark: W8A8D vs BF16

## 文件结构

```
003_operator_bench/
├── shapes.py          # 模型 GEMM shape 配置 (从 HF config.json 提取)
├── bench_ops.py       # 单算子 benchmark 主脚本
└── analyze.py         # 结果分析与对比
```

## 依赖

- Python 3.10+
- PyTorch with Ascend NPU 支持
- `torch_npu` (提供 `npu_dynamic_quant`, `npu_grouped_matmul`, `npu_quant_matmul`)

## GEMM Shape

从 HuggingFace config.json 提取, Decode 阶段 (batch=256, 即 32 prompts × n=8):

| 算子 | 30B-A3B Shape | 718B Shape |
|---|---|---|
| Expert gate+up | **(16, 2048, 1536)** | **(8, 7680, 4096)** |
| Expert down | **(16, 768, 2048)** | **(8, 2048, 7680)** |
| Shared expert | — | **(256, 7680, 4096)** |

## 使用方式

### 1. 查看 shape 信息

```bash
python shapes.py
```

### 2. 运行 benchmark

```bash
# 测试所有模型所有 shape
python bench_ops.py --all --output results.json

# 只测 MoE expert 层
python bench_ops.py --model Qwen3-30B-A3B --category moe --output 30b_moe.json
python bench_ops.py --model Pangu-718B --category moe --output 718b_moe.json

# 只测 Attention 投影层
python bench_ops.py --model Pangu-718B --category attention

# 自定义 shape
python bench_ops.py --custom --M 16 --K 2048 --N 1536
python bench_ops.py --custom --M 8 --K 7680 --N 4096
```

### 3. Sweep M 维度

```bash
# 30B-A3B 的 MoE expert shape: K=2048, N=1536
python bench_ops.py --sweep-m --K 2048 --N 1536 --output sweep_30b.json

# 718B 的 MoE expert shape: K=7680, N=4096
python bench_ops.py --sweep-m --K 7680 --N 4096 --output sweep_718b.json
```

### 4. 分析结果

```bash
# 文本格式
python analyze.py results.json

# Markdown 格式
python analyze.py results.json --markdown

# 和实测值对比验证
python analyze.py 30b_moe.json --validate 0.89
```

### 5. 完整实验流程

```bash
# Step 1: 验证方法论 (30B-A3B)
python bench_ops.py --model Qwen3-30B-A3B --output results_30b.json
python analyze.py results_30b.json --validate 0.89

# Step 2: 外推预测 (718B)
python bench_ops.py --model Pangu-718B --output results_718b.json
python analyze.py results_718b.json

# Step 3: M-sweep 对比
python bench_ops.py --sweep-m --K 2048 --N 1536 --output sweep_30b.json
python bench_ops.py --sweep-m --K 7680 --N 4096 --output sweep_718b.json
```

## 参数说明

### bench_ops.py

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--all` | — | 测试所有模型所有 shape |
| `--model` | — | 指定模型 (`Qwen3-30B-A3B` / `Pangu-718B`) |
| `--sweep-m` | — | Sweep M 维度 |
| `--custom` | — | 自定义 shape |
| `--category` | `all` | 测试类别 (`attention` / `moe` / `dense_ffn`) |
| `--decode-batch` | `256` | decode 并发序列数 (32 prompts × n=8) |
| `--warmup` | `20` | warmup 迭代数 |
| `--repeats` | `100` | 计时迭代数 |
| `--output` | — | 输出 JSON 路径 |

### shapes.py

| 参数 | 说明 |
|---|---|
| `MODELS` | 模型架构字典, 可自行扩展 |
| `get_all_shapes(model, batch)` | 返回所有 GEMM shape |
| `get_moe_shapes(model, batch)` | 返回 MoE expert shape |
| `get_attention_shapes(model, batch)` | 返回 Attention 投影 shape |
