# 单算子 Benchmark: W8A8D vs BF16 MoE Decode 性能分析

## 背景

在 verl GRPO 训练中，vLLM 作为 rollout 推理后端。实测发现 W8A8 Dynamic 量化在
Qwen3-30B-A3B (MoE) 上比 BF16 **慢约 11%** (speedup = 0.89x)，而在更大的 Dense 模型上
(Qwen3-32B: 1.09x, Qwen2.5-72B: 1.17x) 有正收益。

**核心假设**: 30B-A3B 的 MoE expert GEMM shape 太小 (hidden=2048, intermediate=768)，
`npu_dynamic_quant` 的额外开销吃掉了 INT8 matmul 的带宽收益。在更大的 MoE 模型
(盘古 718B, hidden=7680, intermediate=2048) 上，shape 更大，W8A8D 应能获得正收益。

**本实验目的**: 通过单算子级 benchmark 证明上述假设。

## 实验方法

### 算子对比

MoE 层走 `npu_grouped_matmul` 路径 (非 Dense 的 `npu_quant_matmul`)，
Attention 投影层走 Dense 路径。

**MoE Expert 层 (grouped_matmul):**

```
BF16 路径 (per MoE layer):
  npu_grouped_matmul(x_bf16, w_bf16, group_list)       # gmm1: gate+up
  npu_swiglu(out)
  npu_grouped_matmul(activated, w_bf16, group_list)     # gmm2: down

W8A8D 路径 (per MoE layer):
  npu_dynamic_quant(x_bf16) → x_int8, pertoken_scale   # 额外开销
  npu_grouped_matmul(x_int8, w_int8, scale, group_list) # gmm1 (INT8)
  npu_swiglu(out)
  npu_grouped_matmul(activated_int8, w_int8, scale, group_list)  # gmm2 (INT8)

注: 实际部署中 gmm1+swiglu+requant 可能融合为 npu_grouped_matmul_swiglu_quant,
    本实验拆开测量以分析各算子贡献。
```

**Attention 投影层 (quant_matmul):**

```
BF16 路径: torch.matmul(x_bf16, w_bf16)
W8A8D 路径: npu_dynamic_quant(x) → npu_quant_matmul(x_int8, w_int8, scale)
```

### GEMM Shape 来源

从 HuggingFace config.json 提取的精确参数:

| 参数 | Qwen3-30B-A3B | 盘古 718B |
|---|---|---|
| hidden_size | 2048 | 7680 |
| moe_intermediate_size | 768 | 2048 |
| num_experts / top_k | 128 / 8 | 256 / 8 |
| num_hidden_layers | 48 | 61 |
| attention | GQA (32/4) | MLA (q_lora=1536, kv_lora=512) |
| shared expert | 无 | 1 |

Decode 阶段 (batch=256, 即 32 prompts × n=8):

| 算子 | 30B-A3B Shape | 718B Shape |
|---|---|---|
| Expert gate+up | **(16, 2048, 1536)** | **(8, 7680, 4096)** |
| Expert down | **(16, 768, 2048)** | **(8, 2048, 7680)** |
| Shared expert | — | **(256, 7680, 4096)** |

其中 M = batch × top_k / num_experts (每个 expert 平均分到的 token 数)。

### 论证链

```
Phase 1: 验证方法论
  单算子 benchmark (30B-A3B shape) → 预测 speedup
  vs 整模型实测 0.89x → 偏差 < 5% 则方法可信

Phase 2: 外推结论
  单算子 benchmark (718B shape) → 预测 speedup > 1.0
  → 证明更大 MoE shape 下 W8A8D 有正收益

Phase 3 (可选): 交叉验证
  718B profiling → 验证算子时间分布与单算子预测一致
```

## 文件结构

```
operator_bench/
├── README.md          # 本文件
├── shapes.py          # 模型 shape 配置 (从 HF config.json 提取)
├── bench_ops.py       # 单算子 benchmark 主脚本
└── analyze.py         # 结果分析与对比
```

## 依赖

- Python 3.10+
- PyTorch with Ascend NPU 支持
- `torch_npu` (提供 `npu_dynamic_quant`, `npu_grouped_matmul`, `npu_quant_matmul`)

## 使用方式

### 1. 查看 shape 信息

```bash
python shapes.py
```

输出两个模型 decode 阶段所有线性层的 GEMM shape。

### 2. 运行 benchmark

```bash
# 测试所有模型所有 shape
python bench_ops.py --all --output results.json

# 只测 MoE expert 层
python bench_ops.py --model Qwen3-30B-A3B --category moe --output 30b_moe.json
python bench_ops.py --model Pangu-718B --category moe --output 718b_moe.json

# 只测 Attention 投影层
python bench_ops.py --model Pangu-718B --category attention

# 自定义 shape (快速验证)
python bench_ops.py --custom --M 16 --K 2048 --N 1536
python bench_ops.py --custom --M 8 --K 7680 --N 4096
```

### 3. Sweep M 维度 (关键实验)

找出 W8A8D vs BF16 的 breakeven M 值:

```bash
# 30B-A3B 的 MoE expert shape: K=2048, N=1536
python bench_ops.py --sweep-m --K 2048 --N 1536 --output sweep_30b.json

# 718B 的 MoE expert shape: K=7680, N=4096
python bench_ops.py --sweep-m --K 7680 --N 4096 --output sweep_718b.json
```

这个实验最关键——能直观展示 shape 大小如何影响 W8A8D 的收益。
预期结果: 30B shape 在 M=16 时 speedup < 1, 718B shape 在 M=8 时 speedup > 1。

### 4. 分析结果

```bash
# 文本格式
python analyze.py results.json

# Markdown 格式 (可直接贴报告)
python analyze.py results.json --markdown

# 和实测值对比验证
python analyze.py 30b_moe.json --validate 0.89
```

### 5. 完整实验流程 (推荐)

```bash
# Step 1: 验证方法论 (30B-A3B)
python bench_ops.py --model Qwen3-30B-A3B --output results_30b.json
python analyze.py results_30b.json --validate 0.89

# Step 2: 外推预测 (718B)
python bench_ops.py --model Pangu-718B --output results_718b.json
python analyze.py results_718b.json

# Step 3: M-sweep 对比 (可视化 breakeven)
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

## 注意事项

1. **MoE 用 `npu_grouped_matmul`, 不是 `npu_quant_matmul`**
   Dense 模型的线性层走 `npu_quant_matmul` (标准 matmul), MoE expert 层走
   `npu_grouped_matmul` (带 group_list 的分组 matmul), 两者是不同算子。

2. **融合算子未直接测量**
   实际部署中 gmm1+swiglu+requant 大概率走融合路径 (`npu_grouped_matmul_swiglu_quant`),
   本实验拆开测量以分离各算子贡献。融合路径的实际性能应优于拆开测量的总和。
   因此本实验的 W8A8D 预测是**保守估计** (实际更快)。

3. **EP 通信开销不影响对比结论**
   718B 实际部署走 EP (Expert Parallelism), 有 all-to-all 通信开销, 但
   BF16 和 W8A8D 的通信量和模式相同, 不改变计算部分的相对性能对比。

4. **Profiling 互补验证**
   如果能拿到 30B-A3B 的 BF16 和 W8A8D profiling 数据, 可与单算子结果交叉验证,
   进一步确认方法论可靠性。718B 的 profiling (如有) 可作为最终结论的额外证据。
