# verl-rollout-bench

在 Ascend NPU 上对比 BF16 / W8A16 / W8A8 Dynamic 低精度量化对 vLLM 推理吞吐的加速效果,
面向 verl RL GRPO rollout 场景.

## 目标

证明 W8A8 Dynamic 量化在更大规模 MoE 模型 (如 Pangu 718B) 上能有效提升 rollout 效率,
尽管在 Qwen3-30B-A3B 上存在 ~11% 的回退.

## 实验列表

| 编号 | 目录 | 说明 |
|:---|:---|:---|
| 001 | `001_multi_model_quant/` | 多模型 × 多精度吞吐对比 (原始实验) |
| 002 | `002_decode_sweep/` | Decode 长度 Sweep: 不同 output_len 下 BF16 vs W8A8D 加速比 |
| 003 | `003_operator_bench/` | 单算子 Benchmark: 按真实 GEMM shape 对比 BF16 vs W8A8D |
| 004 | `004_profiling/` | Profiling 采集: 算子级耗时对比 (torch_npu / msserviceprofiler) |

## 共享配置

```
config.yaml          # 模型配置 (路径、TP、gpu_mem_util 等)
config_parser.py     # YAML 配置解析器 (供 shell 脚本调用)
```

所有实验脚本通过 `${SCRIPT_DIR}/../config.yaml` 引用共享配置.

## 快速开始

```bash
export MODEL_BASE="/data/l50044498/models"

# 实验 001: 多模型吞吐对比
bash 001_multi_model_quant/run_vllm_benchmark.sh

# 实验 002: Decode 长度 Sweep
bash 002_decode_sweep/run_sweep.sh

# 实验 004: Profiling 采集
bash 004_profiling/profile_offline.sh
```

## 依赖

- Python 3.10+ 、`pyyaml`
- vLLM (with Ascend NPU support, v0.13.0+)
- 实验 003 需要 `torch_npu`
- 实验 004 在线模式需要 `msserviceprofiler==1.2.2`
