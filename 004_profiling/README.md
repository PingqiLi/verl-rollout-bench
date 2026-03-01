# 004 Profiling 实验

采集 Qwen3-30B-A3B decode 阶段的 **算子级** profiling 数据, 对比 BF16 vs W8A8D 各算子耗时.

## Profiling 方式

| | 单步采集 (profile_single_step.py) | 离线编排 (profile_offline.sh) | 在线 (profile_online.sh) |
|:---|:---|:---|:---|
| 工具 | `torch_npu.profiler` | 同左 (调用 profile_single_step.py) | `msserviceprofiler` |
| 采集范围 | **仅 1 prefill + 1 decode step** | 同左 | 按 timelimit 控制 |
| 数据量 | **~5-20 MB** | 同左 | 按配置 |
| **推荐场景** | **单精度快速采集** | **BF16 vs W8A8 一键对比** | 框架级调度分析 |

## 快速开始

### 方式 1: 单步 Profiling (推荐)

```bash
export MODEL_BASE="/data/l50044498/models"

# 直接使用 Python 脚本 (单精度)
python3 profile_single_step.py --model-key qwen3-30b-a3b --quant bf16
python3 profile_single_step.py --model-key qwen3-30b-a3b --quant w8a8

# 或用 shell 脚本一键采集 BF16 + W8A8
bash profile_offline.sh

# eager 模式 (对照)
bash profile_offline.sh --eager

# 只跑 BF16
bash profile_offline.sh --quants "bf16"

# 大 batch
bash profile_offline.sh --batch-size 128
```

默认参数: batch_size=32, max_tokens=2 (1 prefill + 1 decode), warmup=3 步.

### 方式 2: 在线 Profiling

```bash
pip install msserviceprofiler==1.2.2

export MODEL_BASE="/data/l50044498/models"
bash profile_online.sh

# 详细算子信息 (L1 级别)
bash profile_online.sh --acl-level L1
```

### 方式 3: 仅看阶段耗时 (最快)

```bash
VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1 \
VLLM_USE_V1=1 \
python3 -m vllm.entrypoints.cli.main bench throughput \
    --model ${MODEL_BASE}/Qwen3-30B-A3B-Instruct-2507 \
    --dataset-name random --input-len 1 --output-len 128 \
    --num-prompts 4 --n 8 --dtype bfloat16 \
    --tensor-parallel-size 4 --max-model-len 129 \
    --gpu-memory-utilization 0.8 --trust-remote-code
```

输出:
```
Profile execute duration [Decode]: [prepare input]:2.1ms [forward]:4.1ms [post process]:1.2ms
```

## 分析

```bash
# BF16 vs W8A8 算子对比
python3 analyze_profile.py outputs/<timestamp>_graph/

# 只看 MoE 相关算子
python3 analyze_profile.py outputs/<timestamp>_graph/ --filter moe

# 只看量化相关算子
python3 analyze_profile.py outputs/<timestamp>_graph/ --filter quant

# Markdown 输出
python3 analyze_profile.py outputs/<timestamp>_graph/ --markdown
```

手动分析: 上传 `profile_bf16/` 或 `profile_w8a8/` 下的 `*.pt.trace.json` 到 https://ui.perfetto.dev/

## 输出目录

```
004_profiling/
└── outputs/
    └── 20260301_120000_graph/
        ├── profile_bf16/
        │   └── {worker}_{ts}_ascend_pt/
        │       └── *.pt.trace.json
        ├── profile_w8a8/
        │   └── {worker}_{ts}_ascend_pt/
        │       └── *.pt.trace.json
        ├── bf16_profile.log
        └── w8a8_profile.log
```

## 关键算子

### BF16 MoE Forward (单层)

| 算子 | 作用 |
|:---|:---|
| `npu_grouped_matmul` (gmm1) | gate_up 投影 |
| `npu_swiglu` | 激活函数 |
| `npu_grouped_matmul` (gmm2) | down 投影 |

### W8A8D MoE Forward (单层)

| 算子 | 作用 |
|:---|:---|
| `npu_dynamic_quant` | 在线量化 activation (FP→INT8) |
| `npu_grouped_matmul_swiglu_quant` | 融合 gmm1 + swiglu + requant |
| `npu_grouped_matmul` (gmm2) | down 投影 (INT8 权重) |

## 注意事项

1. **Profiling 会拖慢推理**: 采集到的吞吐数据不能作为性能参考, 只看算子耗时分布
2. **一个进程一个精度**: 每种精度需要独立进程, `profile_offline.sh` 自动处理
3. **Trace 格式**: Chrome Trace JSON, 可用 `json.load()` 直接解析
