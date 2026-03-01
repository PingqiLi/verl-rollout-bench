# 004 Profiling 实验

## 目的

采集 Qwen3-30B-A3B decode 阶段的 **算子级** profiling 数据, 对比:
- BF16 单层 forward 各算子耗时
- W8A8 Dynamic 单层 forward 各算子耗时 (含 `npu_dynamic_quant`, `npu_grouped_matmul` 等)
- **确保在图模式 (ACLGraph) 下采集**, 反映真实推理路径

## Profiling 方法总览

本实验提供三种 profiling 方式, 适用于不同场景:

| | 单步采集 (profile_single_step.py) | 离线编排 (profile_offline.sh) | 在线 (profile_online.sh) |
|:---|:---|:---|:---|
| 工具 | `torch_npu.profiler` | 同左 (调用 profile_single_step.py) | `msserviceprofiler` |
| 触发方式 | `LLM.start_profile()` / `stop_profile()` | 自动遍历 bf16 + w8a8 | `SERVICE_PROF_CONFIG_PATH` |
| 采集范围 | **仅 1 prefill + 1 decode step** | 同左 | 按 timelimit 控制 |
| 数据量 | **~5-20 MB** (可控) | 同左 | 按配置 |
| 输出格式 | Trace JSON (Perfetto/TensorBoard) | 同左 | chrome_tracing.json + CSV |
| 算子级数据 | ✅ NPU kernel 级别 | 同左 | ✅ 需开启 `acl_task_time` |
| 图模式 | ✅ 默认 ACLGraph | 同左 | ✅ 默认 ACLGraph |
| **推荐场景** | **单精度快速采集** | **BF16 vs W8A8 一键对比** | 框架级调度分析 |

### 还有哪些方式?

| 方式 | 说明 | 何时使用 |
|:---|:---|:---|
| `VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1` | 轻量级: 输出 prepare/forward/postprocess 阶段耗时 | 快速确认 decode 各阶段占比 |
| `msprof` (CANN 原生) | 最底层: 直接包裹进程采集 NPU kernel trace | 需要 AiCore 利用率、L2 cache 等硬件指标 |
| `msserviceprofiler` + `acl_task_time=2` (MSPTI) | 按算子名过滤, 结构化 CSV 输出 | 需要过滤特定算子 (如只看 matmul) |

## 快速开始

### 方式 1: 单步 Profiling (推荐)

```bash
export MODEL_BASE="/data/l50044498/models"

# 直接使用 Python 脚本 (单精度, 最灵活)
python3 profile_single_step.py --model-key qwen3-30b-a3b --quant bf16
python3 profile_single_step.py --model-key qwen3-30b-a3b --quant w8a8

# 或用 shell 脚本一键采集 BF16 + W8A8 (推荐)
bash profile_offline.sh

# eager 模式 (对照)
bash profile_offline.sh --eager

# 只跑 BF16
bash profile_offline.sh --quants "bf16"

# 大 batch (模拟高并发 decode)
bash profile_offline.sh --batch-size 128
```

默认参数: batch_size=32, max_tokens=2 (1 prefill + 1 decode), warmup=3 步.
每个精度产生约 5-20 MB trace, 比旧方案 (GB 级) 小 100x+.

### 方式 2: 在线 Profiling

```bash
# 需要先安装 msserviceprofiler
pip install msserviceprofiler==1.2.2

export MODEL_BASE="/data/l50044498/models"
bash profile_online.sh

# 详细算子信息 (L1 级别)
bash profile_online.sh --acl-level L1
```

### 方式 3: 仅看阶段耗时 (最快)

不需要本目录的脚本, 直接设环境变量:

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

日志中会输出:
```
Profile execute duration [Decode]: [prepare input]:2.1ms [forward]:4.1ms [post process]:1.2ms
```

## 分析 Profiling 结果

### 自动分析

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

### 手动分析 (Perfetto UI)

1. 打开 https://ui.perfetto.dev/
2. 上传 `profile_bf16/` 或 `profile_w8a8/` 下的 `*.pt.trace.json` 文件
3. 展开 NPU stream, 可以看到每个 kernel 的执行时间
4. 单步采集无需手动排除 warmup (warmup 在 profiling 前完成)

### 手动分析 (MindStudio Insight)

在线 profiling 输出的 `chrome_tracing.json` 可用 MindStudio Insight 打开:
https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html

## 图模式说明

### 为什么要图模式?

vllm-ascend V1 Engine 默认启用 **ACLGraph** (图模式). 生产环境跑的就是图模式,
所以 profiling 也必须在图模式下采集, 才能反映真实的算子执行路径和耗时.

### 图模式下能看到算子吗?

**能**. ACLGraph 做的是 capture/replay:
- Capture 阶段: 记录算子执行序列
- Replay 阶段: 重放同一序列

`torch_npu.profiler` 在 **kernel 级别** 采集, 不受图模式影响.
每个 NPU kernel (如 `npu_grouped_matmul`, `npu_dynamic_quant`) 仍然独立可见.

### Eager vs Graph 对比

如果需要对比两种模式的算子差异:

```bash
# 图模式 (默认)
bash profile_offline.sh

# Eager 模式
bash profile_offline.sh --eager
```

图模式下可能会看到一些融合算子 (fused kernel), 而 eager 模式下是分开的.

## torch_npu.profiler 配置详解

离线脚本使用的 profiler 配置 (在 `vllm_ascend/worker/worker.py` 中):

```python
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=torch_npu.profiler.ExportType.Text,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,  # 详细级别
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    l2_cache=False,          # 不采集 L2 cache 指标 (减少开销)
    op_attr=False,           # 不采集算子属性
    data_simplification=False,
    record_op_args=False,    # 不记录算子参数
)
```

### ProfilerLevel

| 级别 | 说明 |
|:---|:---|
| Level0 | 仅 CPU/NPU 活动时间线 |
| **Level1** | + 算子级别耗时 (默认, 推荐) |
| Level2 | + 算子参数和属性 (数据量大) |

### 如何调整?

如需更详细的数据 (如 L2 cache 命中率, AiCore 指标), 需修改 `vllm-ascend` 源码中
`worker.py` 的 `_init_profiler()` 方法:

```python
# 启用 AiCore 指标
aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreAll,
# 启用 L2 cache 统计
l2_cache=True,
# 记录算子参数 (可看到 shape 信息)
record_op_args=True,
```

## msserviceprofiler 配置详解

在线脚本自动生成的配置 (`ms_service_profiler_config.json`):

```json
{
    "enable": 1,
    "prof_dir": "prof_data",
    "profiler_level": "INFO",
    "acl_task_time": 1,
    "acl_prof_task_time_level": "L0",
    "timelimit": 120,
    "domain": "Request;KVCache;ModelExecute;BatchSchedule"
}
```

### acl_task_time

| 值 | 说明 |
|:---|:---|
| 0 | 不采集算子耗时 |
| **1** | ACL 原生采集 (推荐, 低开销) |
| 2 | MSPTI-based (可过滤算子名, 需 `LD_PRELOAD=libmspti.so`) |

### acl_prof_task_time_level

| 值 | 说明 |
|:---|:---|
| **L0** | 仅 dispatch + execution 耗时 (低开销) |
| L1 | + AscendCL 接口性能 + 算子基础信息 |
| L1,10 | L1 级别, 采集 10 秒 |

### 使用 MSPTI 过滤特定算子

```json
{
    "acl_task_time": 2,
    "kernel_filter": "grouped_matmul;dynamic_quant;quant_matmul"
}
```

需要:
```bash
export LD_PRELOAD=$ASCEND_TOOLKIT_HOME/lib64/libmspti.so
```

## 输出目录结构

```
004_profiling/
└── outputs/
    └── 20260301_120000_graph/          # 一次离线 profiling
        ├── profile_bf16/                # BF16 trace 文件
        │   └── {worker}_{ts}_ascend_pt/
        │       └── *.pt.trace.json
        ├── profile_w8a8/                # W8A8 trace 文件
        │   └── {worker}_{ts}_ascend_pt/
        │       └── *.pt.trace.json
        ├── bf16_profile.log             # profiling 日志
        └── w8a8_profile.log
```

## 期望看到的关键算子

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

对比重点:
- W8A8D 额外的 `npu_dynamic_quant` 开销是否被融合算子的加速抵消
- 融合算子 `npu_grouped_matmul_swiglu_quant` vs 分离的 `npu_grouped_matmul` + `npu_swiglu`

## 注意事项

1. **Profiling 会拖慢推理**: 采集到的吞吐数据不能作为性能参考, 只看算子耗时分布
2. **数据量已可控**: 单步采集约 5-20 MB, 无需担心 GB 级 trace
3. **Warmup 自动处理**: `profile_single_step.py` 在 profiling 前自动 warmup (graph capture + JIT)
4. **一个进程一个精度**: 每种精度需要独立进程 (NPU 上下文限制), `profile_offline.sh` 自动处理
5. **Trace 格式**: Chrome Trace JSON, 可用 `json.load()` 直接解析, 或上传 Perfetto UI 可视化
