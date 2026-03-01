#!/bin/bash
set -euo pipefail

# ======================== 模拟 GRPO Rollout 负载 ========================
# BF16 vs W8A8D, 混合 output_len 分布
# 每个 prompt 的 max_tokens 从分布中采样, 模拟真实 rollout

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/../config.yaml"
PARSER="${SCRIPT_DIR}/../config_parser.py"

# ======================== 日志 ========================
RED='\033[31m'; GREEN='\033[32m'; YELLOW='\033[33m'; BLUE='\033[34m'; NC='\033[0m'
log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ======================== 配置 ========================
cfg() { python3 "${PARSER}" "${CONFIG}" "$@"; }

# 默认值
MODEL_KEY="qwen3-30b-a3b"
QUANTS=("bf16" "w8a8")
INPUT_LEN=512
NUM_PROMPTS=32
NUM_SAMPLES=8
DIST="zipf"
MIN_TOKENS=64
MAX_TOKENS=2048
SEED=42
ASCEND_DEVICES=""
GPU_MEM_UTIL=""

# ======================== 参数解析 ========================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-key)      MODEL_KEY="$2"; shift 2 ;;
        --quants)         IFS=',' read -ra QUANTS <<< "$2"; shift 2 ;;
        --input-len)      INPUT_LEN="$2"; shift 2 ;;
        --num-prompts)    NUM_PROMPTS="$2"; shift 2 ;;
        --num-samples)    NUM_SAMPLES="$2"; shift 2 ;;
        --dist)           DIST="$2"; shift 2 ;;
        --min-tokens)     MIN_TOKENS="$2"; shift 2 ;;
        --max-tokens)     MAX_TOKENS="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --devices)        ASCEND_DEVICES="$2"; shift 2 ;;
        --gpu-mem-util)   GPU_MEM_UTIL="$2"; shift 2 ;;
        --model-base)     export MODEL_BASE="$2"; shift 2 ;;
        *)                log_error "未知参数: $1"; exit 1 ;;
    esac
done

# ======================== 读取模型配置 ========================
DISPLAY_NAME=$(cfg get "${MODEL_KEY}" display)
TP_SIZE=$(cfg get "${MODEL_KEY}" tp)
[[ -z "${GPU_MEM_UTIL}" ]] && GPU_MEM_UTIL=$(cfg get "${MODEL_KEY}" gpu_mem_util)

log_info "模型: ${DISPLAY_NAME} (tp=${TP_SIZE}, gpu_mem_util=${GPU_MEM_UTIL})"
log_info "负载: input=${INPUT_LEN}, prompts=${NUM_PROMPTS}, n=${NUM_SAMPLES}"
log_info "分布: ${DIST} [${MIN_TOKENS}, ${MAX_TOKENS}], seed=${SEED}"

# 解析模型路径
declare -A MODEL_PATHS
for q in "${QUANTS[@]}"; do
    path=$(cfg get-path "${MODEL_KEY}" "${q}" 2>/dev/null || echo "")
    if [[ -z "${path}" ]]; then
        log_warn "${q} 路径未配置, 跳过"
        continue
    fi
    MODEL_PATHS["${q}"]="${path}"
done

if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
    log_error "无可用模型路径"
    exit 1
fi

# ======================== NPU 设备检测 ========================
if [[ -z "${ASCEND_DEVICES}" ]]; then
    if command -v npu-smi &>/dev/null; then
        NUM_NPUS=$(npu-smi info -l 2>/dev/null | grep -c "NPU ID" || echo 0)
    else
        NUM_NPUS=8
        log_warn "npu-smi 不可用, 假设 ${NUM_NPUS} NPU"
    fi
    ASCEND_DEVICES=$(seq -s, 0 $((NUM_NPUS - 1)))
else
    NUM_NPUS=$(echo "${ASCEND_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
fi

IFS=',' read -ra DEV_ARRAY <<< "${ASCEND_DEVICES}"
PARALLEL_SLOTS=$((NUM_NPUS / TP_SIZE))

log_info "NPU: ${NUM_NPUS} 卡, tp=${TP_SIZE}, 可并行 ${PARALLEL_SLOTS} 组"

# ======================== 进程清理 ========================
kill_process_tree() {
    local pid=$1
    local children
    children=$(pgrep -P "$pid" 2>/dev/null || true)
    for child in $children; do
        kill_process_tree "$child"
    done
    kill -9 "$pid" 2>/dev/null || true
}

cleanup_vllm() {
    local pids
    pids=$(pgrep -f "rollout_bench.py" 2>/dev/null || true)
    for pid in $pids; do
        kill_process_tree "$pid"
    done
    sleep 1
}

trap cleanup_vllm EXIT

# ======================== 输出目录 ========================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${SCRIPT_DIR}/outputs/${TIMESTAMP}"
mkdir -p "${RUN_DIR}/results" "${RUN_DIR}/logs"

# 保存实验配置
cat > "${RUN_DIR}/experiment.json" <<EOF
{
    "model": "${MODEL_KEY}",
    "display_name": "${DISPLAY_NAME}",
    "tp_size": ${TP_SIZE},
    "gpu_mem_util": ${GPU_MEM_UTIL},
    "input_len": ${INPUT_LEN},
    "num_prompts": ${NUM_PROMPTS},
    "num_samples": ${NUM_SAMPLES},
    "dist": "${DIST}",
    "min_tokens": ${MIN_TOKENS},
    "max_tokens": ${MAX_TOKENS},
    "seed": ${SEED},
    "quants": [$(printf '"%s", ' "${QUANTS[@]}" | sed 's/, $//')],
    "devices": "${ASCEND_DEVICES}",
    "parallel_slots": ${PARALLEL_SLOTS},
    "timestamp": "${TIMESTAMP}"
}
EOF

# ======================== Benchmark ========================
BENCH_ARGS=(
    --tp "${TP_SIZE}"
    --gpu-mem-util "${GPU_MEM_UTIL}"
    --input-len "${INPUT_LEN}"
    --num-prompts "${NUM_PROMPTS}"
    --num-samples "${NUM_SAMPLES}"
    --dist "${DIST}"
    --min-tokens "${MIN_TOKENS}"
    --max-tokens "${MAX_TOKENS}"
    --seed "${SEED}"
)

# 分配设备组
get_device_group() {
    local group_idx=$1
    local start=$((group_idx * TP_SIZE))
    local end=$((start + TP_SIZE - 1))
    local devices=""
    for i in $(seq "$start" "$end"); do
        [[ -n "$devices" ]] && devices+=","
        devices+="${DEV_ARRAY[$i]}"
    done
    echo "$devices"
}

run_single() {
    local quant=$1
    local model_path=$2
    local devices=$3
    local result_file="${RUN_DIR}/results/${quant}.json"
    local log_file="${RUN_DIR}/logs/${quant}.log"

    log_info "[${quant}] 开始 (devices=${devices})"

    ASCEND_RT_VISIBLE_DEVICES="${devices}" \
    python3 "${SCRIPT_DIR}/rollout_bench.py" \
        --model "${model_path}" \
        "${BENCH_ARGS[@]}" \
        --output "${result_file}" \
        2>&1 | tee "${log_file}"

    local exit_code=$?
    if [[ ${exit_code} -eq 0 ]] && [[ -f "${result_file}" ]]; then
        log_ok "[${quant}] 完成"
        return 0
    else
        log_error "[${quant}] 失败 (exit=${exit_code})"
        return 1
    fi
}

# ======================== 执行 ========================
FAILED=0

if [[ ${PARALLEL_SLOTS} -ge 2 ]] && [[ ${#MODEL_PATHS[@]} -ge 2 ]]; then
    log_info "并行模式: ${PARALLEL_SLOTS} 组 NPU"
    PIDS=()
    group_idx=0
    for q in "${QUANTS[@]}"; do
        [[ -z "${MODEL_PATHS[$q]+x}" ]] && continue
        devices=$(get_device_group $group_idx)
        run_single "$q" "${MODEL_PATHS[$q]}" "$devices" &
        PIDS+=($!)
        group_idx=$((group_idx + 1))
        [[ $group_idx -ge $PARALLEL_SLOTS ]] && break
    done
    for pid in "${PIDS[@]}"; do
        wait "$pid" || FAILED=$((FAILED + 1))
    done
else
    log_info "串行模式"
    default_devices=$(get_device_group 0)
    for q in "${QUANTS[@]}"; do
        [[ -z "${MODEL_PATHS[$q]+x}" ]] && continue
        run_single "$q" "${MODEL_PATHS[$q]}" "$default_devices" \
            || FAILED=$((FAILED + 1))
        cleanup_vllm
    done
fi

# ======================== 分析 ========================
result_count=$(find "${RUN_DIR}/results" -name "*.json" | wc -l | tr -d ' ')
if [[ ${result_count} -ge 1 ]]; then
    log_info "分析结果..."
    python3 "${SCRIPT_DIR}/analyze_rollout.py" "${RUN_DIR}/results"
    python3 "${SCRIPT_DIR}/analyze_rollout.py" "${RUN_DIR}/results" --markdown \
        > "${RUN_DIR}/results/summary.md"
    log_ok "报告: ${RUN_DIR}/results/summary.md"
fi

echo ""
log_ok "完成! 结果目录: ${RUN_DIR}"
[[ ${FAILED} -gt 0 ]] && log_warn "${FAILED} 个 benchmark 失败"
exit ${FAILED}
