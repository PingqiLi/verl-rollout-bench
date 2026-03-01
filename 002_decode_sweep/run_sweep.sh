#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Decode 长度 Sweep 实验
#
# Qwen3-30B-A3B 上对比 BF16 vs W8A8D 在不同 decode 长度下的纯 decode 吞吐.
# input_len=1, output_len 从 256 扫到 16384, n=8.
#
# 自动检测 NPU 数量: 若设备数 >= 2×TP, 则同一 output_len 下
# 不同精度并行运行在不同设备组上 (如 bf16→0,1,2,3  w8a8→4,5,6,7).
#
# 用法:
#   bash run_sweep.sh                          # 默认参数
#   bash run_sweep.sh --num-prompts 8          # 减少 prompt 数
#   bash run_sweep.sh --output-lens "256 1024 4096"  # 自定义 sweep 范围
# ============================================================

# ======================== 颜色日志 ========================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${GREEN}▶${NC} $*"; }

# ======================== 路径 ========================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${REPO_ROOT}/config.yaml"
CONFIG_PARSER="${REPO_ROOT}/config_parser.py"

# 配置读取
cfg() {
    python3 "${CONFIG_PARSER}" "${CONFIG_FILE}" "$@"
}

# ======================== 默认参数 ========================
MODEL_KEY="qwen3-30b-a3b"
QUANTS=("bf16" "w8a8")
OUTPUT_LENS=(256 512 1024 2048 4096 8192 16384)
INPUT_LEN=1
NUM_PROMPTS=32
NUM_SAMPLES=8
GPU_MEM_UTIL_OVERRIDE=""
ASCEND_DEVICES=""   # 空 = 自动检测

# ======================== 参数解析 ========================
show_help() {
    cat <<EOF
用法: bash run_sweep.sh [选项]

Decode 长度 Sweep: Qwen3-30B-A3B 上 BF16 vs W8A8D 纯 decode 吞吐对比
input_len=1 (纯 decode), sweep output_len = 256..16384

自动检测 NPU 数量, 若设备数 >= 2×TP 则不同精度并行运行.
例: 8 NPU / tp=4 → bf16 和 w8a8 同时跑在不同设备组上, 耗时减半.

选项:
  --num-prompts N         prompt 数量 (默认: ${NUM_PROMPTS})
  -n N                    每 prompt 采样数 (默认: ${NUM_SAMPLES})
  --output-lens "L1 L2"   自定义 output_len 列表 (默认: "256 512 1024 2048 4096 8192 16384")
  --model-base DIR        模型根目录 (也可用 MODEL_BASE 环境变量)
  --gpu-mem-util F        覆盖 GPU 显存比例 (默认: 从 config.yaml 读取)
  --devices D             NPU 设备列表 (默认: 自动检测)
  -h, --help              显示帮助

注意:
  output_len 较大时 (>=4096), KV cache 需求显著增加.
  如遇 OOM, 用 --num-prompts 降低并发, 或 --gpu-mem-util 0.9 放宽显存限制.

示例:
  bash run_sweep.sh                                    # 全量 sweep (自动并行)
  bash run_sweep.sh --num-prompts 8 --gpu-mem-util 0.9 # 长序列友好配置
  bash run_sweep.sh --output-lens "256 1024 4096"      # 只跑三个点
  bash run_sweep.sh --devices 0,1,2,3                  # 手动指定 4 卡 (串行)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-prompts)    NUM_PROMPTS="$2";         shift 2 ;;
        -n)               NUM_SAMPLES="$2";         shift 2 ;;
        --output-lens)    IFS=' ' read -ra OUTPUT_LENS <<< "$2"; shift 2 ;;
        --model-base)     export MODEL_BASE="$2";   shift 2 ;;
        --gpu-mem-util)   GPU_MEM_UTIL_OVERRIDE="$2"; shift 2 ;;
        --devices)        ASCEND_DEVICES="$2";      shift 2 ;;
        -h|--help)        show_help; exit 0 ;;
        *)                log_error "未知参数: $1"; show_help; exit 1 ;;
    esac
done

# ======================== 配置加载 ========================
if [[ ! -f "${CONFIG_FILE}" ]]; then
    log_error "找不到配置文件: ${CONFIG_FILE}"
    exit 1
fi

MODEL_BASE="${MODEL_BASE:-$(cfg global model_base)}"
export MODEL_BASE

DISPLAY_NAME=$(cfg get "${MODEL_KEY}" display)
TP_SIZE=$(cfg get "${MODEL_KEY}" tp)
GPU_MEM_UTIL="${GPU_MEM_UTIL_OVERRIDE:-$(cfg get "${MODEL_KEY}" gpu_mem_util)}"

# 模型路径
get_model_path() {
    local quant="$1"
    cfg get-path "${MODEL_KEY}" "${quant}"
}

# ======================== NPU 自动检测 ========================

detect_devices() {
    # 优先使用用户指定的 --devices
    if [[ -n "${ASCEND_DEVICES}" ]]; then
        return
    fi

    # 1. 环境变量
    if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
        ASCEND_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}"
        return
    fi

    # 2. npu-smi 自动检测
    if command -v npu-smi &>/dev/null; then
        local npu_count
        npu_count=$(npu-smi info -l 2>/dev/null \
            | grep -oP 'Total Count\s*:\s*\K\d+' || echo "")
        if [[ -n "${npu_count}" && "${npu_count}" -gt 0 ]]; then
            ASCEND_DEVICES=$(seq -s, 0 $((npu_count - 1)))
            return
        fi
    fi

    # 3. /dev/davinci* 设备文件
    local dev_count
    dev_count=$(ls /dev/davinci[0-9]* 2>/dev/null | wc -l || echo "0")
    if [[ "${dev_count}" -gt 0 ]]; then
        ASCEND_DEVICES=$(seq -s, 0 $((dev_count - 1)))
        return
    fi

    # 兜底: 8 卡
    ASCEND_DEVICES="0,1,2,3,4,5,6,7"
}

# 根据设备数和 TP 计算并行 slot + 设备分组
# DEVICE_GROUPS[0]="0,1,2,3"  DEVICE_GROUPS[1]="4,5,6,7"
setup_parallelism() {
    detect_devices

    IFS=',' read -ra DEVICES_ARR <<< "${ASCEND_DEVICES}"
    NUM_DEVICES=${#DEVICES_ARR[@]}
    NUM_SLOTS=$((NUM_DEVICES / TP_SIZE))

    DEVICE_GROUPS=()
    for ((i = 0; i < NUM_SLOTS; i++)); do
        local start=$((i * TP_SIZE))
        local group=""
        for ((j = 0; j < TP_SIZE; j++)); do
            [[ -n "${group}" ]] && group+=","
            group+="${DEVICES_ARR[$((start + j))]}"
        done
        DEVICE_GROUPS+=("${group}")
    done

    # 并行数不超过 quant 数
    PARALLEL_SLOTS=${NUM_SLOTS}
    if [[ ${PARALLEL_SLOTS} -gt ${#QUANTS[@]} ]]; then
        PARALLEL_SLOTS=${#QUANTS[@]}
    fi
}

# ======================== 进程清理 ========================

# 递归杀掉进程树
kill_process_tree() {
    local pid="$1"
    local signal="${2:-TERM}"

    local children
    children=$(pgrep -P "$pid" 2>/dev/null || true)
    for child in $children; do
        kill_process_tree "$child" "$signal"
    done

    if kill -0 "$pid" 2>/dev/null; then
        kill -"$signal" "$pid" 2>/dev/null || true
    fi
}

# 清理残留的 vllm benchmark 进程
cleanup_stale_workers() {
    local pids
    pids=$(pgrep -f "vllm.entrypoints.cli.main bench" 2>/dev/null || true)
    if [[ -n "${pids}" ]]; then
        log_warn "清理残留 vllm 进程: ${pids}"
        for pid in ${pids}; do
            kill_process_tree "$pid" "KILL"
        done
        sleep 3
    fi
}

cleanup() {
    cleanup_stale_workers
}
trap cleanup EXIT

# ======================== 输出目录 ========================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${SCRIPT_DIR}/outputs/${TIMESTAMP}"
RESULT_DIR="${RUN_DIR}/results"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

# 保存实验配置
save_experiment_config() {
    cat > "${RUN_DIR}/experiment.json" <<EXPEOF
{
    "model": "${MODEL_KEY}",
    "display_name": "${DISPLAY_NAME}",
    "tp_size": ${TP_SIZE},
    "gpu_mem_util": ${GPU_MEM_UTIL},
    "input_len": ${INPUT_LEN},
    "output_lens": [$(IFS=,; echo "${OUTPUT_LENS[*]}")],
    "num_prompts": ${NUM_PROMPTS},
    "num_samples_per_prompt": ${NUM_SAMPLES},
    "quants": ["$(IFS='","'; echo "${QUANTS[*]}")"],
    "devices": "${ASCEND_DEVICES}",
    "parallel_slots": ${PARALLEL_SLOTS},
    "timestamp": "${TIMESTAMP}",
    "note": "ignore_eos=True (vllm 内部硬编码), 输出长度确定"
}
EXPEOF
}

# ======================== 单次 benchmark ========================
run_single() {
    local quant="$1"
    local output_len="$2"
    local devices="$3"

    local model_path
    model_path=$(get_model_path "${quant}")

    if [[ ! -d "${model_path}" ]]; then
        log_warn "跳过 ${quant}: 模型路径不存在 ${model_path}"
        return 1
    fi

    local max_model_len=$((INPUT_LEN + output_len))
    local result_file="${RESULT_DIR}/${quant}_olen${output_len}.json"
    local bench_log="${LOG_DIR}/${quant}_olen${output_len}.log"

    # 构建命令
    local bench_cmd="VLLM_USE_V1=1 ASCEND_RT_VISIBLE_DEVICES=${devices}"
    bench_cmd+=" python3 -m vllm.entrypoints.cli.main bench throughput"
    bench_cmd+=" --model ${model_path}"
    bench_cmd+=" --dataset-name random"
    bench_cmd+=" --input-len ${INPUT_LEN}"
    bench_cmd+=" --output-len ${output_len}"
    bench_cmd+=" --num-prompts ${NUM_PROMPTS}"
    bench_cmd+=" --n ${NUM_SAMPLES}"
    bench_cmd+=" --dtype bfloat16"
    bench_cmd+=" --tensor-parallel-size ${TP_SIZE}"
    bench_cmd+=" --max-model-len ${max_model_len}"
    bench_cmd+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"
    bench_cmd+=" --trust-remote-code"
    bench_cmd+=" --no-enable-chunked-prefill"
    bench_cmd+=" --output-json ${result_file}"

    if [[ "${quant}" != "bf16" ]]; then
        bench_cmd+=" --quantization ascend"
    fi

    if eval "${bench_cmd}" > "${bench_log}" 2>&1; then
        return 0
    else
        return 1
    fi
}

# ======================== 主循环 ========================
main() {
    setup_parallelism
    save_experiment_config

    log_info "=============================================="
    log_info "  Decode 长度 Sweep 实验"
    log_info "=============================================="
    log_info "模型: ${DISPLAY_NAME} (tp=${TP_SIZE}, gpu_mem=${GPU_MEM_UTIL})"
    log_info "参数: input_len=${INPUT_LEN}, n=${NUM_SAMPLES}, num_prompts=${NUM_PROMPTS}"
    log_info "Sweep: output_len = ${OUTPUT_LENS[*]}"
    log_info "精度: ${QUANTS[*]}"
    log_info "设备: ${ASCEND_DEVICES} (${NUM_DEVICES} NPU)"
    if [[ ${PARALLEL_SLOTS} -gt 1 ]]; then
        log_info "并行: ${PARALLEL_SLOTS} 组同时运行 (${NUM_DEVICES} NPU / tp=${TP_SIZE})"
        for i in "${!DEVICE_GROUPS[@]}"; do
            log_info "  slot ${i}: devices ${DEVICE_GROUPS[$i]}"
        done
    else
        log_info "串行: ${NUM_DEVICES} NPU / tp=${TP_SIZE}, 逐个运行"
    fi
    log_info "输出: ${RUN_DIR}"
    log_info "注: vllm 内部 ignore_eos=True, 输出长度严格确定"
    echo ""

    local total_runs=$(( ${#OUTPUT_LENS[@]} * ${#QUANTS[@]} ))
    local run_idx=0
    local failed_runs=()
    local succeeded_runs=()

    for output_len in "${OUTPUT_LENS[@]}"; do
        log_step "output_len = ${output_len}"

        if [[ ${PARALLEL_SLOTS} -gt 1 ]]; then
            # ---- 并行: 不同精度同时跑在不同设备组 ----
            local pids=()
            local tags=()
            local group_idx=0

            for quant in "${QUANTS[@]}"; do
                run_idx=$((run_idx + 1))
                local run_tag="${quant}_olen${output_len}"
                tags+=("${run_tag}")

                log_info "  [${run_idx}/${total_runs}] ${quant^^} → devices ${DEVICE_GROUPS[$group_idx]}"

                run_single "${quant}" "${output_len}" "${DEVICE_GROUPS[$group_idx]}" &
                pids+=($!)
                group_idx=$(( (group_idx + 1) % ${#DEVICE_GROUPS[@]} ))
            done

            # 等待所有并行任务完成
            for i in "${!pids[@]}"; do
                if wait "${pids[$i]}"; then
                    log_ok "  ${tags[$i]} 完成 (日志: ${LOG_DIR}/${tags[$i]}.log)"
                    succeeded_runs+=("${tags[$i]}")
                else
                    log_error "  ${tags[$i]} 失败 (日志: ${LOG_DIR}/${tags[$i]}.log)"
                    failed_runs+=("${tags[$i]}")
                fi
            done

            cleanup_stale_workers
        else
            # ---- 串行: 逐个运行 ----
            for quant in "${QUANTS[@]}"; do
                run_idx=$((run_idx + 1))
                local run_tag="${quant}_olen${output_len}"

                log_step "[${run_idx}/${total_runs}] ${DISPLAY_NAME} / ${quant^^} / output_len=${output_len}"
                log_info "  devices: ${DEVICE_GROUPS[0]}"

                if run_single "${quant}" "${output_len}" "${DEVICE_GROUPS[0]}"; then
                    log_ok "  ${run_tag} 完成"
                    succeeded_runs+=("${run_tag}")
                else
                    log_error "  ${run_tag} 失败 (日志: ${LOG_DIR}/${run_tag}.log)"
                    failed_runs+=("${run_tag}")
                fi

                cleanup_stale_workers
            done
        fi
    done

    # ======================== 汇总 ========================
    echo ""
    log_info "=============================================="
    log_info "  Sweep 完成"
    log_info "=============================================="
    log_info "成功: ${#succeeded_runs[@]} / ${total_runs}"

    if [[ ${#failed_runs[@]} -gt 0 ]]; then
        log_warn "失败的实验:"
        for f in "${failed_runs[@]}"; do
            log_warn "  - ${f}"
        done
    fi

    # 自动分析
    if [[ ${#succeeded_runs[@]} -gt 0 ]] && command -v python3 &>/dev/null; then
        log_step "生成分析报告..."
        python3 "${SCRIPT_DIR}/analyze_sweep.py" "${RESULT_DIR}" --input-len "${INPUT_LEN}"
        echo ""
        python3 "${SCRIPT_DIR}/analyze_sweep.py" "${RESULT_DIR}" --input-len "${INPUT_LEN}" --markdown \
            > "${RESULT_DIR}/summary.md" 2>/dev/null \
            && log_ok "Markdown 报告: ${RESULT_DIR}/summary.md"
        python3 "${SCRIPT_DIR}/analyze_sweep.py" "${RESULT_DIR}" --input-len "${INPUT_LEN}" --csv \
            > "${RESULT_DIR}/summary.csv" 2>/dev/null \
            && log_ok "CSV 报告: ${RESULT_DIR}/summary.csv"
    fi

    log_ok "结果目录: ${RUN_DIR}"
}

main
