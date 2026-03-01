#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Decode 长度 Sweep 实验
#
# 目的: 在 Qwen3-30B-A3B 上对比 BF16 vs W8A8D 在不同 decode 长度下
#       的纯 decode 吞吐, 验证长尾 decode 对量化加速比的影响.
#
# 方法:
#   - input_len=1: 消除 prefill, 近似纯 decode
#   - output_len 从 256 扫到 16384: 覆盖 GRPO 中短回复到长尾回复
#   - ignore_eos: vllm bench throughput 内部已硬编码 ignore_eos=True,
#     输出长度严格等于 output_len, 不会提前终止
#   - n=8: 模拟 GRPO rollout 的多采样
#
# 用法:
#   bash run_sweep.sh                          # 默认参数
#   bash run_sweep.sh --num-prompts 8          # 减少 prompt 数 (长序列防 OOM)
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
ASCEND_DEVICES="0,1,2,3,4,5,6,7"

# ======================== 参数解析 ========================
show_help() {
    cat <<EOF
用法: bash run_sweep.sh [选项]

Decode 长度 Sweep: Qwen3-30B-A3B 上 BF16 vs W8A8D 纯 decode 吞吐对比
input_len=1 (纯 decode), sweep output_len = 256..16384

选项:
  --num-prompts N         prompt 数量 (默认: ${NUM_PROMPTS})
  -n N                    每 prompt 采样数 (默认: ${NUM_SAMPLES})
  --output-lens "L1 L2"   自定义 output_len 列表 (默认: "256 512 1024 2048 4096 8192 16384")
  --model-base DIR        模型根目录 (也可用 MODEL_BASE 环境变量)
  --gpu-mem-util F        覆盖 GPU 显存比例 (默认: 从 config.yaml 读取)
  --devices D             Ascend 设备列表 (默认: ${ASCEND_DEVICES})
  -h, --help              显示帮助

注意:
  output_len 较大时 (>=4096), KV cache 需求显著增加.
  如遇 OOM, 用 --num-prompts 降低并发, 或 --gpu-mem-util 0.9 放宽显存限制.

示例:
  bash run_sweep.sh                                    # 全量 sweep
  bash run_sweep.sh --num-prompts 8 --gpu-mem-util 0.9 # 长序列友好配置
  bash run_sweep.sh --output-lens "256 1024 4096"      # 只跑三个点
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
    "timestamp": "${TIMESTAMP}",
    "note": "ignore_eos=True (vllm 内部硬编码), 输出长度确定"
}
EXPEOF
}

# ======================== 单次 benchmark ========================
run_single() {
    local quant="$1"
    local output_len="$2"

    local model_path
    model_path=$(get_model_path "${quant}")

    if [[ ! -d "${model_path}" ]]; then
        log_warn "跳过 ${quant}: 模型路径不存在 ${model_path}"
        return 1
    fi

    local max_model_len=$((INPUT_LEN + output_len))
    local total_seqs=$((NUM_PROMPTS * NUM_SAMPLES))
    local result_file="${RESULT_DIR}/${quant}_olen${output_len}.json"
    local bench_log="${LOG_DIR}/${quant}_olen${output_len}.log"

    log_info "  ${NUM_PROMPTS} prompts × n=${NUM_SAMPLES} = ${total_seqs} 序列"
    log_info "  input=${INPUT_LEN}, output=${output_len}, max_model_len=${max_model_len}"

    # 环境变量
    local env_prefix=""
    env_prefix+="VLLM_USE_V1=1 "
    env_prefix+="ASCEND_RT_VISIBLE_DEVICES=${ASCEND_DEVICES} "

    # vllm bench throughput 命令
    local bench_cmd="python3 -m vllm.entrypoints.cli.main bench throughput"
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

    # W8A8 量化
    if [[ "${quant}" != "bf16" ]]; then
        bench_cmd+=" --quantization ascend"
    fi

    log_info "  命令: ${bench_cmd}"

    if eval "${env_prefix} ${bench_cmd}" 2>&1 | tee "${bench_log}"; then
        log_ok "  完成: ${result_file}"
        return 0
    else
        log_error "  失败! 查看日志: ${bench_log}"
        return 1
    fi
}

# ======================== 主循环 ========================
main() {
    save_experiment_config

    log_info "=============================================="
    log_info "  Decode 长度 Sweep 实验"
    log_info "=============================================="
    log_info "模型: ${DISPLAY_NAME} (tp=${TP_SIZE}, gpu_mem=${GPU_MEM_UTIL})"
    log_info "参数: input_len=${INPUT_LEN}, n=${NUM_SAMPLES}, num_prompts=${NUM_PROMPTS}"
    log_info "Sweep: output_len = ${OUTPUT_LENS[*]}"
    log_info "精度: ${QUANTS[*]}"
    log_info "输出: ${RUN_DIR}"
    log_info "注: vllm 内部 ignore_eos=True, 输出长度严格确定"
    echo ""

    local total_runs=$(( ${#OUTPUT_LENS[@]} * ${#QUANTS[@]} ))
    local run_idx=0
    local failed_runs=()
    local succeeded_runs=()

    for output_len in "${OUTPUT_LENS[@]}"; do
        log_step "output_len = ${output_len}"

        for quant in "${QUANTS[@]}"; do
            run_idx=$((run_idx + 1))
            local run_tag="${quant}_olen${output_len}"

            log_step "[${run_idx}/${total_runs}] ${DISPLAY_NAME} / ${quant^^} / output_len=${output_len}"

            if run_single "${quant}" "${output_len}"; then
                succeeded_runs+=("${run_tag}")
            else
                failed_runs+=("${run_tag}")
            fi

            # 等待 GPU 内存释放
            sleep 5
        done
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
