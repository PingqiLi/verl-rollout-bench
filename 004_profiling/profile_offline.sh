#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 离线 Profiling: 单步精确采集 (torch_npu profiler)
#
# 采集 Qwen3-30B-A3B decode 阶段的算子级 profiling 数据.
# 调用 profile_single_step.py, 仅采集 1 prefill + 1 decode step.
#
# 用法:
#   bash profile_offline.sh                    # 默认: 图模式, BF16 + W8A8
#   bash profile_offline.sh --eager            # eager 模式
#   bash profile_offline.sh --quants "bf16"    # 只跑 BF16
#   bash profile_offline.sh --batch-size 64    # 大 batch
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

cfg() {
    python3 "${CONFIG_PARSER}" "${CONFIG_FILE}" "$@"
}

# ======================== 默认参数 ========================
MODEL_KEY="qwen3-30b-a3b"
QUANTS=("bf16" "w8a8")

# 单步 profiling 参数
BATCH_SIZE=4            # 同时推理的 prompt 数
MAX_TOKENS=10           # 每 prompt 生成 token 数 (1 prefill + 9 decode, decode 占 ~90%)
WARMUP_STEPS=3          # warmup 步数 (graph capture + JIT)

ENFORCE_EAGER=false
GPU_MEM_UTIL_OVERRIDE=""
ASCEND_DEVICES="0,1,2,3,4,5,6,7"

# ======================== 参数解析 ========================
show_help() {
    cat <<EOF
用法: bash profile_offline.sh [选项]

单步 Profiling: 精确采集 decode step 的算子级 trace (BF16 vs W8A8D)

选项:
  --quants "Q1 Q2"      精度列表 (默认: "bf16 w8a8")
  --batch-size N         batch 大小 (默认: ${BATCH_SIZE})
  --max-tokens N         生成 token 数, 10=1 prefill+9 decode (默认: ${MAX_TOKENS})
  --warmup-steps N       warmup 步数 (默认: ${WARMUP_STEPS})
  --eager                强制 eager 模式 (禁用 ACLGraph)
  --gpu-mem-util F       覆盖 GPU 显存比例
  --model-base DIR       模型根目录
  --devices D            Ascend 设备列表
  -h, --help             显示帮助


示例:
  bash profile_offline.sh                         # 默认: BF16 + W8A8, 图模式
  bash profile_offline.sh --quants "bf16"         # 只跑 BF16
  bash profile_offline.sh --batch-size 128        # 大 batch
  bash profile_offline.sh --eager                 # eager 模式对照
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quants)         IFS=' ' read -ra QUANTS <<< "$2"; shift 2 ;;
        --batch-size)     BATCH_SIZE="$2";          shift 2 ;;
        --max-tokens)     MAX_TOKENS="$2";          shift 2 ;;
        --warmup-steps)   WARMUP_STEPS="$2";        shift 2 ;;
        --eager)          ENFORCE_EAGER=true;        shift ;;
        --gpu-mem-util)   GPU_MEM_UTIL_OVERRIDE="$2"; shift 2 ;;
        --model-base)     export MODEL_BASE="$2";   shift 2 ;;
        --devices)        ASCEND_DEVICES="$2";       shift 2 ;;
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

get_model_path() {
    cfg get-path "${MODEL_KEY}" "$1"
}

# ======================== 输出目录 ========================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODE_TAG="graph"
if [[ "$ENFORCE_EAGER" == true ]]; then
    MODE_TAG="eager"
fi
RUN_DIR="${SCRIPT_DIR}/outputs/${TIMESTAMP}_${MODE_TAG}"
mkdir -p "${RUN_DIR}"

# ======================== 单次 profiling ========================
run_profile() {
    local quant="$1"

    local model_path
    model_path=$(get_model_path "${quant}")

    if [[ ! -d "${model_path}" ]]; then
        log_warn "跳过 ${quant}: 模型路径不存在 ${model_path}"
        return 1
    fi

    local profile_dir="${RUN_DIR}/profile_${quant}"
    local prof_log="${RUN_DIR}/${quant}_profile.log"

    mkdir -p "${profile_dir}"

    log_info "  batch=${BATCH_SIZE}, max_tokens=${MAX_TOKENS}, mode=${MODE_TAG}"
    log_info "  warmup=${WARMUP_STEPS} 步"
    log_info "  输出: ${profile_dir}"

    # 构建 profile_single_step.py 命令
    local prof_cmd="python3 ${SCRIPT_DIR}/profile_single_step.py"
    prof_cmd+=" --model ${model_path}"
    prof_cmd+=" --quant ${quant}"
    prof_cmd+=" --output-dir ${profile_dir}"
    prof_cmd+=" --tp ${TP_SIZE}"
    prof_cmd+=" --gpu-mem-util ${GPU_MEM_UTIL}"
    prof_cmd+=" --batch-size ${BATCH_SIZE}"
    prof_cmd+=" --max-tokens ${MAX_TOKENS}"
    prof_cmd+=" --warmup-steps ${WARMUP_STEPS}"

    # eager 模式
    if [[ "${ENFORCE_EAGER}" == true ]]; then
        prof_cmd+=" --enforce-eager"
    fi

    # 设备选择由环境变量控制
    local env_prefix="ASCEND_RT_VISIBLE_DEVICES=${ASCEND_DEVICES}"

    log_info "  命令: ${prof_cmd}"

    if eval "${env_prefix} ${prof_cmd}" 2>&1 | tee "${prof_log}"; then
        log_ok "  Profiling 完成: ${profile_dir}"
        return 0
    else
        log_error "  Profiling 失败! 查看日志: ${prof_log}"
        return 1
    fi
}

# ======================== 主流程 ========================
main() {
    log_info "=============================================="
    log_info "  单步 Profiling (torch_npu profiler)"
    log_info "=============================================="
    log_info "模型: ${DISPLAY_NAME} (tp=${TP_SIZE}, gpu_mem=${GPU_MEM_UTIL})"
    log_info "参数: batch=${BATCH_SIZE}, max_tokens=${MAX_TOKENS}, warmup=${WARMUP_STEPS}"
    log_info "模式: ${MODE_TAG}"
    log_info "精度: ${QUANTS[*]}"
    log_info "输出: ${RUN_DIR}"
    echo ""

    for quant in "${QUANTS[@]}"; do
        log_step "Profiling: ${DISPLAY_NAME} / ${quant^^} (${MODE_TAG})"

        if run_profile "${quant}"; then
            log_ok "${quant} profiling 完成"
        else
            log_error "${quant} profiling 失败"
        fi

        # 等待 GPU 内存释放 (下一个 quant 需要重新加载模型)
        sleep 10
    done

    echo ""
    log_info "=============================================="
    log_info "  Profiling 采集完成"
    log_info "=============================================="
    log_info "输出目录: ${RUN_DIR}"
    echo ""
    log_info "查看 trace:"
    log_info "  方式 1: 上传到 https://ui.perfetto.dev/"
    log_info "  方式 2: MindStudio Insight 打开"
    echo ""
    log_info "对比分析:"
    log_info "  python3 ${SCRIPT_DIR}/analyze_profile.py ${RUN_DIR}"
}

main
