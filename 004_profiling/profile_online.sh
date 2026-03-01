#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 在线 Profiling: msserviceprofiler + vllm serve
#
# 通过 vllm serve 部署模型, 用 msserviceprofiler 采集:
#   - 框架级: scheduler / KV cache / model execute 各阶段耗时
#   - 算子级: ACL operator dispatch + execution 耗时 (acl_task_time)
#
# 与离线方式的区别:
#   离线 (profile_offline.sh): torch_npu.profiler, 输出 trace 文件 (Perfetto 可视化)
#   在线 (本脚本):             msserviceprofiler, 输出结构化 CSV + chrome_tracing.json
#
# 前置依赖:
#   pip install msserviceprofiler==1.2.2
#
# 用法:
#   bash profile_online.sh                     # 默认: BF16 + W8A8
#   bash profile_online.sh --quants "bf16"     # 只跑 BF16
#   bash profile_online.sh --acl-level L1      # 详细算子信息
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
SERVER_PORT=8080
NUM_REQUESTS=32
INPUT_LEN=1
OUTPUT_LEN=128
ACL_LEVEL="L0"         # L0: 低开销, L1: 详细算子信息
ACL_TASK_TIME=1         # 0: 禁用, 1: ACL原生, 2: MSPTI-based
PROF_TIMELIMIT=120      # profiling 超时 (秒)
ASCEND_DEVICES="0,1,2,3,4,5,6,7"
GPU_MEM_UTIL_OVERRIDE=""

# ======================== 参数解析 ========================
show_help() {
    cat <<EOF
用法: bash profile_online.sh [选项]

在线 Profiling: msserviceprofiler + vllm serve
启动 vllm serve, 发请求采集 profiling 数据

选项:
  --quants "Q1 Q2"     精度列表 (默认: "bf16 w8a8")
  --input-len N         输入长度 (默认: ${INPUT_LEN})
  --output-len N        输出长度 (默认: ${OUTPUT_LEN})
  --num-requests N      请求数量 (默认: ${NUM_REQUESTS})
  --acl-level L0|L1     ACL profiling 级别 (默认: ${ACL_LEVEL})
  --acl-task-time 0|1|2 ACL task time 模式 (默认: ${ACL_TASK_TIME})
  --port N              server 端口 (默认: ${SERVER_PORT})
  --gpu-mem-util F      覆盖 GPU 显存比例
  --model-base DIR      模型根目录
  --devices D           Ascend 设备列表
  -h, --help            显示帮助

ACL Task Time 模式:
  0: 不采集算子耗时
  1: ACL 原生采集 (推荐, 开销较低)
  2: MSPTI-based (更详细, 需 LD_PRELOAD=libmspti.so)

ACL Level:
  L0: 仅采集 dispatch + execution 耗时 (低开销)
  L1: 还采集 AscendCL 接口性能 + 算子基础信息 (全面分析)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quants)         IFS=' ' read -ra QUANTS <<< "$2"; shift 2 ;;
        --input-len)      INPUT_LEN="$2";         shift 2 ;;
        --output-len)     OUTPUT_LEN="$2";        shift 2 ;;
        --num-requests)   NUM_REQUESTS="$2";      shift 2 ;;
        --acl-level)      ACL_LEVEL="$2";         shift 2 ;;
        --acl-task-time)  ACL_TASK_TIME="$2";     shift 2 ;;
        --port)           SERVER_PORT="$2";       shift 2 ;;
        --gpu-mem-util)   GPU_MEM_UTIL_OVERRIDE="$2"; shift 2 ;;
        --model-base)     export MODEL_BASE="$2"; shift 2 ;;
        --devices)        ASCEND_DEVICES="$2";    shift 2 ;;
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

# ======================== 检查依赖 ========================
check_msserviceprofiler() {
    if ! python3 -c "import msserviceprofiler" 2>/dev/null; then
        log_error "msserviceprofiler 未安装. 请执行: pip install msserviceprofiler==1.2.2"
        exit 1
    fi
    log_ok "msserviceprofiler 已安装"
}

# ======================== 输出目录 ========================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${SCRIPT_DIR}/outputs/${TIMESTAMP}_online"
mkdir -p "${RUN_DIR}"

# ======================== 生成 profiler 配置 ========================
generate_profiler_config() {
    local prof_dir="$1"
    local config_file="${prof_dir}/ms_service_profiler_config.json"

    cat > "${config_file}" <<CFGEOF
{
    "enable": 1,
    "prof_dir": "${prof_dir}/prof_data",
    "profiler_level": "INFO",
    "acl_task_time": ${ACL_TASK_TIME},
    "acl_prof_task_time_level": "${ACL_LEVEL}",
    "timelimit": ${PROF_TIMELIMIT},
    "domain": "Request;KVCache;ModelExecute;BatchSchedule"
}
CFGEOF
    echo "${config_file}"
}

# ======================== 进程管理 ========================
SERVER_PID=""

kill_server() {
    if [[ -n "${SERVER_PID}" ]]; then
        log_info "停止 server (PID: ${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        sleep 3
        SERVER_PID=""
    fi
}

trap kill_server EXIT

wait_for_server() {
    local port="$1"
    local max_wait=300
    local elapsed=0
    log_info "等待 server 启动 (端口 ${port}, 最长 ${max_wait}s)..."
    while ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [[ $elapsed -ge $max_wait ]]; then
            log_error "server 启动超时 (${max_wait}s)"
            return 1
        fi
    done
    log_ok "server 已就绪 (${elapsed}s)"
}

# ======================== 发送请求 ========================
send_requests() {
    local model_path="$1"
    local num_requests="$2"

    log_info "发送 ${num_requests} 个请求..."
    for i in $(seq 1 "${num_requests}"); do
        curl -s "http://localhost:${SERVER_PORT}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"${model_path}\",
                \"prompt\": \"Explain the theory of\",
                \"max_completion_tokens\": ${OUTPUT_LEN},
                \"temperature\": 1.0
            }" > /dev/null 2>&1 &
    done
    wait
    log_ok "  ${num_requests} 个请求已完成"
}

# ======================== 单次 profiling ========================
run_online_profile() {
    local quant="$1"

    local model_path
    model_path=$(get_model_path "${quant}")

    if [[ ! -d "${model_path}" ]]; then
        log_warn "跳过 ${quant}: 模型路径不存在 ${model_path}"
        return 1
    fi

    local max_model_len=$((INPUT_LEN + OUTPUT_LEN))
    local prof_dir="${RUN_DIR}/profile_${quant}"
    local server_log="${RUN_DIR}/${quant}_server.log"
    mkdir -p "${prof_dir}"

    # 生成 profiler 配置
    local config_file
    config_file=$(generate_profiler_config "${prof_dir}")

    log_info "  Profiler 配置: ${config_file}"
    log_info "  model_path: ${model_path}"

    # 启动 server (带 profiling 环境变量)
    local env_prefix=""
    env_prefix+="VLLM_USE_V1=1 "
    env_prefix+="ASCEND_RT_VISIBLE_DEVICES=${ASCEND_DEVICES} "
    env_prefix+="SERVICE_PROF_CONFIG_PATH=${config_file} "

    local serve_cmd="python3 -m vllm.entrypoints.cli.main serve ${model_path}"
    serve_cmd+=" --dtype bfloat16"
    serve_cmd+=" --tensor-parallel-size ${TP_SIZE}"
    serve_cmd+=" --max-model-len ${max_model_len}"
    serve_cmd+=" --gpu-memory-utilization ${GPU_MEM_UTIL}"
    serve_cmd+=" --trust-remote-code"
    serve_cmd+=" --port ${SERVER_PORT}"

    # W8A8 量化
    if [[ "${quant}" != "bf16" ]]; then
        serve_cmd+=" --quantization ascend"
    fi

    log_info "  启动 server: ${serve_cmd}"
    eval "${env_prefix} ${serve_cmd}" > "${server_log}" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server "${SERVER_PORT}"; then
        log_error "  server 启动失败! 查看日志: ${server_log}"
        kill_server
        return 1
    fi

    # 发送请求
    send_requests "${model_path}" "${NUM_REQUESTS}"

    # 等待 profiling 数据写入
    sleep 5

    # 停止 server
    kill_server

    # 分析 profiling 数据
    local prof_data_dir="${prof_dir}/prof_data"
    if [[ -d "${prof_data_dir}" ]]; then
        log_info "  分析 profiling 数据..."
        if msserviceprofiler analyze \
            --input-path="${prof_data_dir}" \
            --output-path="${prof_dir}/analysis" 2>/dev/null; then
            log_ok "  分析完成: ${prof_dir}/analysis/"
        else
            log_warn "  自动分析失败, 请手动运行:"
            log_warn "  msserviceprofiler analyze --input-path=${prof_data_dir} --output-path=${prof_dir}/analysis"
        fi
    fi

    log_ok "  Profiling 完成: ${prof_dir}"
}

# ======================== 主流程 ========================
main() {
    check_msserviceprofiler

    log_info "=============================================="
    log_info "  在线 Profiling (msserviceprofiler)"
    log_info "=============================================="
    log_info "模型: ${DISPLAY_NAME} (tp=${TP_SIZE}, gpu_mem=${GPU_MEM_UTIL})"
    log_info "参数: input=${INPUT_LEN}, output=${OUTPUT_LEN}, requests=${NUM_REQUESTS}"
    log_info "ACL: task_time=${ACL_TASK_TIME}, level=${ACL_LEVEL}"
    log_info "精度: ${QUANTS[*]}"
    log_info "输出: ${RUN_DIR}"
    echo ""

    for quant in "${QUANTS[@]}"; do
        log_step "Online Profiling: ${DISPLAY_NAME} / ${quant^^}"

        if run_online_profile "${quant}"; then
            log_ok "${quant} profiling 完成"
        else
            log_error "${quant} profiling 失败"
        fi

        sleep 10
    done

    echo ""
    log_info "=============================================="
    log_info "  Profiling 采集完成"
    log_info "=============================================="
    log_info "输出目录: ${RUN_DIR}"
    echo ""
    log_info "查看结果:"
    log_info "  chrome_tracing.json → MindStudio Insight 或 Perfetto UI"
    log_info "  batch.csv / request.csv → 调度与请求级指标"
    log_info "  service_summary.csv → 服务级汇总"
}

main
