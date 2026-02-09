#!/bin/bash
# ============================================================
# vLLM Online Serving Benchmark Script (Ascend NPU)
#
# 实验矩阵: 3 量化精度 x 3 模型 - 1= 8 组实验 (Qwen3-30B-A3B 不包含W8A16)
#   量化: BF16, W8A16, W8A8
#   模型: Qwen3-1.7B, Pangu-7B, Qwen3-30B-A3B
#
# 用法:
#   正式跑全量实验:     bash run_vllm_benchmark.sh
#   诊断模式(快速验证): bash run_vllm_benchmark.sh --diagnostic
#   指定实验子集:       bash run_vllm_benchmark.sh --models qwen3-1.7b --quants bf16,w8a16
#   不采集profiling:    bash run_vllm_benchmark.sh --no-profile
# ============================================================
set -euo pipefail

# ======================== 默认配置 ========================

# 可通过环境变量预设, 脚本内为 fallback 默认值
MODEL_BASE="${MODEL_BASE:-/data/l50044498/models}"
BENCH_BASE_DIR="${BENCH_BASE_DIR:-.}"

# 服务端口
SERVER_PORT=8080
SERVER_HOST="127.0.0.1"

# vLLM 环境变量
export VLLM_USE_V1=1

# Ascend NPU 环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 时间戳, 用于区分多次实验
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

# Profiling 控制 (默认关闭, profiling 会严重拖慢推理, 性能数字不准)
ENABLE_PROFILING=false
PROFILING_BASE_DIR="${BENCH_BASE_DIR}/benchmark_profiles/${RUN_TIMESTAMP}"

# 结果输出目录 (每次实验独立子目录, 不覆盖)
RESULT_DIR="${BENCH_BASE_DIR}/benchmark_results/${RUN_TIMESTAMP}"
LOG_DIR="${BENCH_BASE_DIR}/benchmark_logs/${RUN_TIMESTAMP}"

# Benchmark 模式
#   offline = 用 vllm bench throughput, 直接调用 LLM.generate(), 不走 HTTP
#             支持 n=8 (每 prompt 多次采样), 高并发, 最接近 verl rollout 的真实行为
#   online  = 用 vllm bench serve, 走 HTTP 在线服务, 支持 profiling 采集
#             但并发受 max-concurrency 限制, 不支持 n 参数, 不能完全模拟 rollout
BENCH_MODE="offline"

# Benchmark 负载参数
# 目标: 用相同负载对比 BF16 vs W8A16 vs W8A8 的 throughput 差异
INPUT_LEN=512
OUTPUT_LEN=256
NUM_PROMPTS=32                  # per-GPU prompt 数
NUM_SAMPLES_PER_PROMPT=8        # 每 prompt 生成 n 个采样, 仅 offline 模式有效
MAX_CONCURRENCY=128             # 仅 online 模式有效
REQUEST_RATE="inf"              # 仅 online 模式有效

# Ascend 图模式 Bucket 配置 (与 input_len 对齐)
BUCKET_MIN=512
BUCKET_MAX=512

# Server 等待超时 (秒)
SERVER_WAIT_TIMEOUT=600

# 诊断模式标志
DIAGNOSTIC_MODE=false

# 不采集 profiling 标志
NO_PROFILE=false

# 用户指定的实验子集
USER_MODELS=""
USER_QUANTS=""
GPU_MEMORY_UTILIZATION_OVERRIDE=""

# ======================== 参数解析 ========================

print_usage() {
    echo "用法: $0 [OPTIONS]"
    echo ""
    echo "选项:"
    echo "  --diagnostic          诊断模式: 只跑 BF16+Qwen3-1.7B, 少量请求, 验证能跑通"
    echo "  --offline             [默认] 离线模式: LLM.generate() 直接推理, 高并发, 最接近 verl rollout"
    echo "  --online              在线服务模式: 走 HTTP, 支持 profiling 采集, 但并发受限"
    echo "  --profile             采集 torch profiling (默认关闭, 会拖慢推理)"
    echo "  --no-profile          不采集 torch profiling (默认)"
    echo "  --models MODELS       逗号分隔的模型列表, 如: qwen3-1.7b,pangu-7b,qwen3-30b-a3b"
    echo "  --quants QUANTS       逗号分隔的量化列表, 如: bf16,w8a16,w8a8_dynamic"
    echo "  --input-len N         输入 prompt 长度 (默认: ${INPUT_LEN})"
    echo "  --output-len N        输出生成长度 (默认: ${OUTPUT_LEN})"
    echo "  --num-prompts N       请求数量 (默认: ${NUM_PROMPTS})"
    echo "  -n N                  每个 prompt 生成 N 个采样 (默认: ${NUM_SAMPLES_PER_PROMPT}, 仅 offline)"
    echo "  --gpu-mem-util F      GPU 显存占用比例 (默认: per-model, 设置后覆盖所有模型)"
    echo "  --max-concurrency N   最大并发数 (默认: ${MAX_CONCURRENCY}, 仅 online)"
    echo "  --port PORT           服务端口 (默认: ${SERVER_PORT}, 仅 online)"
    echo "  --model-base DIR      模型根目录 (或设环境变量 MODEL_BASE, 默认: ${MODEL_BASE})"
    echo "  -h, --help            显示帮助信息"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --diagnostic)
            DIAGNOSTIC_MODE=true
            shift ;;
        --offline)
            BENCH_MODE="offline"
            shift ;;
        --online)
            BENCH_MODE="online"
            shift ;;
        --profile)
            ENABLE_PROFILING=true
            shift ;;
        --no-profile)
            NO_PROFILE=true
            shift ;;
        --models)
            USER_MODELS="$2"
            shift 2 ;;
        --quants)
            USER_QUANTS="$2"
            shift 2 ;;
        --input-len)
            INPUT_LEN="$2"
            shift 2 ;;
        --output-len)
            OUTPUT_LEN="$2"
            shift 2 ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2 ;;
        -n)
            NUM_SAMPLES_PER_PROMPT="$2"
            shift 2 ;;
        --gpu-mem-util)
            GPU_MEMORY_UTILIZATION_OVERRIDE="$2"
            shift 2 ;;
        --max-concurrency)
            MAX_CONCURRENCY="$2"
            shift 2 ;;
        --port)
            SERVER_PORT="$2"
            shift 2 ;;
        --model-base)
            MODEL_BASE="$2"
            shift 2 ;;
        -h|--help)
            print_usage
            exit 0 ;;
        *)
            echo "[ERROR] 未知参数: $1"
            print_usage
            exit 1 ;;
    esac
done

# ======================== 颜色输出 ========================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $(date '+%H:%M:%S') $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date '+%H:%M:%S') $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $*"; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $(date '+%H:%M:%S') $*"; }

# ======================== 实验矩阵定义 ========================

# 模型配置 (只配模型路径、TP、显存等硬件相关参数)
# 负载参数 (input_len, output_len, num_prompts) 用全局配置, 保证公平对比
declare_model_config() {
    # --- Qwen3-1.7B ---
    QWEN3_1_7B_DISPLAY="Qwen3-1.7B"
    QWEN3_1_7B_PATH_BF16="${MODEL_BASE}/qwen3-1.7b"
    QWEN3_1_7B_PATH_W8A16="${MODEL_BASE}/qwen3-1.7b-W8A16"
    QWEN3_1_7B_PATH_W8A8="${MODEL_BASE}/qwen3-1.7b-W8A8D"
    QWEN3_1_7B_TP=1
    QWEN3_1_7B_GPU_MEM_UTIL=0.4

    # --- Pangu-7B ---
    PANGU_7B_DISPLAY="Pangu-7B"
    PANGU_7B_PATH_BF16="${MODEL_BASE}/openPangu-Embedded-7B-V1.1"
    PANGU_7B_PATH_W8A16="${MODEL_BASE}/openPangu-Embedded-7B-V1.1-W8A16"
    PANGU_7B_PATH_W8A8="${MODEL_BASE}/openPangu-Embedded-7B-V1.1-W8A8D"
    PANGU_7B_TP=1
    PANGU_7B_GPU_MEM_UTIL=0.6

    # --- Qwen3-30B-A3B (MoE) ---
    QWEN3_30B_A3B_DISPLAY="Qwen3-30B-A3B"
    QWEN3_30B_A3B_PATH_BF16="${MODEL_BASE}/Qwen3-30B-A3B-Instruct-2507"
    QWEN3_30B_A3B_PATH_W8A8="${MODEL_BASE}/Qwen3-30B-A3B-Instruct-2507-W8A8D"
    QWEN3_30B_A3B_TP=4
    QWEN3_30B_A3B_GPU_MEM_UTIL=0.8
}

# 获取模型配置字段
# 用法: get_model_field <model_key> <field>
# field: DISPLAY, PATH_BF16, PATH_W8A16, PATH_W8A8, TP, MAX_MODEL_LEN, MAX_NUM_SEQS
get_model_field() {
    local model_key="$1"
    local field="$2"
    local var_name=""

    case "$model_key" in
        qwen3-1.7b)    var_name="QWEN3_1_7B_${field}" ;;
        pangu-7b)       var_name="PANGU_7B_${field}" ;;
        qwen3-30b-a3b) var_name="QWEN3_30B_A3B_${field}" ;;
        *)
            log_error "未知模型: $model_key"
            return 1 ;;
    esac
    eval "echo \"\${${var_name}:-}\"" 2>/dev/null || echo ""
}

# 获取对应量化精度的模型路径 (返回空字符串表示该组合不存在)
get_model_path() {
    local model_key="$1"
    local quant="$2"
    local field=""
    case "$quant" in
        bf16)   field="PATH_BF16" ;;
        w8a16)  field="PATH_W8A16" ;;
        w8a8)   field="PATH_W8A8" ;;
        *)
            log_error "未知量化精度: $quant"
            return 1 ;;
    esac

    local var_name=""
    case "$model_key" in
        qwen3-1.7b)    var_name="QWEN3_1_7B_${field}" ;;
        pangu-7b)       var_name="PANGU_7B_${field}" ;;
        qwen3-30b-a3b) var_name="QWEN3_30B_A3B_${field}" ;;
        *)
            log_error "未知模型: $model_key"
            return 1 ;;
    esac

    # eval 兼容 bash 3.x + set -u, 变量未定义时返回空字符串
    eval "echo \"\${${var_name}:-}\""
}

# 获取模型的 gpu_memory_utilization
# CLI --gpu-mem-util 覆盖 > per-model 配置 > 默认 0.9
get_gpu_mem_util() {
    local model_key="$1"

    # CLI 覆盖优先
    if [[ -n "${GPU_MEMORY_UTILIZATION_OVERRIDE}" ]]; then
        echo "${GPU_MEMORY_UTILIZATION_OVERRIDE}"
        return
    fi

    # per-model 硬编码, 不依赖动态变量查找
    case "$model_key" in
        qwen3-1.7b)    echo "${QWEN3_1_7B_GPU_MEM_UTIL:-0.9}" ;;
        pangu-7b)       echo "${PANGU_7B_GPU_MEM_UTIL:-0.9}" ;;
        qwen3-30b-a3b) echo "${QWEN3_30B_A3B_GPU_MEM_UTIL:-0.9}" ;;
        *)              echo "0.9" ;;
    esac
}

# 判断某个 model+quant 组合是否应该跳过
# 返回 0 表示应该跳过, 1 表示不跳过
should_skip_experiment() {
    local model_key="$1"
    local quant="$2"
    local model_path
    model_path=$(get_model_path "$model_key" "$quant")

    # 路径为空 = 该组合未配置 (如 Qwen3-30B-A3B 没有 W8A16)
    if [[ -z "$model_path" ]]; then
        return 0
    fi
    return 1
}

# 获取所有有效实验的列表 (格式: "model_key|quant" 每行一个)
get_valid_experiments() {
    local models=("$@")
    # quants 从全局变量读取, 通过 _QUANTS 数组传入
    for model_key in "${models[@]}"; do
        for quant in "${_QUANTS[@]}"; do
            if ! should_skip_experiment "$model_key" "$quant"; then
                echo "${model_key}|${quant}"
            fi
        done
    done
}

# ======================== 辅助函数 ========================

# 初始化目录
init_dirs() {
    mkdir -p "${RESULT_DIR}" "${LOG_DIR}" "${PROFILING_BASE_DIR}"
}

# 获取实验的 profiling 目录
get_profile_dir() {
    local model_key="$1"
    local quant="$2"
    echo "${PROFILING_BASE_DIR}/${model_key}_${quant}"
}

# 获取实验的结果文件路径
get_result_file() {
    local model_key="$1"
    local quant="$2"
    echo "${RESULT_DIR}/${model_key}_${quant}.json"
}

# 获取实验的 server 日志文件路径
get_server_log() {
    local model_key="$1"
    local quant="$2"
    echo "${LOG_DIR}/server_${model_key}_${quant}.log"
}

# 获取实验的 benchmark 日志文件路径
get_bench_log() {
    local model_key="$1"
    local quant="$2"
    echo "${LOG_DIR}/bench_${model_key}_${quant}.log"
}

# 杀掉某个进程及其所有子进程 (递归)
kill_process_tree() {
    local pid="$1"
    local signal="${2:-TERM}"

    # 先找到所有子进程 (深度优先, 先杀子再杀父)
    local children
    children=$(pgrep -P "$pid" 2>/dev/null || true)
    for child in $children; do
        kill_process_tree "$child" "$signal"
    done

    # 杀当前进程
    if kill -0 "$pid" 2>/dev/null; then
        kill -"$signal" "$pid" 2>/dev/null || true
    fi
}

# 等待进程树全部退出
wait_process_tree_exit() {
    local pid="$1"
    local timeout="$2"
    local elapsed=0

    while [[ $elapsed -lt $timeout ]]; do
        # 检查主进程和所有子进程是否都退出
        if ! kill -0 "$pid" 2>/dev/null; then
            local remaining
            remaining=$(pgrep -P "$pid" 2>/dev/null || true)
            if [[ -z "$remaining" ]]; then
                return 0
            fi
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 1  # 超时
}

# 杀掉指定端口上的 vllm server (含所有 worker 子进程)
kill_server() {
    local port="$1"

    # === 路径 1: 通过记录的主进程 PID 杀整棵进程树 ===
    if [[ -n "${CURRENT_SERVER_PID:-}" ]] && kill -0 "$CURRENT_SERVER_PID" 2>/dev/null; then
        log_info "正在停止 server 进程树 (主 PID: ${CURRENT_SERVER_PID})..."

        # 列出所有子进程用于日志
        local child_pids
        child_pids=$(pgrep -P "$CURRENT_SERVER_PID" 2>/dev/null || true)
        if [[ -n "$child_pids" ]]; then
            log_info "  发现 worker 子进程: ${child_pids}"
        fi

        # 第一轮: SIGTERM 优雅退出 (给 profiling flush 时间)
        kill_process_tree "$CURRENT_SERVER_PID" "TERM"

        # 等待进程树退出 (最长 120s, profiling flush 可能很慢)
        if wait_process_tree_exit "$CURRENT_SERVER_PID" 120; then
            log_ok "Server 进程树已全部退出"
        else
            # 第二轮: SIGKILL 强制杀
            log_warn "Server 进程树未在 120s 内退出, 强制终止..."
            kill_process_tree "$CURRENT_SERVER_PID" "KILL"
            sleep 3
        fi

        CURRENT_SERVER_PID=""
    fi

    # === 路径 2: 兜底清理 — 通过端口和关键字查找残留进程 ===
    # 即使路径 1 执行了, 也做一次兜底检查, 防止有漏网之鱼

    # 2a. 通过端口查找
    local port_pids
    port_pids=$(lsof -ti:"${port}" 2>/dev/null || true)
    if [[ -n "$port_pids" ]]; then
        log_warn "端口 ${port} 仍有残留进程: ${port_pids}, 正在清理..."
        echo "$port_pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi

    # 2b. 通过进程名查找残留的 vllm worker 进程
    #     vLLM V1 的 worker 进程命令行包含 "vllm.entrypoints" 或 "vllm.v1.worker"
    local stale_pids
    stale_pids=$(pgrep -f "vllm.entrypoints.openai.api_server.*--port ${port}" 2>/dev/null || true)
    if [[ -n "$stale_pids" ]]; then
        log_warn "发现残留 vllm 进程: ${stale_pids}, 正在清理..."
        echo "$stale_pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

# 等待 server 就绪
wait_for_server() {
    local port="$1"
    local timeout="$2"
    local url="http://${SERVER_HOST}:${port}/health"
    local elapsed=0

    log_info "等待 server 就绪 (超时: ${timeout}s)..."
    while [[ $elapsed -lt $timeout ]]; do
        if curl -sf "${url}" > /dev/null 2>&1; then
            log_ok "Server 已就绪 (耗时: ${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [[ $((elapsed % 30)) -eq 0 ]]; then
            log_info "已等待 ${elapsed}s..."
        fi
    done

    log_error "Server 在 ${timeout}s 内未就绪"
    return 1
}

# 全局 server PID 追踪
CURRENT_SERVER_PID=""

# 确保退出时清理 server 进程
cleanup() {
    log_info "正在清理..."
    kill_server "$SERVER_PORT"
}
trap cleanup EXIT

# ======================== 核心: 拉起 Server ========================

start_server() {
    local model_key="$1"
    local quant="$2"

    local model_path
    model_path=$(get_model_path "$model_key" "$quant")
    local tp_size
    tp_size=$(get_model_field "$model_key" "TP")
    local gpu_mem_util
    gpu_mem_util=$(get_gpu_mem_util "$model_key")
    local max_model_len=$((INPUT_LEN + OUTPUT_LEN))
    local display_name
    display_name=$(get_model_field "$model_key" "DISPLAY")
    local server_log
    server_log=$(get_server_log "$model_key" "$quant")

    log_step "拉起 Server: ${display_name} / ${quant^^}"
    log_info "  模型路径: ${model_path}"
    log_info "  TP: ${tp_size}, max_model_len: ${max_model_len}, gpu_mem: ${gpu_mem_util}"

    # 检查模型路径是否存在
    if [[ ! -d "$model_path" ]]; then
        log_error "模型路径不存在: ${model_path}"
        return 1
    fi

    # 先杀掉残留进程, 确保端口和显存都被释放
    kill_server "$SERVER_PORT"
    log_info "等待残留进程清理和显存释放 (5s)..."
    sleep 5

    # 构建 server 启动命令
    local server_cmd="python3 -m vllm.entrypoints.openai.api_server"
    server_cmd+=" --port ${SERVER_PORT}"
    server_cmd+=" --model ${model_path}"
    server_cmd+=" --tensor-parallel-size ${tp_size}"
    server_cmd+=" --dtype bfloat16"
    server_cmd+=" --max-model-len ${max_model_len}"
    server_cmd+=" --trust-remote-code"
    server_cmd+=" --gpu-memory-utilization ${gpu_mem_util}"
    server_cmd+=" --disable-log-requests"
    server_cmd+=" --no-enable-chunked-prefill"

    # W8A16 / W8A8 需要指定量化方式
    if [[ "$quant" != "bf16" ]]; then
        server_cmd+=" --quantization ascend"
    fi

    # 构建环境变量
    local env_prefix=""
    env_prefix+="VLLM_USE_V1=1 "
    env_prefix+="VLLM_PROMPT_SEQ_BUCKET_MIN=${BUCKET_MIN} "
    env_prefix+="VLLM_PROMPT_SEQ_BUCKET_MAX=${BUCKET_MAX} "

    # Profiling 环境变量
    if [[ "$ENABLE_PROFILING" == true ]]; then
        local profile_dir
        profile_dir=$(get_profile_dir "$model_key" "$quant")
        mkdir -p "$profile_dir"
        env_prefix+="VLLM_TORCH_PROFILER_DIR=${profile_dir} "
        # 加大 RPC 超时, 防止 profiling flush 超时
        env_prefix+="VLLM_RPC_TIMEOUT=1800000 "
        log_info "  Profiling 输出: ${profile_dir}"
    fi

    log_info "  Server 日志: ${server_log}"
    log_info "  启动命令: ${env_prefix}${server_cmd}"

    # 后台启动 server
    eval "${env_prefix} ${server_cmd}" > "${server_log}" 2>&1 &
    CURRENT_SERVER_PID=$!
    log_info "  Server PID: ${CURRENT_SERVER_PID}"

    # 等待 server 就绪
    if ! wait_for_server "$SERVER_PORT" "$SERVER_WAIT_TIMEOUT"; then
        log_error "Server 启动失败, 查看日志: ${server_log}"
        log_error "=== Server 日志尾部 ==="
        tail -30 "${server_log}" || true
        kill_server "$SERVER_PORT"
        return 1
    fi
}

# ======================== 核心: 运行 Benchmark ========================

run_benchmark() {
    local model_key="$1"
    local quant="$2"
    local num_prompts="$3"

    if [[ "$BENCH_MODE" == "offline" ]]; then
        run_benchmark_offline "$model_key" "$quant" "$num_prompts"
    else
        run_benchmark_online "$model_key" "$quant" "$num_prompts"
    fi
}

# ---- Offline 模式: 直接 LLM.generate(), 最接近 verl rollout ----
# 优势:
#   - 支持 n=8 (每个 prompt 生成 n 个采样), 完全模拟 verl rollout
#   - 所有 prompt 同时提交给 engine, 不受 HTTP max-concurrency 限制
#   - 实际并发 = num_prompts × n = 256 × 8 = 2048 序列
#   - 大 batch → 大 GEMM → 量化加速效果能正确体现
# 局限:
#   - profiling 需要通过环境变量 VLLM_TORCH_PROFILER_DIR + --profile 参数
#   - 输出指标为 throughput (tok/s) 和 elapsed_time, 没有 TTFT/TPOT/E2EL 分解

run_benchmark_offline() {
    local model_key="$1"
    local quant="$2"
    local num_prompts="$3"

    local model_path
    model_path=$(get_model_path "$model_key" "$quant")
    local display_name
    display_name=$(get_model_field "$model_key" "DISPLAY")
    local tp_size
    tp_size=$(get_model_field "$model_key" "TP")
    local gpu_mem_util
    gpu_mem_util=$(get_gpu_mem_util "$model_key")
    local max_model_len=$((INPUT_LEN + OUTPUT_LEN))
    local result_file
    result_file=$(get_result_file "$model_key" "$quant")
    local bench_log
    bench_log=$(get_bench_log "$model_key" "$quant")

    # 安全检查: gpu_mem_util 不能为空
    if [[ -z "${gpu_mem_util}" ]]; then
        log_error "get_gpu_mem_util 返回空值 (model_key=${model_key}), 使用默认 0.9"
        gpu_mem_util="0.9"
    fi

    local total_seqs=$((num_prompts * NUM_SAMPLES_PER_PROMPT))
    log_step "运行 Offline Benchmark: ${display_name} / ${quant^^}"
    log_info "  ${num_prompts} prompts × n=${NUM_SAMPLES_PER_PROMPT} = ${total_seqs} 序列"
    log_info "  input=${INPUT_LEN}, output=${OUTPUT_LEN}, gpu_mem=${gpu_mem_util}"

    # 构建环境变量
    local env_prefix=""
    env_prefix+="VLLM_USE_V1=1 "
    env_prefix+="VLLM_PROMPT_SEQ_BUCKET_MIN=${BUCKET_MIN} "
    env_prefix+="VLLM_PROMPT_SEQ_BUCKET_MAX=${BUCKET_MAX} "

    # Profiling
    local profile_flag=""
    if [[ "$ENABLE_PROFILING" == true ]]; then
        local profile_dir
        profile_dir=$(get_profile_dir "$model_key" "$quant")
        mkdir -p "$profile_dir"
        env_prefix+="VLLM_TORCH_PROFILER_DIR=${profile_dir} "
        profile_flag="--profile"
        log_info "  Profiling 输出: ${profile_dir}"
    fi

    # vllm bench throughput 命令
    local bench_cmd="python3 -m vllm.entrypoints.cli.main bench throughput"
    bench_cmd+=" --model ${model_path}"
    bench_cmd+=" --dataset-name random"
    bench_cmd+=" --input-len ${INPUT_LEN}"
    bench_cmd+=" --output-len ${OUTPUT_LEN}"
    bench_cmd+=" --num-prompts ${num_prompts}"
    bench_cmd+=" --n ${NUM_SAMPLES_PER_PROMPT}"
    bench_cmd+=" --dtype bfloat16"
    bench_cmd+=" --tensor-parallel-size ${tp_size}"
    bench_cmd+=" --max-model-len ${max_model_len}"
    bench_cmd+=" --gpu-memory-utilization ${gpu_mem_util}"
    bench_cmd+=" --trust-remote-code"
    bench_cmd+=" --no-enable-chunked-prefill"
    bench_cmd+=" --output-json ${result_file}"
    bench_cmd+=" ${profile_flag}"

    # W8A16 / W8A8 量化
    if [[ "$quant" != "bf16" ]]; then
        bench_cmd+=" --quantization ascend"
    fi

    log_info "  命令: ${env_prefix}${bench_cmd}"
    log_info "  日志: ${bench_log}"

    # 执行 (offline 模式不需要单独的 server 进程)
    if eval "${env_prefix} ${bench_cmd}" 2>&1 | tee "${bench_log}"; then
        log_ok "Benchmark 完成: ${display_name} / ${quant^^}"
    else
        log_error "Benchmark 失败: ${display_name} / ${quant^^}"
        log_error "查看日志: ${bench_log}"
        return 1
    fi
}

# ---- Online 模式: 走 HTTP 在线服务 (支持细粒度延迟指标) ----
# 优势:
#   - 输出 TTFT, TPOT, ITL, E2EL 等细粒度延迟指标
#   - 通过 HTTP /start_profile /stop_profile 控制 profiling
# 局限:
#   - 并发受 max-concurrency 限制, 不支持 n 参数
#   - 实际并发远低于 verl rollout, 量化加速效果可能被低估

run_benchmark_online() {
    local model_key="$1"
    local quant="$2"
    local num_prompts="$3"

    local model_path
    model_path=$(get_model_path "$model_key" "$quant")
    local display_name
    display_name=$(get_model_field "$model_key" "DISPLAY")
    local result_file
    result_file=$(get_result_file "$model_key" "$quant")
    local bench_log
    bench_log=$(get_bench_log "$model_key" "$quant")
    local profile_flag=""

    if [[ "$ENABLE_PROFILING" == true ]]; then
        profile_flag="--profile"
    fi

    log_step "运行 Online Benchmark: ${display_name} / ${quant^^} (${num_prompts} prompts)"

    local bench_cmd="python3 -m vllm.entrypoints.cli.main bench serve"
    bench_cmd+=" --backend vllm"
    bench_cmd+=" --model ${model_path}"
    bench_cmd+=" --port ${SERVER_PORT}"
    bench_cmd+=" --dataset-name random"
    bench_cmd+=" --random-input-len ${INPUT_LEN}"
    bench_cmd+=" --random-output-len ${OUTPUT_LEN}"
    bench_cmd+=" --random-range-ratio 0.0"
    bench_cmd+=" --ignore-eos"
    bench_cmd+=" --num-prompts ${num_prompts}"
    bench_cmd+=" --request-rate ${REQUEST_RATE}"
    bench_cmd+=" --max-concurrency ${MAX_CONCURRENCY}"
    bench_cmd+=" --percentile-metrics ttft,tpot,itl,e2el"
    bench_cmd+=" --metric-percentiles 50,90,95,99"
    bench_cmd+=" --trust-remote-code"
    bench_cmd+=" --save-result"
    bench_cmd+=" --result-dir ${RESULT_DIR}"
    bench_cmd+=" --result-filename $(basename "${result_file}")"
    bench_cmd+=" ${profile_flag}"

    log_info "  Benchmark 命令: ${bench_cmd}"
    log_info "  Benchmark 日志: ${bench_log}"

    if eval "${bench_cmd}" 2>&1 | tee "${bench_log}"; then
        log_ok "Benchmark 完成: ${display_name} / ${quant^^}"
    else
        log_error "Benchmark 失败: ${display_name} / ${quant^^}"
        log_error "查看日志: ${bench_log}"
        return 1
    fi
}

# ======================== 核心: 停止 Server 并等待 Profiling Flush ========================

stop_server_and_flush_profiling() {
    local model_key="$1"
    local quant="$2"

    log_step "停止 Server 并等待 Profiling 数据落盘..."

    # 杀整棵进程树 (kill_server 内部会等待进程退出)
    kill_server "$SERVER_PORT"

    # 等待 NPU/GPU 显存释放
    # 进程退出后显存释放不是瞬时的, 需要等待设备上下文清理完成
    log_info "等待 NPU/GPU 显存释放 (10s)..."
    sleep 10

    if [[ "$ENABLE_PROFILING" == true ]]; then
        local profile_dir
        profile_dir=$(get_profile_dir "$model_key" "$quant")

        # 检查 profiling 输出
        local profile_files
        profile_files=$(find "${profile_dir}" -type f 2>/dev/null | wc -l)
        if [[ "$profile_files" -gt 0 ]]; then
            log_ok "Profiling 数据已保存: ${profile_dir} (${profile_files} 个文件)"
            ls -lh "${profile_dir}/" 2>/dev/null || true
        else
            log_warn "Profiling 目录为空: ${profile_dir}"
        fi
    fi
}

# ======================== 单组实验流程 ========================

run_single_experiment() {
    local model_key="$1"
    local quant="$2"
    local num_prompts="$3"
    local display_name
    display_name=$(get_model_field "$model_key" "DISPLAY")

    echo ""
    echo "================================================================"
    echo "  实验: ${display_name} / ${quant^^}  [${BENCH_MODE} 模式]"
    echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"

    local exp_start_time
    exp_start_time=$(date +%s)
    local bench_status=0

    if [[ "$BENCH_MODE" == "offline" ]]; then
        # Offline 模式: 不需要 server, 直接运行 benchmark
        if ! run_benchmark "$model_key" "$quant" "$num_prompts"; then
            bench_status=1
        fi
        # offline benchmark 退出后显存自动释放, 等一下确保清理完成
        log_info "等待 NPU/GPU 显存释放 (10s)..."
        sleep 10
    else
        # Online 模式: 需要 server
        # Step 1: 拉起 Server
        if ! start_server "$model_key" "$quant"; then
            log_error "跳过实验: ${display_name} / ${quant^^} (Server 启动失败)"
            return 1
        fi

        # Step 2: 运行 Benchmark
        if ! run_benchmark "$model_key" "$quant" "$num_prompts"; then
            bench_status=1
        fi

        # Step 3: 停止 Server, flush profiling
        stop_server_and_flush_profiling "$model_key" "$quant"
    fi

    local exp_end_time
    exp_end_time=$(date +%s)
    local exp_duration=$(( exp_end_time - exp_start_time ))

    if [[ $bench_status -eq 0 ]]; then
        log_ok "实验完成: ${display_name} / ${quant^^} (耗时: ${exp_duration}s)"
    else
        log_error "实验失败: ${display_name} / ${quant^^} (耗时: ${exp_duration}s)"
    fi

    return $bench_status
}

# ======================== 诊断模式 ========================

run_diagnostic() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              诊断模式 (Diagnostic Mode)                      ║"
    echo "║  模型: Qwen3-1.7B / BF16   模式: ${BENCH_MODE}                ║"
    echo "║  目标: 验证 benchmark 能跑通 + 结果能正确保存               ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    local diag_model="qwen3-1.7b"
    local diag_quant="bf16"
    local diag_num_prompts=4
    local diag_profile_dir
    diag_profile_dir=$(get_profile_dir "$diag_model" "$diag_quant")

    # 强制开启 profiling 用于诊断
    ENABLE_PROFILING=true

    log_step "[诊断 1/5] 检查模型路径..."
    local model_path
    model_path=$(get_model_path "$diag_model" "$diag_quant")
    if [[ -d "$model_path" ]]; then
        log_ok "模型路径存在: ${model_path}"
    else
        log_error "模型路径不存在: ${model_path}"
        log_error "请修改脚本中 MODEL_BASE 变量, 或使用 --model-base 参数指定"
        return 1
    fi

    log_step "[诊断 2/5] 检查 vllm CLI..."
    if python3 -m vllm.entrypoints.cli.main --help > /dev/null 2>&1; then
        log_ok "vllm CLI 可用"
    else
        log_error "vllm CLI 不可用, 请检查 vllm 安装"
        return 1
    fi

    log_step "[诊断 3/5] 启动 Server + 运行 Benchmark (${diag_num_prompts} prompts)..."
    if ! run_single_experiment "$diag_model" "$diag_quant" "$diag_num_prompts"; then
        log_error "诊断实验失败"
        return 1
    fi

    log_step "[诊断 4/5] 验证 benchmark 结果文件..."
    local result_file
    result_file=$(get_result_file "$diag_model" "$diag_quant")
    if [[ -f "$result_file" ]]; then
        log_ok "结果文件已生成: ${result_file}"
        log_info "结果内容预览:"
        python3 -c "
import json
with open('${result_file}') as f:
    data = json.load(f)
keys = ['completed', 'request_throughput', 'output_throughput',
        'total_token_throughput', 'mean_ttft_ms', 'mean_tpot_ms', 'mean_e2el_ms']
for k in keys:
    if k in data:
        v = data[k]
        if isinstance(v, float):
            print(f'  {k}: {v:.2f}')
        else:
            print(f'  {k}: {v}')
" 2>/dev/null || log_warn "无法解析结果文件"
    else
        log_error "结果文件未生成: ${result_file}"
    fi

    log_step "[诊断 5/5] 验证 Profiling 数据..."
    if [[ -d "$diag_profile_dir" ]]; then
        local file_count
        file_count=$(find "${diag_profile_dir}" -type f 2>/dev/null | wc -l)
        if [[ "$file_count" -gt 0 ]]; then
            log_ok "Profiling 数据已保存到: ${diag_profile_dir}"
            log_info "文件列表:"
            ls -lh "${diag_profile_dir}/" 2>/dev/null | head -10
            echo ""
            log_info "可通过以下方式解析 profiling 数据:"
            echo "  方式1 (Ascend): python3 -c \"from torch_npu.profiler.profiler import analyse; analyse(profiler_path='${diag_profile_dir}')\""
            echo "  方式2 (Perfetto): 打开 https://ui.perfetto.dev/ 上传 trace 文件"
        else
            log_warn "Profiling 目录为空 (可能 server 退出太快未来得及 flush)"
            log_info "建议: 增加 num_prompts 或 output_len 让推理时间更长"
        fi
    else
        log_error "Profiling 目录不存在: ${diag_profile_dir}"
    fi

    # 诊断模式也输出汇总表格 (方便验证)
    generate_summary_table

    echo ""
    echo "================================================================"
    echo "  诊断完成"
    echo "  结果目录: ${RESULT_DIR}"
    echo "  日志目录: ${LOG_DIR}"
    echo "  Profiling: ${diag_profile_dir}"
    echo "================================================================"
}

# ======================== 全量实验 ========================

run_full_benchmark() {
    # 确定要跑的模型和量化精度
    local models=("qwen3-1.7b" "pangu-7b" "qwen3-30b-a3b")
    local quants=("bf16" "w8a16" "w8a8")

    if [[ -n "$USER_MODELS" ]]; then
        IFS=',' read -ra models <<< "$USER_MODELS"
    fi
    if [[ -n "$USER_QUANTS" ]]; then
        IFS=',' read -ra quants <<< "$USER_QUANTS"
    fi

    # Profiling 开关
    if [[ "$NO_PROFILE" == true ]]; then
        ENABLE_PROFILING=false
        log_info "Profiling 已禁用"
    fi

    # 计算有效实验数 (跳过未配置的组合, 如 Qwen3-30B-A3B 无 W8A16)
    _QUANTS=("${quants[@]}")
    local valid_experiments
    valid_experiments=$(get_valid_experiments "${models[@]}")
    local total_experiments
    total_experiments=$(echo "$valid_experiments" | wc -l | tr -d ' ')

    # 列出跳过的组合
    local skipped_experiments=()
    for model_key in "${models[@]}"; do
        for quant in "${quants[@]}"; do
            if should_skip_experiment "$model_key" "$quant"; then
                local skip_display
                skip_display=$(get_model_field "$model_key" "DISPLAY")
                skipped_experiments+=("${skip_display}/${quant^^}")
            fi
        done
    done

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           vLLM Serving Benchmark (Ascend NPU)               ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Benchmark 模式: ${BENCH_MODE}"
    echo "  有效实验: ${total_experiments} 组"
    echo "  模型: ${models[*]}"
    echo "  量化: ${quants[*]}"
    if [[ ${#skipped_experiments[@]} -gt 0 ]]; then
        echo "  跳过: ${skipped_experiments[*]} (未配置模型路径)"
    fi
    echo "  输入长度: ${INPUT_LEN}, 输出长度: ${OUTPUT_LEN}"
    if [[ "$BENCH_MODE" == "offline" ]]; then
        local total_seqs=$((NUM_PROMPTS * NUM_SAMPLES_PER_PROMPT))
        echo "  负载 (per GPU): ${NUM_PROMPTS} prompts × n=${NUM_SAMPLES_PER_PROMPT} = ${total_seqs} 序列"
    else
        echo "  请求数量: ${NUM_PROMPTS}, 最大并发: ${MAX_CONCURRENCY}"
    fi
    echo "  gpu_memory_utilization: per-model (1.7B=0.4, 7B=0.6, 30B=0.8)"
    echo "  Profiling: ${ENABLE_PROFILING}"
    echo "  结果目录: ${RESULT_DIR}"
    echo ""

    local run_start_time
    run_start_time=$(date +%s)
    local exp_index=0
    local success_count=0
    local fail_count=0
    local skip_count=0
    local failed_experiments=()

    for model_key in "${models[@]}"; do
        for quant in "${quants[@]}"; do
            # 跳过未配置的组合
            if should_skip_experiment "$model_key" "$quant"; then
                local skip_display
                skip_display=$(get_model_field "$model_key" "DISPLAY")
                log_warn "跳过: ${skip_display} / ${quant^^} (未配置该量化精度的模型路径)"
                skip_count=$((skip_count + 1))
                continue
            fi

            exp_index=$((exp_index + 1))
            local display_name
            display_name=$(get_model_field "$model_key" "DISPLAY")

            echo ""
            log_info "========== 实验 ${exp_index}/${total_experiments}: ${display_name} / ${quant^^} =========="

            if run_single_experiment "$model_key" "$quant" "$NUM_PROMPTS"; then
                success_count=$((success_count + 1))
            else
                fail_count=$((fail_count + 1))
                failed_experiments+=("${display_name}/${quant^^}")
            fi

            # 实验间留出清理时间
            sleep 5
        done
    done

    local run_end_time
    run_end_time=$(date +%s)
    local total_duration=$(( run_end_time - run_start_time ))

    # 打印总结
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                     实验总结                                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  有效实验: ${total_experiments} 组"
    echo "  成功: ${success_count}"
    echo "  失败: ${fail_count}"
    echo "  跳过: ${skip_count}"
    echo "  总耗时: ${total_duration}s ($(( total_duration / 60 ))m $(( total_duration % 60 ))s)"
    echo ""

    if [[ $fail_count -gt 0 ]]; then
        echo "  失败的实验:"
        for exp in "${failed_experiments[@]}"; do
            echo "    - ${exp}"
        done
        echo ""
    fi

    echo "  结果文件:"
    ls -lh "${RESULT_DIR}/"*.json 2>/dev/null || echo "    (无)"
    echo ""

    if [[ "$ENABLE_PROFILING" == true ]]; then
        echo "  Profiling 数据:"
        for model_key in "${models[@]}"; do
            for quant in "${quants[@]}"; do
                if should_skip_experiment "$model_key" "$quant"; then
                    continue
                fi
                local pdir
                pdir=$(get_profile_dir "$model_key" "$quant")
                local fcount
                fcount=$(find "${pdir}" -type f 2>/dev/null | wc -l)
                echo "    ${pdir} (${fcount} files)"
            done
        done
        echo ""
    fi

    echo "  日志文件:"
    ls -lh "${LOG_DIR}/"*.log 2>/dev/null || echo "    (无)"
    echo ""

    # 生成汇总对比表格
    generate_summary_table

    return $fail_count
}

# ======================== 汇总表格生成 ========================

generate_summary_table() {
    log_step "生成汇总对比表格..."

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    python3 "${script_dir}/summarize_benchmark.py" "${RESULT_DIR}" --markdown
}

# ======================== 主入口 ========================

main() {
    declare_model_config
    init_dirs

    log_info "脚本启动: $(date '+%Y-%m-%d %H:%M:%S')"
    log_info "工作目录: $(pwd)"

    if [[ "$DIAGNOSTIC_MODE" == true ]]; then
        run_diagnostic
    else
        run_full_benchmark
    fi
}

main
