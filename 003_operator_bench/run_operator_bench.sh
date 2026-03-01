#!/bin/bash
set -euo pipefail

# ======================== 单算子 Benchmark: BF16 vs W8A8D ========================
# 完整流程:
#   1. 对 30B-A3B 和 718B 的所有 GEMM shape 做 BF16 vs W8A8D 对比
#   2. M-sweep: 找 breakeven point (30B shape 和 718B shape 各一条曲线)
#   3. 分析: 预测整模型 speedup, 对比 001 实测值

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ======================== 日志 ========================
RED='\033[31m'; GREEN='\033[32m'; YELLOW='\033[33m'; BLUE='\033[34m'; NC='\033[0m'
log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step()  { echo -e "\n${GREEN}========== $* ==========${NC}"; }

# ======================== 默认值 ========================
DECODE_BATCH=256
WARMUP=20
REPEATS=100
VALIDATE_SPEEDUP="0.89"       # 001 实测: 30B-A3B W8A8D = 0.89x
SKIP_SWEEP=false
DEVICE=0

# ======================== 参数解析 ========================
while [[ $# -gt 0 ]]; do
    case $1 in
        --decode-batch)   DECODE_BATCH="$2"; shift 2 ;;
        --warmup)         WARMUP="$2"; shift 2 ;;
        --repeats)        REPEATS="$2"; shift 2 ;;
        --validate)       VALIDATE_SPEEDUP="$2"; shift 2 ;;
        --skip-sweep)     SKIP_SWEEP=true; shift ;;
        --device)         DEVICE="$2"; shift 2 ;;
        *)                log_error "未知参数: $1"; exit 1 ;;
    esac
done

# ======================== 输出目录 ========================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${SCRIPT_DIR}/outputs/${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

log_info "参数: decode_batch=${DECODE_BATCH}, warmup=${WARMUP}, repeats=${REPEATS}"
log_info "设备: NPU ${DEVICE}"
log_info "输出: ${RUN_DIR}"

# 公共参数
BENCH_ARGS=(
    --decode-batch "${DECODE_BATCH}"
    --warmup "${WARMUP}"
    --repeats "${REPEATS}"
)

export ASCEND_RT_VISIBLE_DEVICES="${DEVICE}"
FAILED=0

# ======================== Step 1: 全量 shape benchmark ========================
log_step "Step 1: 全量 shape benchmark (30B-A3B + 718B)"

if ASCEND_RT_VISIBLE_DEVICES="${DEVICE}" \
   python3 "${SCRIPT_DIR}/bench_ops.py" \
       --all "${BENCH_ARGS[@]}" \
       --output "${RUN_DIR}/results_all.json" \
       2>&1 | tee "${RUN_DIR}/bench_all.log"; then
    log_ok "全量 benchmark 完成"
else
    log_error "全量 benchmark 失败"
    FAILED=$((FAILED + 1))
fi

# ======================== Step 2: M-sweep ========================
if [[ "${SKIP_SWEEP}" == "false" ]]; then
    log_step "Step 2: M-sweep (找 breakeven point)"

    # 30B-A3B 的 MoE expert shape: K=2048, N=1536
    log_info "Sweep: 30B-A3B shape (K=2048, N=1536)"
    if ASCEND_RT_VISIBLE_DEVICES="${DEVICE}" \
       python3 "${SCRIPT_DIR}/bench_ops.py" \
           --sweep-m --K 2048 --N 1536 \
           "${BENCH_ARGS[@]}" \
           --output "${RUN_DIR}/sweep_30b.json" \
           2>&1 | tee "${RUN_DIR}/sweep_30b.log"; then
        log_ok "30B sweep 完成"
    else
        log_error "30B sweep 失败"
        FAILED=$((FAILED + 1))
    fi

    # 718B 的 MoE expert shape: K=7680, N=4096
    log_info "Sweep: 718B shape (K=7680, N=4096)"
    if ASCEND_RT_VISIBLE_DEVICES="${DEVICE}" \
       python3 "${SCRIPT_DIR}/bench_ops.py" \
           --sweep-m --K 7680 --N 4096 \
           "${BENCH_ARGS[@]}" \
           --output "${RUN_DIR}/sweep_718b.json" \
           2>&1 | tee "${RUN_DIR}/sweep_718b.log"; then
        log_ok "718B sweep 完成"
    else
        log_error "718B sweep 失败"
        FAILED=$((FAILED + 1))
    fi
else
    log_info "跳过 M-sweep (--skip-sweep)"
fi

# ======================== Step 3: 分析 ========================
log_step "Step 3: 分析结果"

# 全量结果 + 验证
if [[ -f "${RUN_DIR}/results_all.json" ]]; then
    log_info "分析全量结果 (validate=${VALIDATE_SPEEDUP}x)"
    python3 "${SCRIPT_DIR}/analyze.py" \
        "${RUN_DIR}/results_all.json" \
        --validate "${VALIDATE_SPEEDUP}" \
        2>&1 | tee "${RUN_DIR}/analysis.txt"

    python3 "${SCRIPT_DIR}/analyze.py" \
        "${RUN_DIR}/results_all.json" \
        --validate "${VALIDATE_SPEEDUP}" \
        --markdown > "${RUN_DIR}/analysis.md"
    log_ok "全量分析: ${RUN_DIR}/analysis.md"
fi

# Sweep 结果
for sweep_file in "${RUN_DIR}"/sweep_*.json; do
    [[ -f "${sweep_file}" ]] || continue
    name=$(basename "${sweep_file}" .json)
    log_info "分析 ${name}"
    python3 "${SCRIPT_DIR}/analyze.py" "${sweep_file}" \
        2>&1 | tee -a "${RUN_DIR}/analysis.txt"
    python3 "${SCRIPT_DIR}/analyze.py" "${sweep_file}" \
        --markdown >> "${RUN_DIR}/analysis.md"
done


# 生成论证报告
log_info "生成论证报告..."
python3 "${SCRIPT_DIR}/generate_report.py" "${RUN_DIR}" \
    --output "${RUN_DIR}/report.md" 2>&1
log_ok "报告: ${RUN_DIR}/report.md"

# ======================== 完成 ========================
echo ""
log_ok "完成! 结果目录: ${RUN_DIR}"
echo ""
log_info "文件列表:"
ls -1 "${RUN_DIR}"/*.json "${RUN_DIR}"/*.txt "${RUN_DIR}"/*.md 2>/dev/null | while read -r f; do
    echo "  $(basename "$f")"
done
echo ""
[[ ${FAILED} -gt 0 ]] && log_warn "${FAILED} 个步骤失败"
exit ${FAILED}
