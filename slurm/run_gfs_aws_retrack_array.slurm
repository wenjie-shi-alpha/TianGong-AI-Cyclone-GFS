#!/bin/bash
#SBATCH --job-name=gfs_aws_retrack
#SBATCH --output=logs/gfs_retrack_%j.out
#SBATCH --error=logs/gfs_retrack_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=26
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --qos=normal
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# =================== 可调参数 (通过 sbatch --export=... 覆盖) ===================
DEFAULT_PROJECT_DIR="/scratch/users/ziqianx/TianGong-AI-Cyclone-GFS"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
TRACKS_CSV="${TRACKS_CSV:-}"
ENV_PREFIX="/scratch/users/ziqianx/conda/envs/cyclone"
LEAD_DAYS="${LEAD_DAYS:-10}"
STORMS_FILTER="${STORMS_FILTER:-}"       # 可选: 逗号分隔的 storm id
KEEP_NC="${KEEP_NC:-0}"
PIPELINE_MODE="${PIPELINE_MODE:-grib}"   # grib | gee | cds
REQUIRE_FULL_ENV_EXTRACTION="${REQUIRE_FULL_ENV_EXTRACTION:-1}"
GEE_PROJECT="${GEE_PROJECT:-eminent-glider-467006-r0}"
GEE_SERVICE_ACCOUNT_KEY_JSON="${GEE_SERVICE_ACCOUNT_KEY_JSON:-}"
GEE_OUTPUT_SUBDIR="${GEE_OUTPUT_SUBDIR:-gee_pipeline}"
GEE_ANALYSIS_ONLY="${GEE_ANALYSIS_ONLY:-0}"
GEE_MAX_FORECAST_HOUR="${GEE_MAX_FORECAST_HOUR:-120}"
GEE_TIME_WINDOW_HOURS="${GEE_TIME_WINDOW_HOURS:-6}"
GEE_TEMPORAL_SPAN_HOURS="${GEE_TEMPORAL_SPAN_HOURS:-120}"
GEE_MAX_STEPS="${GEE_MAX_STEPS:-40}"
GEE_SPATIAL_PAD_DEG="${GEE_SPATIAL_PAD_DEG:-12.0}"
GEE_SLEEP_SEC="${GEE_SLEEP_SEC:-0.5}"
CDS_OUTPUT_SUBDIR="${CDS_OUTPUT_SUBDIR:-cds_output}"
CDS_WORKERS="${CDS_WORKERS:-4}"
CDS_NO_CLEAN="${CDS_NO_CLEAN:-0}"
CDS_MAX_POINTS="${CDS_MAX_POINTS:-}"

# =================== 性能参数 (26 CPU / 200G RAM 优化) ===================
#
# 管线阶段与资源特征:
#   阶段         类型      瓶颈           资源消耗
#   ─────────────────────────────────────────────────
#   S3 下载      I/O密集   网络带宽        低CPU, 低内存
#   cfgrib解析   CPU密集   CPU + 内存      高CPU, ~4GB/进程
#   NC写入       I/O密集   磁盘/锁串行     低CPU, 内存峰值
#   追踪+提取    CPU密集   CPU + 内存      高CPU, ~8-12GB/进程
#
# 默认策略: 稳定优先 (降低并发争抢)
#   - 2 cycle-workers × 4 download-threads = 8 并发下载
#   - 10 parse-workers 共享进程池
#   - 4 extract-processes 并行分析
#   - 下载完整性: 至少满足 majority 比例后进入追踪
#
CYCLE_WORKERS="${CYCLE_WORKERS:-2}"
DOWNLOAD_THREADS="${DOWNLOAD_THREADS:-4}"
PARSE_WORKERS="${PARSE_WORKERS:-10}"
EXTRACT_PROCESSES="${EXTRACT_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GFS_DOWNLOAD_TIMEOUT_SEC="${GFS_DOWNLOAD_TIMEOUT_SEC:-1200}"
GFS_CFGRIB_TIMEOUT_SEC="${GFS_CFGRIB_TIMEOUT_SEC:-1200}"
MIN_GRIB_FILES_PER_CYCLE="${MIN_GRIB_FILES_PER_CYCLE:-20}"
MIN_FORECAST_SPAN_HOURS="${MIN_FORECAST_SPAN_HOURS:-96}"
MIN_GRIB_SUCCESS_RATIO="${MIN_GRIB_SUCCESS_RATIO:-0.60}"
SKIP_HEAD_CYCLES="${SKIP_HEAD_CYCLES:-7}"

# =================== 环境初始化 ===================
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
  fi
fi

if command -v module >/dev/null 2>&1; then
  for mod in mamba micromamba miniconda3 anaconda3 miniforge3 mambaforge; do
    if module --ignore_cache load "$mod" >/dev/null 2>&1 || module load "$mod" >/dev/null 2>&1; then
      echo "Loaded module: $mod"
      break
    fi
  done
fi

# 项目目录回退解析
if [[ ! -d "$PROJECT_DIR/src" ]]; then
  CANDIDATES=(
    "$DEFAULT_PROJECT_DIR"
    "/scratch/users/${USER}/TianGong-AI-Cyclone-GFS"
    "${SLURM_SUBMIT_DIR:-}"
  )
  for cand in "${CANDIDATES[@]}"; do
    [[ -z "$cand" ]] && continue
    if [[ -d "$cand/src" ]]; then
      PROJECT_DIR="$cand"
      break
    fi
  done
fi

if [[ ! -d "$PROJECT_DIR/src" ]]; then
  echo "❌ PROJECT_DIR invalid: $PROJECT_DIR"
  exit 2
fi

if [[ -z "$TRACKS_CSV" ]]; then
  TRACKS_CSV="$PROJECT_DIR/input/matched_cyclone_tracks.csv"
fi
if [[ ! -f "$TRACKS_CSV" ]]; then
  echo "❌ TRACKS_CSV not found: $TRACKS_CSV"
  exit 2
fi

# 输出目录: 直接放在共享存储上
OUTPUT_ROOT="$PROJECT_DIR/hpc_runs/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_ROOT"/{final_single_output,track_single,logs}

if [[ "$PIPELINE_MODE" != "grib" && "$PIPELINE_MODE" != "gee" && "$PIPELINE_MODE" != "cds" ]]; then
  echo "❌ Unsupported PIPELINE_MODE: $PIPELINE_MODE (expect grib, gee or cds)"
  exit 2
fi

if [[ "$REQUIRE_FULL_ENV_EXTRACTION" == "1" && "$PIPELINE_MODE" == "gee" ]]; then
  echo "❌ PIPELINE_MODE=gee does not support full GRIB->NC weather-system extraction."
  echo "   Use PIPELINE_MODE=grib for full tracking + environmental systems."
  echo "   If you intentionally want simplified GEE summaries, set REQUIRE_FULL_ENV_EXTRACTION=0."
  exit 2
fi

if [[ -n "$GEE_SERVICE_ACCOUNT_KEY_JSON" && ! -f "$GEE_SERVICE_ACCOUNT_KEY_JSON" ]]; then
  echo "❌ GEE_SERVICE_ACCOUNT_KEY_JSON not found: $GEE_SERVICE_ACCOUNT_KEY_JSON"
  exit 2
fi

# Conda/Mamba 环境
if [[ ! -d "$ENV_PREFIX" ]]; then
  echo "❌ ENV_PREFIX not found: $ENV_PREFIX"
  exit 2
fi
if command -v conda >/dev/null 2>&1; then
  ENV_RUNNER=(conda run -p "$ENV_PREFIX")
elif command -v mamba >/dev/null 2>&1; then
  ENV_RUNNER=(mamba run -p "$ENV_PREFIX")
elif command -v micromamba >/dev/null 2>&1; then
  ENV_RUNNER=(micromamba run -p "$ENV_PREFIX")
else
  echo "❌ Need conda/mamba/micromamba to run env prefix: $ENV_PREFIX"
  exit 2
fi

# 参数校验
for varname in CYCLE_WORKERS DOWNLOAD_THREADS PARSE_WORKERS EXTRACT_PROCESSES BATCH_SIZE SKIP_HEAD_CYCLES MIN_GRIB_FILES_PER_CYCLE MIN_FORECAST_SPAN_HOURS; do
  eval "val=\$$varname"
  if [[ ! "$val" =~ ^[0-9]+$ ]]; then
    echo "❌ $varname must be a non-negative integer, got: $val"
    exit 2
  fi
done

if ! awk "BEGIN{exit !($MIN_GRIB_SUCCESS_RATIO >= 0 && $MIN_GRIB_SUCCESS_RATIO <= 1)}"; then
  echo "❌ MIN_GRIB_SUCCESS_RATIO must be in [0,1], got: $MIN_GRIB_SUCCESS_RATIO"
  exit 2
fi

ulimit -n 8192 || true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export GFS_DOWNLOAD_TIMEOUT_SEC
export GFS_CFGRIB_TIMEOUT_SEC
export MIN_GRIB_FILES_PER_CYCLE
export MIN_FORECAST_SPAN_HOURS
export MIN_GRIB_SUCCESS_RATIO
export SKIP_HEAD_CYCLES

# =================== 工作目录设置 ===================
# 使用节点本地存储加速I/O, 输出通过 symlink 直接写入共享存储
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
RUN_DIR="$LOCAL_BASE/gfs_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"/{logs,input,output,data}
cd "$RUN_DIR"

cp -f "$TRACKS_CSV" "input/tracks.csv"

# Symlink 输出目录到共享存储 (结果直接写入 OUTPUT_ROOT, 不需要事后同步)
if [[ "$PIPELINE_MODE" == "grib" || "$PIPELINE_MODE" == "cds" ]]; then
  ln -sfn "$OUTPUT_ROOT/final_single_output" "$RUN_DIR/final_single_output"
  ln -sfn "$OUTPUT_ROOT/track_single" "$RUN_DIR/track_single"
  if [[ "$KEEP_NC" == "1" ]]; then
    mkdir -p "$OUTPUT_ROOT/grib_nc"
    ln -sfn "$OUTPUT_ROOT/grib_nc" "$RUN_DIR/data/grib_nc"
  fi
else
  mkdir -p "$OUTPUT_ROOT/$GEE_OUTPUT_SUBDIR"
  ln -sfn "$OUTPUT_ROOT/$GEE_OUTPUT_SUBDIR" "$RUN_DIR/$GEE_OUTPUT_SUBDIR"
fi

# 写入作业状态文件
STATUS_FILE="$OUTPUT_ROOT/job_status_${SLURM_JOB_ID}.txt"
write_status() {
  cat > "$STATUS_FILE" <<EOF
state=$1
updated_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
job_id=${SLURM_JOB_ID:-}
pipeline_mode=$PIPELINE_MODE
run_dir=$RUN_DIR
project_dir=$PROJECT_DIR
output_root=$OUTPUT_ROOT
EOF
}

# 退出时同步日志
cleanup() {
  local exit_code=$?
  # 同步本地日志到共享存储
  if [[ -d "$RUN_DIR/logs" ]]; then
    rsync -a "$RUN_DIR/logs/" "$OUTPUT_ROOT/logs/" 2>/dev/null || true
  fi
  if [[ "$exit_code" -eq 0 ]]; then
    write_status "finished"
  else
    write_status "failed($exit_code)"
  fi
}
trap cleanup EXIT
write_status "running"

# =================== 打印作业配置 ===================
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 26)}"
echo "=== Job Configuration ==="
echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "RUN_DIR=$RUN_DIR"
echo "PIPELINE_MODE=$PIPELINE_MODE"
echo "TRACKS_CSV=$TRACKS_CSV"
echo "STORMS_FILTER=${STORMS_FILTER:-<ALL>}"
echo ""
echo "=== Resource Allocation ==="
echo "CPUS=$CPUS  MEM=200G"
echo "CYCLE_WORKERS=$CYCLE_WORKERS"
echo "DOWNLOAD_THREADS=$DOWNLOAD_THREADS"
echo "PARSE_WORKERS=$PARSE_WORKERS"
echo "EXTRACT_PROCESSES=$EXTRACT_PROCESSES"
echo "BATCH_SIZE=$BATCH_SIZE (0=auto)"
echo "GFS_DOWNLOAD_TIMEOUT_SEC=$GFS_DOWNLOAD_TIMEOUT_SEC"
echo "GFS_CFGRIB_TIMEOUT_SEC=$GFS_CFGRIB_TIMEOUT_SEC"
echo "MIN_GRIB_FILES_PER_CYCLE=$MIN_GRIB_FILES_PER_CYCLE"
echo "MIN_FORECAST_SPAN_HOURS=$MIN_FORECAST_SPAN_HOURS"
echo "MIN_GRIB_SUCCESS_RATIO=$MIN_GRIB_SUCCESS_RATIO"
echo "SKIP_HEAD_CYCLES=$SKIP_HEAD_CYCLES"
echo ""

# =================== 执行管线 ===================
STATUS=0
LOG_FILE="$OUTPUT_ROOT/logs/run_${SLURM_JOB_ID}.log"

if [[ "$PIPELINE_MODE" == "grib" || "$PIPELINE_MODE" == "cds" ]]; then
  if [[ "$PIPELINE_MODE" == "cds" ]]; then
    echo "ℹ️ PIPELINE_MODE=cds uses GFS/AWS GRIB cycle workflow."
  fi

  CMD=(
    python3 -u "$PROJECT_DIR/src/assemble_and_run_grib.py"
    --tracks "input/tracks.csv"
    --cycle-workers "$CYCLE_WORKERS"
    --download-threads "$DOWNLOAD_THREADS"
    --parse-workers "$PARSE_WORKERS"
    --processes "$EXTRACT_PROCESSES"
    --grib-dir "$RUN_DIR/grib_cache"
    --batch-size "$BATCH_SIZE"
    --skip-head-cycles "$SKIP_HEAD_CYCLES"
  )

  # 如果指定了 storm filter, 传入 --storms 参数
  if [[ -n "$STORMS_FILTER" ]]; then
    IFS=',' read -r -a STORMS_ARR <<< "$STORMS_FILTER"
    CMD+=(--storms)
    for sid in "${STORMS_ARR[@]}"; do
      sid="${sid//[[:space:]]/}"
      [[ -n "$sid" ]] && CMD+=("$sid")
    done
  fi

  if [[ "$KEEP_NC" == "1" ]]; then
    CMD+=(--keep-nc --keep-grib)
  fi

  set +e
  "${ENV_RUNNER[@]}" "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
  STATUS=${PIPESTATUS[0]}
  set -e

else
  # GEE 管线: 逐 cycle 处理
  GEE_OUT_DIR="$RUN_DIR/$GEE_OUTPUT_SUBDIR"
  mkdir -p "$GEE_OUT_DIR"

  # 计算需要处理的 cycles
  CYCLE_LIST_FILE="$RUN_DIR/input/gee_cycles.txt"
  CYCLE_SELECT_SCRIPT="$RUN_DIR/_gee_cycle_select.py"
  cat > "$CYCLE_SELECT_SCRIPT" <<'PY'
import os, sys, io, contextlib
from datetime import datetime
project_dir, tracks_csv, lead_days, storms_filter = sys.argv[1:]
lead_days = int(lead_days)
sys.path.insert(0, os.path.join(project_dir, "src"))
from assemble_and_run_grib import compute_needed_cycles
storm_filter = None
if storms_filter.strip():
    storm_filter = [s.strip() for s in storms_filter.split(",") if s.strip()]
with contextlib.redirect_stdout(io.StringIO()):
    cycle_dts, _ = compute_needed_cycles(tracks_csv, lead_days=lead_days, storm_filter=storm_filter)
for dt in cycle_dts:
    print(dt.strftime("%Y%m%dT%H"))
PY

  set +e
  "${ENV_RUNNER[@]}" python3 "$CYCLE_SELECT_SCRIPT" "$PROJECT_DIR" "input/tracks.csv" "$LEAD_DAYS" "$STORMS_FILTER" > "$CYCLE_LIST_FILE" 2>&1
  GEE_SELECT_STATUS=$?
  set -e

  if [[ "$GEE_SELECT_STATUS" -ne 0 ]]; then
    echo "❌ Failed to compute GEE cycles (exit=$GEE_SELECT_STATUS)"
    cat "$CYCLE_LIST_FILE"
    exit "$GEE_SELECT_STATUS"
  fi

  mapfile -t CYCLES < "$CYCLE_LIST_FILE"
  if [[ ${#CYCLES[@]} -eq 0 ]]; then
    echo "⏹️ No cycles to process."
    exit 0
  fi
  echo "📋 GEE cycles to process: ${#CYCLES[@]}"

  STORM_ARGS=()
  if [[ -n "$STORMS_FILTER" ]]; then
    IFS=',' read -r -a STORMS_ARR <<< "$STORMS_FILTER"
    for sid in "${STORMS_ARR[@]}"; do
      sid="${sid//[[:space:]]/}"
      [[ -n "$sid" ]] && STORM_ARGS+=("$sid")
    done
  fi

  for cycle_tag in "${CYCLES[@]}"; do
    cycle_iso="${cycle_tag:0:4}-${cycle_tag:4:2}-${cycle_tag:6:2}T${cycle_tag:9:2}:00:00Z"
    echo "▶️ GEE cycle: $cycle_tag ($cycle_iso)" | tee -a "$LOG_FILE"

    GEE_CMD=(
      python3 -u "$PROJECT_DIR/src/gee_gfs_pipeline.py"
      --initials-csv "input/tracks.csv"
      --start-time "$cycle_iso"
      --time-window-hours "$GEE_TIME_WINDOW_HOURS"
      --temporal-span-hours "$GEE_TEMPORAL_SPAN_HOURS"
      --max-steps "$GEE_MAX_STEPS"
      --max-forecast-hour "$GEE_MAX_FORECAST_HOUR"
      --spatial-pad-deg "$GEE_SPATIAL_PAD_DEG"
      --sleep-sec "$GEE_SLEEP_SEC"
      --output-dir "$GEE_OUT_DIR"
    )
    [[ "$GEE_ANALYSIS_ONLY" == "1" ]] && GEE_CMD+=(--analysis-only)
    [[ -n "$GEE_PROJECT" ]] && GEE_CMD+=(--project "$GEE_PROJECT")
    [[ -n "$GEE_SERVICE_ACCOUNT_KEY_JSON" ]] && GEE_CMD+=(--service-account-key-json "$GEE_SERVICE_ACCOUNT_KEY_JSON")
    if [[ ${#STORM_ARGS[@]} -gt 0 ]]; then
      GEE_CMD+=(--storm-id "${STORM_ARGS[@]}")
    fi

    set +e
    "${ENV_RUNNER[@]}" "${GEE_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
    CYCLE_STATUS=${PIPESTATUS[0]}
    set -e
    if [[ "$CYCLE_STATUS" -ne 0 ]]; then
      STATUS="$CYCLE_STATUS"
      echo "❌ GEE cycle failed: $cycle_tag (exit=$CYCLE_STATUS)" | tee -a "$LOG_FILE"
      break
    fi
  done
fi

# =================== 最终状态 ===================
echo ""
echo "Pipeline exit code: $STATUS"
echo "Output directory: $OUTPUT_ROOT"

exit "$STATUS"
