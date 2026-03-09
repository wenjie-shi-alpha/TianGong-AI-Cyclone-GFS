#!/usr/bin/env bash
# =============================================================================
# submit_gfs_aws_retrack.sh - 提交 GFS 台风追踪管线到 Slurm
#
# 用法:
#   bash submit_gfs_aws_retrack.sh                    # 使用默认参数
#   STORMS_FILTER="AL092023" bash submit_gfs_aws_retrack.sh   # 只处理特定台风
# =============================================================================
set -euo pipefail

DEFAULT_PROJECT_DIR="/scratch/users/ziqianx/TianGong-AI-Cyclone-GFS"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
TRACKS_CSV="${TRACKS_CSV:-}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-}"
STORMS_FILTER="${STORMS_FILTER:-}"
PIPELINE_MODE="${PIPELINE_MODE:-grib}"
KEEP_NC="${KEEP_NC:-0}"
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

# OnDemand 脚本同步
OOD_SYNC="${OOD_SYNC:-1}"
OOD_PROJECT_DIR="${OOD_PROJECT_DIR:-/home/users/ziqianx/ondemand/data/sys/myjobs/projects/default/3}"
OOD_SCRIPT_NAME="${OOD_SCRIPT_NAME:-run_cyclone_track_extract.slurm}"

# ── 路径解析 ──
if [[ ! -d "$PROJECT_DIR/src" ]]; then
  CANDIDATES=(
    "$DEFAULT_PROJECT_DIR"
    "/scratch/users/${USER}/TianGong-AI-Cyclone-GFS"
    "${PWD}"
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

if [[ -z "$SBATCH_SCRIPT" ]]; then
  SBATCH_SCRIPT="$PROJECT_DIR/slurm/run_gfs_aws_retrack_array.slurm"
fi
if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "❌ SBATCH script not found: $SBATCH_SCRIPT"
  exit 2
fi

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

# ── OnDemand 同步 ──
if [[ "$OOD_SYNC" == "1" ]]; then
  echo "Syncing sbatch script to OnDemand project dir ..."
  mkdir -p "$OOD_PROJECT_DIR"
  cp -f "$SBATCH_SCRIPT" "$OOD_PROJECT_DIR/$OOD_SCRIPT_NAME"
  chmod 644 "$OOD_PROJECT_DIR/$OOD_SCRIPT_NAME"
  echo "OOD script ready: $OOD_PROJECT_DIR/$OOD_SCRIPT_NAME"
fi

# ── 确保 logs 目录存在 ──
mkdir -p "$PROJECT_DIR/logs"

# ── 提交作业 ──
echo "=== Submitting GFS Retrack Pipeline ==="
echo "PROJECT_DIR=$PROJECT_DIR"
echo "TRACKS_CSV=$TRACKS_CSV"
echo "PIPELINE_MODE=$PIPELINE_MODE"
echo "STORMS_FILTER=${STORMS_FILTER:-<ALL>}"

JOB_RAW="$(
sbatch --parsable \
  --chdir="$PROJECT_DIR" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",TRACKS_CSV="$TRACKS_CSV",STORMS_FILTER="$STORMS_FILTER",KEEP_NC="$KEEP_NC",PIPELINE_MODE="$PIPELINE_MODE",REQUIRE_FULL_ENV_EXTRACTION="$REQUIRE_FULL_ENV_EXTRACTION",GEE_PROJECT="$GEE_PROJECT",GEE_SERVICE_ACCOUNT_KEY_JSON="$GEE_SERVICE_ACCOUNT_KEY_JSON",GEE_OUTPUT_SUBDIR="$GEE_OUTPUT_SUBDIR",GEE_ANALYSIS_ONLY="$GEE_ANALYSIS_ONLY",GEE_MAX_FORECAST_HOUR="$GEE_MAX_FORECAST_HOUR",GEE_TIME_WINDOW_HOURS="$GEE_TIME_WINDOW_HOURS",GEE_TEMPORAL_SPAN_HOURS="$GEE_TEMPORAL_SPAN_HOURS",GEE_MAX_STEPS="$GEE_MAX_STEPS",GEE_SPATIAL_PAD_DEG="$GEE_SPATIAL_PAD_DEG",GEE_SLEEP_SEC="$GEE_SLEEP_SEC",CDS_OUTPUT_SUBDIR="$CDS_OUTPUT_SUBDIR",CDS_WORKERS="$CDS_WORKERS",CDS_NO_CLEAN="$CDS_NO_CLEAN",CDS_MAX_POINTS="$CDS_MAX_POINTS" \
  "$SBATCH_SCRIPT"
)"
JOB_ID="${JOB_RAW%%;*}"
echo "✅ Job submitted: $JOB_ID"
echo "   Output: $PROJECT_DIR/hpc_runs/$JOB_ID/"
echo "   Logs:   $PROJECT_DIR/logs/gfs_retrack_${JOB_ID}.out"
