#!/usr/bin/env bash
set -euo pipefail

DEFAULT_PROJECT_DIR="/scratch/users/ziqianx/TianGong-AI-Cyclone-GFS"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
TRACKS_CSV="${TRACKS_CSV:-}"
# 强制使用指定环境（用户要求）
ENV_PREFIX="/scratch/users/ziqianx/conda/envs/cyclone"
LEAD_DAYS="${LEAD_DAYS:-10}"
CYCLES_PER_TASK="${CYCLES_PER_TASK:-32}"
STORMS_FILTER="${STORMS_FILTER:-}"      # optional comma-separated storm ids
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-6}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-}"
MERGE_OUTPUT_ROOT="${MERGE_OUTPUT_ROOT:-}"
PIPELINE_MODE="${PIPELINE_MODE:-grib}"  # grib | gee | cds
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
ARCHIVE_ON_COMPLETE="${ARCHIVE_ON_COMPLETE:-1}"
ARCHIVE_NAME="${ARCHIVE_NAME:-}"
ARCHIVE_DEPENDENCY="${ARCHIVE_DEPENDENCY:-afterany}"  # afterany | afterok
ARCHIVE_SBATCH_SCRIPT="${ARCHIVE_SBATCH_SCRIPT:-}"
OOD_SYNC="${OOD_SYNC:-1}"
OOD_PROJECT_DIR="${OOD_PROJECT_DIR:-/home/users/ziqianx/ondemand/data/sys/myjobs/projects/default/3}"
OOD_SCRIPT_NAME="${OOD_SCRIPT_NAME:-run_cyclone_track_extract.slurm}"

# Try robust fallback resolution when caller uses an old HOME default.
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

if [[ -z "$ARCHIVE_SBATCH_SCRIPT" ]]; then
  ARCHIVE_SBATCH_SCRIPT="$PROJECT_DIR/slurm/package_merge_output.slurm"
fi
if [[ "$ARCHIVE_ON_COMPLETE" == "1" && ! -f "$ARCHIVE_SBATCH_SCRIPT" ]]; then
  echo "❌ ARCHIVE_SBATCH_SCRIPT not found: $ARCHIVE_SBATCH_SCRIPT"
  exit 2
fi

if [[ -z "$MERGE_OUTPUT_ROOT" ]]; then
  MERGE_OUTPUT_ROOT="$PROJECT_DIR/hpc_runs/retrack_resume"
fi

if [[ "$PIPELINE_MODE" != "grib" && "$PIPELINE_MODE" != "gee" && "$PIPELINE_MODE" != "cds" ]]; then
  echo "❌ Unsupported PIPELINE_MODE: $PIPELINE_MODE (expect grib, gee or cds)"
  exit 2
fi

if [[ "$PIPELINE_MODE" == "cds" ]]; then
  echo "ℹ️ PIPELINE_MODE=cds uses the GFS/AWS GRIB cycle workflow (same logic as grib)."
fi

if [[ "$REQUIRE_FULL_ENV_EXTRACTION" == "1" && "$PIPELINE_MODE" == "gee" ]]; then
  echo "❌ PIPELINE_MODE=gee does not support full GRIB->NC weather-system extraction."
  echo "   Use PIPELINE_MODE=grib for full tracking + environmental systems."
  echo "   If you intentionally want simplified GEE summaries, set REQUIRE_FULL_ENV_EXTRACTION=0."
  exit 2
fi

if [[ ! "$ARRAY_CONCURRENCY" =~ ^[0-9]+$ ]] || [[ "$ARRAY_CONCURRENCY" -lt 1 ]]; then
  echo "❌ ARRAY_CONCURRENCY must be a positive integer, got: $ARRAY_CONCURRENCY"
  exit 2
fi

if [[ -n "$GEE_SERVICE_ACCOUNT_KEY_JSON" && ! -f "$GEE_SERVICE_ACCOUNT_KEY_JSON" ]]; then
  echo "❌ GEE_SERVICE_ACCOUNT_KEY_JSON not found: $GEE_SERVICE_ACCOUNT_KEY_JSON"
  exit 2
fi

if [[ "$ARCHIVE_ON_COMPLETE" == "1" ]]; then
  if [[ "$ARCHIVE_DEPENDENCY" != "afterany" && "$ARCHIVE_DEPENDENCY" != "afterok" ]]; then
    echo "❌ ARCHIVE_DEPENDENCY must be one of: afterany, afterok"
    exit 2
  fi
fi

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

if [[ "$OOD_SYNC" == "1" ]]; then
  echo "Syncing sbatch script to OnDemand project dir ..."
  mkdir -p "$OOD_PROJECT_DIR"
  cp -f "$SBATCH_SCRIPT" "$OOD_PROJECT_DIR/$OOD_SCRIPT_NAME"
  chmod 644 "$OOD_PROJECT_DIR/$OOD_SCRIPT_NAME"
  if [[ ! -r "$OOD_PROJECT_DIR/$OOD_SCRIPT_NAME" ]]; then
    echo "❌ OOD script copy failed or unreadable: $OOD_PROJECT_DIR/$OOD_SCRIPT_NAME"
    exit 2
  fi
  echo "OOD script ready: $OOD_PROJECT_DIR/$OOD_SCRIPT_NAME"
fi

TOTAL_CYCLES="$(
"${ENV_RUNNER[@]}" python3 - "$PROJECT_DIR" "$TRACKS_CSV" "$LEAD_DAYS" "$STORMS_FILTER" "$MERGE_OUTPUT_ROOT" "$PIPELINE_MODE" <<'PY'
import os
import sys
import io
import contextlib
from pathlib import Path
from datetime import datetime

project_dir, tracks_csv, lead_days, storms_filter, merge_root, pipeline_mode = sys.argv[1:]
lead_days = int(lead_days)
sys.path.insert(0, os.path.join(project_dir, "src"))
from assemble_and_run_grib import compute_needed_cycles

storm_filter = None
if storms_filter.strip():
    storm_filter = [s.strip() for s in storms_filter.split(",") if s.strip()]

with contextlib.redirect_stdout(io.StringIO()):
    cycle_dts, _ = compute_needed_cycles(
        tracks_csv,
        lead_days=lead_days,
        storm_filter=storm_filter,
    )

all_tags = [dt.strftime("%Y%m%dT%H") for dt in cycle_dts]
if pipeline_mode == "gee":
    todo = len(all_tags)
else:
    done_stems = set()
    final_dir = Path(merge_root) / "final_single_output"
    if final_dir.exists():
        for jf in final_dir.glob("*_TC_Analysis_*.json"):
            stem = jf.stem
            if "_TC_Analysis_" in stem:
                done_stems.add(stem.split("_TC_Analysis_", 1)[0])

    todo = 0
    for tag in all_tags:
        init_tag = datetime.strptime(tag, "%Y%m%dT%H").strftime("%Y-%m-%d%H")
        nc_stem = f"gfs_{init_tag}_f000_f240_6h"
        if nc_stem not in done_stems:
            todo += 1

print(todo)
PY
)"

if [[ "$TOTAL_CYCLES" -eq 0 ]]; then
  echo "⏹️ No cycles to process."
  exit 0
fi

TASKS=$(( (TOTAL_CYCLES + CYCLES_PER_TASK - 1) / CYCLES_PER_TASK ))
LAST_TASK=$(( TASKS - 1 ))

echo "TOTAL_CYCLES=$TOTAL_CYCLES"
echo "CYCLES_PER_TASK=$CYCLES_PER_TASK"
echo "TASKS=$TASKS"
echo "PIPELINE_MODE=$PIPELINE_MODE"
echo "Submitting: --array=0-${LAST_TASK}%${ARRAY_CONCURRENCY}"

mkdir -p "$PROJECT_DIR/logs"

ARRAY_JOB_RAW="$(
sbatch --parsable \
  --chdir="$PROJECT_DIR" \
  --array="0-${LAST_TASK}%${ARRAY_CONCURRENCY}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",TRACKS_CSV="$TRACKS_CSV",LEAD_DAYS="$LEAD_DAYS",CYCLES_PER_TASK="$CYCLES_PER_TASK",STORMS_FILTER="$STORMS_FILTER",ARRAY_CONCURRENCY="$ARRAY_CONCURRENCY",MERGE_OUTPUT_ROOT="$MERGE_OUTPUT_ROOT",PIPELINE_MODE="$PIPELINE_MODE",REQUIRE_FULL_ENV_EXTRACTION="$REQUIRE_FULL_ENV_EXTRACTION",GEE_PROJECT="$GEE_PROJECT",GEE_SERVICE_ACCOUNT_KEY_JSON="$GEE_SERVICE_ACCOUNT_KEY_JSON",GEE_OUTPUT_SUBDIR="$GEE_OUTPUT_SUBDIR",GEE_ANALYSIS_ONLY="$GEE_ANALYSIS_ONLY",GEE_MAX_FORECAST_HOUR="$GEE_MAX_FORECAST_HOUR",GEE_TIME_WINDOW_HOURS="$GEE_TIME_WINDOW_HOURS",GEE_TEMPORAL_SPAN_HOURS="$GEE_TEMPORAL_SPAN_HOURS",GEE_MAX_STEPS="$GEE_MAX_STEPS",GEE_SPATIAL_PAD_DEG="$GEE_SPATIAL_PAD_DEG",GEE_SLEEP_SEC="$GEE_SLEEP_SEC",CDS_OUTPUT_SUBDIR="$CDS_OUTPUT_SUBDIR",CDS_WORKERS="$CDS_WORKERS",CDS_NO_CLEAN="$CDS_NO_CLEAN",CDS_MAX_POINTS="$CDS_MAX_POINTS" \
  "$SBATCH_SCRIPT"
)"
ARRAY_JOB_ID="${ARRAY_JOB_RAW%%;*}"
echo "✅ Array job submitted: $ARRAY_JOB_ID"

if [[ "$ARCHIVE_ON_COMPLETE" == "1" ]]; then
  if [[ -z "$ARCHIVE_NAME" ]]; then
    RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    ARCHIVE_NAME="retrack_${PIPELINE_MODE}_${ARRAY_JOB_ID}_${RUN_STAMP}.tar.gz"
  fi

  PACK_JOB_RAW="$(
  sbatch --parsable \
    --dependency="${ARCHIVE_DEPENDENCY}:${ARRAY_JOB_ID}" \
    --chdir="$PROJECT_DIR" \
    --export=ALL,MERGE_OUTPUT_ROOT="$MERGE_OUTPUT_ROOT",PIPELINE_MODE="$PIPELINE_MODE",GEE_OUTPUT_SUBDIR="$GEE_OUTPUT_SUBDIR",CDS_OUTPUT_SUBDIR="$CDS_OUTPUT_SUBDIR",ARCHIVE_NAME="$ARCHIVE_NAME" \
    "$ARCHIVE_SBATCH_SCRIPT"
  )"
  PACK_JOB_ID="${PACK_JOB_RAW%%;*}"

  echo "📦 Archive job submitted: $PACK_JOB_ID"
  echo "   dependency=${ARCHIVE_DEPENDENCY}:${ARRAY_JOB_ID}"
  echo "   archive=$MERGE_OUTPUT_ROOT/$ARCHIVE_NAME"
else
  echo "ℹ️ ARCHIVE_ON_COMPLETE=0, skip archive job submission."
fi
