#!/bin/bash
#SBATCH --job-name=gfs_aws_retrack
#SBATCH --output=logs/gfs_aws_retrack_%A_%a.out
#SBATCH --error=logs/gfs_aws_retrack_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# --------------- Runtime knobs (override via sbatch --export=...) ---------------
DEFAULT_PROJECT_DIR="/scratch/users/ziqianx/TianGong-AI-Cyclone-GFS"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
TRACKS_CSV="${TRACKS_CSV:-}"
# 强制使用指定环境（用户要求）
ENV_PREFIX="/scratch/users/ziqianx/conda/envs/cyclone"
LEAD_DAYS="${LEAD_DAYS:-10}"
CYCLES_PER_TASK="${CYCLES_PER_TASK:-32}"
STORMS_FILTER="${STORMS_FILTER:-}"    # optional comma-separated storm ids
KEEP_NC="${KEEP_NC:-0}"
PIPELINE_MODE="${PIPELINE_MODE:-grib}"  # grib | gee | cds
REQUIRE_FULL_ENV_EXTRACTION="${REQUIRE_FULL_ENV_EXTRACTION:-1}"
# 1: resume mode (skip cycles already merged under MERGE_OUTPUT_ROOT/final_single_output)
# 0: always process all computed cycles
RESUME_COMPLETED_CYCLES="${RESUME_COMPLETED_CYCLES:-1}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-1}"  # passed from submit script
MAX_GLOBAL_DOWNLOAD_THREADS="${MAX_GLOBAL_DOWNLOAD_THREADS:-192}"
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

# Optional output merge target (shared filesystem).
# If empty, outputs stay in per-task RUN_DIR.
MERGE_OUTPUT_ROOT="${MERGE_OUTPUT_ROOT:-}"

# --------------- Environment bootstrap ---------------
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/etc/profile.d/modules.sh
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

# Try robust fallback resolution when running from OnDemand copies.
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

if [[ -z "$MERGE_OUTPUT_ROOT" ]]; then
  MERGE_OUTPUT_ROOT="$PROJECT_DIR/hpc_runs/${SLURM_JOB_ID}"
fi

if [[ "$PIPELINE_MODE" != "grib" && "$PIPELINE_MODE" != "gee" && "$PIPELINE_MODE" != "cds" ]]; then
  echo "❌ Unsupported PIPELINE_MODE: $PIPELINE_MODE (expect grib, gee or cds)"
  exit 2
fi

if [[ "$RESUME_COMPLETED_CYCLES" != "0" && "$RESUME_COMPLETED_CYCLES" != "1" ]]; then
  echo "❌ RESUME_COMPLETED_CYCLES must be 0 or 1, got: $RESUME_COMPLETED_CYCLES"
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

# --------------- Resource-aware tuning (single node) ---------------
CPUS="${SLURM_CPUS_PER_TASK:-64}"

if [[ -z "${CYCLE_WORKERS:-}" ]]; then
  CYCLE_WORKERS=$(( CPUS / 8 ))
  (( CYCLE_WORKERS < 2 )) && CYCLE_WORKERS=2
  (( CYCLE_WORKERS > 8 )) && CYCLE_WORKERS=8
fi

if [[ -z "${DOWNLOAD_THREADS:-}" ]]; then
  DOWNLOAD_THREADS=$(( CPUS / 4 ))
  (( DOWNLOAD_THREADS < 8 )) && DOWNLOAD_THREADS=8
  (( DOWNLOAD_THREADS > 24 )) && DOWNLOAD_THREADS=24
fi

# Guardrail: cap per-task downloader threads based on array concurrency budget,
# so aggregate S3 request fan-out stays bounded.
if [[ "$ARRAY_CONCURRENCY" =~ ^[0-9]+$ ]] && [[ "$MAX_GLOBAL_DOWNLOAD_THREADS" =~ ^[0-9]+$ ]]; then
  (( ARRAY_CONCURRENCY < 1 )) && ARRAY_CONCURRENCY=1
  (( MAX_GLOBAL_DOWNLOAD_THREADS < 8 )) && MAX_GLOBAL_DOWNLOAD_THREADS=8
  PER_TASK_DL_CAP=$(( MAX_GLOBAL_DOWNLOAD_THREADS / ARRAY_CONCURRENCY ))
  (( PER_TASK_DL_CAP < 4 )) && PER_TASK_DL_CAP=4
  if (( DOWNLOAD_THREADS > PER_TASK_DL_CAP )); then
    echo "⚠️ DOWNLOAD_THREADS capped: ${DOWNLOAD_THREADS} -> ${PER_TASK_DL_CAP} (ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY}, MAX_GLOBAL_DOWNLOAD_THREADS=${MAX_GLOBAL_DOWNLOAD_THREADS})"
    DOWNLOAD_THREADS="$PER_TASK_DL_CAP"
  fi
fi

if [[ -z "${PARSE_WORKERS:-}" ]]; then
  PARSE_WORKERS=$(( CPUS - 8 ))
  (( PARSE_WORKERS < 8 )) && PARSE_WORKERS=8
fi

if [[ -z "${EXTRACT_PROCESSES:-}" ]]; then
  EXTRACT_PROCESSES=$(( CPUS / 2 ))
  (( EXTRACT_PROCESSES < 4 )) && EXTRACT_PROCESSES=4
  (( EXTRACT_PROCESSES > 32 )) && EXTRACT_PROCESSES=32
fi

ulimit -n 8192 || true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

# --------------- Local fast workspace (avoid shared FS bottleneck) ---------------
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  IS_ARRAY_JOB=1
  ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID}"
  ARRAY_TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"
  CYCLES_PER_TASK_EFFECTIVE="$CYCLES_PER_TASK"
  echo "ℹ️ Array task detected: array_id=${ARRAY_TASK_ID}, array_count=${ARRAY_TASK_COUNT}"
else
  IS_ARRAY_JOB=0
  ARRAY_TASK_ID=0
  ARRAY_TASK_COUNT=1
  CYCLES_PER_TASK_EFFECTIVE=0
  echo "ℹ️ Direct Slurm submission detected: running all pending cycles in one task."
fi

RUN_DIR="${RUN_DIR:-$LOCAL_BASE/gfs_${SLURM_JOB_ID}_${ARRAY_TASK_ID}}"
mkdir -p "$RUN_DIR"/{logs,input,output}
cd "$RUN_DIR"

cp -f "$TRACKS_CSV" "input/tracks.csv"

# --------------- Build non-overlapping cycle subset for this array task ---------------
mapfile -t CYCLES < <(
  "${ENV_RUNNER[@]}" python3 - "$PROJECT_DIR" "input/tracks.csv" "$LEAD_DAYS" "$CYCLES_PER_TASK_EFFECTIVE" "$ARRAY_TASK_ID" "$ARRAY_TASK_COUNT" "$STORMS_FILTER" "$MERGE_OUTPUT_ROOT" "$PIPELINE_MODE" "$IS_ARRAY_JOB" "$RESUME_COMPLETED_CYCLES" <<'PY'
import os
import sys
import io
import contextlib
from pathlib import Path
from datetime import datetime

project_dir, tracks_csv, lead_days, cycles_per_task, task_id, task_count, storms_filter, merge_root, pipeline_mode, is_array_job, resume_completed_cycles = sys.argv[1:]
lead_days = int(lead_days)
cycles_per_task = int(cycles_per_task)
task_id = int(task_id)
task_count = int(task_count)
is_array_job = int(is_array_job) == 1
resume_completed_cycles = int(resume_completed_cycles) == 1

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

if pipeline_mode == "gee" or not resume_completed_cycles:
    tags = all_tags
else:
    done_stems = set()
    final_dir = Path(merge_root) / "final_single_output"
    if final_dir.exists():
        for jf in final_dir.glob("*_TC_Analysis_*.json"):
            stem = jf.stem
            if "_TC_Analysis_" in stem:
                done_stems.add(stem.split("_TC_Analysis_", 1)[0])

    tags = []
    for tag in all_tags:
        init_tag = datetime.strptime(tag, "%Y%m%dT%H").strftime("%Y-%m-%d%H")
        nc_stem = f"gfs_{init_tag}_f000_f240_6h"
        if nc_stem not in done_stems:
            tags.append(tag)

if not tags:
    sys.exit(0)

if is_array_job:
    if cycles_per_task <= 0:
        raise SystemExit("❌ CYCLES_PER_TASK must be > 0 in array mode")
    # deterministic chunking: contiguous chunks + array index
    chunks = [tags[i:i + cycles_per_task] for i in range(0, len(tags), cycles_per_task)]
    # if user submitted more array tasks than chunks, silently no-op extra tasks
    if task_id >= len(chunks):
        sys.exit(0)
    selected = chunks[task_id]
else:
    selected = tags

for tag in selected:
    print(tag)
PY
)

if [[ ${#CYCLES[@]} -eq 0 ]]; then
  if [[ "$IS_ARRAY_JOB" == "1" ]]; then
    echo "⏹️ No cycles assigned to this array task (array_id=${ARRAY_TASK_ID}). Exit."
  else
    echo "⏹️ No pending cycles to process. Exit."
  fi
  exit 0
fi

echo "=== Job meta ==="
echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
if [[ "$IS_ARRAY_JOB" == "1" ]]; then
  echo "TASK_MODE=array"
  echo "SLURM_ARRAY_TASK_ID=${ARRAY_TASK_ID}"
  echo "SLURM_ARRAY_TASK_COUNT=${ARRAY_TASK_COUNT}"
else
  echo "TASK_MODE=single"
  echo "SLURM_ARRAY_TASK_ID=N/A"
  echo "SLURM_ARRAY_TASK_COUNT=N/A"
fi
echo "RESUME_COMPLETED_CYCLES=${RESUME_COMPLETED_CYCLES}"
echo "RUN_DIR=$RUN_DIR"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "PIPELINE_MODE=$PIPELINE_MODE"
echo "CPUS=$CPUS CYCLE_WORKERS=$CYCLE_WORKERS DOWNLOAD_THREADS=$DOWNLOAD_THREADS PARSE_WORKERS=$PARSE_WORKERS EXTRACT_PROCESSES=$EXTRACT_PROCESSES"
echo "Assigned items: ${#CYCLES[@]} (first=${CYCLES[0]}, last=${CYCLES[-1]})"

STATUS=0
LOG_FILE="logs/run_${SLURM_JOB_ID}_${ARRAY_TASK_ID}.log"

if [[ "$PIPELINE_MODE" == "grib" || "$PIPELINE_MODE" == "cds" ]]; then
  if [[ "$PIPELINE_MODE" == "cds" ]]; then
    echo "ℹ️ PIPELINE_MODE=cds uses GFS/AWS GRIB cycle workflow."
  fi
  CMD=(
    python3 -u "$PROJECT_DIR/src/assemble_and_run_grib.py"
    --tracks "input/tracks.csv"
    --cycles "${CYCLES[@]}"
    --cycle-workers "$CYCLE_WORKERS"
    --download-threads "$DOWNLOAD_THREADS"
    --parse-workers "$PARSE_WORKERS"
    --processes "$EXTRACT_PROCESSES"
    --grib-dir "$RUN_DIR/grib_cache"
  )

  if [[ "$KEEP_NC" == "1" ]]; then
    CMD+=(--keep-nc --keep-grib)
  fi

  set +e
  "${ENV_RUNNER[@]}" "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
  STATUS=${PIPESTATUS[0]}
  set -e
else
  GEE_OUT_DIR="$RUN_DIR/$GEE_OUTPUT_SUBDIR"
  mkdir -p "$GEE_OUT_DIR"

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

    if [[ "$GEE_ANALYSIS_ONLY" == "1" ]]; then
      GEE_CMD+=(--analysis-only)
    fi
    if [[ -n "$GEE_PROJECT" ]]; then
      GEE_CMD+=(--project "$GEE_PROJECT")
    fi
    if [[ -n "$GEE_SERVICE_ACCOUNT_KEY_JSON" ]]; then
      GEE_CMD+=(--service-account-key-json "$GEE_SERVICE_ACCOUNT_KEY_JSON")
    fi
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

echo "Pipeline exit code: $STATUS"

# --------------- Merge outputs to shared storage ---------------
if [[ -n "$MERGE_OUTPUT_ROOT" ]]; then
  if [[ "$PIPELINE_MODE" == "grib" || "$PIPELINE_MODE" == "cds" ]]; then
    mkdir -p "$MERGE_OUTPUT_ROOT"/{final_single_output,track_single,logs}
    if [[ -d "$RUN_DIR/final_single_output" ]]; then
      rsync -a --ignore-existing "$RUN_DIR/final_single_output/" "$MERGE_OUTPUT_ROOT/final_single_output/"
    fi
    if [[ -d "$RUN_DIR/track_single" ]]; then
      rsync -a --ignore-existing "$RUN_DIR/track_single/" "$MERGE_OUTPUT_ROOT/track_single/"
    fi
    rsync -a "$RUN_DIR/logs/" "$MERGE_OUTPUT_ROOT/logs/"
  else
    mkdir -p "$MERGE_OUTPUT_ROOT/$GEE_OUTPUT_SUBDIR" "$MERGE_OUTPUT_ROOT/logs"
    if [[ -d "$RUN_DIR/$GEE_OUTPUT_SUBDIR" ]]; then
      rsync -a --ignore-existing "$RUN_DIR/$GEE_OUTPUT_SUBDIR/" "$MERGE_OUTPUT_ROOT/$GEE_OUTPUT_SUBDIR/"
    fi
    rsync -a "$RUN_DIR/logs/" "$MERGE_OUTPUT_ROOT/logs/"
  fi
fi

exit "$STATUS"
