#!/bin/bash
#SBATCH --job-name=gfs_aws_retrack
#SBATCH --output=logs/gfs_aws_retrack_%A_%a.out
#SBATCH --error=logs/gfs_aws_retrack_%A_%a.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# --------------- Runtime knobs (override via sbatch --export=...) ---------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/TianGong-AI-Cyclone-GFS}"
TRACKS_CSV="${TRACKS_CSV:-$PROJECT_DIR/input/matched_cyclone_tracks.csv}"
# 强制使用指定环境（用户要求）
ENV_PREFIX="/scratch/users/ziqianx/conda/envs/cyclone"
LEAD_DAYS="${LEAD_DAYS:-10}"
CYCLES_PER_TASK="${CYCLES_PER_TASK:-32}"
STORMS_FILTER="${STORMS_FILTER:-}"    # optional comma-separated storm ids
KEEP_NC="${KEEP_NC:-0}"

# Optional output merge target (shared filesystem).
# If empty, outputs stay in per-task RUN_DIR.
MERGE_OUTPUT_ROOT="${MERGE_OUTPUT_ROOT:-$PROJECT_DIR/hpc_runs/${SLURM_JOB_ID}}"

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

if [[ ! -d "$PROJECT_DIR/src" ]]; then
  echo "❌ PROJECT_DIR invalid: $PROJECT_DIR"
  exit 2
fi
if [[ ! -f "$TRACKS_CSV" ]]; then
  echo "❌ TRACKS_CSV not found: $TRACKS_CSV"
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
RUN_DIR="${RUN_DIR:-$LOCAL_BASE/gfs_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$RUN_DIR"/{logs,input,output}
cd "$RUN_DIR"

cp -f "$TRACKS_CSV" "input/tracks.csv"

# --------------- Build non-overlapping cycle subset for this array task ---------------
mapfile -t CYCLES < <(
  "${ENV_RUNNER[@]}" python3 - "$PROJECT_DIR" "input/tracks.csv" "$LEAD_DAYS" "$CYCLES_PER_TASK" "${SLURM_ARRAY_TASK_ID:-0}" "${SLURM_ARRAY_TASK_COUNT:-1}" "$STORMS_FILTER" "$MERGE_OUTPUT_ROOT" <<'PY'
import os
import sys
import io
import contextlib
from pathlib import Path
from datetime import datetime

project_dir, tracks_csv, lead_days, cycles_per_task, task_id, task_count, storms_filter, merge_root = sys.argv[1:]
lead_days = int(lead_days)
cycles_per_task = int(cycles_per_task)
task_id = int(task_id)
task_count = int(task_count)

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

# deterministic chunking: contiguous chunks + array index
chunk_size = cycles_per_task
chunks = [tags[i:i + chunk_size] for i in range(0, len(tags), chunk_size)]

# if user submitted more array tasks than chunks, silently no-op extra tasks
if task_id >= len(chunks):
    sys.exit(0)

for tag in chunks[task_id]:
    print(tag)
PY
)

if [[ ${#CYCLES[@]} -eq 0 ]]; then
  echo "⏹️ No cycles assigned to this task (array_id=${SLURM_ARRAY_TASK_ID:-0}). Exit."
  exit 0
fi

echo "=== Job meta ==="
echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-}"
echo "RUN_DIR=$RUN_DIR"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "CPUS=$CPUS CYCLE_WORKERS=$CYCLE_WORKERS DOWNLOAD_THREADS=$DOWNLOAD_THREADS PARSE_WORKERS=$PARSE_WORKERS EXTRACT_PROCESSES=$EXTRACT_PROCESSES"
echo "Assigned cycles: ${#CYCLES[@]} (first=${CYCLES[0]}, last=${CYCLES[-1]})"

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
"${ENV_RUNNER[@]}" "${CMD[@]}" 2>&1 | tee "logs/run_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
STATUS=${PIPESTATUS[0]}
set -e

echo "Pipeline exit code: $STATUS"

# --------------- Merge outputs to shared storage ---------------
if [[ -n "$MERGE_OUTPUT_ROOT" ]]; then
  mkdir -p "$MERGE_OUTPUT_ROOT"/{final_single_output,track_single,logs}
  if [[ -d "$RUN_DIR/final_single_output" ]]; then
    rsync -a --ignore-existing "$RUN_DIR/final_single_output/" "$MERGE_OUTPUT_ROOT/final_single_output/"
  fi
  if [[ -d "$RUN_DIR/track_single" ]]; then
    rsync -a --ignore-existing "$RUN_DIR/track_single/" "$MERGE_OUTPUT_ROOT/track_single/"
  fi
  rsync -a "$RUN_DIR/logs/" "$MERGE_OUTPUT_ROOT/logs/"
fi

exit "$STATUS"
