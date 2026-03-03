#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/TianGong-AI-Cyclone-GFS}"
TRACKS_CSV="${TRACKS_CSV:-$PROJECT_DIR/input/matched_cyclone_tracks.csv}"
# 强制使用指定环境（用户要求）
ENV_PREFIX="/scratch/users/ziqianx/conda/envs/cyclone"
LEAD_DAYS="${LEAD_DAYS:-10}"
CYCLES_PER_TASK="${CYCLES_PER_TASK:-32}"
STORMS_FILTER="${STORMS_FILTER:-}"      # optional comma-separated storm ids
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-6}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$PROJECT_DIR/slurm/run_gfs_aws_retrack_array.slurm}"
MERGE_OUTPUT_ROOT="${MERGE_OUTPUT_ROOT:-$PROJECT_DIR/hpc_runs/retrack_resume}"

if [[ ! -f "$TRACKS_CSV" ]]; then
  echo "❌ TRACKS_CSV not found: $TRACKS_CSV"
  exit 2
fi
if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "❌ SBATCH script not found: $SBATCH_SCRIPT"
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

TOTAL_CYCLES="$(
"${ENV_RUNNER[@]}" python3 - "$PROJECT_DIR" "$TRACKS_CSV" "$LEAD_DAYS" "$STORMS_FILTER" "$MERGE_OUTPUT_ROOT" <<'PY'
import os
import sys
import io
import contextlib
from pathlib import Path
from datetime import datetime

project_dir, tracks_csv, lead_days, storms_filter, merge_root = sys.argv[1:]
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
echo "Submitting: --array=0-${LAST_TASK}%${ARRAY_CONCURRENCY}"

mkdir -p "$PROJECT_DIR/logs"

sbatch \
  --chdir="$PROJECT_DIR" \
  --array="0-${LAST_TASK}%${ARRAY_CONCURRENCY}" \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",TRACKS_CSV="$TRACKS_CSV",LEAD_DAYS="$LEAD_DAYS",CYCLES_PER_TASK="$CYCLES_PER_TASK",STORMS_FILTER="$STORMS_FILTER",MERGE_OUTPUT_ROOT="$MERGE_OUTPUT_ROOT" \
  "$SBATCH_SCRIPT"
