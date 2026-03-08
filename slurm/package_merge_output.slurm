#!/bin/bash
#SBATCH --job-name=gfs_result_pack
#SBATCH --output=logs/gfs_result_pack_%j.out
#SBATCH --error=logs/gfs_result_pack_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

MERGE_OUTPUT_ROOT="${MERGE_OUTPUT_ROOT:-}"
PIPELINE_MODE="${PIPELINE_MODE:-grib}"  # grib | gee | cds
GEE_OUTPUT_SUBDIR="${GEE_OUTPUT_SUBDIR:-gee_pipeline}"
CDS_OUTPUT_SUBDIR="${CDS_OUTPUT_SUBDIR:-cds_output}"
ARCHIVE_NAME="${ARCHIVE_NAME:-}"

if [[ -z "$MERGE_OUTPUT_ROOT" ]]; then
  echo "❌ MERGE_OUTPUT_ROOT is empty"
  exit 2
fi

if [[ "$PIPELINE_MODE" != "grib" && "$PIPELINE_MODE" != "gee" && "$PIPELINE_MODE" != "cds" ]]; then
  echo "❌ Unsupported PIPELINE_MODE: $PIPELINE_MODE"
  exit 2
fi

mkdir -p "$MERGE_OUTPUT_ROOT"
mkdir -p logs

if [[ -z "$ARCHIVE_NAME" ]]; then
  RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  ARCHIVE_NAME="retrack_${PIPELINE_MODE}_${SLURM_JOB_ID}_${RUN_STAMP}.tar.gz"
fi
ARCHIVE_NAME="$(basename -- "$ARCHIVE_NAME")"

declare -a CANDIDATES
if [[ "$PIPELINE_MODE" == "grib" || "$PIPELINE_MODE" == "cds" ]]; then
  CANDIDATES=("final_single_output" "track_single" "logs")
else
  CANDIDATES=("$GEE_OUTPUT_SUBDIR" "logs")
fi

declare -a PAYLOAD
for rel in "${CANDIDATES[@]}"; do
  if [[ -e "$MERGE_OUTPUT_ROOT/$rel" ]]; then
    PAYLOAD+=("$rel")
  fi
done

if [[ ${#PAYLOAD[@]} -eq 0 ]]; then
  echo "❌ No output directories found under $MERGE_OUTPUT_ROOT"
  exit 3
fi

TMP_ARCHIVE="$MERGE_OUTPUT_ROOT/.${ARCHIVE_NAME}.tmp"
FINAL_ARCHIVE="$MERGE_OUTPUT_ROOT/${ARCHIVE_NAME}"

echo "📦 Packing payload: ${PAYLOAD[*]}"
echo "📁 Source: $MERGE_OUTPUT_ROOT"

tar -C "$MERGE_OUTPUT_ROOT" -czf "$TMP_ARCHIVE" "${PAYLOAD[@]}"
mv -f "$TMP_ARCHIVE" "$FINAL_ARCHIVE"
sha256sum "$FINAL_ARCHIVE" > "${FINAL_ARCHIVE}.sha256"

echo "✅ Archive ready: $FINAL_ARCHIVE"
echo "✅ Checksum: ${FINAL_ARCHIVE}.sha256"
