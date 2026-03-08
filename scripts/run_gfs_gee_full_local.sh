#!/usr/bin/env bash
set -euo pipefail

# Local direct GEE runner (no Slurm required).
# Runs cycle-by-cycle and optionally packs all outputs once at the end.
# Usage:
#   bash scripts/run_gfs_gee_full_local.sh [PROJECT_ID] [SERVICE_ACCOUNT_KEY_JSON]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

TRACKS_CSV="${TRACKS_CSV:-$PROJECT_DIR/input/matched_cyclone_tracks.csv}"
LEAD_DAYS="${LEAD_DAYS:-10}"
STORMS_FILTER="${STORMS_FILTER:-}"   # comma-separated storm IDs, optional

GEE_OUTPUT_SUBDIR="${GEE_OUTPUT_SUBDIR:-gee_pipeline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/gee_runs/$(date -u +%Y%m%dT%H%M%SZ)}"
GEE_OUTPUT_DIR="$OUTPUT_ROOT/$GEE_OUTPUT_SUBDIR"
LOG_DIR="$OUTPUT_ROOT/logs"

PROJECT_ID_ARG="${1:-}"
SERVICE_KEY_ARG="${2:-}"

# Align with Cyclone_next default project behavior.
GEE_PROJECT="${GEE_PROJECT:-${PROJECT_ID_ARG:-eminent-glider-467006-r0}}"
GEE_SERVICE_ACCOUNT_KEY_JSON="${GEE_SERVICE_ACCOUNT_KEY_JSON:-$SERVICE_KEY_ARG}"
GEE_ANALYSIS_ONLY="${GEE_ANALYSIS_ONLY:-0}"
GEE_MAX_FORECAST_HOUR="${GEE_MAX_FORECAST_HOUR:-120}"
GEE_TIME_WINDOW_HOURS="${GEE_TIME_WINDOW_HOURS:-6}"
GEE_TEMPORAL_SPAN_HOURS="${GEE_TEMPORAL_SPAN_HOURS:-120}"
GEE_MAX_STEPS="${GEE_MAX_STEPS:-40}"
GEE_SPATIAL_PAD_DEG="${GEE_SPATIAL_PAD_DEG:-12.0}"
GEE_SLEEP_SEC="${GEE_SLEEP_SEC:-0.5}"

SKIP_DONE="${SKIP_DONE:-1}"          # skip cycle when .ok marker exists
ARCHIVE_ON_COMPLETE="${ARCHIVE_ON_COMPLETE:-1}"
ARCHIVE_NAME="${ARCHIVE_NAME:-}"
DRY_RUN="${DRY_RUN:-0}"
PRECHECK_AUTH="${PRECHECK_AUTH:-1}"

if [[ ! -f "$TRACKS_CSV" ]]; then
  echo "❌ TRACKS_CSV not found: $TRACKS_CSV"
  exit 2
fi

if [[ -n "$GEE_SERVICE_ACCOUNT_KEY_JSON" && ! -f "$GEE_SERVICE_ACCOUNT_KEY_JSON" ]]; then
  echo "❌ GEE_SERVICE_ACCOUNT_KEY_JSON not found: $GEE_SERVICE_ACCOUNT_KEY_JSON"
  exit 2
fi

if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

mkdir -p "$GEE_OUTPUT_DIR" "$LOG_DIR"

echo "GEE_PROJECT=$GEE_PROJECT"
echo "⚠️ This script runs simplified GEE tracking/summaries and does NOT perform AWS GRIB->NC full environmental-system extraction."

# Preflight auth check (fail fast) to avoid wasting cycle loops.
if [[ "$DRY_RUN" != "1" && "$PRECHECK_AUTH" == "1" ]]; then
set +e
"$PYTHON_BIN" - "$GEE_PROJECT" "$GEE_SERVICE_ACCOUNT_KEY_JSON" <<'PY'
import json
import sys
from pathlib import Path
import ee

project, key = sys.argv[1], sys.argv[2]
try:
    if key:
        key_path = Path(key)
        info = json.loads(key_path.read_text(encoding="utf-8"))
        client_email = info.get("client_email")
        if not client_email:
            raise RuntimeError(f"service account key missing client_email: {key_path}")
        creds = ee.ServiceAccountCredentials(client_email, str(key_path))
        ee.Initialize(credentials=creds, project=project or None)
    else:
        ee.Initialize(project=project or None)
    print("GEE_AUTH_OK")
except Exception as exc:
    print(f"GEE_AUTH_FAILED: {type(exc).__name__}: {exc}")
    raise
PY
auth_rc=$?
set -e
if [[ "$auth_rc" -ne 0 ]]; then
  echo "❌ GEE preflight auth failed. Fix auth/project/network first."
  exit "$auth_rc"
fi
fi

mapfile -t CYCLES < <(
  "$PYTHON_BIN" - "$TRACKS_CSV" "$LEAD_DAYS" "$STORMS_FILTER" <<'PY'
import sys
from datetime import timedelta
import pandas as pd

tracks_csv, lead_days, storms_filter = sys.argv[1:]
lead_days = int(lead_days)
df = pd.read_csv(tracks_csv)

if "datetime" not in df.columns:
    raise SystemExit("CSV missing required column: datetime")

storm_col = "storm_id" if "storm_id" in df.columns else "SID"
if storm_col not in df.columns:
    raise SystemExit("CSV missing required storm id column: storm_id or SID")

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime"]).copy()
df[storm_col] = df[storm_col].astype(str)

if storms_filter.strip():
    allow = {s.strip() for s in storms_filter.split(",") if s.strip()}
    df = df[df[storm_col].isin(allow)]

if df.empty:
    raise SystemExit(0)

s3_data_start = pd.Timestamp("2021-03-23")
cycle_hours = (0, 12)
storm_ranges = df.groupby(storm_col)["datetime"].agg(["min", "max"])

tags = set()
for _, row in storm_ranges.iterrows():
    s_start = pd.Timestamp(row["min"])
    s_end = pd.Timestamp(row["max"])
    earliest = max(s_start - timedelta(days=lead_days), s3_data_start)
    latest = s_end
    if latest < s3_data_start:
        continue
    day = earliest.normalize()
    while day <= latest:
        for hour in cycle_hours:
            cyc = day + timedelta(hours=hour)
            if earliest <= cyc <= latest:
                tags.add(cyc.strftime("%Y%m%dT%H"))
        day += timedelta(days=1)

for tag in sorted(tags):
    print(tag)
PY
)

if [[ ${#CYCLES[@]} -eq 0 ]]; then
  echo "⏹️ No cycles to process."
  exit 0
fi

echo "PROJECT_DIR=$PROJECT_DIR"
echo "TRACKS_CSV=$TRACKS_CSV"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "GEE_OUTPUT_DIR=$GEE_OUTPUT_DIR"
echo "CYCLES=${#CYCLES[@]} (first=${CYCLES[0]}, last=${CYCLES[-1]})"

cat > "$OUTPUT_ROOT/run_config.env" <<EOF
PROJECT_DIR=$PROJECT_DIR
TRACKS_CSV=$TRACKS_CSV
LEAD_DAYS=$LEAD_DAYS
STORMS_FILTER=$STORMS_FILTER
GEE_PROJECT=$GEE_PROJECT
GEE_SERVICE_ACCOUNT_KEY_JSON=$GEE_SERVICE_ACCOUNT_KEY_JSON
GEE_ANALYSIS_ONLY=$GEE_ANALYSIS_ONLY
GEE_MAX_FORECAST_HOUR=$GEE_MAX_FORECAST_HOUR
GEE_TIME_WINDOW_HOURS=$GEE_TIME_WINDOW_HOURS
GEE_TEMPORAL_SPAN_HOURS=$GEE_TEMPORAL_SPAN_HOURS
GEE_MAX_STEPS=$GEE_MAX_STEPS
GEE_SPATIAL_PAD_DEG=$GEE_SPATIAL_PAD_DEG
GEE_SLEEP_SEC=$GEE_SLEEP_SEC
SKIP_DONE=$SKIP_DONE
ARCHIVE_ON_COMPLETE=$ARCHIVE_ON_COMPLETE
DRY_RUN=$DRY_RUN
PRECHECK_AUTH=$PRECHECK_AUTH
EOF

if [[ "$DRY_RUN" == "1" ]]; then
  echo "ℹ️ DRY_RUN=1, commands will be printed only."
fi

STATUS=0
DONE=0
SKIPPED=0
FAILED=0

STORM_ARGS=()
if [[ -n "$STORMS_FILTER" ]]; then
  IFS=',' read -r -a _SID_ARR <<< "$STORMS_FILTER"
  for sid in "${_SID_ARR[@]}"; do
    sid="${sid//[[:space:]]/}"
    [[ -n "$sid" ]] && STORM_ARGS+=("$sid")
  done
fi

for cycle_tag in "${CYCLES[@]}"; do
  cycle_iso="${cycle_tag:0:4}-${cycle_tag:4:2}-${cycle_tag:6:2}T${cycle_tag:9:2}:00:00Z"
  ok_mark="$LOG_DIR/cycle_${cycle_tag}.ok"
  run_log="$LOG_DIR/cycle_${cycle_tag}.log"

  if [[ "$SKIP_DONE" == "1" && -f "$ok_mark" ]]; then
    echo "⏭️ Skip done cycle: $cycle_tag"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  CMD=(
    "$PYTHON_BIN" -u "$PROJECT_DIR/src/gee_gfs_pipeline.py"
    --initials-csv "$TRACKS_CSV"
    --start-time "$cycle_iso"
    --time-window-hours "$GEE_TIME_WINDOW_HOURS"
    --temporal-span-hours "$GEE_TEMPORAL_SPAN_HOURS"
    --max-steps "$GEE_MAX_STEPS"
    --max-forecast-hour "$GEE_MAX_FORECAST_HOUR"
    --spatial-pad-deg "$GEE_SPATIAL_PAD_DEG"
    --sleep-sec "$GEE_SLEEP_SEC"
    --output-dir "$GEE_OUTPUT_DIR"
  )

  if [[ "$GEE_ANALYSIS_ONLY" == "1" ]]; then
    CMD+=(--analysis-only)
  fi
  if [[ -n "$GEE_PROJECT" ]]; then
    CMD+=(--project "$GEE_PROJECT")
  fi
  if [[ -n "$GEE_SERVICE_ACCOUNT_KEY_JSON" ]]; then
    CMD+=(--service-account-key-json "$GEE_SERVICE_ACCOUNT_KEY_JSON")
  fi
  if [[ ${#STORM_ARGS[@]} -gt 0 ]]; then
    CMD+=(--storm-id "${STORM_ARGS[@]}")
  fi

  echo "▶️ [$((DONE + SKIPPED + FAILED + 1))/${#CYCLES[@]}] $cycle_tag ($cycle_iso)"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '   '; printf '%q ' "${CMD[@]}"; printf '\n'
    DONE=$((DONE + 1))
    continue
  fi

  set +e
  "${CMD[@]}" >"$run_log" 2>&1
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    : > "$ok_mark"
    DONE=$((DONE + 1))
    echo "✅ cycle done: $cycle_tag"
  else
    FAILED=$((FAILED + 1))
    STATUS="$rc"
    echo "❌ cycle failed: $cycle_tag (exit=$rc, log=$run_log)"
  fi
done

echo "=== Summary ==="
echo "done=$DONE skipped=$SKIPPED failed=$FAILED total=${#CYCLES[@]}"

if [[ "$DRY_RUN" != "1" && "$ARCHIVE_ON_COMPLETE" == "1" ]]; then
  if [[ -z "$ARCHIVE_NAME" ]]; then
    ARCHIVE_NAME="gee_full_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
  fi
  ARCHIVE_NAME="$(basename -- "$ARCHIVE_NAME")"
  TMP_ARCHIVE="$OUTPUT_ROOT/.${ARCHIVE_NAME}.tmp"
  FINAL_ARCHIVE="$OUTPUT_ROOT/${ARCHIVE_NAME}"

  PAYLOAD=()
  [[ -d "$GEE_OUTPUT_DIR" ]] && PAYLOAD+=("$(realpath --relative-to="$OUTPUT_ROOT" "$GEE_OUTPUT_DIR")")
  [[ -d "$LOG_DIR" ]] && PAYLOAD+=("$(realpath --relative-to="$OUTPUT_ROOT" "$LOG_DIR")")
  if [[ ${#PAYLOAD[@]} -gt 0 ]]; then
    tar -C "$OUTPUT_ROOT" -czf "$TMP_ARCHIVE" "${PAYLOAD[@]}"
    mv -f "$TMP_ARCHIVE" "$FINAL_ARCHIVE"
    sha256sum "$FINAL_ARCHIVE" > "${FINAL_ARCHIVE}.sha256"
    echo "📦 archive: $FINAL_ARCHIVE"
    echo "🔐 checksum: ${FINAL_ARCHIVE}.sha256"
  else
    echo "⚠️ no payload found for archive under $OUTPUT_ROOT"
  fi
fi

exit "$STATUS"
