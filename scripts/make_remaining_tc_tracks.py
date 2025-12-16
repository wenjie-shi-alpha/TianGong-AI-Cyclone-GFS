#!/usr/bin/env python3
"""Create a reduced TC-tracks CSV to resume `src/process_by_tc_lifetime.py`.

This script inspects a previous `process_by_tc_lifetime` run log to find:
  1) the last cyclone ID that started processing, and
  2) (optionally) the first *incomplete* cycle for that cyclone.

It then writes a new TC-tracks CSV containing:
  - all storms from that last cyclone onward (preserving original order), and
  - for the last cyclone, only track points at/after the incomplete cycle time
    (so already-finished cycles are not re-selected by lifetime filtering).
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TC_RE = re.compile(r"🌀 处理台风 \[([^\]]+)\]")
CYCLE_RE = re.compile(r"▶️ cycle (\d{4}-\d{2}-\d{2} \d{2}:\d{2})Z")
DONE_RE = re.compile(r"✅ 追踪和环境提取完成")


def _parse_utc_cycle(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


def _find_last_tc_and_resume_cycle(log_path: Path) -> tuple[str, datetime | None]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    last_tc: str | None = None
    last_tc_line_idx: int | None = None
    for idx, line in enumerate(lines):
        match = TC_RE.search(line)
        if match:
            last_tc = match.group(1).strip()
            last_tc_line_idx = idx

    if last_tc is None or last_tc_line_idx is None:
        raise RuntimeError(f"未能从日志中解析出最后处理的台风ID: {log_path}")

    cycle_starts: list[tuple[int, datetime]] = []
    for idx in range(last_tc_line_idx, len(lines)):
        match = CYCLE_RE.search(lines[idx])
        if match:
            cycle_starts.append((idx, _parse_utc_cycle(match.group(1))))

    if not cycle_starts:
        return last_tc, None

    resume_cycle: datetime | None = None
    for i, (start_idx, cycle_dt) in enumerate(cycle_starts):
        end_idx = cycle_starts[i + 1][0] if i + 1 < len(cycle_starts) else len(lines)
        block = lines[start_idx:end_idx]
        completed = any(DONE_RE.search(l) for l in block)
        if not completed:
            resume_cycle = cycle_dt
            break

    return last_tc, resume_cycle


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a remaining TC-tracks CSV for resuming processing.")
    parser.add_argument("--log", type=Path, default=Path("run_process_by_tc_lifetime.log"))
    parser.add_argument("--tc-tracks", type=Path, default=Path("input/matched_cyclone_tracks_2021onwards.csv"))
    parser.add_argument("--out", type=Path, default=Path("input/matched_cyclone_tracks_remaining.csv"))
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="手动指定最后一个台风的恢复起点(UTC), 例如 2021-11-06T12:00:00Z；不指定则从日志自动解析",
    )
    args = parser.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(f"日志不存在: {args.log}")
    if not args.tc_tracks.exists():
        raise FileNotFoundError(f"台风轨迹CSV不存在: {args.tc_tracks}")

    last_tc, resume_cycle = _find_last_tc_and_resume_cycle(args.log)
    if args.resume_from:
        text = args.resume_from.strip().replace("z", "Z")
        if text.endswith("Z"):
            text = text[:-1]
        resume_cycle = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)

    df = pd.read_csv(args.tc_tracks)
    if "storm_id" not in df.columns:
        raise RuntimeError(f"缺少 storm_id 列: {args.tc_tracks}")
    if "datetime" not in df.columns:
        raise RuntimeError(f"缺少 datetime 列: {args.tc_tracks}")

    ordered_ids = df["storm_id"].drop_duplicates().tolist()
    if last_tc not in ordered_ids:
        raise RuntimeError(f"日志中的 storm_id={last_tc} 不在 {args.tc_tracks} 中")
    start_pos = ordered_ids.index(last_tc)
    remaining_ids = ordered_ids[start_pos:]

    remaining = df[df["storm_id"].isin(remaining_ids)].copy()
    if resume_cycle is not None:
        remaining["datetime"] = pd.to_datetime(remaining["datetime"], utc=True, errors="coerce")
        is_last = remaining["storm_id"] == last_tc
        last_block = remaining.loc[is_last].copy()
        other_block = remaining.loc[~is_last].copy()

        after_or_eq = last_block.loc[last_block["datetime"] >= resume_cycle].copy()
        before = last_block.loc[last_block["datetime"] < resume_cycle].copy()
        if not before.empty:
            last_before_row = before.loc[[before["datetime"].idxmax()]]
            last_block = pd.concat([last_before_row, after_or_eq], ignore_index=True)
        else:
            last_block = after_or_eq

        remaining = pd.concat([last_block, other_block], ignore_index=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    remaining.to_csv(args.out, index=False)

    remaining_unique = remaining["storm_id"].drop_duplicates().tolist()
    print(f"✅ 已生成剩余台风轨迹CSV: {args.out}")
    print(f"- 日志最后处理台风: {last_tc}")
    print(f"- 恢复起点(未完成cycle): {resume_cycle.isoformat() if resume_cycle else 'None'}")
    print(f"- 原始台风数: {len(ordered_ids)}")
    print(f"- 剩余台风数: {len(remaining_unique)}")
    print(f"- 输出行数: {len(remaining)}")
    if remaining_unique:
        print(f"- 剩余首个台风: {remaining_unique[0]}")
        print(f"- 剩余最后台风: {remaining_unique[-1]}")


if __name__ == "__main__":
    main()
