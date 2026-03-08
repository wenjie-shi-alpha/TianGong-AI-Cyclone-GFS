# %% [markdown]
# # GFS Pipeline Launcher (16GB RAM / 4 CPU)
#
# This notebook-style script installs dependencies and starts the
# AWS GFS -> NC -> tracking -> environment-extraction pipeline safely
# on a low-resource machine.
#
# Default strategy for stability:
# - `cycle_workers=1` (avoid multi-cycle memory spikes)
# - small `parse_workers` and `processes`
# - memory watchdog during run
#
# Open this file in JupyterLab/VS Code and run cells from top to bottom.

# %%
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path("/root/TianGong-AI-Cyclone-GFS").resolve()
assert PROJECT_DIR.exists(), f"Project dir not found: {PROJECT_DIR}"
os.chdir(PROJECT_DIR)

print(f"PROJECT_DIR={PROJECT_DIR}")
print(f"PYTHON={sys.executable}")


# %%
def run_cmd(cmd: list[str], check: bool = True) -> int:
    print("$", " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(cmd)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


def pip_install(args: list[str]) -> None:
    run_cmd([sys.executable, "-m", "pip", *args], check=True)


# %% [markdown]
# ## 1) Dependency install

# %%
# Upgrade packaging tools first
pip_install(["install", "--upgrade", "pip", "setuptools", "wheel"])

# Core project dependencies
pip_install(["install", "-r", "requirements.txt"])

# Utility dependency for memory check (lightweight)
pip_install(["install", "psutil"])

print("Dependency installation done.")


# %% [markdown]
# ## 2) Runtime config

# %%
@dataclass
class RunConfig:
    tracks_csv: Path
    storms_filter: list[str] | None
    lead_days: int
    cycle_workers: int
    download_threads: int
    parse_workers: int
    extract_processes: int
    batch_size: int
    grib_dir: Path
    max_mem_ratio: float
    dry_run_cycle: str | None


# Input settings (edit if needed)
TRACKS_CSV = PROJECT_DIR / "input" / "matched_cyclone_tracks.csv"
STORMS_FILTER = None
# Example:
# STORMS_FILTER = ["2021093S10103"]

LEAD_DAYS = 10
DRY_RUN_CYCLE = None
# Example smoke test:
# DRY_RUN_CYCLE = "20210403T00"

# Safety controls (for 16GB/4CPU)
MAX_MEM_RATIO = 0.90


# %%
import psutil

cpu_total = os.cpu_count() or 4
cpu_use = min(4, max(1, cpu_total))
mem_total_gb = psutil.virtual_memory().total / (1024**3)

# Conservative defaults for 16G/4C
cycle_workers = 1
# Moderate network parallelism; cheap on memory, useful for speed.
download_threads = min(6, max(3, cpu_use))
# cfgrib parsing is memory-heavy; keep small.
parse_workers = 2 if mem_total_gb >= 14 and cpu_use >= 4 else 1
# Environment extraction can use multiprocessing; keep low.
extract_processes = 2 if mem_total_gb >= 14 and cpu_use >= 4 else 1

# Batch size should not exceed extraction processes under low memory.
batch_size = max(1, min(2, extract_processes))

config = RunConfig(
    tracks_csv=TRACKS_CSV,
    storms_filter=STORMS_FILTER,
    lead_days=LEAD_DAYS,
    cycle_workers=cycle_workers,
    download_threads=download_threads,
    parse_workers=parse_workers,
    extract_processes=extract_processes,
    batch_size=batch_size,
    grib_dir=PROJECT_DIR / "data" / "grib_cache",
    max_mem_ratio=MAX_MEM_RATIO,
    dry_run_cycle=DRY_RUN_CYCLE,
)

print("Detected resources:")
print(f"- CPU total: {cpu_total} -> use <= {cpu_use}")
print(f"- RAM total: {mem_total_gb:.1f} GB")
print("Planned pipeline params:")
print(config)


# %% [markdown]
# ## 3) Build pipeline command

# %%
assert config.tracks_csv.exists(), f"Tracks CSV not found: {config.tracks_csv}"
config.grib_dir.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable,
    "-u",
    "src/assemble_and_run_grib.py",
    "--tracks",
    str(config.tracks_csv),
    "--lead-days",
    str(config.lead_days),
    "--cycle-workers",
    str(config.cycle_workers),
    "--download-threads",
    str(config.download_threads),
    "--parse-workers",
    str(config.parse_workers),
    "--processes",
    str(config.extract_processes),
    "--batch-size",
    str(config.batch_size),
    "--grib-dir",
    str(config.grib_dir),
]

if config.storms_filter:
    cmd.extend(["--storms", *config.storms_filter])

if config.dry_run_cycle:
    cmd.extend(["--cycles", config.dry_run_cycle])

print("Pipeline command:")
print(" ".join(shlex.quote(x) for x in cmd))


# %% [markdown]
# ## 4) Start pipeline with memory watchdog

# %%
# Keep BLAS threads low to avoid CPU oversubscription and memory pressure.
env = os.environ.copy()
env["PYTHONPATH"] = str(PROJECT_DIR / "src")
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
env["OPENBLAS_NUM_THREADS"] = "1"
env["NUMEXPR_NUM_THREADS"] = "1"

log_dir = PROJECT_DIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
run_ts = time.strftime("%Y%m%d_%H%M%S")
log_path = log_dir / f"notebook_pipeline_{run_ts}.log"
print(f"Log file: {log_path}")

stop_flag = {"stop": False}


def watchdog(pid: int, max_ratio: float) -> None:
    proc = psutil.Process(pid)
    while not stop_flag["stop"]:
        try:
            vm = psutil.virtual_memory()
            mem_ratio = vm.percent / 100.0
            rss_gb = proc.memory_info().rss / (1024**3)
            print(
                f"[watchdog] sys_mem={vm.percent:.1f}% proc_rss={rss_gb:.2f}GB "
                f"cpu={proc.cpu_percent(interval=0.5):.1f}%"
            )
            if mem_ratio >= max_ratio:
                print(
                    f"[watchdog] Memory >= {max_ratio*100:.0f}% -> send SIGINT for safe stop"
                )
                proc.send_signal(signal.SIGINT)
                return
            time.sleep(10)
        except psutil.NoSuchProcess:
            return
        except Exception as exc:
            print(f"[watchdog] warning: {exc}")
            time.sleep(10)


with log_path.open("w", encoding="utf-8") as lf:
    lf.write("CMD: " + " ".join(shlex.quote(x) for x in cmd) + "\n")
    lf.flush()

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )

    t = threading.Thread(target=watchdog, args=(p.pid, config.max_mem_ratio), daemon=True)
    t.start()

    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="")
        lf.write(line)
        lf.flush()

    rc = p.wait()
    stop_flag["stop"] = True
    t.join(timeout=1)

print(f"Pipeline exit code: {rc}")
if rc != 0:
    raise RuntimeError(
        "Pipeline failed/stopped. Check log and lower concurrency "
        "(parse_workers/processes)."
    )


# %% [markdown]
# ## 5) Quick result check

# %%
final_dir = PROJECT_DIR / "final_single_output"
track_dir = PROJECT_DIR / "track_single"

n_json = len(list(final_dir.glob("*_TC_Analysis_*.json"))) if final_dir.exists() else 0
n_track = len(list(track_dir.glob("track_*.csv"))) if track_dir.exists() else 0

print(f"final_single_output JSON count: {n_json}")
print(f"track_single CSV count: {n_track}")
print(f"Latest run log: {log_path}")

print("Done.")
