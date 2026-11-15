"""Capture GPU occupancy before any process belonging to an eval is started."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oracle_plus.runtime_metrics import GpuMemoryMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monitor = GpuMemoryMonitor(args.cuda_visible_devices)
    try:
        snapshot = monitor.summary()
    finally:
        monitor.close()
    if "error" in snapshot:
        raise RuntimeError(
            f"cannot capture the pre-eval GPU baseline: {snapshot['error']}"
        )
    payload = {
        "measurement": (
            "NVML global GPU memory minus the snapshot helper process; "
            "captured before F2G and Qwen are started"
        ),
        "gpu_idx": snapshot.get("gpu_idx"),
        "total_memory_gb": snapshot.get("total_memory_gb"),
        "global_used_before_eval_gb": snapshot.get(
            "peak_global_used_during_eval_gb"
        ),
        "other_programs_used_before_eval_gb": snapshot.get(
            "other_processes_before_eval_gb"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
