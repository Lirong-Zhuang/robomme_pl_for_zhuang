"""Runtime timing and best-effort NVIDIA GPU-memory measurements."""

from __future__ import annotations

import os
import statistics
from typing import Any


GIB = 1024 * 1024 * 1024


def _gb(value: int | None) -> float | None:
    """Return GPU memory in binary gigabytes (1024**3 bytes)."""
    return round(value / GIB, 2) if value is not None else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class GpuMemoryMonitor:
    """Capture one physical GPU without making NVML a hard dependency."""

    def __init__(self, cuda_visible_devices: str) -> None:
        self.cuda_visible_devices = cuda_visible_devices
        self.available = False
        self.error: str | None = None
        self._nvml: Any = None
        self._handle: Any = None
        self._baseline: dict[str, int] | None = None
        self._after_model: dict[str, int] | None = None
        self._peak: dict[str, int] | None = None
        self.physical_gpu_index: int | None = None
        self.gpu_identifier: str | None = None
        self.gpu_name: str | None = None
        self.total_bytes: int | None = None

        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            identifier = cuda_visible_devices.split(",", maxsplit=1)[0].strip()
            if not identifier:
                identifier = "0"
            self.gpu_identifier = identifier
            if identifier.isdigit():
                self.physical_gpu_index = int(identifier)
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(
                    self.physical_gpu_index
                )
            else:
                self._handle = pynvml.nvmlDeviceGetHandleByUUID(identifier.encode())
                self.physical_gpu_index = int(
                    pynvml.nvmlDeviceGetIndex(self._handle)
                )
            raw_name = pynvml.nvmlDeviceGetName(self._handle)
            self.gpu_name = (
                raw_name.decode("utf-8", errors="replace")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            self.available = True
            self._baseline = self._snapshot()
            self._peak = dict(self._baseline)
            self.total_bytes = self._baseline["total_bytes"]
        except Exception as error:  # NVML varies across driver/pynvml versions.
            self.error = f"{type(error).__name__}: {error}"

    def _process_used_bytes(self) -> int:
        process_bytes: dict[int, int] = {}
        for function_name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ):
            function = getattr(self._nvml, function_name, None)
            if function is None:
                continue
            try:
                processes = function(self._handle)
            except Exception:
                continue
            for process in processes:
                if int(process.pid) != os.getpid():
                    continue
                used = getattr(process, "usedGpuMemory", None)
                if isinstance(used, int) and 0 <= used < 2**63:
                    process_bytes[int(process.pid)] = max(
                        process_bytes.get(int(process.pid), 0), used
                    )
        return sum(process_bytes.values())

    def _snapshot(self) -> dict[str, int]:
        memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        process_used = self._process_used_bytes()
        return {
            "total_bytes": int(memory.total),
            "global_used_bytes": int(memory.used),
            "current_process_used_bytes": process_used,
        }

    def sample(self) -> None:
        if not self.available:
            return
        try:
            current = self._snapshot()
            assert self._peak is not None
            for key in ("global_used_bytes", "current_process_used_bytes"):
                self._peak[key] = max(self._peak[key], current[key])
        except Exception as error:
            self.error = f"Sampling failed: {type(error).__name__}: {error}"

    def mark_model_loaded(self) -> None:
        if not self.available:
            return
        try:
            self._after_model = self._snapshot()
            self.sample()
        except Exception as error:
            self.error = f"Model-load sampling failed: {type(error).__name__}: {error}"

    def summary(self) -> dict[str, Any]:
        if not self.available or self._baseline is None or self._peak is None:
            return {
                "available": False,
                "cuda_visible_devices": self.cuda_visible_devices,
                "error": self.error or "NVML is unavailable",
            }
        baseline = self._baseline
        after_model = self._after_model
        other_before = max(
            baseline["global_used_bytes"]
            - baseline["current_process_used_bytes"],
            0,
        )
        result = {
            "available": True,
            "cuda_visible_devices": self.cuda_visible_devices,
            "physical_gpu_index": self.physical_gpu_index,
            "gpu_identifier": self.gpu_identifier,
            "gpu_name": self.gpu_name,
            "total_memory_gb": _gb(baseline["total_bytes"]),
            "global_used_before_model_load_gb": _gb(
                baseline["global_used_bytes"]
            ),
            "current_process_before_model_load_gb": _gb(
                baseline["current_process_used_bytes"]
            ),
            "other_processes_before_model_load_gb": _gb(other_before),
            "global_used_after_model_load_gb": _gb(
                after_model["global_used_bytes"] if after_model else None
            ),
            "current_process_after_model_load_gb": _gb(
                after_model["current_process_used_bytes"] if after_model else None
            ),
            "test_increment_after_model_load_gb": _gb(
                max(
                    after_model["global_used_bytes"]
                    - baseline["global_used_bytes"],
                    0,
                )
                if after_model
                else None
            ),
            "peak_global_used_during_test_gb": _gb(
                self._peak["global_used_bytes"]
            ),
            "peak_current_process_during_test_gb": _gb(
                self._peak["current_process_used_bytes"]
            ),
            "test_peak_increment_over_baseline_gb": _gb(
                max(
                    self._peak["global_used_bytes"]
                    - baseline["global_used_bytes"],
                    0,
                )
            ),
            "baseline_definition": (
                "Captured inside run_test.py after CUDA_VISIBLE_DEVICES is set "
                "and before the model engine is imported or loaded."
            ),
            "error": self.error,
        }
        try:
            import torch

            if torch.cuda.is_available():
                result.update(
                    {
                        "torch_current_allocated_gb": _gb(
                            int(torch.cuda.memory_allocated(0))
                        ),
                        "torch_current_reserved_gb": _gb(
                            int(torch.cuda.memory_reserved(0))
                        ),
                        "torch_peak_allocated_gb": _gb(
                            int(torch.cuda.max_memory_allocated(0))
                        ),
                        "torch_peak_reserved_gb": _gb(
                            int(torch.cuda.max_memory_reserved(0))
                        ),
                    }
                )
        except Exception as error:
            result["torch_memory_error"] = f"{type(error).__name__}: {error}"
        return result

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


def timing_summary(
    step_times_ms: list[float],
    *,
    model_load_seconds: float,
) -> dict[str, Any]:
    """Summarize single-step wall-clock VLM latency."""
    return {
        "steps": len(step_times_ms),
        "model_load_seconds": round(model_load_seconds, 3),
        "total_step_vlm_response_seconds": round(sum(step_times_ms) / 1000, 3),
        "average_step_vlm_response_ms": (
            round(statistics.fmean(step_times_ms), 3) if step_times_ms else None
        ),
        "median_step_vlm_response_ms": (
            round(statistics.median(step_times_ms), 3) if step_times_ms else None
        ),
        "p95_step_vlm_response_ms": (
            round(_percentile(step_times_ms, 0.95), 3) if step_times_ms else None
        ),
        "min_step_vlm_response_ms": (
            round(min(step_times_ms), 3) if step_times_ms else None
        ),
        "max_step_vlm_response_ms": (
            round(max(step_times_ms), 3) if step_times_ms else None
        ),
        "measurement_scope": (
            "Wall-clock time from constructing the image-bearing InferRequest "
            "through engine.infer returning the model output; batch_size is 1."
        ),
    }
