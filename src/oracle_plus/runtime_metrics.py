"""Best-effort GPU-memory and per-decision latency summaries."""

from __future__ import annotations

import os
from typing import Any


GIB = 1024**3


def _gib(value: int | None) -> float | None:
    return round(value / GIB, 3) if value is not None else None


def latency_summary(values_ms: list[float], *, scope: str) -> dict[str, Any]:
    """Summarize one latency sample per VLM or mapped-action request."""
    return {
        "steps": len(values_ms),
        "total_seconds": round(sum(values_ms) / 1000, 3),
        "average_ms": (
            round(sum(values_ms) / len(values_ms), 3) if values_ms else None
        ),
        "measurement_scope": scope,
    }


class GpuMemoryMonitor:
    """Measure one visible physical NVIDIA GPU without requiring NVML."""

    def __init__(self, cuda_visible_devices: str) -> None:
        self.cuda_visible_devices = cuda_visible_devices
        self._nvml_ready = False
        self.error: str | None = None
        self._nvml: Any = None
        self._handle: Any = None
        self._baseline: dict[str, int] | None = None
        self._peak: dict[str, int] | None = None
        self.physical_gpu_index: int | None = None
        self.gpu_identifier: str | None = None
        self.gpu_name: str | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            identifier = cuda_visible_devices.split(",", maxsplit=1)[0].strip() or "0"
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
            self._nvml_ready = True
            self._baseline = self._snapshot()
            self._peak = dict(self._baseline)
        except Exception as error:
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
        return {
            "total_bytes": int(memory.total),
            "global_used_bytes": int(memory.used),
            "current_process_used_bytes": self._process_used_bytes(),
        }

    def sample(self) -> None:
        if not self._nvml_ready:
            return
        try:
            current = self._snapshot()
            assert self._peak is not None
            for key in ("global_used_bytes", "current_process_used_bytes"):
                self._peak[key] = max(self._peak[key], current[key])
        except Exception as error:
            self.error = f"Sampling failed: {type(error).__name__}: {error}"

    def mark_model_loaded(self) -> None:
        self.sample()

    def summary(self) -> dict[str, Any]:
        if not self._nvml_ready or self._baseline is None or self._peak is None:
            return {
                "cuda_visible_devices": self.cuda_visible_devices,
                "error": self.error or "NVML is unavailable",
            }
        baseline = self._baseline
        other_processes_before_eval = max(
            baseline["global_used_bytes"]
            - baseline["current_process_used_bytes"],
            0,
        )
        result: dict[str, Any] = {
            "cuda_visible_devices": self.cuda_visible_devices,
            "physical_gpu_index": self.physical_gpu_index,
            "gpu_identifier": self.gpu_identifier,
            "gpu_name": self.gpu_name,
            "total_memory_gb": _gib(baseline["total_bytes"]),
            "other_processes_before_eval_gb": _gib(
                other_processes_before_eval
            ),
            "peak_global_used_during_eval_gb": _gib(
                self._peak["global_used_bytes"]
            ),
            "current_process_peak_during_entire_eval_gb": _gib(
                self._peak["current_process_used_bytes"]
            ),
            "baseline_definition": (
                "other_processes_before_eval_gb is global GPU use minus this "
                "process at monitor startup, before loading Qwen3.8-27B."
            ),
            "error": self.error,
        }
        return result

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
