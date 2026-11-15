"""Serve generic Qwen3.8 JSON inference for init and in-task planning."""

from __future__ import annotations

import argparse
import asyncio
import http
import logging
import os
import time
import traceback
from typing import Any

import websockets
import websockets.asyncio.server as websocket_server

from openpi_client import msgpack_numpy
from oracle_plus.runtime_metrics import GpuMemoryMonitor
from oracle_plus.runtime_metrics import latency_summary
from oracle_plus.vlm import VLMSubgoalPlanner27B


LOGGER = logging.getLogger(__name__)
PLANNING_PHASES = ("init_planning", "in_task_planning", "auxiliary")


def memory_summary(values_gb: list[float]) -> dict[str, Any]:
    return {
        "average_current_process_gb": (
            round(sum(values_gb) / len(values_gb), 3) if values_gb else None
        ),
        "peak_current_process_gb": (
            round(max(values_gb), 3) if values_gb else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--attn-impl",
        choices=("flash_attention_2", "sdpa"),
        default="flash_attention_2",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


class AgenticVLMService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.gpu_monitor = GpuMemoryMonitor(
            os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        )
        gpu_status = self.gpu_monitor.summary()
        if "error" in gpu_status:
            raise RuntimeError(
                f"NVML metrics are required for this eval: {gpu_status['error']}"
            )
        started = time.perf_counter()
        self.planner = VLMSubgoalPlanner27B(
            model_path=args.model_path,
            attn_impl=args.attn_impl,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        self.model_load_seconds = time.perf_counter() - started
        self.gpu_monitor.mark_model_loaded()
        model_gpu = self.gpu_monitor.summary().get(
            "current_process_peak_during_entire_eval_gb"
        )
        if not isinstance(model_gpu, (int, float)) or model_gpu <= 0:
            raise RuntimeError(
                "NVML did not report GPU memory for the Qwen service process"
            )
        self.vlm_times_ms: list[float] = []
        self.request_times_ms: list[float] = []
        self.phase_vlm_times_ms = {phase: [] for phase in PLANNING_PHASES}
        self.phase_request_times_ms = {phase: [] for phase in PLANNING_PHASES}
        self.phase_process_gpu_gb = {phase: [] for phase in PLANNING_PHASES}
        self.inference_config = {
            "attn_impl": args.attn_impl,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "request_seed": self.planner.inference_seed,
            "enable_thinking": False,
            "response_prefix": self.planner.response_prefix,
            "image_max_token_num": os.environ.get("IMAGE_MAX_TOKEN_NUM"),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol": "agentic_vlm_json_v1",
            "model_family": "Qwen3.8-27B",
            "model_path": self.planner.model_path,
            "inference_config": self.inference_config,
            "model_load_seconds": round(self.model_load_seconds, 3),
        }

    def metrics(self) -> dict[str, Any]:
        self.gpu_monitor.sample()
        gpu = self.gpu_monitor.summary()
        gpu["baseline_scope"] = (
            "VLM service startup; use summary.gpu_before_eval for the GPU "
            "occupancy captured before both F2G and Qwen"
        )
        return {
            **self.metadata(),
            "gpu": gpu,
            "vlm_single_call": latency_summary(self.vlm_times_ms),
            "request_to_json": latency_summary(self.request_times_ms),
            "planning_phases": {
                phase: {
                    "vlm_inference": latency_summary(
                        self.phase_vlm_times_ms[phase]
                    ),
                    "request_to_json": latency_summary(
                        self.phase_request_times_ms[phase]
                    ),
                    "current_process_gpu_memory": memory_summary(
                        self.phase_process_gpu_gb[phase]
                    ),
                }
                for phase in PLANNING_PHASES
            },
        }

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        received = time.perf_counter()
        operation = request.get("operation")
        if operation == "metrics":
            return self.metrics()
        if operation != "infer_json":
            raise ValueError(f"unsupported operation: {operation!r}")
        messages = request.get("messages")
        images = request.get("images", [])
        planning_phase = request.get("planning_phase")
        if not isinstance(messages, list) or not all(
            isinstance(message, dict) for message in messages
        ):
            raise ValueError("messages must be a list of objects")
        if not isinstance(images, list) or not all(
            isinstance(image, str) for image in images
        ):
            raise ValueError("images must be a list of paths")
        if planning_phase not in PLANNING_PHASES:
            raise ValueError(
                f"planning_phase must be one of {PLANNING_PHASES}, got "
                f"{planning_phase!r}"
            )
        sample_index, _ = self.gpu_monitor.begin_interval()
        try:
            value, raw_response, latency_ms, _ = self.planner.infer_json_object(
                messages=messages,
                images=images,
            )
        finally:
            gpu_interval = self.gpu_monitor.end_interval(sample_index)
        request_ms = (time.perf_counter() - received) * 1000
        self.vlm_times_ms.append(latency_ms)
        self.request_times_ms.append(request_ms)
        self.phase_vlm_times_ms[planning_phase].append(latency_ms)
        self.phase_request_times_ms[planning_phase].append(request_ms)
        process_gpu_gb = gpu_interval.get("average_current_process_gb")
        if isinstance(process_gpu_gb, (int, float)):
            self.phase_process_gpu_gb[planning_phase].append(
                float(process_gpu_gb)
            )
        return {
            "json": value,
            "raw_response": raw_response,
            "vlm_latency_ms": latency_ms,
            "request_to_json_ms": request_ms,
        }

    def close(self) -> None:
        self.gpu_monitor.close()


def health_check(
    connection: websocket_server.ServerConnection,
    request: websocket_server.Request,
) -> websocket_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def handle_connection(
    connection: websocket_server.ServerConnection,
    service: AgenticVLMService,
) -> None:
    LOGGER.info("Connection from %s opened", connection.remote_address)
    packer = msgpack_numpy.Packer()
    await connection.send(packer.pack(service.metadata()))
    while True:
        try:
            request = msgpack_numpy.unpackb(await connection.recv())
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            await connection.send(packer.pack(service.handle(request)))
        except websockets.ConnectionClosed:
            return
        except Exception:
            error_traceback = traceback.format_exc()
            LOGGER.error("Agentic VLM request failed:\n%s", error_traceback)
            await connection.send(error_traceback)


async def run_server(
    args: argparse.Namespace,
    service: AgenticVLMService,
) -> None:
    async with websocket_server.serve(
        lambda connection: handle_connection(connection, service),
        args.host,
        args.port,
        compression=None,
        max_size=None,
        process_request=health_check,
    ) as server:
        LOGGER.info("Agentic VLM server listening on %s:%s", args.host, args.port)
        await server.serve_forever()


def main() -> None:
    args = parse_args()
    service = AgenticVLMService(args)
    try:
        asyncio.run(run_server(args, service))
    finally:
        service.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
