"""Serve Qwen3.8-27B inference for Oracle Plus over the VLA WebSocket protocol."""

from __future__ import annotations

import argparse
import asyncio
import http
import logging
import os
import time
import traceback
from dataclasses import asdict
from typing import Any

import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames

from openpi_client import msgpack_numpy
from oracle_plus.runtime_metrics import GpuMemoryMonitor
from oracle_plus.runtime_metrics import latency_summary
from oracle_plus.vlm import VLMSubgoalPlanner27B


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/data/public/qwen/Qwen3.8-27B-FP8",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--attn-impl",
        choices=("flash_attention_2", "sdpa"),
        default="flash_attention_2",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


class OraclePlusVLMService:
    """Own the model and translate msgpack requests into planner calls."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.gpu_monitor = GpuMemoryMonitor(
            os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        )
        model_load_started = time.perf_counter()
        self.planner = VLMSubgoalPlanner27B(
            model_path=args.model_path,
            attn_impl=args.attn_impl,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        self.model_load_seconds = time.perf_counter() - model_load_started
        self.gpu_monitor.mark_model_loaded()
        self.vlm_times_ms: list[float] = []
        self.request_to_decision_times_ms: list[float] = []
        self.inference_config = {
            "attn_impl": args.attn_impl,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "request_seed": self.planner.inference_seed,
            "enable_thinking": False,
            "image_max_token_num": os.environ.get("IMAGE_MAX_TOKEN_NUM"),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol": "oracle_plus_vlm_v1",
            "model_family": "Qwen3.8-27B",
            "model_path": self.planner.model_path,
            "inference_config": self.inference_config,
            "model_load_seconds": round(self.model_load_seconds, 3),
        }

    def metrics(self) -> dict[str, Any]:
        self.gpu_monitor.sample()
        return {
            **self.metadata(),
            "gpu": self.gpu_monitor.summary(),
            "vlm_single_step": latency_summary(
                self.vlm_times_ms,
                scope=(
                    "One Qwen3.8-27B engine.infer call after InferRequest "
                    "construction until raw model output returns. WebSocket "
                    "transfer and request unpacking are excluded."
                ),
            ),
            "request_to_decision_single_step": latency_summary(
                self.request_to_decision_times_ms,
                scope=(
                    "One server-side request after InferRequest construction "
                    "through JSON parsing and decision validation. WebSocket "
                    "transfer is excluded."
                ),
            ),
        }

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("operation", "infer")
        if operation == "metrics":
            return self.metrics()
        if operation != "infer":
            raise ValueError(f"Unsupported VLM service operation: {operation!r}")
        messages = request.get("messages")
        images = request.get("images")
        include_description = request.get("include_description")
        if not isinstance(messages, list) or not all(
            isinstance(message, dict) for message in messages
        ):
            raise ValueError("VLM request messages must be a list of objects")
        if not isinstance(images, list) or not all(
            isinstance(image, str) for image in images
        ):
            raise ValueError("VLM request images must be a list of paths")
        if not isinstance(include_description, bool):
            raise ValueError("VLM request include_description must be boolean")
        decision, raw_response, latency_ms, request_to_decision_ms = (
            self.planner.infer(
                messages=messages,
                images=images,
                include_description=include_description,
            )
        )
        self.vlm_times_ms.append(latency_ms)
        self.request_to_decision_times_ms.append(request_to_decision_ms)
        self.gpu_monitor.sample()
        decision_payload = asdict(decision)
        if not include_description:
            decision_payload.pop("description", None)
        return {
            "decision": decision_payload,
            "raw_response": raw_response,
            "vlm_latency_ms": latency_ms,
            "request_to_decision_ms": request_to_decision_ms,
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
    service: OraclePlusVLMService,
) -> None:
    LOGGER.info("Connection from %s opened", connection.remote_address)
    packer = msgpack_numpy.Packer()
    await connection.send(packer.pack(service.metadata()))
    while True:
        try:
            request = msgpack_numpy.unpackb(await connection.recv())
            if not isinstance(request, dict):
                raise ValueError("VLM WebSocket request must be an object")
            await connection.send(packer.pack(service.handle(request)))
        except websockets.ConnectionClosed:
            LOGGER.info("Connection from %s closed", connection.remote_address)
            return
        except Exception:
            await connection.send(traceback.format_exc())
            await connection.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Internal VLM server error; traceback sent to client.",
            )
            raise


async def run_server(
    args: argparse.Namespace,
    service: OraclePlusVLMService,
) -> None:
    async with websocket_server.serve(
        lambda connection: handle_connection(connection, service),
        args.host,
        args.port,
        compression=None,
        max_size=None,
        process_request=health_check,
    ) as server:
        LOGGER.info("Oracle Plus VLM server listening on %s:%s", args.host, args.port)
        await server.serve_forever()


def main() -> None:
    args = parse_args()
    service = OraclePlusVLMService(args)
    try:
        asyncio.run(run_server(args, service))
    finally:
        service.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
