"""Persistent localhost bridge between RoboMME and a separate F2G environment.

Run this file with the Python executable from the ``f2g`` environment.  The
RoboMME process sends compressed NumPy arrays over localhost, so neither
environment imports the other's PyTorch/CUDA packages.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


MAX_REQUEST_BYTES = 128 * 1024 * 1024


def _json_array(value: Any) -> list:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value).tolist()


class F2GServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], estimator: Any, checkpoint: str):
        super().__init__(address, F2GRequestHandler)
        self.estimator = estimator
        self.checkpoint = checkpoint
        self.inference_lock = threading.Lock()


class F2GRequestHandler(BaseHTTPRequestHandler):
    server: F2GServer

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send_json(404, {"error": "not found"})
            return
        self._send_json(
            200,
            {
                "status": "ok",
                "checkpoint": self.server.checkpoint,
                "protocol": "f2g-npz-v1",
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        if not 0 < length <= MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "request body is empty or too large"})
            return
        try:
            with np.load(io.BytesIO(self.rfile.read(length)), allow_pickle=False) as data:
                expected = {"rgb", "depth", "intrinsic", "mask", "depth_scale"}
                if set(data.files) != expected:
                    raise ValueError(
                        f"npz fields must be exactly {sorted(expected)}"
                    )
                rgb = data["rgb"]
                depth = data["depth"]
                intrinsic = data["intrinsic"]
                mask = data["mask"]
                depth_scale = float(data["depth_scale"])
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("rgb must be HxWx3 uint8 in RGB order")
            if depth.shape != rgb.shape[:2] or mask.shape != rgb.shape[:2]:
                raise ValueError("rgb, depth, and mask must be aligned")
            if intrinsic.shape != (3, 3):
                raise ValueError("intrinsic must have shape 3x3")
            if np.count_nonzero(mask) < 50:
                raise ValueError("mask must contain at least 50 foreground pixels")
            with self.server.inference_lock:
                result = self.server.estimator.predict(
                    rgb,
                    depth,
                    intrinsic,
                    mask,
                    depth_scale=depth_scale,
                    return_shape=False,
                )
            response = {
                "T_cam_obj": _json_array(result["T_cam_obj"]),
                "size_m": _json_array(result["size_m"]),
                "coordinate_frame": "camera: x right, y down, z forward",
                "protocol": "f2g-npz-v1",
            }
        except Exception as error:
            self._send_json(
                422,
                {"error": type(error).__name__, "detail": str(error)},
            )
            return
        self._send_json(200, response)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[f2g-server] {self.address_string()} {format % args}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--radio-repository",
        type=Path,
        help="optional RADIO source checkout; exported as F2G_RADIO_REPOSITORY",
    )
    parser.add_argument(
        "--radio-checkpoint",
        type=Path,
        help="optional RADIO checkpoint; exported as F2G_RADIO_CHECKPOINT",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.radio_repository:
        os.environ["F2G_RADIO_REPOSITORY"] = str(
            args.radio_repository.expanduser().resolve()
        )
    if args.radio_checkpoint:
        os.environ["F2G_RADIO_CHECKPOINT"] = str(
            args.radio_checkpoint.expanduser().resolve()
        )
    # Import only after environment variables are set, and only in the f2g env.
    from f2g_pose import PoseEstimator

    estimator = PoseEstimator(str(checkpoint))
    server = F2GServer((args.host, args.port), estimator, str(checkpoint))
    print(f"F2G ready at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
