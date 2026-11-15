"""Small dependency-free client for the persistent F2G inference process."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

import numpy as np

from .types import CameraFrame
from .types import F2GPrediction


class F2GServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class F2GClient:
    """Call a localhost F2G service without importing its PyTorch environment."""

    base_url: str = "http://127.0.0.1:8765"
    timeout_s: float = 120.0

    def _request(self, request: Request) -> bytes:
        try:
            with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
                return response.read()
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise F2GServiceError(
                f"F2G service returned HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError) as error:
            raise F2GServiceError(
                f"cannot reach F2G service at {self.base_url}: {error}"
            ) from error

    def health(self) -> dict:
        payload = self._request(Request(f"{self.base_url.rstrip('/')}/health"))
        result = json.loads(payload.decode("utf-8"))
        if result.get("status") != "ok":
            raise F2GServiceError(f"F2G service is unhealthy: {result}")
        return result

    def predict(self, frame: CameraFrame, mask: np.ndarray) -> F2GPrediction:
        mask = np.asarray(mask)
        if mask.shape != frame.depth.shape:
            raise ValueError("F2G mask must be aligned with RGB and depth")
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            rgb=frame.rgb,
            depth=frame.depth,
            intrinsic=frame.intrinsic,
            mask=mask.astype(np.uint8) * 255,
            depth_scale=np.asarray(frame.depth_scale, dtype=np.float64),
        )
        request = Request(
            f"{self.base_url.rstrip('/')}/predict",
            data=buffer.getvalue(),
            headers={"Content-Type": "application/x-npz"},
            method="POST",
        )
        payload = self._request(request)
        result = json.loads(payload.decode("utf-8"))
        try:
            transform = result["T_cam_obj"]
            size = result["size_m"]
        except KeyError as error:
            raise F2GServiceError(
                f"F2G response is missing {error.args[0]!r}: {result}"
            ) from error
        return F2GPrediction(
            object_to_camera=np.asarray(transform, dtype=np.float64),
            size_m=np.asarray(size, dtype=np.float64),
            metadata={
                key: value
                for key, value in result.items()
                if key not in {"T_cam_obj", "size_m"}
            },
        )
