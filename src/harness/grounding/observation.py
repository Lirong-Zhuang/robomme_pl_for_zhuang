"""Convert RoboMME columnar observations to one validated camera frame."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .types import CameraFrame


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _last(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("camera observation list is empty")
        return value[-1]
    return value


def _image(value: Any, *, channels: int | None) -> np.ndarray:
    array = _to_numpy(_last(value))
    while array.ndim > (3 if channels else 2) and array.shape[0] == 1:
        array = array[0]
    if channels is not None and array.ndim == 4:
        array = array[-1]
    if channels is None and array.ndim == 3:
        if array.shape[-1] == 1:
            array = array[..., 0]
        else:
            array = array[-1]
    return array


def _matrix(value: Any, rows: int, columns: int) -> np.ndarray:
    array = _to_numpy(_last(value))
    if array.size < rows * columns:
        raise ValueError(f"camera matrix has fewer than {rows * columns} values")
    return array.reshape(-1, rows, columns)[-1]


def camera_frame_from_robomme(
    obs: Mapping[str, Any],
    info: Mapping[str, Any],
    *,
    depth_scale: float,
    image_ref: str = "current_front",
) -> CameraFrame:
    """Read the latest aligned front-camera frame from reset/step output.

    ``depth_scale`` is deliberately required because RoboMME currently records
    depth as int16 without persisting its unit.
    """

    required_obs = (
        "front_rgb_list",
        "front_depth_list",
        "front_camera_extrinsic_list",
    )
    missing_obs = [key for key in required_obs if key not in obs]
    if missing_obs:
        raise ValueError(
            "RoboMME observation is missing grounding fields: "
            + ", ".join(missing_obs)
        )
    if "front_camera_intrinsic" not in info:
        raise ValueError("RoboMME info is missing front_camera_intrinsic")
    rgb = _image(obs["front_rgb_list"], channels=3)
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating) and np.nanmax(rgb) <= 1.0:
            rgb = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    depth = _image(obs["front_depth_list"], channels=None)
    intrinsic = _matrix(info["front_camera_intrinsic"], 3, 3)
    extrinsic = _matrix(obs["front_camera_extrinsic_list"], 3, 4)
    return CameraFrame(
        rgb=rgb,
        depth=depth,
        intrinsic=intrinsic,
        world_to_camera=extrinsic,
        depth_scale=depth_scale,
        image_ref=image_ref,
    )


def depth_statistics(frame: CameraFrame) -> dict[str, float | int]:
    """Return a small diagnostic used to verify the server's depth unit."""

    depth_m = frame.depth_m
    valid = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    if valid.size == 0:
        raise ValueError("front depth has no positive finite values")
    return {
        "valid_pixels": int(valid.size),
        "min_m": float(np.min(valid)),
        "median_m": float(np.median(valid)),
        "max_m": float(np.max(valid)),
    }
