"""Runtime data types used between OpenCV, F2G, and action binding."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    """One aligned front-camera RGB-D observation.

    ``world_to_camera`` follows ManiSkill's ``extrinsic_cv`` convention.
    ``depth_scale`` is metres per raw depth unit (0.001 for millimetres).
    """

    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    world_to_camera: np.ndarray
    depth_scale: float
    image_ref: str = "current_front"

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth)
        intrinsic = np.asarray(self.intrinsic, dtype=np.float64)
        extrinsic = np.asarray(self.world_to_camera, dtype=np.float64)
        depth_scale = float(self.depth_scale)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape HxWx3")
        if rgb.dtype != np.uint8:
            raise ValueError("rgb must be uint8 in RGB order")
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
            raise ValueError("depth must have shape HxW aligned with rgb")
        if not np.issubdtype(depth.dtype, np.number):
            raise ValueError("depth must be numeric")
        if np.issubdtype(depth.dtype, np.signedinteger) and np.any(depth < 0):
            raise ValueError(
                "depth contains negative integers; check int16 overflow before F2G"
            )
        if intrinsic.shape != (3, 3):
            raise ValueError("intrinsic must have shape 3x3")
        if extrinsic.shape == (3, 4):
            extrinsic_4x4 = np.eye(4, dtype=np.float64)
            extrinsic_4x4[:3, :4] = extrinsic
            extrinsic = extrinsic_4x4
        if extrinsic.shape != (4, 4):
            raise ValueError("world_to_camera must have shape 3x4 or 4x4")
        if not np.all(np.isfinite(intrinsic)) or not np.all(np.isfinite(extrinsic)):
            raise ValueError("camera matrices must contain finite values")
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ValueError("camera focal lengths must be positive")
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError("depth_scale must be finite and positive")
        if not self.image_ref.strip():
            raise ValueError("image_ref must be non-empty")
        object.__setattr__(self, "rgb", np.ascontiguousarray(rgb))
        object.__setattr__(self, "depth", np.ascontiguousarray(depth))
        object.__setattr__(self, "intrinsic", intrinsic)
        object.__setattr__(self, "world_to_camera", extrinsic)
        object.__setattr__(self, "depth_scale", depth_scale)

    @property
    def depth_m(self) -> np.ndarray:
        return self.depth.astype(np.float32, copy=False) * self.depth_scale


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    confidence: float
    profile: str
    point_xy: tuple[float, float] | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask)
        if mask.ndim != 2:
            raise ValueError("segmentation mask must have shape HxW")
        mask = np.ascontiguousarray(mask.astype(bool, copy=False))
        pixels = int(np.count_nonzero(mask))
        if pixels < 50:
            raise ValueError(
                f"segmentation mask has {pixels} pixels; F2G requires at least 50"
            )
        confidence = float(self.confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("segmentation confidence must be between 0 and 1")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class F2GPrediction:
    object_to_camera: np.ndarray
    size_m: np.ndarray
    points_camera: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        transform = np.asarray(self.object_to_camera, dtype=np.float64)
        size = np.asarray(self.size_m, dtype=np.float64).reshape(-1)
        if transform.shape != (4, 4):
            raise ValueError("F2G T_cam_obj must have shape 4x4")
        if size.shape != (3,) or np.any(size <= 0):
            raise ValueError("F2G size_m must contain three positive values")
        if not np.all(np.isfinite(transform)) or not np.all(np.isfinite(size)):
            raise ValueError("F2G prediction must contain finite values")
        points = self.points_camera
        if points is not None:
            points = np.asarray(points, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("F2G points_camera must have shape Nx3")
            if not np.all(np.isfinite(points)):
                raise ValueError("F2G points_camera must contain finite values")
        object.__setattr__(self, "object_to_camera", transform)
        object.__setattr__(self, "size_m", size)
        object.__setattr__(self, "points_camera", points)
