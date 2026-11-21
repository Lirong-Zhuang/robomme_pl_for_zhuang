"""Coordinate conversion from F2G camera axes to RoboMME world axes."""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import CameraFrame


def image_point_to_world(
    frame: CameraFrame,
    point_xy: tuple[float, float],
    *,
    search_radius_px: int = 3,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    """Back-project one VLM point using the nearest valid aligned depth pixel."""

    if search_radius_px < 0:
        raise ValueError("search_radius_px must be non-negative")
    x, y = (float(point_xy[0]), float(point_xy[1]))
    height, width = frame.depth.shape
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"point_xy={point_xy} is outside the image")
    center_x = min(max(int(round(x)), 0), width - 1)
    center_y = min(max(int(round(y)), 0), height - 1)
    depth_m = frame.depth_m

    candidates: list[tuple[float, int, int, float]] = []
    for pixel_y in range(
        max(0, center_y - search_radius_px),
        min(height, center_y + search_radius_px + 1),
    ):
        for pixel_x in range(
            max(0, center_x - search_radius_px),
            min(width, center_x + search_radius_px + 1),
        ):
            value = float(depth_m[pixel_y, pixel_x])
            if np.isfinite(value) and value > 0:
                distance = (pixel_x - x) ** 2 + (pixel_y - y) ** 2
                candidates.append((distance, pixel_x, pixel_y, value))
    if not candidates:
        raise ValueError(
            f"no valid depth within {search_radius_px}px of point_xy={point_xy}"
        )
    _, depth_x, depth_y, depth_value = min(candidates)
    camera_ray = np.linalg.inv(frame.intrinsic) @ np.asarray(
        [float(depth_x), float(depth_y), 1.0], dtype=np.float64
    )
    camera_point = camera_ray * depth_value
    world_point = np.linalg.inv(frame.world_to_camera) @ np.asarray(
        [*camera_point, 1.0], dtype=np.float64
    )
    if not np.all(np.isfinite(world_point[:3])):
        raise ValueError("point-depth back-projection produced non-finite world coordinates")
    return (
        tuple(float(item) for item in world_point[:3]),
        {
            "requested_point_xy": [x, y],
            "depth_pixel_xy": [depth_x, depth_y],
            "depth_m": depth_value,
            "search_radius_px": search_radius_px,
        },
    )


def object_to_world(
    world_to_camera: np.ndarray,
    object_to_camera: np.ndarray,
) -> np.ndarray:
    """Return T_world_obj from ManiSkill T_cam_world and F2G T_cam_obj."""

    world_to_camera = np.asarray(world_to_camera, dtype=np.float64)
    object_to_camera = np.asarray(object_to_camera, dtype=np.float64)
    if world_to_camera.shape == (3, 4):
        expanded = np.eye(4, dtype=np.float64)
        expanded[:3, :4] = world_to_camera
        world_to_camera = expanded
    if world_to_camera.shape != (4, 4) or object_to_camera.shape != (4, 4):
        raise ValueError("transforms must have shape 4x4 (or 3x4 extrinsic)")
    return np.linalg.inv(world_to_camera) @ object_to_camera


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> tuple[float, ...]:
    """Convert a proper 3x3 rotation matrix to a normalized WXYZ quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape 3x3")
    # Project small numeric errors onto SO(3) before converting.
    u, _, vh = np.linalg.svd(matrix)
    matrix = u @ vh
    if np.linalg.det(matrix) < 0:
        u[:, -1] *= -1
        matrix = u @ vh
    trace = float(np.trace(matrix))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(matrix)))
        if i == 0:
            s = np.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    return tuple(float(item) for item in quaternion)
