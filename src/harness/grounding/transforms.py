"""Coordinate conversion from F2G camera axes to RoboMME world axes."""

from __future__ import annotations

import numpy as np


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
