"""Shared overlays for OpenCV masks and F2G camera-frame predictions."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from .types import SegmentationResult


YELLOW_RGB = (255, 255, 0)
GREEN_RGB = (0, 255, 0)
AXIS_COLORS_RGB = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 80, 255),
)


def _box_corners(size_m: np.ndarray) -> np.ndarray:
    half_x, half_y, half_z = np.asarray(size_m, dtype=np.float64) / 2.0
    return np.asarray(
        [
            [-half_x, -half_y, -half_z],
            [half_x, -half_y, -half_z],
            [half_x, half_y, -half_z],
            [-half_x, half_y, -half_z],
            [-half_x, -half_y, half_z],
            [half_x, -half_y, half_z],
            [half_x, half_y, half_z],
            [-half_x, half_y, half_z],
        ],
        dtype=np.float64,
    )


def _project_points(
    points_object: np.ndarray,
    object_to_camera: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_object, dtype=np.float64)
    transform = np.asarray(object_to_camera, dtype=np.float64)
    camera = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    if np.any(camera[:, 2] <= 1e-6):
        raise ValueError("F2G box contains points behind the camera")
    homogeneous = (np.asarray(intrinsic, dtype=np.float64) @ camera.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _draw_point(output: np.ndarray, point_xy: tuple[float, float] | None) -> None:
    if point_xy is None:
        return
    point = (int(round(point_xy[0])), int(round(point_xy[1])))
    cv2.circle(output, point, 5, YELLOW_RGB, -1)


def draw_mask_overlay(
    rgb: np.ndarray,
    segmentations: Iterable[SegmentationResult],
) -> np.ndarray:
    """Paint all selected mask pixels black and VLM points yellow."""

    output = np.asarray(rgb, dtype=np.uint8).copy()
    segmentations = tuple(segmentations)
    for segmentation in segmentations:
        if segmentation.mask.shape != output.shape[:2]:
            raise ValueError("segmentation mask must be aligned with RGB")
        output[segmentation.mask] = (0, 0, 0)
    for segmentation in segmentations:
        _draw_point(output, segmentation.point_xy)
    return output


def draw_grounding_overlay(
    rgb: np.ndarray,
    *,
    intrinsic: np.ndarray,
    predictions: Iterable[
        tuple[np.ndarray, np.ndarray, tuple[float, float] | None]
    ],
) -> np.ndarray:
    """Draw all F2G oriented boxes, axes, and their corresponding VLM points."""

    output = np.asarray(rgb, dtype=np.uint8).copy()
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    points: list[tuple[float, float] | None] = []
    for object_to_camera, size_m, point_xy in predictions:
        size = np.asarray(size_m, dtype=np.float64)
        pixels = np.rint(
            _project_points(
                _box_corners(size),
                object_to_camera,
                intrinsic,
            )
        ).astype(int)
        for start, end in edges:
            cv2.line(
                output,
                tuple(int(value) for value in pixels[start]),
                tuple(int(value) for value in pixels[end]),
                GREEN_RGB,
                2,
                cv2.LINE_AA,
            )

        axis_points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [size[0] / 2.0, 0.0, 0.0],
                [0.0, size[1] / 2.0, 0.0],
                [0.0, 0.0, size[2] / 2.0],
            ]
        )
        axes = np.rint(
            _project_points(axis_points, object_to_camera, intrinsic)
        ).astype(int)
        origin = tuple(int(value) for value in axes[0])
        for endpoint, color in zip(
            axes[1:], AXIS_COLORS_RGB, strict=True
        ):
            cv2.line(
                output,
                origin,
                tuple(int(value) for value in endpoint),
                color,
                3,
                cv2.LINE_AA,
            )
        points.append(point_xy)

    for point_xy in points:
        _draw_point(output, point_xy)
    return output
