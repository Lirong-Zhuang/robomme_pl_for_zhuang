"""Manual mask -> F2G -> projected 3D grounding-box diagnostic.

Required services and environments:

1. In terminal A, start the service in the micromamba ``f2g`` environment,
   using the same paths configured in ``scripts/eval_agentic.sh``::

       micromamba run -n f2g python scripts/f2g_pose_server.py \
       --checkpoint /path/to/f2g-pose.pt \
       --radio-repository /path/to/RADIO \
       --radio-checkpoint /path/to/radio-v2.5-l_half.pth.tar \
       --host 127.0.0.1 --port 8765

2. Edit the configuration block below, then run this file from the repository
   root in the ``robomme`` environment::

       micromamba activate robomme
       python src/harness/tests/test_grounding.py

The PNG and NPZ must both be placed in ``test_resource``. The NPZ is generated
by Agentic Eval before every grounding call and contains aligned depth,
intrinsic, world-to-camera, and depth-scale arrays. Only two images are written:
the black mask overlay and the projected F2G grounding box, both with the
manually supplied point rendered in yellow.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
TEST_DIR = Path(__file__).resolve().parent
for import_root in (SRC_ROOT, TEST_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from harness.grounding import F2GClient
from harness.grounding.types import CameraFrame
import test_opencv_manual_mask as manual_mask


# Experiment configuration. Edit these values; the run command stays one line.
IMAGE_NAME = "red_cube_only.png"
COLOR_HINT = "red"
POINT_XY = (120, 110)
OUTPUT_NAME = "red_grounding_1"
GROUNDING_INPUT_NAME = "grounding_input.npz"
F2G_URL = "http://127.0.0.1:8765"
MIN_COMPONENT_AREA = 50
MAX_COMPONENT_AREA_FRACTION = 0.08
MORPHOLOGY_PX = 3
HSV_HUE_TOLERANCE = 8
HSV_MIN_SATURATION = 120
HSV_MIN_VALUE = 20
WHITE_MAX_SATURATION = 55
WHITE_MIN_VALUE = 160
BLACK_MAX_VALUE = 70


def _grounding_input_name() -> str:
    name = GROUNDING_INPUT_NAME.strip()
    if not name or Path(name).name != name:
        raise ValueError(
            "GROUNDING_INPUT_NAME must be one NPZ filename inside test_resource"
        )
    return name


def _box_corners(size_m: object, np: object) -> object:
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
    points_object: object,
    object_to_camera: object,
    intrinsic: object,
    np: object,
) -> object:
    points = np.asarray(points_object, dtype=np.float64)
    transform = np.asarray(object_to_camera, dtype=np.float64)
    camera = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    if np.any(camera[:, 2] <= 1e-6):
        raise ValueError("F2G box contains points behind the camera")
    homogeneous = (np.asarray(intrinsic, dtype=np.float64) @ camera.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _draw_grounding_box(
    rgb: object,
    *,
    object_to_camera: object,
    size_m: object,
    intrinsic: object,
    point_xy: tuple[int, int],
    cv2: object,
    np: object,
) -> object:
    output = rgb.copy()
    corners = _box_corners(size_m, np)
    pixels = _project_points(corners, object_to_camera, intrinsic, np)
    pixels = np.rint(pixels).astype(int)
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
    for start, end in edges:
        cv2.line(
            output,
            tuple(int(value) for value in pixels[start]),
            tuple(int(value) for value in pixels[end]),
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    size = np.asarray(size_m, dtype=np.float64)
    axis_points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [size[0] / 2.0, 0.0, 0.0],
            [0.0, size[1] / 2.0, 0.0],
            [0.0, 0.0, size[2] / 2.0],
        ]
    )
    axes = np.rint(
        _project_points(axis_points, object_to_camera, intrinsic, np)
    ).astype(int)
    origin = tuple(int(value) for value in axes[0])
    for endpoint, color in zip(
        axes[1:],
        ((255, 0, 0), (0, 255, 0), (0, 80, 255)),
        strict=True,
    ):
        cv2.line(
            output,
            origin,
            tuple(int(value) for value in endpoint),
            color,
            3,
            cv2.LINE_AA,
        )
    cv2.circle(output, point_xy, 5, (255, 255, 0), -1)
    return output


def run_manual_grounding_test() -> None:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "the robomme environment must provide NumPy and OpenCV"
        ) from error

    manual_mask.IMAGE_NAME = IMAGE_NAME
    manual_mask.COLOR_HINT = COLOR_HINT
    manual_mask.POINT_XY = POINT_XY
    manual_mask.OUTPUT_NAME = OUTPUT_NAME
    manual_mask.MIN_COMPONENT_AREA = MIN_COMPONENT_AREA
    manual_mask.MAX_COMPONENT_AREA_FRACTION = MAX_COMPONENT_AREA_FRACTION
    manual_mask.MORPHOLOGY_PX = MORPHOLOGY_PX
    manual_mask.HSV_HUE_TOLERANCE = HSV_HUE_TOLERANCE
    manual_mask.HSV_MIN_SATURATION = HSV_MIN_SATURATION
    manual_mask.HSV_MIN_VALUE = HSV_MIN_VALUE
    manual_mask.WHITE_MAX_SATURATION = WHITE_MAX_SATURATION
    manual_mask.WHITE_MIN_VALUE = WHITE_MIN_VALUE
    manual_mask.BLACK_MAX_VALUE = BLACK_MAX_VALUE
    mask_result = manual_mask.run_manual_rgb_component_mask()
    rgb = mask_result["rgb"]
    mask = mask_result["mask"]
    point_xy = mask_result["point_xy"]
    output_name = str(mask_result["output_name"])
    input_path = manual_mask.RESOURCE_DIR / _grounding_input_name()
    if not input_path.is_file():
        raise FileNotFoundError(
            f"grounding input does not exist: {input_path}; copy "
            "grounding/ep_<id>/call_<id>/grounding_input.npz from an eval run"
        )

    with np.load(input_path, allow_pickle=False) as data:
        required = {"depth", "intrinsic", "depth_scale"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"grounding input NPZ is missing {sorted(missing)}")
        depth = data["depth"]
        intrinsic = data["intrinsic"]
        depth_scale = float(np.asarray(data["depth_scale"]).reshape(()))
        world_to_camera = (
            data["world_to_camera"]
            if "world_to_camera" in data.files
            else np.eye(4, dtype=np.float64)
        )
    if depth.shape != rgb.shape[:2]:
        raise ValueError(
            f"PNG shape {rgb.shape[:2]} does not match depth shape {depth.shape}"
        )

    frame = CameraFrame(
        rgb=rgb,
        depth=depth,
        intrinsic=intrinsic,
        world_to_camera=world_to_camera,
        depth_scale=depth_scale,
        image_ref="manual_grounding_front",
    )
    client = F2GClient(
        base_url=F2G_URL,
    )
    health = client.health()
    prediction = client.predict(frame, mask)
    grounded_rgb = _draw_grounding_box(
        rgb,
        object_to_camera=prediction.object_to_camera,
        size_m=prediction.size_m,
        intrinsic=intrinsic,
        point_xy=point_xy,
        cv2=cv2,
        np=np,
    )
    manual_mask.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grounding_path = manual_mask.OUTPUT_DIR / f"{output_name}_grounding.png"
    cv2.imwrite(
        str(grounding_path),
        cv2.cvtColor(grounded_rgb, cv2.COLOR_RGB2BGR),
    )
    print(
        json.dumps(
            {
                "f2g_health": health,
                "T_cam_obj": prediction.object_to_camera.tolist(),
                "size_m": prediction.size_m.tolist(),
                "f2g_metadata": dict(prediction.metadata),
                "artifacts": {
                    "mask_overlay": str(mask_result["mask_overlay_path"]),
                    "grounding": str(grounding_path.resolve()),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    assert grounding_path.is_file()


def test_manual_grounding() -> None:
    """Optional pytest entry point; direct Python execution needs no pytest."""

    import pytest

    if os.environ.get("RUN_AGENTIC_GROUNDING_TEST") != "1":
        pytest.skip("set RUN_AGENTIC_GROUNDING_TEST=1 to call F2G")
    run_manual_grounding_test()


if __name__ == "__main__":
    run_manual_grounding_test()
