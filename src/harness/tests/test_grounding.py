"""Manual mask -> F2G -> projected 3D grounding-box diagnostic.

Required services and environments:

1. In terminal A, start the configured standalone F2G service::

       bash scripts/serve_f2g_for_test.sh

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
from harness.grounding.visualization import draw_grounding_overlay
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
    grounded_rgb = draw_grounding_overlay(
        rgb,
        intrinsic=intrinsic,
        predictions=[
            (
                prediction.object_to_camera,
                prediction.size_m,
                point_xy,
            )
        ],
    )
    manual_mask.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grounding_path = (
        manual_mask.OUTPUT_DIR / f"{output_name}_grounding_overlay.png"
    )
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
                    "grounding_overlay": str(grounding_path.resolve()),
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
