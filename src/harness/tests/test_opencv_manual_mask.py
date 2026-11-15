"""Manual RGB-component mask diagnostic for one RoboMME observation.

Required environment: micromamba ``robomme`` (Python, NumPy, OpenCV).

Run from the repository root::

    micromamba activate robomme
    python src/harness/tests/test_opencv_manual_mask.py

Edit the configuration constants below before running. Direct execution does
not require pytest. If pytest is installed, its wrapper remains skipped unless
``RUN_AGENTIC_OPENCV_MASK_TEST=1``.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
RESOURCE_DIR = Path(__file__).resolve().parent / "test_resource"
OUTPUT_DIR = REPO_ROOT / "runs" / "tests" / "grounding"

# Manual defaults. These can be edited directly for a quick local experiment.
IMAGE_NAME = "red_cube_only.png"
COLOR_HINT = "red"
POINT_XY = (120, 110)
OUTPUT_NAME = "red_test_3"
MIN_COMPONENT_AREA = 50
MAX_COMPONENT_AREA_FRACTION = 0.08
MORPHOLOGY_PX = 3
HSV_HUE_TOLERANCE = 8
HSV_MIN_SATURATION = 120
HSV_MIN_VALUE = 20
WHITE_MAX_SATURATION = 55
WHITE_MIN_VALUE = 160
BLACK_MAX_VALUE = 70


def _manual_settings() -> tuple[str, str, tuple[int, int], str]:
    from harness.grounding.opencv_segmenter import RGB_PROTOTYPES

    image_name = IMAGE_NAME.strip()
    color_hint = COLOR_HINT.strip().lower()
    point_xy = (int(POINT_XY[0]), int(POINT_XY[1]))
    output_name = OUTPUT_NAME.strip()
    if not image_name or Path(image_name).name != image_name:
        raise ValueError("IMAGE_NAME must be one filename inside test_resource")
    if color_hint not in RGB_PROTOTYPES:
        raise ValueError(f"COLOR_HINT must be one of {sorted(RGB_PROTOTYPES)}")
    if (
        not output_name
        or Path(output_name).name != output_name
        or any(
            not (character.isalnum() or character in {"-", "_"})
            for character in output_name
        )
    ):
        raise ValueError(
            "OUTPUT_NAME may contain only letters, numbers, "
            "hyphens, and underscores"
        )
    return image_name, color_hint, point_xy, output_name


def run_manual_rgb_component_mask() -> dict[str, object]:
    """Write the matching component nearest a manually supplied image point."""

    try:
        import cv2
        import numpy as np
        from harness.grounding.opencv_segmenter import DEFAULT_PROFILES
        from harness.grounding.opencv_segmenter import OpenCVSegmenter
        from harness.grounding.types import CameraFrame
        from harness.grounding.visualization import draw_mask_overlay
    except ImportError as error:
        raise RuntimeError(
            "the robomme environment must provide NumPy and OpenCV"
        ) from error
    image_name, color_hint, point_xy, output_name = _manual_settings()
    image_path = RESOURCE_DIR / image_name
    if not image_path.is_file():
        raise FileNotFoundError(
            f"test image does not exist: {image_path}; place the selected image "
            "inside src/harness/tests/test_resource first"
        )

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"OpenCV could not decode test image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    point_x, point_y = point_xy
    if not (0 <= point_x < width and 0 <= point_y < height):
        raise ValueError(
            f"point_xy={point_xy} is outside image bounds width={width}, "
            f"height={height}"
        )

    profile_name = "manual_colored"
    profile = replace(
        DEFAULT_PROFILES["compact_colored"],
        min_area=MIN_COMPONENT_AREA,
        max_area_fraction=MAX_COMPONENT_AREA_FRACTION,
        morphology_px=MORPHOLOGY_PX,
        hsv_hue_tolerance=HSV_HUE_TOLERANCE,
        hsv_min_saturation=HSV_MIN_SATURATION,
        hsv_min_value=HSV_MIN_VALUE,
        white_max_saturation=WHITE_MAX_SATURATION,
        white_min_value=WHITE_MIN_VALUE,
        black_max_value=BLACK_MAX_VALUE,
    )
    frame = CameraFrame(
        rgb=rgb,
        depth=np.ones((height, width), dtype=np.uint16),
        intrinsic=np.eye(3, dtype=np.float64),
        world_to_camera=np.eye(4, dtype=np.float64),
        depth_scale=1.0,
        image_ref="manual_mask_front",
    )
    segmentation = OpenCVSegmenter(
        profiles={profile_name: profile},
        color_profile=profile,
    ).segment(
        frame,
        point_xy=point_xy,
        color_hint=color_hint,
        profile_name=profile_name,
    )
    selected_mask = segmentation.mask
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_stem = output_name
    overlay_path = OUTPUT_DIR / f"{artifact_stem}_mask_overlay.png"
    for legacy_suffix in (
        "raw_mask.png",
        "selected_mask.png",
        "overlay.png",
        "diagnostics.json",
        "grounding.png",
    ):
        (OUTPUT_DIR / f"{artifact_stem}_{legacy_suffix}").unlink(missing_ok=True)

    overlay_rgb = draw_mask_overlay(rgb, [segmentation])
    cv2.imwrite(
        str(overlay_path),
        cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
    )

    diagnostics = {
        "image_path": str(image_path.resolve()),
        "image_shape_hw": [height, width],
        "color_hint": color_hint,
        "point_xy": list(point_xy),
        "output_name": output_name,
        "segmentation_profile": profile_name,
        "confidence": segmentation.confidence,
        "selected": dict(segmentation.diagnostics),
        "mask_overlay": str(overlay_path.resolve()),
    }
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))

    assert int(np.count_nonzero(selected_mask)) >= MIN_COMPONENT_AREA
    assert overlay_path.is_file()
    return {
        "rgb": rgb,
        "mask": selected_mask,
        "point_xy": point_xy,
        "output_name": output_name,
        "image_path": image_path,
        "mask_overlay_path": overlay_path,
        "diagnostics": diagnostics,
    }


def test_nearest_manual_rgb_component_mask() -> None:
    """Optional pytest entry point; direct Python execution needs no pytest."""

    import pytest

    if os.environ.get("RUN_AGENTIC_OPENCV_MASK_TEST") != "1":
        pytest.skip("set RUN_AGENTIC_OPENCV_MASK_TEST=1 to generate mask artifacts")
    run_manual_rgb_component_mask()


if __name__ == "__main__":
    run_manual_rgb_component_mask()
