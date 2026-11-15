"""Manual RGB-component mask diagnostic for one RoboMME observation.

Required environment: micromamba ``robomme`` (Python, NumPy, OpenCV).

Run from the repository root::

    micromamba activate robomme
    python src/harness/tests/test_opencv_manual_mask.py

Edit the constants below or override them with the documented environment
variables. Running this file directly does not require pytest. If pytest is
installed, its wrapper test remains skipped unless
``RUN_AGENTIC_OPENCV_MASK_TEST=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
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

# Trusted RGB prototypes used by this diagnostic. OpenCV converts the selected
# prototype to HSV so illumination changes do not exclude darker object faces.
RGB_PROTOTYPES = {
    "red": (220, 45, 45),
    "green": (45, 180, 70),
    "blue": (45, 90, 220),
    "yellow": (235, 210, 40),
    "orange": (235, 130, 35),
    "purple": (145, 65, 190),
    "cyan": (40, 190, 200),
    "magenta": (210, 55, 175),
    "white": (235, 235, 235),
    "black": (25, 25, 25),
}


def _manual_settings() -> tuple[str, str, tuple[int, int], str]:
    image_name = os.environ.get("AGENTIC_MASK_IMAGE", IMAGE_NAME).strip()
    color_hint = os.environ.get("AGENTIC_MASK_COLOR", COLOR_HINT).strip().lower()
    point_xy = (
        int(os.environ.get("AGENTIC_MASK_X", POINT_XY[0])),
        int(os.environ.get("AGENTIC_MASK_Y", POINT_XY[1])),
    )
    output_name = os.environ.get(
        "AGENTIC_MASK_OUTPUT_NAME",
        OUTPUT_NAME,
    ).strip()
    if not image_name or Path(image_name).name != image_name:
        raise ValueError("AGENTIC_MASK_IMAGE must be one filename inside test_resource")
    if color_hint not in RGB_PROTOTYPES:
        raise ValueError(
            f"AGENTIC_MASK_COLOR must be one of {sorted(RGB_PROTOTYPES)}"
        )
    if (
        not output_name
        or Path(output_name).name != output_name
        or any(
            not (character.isalnum() or character in {"-", "_"})
            for character in output_name
        )
    ):
        raise ValueError(
            "AGENTIC_MASK_OUTPUT_NAME may contain only letters, numbers, "
            "hyphens, and underscores"
        )
    return image_name, color_hint, point_xy, output_name


def run_manual_rgb_component_mask() -> None:
    """Write the matching component nearest a manually supplied image point."""

    try:
        import cv2
        import numpy as np
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

    prototype = np.asarray(RGB_PROTOTYPES[color_hint], dtype=np.float32)
    distance = np.linalg.norm(rgb.astype(np.float32) - prototype, axis=2)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    prototype_hsv = cv2.cvtColor(
        np.asarray([[RGB_PROTOTYPES[color_hint]]], dtype=np.uint8),
        cv2.COLOR_RGB2HSV,
    )[0, 0]
    if color_hint == "white":
        raw_mask = (
            (hsv[..., 1] <= WHITE_MAX_SATURATION)
            & (hsv[..., 2] >= WHITE_MIN_VALUE)
        )
        mask_rule = "low_saturation_high_value"
    elif color_hint == "black":
        raw_mask = hsv[..., 2] <= BLACK_MAX_VALUE
        mask_rule = "low_value"
    else:
        hue_delta = np.abs(
            hsv[..., 0].astype(np.int16) - int(prototype_hsv[0])
        )
        circular_hue_delta = np.minimum(hue_delta, 180 - hue_delta)
        raw_mask = (
            (circular_hue_delta <= HSV_HUE_TOLERANCE)
            & (hsv[..., 1] >= HSV_MIN_SATURATION)
            & (hsv[..., 2] >= HSV_MIN_VALUE)
        )
        mask_rule = "circular_hue_saturation_value"
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPHOLOGY_PX, MORPHOLOGY_PX),
    )
    cleaned = cv2.morphologyEx(
        raw_mask.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        kernel,
    )
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel) > 0

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned.astype(np.uint8),
        connectivity=8,
    )
    image_area = height * width
    candidates = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_COMPONENT_AREA:
            continue
        if area > MAX_COMPONENT_AREA_FRACTION * image_area:
            continue
        component_mask = labels == label
        if component_mask[point_y, point_x]:
            distance_px = 0.0
        else:
            inverse = (~component_mask).astype(np.uint8)
            distance_map = cv2.distanceTransform(inverse, cv2.DIST_L2, 3)
            distance_px = float(distance_map[point_y, point_x])
        candidates.append(
            {
                "label": label,
                "mask": component_mask,
                "area_px": area,
                "bbox_xywh": [
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_HEIGHT]),
                ],
                "center_xy": [float(value) for value in centroids[label]],
                "point_distance_px": distance_px,
                "mean_rgb_distance": float(np.mean(distance[component_mask])),
            }
        )
    if not candidates:
        raise RuntimeError(
            f"no {color_hint!r} component survived mask_rule={mask_rule}, "
            f"min_area={MIN_COMPONENT_AREA}, and "
            f"max_area_fraction={MAX_COMPONENT_AREA_FRACTION}"
        )

    selected = min(
        candidates,
        key=lambda item: (item["point_distance_px"], -item["area_px"]),
    )
    selected_mask = selected.pop("mask")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_stem = output_name
    raw_mask_path = OUTPUT_DIR / f"{artifact_stem}_raw_mask.png"
    selected_mask_path = OUTPUT_DIR / f"{artifact_stem}_selected_mask.png"
    overlay_path = OUTPUT_DIR / f"{artifact_stem}_overlay.png"
    diagnostics_path = OUTPUT_DIR / f"{artifact_stem}_diagnostics.json"

    cv2.imwrite(str(raw_mask_path), raw_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(selected_mask_path), selected_mask.astype(np.uint8) * 255)
    overlay_rgb = rgb.copy()
    overlay_rgb[selected_mask] = (0, 0, 0)
    # Match Oracle Plus target-point rendering: a filled 5 px RGB yellow dot.
    cv2.circle(overlay_rgb, point_xy, 5, (255, 255, 0), -1)
    cv2.imwrite(
        str(overlay_path),
        cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
    )

    diagnostics = {
        "image_path": str(image_path.resolve()),
        "image_shape_hw": [height, width],
        "color_hint": color_hint,
        "prototype_rgb": list(RGB_PROTOTYPES[color_hint]),
        "prototype_hsv": [int(value) for value in prototype_hsv],
        "point_xy": list(point_xy),
        "output_name": output_name,
        "mask_rule": mask_rule,
        "hsv_thresholds": {
            "hue_tolerance": HSV_HUE_TOLERANCE,
            "min_saturation": HSV_MIN_SATURATION,
            "min_value": HSV_MIN_VALUE,
            "white_max_saturation": WHITE_MAX_SATURATION,
            "white_min_value": WHITE_MIN_VALUE,
            "black_max_value": BLACK_MAX_VALUE,
        },
        "raw_mask_pixels": int(np.count_nonzero(raw_mask)),
        "candidate_count": len(candidates),
        "selected": selected,
        "candidates": [
            {key: value for key, value in candidate.items() if key != "mask"}
            for candidate in candidates
        ],
        "artifacts": {
            "raw_mask": str(raw_mask_path.resolve()),
            "selected_mask": str(selected_mask_path.resolve()),
            "overlay": str(overlay_path.resolve()),
        },
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))

    assert int(np.count_nonzero(selected_mask)) >= MIN_COMPONENT_AREA
    assert raw_mask_path.is_file()
    assert selected_mask_path.is_file()
    assert overlay_path.is_file()
    assert diagnostics_path.is_file()


def test_nearest_manual_rgb_component_mask() -> None:
    """Optional pytest entry point; direct Python execution needs no pytest."""

    import pytest

    if os.environ.get("RUN_AGENTIC_OPENCV_MASK_TEST") != "1":
        pytest.skip("set RUN_AGENTIC_OPENCV_MASK_TEST=1 to generate mask artifacts")
    run_manual_rgb_component_mask()


if __name__ == "__main__":
    run_manual_rgb_component_mask()
