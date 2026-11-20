"""Manual mask -> F2G -> self-contained interactive pose viewer.

Preferred one-terminal launcher::

       bash scripts/test_grounding_pos.sh

The launcher starts and stops F2G automatically.  For separate service and
test processes, use the following two-terminal workflow:

1. In terminal A, start the configured standalone F2G service::

       bash scripts/serve_f2g_for_test.sh

2. Edit the experiment configuration below, then run this file from the
   repository root in the ``robomme`` environment::

       micromamba activate robomme
       python src/harness/tests/test_grounding_pos.py

The configured PNG and ``grounding_input.npz`` must both be present in
``test_resource``.  This diagnostic writes a grounding-overlay PNG and one
self-contained HTML page containing that overlay, the observed masked RGB-D
cloud, F2G's completed cloud, its oriented bounding box, and its object-frame
axes.  Drag the 3D view to rotate, use the mouse wheel to zoom, and press ``r``
to reset the view.
"""

from __future__ import annotations

import base64
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
TEST_DIR = Path(__file__).resolve().parent
for import_root in (SRC_ROOT, TEST_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from harness.grounding import F2GClient
from harness.grounding.opencv_segmenter import DEFAULT_PROFILES
from harness.grounding.opencv_segmenter import OpenCVSegmenter
from harness.grounding.types import CameraFrame
import test_opencv_manual_mask as manual_mask


# Experiment configuration. Edit these values; the run command stays one line.
IMAGE_NAME = "red_cube_only.png"
COLOR_HINT = "red"
POINT_XY = (120, 110)
OUTPUT_DIR = Path(
    os.environ.get("GROUNDING_OUTPUT_DIR", str(manual_mask.OUTPUT_DIR))
).expanduser()
GROUNDING_INPUT_NAME = "grounding_input.npz"
F2G_URL = "http://127.0.0.1:8765"
MAX_DISPLAY_POINTS = 2048
MIN_COMPONENT_AREA = 50
MAX_COMPONENT_AREA_FRACTION = 0.08
MORPHOLOGY_PX = 3
HSV_HUE_TOLERANCE = 8
HSV_MIN_SATURATION = 120
HSV_MIN_VALUE = 20
WHITE_MAX_SATURATION = 55
WHITE_MIN_VALUE = 160
BLACK_MAX_VALUE = 70


def _safe_resource_name(value: str, *, setting: str) -> str:
    name = value.strip()
    if not name or Path(name).name != name:
        raise ValueError(f"{setting} must be one filename inside test_resource")
    return name


def _load_rgb_and_grounding_input() -> tuple[Any, Any, Any, float, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "the robomme environment must provide NumPy and OpenCV"
        ) from error

    image_name = _safe_resource_name(IMAGE_NAME, setting="IMAGE_NAME")
    input_name = _safe_resource_name(
        GROUNDING_INPUT_NAME,
        setting="GROUNDING_INPUT_NAME",
    )
    image_path = manual_mask.RESOURCE_DIR / image_name
    if not image_path.is_file():
        raise FileNotFoundError(
            f"test image does not exist: {image_path}; place it inside "
            "src/harness/tests/test_resource"
        )
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"OpenCV could not decode test image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    input_path = manual_mask.RESOURCE_DIR / input_name
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
        intrinsic = np.asarray(data["intrinsic"], dtype=np.float64)
        depth_scale = float(np.asarray(data["depth_scale"]).reshape(()))
        world_to_camera = (
            np.asarray(data["world_to_camera"], dtype=np.float64)
            if "world_to_camera" in data.files
            else np.eye(4, dtype=np.float64)
        )
    if depth.shape != rgb.shape[:2]:
        raise ValueError(
            f"PNG shape {rgb.shape[:2]} does not match depth shape {depth.shape}"
        )
    return rgb, depth, intrinsic, depth_scale, world_to_camera


def _segment_object(frame: CameraFrame) -> Any:
    color_hint = COLOR_HINT.strip().lower()
    point_xy = (int(POINT_XY[0]), int(POINT_XY[1]))
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
    return OpenCVSegmenter(
        profiles={profile_name: profile},
        color_profile=profile,
    ).segment(
        frame,
        point_xy=point_xy,
        color_hint=color_hint,
        profile_name=profile_name,
    )


def _masked_camera_cloud(
    frame: CameraFrame,
    mask: Any,
    *,
    max_points: int,
) -> tuple[Any, Any, int]:
    """Back-project valid masked depth pixels into F2G camera coordinates."""

    import numpy as np

    if max_points <= 0:
        raise ValueError("MAX_DISPLAY_POINTS must be positive")
    depth_m = np.asarray(frame.depth, dtype=np.float64) * frame.depth_scale
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth_m) & (depth_m > 0)
    pixel_y, pixel_x = np.nonzero(valid)
    total_points = int(pixel_x.size)
    if total_points == 0:
        raise ValueError("selected mask contains no valid depth pixels")
    if total_points > max_points:
        selected = np.linspace(0, total_points - 1, max_points, dtype=np.int64)
        pixel_x = pixel_x[selected]
        pixel_y = pixel_y[selected]

    z = depth_m[pixel_y, pixel_x]
    fx = float(frame.intrinsic[0, 0])
    fy = float(frame.intrinsic[1, 1])
    cx = float(frame.intrinsic[0, 2])
    cy = float(frame.intrinsic[1, 2])
    if fx == 0 or fy == 0:
        raise ValueError("camera focal lengths must be non-zero")
    points = np.column_stack(
        (
            (pixel_x.astype(np.float64) - cx) * z / fx,
            (pixel_y.astype(np.float64) - cy) * z / fy,
            z,
        )
    )
    colors = np.asarray(frame.rgb[pixel_y, pixel_x], dtype=np.uint8)
    return points, colors, total_points


def _png_data_url(rgb: Any) -> str:
    import cv2

    success, encoded = cv2.imencode(
        ".png",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    )
    if not success:
        raise RuntimeError("OpenCV could not encode the source image")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _viewer_html(
    *,
    rgb: Any,
    point_xy: tuple[int, int],
    points_camera: Any,
    colors_rgb: Any,
    total_masked_points: int,
    object_to_camera: Any,
    size_m: Any,
    segmentation: Any,
    f2g_metadata: dict[str, Any],
    completed_points_camera: Any | None = None,
    ground_truth_object_to_camera: Any | None = None,
    ground_truth_size_m: Any | None = None,
) -> str:
    import numpy as np

    transform = np.asarray(object_to_camera, dtype=np.float64)
    size = np.asarray(size_m, dtype=np.float64).reshape(3)
    points = np.asarray(points_camera, dtype=np.float64)
    colors = np.asarray(colors_rgb, dtype=np.uint8)
    completed_points = (
        np.empty((0, 3), dtype=np.float64)
        if completed_points_camera is None
        else np.asarray(completed_points_camera, dtype=np.float64)
    )
    if completed_points.ndim != 2 or completed_points.shape[1] != 3:
        raise ValueError("completed F2G points must have shape Nx3")
    if not np.all(np.isfinite(completed_points)):
        raise ValueError("completed F2G points must be finite")
    if completed_points.shape[0] > 8192:
        selected = np.linspace(
            0,
            completed_points.shape[0] - 1,
            8192,
            dtype=np.int64,
        )
        completed_points = completed_points[selected]
    origin = transform[:3, 3]
    rotation = transform[:3, :3]
    half = size / 2.0
    local_corners = np.asarray(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    box_corners = (rotation @ local_corners.T).T + origin
    ground_truth_box_corners: list[list[float]] = []
    ground_truth_text = "not supplied"
    if ground_truth_object_to_camera is not None or ground_truth_size_m is not None:
        if ground_truth_object_to_camera is None or ground_truth_size_m is None:
            raise ValueError(
                "ground-truth transform and size must be supplied together"
            )
        ground_truth_transform = np.asarray(
            ground_truth_object_to_camera,
            dtype=np.float64,
        )
        ground_truth_size = np.asarray(
            ground_truth_size_m,
            dtype=np.float64,
        ).reshape(3)
        if ground_truth_transform.shape != (4, 4):
            raise ValueError("ground-truth transform must have shape 4x4")
        ground_truth_half = ground_truth_size / 2.0
        ground_truth_local_corners = np.sign(local_corners) * ground_truth_half
        ground_truth_box_corners_array = (
            ground_truth_transform[:3, :3]
            @ ground_truth_local_corners.T
        ).T + ground_truth_transform[:3, 3]
        ground_truth_box_corners = (
            ground_truth_box_corners_array.round(7).tolist()
        )
        ground_truth_text = json.dumps(
            {
                "T_cam_obj": ground_truth_transform.tolist(),
                "size_m": ground_truth_size.tolist(),
            },
            indent=2,
        )
    axis_length = max(float(np.max(size)) * 0.8, 0.02)
    axes = np.stack(
        (
            origin,
            origin + rotation[:, 0] * axis_length,
            origin + rotation[:, 1] * axis_length,
            origin + rotation[:, 2] * axis_length,
        )
    )
    payload = {
        "points": points.round(7).tolist(),
        "colors": colors.tolist(),
        "completedPoints": completed_points.round(7).tolist(),
        "boxCorners": box_corners.round(7).tolist(),
        "groundTruthBoxCorners": ground_truth_box_corners,
        "axes": axes.round(7).tolist(),
        "origin": origin.round(7).tolist(),
    }
    payload_json = json.dumps(payload, separators=(",", ":"))
    matrix_text = json.dumps(transform.tolist(), indent=2)
    size_text = json.dumps(size.tolist())
    metadata_text = json.dumps(f2g_metadata, ensure_ascii=False, indent=2)
    diagnostics_text = json.dumps(
        dict(segmentation.diagnostics),
        ensure_ascii=False,
        indent=2,
    )
    image_height, image_width = rgb.shape[:2]
    dot_left = 100.0 * point_xy[0] / image_width
    dot_top = 100.0 * point_xy[1] / image_height
    image_data_url = _png_data_url(rgb)

    # All variable strings below are produced from JSON or base64 data owned by
    # this diagnostic.  The page is self-contained and does not load a CDN.
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>F2G interactive pose viewer</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #080b10; color: #edf2f7; }}
    header {{ padding: 14px 18px; border-bottom: 1px solid #293241; background: #0e141d; }}
    h1 {{ margin: 0; font-size: 18px; font-weight: 650; }}
    .subtitle {{ margin-top: 5px; color: #9fb0c3; font-size: 13px; }}
    main {{ display: grid; grid-template-columns: minmax(280px, 38%) 1fr; min-height: calc(100vh - 67px); }}
    .left {{ padding: 16px; border-right: 1px solid #293241; overflow: auto; }}
    .image-wrap {{ position: relative; width: 100%; border: 1px solid #344154; background: #111827; }}
    .image-wrap img {{ display: block; width: 100%; height: auto; image-rendering: auto; }}
    .image-point {{ position: absolute; left: {dot_left:.6f}%; top: {dot_top:.6f}%; width: 12px; height: 12px; border-radius: 50%; background: #ffe600; border: 2px solid #161616; transform: translate(-50%, -50%); }}
    .cards {{ display: grid; gap: 10px; margin-top: 12px; }}
    .card {{ border: 1px solid #293241; background: #0e141d; border-radius: 8px; padding: 10px 12px; }}
    .label {{ color: #8fa4ba; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }}
    .value {{ margin-top: 4px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; white-space: pre-wrap; overflow-wrap: anywhere; color: #dce7f2; }}
    details {{ margin-top: 10px; border-top: 1px solid #293241; padding-top: 8px; }}
    summary {{ cursor: pointer; color: #b6c7d8; font-size: 12px; }}
    .viewer {{ position: relative; min-width: 0; min-height: 500px; overflow: hidden; }}
    canvas {{ display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }}
    canvas.dragging {{ cursor: grabbing; }}
    .hud {{ position: absolute; left: 16px; top: 14px; pointer-events: none; text-shadow: 0 1px 2px #000; }}
    .hud-title {{ color: #f4f7fb; font-size: 14px; font-weight: 650; }}
    .hud-line {{ margin-top: 4px; color: #a9bbcd; font: 12px ui-monospace, monospace; }}
    .legend {{ position: absolute; right: 16px; bottom: 14px; padding: 8px 10px; border: 1px solid #344154; border-radius: 7px; background: rgba(9, 13, 19, .82); font: 11px ui-monospace, monospace; pointer-events: none; }}
    .swatch {{ display: inline-block; width: 10px; height: 3px; margin-right: 5px; vertical-align: middle; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} .left {{ border-right: 0; border-bottom: 1px solid #293241; }} .viewer {{ min-height: 62vh; }} }}
  </style>
</head>
<body>
  <header>
    <h1>F2G interactive pose viewer</h1>
    <div class="subtitle">Observed masked RGB-D cloud with F2G pose, metric OBB, and object-frame axes</div>
  </header>
  <main>
    <section class="left">
      <div class="image-wrap">
        <img src="{image_data_url}" alt="F2G source RGB image">
        <div class="image-point" title="VLM point {point_xy}"></div>
      </div>
      <div class="cards">
        <div class="card"><div class="label">T_cam_obj</div><div class="value">{matrix_text}</div></div>
        <div class="card"><div class="label">Extent XYZ, metres</div><div class="value">{size_text}</div></div>
        <div class="card"><div class="label">Point clouds</div><div class="value">{len(points)} displayed / {total_masked_points} valid masked depth points · {len(completed_points)} F2G completed points</div></div>
        <div class="card"><div class="label">Simulator ground truth</div><div class="value">{ground_truth_text}</div></div>
      </div>
      <details><summary>Segmentation diagnostics</summary><div class="value">{diagnostics_text}</div></details>
      <details><summary>F2G metadata</summary><div class="value">{metadata_text}</div></details>
    </section>
    <section class="viewer" id="viewer">
      <canvas id="canvas"></canvas>
      <div class="hud">
        <div class="hud-title">F2G pose | camera frame</div>
        <div class="hud-line">drag: rotate · wheel: zoom · r: reset</div>
        <div class="hud-line" id="angles"></div>
      </div>
      <div class="legend">
        <div><span class="swatch" style="background:#9aa0a6"></span>F2G completed cloud</div>
        <div><span class="swatch" style="background:#ffd43b"></span>F2G OBB</div>
        <div><span class="swatch" style="background:#48dbe8"></span>Simulator GT OBB</div>
        <div><span class="swatch" style="background:#ff4b4b"></span>X axis</div>
        <div><span class="swatch" style="background:#42d66b"></span>Y axis</div>
        <div><span class="swatch" style="background:#4b91ff"></span>Z axis</div>
      </div>
    </section>
  </main>
  <script>
    "use strict";
    const data = {payload_json};
    const canvas = document.getElementById("canvas");
    const viewer = document.getElementById("viewer");
    const angles = document.getElementById("angles");
    const ctx = canvas.getContext("2d");
    const boxEdges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    const axisColors = ["#ff4b4b", "#42d66b", "#4b91ff"];
    let yaw = -0.65;
    let pitch = -0.45;
    let zoom = 1.0;
    let dragging = false;
    let previousX = 0;
    let previousY = 0;

    function resetView() {{ yaw = -0.65; pitch = -0.45; zoom = 1.0; draw(); }}
    function rotated(point) {{
      const x = point[0] - data.origin[0];
      const y = point[1] - data.origin[1];
      const z = point[2] - data.origin[2];
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x1 = cy * x + sy * z;
      const z1 = -sy * x + cy * z;
      return [x1, cp * y - sp * z1, sp * y + cp * z1];
    }}
    function sceneSpan() {{
      let maximum = 0.001;
      for (const point of data.points.concat(data.completedPoints, data.boxCorners, data.groundTruthBoxCorners, data.axes)) {{
        maximum = Math.max(maximum, Math.abs(point[0] - data.origin[0]), Math.abs(point[1] - data.origin[1]), Math.abs(point[2] - data.origin[2]));
      }}
      return maximum * 2.35;
    }}
    function drawLine(a, b, color, width) {{
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
    }}
    function draw() {{
      const dpr = window.devicePixelRatio || 1;
      const width = viewer.clientWidth;
      const height = viewer.clientHeight;
      if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {{
        canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
      }}
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const gradient = ctx.createRadialGradient(width * .5, height * .46, 20, width * .5, height * .5, Math.max(width, height) * .72);
      gradient.addColorStop(0, "#17202d"); gradient.addColorStop(1, "#070a0f");
      ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
      const scale = Math.min(width, height) * .78 / sceneSpan() * zoom;
      const project = point => {{ const p = rotated(point); return [width / 2 + p[0] * scale, height / 2 - p[1] * scale, p[2]]; }};

      const projectedCompleted = data.completedPoints.map(project).sort((a, b) => a[2] - b[2]);
      for (const point of projectedCompleted) {{
        ctx.fillStyle = "#9aa0a6";
        ctx.beginPath(); ctx.arc(point[0], point[1], 1.6, 0, Math.PI * 2); ctx.fill();
      }}

      const projectedPoints = data.points.map((point, index) => {{ const p = project(point); return [p[0], p[1], p[2], index]; }});
      projectedPoints.sort((a, b) => a[2] - b[2]);
      for (const point of projectedPoints) {{
        const color = data.colors[point[3]];
        ctx.fillStyle = `rgb(${{Math.max(45, color[0])}},${{Math.max(35, color[1])}},${{Math.max(20, color[2])}})`;
        ctx.beginPath(); ctx.arc(point[0], point[1], 2.15, 0, Math.PI * 2); ctx.fill();
      }}

      const box = data.boxCorners.map(project);
      for (const edge of boxEdges) drawLine(box[edge[0]], box[edge[1]], "#ffd43b", 2.0);
      if (data.groundTruthBoxCorners.length === 8) {{
        const groundTruthBox = data.groundTruthBoxCorners.map(project);
        for (const edge of boxEdges) drawLine(groundTruthBox[edge[0]], groundTruthBox[edge[1]], "#48dbe8", 2.4);
      }}
      const projectedAxes = data.axes.map(project);
      for (let index = 0; index < 3; index += 1) drawLine(projectedAxes[0], projectedAxes[index + 1], axisColors[index], 3.0);
      ctx.fillStyle = "#ffffff"; ctx.beginPath(); ctx.arc(projectedAxes[0][0], projectedAxes[0][1], 3.2, 0, Math.PI * 2); ctx.fill();
      angles.textContent = `yaw ${{(yaw * 180 / Math.PI).toFixed(1)}}° · pitch ${{(pitch * 180 / Math.PI).toFixed(1)}}° · zoom ${{zoom.toFixed(2)}}×`;
    }}
    canvas.addEventListener("pointerdown", event => {{ dragging = true; previousX = event.clientX; previousY = event.clientY; canvas.classList.add("dragging"); canvas.setPointerCapture(event.pointerId); }});
    canvas.addEventListener("pointermove", event => {{ if (!dragging) return; yaw += (event.clientX - previousX) * 0.008; pitch = Math.max(-1.5, Math.min(1.5, pitch + (event.clientY - previousY) * 0.008)); previousX = event.clientX; previousY = event.clientY; draw(); }});
    canvas.addEventListener("pointerup", event => {{ dragging = false; canvas.classList.remove("dragging"); canvas.releasePointerCapture(event.pointerId); }});
    canvas.addEventListener("pointercancel", () => {{ dragging = false; canvas.classList.remove("dragging"); }});
    canvas.addEventListener("wheel", event => {{ event.preventDefault(); zoom = Math.max(.2, Math.min(8, zoom * Math.exp(-event.deltaY * .001))); draw(); }}, {{ passive: false }});
    window.addEventListener("keydown", event => {{ if (event.key.toLowerCase() === "r") resetView(); }});
    new ResizeObserver(draw).observe(viewer);
    draw();
  </script>
</body>
</html>
"""


def run_manual_grounding_pose_test() -> None:
    import cv2
    import numpy as np

    from harness.grounding.visualization import draw_grounding_overlay
    from harness.grounding.visualization import draw_mask_overlay

    rgb, depth, intrinsic, depth_scale, world_to_camera = (
        _load_rgb_and_grounding_input()
    )
    frame = CameraFrame(
        rgb=rgb,
        depth=depth,
        intrinsic=intrinsic,
        world_to_camera=world_to_camera,
        depth_scale=depth_scale,
        image_ref="manual_grounding_pose_front",
    )
    segmentation = _segment_object(frame)
    points_camera, colors_rgb, total_points = _masked_camera_cloud(
        frame,
        segmentation.mask,
        max_points=MAX_DISPLAY_POINTS,
    )
    client = F2GClient(base_url=F2G_URL)
    health = client.health()
    prediction = client.predict(frame, segmentation.mask)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    viewer_path = (OUTPUT_DIR / "pose_viewer.html").resolve()
    overlay_path = (OUTPUT_DIR / "grounding_overlay.png").resolve()
    masked_rgb = draw_mask_overlay(rgb, [segmentation])
    overlay_rgb = draw_grounding_overlay(
        masked_rgb,
        intrinsic=intrinsic,
        predictions=[
            (
                prediction.object_to_camera,
                prediction.size_m,
                segmentation.point_xy,
            )
        ],
    )
    if not cv2.imwrite(
        str(overlay_path),
        cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
    ):
        raise RuntimeError(f"failed to write grounding overlay: {overlay_path}")
    viewer_path.write_text(
        _viewer_html(
            rgb=rgb,
            point_xy=(int(POINT_XY[0]), int(POINT_XY[1])),
            points_camera=points_camera,
            colors_rgb=colors_rgb,
            total_masked_points=total_points,
            object_to_camera=prediction.object_to_camera,
            size_m=prediction.size_m,
            segmentation=segmentation,
            f2g_metadata=dict(prediction.metadata),
            completed_points_camera=prediction.points_camera,
        ),
        encoding="utf-8",
    )
    viewer_url_base = os.environ.get("POSE_VIEWER_URL_BASE", "").rstrip("/")
    viewer_url = (
        f"{viewer_url_base}/{quote(viewer_path.name)}"
        if viewer_url_base
        else None
    )
    print(
        json.dumps(
            {
                "f2g_health": health,
                "T_cam_obj": prediction.object_to_camera.tolist(),
                "size_m": prediction.size_m.tolist(),
                "f2g_metadata": dict(prediction.metadata),
                "point_cloud": {
                    "source": "masked_rgbd_observation",
                    "valid_points": total_points,
                    "displayed_points": int(points_camera.shape[0]),
                },
                "segmentation": {
                    "confidence": segmentation.confidence,
                    "diagnostics": dict(segmentation.diagnostics),
                },
                "artifacts": {
                    "pose_viewer": str(viewer_path),
                    "pose_viewer_url": viewer_url,
                    "grounding_overlay": str(overlay_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    assert np.isfinite(prediction.object_to_camera).all()
    assert viewer_path.is_file()
    assert overlay_path.is_file()


def test_viewer_embeds_completed_f2g_points_in_gray() -> None:
    from types import SimpleNamespace

    import numpy as np

    transform = np.eye(4)
    transform[2, 3] = 1.0
    html = _viewer_html(
        rgb=np.zeros((20, 20, 3), dtype=np.uint8),
        point_xy=(10, 10),
        points_camera=np.asarray([[0.0, 0.0, 1.0]]),
        colors_rgb=np.asarray([[255, 0, 0]], dtype=np.uint8),
        total_masked_points=1,
        object_to_camera=transform,
        size_m=np.asarray([0.1, 0.1, 0.1]),
        segmentation=SimpleNamespace(diagnostics={}),
        f2g_metadata={},
        completed_points_camera=np.asarray(
            [[0.0, 0.0, 1.0], [0.01, 0.0, 1.0]]
        ),
    )

    assert '"completedPoints":[[0.0,0.0,1.0],[0.01,0.0,1.0]]' in html
    assert "2 F2G completed points" in html
    assert "#9aa0a6" in html


def test_manual_grounding_pose() -> None:
    """Optional pytest entry point; direct Python execution needs no pytest."""

    import pytest

    if os.environ.get("RUN_AGENTIC_GROUNDING_POSE_TEST") != "1":
        pytest.skip("set RUN_AGENTIC_GROUNDING_POSE_TEST=1 to call F2G")
    run_manual_grounding_pose_test()


if __name__ == "__main__":
    run_manual_grounding_pose_test()
