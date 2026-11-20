"""BinFill reset -> simulator GT point -> OpenCV mask -> F2G diagnostic.

This test deliberately bypasses only the Agentic VLM point. It creates one raw
BinFill evaluation scene, calls only ``env.reset()``, picks
the most visible red cube from the simulator segmentation IDs, and uses the
center of its GT mask only as the input point. OpenCV then selects the nearest
red component in the grounding-camera RGB image, and only that OpenCV mask is
sent to F2G. The simulator GT mask and OBB are retained only for evaluation.

Run through the one-terminal launcher from the repository root::

    bash scripts/test_grounding_env_gt.sh

Set ``BIN_FILL_EPISODE_ID`` to reproduce a different test episode.  No robot
action, demonstration wrapper, or planner is invoked.
"""

from __future__ import annotations

from dataclasses import replace
import itertools
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
THIRD_PARTY_ROOT = REPO_ROOT / "third_party" / "robomme_benchmark" / "src"
TEST_DIR = Path(__file__).resolve().parent
for import_root in (SRC_ROOT, THIRD_PARTY_ROOT, TEST_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from harness.grounding import F2GClient
from harness.grounding import OpenCVSegmenter
from harness.grounding.types import CameraFrame
from harness.grounding.visualization import draw_grounding_overlay
from harness.grounding.visualization import draw_mask_overlay
from robomme.env_record_wrapper import BenchmarkEnvBuilder
import robomme.robomme_env  # noqa: F401  # register BinFill with Gymnasium
from robomme.robomme_env.BinFill import BinFill
import test_grounding_pos as pose_viewer


EPISODE_ID = int(os.environ.get("BIN_FILL_EPISODE_ID", "0"))
F2G_URL = os.environ.get("F2G_URL", "http://127.0.0.1:8765")
DEPTH_SCALE = float(os.environ.get("ROBOMME_DEPTH_SCALE", "0.001"))
GROUNDING_CAMERA_RESOLUTION = int(
    os.environ.get("GROUNDING_CAMERA_RESOLUTION", "512")
)
GROUNDING_CAMERA_NAME = "grounding_camera"
MAX_DISPLAY_POINTS = 4096
OUTPUT_DIR = Path(
    os.environ.get(
        "GROUNDING_OUTPUT_DIR",
        str(REPO_ROOT / "runs" / "tests" / "grounding"),
    )
).expanduser()


class _BinFillWithGroundingCamera(BinFill):
    """Keep the policy camera unchanged and add a native grounding camera."""

    @property
    def _default_sensor_configs(self):
        base_configs = list(super()._default_sensor_configs)
        if len(base_configs) != 1 or base_configs[0].uid != "base_camera":
            raise RuntimeError(
                "BinFill camera configuration changed; expected one base_camera"
            )
        grounding_camera = replace(
            base_configs[0],
            uid=GROUNDING_CAMERA_NAME,
            width=GROUNDING_CAMERA_RESOLUTION,
            height=GROUNDING_CAMERA_RESOLUTION,
        )
        return [*base_configs, grounding_camera]


def _to_numpy(value: Any) -> Any:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _image(value: Any, *, channels: int | None) -> Any:
    array = _to_numpy(value)
    while array.ndim > (3 if channels is not None else 2) and array.shape[0] == 1:
        array = array[0]
    if channels is not None:
        if array.ndim != 3 or array.shape[-1] != channels:
            raise ValueError(f"expected HxWx{channels} image, got {array.shape}")
    else:
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError(f"expected HxW image, got {array.shape}")
    return array


def _matrix(value: Any, rows: int, columns: int) -> Any:
    import numpy as np

    array = _to_numpy(value)
    if array.size < rows * columns:
        raise ValueError(f"camera matrix has fewer than {rows * columns} values")
    return np.asarray(array, dtype=np.float64).reshape(-1, rows, columns)[-1]


def _raw_camera_frame(
    observation: dict[str, Any],
    *,
    camera_name: str = GROUNDING_CAMERA_NAME,
) -> tuple[CameraFrame, Any]:
    import numpy as np

    sensor_data = observation["sensor_data"][camera_name]
    sensor_param = observation["sensor_param"][camera_name]
    rgb = _image(sensor_data["rgb"], channels=3)
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating) and float(np.nanmax(rgb)) <= 1.0:
            rgb = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    depth = _image(sensor_data["depth"], channels=None)
    segmentation = _image(sensor_data["segmentation"], channels=None)
    intrinsic = _matrix(sensor_param["intrinsic_cv"], 3, 3)
    extrinsic = _matrix(sensor_param["extrinsic_cv"], 3, 4)
    return (
        CameraFrame(
            rgb=np.ascontiguousarray(rgb),
            depth=np.ascontiguousarray(depth),
            intrinsic=intrinsic,
            world_to_camera=extrinsic,
            depth_scale=DEPTH_SCALE,
            image_ref=(
                f"binfill_test_ep_{EPISODE_ID}_{camera_name}_"
                f"{rgb.shape[1]}x{rgb.shape[0]}_reset"
            ),
        ),
        np.asarray(segmentation),
    )


def _actor_segmentation_ids(base_env: Any, actor: Any) -> list[int]:
    mapping = getattr(base_env, "segmentation_id_map", None)
    if not isinstance(mapping, dict):
        raise RuntimeError("BinFill did not expose segmentation_id_map")
    actor_name = getattr(actor, "name", None)
    ids = []
    for segmentation_id, mapped_actor in mapping.items():
        if mapped_actor is actor or (
            actor_name is not None
            and getattr(mapped_actor, "name", None) == actor_name
        ):
            ids.append(int(segmentation_id))
    return sorted(ids)


def _most_visible_red_cube(base_env: Any, segmentation: Any) -> tuple[Any, list[int], Any]:
    import numpy as np

    red_cubes = list(getattr(base_env, "red_cubes", []))
    if not red_cubes:
        raise RuntimeError(
            f"BinFill test episode {EPISODE_ID} contains no red cube; set "
            "BIN_FILL_EPISODE_ID to a red-cube episode"
        )
    candidates = []
    for actor in red_cubes:
        ids = _actor_segmentation_ids(base_env, actor)
        mask = np.isin(segmentation, ids)
        candidates.append((int(np.count_nonzero(mask)), actor, ids, mask))
    visible_pixels, actor, ids, mask = max(candidates, key=lambda item: item[0])
    if visible_pixels < 50:
        counts = {
            str(getattr(candidate[1], "name", "unnamed")): candidate[0]
            for candidate in candidates
        }
        raise RuntimeError(
            "no red cube has the 50 visible pixels required by F2G; "
            f"per-cube counts: {counts}"
        )
    return actor, ids, mask


def _actor_world_transform(actor: Any) -> Any:
    import numpy as np

    pose = actor.pose
    if not hasattr(pose, "to_transformation_matrix"):
        raise RuntimeError("selected actor pose has no transformation matrix")
    transform = _to_numpy(pose.to_transformation_matrix())
    transform = np.asarray(transform, dtype=np.float64).reshape(-1, 4, 4)[-1]
    return transform


def _cube_size_m(base_env: Any) -> Any:
    import numpy as np

    half_size = np.asarray(base_env.cube_half_size, dtype=np.float64).reshape(-1)
    if half_size.size == 1:
        half_size = np.repeat(half_size, 3)
    if half_size.size != 3:
        raise ValueError(f"unexpected cube_half_size: {base_env.cube_half_size!r}")
    return half_size * 2.0


def _cube_symmetry_rotation_error_deg(predicted: Any, ground_truth: Any) -> float:
    import numpy as np

    symmetries = []
    for permutation in itertools.permutations(range(3)):
        permutation_matrix = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            symmetry = permutation_matrix @ np.diag(signs)
            if np.linalg.det(symmetry) > 0.5:
                symmetries.append(symmetry)
    errors = []
    for symmetry in symmetries:
        relative = predicted.T @ (ground_truth @ symmetry)
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        errors.append(float(np.degrees(np.arccos(cosine))))
    return min(errors)


def _point_cloud_gt_consistency(
    points_camera: Any,
    ground_truth_camera: Any,
    ground_truth_size: Any,
    *,
    tolerance_m: float = 0.005,
) -> dict[str, float | int]:
    """Measure whether GT-masked depth points lie inside the simulator cube."""

    import numpy as np

    points = np.asarray(points_camera, dtype=np.float64)
    transform = np.asarray(ground_truth_camera, dtype=np.float64)
    half_size = np.asarray(ground_truth_size, dtype=np.float64) / 2.0
    local_points = (
        transform[:3, :3].T @ (points - transform[:3, 3]).T
    ).T
    outside_excess = np.maximum(np.abs(local_points) - half_size, 0.0)
    maximum_excess = np.max(outside_excess, axis=1)
    return {
        "checked_points": int(points.shape[0]),
        "tolerance_m": float(tolerance_m),
        "inside_fraction": float(np.mean(maximum_excess <= tolerance_m)),
        "mean_outside_excess_m": float(np.mean(maximum_excess)),
        "max_outside_excess_m": float(np.max(maximum_excess)),
    }


def run_environment_ground_truth_test() -> None:
    import cv2
    import numpy as np

    if EPISODE_ID < 0:
        raise ValueError("BIN_FILL_EPISODE_ID must be non-negative")
    if GROUNDING_CAMERA_RESOLUTION <= 0:
        raise ValueError("GROUNDING_CAMERA_RESOLUTION must be positive")
    builder = BenchmarkEnvBuilder(
        env_id="BinFill",
        dataset="test",
        action_space="joint_angle",
        gui_render=False,
    )
    seed, difficulty = builder.resolve_episode(EPISODE_ID)
    env_kwargs: dict[str, Any] = {
        "obs_mode": "rgb+depth+segmentation",
        "control_mode": "pd_joint_pos",
        "render_mode": "rgb_array",
        "reward_mode": "dense",
    }
    if seed is not None:
        env_kwargs["seed"] = seed
    if difficulty:
        env_kwargs["difficulty"] = difficulty

    env = _BinFillWithGroundingCamera(**env_kwargs)
    try:
        # This is intentionally the only environment transition in this test.
        observation, _ = env.reset()
        base_env = env.unwrapped
        frame, raw_segmentation = _raw_camera_frame(observation)
        base_camera_rgb = _image(
            observation["sensor_data"]["base_camera"]["rgb"],
            channels=3,
        )
        actor, segmentation_ids, simulator_gt_mask = _most_visible_red_cube(
            base_env,
            raw_segmentation,
        )
        pixel_y, pixel_x = np.nonzero(simulator_gt_mask)
        point_xy = (
            int(np.rint(np.mean(pixel_x))),
            int(np.rint(np.mean(pixel_y))),
        )
        segmentation = OpenCVSegmenter().segment(
            frame,
            point_xy=point_xy,
            color_hint="red",
            profile_name="compact_colored",
        )
        segmentation = replace(
            segmentation,
            diagnostics={
                **dict(segmentation.diagnostics),
                "source": f"BinFill {GROUNDING_CAMERA_NAME} RGB via OpenCV",
                "input_point_source": "simulator GT mask centroid",
                "actor_name": str(getattr(actor, "name", "unnamed")),
                "segmentation_ids": segmentation_ids,
                "simulator_gt_mask_pixels": int(
                    np.count_nonzero(simulator_gt_mask)
                ),
                "bypassed": ["VLM point"],
                "environment_transitions": ["reset"],
                "robot_actions": 0,
            },
        )
        mask = segmentation.mask
        mask_intersection = int(np.count_nonzero(mask & simulator_gt_mask))
        mask_union = int(np.count_nonzero(mask | simulator_gt_mask))
        opencv_mask_comparison = {
            "opencv_mask_pixels": int(np.count_nonzero(mask)),
            "simulator_gt_mask_pixels": int(
                np.count_nonzero(simulator_gt_mask)
            ),
            "intersection_pixels": mask_intersection,
            "iou": float(mask_intersection / mask_union) if mask_union else 1.0,
        }
        points_camera, colors_rgb, total_points = pose_viewer._masked_camera_cloud(
            frame,
            mask,
            max_points=MAX_DISPLAY_POINTS,
        )

        ground_truth_world = _actor_world_transform(actor)
        ground_truth_camera = frame.world_to_camera @ ground_truth_world
        ground_truth_size = _cube_size_m(base_env)
        point_cloud_consistency = _point_cloud_gt_consistency(
            points_camera,
            ground_truth_camera,
            ground_truth_size,
        )

        client = F2GClient(
            base_url=F2G_URL,
            input_size=(GROUNDING_CAMERA_RESOLUTION, GROUNDING_CAMERA_RESOLUTION),
        )
        health = client.health()
        prediction = client.predict(frame, mask)
        if prediction.points_camera is None:
            raise RuntimeError(
                "F2G response did not include its completed point cloud"
            )

        predicted_transform = np.asarray(
            prediction.object_to_camera,
            dtype=np.float64,
        )
        predicted_size = np.asarray(prediction.size_m, dtype=np.float64)
        center_error_m = float(
            np.linalg.norm(
                predicted_transform[:3, 3] - ground_truth_camera[:3, 3]
            )
        )
        size_abs_error_m = np.abs(predicted_size - ground_truth_size)
        rotation_error_deg = _cube_symmetry_rotation_error_deg(
            predicted_transform[:3, :3],
            ground_truth_camera[:3, :3],
        )
        if point_cloud_consistency["inside_fraction"] < 0.9:
            diagnosis_hint = (
                "OpenCV-masked depth cloud is inconsistent with the simulator "
                "GT box; inspect the OpenCV mask or RGB-depth alignment"
            )
        elif (
            center_error_m > 0.02
            or float(np.max(size_abs_error_m)) > 0.02
            or rotation_error_deg > 20.0
        ):
            diagnosis_hint = (
                "OpenCV-masked depth cloud agrees with the simulator, but F2G has a "
                "large pose/size error; the remaining failure is downstream of "
                "mask acquisition"
            )
        else:
            diagnosis_hint = (
                "both the GT-masked depth cloud and F2G are within the diagnostic "
                "thresholds in this scene"
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"binfill_ep_{EPISODE_ID}_red_opencv_mask"
        viewer_path = (OUTPUT_DIR / f"{stem}_pose_viewer.html").resolve()
        overlay_path = (OUTPUT_DIR / f"{stem}_grounding_overlay.png").resolve()
        input_path = (OUTPUT_DIR / f"{stem}_input.npz").resolve()
        masked_rgb = draw_mask_overlay(frame.rgb, [segmentation])
        overlay_rgb = draw_grounding_overlay(
            masked_rgb,
            intrinsic=frame.intrinsic,
            predictions=[
                (
                    predicted_transform,
                    predicted_size,
                    point_xy,
                )
            ],
        )
        if not cv2.imwrite(
            str(overlay_path),
            cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
        ):
            raise RuntimeError(f"failed to write grounding overlay: {overlay_path}")
        viewer_path.write_text(
            pose_viewer._viewer_html(
                rgb=overlay_rgb,
                point_xy=point_xy,
                points_camera=points_camera,
                colors_rgb=colors_rgb,
                total_masked_points=total_points,
                object_to_camera=predicted_transform,
                size_m=predicted_size,
                segmentation=segmentation,
                f2g_metadata=dict(prediction.metadata),
                completed_points_camera=prediction.points_camera,
                ground_truth_object_to_camera=ground_truth_camera,
                ground_truth_size_m=ground_truth_size,
            ),
            encoding="utf-8",
        )
        np.savez_compressed(
            input_path,
            rgb=frame.rgb,
            depth=frame.depth,
            intrinsic=frame.intrinsic,
            world_to_camera=frame.world_to_camera,
            depth_scale=np.asarray(frame.depth_scale),
            opencv_mask=mask,
            simulator_gt_mask=simulator_gt_mask,
            simulator_T_cam_obj=ground_truth_camera,
            simulator_size_m=ground_truth_size,
            f2g_T_cam_obj=predicted_transform,
            f2g_size_m=predicted_size,
            f2g_completed_points_cam_m=prediction.points_camera,
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
                    "test_boundary": {
                        "scene": "BinFill raw eval reset",
                        "episode_id": EPISODE_ID,
                        "seed": seed,
                        "difficulty": difficulty,
                        "robot_actions": 0,
                        "input_point_source": "simulator GT mask centroid",
                        "mask_source": (
                            "OpenCV nearest red component from grounding_camera RGB"
                        ),
                        "cameras": {
                            "base_camera": {
                                "width": int(base_camera_rgb.shape[1]),
                                "height": int(base_camera_rgb.shape[0]),
                            },
                            GROUNDING_CAMERA_NAME: {
                                "width": int(frame.rgb.shape[1]),
                                "height": int(frame.rgb.shape[0]),
                            },
                        },
                        "bypassed": ["VLM point"],
                    },
                    "selected_actor": {
                        "name": str(getattr(actor, "name", "unnamed")),
                        "segmentation_ids": segmentation_ids,
                        "input_point_xy": list(point_xy),
                    },
                    "opencv_mask_vs_simulator_gt": opencv_mask_comparison,
                    "f2g_health": health,
                    "comparison": {
                        "center_error_m": center_error_m,
                        "size_abs_error_m": size_abs_error_m.tolist(),
                        "cube_symmetry_rotation_error_deg": rotation_error_deg,
                        "note": (
                            "cube rotation error is minimized over the 24 proper "
                            "cube symmetries"
                        ),
                    },
                    "point_cloud_vs_simulator_gt": point_cloud_consistency,
                    "diagnosis_hint": diagnosis_hint,
                    "simulator": {
                        "T_cam_obj": ground_truth_camera.tolist(),
                        "size_m": ground_truth_size.tolist(),
                    },
                    "f2g": {
                        "T_cam_obj": predicted_transform.tolist(),
                        "size_m": predicted_size.tolist(),
                        "completed_point_count": int(
                            prediction.points_camera.shape[0]
                        ),
                        "metadata": dict(prediction.metadata),
                    },
                    "artifacts": {
                        "pose_viewer": str(viewer_path),
                        "pose_viewer_url": viewer_url,
                        "grounding_overlay": str(overlay_path),
                        "reproduction_npz": str(input_path),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        assert viewer_path.is_file()
        assert overlay_path.is_file()
        assert input_path.is_file()
    finally:
        env.close()


def test_environment_ground_truth_grounding() -> None:
    import pytest

    if os.environ.get("RUN_AGENTIC_GROUNDING_ENV_GT_TEST") != "1":
        pytest.skip("set RUN_AGENTIC_GROUNDING_ENV_GT_TEST=1 to run BinFill + F2G")
    run_environment_ground_truth_test()


if __name__ == "__main__":
    run_environment_ground_truth_test()
