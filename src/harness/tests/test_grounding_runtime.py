from __future__ import annotations

import io
import json

import numpy as np
import pytest

from harness.grounding.f2g_client import F2GClient
from harness.grounding.grounder import GroundingPipeline
from harness.grounding.observation import camera_frame_from_maniskill
from harness.grounding.observation import camera_frame_from_robomme
from harness.grounding.observation import depth_statistics
from harness.grounding.observation import remap_grounding_points
from harness.grounding.transforms import object_to_world
from harness.grounding.transforms import rotation_matrix_to_quaternion_wxyz
from harness.grounding.types import CameraFrame
from harness.grounding.types import SegmentationResult


def _camera_inputs() -> tuple[dict, dict]:
    obs = {
        "front_rgb_list": np.zeros((12, 16, 3), dtype=np.uint8),
        "front_depth_list": np.full((12, 16), 700, dtype=np.int16),
        "front_camera_extrinsic_list": np.column_stack(
            (np.eye(3), np.asarray([0.1, 0.2, 0.3]))
        ),
    }
    info = {
        "front_camera_intrinsic": np.asarray(
            [[100, 0, 8], [0, 100, 6], [0, 0, 1]], dtype=np.float32
        )
    }
    return obs, info


def test_robomme_adapter_preserves_aligned_raw_depth() -> None:
    obs, info = _camera_inputs()
    frame = camera_frame_from_robomme(obs, info, depth_scale=0.001)

    assert frame.rgb.shape == (12, 16, 3)
    assert frame.depth.shape == (12, 16)
    assert frame.world_to_camera.shape == (4, 4)
    assert depth_statistics(frame)["median_m"] == pytest.approx(0.7)


def test_adapter_rejects_signed_depth_overflow() -> None:
    obs, info = _camera_inputs()
    obs["front_depth_list"][0, 0] = -1

    with pytest.raises(ValueError, match="negative integers"):
        camera_frame_from_robomme(obs, info, depth_scale=0.001)


def test_maniskill_adapter_selects_named_grounding_camera() -> None:
    rgb = np.full((1, 24, 32, 3), 127, dtype=np.uint8)
    depth = np.full((1, 24, 32, 1), 800, dtype=np.int16)
    intrinsic = np.asarray(
        [[[200, 0, 16], [0, 200, 12], [0, 0, 1]]], dtype=np.float32
    )
    extrinsic = np.asarray(
        [[[1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 1, 0.3]]],
        dtype=np.float32,
    )
    raw_obs = {
        "sensor_data": {
            "grounding_camera": {"rgb": rgb, "depth": depth},
        },
        "sensor_param": {
            "grounding_camera": {
                "intrinsic_cv": intrinsic,
                "extrinsic_cv": extrinsic,
            },
        },
    }

    frame = camera_frame_from_maniskill(
        raw_obs,
        camera_name="grounding_camera",
        depth_scale=0.001,
    )

    assert frame.rgb.shape == (24, 32, 3)
    assert frame.depth.shape == (24, 32)
    assert depth_statistics(frame)["median_m"] == pytest.approx(0.8)
    np.testing.assert_allclose(frame.intrinsic, intrinsic[0])
    np.testing.assert_allclose(frame.world_to_camera[:3], extrinsic[0])


def test_grounding_points_are_remapped_to_high_resolution_camera() -> None:
    grounding = {
        "status": "ok",
        "image_ref": "current_front",
        "targets": {
            "object": {"point_xy": [128.0, 64.0], "color_hint": "red"},
        },
    }

    remapped = remap_grounding_points(
        grounding,
        source_shape=(256, 256),
        target_shape=(768, 768),
    )

    assert remapped["targets"]["object"]["point_xy"] == pytest.approx(
        [385.0, 193.0]
    )
    assert grounding["targets"]["object"]["point_xy"] == [128.0, 64.0]


def test_opencv_spatial_thresholds_scale_for_grounding_camera() -> None:
    from harness.grounding.opencv_segmenter import DEFAULT_PROFILES
    from harness.grounding.opencv_segmenter import _image_pixel_scale
    from harness.grounding.opencv_segmenter import _scale_profile

    pixel_scale = _image_pixel_scale((768, 768))
    button = _scale_profile(
        DEFAULT_PROFILES["button"],
        pixel_scale=pixel_scale,
    )
    container = _scale_profile(
        DEFAULT_PROFILES["container"],
        pixel_scale=pixel_scale,
    )

    assert pixel_scale == pytest.approx(3.0)
    assert button.min_area == 450
    assert button.depth_search_radius_px == 195
    assert button.morphology_px == 9
    assert container.min_area == 900
    assert container.depth_search_radius_px == 330
    assert container.morphology_px == 21
    assert button.depth_tolerance_m == DEFAULT_PROFILES["button"].depth_tolerance_m
    assert button.max_area_fraction == DEFAULT_PROFILES["button"].max_area_fraction


def test_opencv_spatial_thresholds_do_not_shrink_below_reference_resolution() -> None:
    from harness.grounding.opencv_segmenter import DEFAULT_PROFILES
    from harness.grounding.opencv_segmenter import _image_pixel_scale
    from harness.grounding.opencv_segmenter import _scale_profile

    pixel_scale = _image_pixel_scale((40, 40))
    actual = _scale_profile(
        DEFAULT_PROFILES["compact_colored"],
        pixel_scale=pixel_scale,
    )

    assert pixel_scale == pytest.approx(1.0)
    assert actual == DEFAULT_PROFILES["compact_colored"]


@pytest.mark.parametrize("input_size", [(256, 256), (512, 512), (768, 768)])
def test_f2g_client_resizes_only_serialized_inputs(
    monkeypatch: pytest.MonkeyPatch,
    input_size: tuple[int, int],
) -> None:
    obs, info = _camera_inputs()
    frame = camera_frame_from_robomme(obs, info, depth_scale=0.001)
    mask = np.zeros((12, 16), dtype=bool)
    mask[2:10, 4:12] = True
    captured: dict[str, np.ndarray] = {}

    def fake_request(_client: F2GClient, request) -> bytes:
        with np.load(io.BytesIO(request.data)) as payload:
            captured.update({key: payload[key] for key in payload.files})
        return json.dumps(
            {
                "T_cam_obj": np.eye(4).tolist(),
                "size_m": [0.1, 0.1, 0.1],
                "points_cam_m": [[0.0, 0.0, 0.7], [0.01, 0.0, 0.7]],
            }
        ).encode()

    monkeypatch.setattr(F2GClient, "_request", fake_request)
    prediction = F2GClient(input_size=input_size).predict(frame, mask)

    target_height, target_width = input_size
    assert captured["rgb"].shape == (target_height, target_width, 3)
    assert captured["depth"].shape == input_size
    assert captured["mask"].shape == input_size
    np.testing.assert_allclose(
        captured["intrinsic"],
        [
            [100 * target_width / 16, 0, 8 * target_width / 16],
            [0, 100 * target_height / 12, 6 * target_height / 12],
            [0, 0, 1],
        ],
    )
    assert frame.rgb.shape == (12, 16, 3)
    np.testing.assert_allclose(
        prediction.points_camera,
        [[0.0, 0.0, 0.7], [0.01, 0.0, 0.7]],
    )
    assert "points_cam_m" not in prediction.metadata


def test_object_to_world_uses_world_to_camera_extrinsic() -> None:
    world_to_camera = np.eye(4)
    world_to_camera[:3, 3] = [1, 0, 0]
    object_to_camera = np.eye(4)
    object_to_camera[:3, 3] = [1.2, 0.3, 0.4]

    world_object = object_to_world(world_to_camera, object_to_camera)

    assert world_object[:3, 3] == pytest.approx([0.2, 0.3, 0.4])
    assert rotation_matrix_to_quaternion_wxyz(world_object[:3, :3]) == pytest.approx(
        [1, 0, 0, 0]
    )


@pytest.mark.parametrize(
    "action_name",
    ["solve_put_into_bin", "solve_button", "solve_button_ready"],
)
def test_env_target_actions_require_no_grounding_backend(action_name: str) -> None:
    class ForbiddenSegmenter:
        def segment(self, *args, **kwargs):
            raise AssertionError("location_grounding must not call OpenCV")

    class ForbiddenF2G:
        def predict(self, *args, **kwargs):
            raise AssertionError("location_grounding must not call F2G")

    frame = CameraFrame(
        rgb=np.zeros((5, 5, 3), dtype=np.uint8),
        depth=np.full((5, 5), 1000, dtype=np.uint16),
        intrinsic=np.asarray([[100, 0, 2], [0, 100, 2], [0, 0, 1]]),
        world_to_camera=np.eye(4),
        depth_scale=0.001,
    )
    grounded = GroundingPipeline(
        segmenter=ForbiddenSegmenter(),
        f2g_client=ForbiddenF2G(),
    ).build_action_request(
        action_name=action_name,
        grounding={
            "status": "not_required",
            "image_ref": None,
            "targets": {},
        },
        frame=frame,
    )

    assert grounded.masks == {}
    assert grounded.request.targets == {}


def _synthetic_frame(rgb: np.ndarray, depth_mm: np.ndarray) -> CameraFrame:
    return CameraFrame(
        rgb=rgb,
        depth=depth_mm,
        intrinsic=np.asarray([[100, 0, 20], [0, 100, 20], [0, 0, 1]]),
        world_to_camera=np.eye(4),
        depth_scale=0.001,
    )


def test_colored_profile_selects_nearest_matching_component() -> None:
    pytest.importorskip("cv2")
    from harness.grounding.opencv_segmenter import OpenCVSegmenter

    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    rgb[5:15, 6:16] = [220, 45, 45]
    rgb[25:35, 25:35] = [220, 45, 45]
    depth = np.full((40, 40), 700, dtype=np.int16)

    result = OpenCVSegmenter().segment(
        _synthetic_frame(rgb, depth),
        point_xy=(10, 10),
        color_hint="red",
        profile_name="compact_colored",
    )

    assert result.mask[10, 10]
    assert not result.mask[30, 30]


def test_colored_profile_keeps_shaded_face_and_has_no_distance_rejection() -> None:
    pytest.importorskip("cv2")
    from harness.grounding.opencv_segmenter import OpenCVSegmenter

    rgb = np.full((120, 120, 3), [80, 70, 60], dtype=np.uint8)
    rgb[8:18, 8:18] = [220, 45, 45]
    rgb[18:28, 8:18] = [100, 18, 18]
    depth = np.full((120, 120), 700, dtype=np.int16)

    result = OpenCVSegmenter().segment(
        _synthetic_frame(rgb, depth),
        point_xy=(100, 100),
        color_hint="red",
        profile_name="compact_colored",
    )

    assert result.mask[12, 12]
    assert result.mask[22, 12]
    assert result.diagnostics["point_distance_px"] > 70
    assert result.diagnostics["mask_rule"] == "circular_hue_saturation_value"


def test_shared_mask_overlay_paints_mask_black_and_point_yellow() -> None:
    pytest.importorskip("cv2")
    from harness.grounding.visualization import draw_mask_overlay

    rgb = np.full((20, 20, 3), 180, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    segmentation = SegmentationResult(
        mask=mask,
        confidence=1.0,
        profile="compact_colored",
        point_xy=(10, 10),
    )

    overlay = draw_mask_overlay(rgb, [segmentation])

    assert overlay[6, 6].tolist() == [0, 0, 0]
    assert overlay[10, 10].tolist() == [255, 255, 0]


def test_shared_grounding_overlay_draws_f2g_box_and_point() -> None:
    pytest.importorskip("cv2")
    from harness.grounding.visualization import draw_grounding_overlay

    rgb = np.full((100, 100, 3), [180, 160, 140], dtype=np.uint8)
    intrinsic = np.asarray([[100, 0, 50], [0, 100, 50], [0, 0, 1]])
    object_to_camera = np.eye(4)
    object_to_camera[2, 3] = 1.0

    overlay = draw_grounding_overlay(
        rgb,
        intrinsic=intrinsic,
        predictions=[
            (object_to_camera, np.asarray([0.2, 0.2, 0.2]), (20, 20))
        ],
    )

    assert overlay[20, 20].tolist() == [255, 255, 0]
    assert np.any(np.all(overlay == [0, 255, 0], axis=2))
    # The F2G overlay is drawn directly on the camera RGB. In particular, it
    # must not inherit the black region used by the separate OpenCV mask view.
    assert overlay[45, 45].tolist() == rgb[45, 45].tolist()


def test_target_profile_can_use_trusted_purple_color() -> None:
    pytest.importorskip("cv2")
    from harness.grounding.opencv_segmenter import OpenCVSegmenter

    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    rgb[15:25, 15:25] = [145, 65, 190]
    depth = np.full((40, 40), 700, dtype=np.int16)

    result = OpenCVSegmenter().segment(
        _synthetic_frame(rgb, depth),
        point_xy=(20, 20),
        color_hint="purple",
        profile_name="target",
    )

    assert result.mask[20, 20]
    assert not result.mask[0, 0]


def test_position_profile_uses_nearby_depth_connected_region() -> None:
    pytest.importorskip("cv2")
    from harness.grounding.opencv_segmenter import OpenCVSegmenter

    rgb = np.full((40, 40, 3), 100, dtype=np.uint8)
    depth = np.full((40, 40), 1000, dtype=np.int16)
    depth[12:22, 14:24] = 700

    result = OpenCVSegmenter().segment(
        _synthetic_frame(rgb, depth),
        point_xy=(20, 18),
        color_hint=None,
        profile_name="button",
    )

    assert result.mask[18, 20]
    assert not result.mask[0, 0]
