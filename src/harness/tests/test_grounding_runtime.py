from __future__ import annotations

import numpy as np
import pytest

from harness.grounding.observation import camera_frame_from_robomme
from harness.grounding.observation import depth_statistics
from harness.grounding.transforms import object_to_world
from harness.grounding.transforms import rotation_matrix_to_quaternion_wxyz
from harness.grounding.types import CameraFrame


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


def test_target_profile_can_use_trusted_purple_color() -> None:
    pytest.importorskip("cv2")
    from harness.grounding.opencv_segmenter import OpenCVSegmenter

    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    rgb[12:28, 12:28] = [145, 65, 190]
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
