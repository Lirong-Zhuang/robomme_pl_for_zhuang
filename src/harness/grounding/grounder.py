"""End-to-end VLM hint -> OpenCV mask -> F2G -> ActionRequest pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Mapping

from harness.actions.types import ActionRequest
from harness.actions.types import ObjectEstimate
from harness.actions.types import Pose3D
from harness.actions.types import Size3D

from .contracts import GroundingContractRegistry
from .f2g_client import F2GClient
from .opencv_segmenter import OpenCVSegmenter
from .transforms import object_to_world
from .transforms import rotation_matrix_to_quaternion_wxyz
from .types import CameraFrame
from .types import F2GPrediction
from .types import SegmentationResult


DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "action_candidates"
    / "action_catalog.json"
)


@dataclass(frozen=True)
class GroundingOutput:
    """Action-ready estimates plus masks retained for logging/debugging."""

    request: ActionRequest
    masks: Mapping[str, SegmentationResult]


class GroundingPipeline:
    def __init__(
        self,
        *,
        f2g_client: F2GClient | None = None,
        segmenter: OpenCVSegmenter | None = None,
        catalog_path: Path = DEFAULT_CATALOG_PATH,
    ) -> None:
        self.f2g_client = f2g_client or F2GClient()
        self.segmenter = segmenter or OpenCVSegmenter()
        self.contracts = GroundingContractRegistry.from_catalog(catalog_path)

    def _estimate(
        self,
        *,
        frame: CameraFrame,
        role: str,
        color_hint: str | None,
        segmentation: SegmentationResult,
        index: int | None = None,
    ) -> ObjectEstimate:
        prediction: F2GPrediction = self.f2g_client.predict(
            frame, segmentation.mask
        )
        world_transform = object_to_world(
            frame.world_to_camera, prediction.object_to_camera
        )
        suffix = f"_{index}" if index is not None else ""
        return ObjectEstimate(
            object_id=f"{frame.image_ref}:{role}{suffix}",
            label=color_hint or role,
            pose=Pose3D(
                position_xyz=tuple(float(item) for item in world_transform[:3, 3]),
                quaternion_wxyz=rotation_matrix_to_quaternion_wxyz(
                    world_transform[:3, :3]
                ),
                frame="world",
            ),
            size=Size3D(tuple(float(item) for item in prediction.size_m)),
            # F2G's public inference output has no calibrated confidence.  Keep
            # the OpenCV mask-selection confidence instead of inventing one.
            confidence=segmentation.confidence,
            metadata={
                "source": "opencv+f2g",
                "image_ref": frame.image_ref,
                "segmentation_profile": segmentation.profile,
                "segmentation": dict(segmentation.diagnostics),
                "T_cam_obj": prediction.object_to_camera.tolist(),
                "f2g": dict(prediction.metadata),
            },
        )

    def build_action_request(
        self,
        *,
        action_name: str,
        grounding: Mapping[str, Any],
        frame: CameraFrame,
        arguments: Mapping[str, Any] | None = None,
    ) -> GroundingOutput:
        """Ground exactly the roles declared by the trusted action catalog."""

        contract = self.contracts.contract(action_name)
        normalized = self.contracts.validate(
            action_name,
            grounding,
            image_shape=frame.rgb.shape[:2],
        )
        if normalized["status"] == "ambiguous":
            raise ValueError("VLM grounding is ambiguous; do not execute the action")
        mode = contract["mode"]
        action_arguments = dict(arguments or {})
        if mode == "none":
            return GroundingOutput(
                request=ActionRequest(action_name, arguments=action_arguments),
                masks={},
            )
        if mode == "scene_target_set":
            target_spec = contract["targets"][0]
            role = target_spec["role"]
            segmentations = self.segmenter.segment_scene_targets(frame)
            indexed = [
                (
                    self._estimate(
                        frame=frame,
                        role=role,
                        color_hint=None,
                        segmentation=segmentation,
                        index=index,
                    ),
                    segmentation,
                )
                for index, segmentation in enumerate(segmentations)
            ]
            # PatternLock builds row-major in world x/y and RouteStick builds
            # indices from negative to positive world y. Sorting by world y
            # preserves RouteStick's index semantics after camera projection.
            indexed.sort(key=lambda pair: pair[0].pose.position_xyz[1])
            estimates = tuple(pair[0] for pair in indexed)
            return GroundingOutput(
                request=ActionRequest(
                    action_name,
                    target_sets={role: estimates},
                    arguments=action_arguments,
                ),
                masks={
                    f"{role}_{index}": pair[1]
                    for index, pair in enumerate(indexed)
                },
            )
        if normalized["image_ref"] != frame.image_ref:
            raise ValueError(
                f"VLM image_ref={normalized['image_ref']!r} does not match "
                f"camera frame {frame.image_ref!r}"
            )
        estimates: dict[str, ObjectEstimate] = {}
        masks: dict[str, SegmentationResult] = {}
        target_specs = {item["role"]: item for item in contract["targets"]}
        for role, hint in normalized["targets"].items():
            spec = target_specs[role]
            segmentation_color = (
                spec.get("fixed_color")
                if spec.get("color_hint") == "fixed"
                else hint["color_hint"]
            )
            segmentation = self.segmenter.segment(
                frame,
                point_xy=tuple(hint["point_xy"]),
                color_hint=segmentation_color,
                profile_name=spec["segmentation_profile"],
            )
            masks[role] = segmentation
            estimates[role] = self._estimate(
                frame=frame,
                role=role,
                color_hint=segmentation_color,
                segmentation=segmentation,
            )
        return GroundingOutput(
            request=ActionRequest(
                action_name,
                targets=estimates,
                arguments=action_arguments,
            ),
            masks=masks,
        )
