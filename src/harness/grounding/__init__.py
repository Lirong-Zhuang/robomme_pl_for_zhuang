"""Trusted boundary between VLM 2D hints and metric perception."""

from typing import Any

from .contracts import GroundingContractRegistry
from .contracts import normalize_vlm_grounding


__all__ = [
    "CameraFrame",
    "F2GClient",
    "F2GServiceError",
    "GroundingContractRegistry",
    "GroundingOutput",
    "GroundingPipeline",
    "OpenCVSegmenter",
    "camera_frame_from_robomme",
    "depth_statistics",
    "normalize_vlm_grounding",
]


def __getattr__(name: str) -> Any:
    """Keep contract validation usable without importing OpenCV/SAPIEN."""

    if name in {"F2GClient", "F2GServiceError"}:
        from . import f2g_client

        return getattr(f2g_client, name)
    if name in {"GroundingOutput", "GroundingPipeline"}:
        from . import grounder

        return getattr(grounder, name)
    if name in {"camera_frame_from_robomme", "depth_statistics"}:
        from . import observation

        return getattr(observation, name)
    if name == "OpenCVSegmenter":
        from .opencv_segmenter import OpenCVSegmenter

        return OpenCVSegmenter
    if name == "CameraFrame":
        from .types import CameraFrame

        return CameraFrame
    raise AttributeError(name)
