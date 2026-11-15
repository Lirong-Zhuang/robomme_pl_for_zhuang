"""Data contracts shared by grounding, actions, and the motion backend.

The perception layer must convert OpenCV/F2G outputs into these types before an
action is called.  Actions therefore never receive a SAPIEN actor as a target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal
from typing import Mapping


Vector3 = tuple[float, float, float]
QuaternionWXYZ = tuple[float, float, float, float]
ActionStatus = Literal["success", "fail", "timeout", "error", "ongoing"]


def _finite_tuple(name: str, value: tuple[float, ...], length: int) -> tuple[float, ...]:
    if len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    converted = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in converted):
        raise ValueError(f"{name} must contain only finite values")
    return converted


@dataclass(frozen=True)
class Pose3D:
    """Rigid pose in a named frame, expressed in metres and WXYZ quaternion order."""

    position_xyz: Vector3
    quaternion_wxyz: QuaternionWXYZ
    frame: str = "world"

    def __post_init__(self) -> None:
        position = _finite_tuple("position_xyz", self.position_xyz, 3)
        quaternion = _finite_tuple("quaternion_wxyz", self.quaternion_wxyz, 4)
        norm = math.sqrt(sum(component * component for component in quaternion))
        if norm < 1e-8:
            raise ValueError("quaternion_wxyz must have non-zero norm")
        frame = self.frame.strip()
        if not frame:
            raise ValueError("frame must be non-empty")
        object.__setattr__(self, "position_xyz", position)
        object.__setattr__(
            self,
            "quaternion_wxyz",
            tuple(component / norm for component in quaternion),
        )
        object.__setattr__(self, "frame", frame)


@dataclass(frozen=True)
class Size3D:
    """Full oriented-box dimensions along the object's local x/y/z axes."""

    xyz: Vector3

    def __post_init__(self) -> None:
        size = _finite_tuple("size.xyz", self.xyz, 3)
        if any(component <= 0 for component in size):
            raise ValueError("size.xyz values must be positive")
        object.__setattr__(self, "xyz", size)


@dataclass(frozen=True)
class ObjectEstimate:
    """Perception-owned object geometry; no simulator object is stored here."""

    object_id: str
    pose: Pose3D
    size: Size3D
    label: str | None = None
    confidence: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object_id = self.object_id.strip()
        if not object_id:
            raise ValueError("object_id must be non-empty")
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class ActionRequest:
    """One grounded call to a catalog action.

    ``targets`` is keyed by semantic roles declared by the action, for example
    ``object``, ``button``, ``container``, ``target``, ``peg`` or ``box``.
    ``target_sets`` carries candidate collections needed by PatternLock/RouteStick.
    """

    action_name: str
    targets: Mapping[str, ObjectEstimate] = field(default_factory=dict)
    target_sets: Mapping[str, tuple[ObjectEstimate, ...]] = field(default_factory=dict)
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.action_name.strip()
        if not name:
            raise ValueError("action_name must be non-empty")
        object.__setattr__(self, "action_name", name)


@dataclass(frozen=True)
class ActionResult:
    """Normalized result returned to the harness after one high-level action."""

    status: ActionStatus
    message: str = ""
    stopped: bool = False
    simulator_frame_count: int = 0
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.simulator_frame_count < 0:
            raise ValueError("simulator_frame_count must be non-negative")
