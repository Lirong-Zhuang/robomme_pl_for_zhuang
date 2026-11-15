"""Oracle-equivalent geometry helpers used to construct action waypoints.

Keep this module independent of env, SAPIEN and MPLib so it can be unit-tested
with perception outputs alone.  Do not invent a new grasp or motion policy here:
port the math from the source named in ``oracle_reference.py`` and replace only
the actor pose/OBB reads.
"""

from __future__ import annotations

from .types import Pose3D


def replace_position(pose: Pose3D, position_xyz: tuple[float, float, float]) -> Pose3D:
    """Return ``pose`` with a new position and unchanged orientation/frame."""

    return Pose3D(
        position_xyz=position_xyz,
        quaternion_wxyz=pose.quaternion_wxyz,
        frame=pose.frame,
    )


def translate_world(pose: Pose3D, offset_xyz: tuple[float, float, float]) -> Pose3D:
    """Translate a pose along canonical world/base axes."""

    return replace_position(
        pose,
        tuple(value + offset for value, offset in zip(pose.position_xyz, offset_xyz, strict=True)),
    )


