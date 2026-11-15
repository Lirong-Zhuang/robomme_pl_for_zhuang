"""Oracle-style button functions with explicit button pose input."""

from __future__ import annotations

from typing import Any

import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from robomme.logging_utils import logger
from robomme.robomme_env.utils.subgoal_planner_func import _record_waypoint

from .types import Pose3D


def solve_button(env: Any, planner: Any, *, button_pose: Pose3D, steps_press: int | None = None, interval: int = 20, without_hold: bool = False) -> Any:
    """Port Oracle ``solve_button`` and replace only ``obj.pose``."""

    FINGER_LENGTH = 0.025
    env=env.unwrapped
    position=list(button_pose.position_xyz)
    ready_position=position.copy()
    ready_position[2]=0.15

    angles = torch.deg2rad(torch.tensor([180.0, 0.0, 0.0], dtype=torch.float32))
    rotate = matrix_to_quaternion(
        euler_angles_to_matrix(angles, convention="XYZ")
    )
    _record_waypoint(
        env,
        "solve_button",
        "open",
        waypoint_p=ready_position,
        waypoint_q=rotate,
    )
    if without_hold==False:
        planner.move_to_pose_with_screw(sapien.Pose(p=ready_position,q=rotate))

    _record_waypoint(
        env,
        "solve_button",
        "close",
        waypoint_p=ready_position,
        waypoint_q=rotate,
    )
    planner.close_gripper()

    steps=env.elapsed_steps.item()
    logger.debug("press button at step %s", steps)
    rotate_list = rotate.tolist() if hasattr(rotate, 'tolist') else rotate

    _record_waypoint(
        env,
        "solve_button",
        "close",
        waypoint_p=position,
        waypoint_q=rotate,
    )
    planner.move_to_pose_with_screw(sapien.Pose(p=position,q=rotate))

    _record_waypoint(
        env,
        "solve_button",
        "close",
        waypoint_p=ready_position,
        waypoint_q=rotate,
    )
    planner.move_to_pose_with_screw(sapien.Pose(p=ready_position, q=rotate))


def solve_button_ready(env: Any, planner: Any, *, button_pose: Pose3D) -> Any:
    """Port Oracle ``solve_button_ready`` and replace only ``obj.pose``."""

    FINGER_LENGTH = 0.025
    env=env.unwrapped
    position=list(button_pose.position_xyz)
    ready_position=position.copy()
    ready_position[2]=0.15

    angles = torch.deg2rad(torch.tensor([180.0, 0.0, 0.0], dtype=torch.float32))
    rotate = matrix_to_quaternion(
        euler_angles_to_matrix(angles, convention="XYZ")
    )

    _record_waypoint(
        env,
        "solve_button_ready",
        "close",
        waypoint_p=ready_position,
        waypoint_q=rotate,
    )
    planner.move_to_pose_with_screw(sapien.Pose(p=ready_position,q=rotate))
    planner.close_gripper()
