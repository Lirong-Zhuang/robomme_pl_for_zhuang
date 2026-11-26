"""Oracle-style button functions using the simulator's semantic button actor."""

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

from .env_targets import button_position


def solve_button(
    env: Any,
    planner: Any,
    *,
    steps_press: int | None = None,
    interval: int = 20,
    without_hold: bool = False,
    ordinal: str | None = None,
) -> Any:
    """Port Oracle ``solve_button`` and read ``obj.pose`` from the environment."""

    env = env.unwrapped
    position = button_position(env, ordinal=ordinal)
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
    if not without_hold:
        planner.move_to_pose_with_screw(sapien.Pose(p=ready_position,q=rotate))

    _record_waypoint(
        env,
        "solve_button",
        "close",
        waypoint_p=ready_position,
        waypoint_q=rotate,
    )
    planner.close_gripper()

    steps = env.elapsed_steps.item()
    logger.debug("press button at step %s", steps)

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


def solve_button_ready(env: Any, planner: Any) -> Any:
    """Port Oracle ``solve_button_ready`` using the environment's button pose."""

    env = env.unwrapped
    position = button_position(env)
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
