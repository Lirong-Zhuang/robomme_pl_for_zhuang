"""Oracle-style pushing functions with explicit object/target geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_multiply,
)
from robomme.robomme_env.utils.subgoal_planner_func import _record_waypoint

from .execution_state import move_cube_obj_flag_from_grasped_endpoint
from .types import Pose3D
from .types import Size3D


def direction_from_grounded_positions(
    obj_pose: Pose3D,
    target_pose: Pose3D,
) -> int:
    """Match MoveCube's execution-scene direction rule using F2G world poses."""

    obj_pos = np.asarray(obj_pose.position_xyz, dtype=np.float64)
    target_pos = np.asarray(target_pose.position_xyz, dtype=np.float64)
    if np.linalg.norm(target_pos[:2] - obj_pos[:2]) < 1e-6:
        raise ValueError("cube and target positions are too close to derive direction")
    return 1 if obj_pos[1] > target_pos[1] else -1


def solve_push_to_target(env: Any, planner: Any, *, obj_pose: Pose3D, obj_size: Size3D, target_pose: Pose3D) -> Any:
    """Port Oracle ``solve_push_to_target``."""

    planner.open_gripper()
    FINGER_LENGTH = 0.025
    env = env.unwrapped

    obj_pos = np.asarray(obj_pose.position_xyz, dtype=np.float32)
    target_pos = np.asarray(target_pose.position_xyz, dtype=np.float32)

    push_direction_xy = target_pos[:2] - obj_pos[:2]
    push_direction_xy = push_direction_xy / np.linalg.norm(push_direction_xy)
    push_direction_3d = np.array([push_direction_xy[0], push_direction_xy[1], 0])

    x_axis = np.array([push_direction_xy[0], push_direction_xy[1], 0])
    z_axis = np.array([0, 0, -1])
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        raise ValueError("Push direction invalid; cannot construct gripper frame.")
    y_axis = y_axis / y_norm

    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    rotation_matrix_torch = torch.from_numpy(rotation_matrix).float().unsqueeze(0)
    push_quat = matrix_to_quaternion(rotation_matrix_torch)[0]

    if  target_pos[0] < obj_pos[0] :
        z_rotation_angles = torch.deg2rad(
            torch.tensor([0.0, 0.0, 180], dtype=torch.float32)
        )
        z_rotation_matrix = euler_angles_to_matrix(z_rotation_angles, convention="XYZ")
        z_rotation_quat = matrix_to_quaternion(z_rotation_matrix.unsqueeze(0))[0]
        push_quat = quaternion_multiply(
            push_quat.unsqueeze(0), z_rotation_quat.unsqueeze(0)
        )[0]

    push_pose = sapien.Pose(p=obj_pos, q=push_quat.detach().cpu().numpy())

    offset_distance = 0.05
    start_pos = obj_pos - push_direction_3d * offset_distance
    start_pos[2] = push_pose.p[2]

    reach_pose_q = push_pose.q.tolist() if hasattr(push_pose.q, 'tolist') else list(push_pose.q)

    _record_waypoint(
        env,
        "solve_push_to_target",
        "close",
        waypoint_p=start_pos,
        waypoint_q=reach_pose_q,
    )
    planner.move_to_pose_with_screw(sapien.Pose(p=start_pos.tolist(), q=reach_pose_q))

    planner.close_gripper()

    cube_half_size = float(obj_size.xyz[0]) / 2.0
    end_pos = target_pos.copy() - push_direction_3d * cube_half_size
    end_pos[2] = start_pos[2]

    _record_waypoint(
        env,
        "solve_push_to_target",
        "close",
        waypoint_p=end_pos,
        waypoint_q=reach_pose_q,
    )
    planner.move_to_pose_with_screw(sapien.Pose(p=end_pos.tolist(), q=reach_pose_q))

    _record_waypoint(
        env,
        "solve_push_to_target",
        "open",
        waypoint_p=end_pos,
        waypoint_q=reach_pose_q,
    )
    planner.open_gripper()


def solve_push_to_target_with_peg(
    env: Any,
    planner: Any,
    *,
    obj_pose: Pose3D,
    target_pose: Pose3D,
) -> Any:
    """Push with a peg using geometry-derived direction and saved peg-end state."""

    FINGER_LENGTH = 0.025
    direction = direction_from_grounded_positions(obj_pose, target_pose)
    obj_flag = move_cube_obj_flag_from_grasped_endpoint(env)
    env = env.unwrapped

    obj_pos = np.asarray(obj_pose.position_xyz, dtype=np.float32)
    target_pos = np.asarray(target_pose.position_xyz, dtype=np.float32)

    push_direction_xy = target_pos[:2] - obj_pos[:2]
    push_direction_xy = push_direction_xy / np.linalg.norm(push_direction_xy)
    push_direction_3d = np.array([push_direction_xy[0], push_direction_xy[1], 0])

    x_axis = np.array([push_direction_xy[0], push_direction_xy[1], 0])
    z_axis = np.array([0, 0, -1])
    y_axis = np.cross(z_axis, x_axis)

    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        raise ValueError("Invalid push direction; failed to construct gripper frame.")
    y_axis = y_axis / y_norm

    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    rotation_matrix_torch = torch.from_numpy(rotation_matrix).float().unsqueeze(0)
    base_quat = matrix_to_quaternion(rotation_matrix_torch)[0]

    z_rotation_angles = torch.deg2rad(
        torch.tensor([0.0, 0.0, 90.0 * direction * obj_flag], dtype=torch.float32)
    )
    z_rotation_matrix = euler_angles_to_matrix(z_rotation_angles, convention="XYZ")
    z_rotation_quat = matrix_to_quaternion(z_rotation_matrix.unsqueeze(0))[0]

    push_quat = quaternion_multiply(
        base_quat.unsqueeze(0), z_rotation_quat.unsqueeze(0)
    )[0].cpu().numpy()

    push_pose = sapien.Pose(p=obj_pos, q=push_quat)

    offset_distance = 0.1
    start_pos = obj_pos - push_direction_3d * offset_distance
    start_pos[2] = obj_pos[2]

    lateral_unit = np.array([-push_direction_xy[1], push_direction_xy[0], 0], dtype=np.float32)
    lateral_norm = np.linalg.norm(lateral_unit[:2])

    if lateral_norm < 1e-6:
        lateral_unit = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        lateral_unit /= lateral_norm

    lateral_distance = 0.1 * direction
    start_pos = start_pos - lateral_unit * lateral_distance

    reach_pose_q = push_pose.q.tolist() if hasattr(push_pose.q, "tolist") else list(push_pose.q)

    start_ready_pos = start_pos.copy()

    _record_waypoint(
        env,
        "solve_push_to_target_with_peg",
        "close",
        waypoint_p=start_pos,
        waypoint_q=reach_pose_q,
    )
    planner.move_to_pose_with_screw(sapien.Pose(p=start_ready_pos.tolist(), q=reach_pose_q))
    planner.move_to_pose_with_screw(sapien.Pose(p=start_pos.tolist(), q=reach_pose_q))

    planner.close_gripper()

    end_pos = target_pos - push_direction_3d * 0.03
    end_pos[2] = start_pos[2]
    end_pos = end_pos - lateral_unit * lateral_distance
    _record_waypoint(
        env,
        "solve_push_to_target_with_peg",
        "close",
        waypoint_p=end_pos,
        waypoint_q=reach_pose_q,
    )
    planner.move_to_pose_with_screw(sapien.Pose(p=end_pos.tolist(), q=reach_pose_q))

    _record_waypoint(
        env,
        "solve_push_to_target_with_peg",
        "open",
        waypoint_p=end_pos,
        waypoint_q=reach_pose_q,
    )
    planner.open_gripper()
