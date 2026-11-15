"""Oracle-style pickup and placement functions with explicit geometry inputs.

Fill these first: together with ``button.py`` they form a complete BinFill
vertical slice. Port each referenced RoboMME function body and replace only actor
geometry reads; keep env, planner, waypoints, constants, and gripper timing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import trimesh

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
)
from robomme.robomme_env.utils.subgoal_planner_func import _record_waypoint

from .types import Pose3D
from .types import Size3D


def solve_pickup(env: Any, planner: Any, obj_pose: Pose3D, obj_size: Size3D) -> Any:

    planner.open_gripper()

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # retrieves the object oriented bounding box (trimesh box object)
    obj_sapien_pose = sapien.Pose(
        p=list(obj_pose.position_xyz),
        q=list(obj_pose.quaternion_wxyz),
    )

    obb = trimesh.primitives.Box(
        extents=np.asarray(obj_size.xyz),
        transform=obj_sapien_pose.to_transformation_matrix(),
    )

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, np.asarray(obj_pose.position_xyz))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose_p = grasp_pose.p.tolist() if hasattr(grasp_pose.p, 'tolist') else list(grasp_pose.p)
    reach_pose_q = grasp_pose.q.tolist() if hasattr(grasp_pose.q, 'tolist') else list(grasp_pose.q)
    reach_pose_p[2]=0.15
    _record_waypoint(
        env,
        "solve_pickup",
        "open",
        waypoint_p=reach_pose_p,
        waypoint_q=reach_pose_q,
    )
    planner.move_to_pose_with_screw(sapien.Pose(p=reach_pose_p,q=reach_pose_q))

    planner.open_gripper()

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    grasp_pose_p = grasp_pose.p.tolist() if hasattr(grasp_pose.p, 'tolist') else list(grasp_pose.p)
    grasp_pose_q = grasp_pose.q.tolist() if hasattr(grasp_pose.q, 'tolist') else list(grasp_pose.q)

    _record_waypoint(
        env,
        "solve_pickup",
        "close",
        waypoint_p=grasp_pose_p,
        waypoint_q=grasp_pose_q,
    )
    planner.move_to_pose_with_screw(grasp_pose)

    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose_P=list(obj_pose.position_xyz)
    goal_pose_P[2]=0.15
    goal_pose = sapien.Pose(goal_pose_P, grasp_pose.q)
    _record_waypoint(
        env,
        "solve_pickup",
        "close",
        waypoint_p=goal_pose_P,
        waypoint_q=grasp_pose_q,
    )
    res = planner.move_to_pose_with_screw(goal_pose)

    #planner.close_gripper()

    planner.close()
    return res


def solve_put_into_bin(env: Any, planner: Any, *, target_pose: Pose3D) -> Any:
    """Port Oracle ``solve_putonto_whenhold_binspecial``."""

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    grasp_pose_q=env.agent.tcp.pose.q.tolist()[0]

    goal_pose_P=list(target_pose.position_xyz)
    goal_pose_P[2]=0.2
    goal_pose = sapien.Pose(goal_pose_P, grasp_pose_q)
    _record_waypoint(
        env,
        "solve_putonto_whenhold_binspecial",
        "close",
        waypoint_p=goal_pose_P,
        waypoint_q=grasp_pose_q,
    )
    res = planner.move_to_pose_with_screw(goal_pose)

    _record_waypoint(
        env,
        "solve_putonto_whenhold_binspecial",
        "open",
        waypoint_p=goal_pose_P,
        waypoint_q=grasp_pose_q,
    )
    planner.open_gripper()

    goal_pose_P=list(target_pose.position_xyz)
    goal_pose_P[0]=goal_pose_P[0]-0.1
    goal_pose_P[2]=0.2
    goal_pose = sapien.Pose(goal_pose_P, grasp_pose_q)
    res = planner.move_to_pose_with_screw(goal_pose)
    planner.close()
    return res


def solve_putonto_target(env: Any, planner: Any, *, target_pose: Pose3D, height: float | None = None) -> Any:
    """Port Oracle ``solve_putonto_whenhold``."""

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    grasp_pose_q=env.agent.tcp.pose.q.tolist()[0]

    goal_pose_P_prepare=list(target_pose.position_xyz)
    goal_pose_P_prepare[2]=0.15
    goal_pose = sapien.Pose(goal_pose_P_prepare, grasp_pose_q)
    _record_waypoint(
        env,
        "solve_putonto_whenhold",
        "close",
        waypoint_p=goal_pose_P_prepare,
        waypoint_q=grasp_pose_q,
    )
    res = planner.move_to_pose_with_screw(goal_pose)

    goal_pose_P=list(target_pose.position_xyz)
    if height is not None:
        goal_pose_P[2]=height

    goal_pose = sapien.Pose(goal_pose_P, grasp_pose_q)
    _record_waypoint(
        env,
        "solve_putonto_whenhold",
        "open",
        waypoint_p=goal_pose_P,
        waypoint_q=grasp_pose_q,
    )
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.open_gripper()

    goal_pose_P=list(target_pose.position_xyz)
    goal_pose_P[2]=0.15
    goal_pose = sapien.Pose(goal_pose_P, grasp_pose_q)
    _record_waypoint(
        env,
        "solve_putonto_whenhold",
        "open",
        waypoint_p=goal_pose_P,
        waypoint_q=grasp_pose_q,
    )
    res = planner.move_to_pose_with_screw(goal_pose)
    planner.close()
    return res


def solve_putdown(env: Any, planner: Any, *, release_z: float = 0.07) -> Any:
    """Port Oracle ``solve_putdown_whenhold``; no target actor is needed."""

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    grasp_pose_q=env.agent.tcp.pose.q.tolist()[0]

    goal_pose_P=env.agent.tcp.pose.p.tolist()[0]
    goal_pose_P[2]=release_z
    goal_pose = sapien.Pose(goal_pose_P,grasp_pose_q)
    _record_waypoint(
        env,
        "solve_putdown_whenhold",
        "open",
        waypoint_p=goal_pose_P,
        waypoint_q=grasp_pose_q,
    )
    res = planner.move_to_pose_with_screw(goal_pose)
    planner.open_gripper()

    goal_pose_P=env.agent.tcp.pose.p.tolist()[0]
    goal_pose_P[2]=0.15

    goal_pose = sapien.Pose(goal_pose_P, grasp_pose_q)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res


def solve_pickup_bin(env: Any, planner: Any, *, obj_pose: Pose3D, obj_size: Size3D) -> Any:
    """Port Oracle ``solve_pickup_bin`` using pose/size instead of actor obj."""

    planner.open_gripper()

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    obj_sapien_pose = sapien.Pose(
        p=list(obj_pose.position_xyz),
        q=list(obj_pose.quaternion_wxyz),
    )
    obb = trimesh.primitives.Box(
        extents=np.asarray(obj_size.xyz),
        transform=obj_sapien_pose.to_transformation_matrix(),
    )

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, np.asarray(obj_pose.position_xyz))

    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.15])
    reach_pose_p=reach_pose.p.tolist()
    reach_pose_p[2]=0.2
    reach_pose_q=reach_pose.q.tolist()
    reach_pose_fix=sapien.Pose(reach_pose_p,reach_pose_q)
    _record_waypoint(
        env,
        "solve_pickup_bin",
        "open",
        waypoint_p=reach_pose_p,
        waypoint_q=reach_pose_q,
    )
    planner.move_to_pose_with_screw(reach_pose_fix)
    planner.open_gripper()

    grasp_pose_up=grasp_pose * sapien.Pose([0, 0, -0.01])
    _record_waypoint(
        env,
        "solve_pickup_bin",
        "close",
        waypoint_p=np.asarray(grasp_pose_up.p, dtype=np.float32).reshape(-1),
        waypoint_q=np.asarray(grasp_pose_up.q, dtype=np.float32).reshape(-1),
    )
    planner.move_to_pose_with_screw(grasp_pose_up)
    planner.close_gripper()

    goal_pose_P=list(obj_pose.position_xyz)
    goal_pose_P[2]=0.2
    goal_pose = sapien.Pose(goal_pose_P, grasp_pose.q)
    _record_waypoint(
        env,
        "solve_pickup_bin",
        "close",
        waypoint_p=goal_pose_P,
        waypoint_q=np.asarray(grasp_pose.q, dtype=np.float32).reshape(-1),
    )
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res


def solve_swingonto_whenhold(env: Any, planner: Any, *, target_pose: Pose3D, height: float = 0.05) -> Any:
    """Port Oracle ``solve_swingonto_whenhold``."""

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    grasp_pose_q=env.agent.tcp.pose.q.tolist()[0]
    goal_pose_P=list(target_pose.position_xyz)
    goal_pose_P[2]=height
    goal_pose = sapien.Pose(goal_pose_P, grasp_pose_q)
    _record_waypoint(
        env,
        "solve_swingonto_whenhold",
        "close",
        waypoint_p=goal_pose_P,
        waypoint_q=grasp_pose_q,
    )
    res = planner.move_to_pose_with_screw(goal_pose)
    planner.close()
    return res
