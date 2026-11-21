"""Oracle-style peg functions with explicit peg and box poses."""

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
from robomme.logging_utils import logger
from robomme.robomme_env.utils.subgoal_planner_func import _record_waypoint

from .execution_state import insert_peg_obj_flag_from_grasped_endpoint
from .execution_state import remember_move_cube_center_peg_grasp
from .execution_state import remember_grasped_peg_endpoint_color
from .types import Pose3D


def _grasp_and_lift_peg(
    env: Any,
    planner: Any,
    *,
    peg_pose: Pose3D,
    peg_color: str | None,
    action_name: str,
) -> Any:
    """Shared Oracle peg pickup with whole-peg or endpoint state."""

    planner.open_gripper()
    lift_height = 0.2

    grasp_pose_p = np.asarray(peg_pose.position_xyz, dtype=np.float32).reshape(-1)
    grasp_pose_q = np.asarray(peg_pose.quaternion_wxyz, dtype=np.float32).reshape(-1)

    flip_angles = torch.tensor([[np.pi, 0.0, 0.0]], dtype=torch.float32)
    flip_matrix = euler_angles_to_matrix(flip_angles, convention="XYZ")
    flip_quat = matrix_to_quaternion(flip_matrix)[0]

    grasp_pose_q_tensor = torch.from_numpy(grasp_pose_q).to(
        dtype=torch.float32, device=flip_quat.device
    )
    rotated_quat = quaternion_multiply(
        grasp_pose_q_tensor.unsqueeze(0), flip_quat.unsqueeze(0)
    ).squeeze(0)
    grasp_pose_q = rotated_quat.detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(grasp_pose_q)
    if norm > 0:
        grasp_pose_q /= norm

    lifted_pose_p = grasp_pose_p.copy()
    lifted_pose_p[2] = lift_height

    _record_waypoint(
        env,
        action_name,
        "open",
        waypoint_p=lifted_pose_p,
        waypoint_q=grasp_pose_q,
    )
    planner.move_to_pose_with_screw(sapien.Pose(p=lifted_pose_p, q=grasp_pose_q))

    _record_waypoint(
        env,
        action_name,
        "close",
        waypoint_p=grasp_pose_p,
        waypoint_q=grasp_pose_q,
    )
    planner.move_to_pose_with_screw(sapien.Pose(p=grasp_pose_p, q=grasp_pose_q))
    planner.close_gripper()

    lifted_pose_p = grasp_pose_p.copy()
    lifted_pose_p[2] = lift_height
    _record_waypoint(
        env,
        action_name,
        "close",
        waypoint_p=lifted_pose_p,
        waypoint_q=grasp_pose_q,
    )
    planner.move_to_pose_with_screw(sapien.Pose(p=lifted_pose_p, q=grasp_pose_q))

    planner.close_gripper()

    current_grasp_pose = sapien.Pose(p=grasp_pose_p, q=grasp_pose_q)
    env.current_grasp_pose = current_grasp_pose
    if peg_color is None:
        remember_move_cube_center_peg_grasp(env)
    else:
        # Preserve the actually grounded endpoint for the subsequent peg action.
        remember_grasped_peg_endpoint_color(env, peg_color)


def grasp_and_lift_peg(
    env: Any,
    planner: Any,
    *,
    peg_pose: Pose3D,
) -> Any:
    """Grasp the whole peg near its grounded center for MoveCube."""

    return _grasp_and_lift_peg(
        env,
        planner,
        peg_pose=peg_pose,
        peg_color=None,
        action_name="grasp_and_lift_peg",
    )


def grasp_and_lift_peg_side(
    env: Any,
    planner: Any,
    *,
    peg_pose: Pose3D,
    peg_color: str,
) -> Any:
    """Grasp one VLM-selected black/white peg half for InsertPeg."""

    return _grasp_and_lift_peg(
        env,
        planner,
        peg_pose=peg_pose,
        peg_color=peg_color,
        action_name="grasp_and_lift_peg_side",
    )


def insert_peg(env: Any, planner: Any, *, peg_pose: Pose3D, box_pose: Pose3D, direction: int, cut_retreat: bool = False) -> Any:
    """Port Oracle ``insert_peg`` with explicit box/insert-object poses."""

    obj_flag = insert_peg_obj_flag_from_grasped_endpoint(env)

    box_sapien_pose = sapien.Pose(
        p=list(box_pose.position_xyz),
        q=list(box_pose.quaternion_wxyz),
    )
    peg_sapien_pose = sapien.Pose(
        p=list(peg_pose.position_xyz),
        q=list(peg_pose.quaternion_wxyz),
    )
    peg_to_tcp = peg_sapien_pose.inv() * env.agent.tcp.pose

    def _compute_insert_pose():
        pose = box_sapien_pose * peg_to_tcp
        if obj_flag == -1:
            pose = pose * sapien.Pose(q=[0, 0, 0, 1])
        if direction == -1:
            pose = pose * sapien.Pose(q=[0, 0, 0, 1])
        return pose

    def _resolve_target_pose(offset):
        current_pose = _compute_insert_pose()
        offset_vec = np.asarray(offset, dtype=np.float32).reshape(-1).copy()
        if obj_flag == 1 and direction == -1 and offset_vec[0] < 0:
            relative_p = np.asarray(peg_to_tcp.p, dtype=np.float32).reshape(-1)
            offset_vec[0] += relative_p[0]
        return current_pose * sapien.Pose(offset_vec.tolist())

    def _record_target_waypoint(target_pose, waypoint_type="close"):
        _record_waypoint(
            env,
            "insert_peg",
            waypoint_type,
            waypoint_p=np.asarray(target_pose.p, dtype=np.float32).reshape(-1),
            waypoint_q=np.asarray(target_pose.q, dtype=np.float32).reshape(-1),
        )

    def _move_with_offset(offset, *, target_pose=None):
        if target_pose is None:
            target_pose = _resolve_target_pose(offset)
        planner.move_to_pose_with_screw(target_pose)

    def _move_with_offset_with_break(offset, *, target_pose=None):
        end_steps = getattr(env, "end_steps", None)
        if end_steps is not None and int(getattr(env, "elapsed_steps", 0)) > end_steps + 3:
            return True

        if target_pose is None:
            target_pose = _resolve_target_pose(offset)

        pose_for_plan = planner._transform_pose_for_planning(target_pose)
        pose_p = np.asarray(pose_for_plan.p, dtype=np.float32).reshape(-1)
        pose_q = np.asarray(pose_for_plan.q, dtype=np.float32).reshape(-1)
        result = planner.planner.plan_screw(
            np.concatenate([pose_p, pose_q]),
            planner.robot.get_qpos().cpu().numpy()[0],
            time_step=planner.base_env.control_timestep,
            use_point_cloud=planner.use_point_cloud,
        )
        if result["status"] != "Success":
            return False

        n_step = result["position"].shape[0]
        for i in range(n_step):
            if end_steps is not None and int(getattr(env, "elapsed_steps", 0)) > end_steps + 3:
                logger.debug("break early")
                return True
            qpos = result["position"][i]
            if planner.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][i]
                action = np.hstack([qpos, qvel, planner.gripper_state])
            else:
                action = np.hstack([qpos, planner.gripper_state])
            planner.env.step(action)
            planner.elapsed_steps += 1
        return True

    def _record_and_move(offset, *, with_break=False, waypoint_type="close"):
        target_pose = _resolve_target_pose(offset)
        _record_target_waypoint(target_pose, waypoint_type=waypoint_type)
        if with_break:
            return _move_with_offset_with_break(offset, target_pose=target_pose)
        _move_with_offset(offset, target_pose=target_pose)
        return True

    if obj_flag==-1:
        _record_and_move([0.2, 0, -0.15])
        _record_and_move([0.2, 0, 0])

        _move_with_offset([0.15 , 0, 0])
        if cut_retreat!=True:
            _record_and_move([-0.05, 0, 0])
        else:
            logger.debug(
                f"cut_retreat mode (obj=-1): "
                f"elapsed_steps={int(getattr(env, 'elapsed_steps', 0))}, "
                f"end_steps={getattr(env, 'end_steps', None)}"
            )
            _record_and_move([-0.05, 0, 0], with_break=True)
    else:
        _record_and_move([-0.2, 0, -0.15])
        _record_and_move([-0.2, 0, 0])

        _move_with_offset([-0.15 , 0, 0])
        if cut_retreat!=True:
            _record_and_move([-0.05, 0, 0])
        else:
            logger.debug(
                f"cut_retreat mode (obj=1): "
                f"elapsed_steps={int(getattr(env, 'elapsed_steps', 0))}, "
                f"end_steps={getattr(env, 'end_steps', None)}"
            )
            _record_and_move([-0.05, 0, 0], with_break=True)
