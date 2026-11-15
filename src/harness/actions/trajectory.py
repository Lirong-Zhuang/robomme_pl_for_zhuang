"""Oracle-style PatternLock and RouteStick functions using perceived targets."""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch
from robomme.logging_utils import logger
from robomme.robomme_env.utils.subgoal_planner_func import _record_waypoint

from .types import Pose3D


def _solve_swingonto(env: Any, planner: Any, target_pose: Pose3D, record_swing_qpos: bool = False) -> Any:
    env = env.unwrapped
    current_qpos = env.agent.tcp.pose.q.reshape(-1, 4)[0]
    goal_pose_P=list(target_pose.position_xyz)
    goal_pose_P[2]=0.07
    goal_pose = sapien.Pose(goal_pose_P, current_qpos)
    for i in range(2):
        if i == 1 :
            _record_waypoint(env, "solve_swingonto", "close", waypoint_p=goal_pose_P, waypoint_q=current_qpos)
        res = planner.move_to_pose_with_screw(goal_pose)
    try:
        planner.close_gripper()
    except:
        AttributeError
    if record_swing_qpos==True:
        env.swing_qpos=env.agent.robot.qpos
    return res


def _solve_swingonto_with_direction(env: Any, planner: Any, target_pose: Pose3D, radius: float = 0.1, direction: str = "counterclockwise") -> Any:
    """Port Oracle ``solve_swingonto_withDirection`` with explicit target pose."""

    start_pos = env.agent.tcp.pose.p.reshape(-1, 3)[0]
    end_pos = np.asarray(target_pose.position_xyz, dtype=np.float32)
    start_xy = np.asarray(start_pos[:2], dtype=np.float32)
    end_xy = np.asarray(end_pos[:2], dtype=np.float32)
    chord_vec = end_xy - start_xy
    chord_len = np.linalg.norm(chord_vec)
    current_qpos = env.agent.tcp.pose.q.reshape(-1, 4)[0].tolist()

    init_qpos_tensor = planner.robot.get_qpos()
    qpos_device = init_qpos_tensor.device if hasattr(init_qpos_tensor, "device") else None
    qpos_dtype = init_qpos_tensor.dtype if hasattr(init_qpos_tensor, "dtype") else torch.float32
    init_qpos = (init_qpos_tensor.detach().cpu().numpy() if hasattr(init_qpos_tensor, "detach") else np.asarray(init_qpos_tensor)).reshape(-1)
    plan_start_qpos = init_qpos.copy()

    waypoints = []
    if chord_len < 1e-6:
        goal_p = end_pos.tolist()
        goal_p[2] = 0.07
        waypoints.append(sapien.Pose(goal_p, current_qpos))
    else:
        lateral_offset = float(max(radius, 1e-4))
        dir_unit = chord_vec / chord_len
        perp = np.array([-dir_unit[1], dir_unit[0]])
        direction_l = str(direction).lower()
        if direction_l == "counterclockwise":
            sign = -1.0
        elif direction_l == "clockwise":
            sign = 1.0
        else:
            sign = 1.0 if direction_l == "left" else -1.0
        control_xy = (start_xy + end_xy) / 2.0 + sign * lateral_offset * perp
        num_steps = 45
        for t in np.linspace(0.0, 1.0, num_steps):
            one_minus_t = 1.0 - t
            waypoint_xy = ((one_minus_t ** 2) * start_xy + 2 * one_minus_t * t * control_xy + (t ** 2) * end_xy)
            goal_p = [float(waypoint_xy[0]), float(waypoint_xy[1]), 0.07]
            waypoints.append(sapien.Pose(goal_p, current_qpos))
    if waypoints:
        last_p = np.asarray(waypoints[-1].p, dtype=np.float32).reshape(-1).tolist()
        last_q = np.asarray(waypoints[-1].q, dtype=np.float32).reshape(-1).tolist()
        for _ in range(5):
            waypoints.append(sapien.Pose(last_p, last_q))
    logger.debug(" get waypoint")

    positions = []
    last_res = None
    plan_start_qpos_full = planner.planner.pad_qpos(plan_start_qpos.copy())
    for idx, wp in enumerate(waypoints):
        pose_for_plan = planner._transform_pose_for_planning(wp)
        pose_p = np.asarray(pose_for_plan.p, dtype=np.float32).reshape(-1)
        pose_q = np.asarray(pose_for_plan.q, dtype=np.float32).reshape(-1)
        goal_world = np.concatenate([pose_p, pose_q])
        goal_base = planner.planner.transform_goal_to_wrt_base(goal_world)
        ik_status, ik_solutions = planner.planner.IK(goal_base, plan_start_qpos_full.copy())
        if ik_status != "Success" or len(ik_solutions) == 0:
            logger.debug(f"IK failed at waypoint {idx}: {ik_status}")
            continue
        qpos_sol = ik_solutions[0]
        padded_qpos = plan_start_qpos_full.copy()
        padded_qpos[: qpos_sol.shape[0]] = qpos_sol
        positions.append(padded_qpos)
        plan_start_qpos_full = padded_qpos

    if len(positions) == 0:
        logger.debug("No IK solutions found for waypoints, aborting follow_path.")
        return None

    full_positions = np.stack(positions, axis=0)
    mid_idx = len(full_positions) // 2
    traj_res_1 = {"status": "Success", "position": full_positions[:mid_idx]}
    if planner.control_mode == "pd_joint_pos_vel":
        traj_res_1["velocity"] = np.zeros_like(traj_res_1["position"])
    mid_pose = waypoints[mid_idx - 1] if mid_idx > 0 else waypoints[0]
    _record_waypoint(env, "solve_swingonto_withDirection", "close", waypoint_p=np.asarray(mid_pose.p, dtype=np.float32).reshape(-1), waypoint_q=np.asarray(mid_pose.q, dtype=np.float32).reshape(-1))
    planner.follow_path(traj_res_1)

    traj_res_2 = {"status": "Success", "position": full_positions[mid_idx:]}
    if planner.control_mode == "pd_joint_pos_vel":
        traj_res_2["velocity"] = np.zeros_like(traj_res_2["position"])
    end_pose = waypoints[-1]
    _record_waypoint(env, "solve_swingonto_withDirection", "close", waypoint_p=np.asarray(end_pose.p, dtype=np.float32).reshape(-1), waypoint_q=np.asarray(end_pose.q, dtype=np.float32).reshape(-1))
    last_res = planner.follow_path(traj_res_2)
    return last_res


def solve_pattern_direction(env: Any, planner: Any, *, target_poses: tuple[Pose3D, ...], direction: str) -> Any:
    directions = ["forward", "backward", "left", "right", "forward-left", "forward-right", "backward-left", "backward-right"]
    dir_vectors = {
        "forward": np.array([1.0, 0.0]), "backward": np.array([-1.0, 0.0]),
        "left": np.array([0.0, 1.0]), "right": np.array([0.0, -1.0]),
        "forward-left": np.array([1.0, 1.0]) / np.sqrt(2.0), "forward-right": np.array([1.0, -1.0]) / np.sqrt(2.0),
        "backward-left": np.array([-1.0, 1.0]) / np.sqrt(2.0), "backward-right": np.array([-1.0, -1.0]) / np.sqrt(2.0),
    }
    eps = 1e-6
    dir_filters = {
        "forward": lambda dx, dy: dx > eps, "backward": lambda dx, dy: dx < -eps,
        "left": lambda dx, dy: dy > eps, "right": lambda dx, dy: dy < -eps,
        "forward-left": lambda dx, dy: dx > eps and dy > eps, "forward-right": lambda dx, dy: dx > eps and dy < -eps,
        "backward-left": lambda dx, dy: dx < -eps and dy > eps, "backward-right": lambda dx, dy: dx < -eps and dy < -eps,
    }
    if direction not in directions:
        raise ValueError(f"Unsupported PatternLock direction: {direction!r}")

    tcp_pose = np.asarray(env.agent.tcp.pose.p).reshape(-1)
    if tcp_pose.size < 2:
        raise ValueError("TCP pose does not provide x/y coordinates.")
    tcp_xy = tcp_pose[:2]
    dist_list = []
    for target_pose in target_poses:
        target_xy = np.asarray(target_pose.position_xyz, dtype=np.float32)[:2]
        dist_list.append((float(np.linalg.norm(target_xy - tcp_xy)), target_pose, target_xy))
    if not dist_list:
        raise ValueError("PatternLock requires perceived target poses.")
    dist_list.sort(key=lambda item: item[0])
    ref_target, ref_xy = dist_list[0][1], dist_list[0][2]

    vec = dir_vectors[direction]
    filt = dir_filters[direction]
    best = None
    best_score = None
    for candidate in target_poses:
        if candidate is ref_target:
            continue
        candidate_xy = np.asarray(candidate.position_xyz, dtype=np.float32)[:2]
        delta = candidate_xy - ref_xy
        dx, dy = float(delta[0]), float(delta[1])
        if not filt(dx, dy):
            continue
        dist = float(np.linalg.norm(delta))
        if dist < eps:
            continue
        if dist>0.2:
            continue
        align = float(np.dot(delta / dist, vec))
        score = (-align*0.5, dist)
        if best_score is None or score < best_score:
            best_score = score
            best = candidate
    if best is None:
        logger.debug(f"[PatternLock] No candidate in direction '{direction}', using nearest target.")
        best = ref_target
    return [_solve_swingonto(env, planner, best), _solve_swingonto(env, planner, best)]


def solve_swingonto_with_direction(env: Any, planner: Any, *, target_poses: tuple[Pose3D, ...], side: str, direction: str, radius: float = 0.2, route_button_indices: tuple[int, ...] = (0, 2, 4, 6, 8)) -> Any:
    dir_map = {"clock": "clockwise", "clockwise": "clockwise", "cw": "clockwise", "counterclock": "counterclockwise", "counterclockwise": "counterclockwise", "anticlockwise": "counterclockwise", "ccw": "counterclockwise"}
    direction_norm = dir_map.get(str(direction).lower(), "counterclockwise")
    allowed_set = set(int(index) for index in route_button_indices)
    allowed_candidates = [(index, target_pose) for index, target_pose in enumerate(target_poses) if index in allowed_set]

    tcp_pose = np.asarray(env.agent.tcp.pose.p).reshape(-1)
    tcp_xy = tcp_pose[:2] if tcp_pose.size >= 2 else None
    dist_list = []
    for index, target_pose in allowed_candidates:
        target_xy = np.asarray(target_pose.position_xyz, dtype=np.float32)[:2]
        if tcp_xy is not None:
            dist_list.append((np.linalg.norm(target_xy - tcp_xy), index, target_pose))
    if dist_list:
        _, ref_idx, ref_target = min(dist_list, key=lambda item: item[0])
    elif allowed_candidates:
        ref_idx, ref_target = allowed_candidates[0]
    else:
        ref_idx, ref_target = 0, None

    candidates = [(index, target_pose) for index, target_pose in allowed_candidates if target_pose is not ref_target]
    if not candidates:
        raise ValueError("RouteStick: no available targets to swing onto.")

    side_l = str(side).lower()
    side_candidates = []
    for index, target_pose in candidates:
        if side_l == "left" and index > ref_idx:
            side_candidates.append((index, target_pose))
        elif side_l == "right" and index < ref_idx:
            side_candidates.append((index, target_pose))

    if not side_candidates:
        fallback_target = None
        min_dist = float('inf')
        for index, target_pose in allowed_candidates:
            target_xy = np.asarray(target_pose.position_xyz, dtype=np.float32)[:2]
            if tcp_xy is not None:
                distance = np.linalg.norm(target_xy - tcp_xy)
                if distance < min_dist:
                    min_dist = distance
                    fallback_target = target_pose
        if fallback_target is None:
            if candidates:
                fallback_target = allowed_candidates[0][1]
            else:
                raise ValueError("RouteStick: failed to determine a target for the selected side.")
        logger.debug(f"RouteStick: failed to determine a target for side '{side_l}'. Fallback to nearest target.")
        return _solve_swingonto(env, planner, fallback_target)

    if side_l == "left":
        _target_idx, target = min(side_candidates, key=lambda item: item[0] - ref_idx)
    else:
        _target_idx, target = min(side_candidates, key=lambda item: ref_idx - item[0])
    return _solve_swingonto_with_direction(env, planner, target, radius=radius, direction=direction_norm)
