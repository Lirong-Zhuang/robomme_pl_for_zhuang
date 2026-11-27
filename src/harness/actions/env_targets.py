"""Resolve simulator-owned targets used by privileged Oracle-style actions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .execution_state import require_grasped_peg_endpoint_color
from .types import ActionRequest, ObjectEstimate, Pose3D, Size3D


def _base_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def actor_position(actor: Any, *, target_name: str) -> list[float]:
    """Return the first environment actor position as three finite floats."""

    pose = getattr(actor, "pose", None)
    position = getattr(pose, "p", None)
    if position is None:
        raise RuntimeError(f"environment target {target_name!r} has no pose.p")
    values = _numpy(position).reshape(-1, 3)[0]
    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"environment target {target_name!r} has a non-finite position"
        )
    return [float(item) for item in values]


def _actor(base: Any, name: str) -> Any:
    actor = getattr(base, name, None)
    if actor is None:
        raise RuntimeError(f"environment target {name!r} is unavailable")
    return actor


def _estimate(actor: Any, name: str, *, size: Size3D | None = None) -> ObjectEstimate:
    orientation = getattr(getattr(actor, "pose", None), "q", None)
    if orientation is None:
        raise RuntimeError(f"environment target {name!r} has no pose.q")
    return ObjectEstimate(
        object_id=name,
        pose=Pose3D(
            tuple(actor_position(actor, target_name=name)),
            tuple(_numpy(orientation).reshape(-1, 4)[0]),
        ),
        size=size,
        metadata={"source": "environment"},
    )


def _free_peg_end(base: Any) -> ObjectEstimate:
    color = require_grasped_peg_endpoint_color(base)
    heads = _actor(base, "peg_heads")
    tails = _actor(base, "peg_tails")
    matches = []
    for index, (head, tail) in enumerate(zip(heads, tails, strict=True)):
        held, free = (head, tail) if color == "black" else (tail, head)
        if bool(_numpy(base.agent.is_grasping(held)).item()):
            matches.append((index, free))
    if len(matches) != 1:
        raise RuntimeError(f"expected one grasped {color} peg endpoint, found {len(matches)}")
    index, free = matches[0]
    return _estimate(free, f"peg_{index}_free_end")


def bind_environment_geometry(env: Any, request: ActionRequest) -> ActionRequest:
    """Supply live geometry without consulting the benchmark's intended plan."""
    name = request.action_name
    base = _base_env(env)
    if name in {"solve_push_to_target", "solve_push_to_target_with_peg"}:
        # MoveCube's first cube/disk are the execution pair; the second is the demo pair.
        size = None
        if name == "solve_push_to_target":
            diameter = 2.0 * float(base.cube_half_size)
            size = Size3D((diameter, diameter, diameter))
        targets = {
            "object": _estimate(_actor(base, "cube"), "cube", size=size),
            "target": _estimate(_actor(base, "goal_site"), "goal_site"),
        }
    elif name == "solve_putonto_target":
        actor_name = "goal_site" if getattr(base, "goal_site", None) is not None else "target"
        targets = {"target": _estimate(_actor(base, actor_name), actor_name)}
    elif name == "solve_swingonto_whenhold":
        side = request.arguments.get("side")
        if side not in {"left", "right"}:
            raise ValueError("swing destination must specify robot-left or robot-right")
        actor_name = f"target_{side}"
        targets = {"target": _estimate(_actor(base, actor_name), actor_name)}
    elif name == "insert_peg":
        targets = {"peg": _free_peg_end(base), "box": _estimate(_actor(base, "box"), "box")}
    elif name in {"solve_pattern_direction", "solve_swingonto_with_direction"}:
        candidates = _actor(base, "targets_grid")
        estimates = tuple(_estimate(actor, f"grid_{i}") for i, actor in enumerate(candidates))
        if not estimates:
            raise RuntimeError("environment target grid is empty")
        # Preserve bench grid indices, including RouteStick's intermediate obstacles.
        return replace(request, targets={}, target_sets={"targets": estimates})
    else:
        return request
    return replace(request, targets=targets, target_sets={})


def bin_position(env: Any) -> list[float]:
    """Match Oracle Plus by reading BinFill's board-with-hole actor directly."""

    base = _base_env(env)
    actor = getattr(base, "board_with_hole", None)
    if actor is None:
        raise RuntimeError("environment bin target 'board_with_hole' is unavailable")
    return actor_position(actor, target_name="board_with_hole")


def button_position(env: Any, *, ordinal: str | None = None) -> list[float]:
    """Read the task's semantic button actor directly from the environment."""

    base = _base_env(env)
    task_name = type(base).__name__.casefold()
    normalized_ordinal = ordinal.strip().casefold() if ordinal else None
    if normalized_ordinal == "first":
        names = ("button_left", "button")
    elif normalized_ordinal == "second":
        names = ("button_right", "button")
    elif normalized_ordinal is not None:
        raise ValueError(f"unsupported button ordinal: {ordinal!r}")
    else:
        # Oracle Plus uses button_left for ButtonUnmask variants and button for
        # the other single-button tasks. Keep the same preference and fallback.
        names = (
            ("button_left", "button")
            if "buttonunmask" in task_name
            else ("button", "button_left")
        )
    for name in names:
        actor = getattr(base, name, None)
        if actor is not None:
            return actor_position(actor, target_name=name)
    raise RuntimeError(
        f"environment button target is unavailable; tried {list(names)}"
    )
