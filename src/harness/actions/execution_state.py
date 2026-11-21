"""Small trusted execution-state values shared by consecutive actions."""

from __future__ import annotations

from typing import Any


_PEG_COLOR_ATTRIBUTE = "_agentic_grasped_peg_endpoint_color"
_VALID_PEG_COLORS = {"black", "white"}
_MOVE_CUBE_CENTER_GRASP = "center"


def _state_owner(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def remember_grasped_peg_endpoint_color(env: Any, color: str) -> None:
    """Record the visibly grounded peg half selected by the pickup action."""

    normalized = color.strip().lower() if isinstance(color, str) else None
    if normalized not in _VALID_PEG_COLORS:
        raise ValueError("peg endpoint color must be 'black' or 'white'")
    setattr(_state_owner(env), _PEG_COLOR_ATTRIBUTE, normalized)


def remember_move_cube_center_peg_grasp(env: Any) -> None:
    """Record MoveCube's task-specific whole-peg center grasp."""

    setattr(_state_owner(env), _PEG_COLOR_ATTRIBUTE, _MOVE_CUBE_CENTER_GRASP)


def require_grasped_peg_endpoint_color(env: Any) -> str:
    """Return the black/white endpoint state produced by peg pickup."""

    value = getattr(_state_owner(env), _PEG_COLOR_ATTRIBUTE, None)
    if value not in _VALID_PEG_COLORS:
        raise RuntimeError(
            "peg action requires a preceding successful pickup that records the "
            "black or white endpoint"
        )
    return str(value)


def move_cube_obj_flag_from_grasped_endpoint(env: Any) -> int:
    """Map endpoint or task-specific center grasp to MoveCube's Oracle flag."""

    value = getattr(_state_owner(env), _PEG_COLOR_ATTRIBUTE, None)
    if value == _MOVE_CUBE_CENTER_GRASP:
        # A centered grasp makes the two peg ends interchangeable for MoveCube;
        # retain one deterministic legacy hook orientation.
        return 1
    return 1 if require_grasped_peg_endpoint_color(env) == "black" else -1


def insert_peg_obj_flag_from_grasped_endpoint(env: Any) -> int:
    """Map InsertPeg's grasped endpoint convention to its legacy Oracle flag."""

    return -1 if require_grasped_peg_endpoint_color(env) == "black" else 1
