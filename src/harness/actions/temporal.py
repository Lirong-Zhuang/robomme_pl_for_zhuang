"""Oracle-style action whose primary effect is advancing simulator time."""

from __future__ import annotations

from typing import Any


def solve_hold_abs_timestep(env: Any, planner: Any, *, absolute_timestep: int) -> Any:
    """Port Oracle ``solve_hold_obj_absTimestep`` with the same absolute timing."""

    if not isinstance(absolute_timestep, int) or isinstance(absolute_timestep, bool):
        raise ValueError("absolute_timestep must be an integer")
    while int(getattr(env, "elapsed_steps", 0)) < absolute_timestep:
        planner.close_gripper()
    return None
