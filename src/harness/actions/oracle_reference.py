"""Auditable mapping from catalog actions to their Oracle/Oracle Plus sources.

These references are the behavioral specifications for the new implementations.
Port the referenced waypoint order and constants first; only replace simulator
actor geometry reads with :class:`ObjectEstimate` values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OracleReference:
    public_name: str
    source_file: str
    source_symbol: str
    adaptation: str


_PLANNER_FUNCTIONS = (
    "third_party/robomme_benchmark/src/robomme/robomme_env/utils/"
    "subgoal_planner_func.py"
)
_ORACLE_OPTIONS = (
    "third_party/robomme_benchmark/src/robomme/robomme_env/utils/vqa_options.py"
)


ORACLE_ACTION_REFERENCES: dict[str, OracleReference] = {
    "solve_pickup": OracleReference(
        "solve_pickup",
        _PLANNER_FUNCTIONS,
        "solve_pickup",
        "Replace obj pose/OBB reads with the grounded object estimate.",
    ),
    "solve_put_into_bin": OracleReference(
        "solve_put_into_bin",
        _PLANNER_FUNCTIONS,
        "solve_putonto_whenhold_binspecial",
        "Replace board_with_hole pose reads with the grounded container estimate.",
    ),
    "solve_button": OracleReference(
        "solve_button",
        _PLANNER_FUNCTIONS,
        "solve_button",
        "Replace button actor pose reads with the grounded button estimate.",
    ),
    "grasp_and_lift_peg": OracleReference(
        "grasp_and_lift_peg",
        _PLANNER_FUNCTIONS,
        "grasp_and_lift_peg_side",
        "Ground the whole peg for MoveCube and use a deterministic centered-grasp "
        "hook orientation without endpoint selection.",
    ),
    "grasp_and_lift_peg_side": OracleReference(
        "grasp_and_lift_peg_side",
        _PLANNER_FUNCTIONS,
        "grasp_and_lift_peg_side",
        "Replace peg-end actor pose reads with the grounded peg estimate.",
    ),
    "solve_push_to_target_with_peg": OracleReference(
        "solve_push_to_target_with_peg",
        _PLANNER_FUNCTIONS,
        "solve_push_to_target_with_peg",
        "Replace cube/goal actor pose reads; derive direction from grounded positions "
        "and map the task-specific centered peg grasp to a deterministic legacy "
        "MoveCube obj_flag.",
    ),
    "solve_push_to_target": OracleReference(
        "solve_push_to_target",
        _PLANNER_FUNCTIONS,
        "solve_push_to_target",
        "Replace cube/goal actor pose reads with grounded estimates.",
    ),
    "solve_putonto_target": OracleReference(
        "solve_putonto_target",
        _PLANNER_FUNCTIONS,
        "solve_putonto_whenhold",
        "Replace target actor pose reads with the grounded target estimate.",
    ),
    "solve_putdown": OracleReference(
        "solve_putdown",
        _PLANNER_FUNCTIONS,
        "solve_putdown_whenhold",
        "No semantic actor is needed; preserve the TCP-relative release motion.",
    ),
    "solve_button_ready": OracleReference(
        "solve_button_ready",
        _PLANNER_FUNCTIONS,
        "solve_button_ready",
        "Replace button actor pose reads with the grounded button estimate.",
    ),
    "solve_hold_abs_timestep": OracleReference(
        "solve_hold_abs_timestep",
        _PLANNER_FUNCTIONS,
        "solve_hold_obj_absTimestep",
        "Preserve absolute environment-timestep semantics.",
    ),
    "solve_pickup_bin": OracleReference(
        "solve_pickup_bin",
        _PLANNER_FUNCTIONS,
        "solve_pickup_bin",
        "Replace container actor pose/OBB reads with the grounded estimate.",
    ),
    "solve_swingonto_whenhold": OracleReference(
        "solve_swingonto_whenhold",
        _PLANNER_FUNCTIONS,
        "solve_swingonto_whenhold",
        "Replace target actor pose reads with the grounded estimate.",
    ),
    "insert_peg": OracleReference(
        "insert_peg",
        _PLANNER_FUNCTIONS,
        "insert_peg",
        "Replace env.box/insert_obj poses; resolve left/right from subgoal word tokens, "
        "map saved black/white pickup state to InsertPeg's legacy obj_flag, and preserve "
        "cut_retreat behavior.",
    ),
    "solve_pattern_direction": OracleReference(
        "solve_pattern_direction",
        _ORACLE_OPTIONS,
        "_options_patternlock._solve_direction + solve_swingonto",
        "Replace grid actor positions with perceived target positions.",
    ),
    "solve_swingonto_with_direction": OracleReference(
        "solve_swingonto_with_direction",
        _ORACLE_OPTIONS,
        "_options_routestick._solve_route + solve_swingonto_withDirection",
        "Replace route actor positions with perceived targets; preserve Bezier/IK motion.",
    ),
}
