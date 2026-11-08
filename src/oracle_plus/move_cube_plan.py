"""Lock MoveCube imitation to the strategy inferred from its demonstration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .vlm import SubgoalDecision


STRATEGY_PLANS: dict[str, tuple[str, ...]] = {
    "A_GRIPPER_PUSH": ("close gripper and push the cube to the target",),
    "B_CUBE_PICK_PLACE": (
        "pick up the cube",
        "place the cube onto the target",
    ),
    "C_PEG_HOOK": (
        "pick up the peg",
        "hook the cube to the target with the peg",
    ),
}

STRATEGY_EVIDENCE: dict[str, dict[str, str | bool]] = {
    "A_GRIPPER_PUSH": {
        "carried_object": "neither",
        "cube_leaves_tabletop": False,
        "cube_has_visible_table_clearance": False,
        "cube_moves_with_vertical_lift": False,
        "peg_moves_with_gripper": False,
        "peg_visibly_held": False,
    },
    "B_CUBE_PICK_PLACE": {
        "carried_object": "cube",
        "cube_leaves_tabletop": True,
        "cube_has_visible_table_clearance": True,
        "cube_moves_with_vertical_lift": True,
        "peg_moves_with_gripper": False,
        "peg_visibly_held": False,
    },
    "C_PEG_HOOK": {
        "carried_object": "peg",
        "cube_leaves_tabletop": False,
        "cube_has_visible_table_clearance": False,
        "cube_moves_with_vertical_lift": False,
        "peg_moves_with_gripper": True,
        "peg_visibly_held": True,
    },
}


@dataclass(frozen=True)
class LockedMoveCubePlan:
    strategy: str
    subgoals: tuple[str, ...]
    action_names: tuple[str, ...]


def canonicalize_move_cube_strategy(
    decision: SubgoalDecision,
) -> SubgoalDecision:
    """Convert a classification-only audit result into a trusted plan shape."""

    if decision.strategy is None:
        raise ValueError("MoveCube strategy audit must include strategy")
    strategy = decision.strategy.strip().upper()
    try:
        subgoals = STRATEGY_PLANS[strategy]
        evidence = STRATEGY_EVIDENCE[strategy]
    except KeyError as error:
        raise ValueError(
            f"Unknown MoveCube strategy {decision.strategy!r}; "
            f"expected one of {tuple(STRATEGY_PLANS)}"
        ) from error
    return SubgoalDecision(
        completed=False,
        next_subgoal=subgoals[0],
        point=None,
        description=decision.description,
        subgoal_sequence=subgoals,
        strategy=strategy,
        evidence=dict(evidence),
    )


def lock_move_cube_plan(
    decision: SubgoalDecision,
    *,
    resolve_action: Callable[[str, str], str | None],
) -> tuple[LockedMoveCubePlan, SubgoalDecision]:
    """Validate and canonicalize the initial demonstration-derived plan."""

    if decision.strategy is None:
        raise ValueError("MoveCube initial decision must include strategy")
    strategy = decision.strategy.strip().upper()
    try:
        canonical_subgoals = STRATEGY_PLANS[strategy]
    except KeyError as error:
        raise ValueError(
            f"Unknown MoveCube strategy {decision.strategy!r}; "
            f"expected one of {tuple(STRATEGY_PLANS)}"
        ) from error
    if decision.completed:
        raise ValueError("MoveCube initial decision must set completed=false")
    if not decision.subgoal_sequence:
        raise ValueError("MoveCube initial decision must include subgoal_sequence")
    if decision.evidence is None:
        raise ValueError("MoveCube initial decision must include evidence")
    expected_evidence = STRATEGY_EVIDENCE[strategy]
    reported_evidence = {
        key: decision.evidence.get(key) for key in expected_evidence
    }
    if reported_evidence != expected_evidence:
        raise ValueError(
            f"MoveCube strategy {strategy} requires evidence {expected_evidence}, "
            f"but the model reported {reported_evidence}"
        )

    resolved_expected_actions = tuple(
        resolve_action("MoveCube", subgoal) for subgoal in canonical_subgoals
    )
    if any(action is None for action in resolved_expected_actions):
        raise AssertionError("A canonical MoveCube subgoal unexpectedly resolved to STOP")
    expected_actions = tuple(
        action for action in resolved_expected_actions if action is not None
    )
    planned_actions = tuple(
        resolve_action("MoveCube", subgoal)
        for subgoal in decision.subgoal_sequence
    )
    if planned_actions != expected_actions:
        raise ValueError(
            f"MoveCube strategy {strategy} requires actions {expected_actions}, "
            f"but subgoal_sequence resolves to {planned_actions}"
        )
    initial_action = resolve_action("MoveCube", decision.next_subgoal)
    if initial_action != expected_actions[0]:
        raise ValueError(
            f"MoveCube strategy {strategy} must start with {expected_actions[0]!r}, "
            f"not {initial_action!r}"
        )

    locked = LockedMoveCubePlan(
        strategy=strategy,
        subgoals=canonical_subgoals,
        action_names=expected_actions,
    )
    canonical_decision = SubgoalDecision(
        completed=False,
        next_subgoal=canonical_subgoals[0],
        point=None,
        description=decision.description,
        subgoal_sequence=canonical_subgoals,
        strategy=strategy,
        evidence=dict(expected_evidence),
    )
    return locked, canonical_decision


def gate_move_cube_followup(
    model_decision: SubgoalDecision,
    *,
    locked_plan: LockedMoveCubePlan,
    active_index: int,
    active_subgoal: str,
    resolve_action: Callable[[str, str], str | None],
) -> tuple[SubgoalDecision, str, int, dict[str, object]]:
    """Use the VLM only as a completion judge for the locked MoveCube plan."""

    if not 0 <= active_index < len(locked_plan.subgoals):
        raise IndexError(f"MoveCube plan index is out of range: {active_index}")
    expected_active_action = locked_plan.action_names[active_index]
    actual_active_action = resolve_action("MoveCube", active_subgoal)
    if actual_active_action != expected_active_action:
        raise RuntimeError(
            f"MoveCube locked plan expected active action "
            f"{expected_active_action!r}, found {actual_active_action!r}"
        )

    next_index = active_index
    reason = "model reported the current subgoal incomplete"
    if model_decision.completed:
        if active_index + 1 < len(locked_plan.subgoals):
            next_index = active_index + 1
            reason = "advanced to the deterministic successor in the locked plan"
        else:
            # A successful final primitive terminates through the environment
            # before this follow-up. If the environment is still ongoing, do
            # not let a visual early-stop claim turn into task success.
            reason = (
                "final subgoal was visually reported complete but the environment "
                "is still ongoing; kept the final locked subgoal active"
            )

    next_subgoal = locked_plan.subgoals[next_index]
    next_action = locked_plan.action_names[next_index]
    gated_decision = SubgoalDecision(
        completed=next_index != active_index,
        next_subgoal=next_subgoal,
        point=None,
        description=model_decision.description,
    )
    gate = {
        "strategy": locked_plan.strategy,
        "active_index": active_index,
        "next_index": next_index,
        "reported_completed": model_decision.completed,
        "reported_next_subgoal": model_decision.next_subgoal,
        "enforced_next_subgoal": next_subgoal,
        "overrode_model": gated_decision != model_decision,
        "reason": reason,
    }
    return gated_decision, next_action, next_index, gate
