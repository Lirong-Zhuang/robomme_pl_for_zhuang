from __future__ import annotations

import pytest

from oracle_plus.move_cube_plan import STRATEGY_EVIDENCE
from oracle_plus.move_cube_plan import STRATEGY_PLANS
from oracle_plus.move_cube_plan import gate_move_cube_followup
from oracle_plus.move_cube_plan import lock_move_cube_plan
from oracle_plus.prompts import get_prompt
from oracle_plus.prompts import resolve_action_name
from oracle_plus.vlm import SubgoalDecision


@pytest.mark.parametrize(("strategy", "subgoals"), STRATEGY_PLANS.items())
def test_move_cube_initial_strategy_locks_a_canonical_plan(
    strategy: str,
    subgoals: tuple[str, ...],
) -> None:
    model_decision = SubgoalDecision(
        completed=False,
        next_subgoal=subgoals[0],
        point=None,
        description="Visible temporal object-motion evidence.",
        subgoal_sequence=subgoals,
        strategy=strategy,
        evidence=STRATEGY_EVIDENCE[strategy],
    )

    locked, canonical = lock_move_cube_plan(
        model_decision,
        resolve_action=resolve_action_name,
    )

    assert locked.strategy == strategy
    assert locked.subgoals == subgoals
    assert canonical.next_subgoal == subgoals[0]
    assert canonical.subgoal_sequence == subgoals


def test_move_cube_rejects_strategy_and_sequence_mismatch() -> None:
    with pytest.raises(ValueError, match="requires actions"):
        lock_move_cube_plan(
            SubgoalDecision(
                completed=False,
                next_subgoal="pick up the cube",
                point=None,
                description="The cube appears to be carried.",
                subgoal_sequence=STRATEGY_PLANS["B_CUBE_PICK_PLACE"],
                strategy="C_PEG_HOOK",
                evidence=STRATEGY_EVIDENCE["C_PEG_HOOK"],
            ),
            resolve_action=resolve_action_name,
        )


def test_move_cube_rejects_strategy_and_evidence_mismatch() -> None:
    subgoals = STRATEGY_PLANS["C_PEG_HOOK"]
    with pytest.raises(ValueError, match="requires evidence"):
        lock_move_cube_plan(
            SubgoalDecision(
                completed=False,
                next_subgoal=subgoals[0],
                point=None,
                description="The cube, rather than the peg, appears lifted.",
                subgoal_sequence=subgoals,
                strategy="C_PEG_HOOK",
                evidence=STRATEGY_EVIDENCE["B_CUBE_PICK_PLACE"],
            ),
            resolve_action=resolve_action_name,
        )


def test_move_cube_rejects_pick_place_without_vertical_lift_evidence() -> None:
    subgoals = STRATEGY_PLANS["B_CUBE_PICK_PLACE"]
    with pytest.raises(ValueError, match="requires evidence"):
        lock_move_cube_plan(
            SubgoalDecision(
                completed=False,
                next_subgoal=subgoals[0],
                point=None,
                description=(
                    "The cube overlaps the fingers and moves horizontally with them."
                ),
                subgoal_sequence=subgoals,
                strategy="B_CUBE_PICK_PLACE",
                evidence={
                    "carried_object": "cube",
                    "cube_leaves_tabletop": True,
                    "peg_moves_with_gripper": False,
                },
            ),
            resolve_action=resolve_action_name,
        )


def test_move_cube_followup_cannot_jump_to_another_strategy() -> None:
    subgoals = STRATEGY_PLANS["C_PEG_HOOK"]
    locked, _ = lock_move_cube_plan(
        SubgoalDecision(
            completed=False,
            next_subgoal=subgoals[0],
            point=None,
            description="The peg is held by the gripper.",
            subgoal_sequence=subgoals,
            strategy="C_PEG_HOOK",
            evidence=STRATEGY_EVIDENCE["C_PEG_HOOK"],
        ),
        resolve_action=resolve_action_name,
    )
    wrong_strategy_decision = SubgoalDecision(
        completed=False,
        next_subgoal="pick up the cube",
        point=None,
        description="The current peg pickup is not visibly complete.",
    )

    gated, action_name, next_index, gate = gate_move_cube_followup(
        wrong_strategy_decision,
        locked_plan=locked,
        active_index=0,
        active_subgoal=subgoals[0],
        resolve_action=resolve_action_name,
    )

    assert gated.next_subgoal == "pick up the peg"
    assert gated.completed is False
    assert action_name == "grasp_and_lift_peg_side"
    assert next_index == 0
    assert gate["overrode_model"] is True


def test_move_cube_completion_advances_only_to_locked_successor() -> None:
    subgoals = STRATEGY_PLANS["C_PEG_HOOK"]
    locked, _ = lock_move_cube_plan(
        SubgoalDecision(
            completed=False,
            next_subgoal=subgoals[0],
            point=None,
            description="The peg is held by the gripper.",
            subgoal_sequence=subgoals,
            strategy="C_PEG_HOOK",
            evidence=STRATEGY_EVIDENCE["C_PEG_HOOK"],
        ),
        resolve_action=resolve_action_name,
    )

    gated, action_name, next_index, _ = gate_move_cube_followup(
        SubgoalDecision(
            completed=True,
            next_subgoal="place the cube onto the target",
            point=None,
            description="The peg is now held above the tabletop.",
        ),
        locked_plan=locked,
        active_index=0,
        active_subgoal=subgoals[0],
        resolve_action=resolve_action_name,
    )

    assert gated.next_subgoal == "hook the cube to the target with the peg"
    assert gated.completed is True
    assert action_name == "solve_push_to_target_with_peg"
    assert next_index == 1


def test_move_cube_final_step_repeats_until_environment_stops() -> None:
    subgoals = STRATEGY_PLANS["A_GRIPPER_PUSH"]
    locked, _ = lock_move_cube_plan(
        SubgoalDecision(
            completed=False,
            next_subgoal=subgoals[0],
            point=None,
            description="The empty closed gripper directly contacts the cube.",
            subgoal_sequence=subgoals,
            strategy="A_GRIPPER_PUSH",
            evidence=STRATEGY_EVIDENCE["A_GRIPPER_PUSH"],
        ),
        resolve_action=resolve_action_name,
    )

    gated, action_name, next_index, gate = gate_move_cube_followup(
        SubgoalDecision(
            completed=True,
            next_subgoal="STOP",
            point=None,
            description="The cube appears to have reached the target.",
        ),
        locked_plan=locked,
        active_index=0,
        active_subgoal=subgoals[0],
        resolve_action=resolve_action_name,
    )

    assert gated.next_subgoal == subgoals[0]
    assert gated.completed is False
    assert action_name == "solve_push_to_target"
    assert next_index == 0
    assert "environment is still ongoing" in str(gate["reason"])


def test_move_cube_prompt_demands_object_identity_evidence() -> None:
    prompt = get_prompt(
        "MoveCube",
        include_description=True,
        include_wrist=True,
    ).system_prompt

    assert '"strategy": "A_GRIPPER_PUSH, B_CUBE_PICK_PLACE, or C_PEG_HOOK"' in prompt
    assert "A closed gripper alone is NEVER evidence" in prompt
    assert "if the gripper carries the peg, it is C_PEG_HOOK" in prompt


def test_move_cube_strategy_and_evidence_parse_from_vlm_json() -> None:
    decision = SubgoalDecision.from_mapping(
        {
            "description": "The long peg moves with the gripper while the cube stays on the table.",
            "strategy": "C_PEG_HOOK",
            "evidence": STRATEGY_EVIDENCE["C_PEG_HOOK"],
            "subgoal_sequence": list(STRATEGY_PLANS["C_PEG_HOOK"]),
            "completed": False,
            "next_subgoal": "pick up the peg",
            "point": None,
        },
        include_description=True,
    )

    assert decision.strategy == "C_PEG_HOOK"
    assert decision.evidence == STRATEGY_EVIDENCE["C_PEG_HOOK"]
