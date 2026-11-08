"""MoveCube-specific demonstration prompt and deterministic action mapping."""

from __future__ import annotations

import re

from .base import FOLLOWUP_USER_TEMPLATE
from .base import FOLLOWUP_USER_TEMPLATE_OVERALL_ONLY
from .base import INITIAL_DEMONSTRATION_USER_TEMPLATE
from .base import INITIAL_DEMONSTRATION_USER_TEMPLATE_OVERALL_ONLY
from .base import TaskPromptSpec
from .base import build_system_prompt


TASK_NAME = "MoveCube"

NOTES = """1. The purple disk is the target. The small block is the cube. The
long thin object is the peg.
2. Infer exactly one strategy from temporal object motion, not from whether the
gripper fingers are open or closed:
(A_GRIPPER_PUSH) The closed gripper carries neither the cube nor the peg. The cube
stays on the tabletop and moves through direct contact with the empty gripper.
(B_CUBE_PICK_PLACE) The cube itself is visibly between the fingers, leaves the
tabletop, and moves rigidly together with the gripper before being placed.
(C_PEG_HOOK) The long thin peg is visibly between the fingers and moves rigidly
with the gripper before the cube moves. The cube itself stays on the tabletop and
is moved through contact with the held peg. This can look like pushing rather
than literal hooking; if the gripper carries the peg, it is C_PEG_HOOK.
3. A closed gripper alone is NEVER evidence that the cube was picked up. Decide
what is attached to the gripper: cube means B, peg means C, neither means A.
4. In the initial description, state the directly visible object-identity and
motion evidence supporting the selected strategy. Do not merely say that the
gripper is closed.
5. Set evidence consistently: A means carried_object=neither,
cube_leaves_tabletop=false, peg_moves_with_gripper=false; B means cube, true,
false; C means peg, false, true.
6. Reproduce only the demonstrated strategy in the new execution. Do not mix
actions from different strategies.
7. If the active subgoal is not visibly complete, keep it active.
8. Every MoveCube action uses point=null; the task has fixed cube, target, and
peg objects."""

SUBGOALS = """- close gripper and push the cube to the target
- pick up the cube
- place the cube onto the target
- pick up the peg
- hook the cube to the target with the peg
- STOP, only after the demonstrated strategy has completed"""

COMBINED_EXAMPLE = """Classify by the object carried by the gripper:
A_GRIPPER_PUSH: the closed gripper is empty; neither cube nor peg is carried;
the cube stays on the table and moves through direct gripper contact.
Plan: close gripper and push the cube to the target.
B_CUBE_PICK_PLACE: the cube is between the fingers, leaves the table, and moves
together with the gripper.
Plan: pick up the cube; place the cube onto the target.
C_PEG_HOOK: the long peg is between the fingers and moves with the gripper; the
cube stays on the table and is moved through contact with the held peg.
Plan: pick up the peg; hook the cube to the target with the peg."""

POINT_RULES = """5. For every MoveCube subgoal and for STOP, output point as null.
6. Never output point=[y, x] for this task."""


INITIAL_SCHEMA_WITH_DESCRIPTION = """```json
{
  "description": "visible temporal evidence identifying what the gripper carries; at most 50 English words and two short sentences",
  "strategy": "A_GRIPPER_PUSH, B_CUBE_PICK_PLACE, or C_PEG_HOOK",
  "evidence": {
    "carried_object": "neither, cube, or peg",
    "cube_leaves_tabletop": true or false,
    "peg_moves_with_gripper": true or false
  },
  "subgoal_sequence": ["the exact ordered subgoals for that strategy"],
  "completed": false,
  "next_subgoal": "the first subgoal",
  "point": null
}
```"""


INITIAL_SCHEMA_WITHOUT_DESCRIPTION = """```json
{
  "strategy": "A_GRIPPER_PUSH, B_CUBE_PICK_PLACE, or C_PEG_HOOK",
  "evidence": {
    "carried_object": "neither, cube, or peg",
    "cube_leaves_tabletop": true or false,
    "peg_moves_with_gripper": true or false
  },
  "subgoal_sequence": ["the exact ordered subgoals for that strategy"],
  "completed": false,
  "next_subgoal": "the first subgoal",
  "point": null
}
```"""


SUBGOAL_ACTION_RULES: tuple[tuple[str, str], ...] = (
    (r"\bhook\b.*\bcube\b.*\btarget\b.*\bpeg\b", "solve_push_to_target_with_peg"),
    (r"\b(?:close\b.*\bgripper\b.*)?\bpush\b.*\bcube\b.*\btarget\b", "solve_push_to_target"),
    (r"\bpick\s+up\b.*\bcube\b", "solve_pickup"),
    (r"\b(?:place|put)\b.*\bcube\b.*\b(?:onto|on)\b.*\btarget\b", "solve_putonto_whenhold"),
    (r"\bpick\s+up\b.*\bpeg\b", "grasp_and_lift_peg_side"),
)

POINT_ACTION_NAMES: tuple[str, ...] = ()


def _prompt(*, include_description: bool, include_wrist: bool) -> TaskPromptSpec:
    suffix = "with_description" if include_description else "without_description"
    view = "paired" if include_wrist else "overall_only"
    return TaskPromptSpec(
        task=TASK_NAME,
        version=f"move_cube_oracle_planner_demo_v2_{view}_{suffix}",
        include_description=include_description,
        include_wrist=include_wrist,
        initial_observation_mode="demonstration",
        system_prompt=build_system_prompt(
            subgoals=SUBGOALS,
            notes=NOTES,
            example=COMBINED_EXAMPLE,
            include_description=include_description,
            point_rules=POINT_RULES,
            initial_schema=(
                INITIAL_SCHEMA_WITH_DESCRIPTION
                if include_description
                else INITIAL_SCHEMA_WITHOUT_DESCRIPTION
            ),
            description_rule_override=(
                "Include a non-empty description of no more than 50 English words "
                "and at most two short sentences. On the initial call, state "
                "directly visible temporal evidence identifying whether the "
                "gripper carries the cube, the peg, or neither. On follow-up "
                "calls, state only visible completion evidence for the current "
                "subgoal."
                if include_description
                else None
            ),
        ),
        initial_user_template=(
            INITIAL_DEMONSTRATION_USER_TEMPLATE
            if include_wrist
            else INITIAL_DEMONSTRATION_USER_TEMPLATE_OVERALL_ONLY
        ),
        followup_user_template=(
            FOLLOWUP_USER_TEMPLATE
            if include_wrist
            else FOLLOWUP_USER_TEMPLATE_OVERALL_ONLY
        ),
    )


PROMPT_WITH_DESCRIPTION = _prompt(include_description=True, include_wrist=True)
PROMPT_WITHOUT_DESCRIPTION = _prompt(include_description=False, include_wrist=True)
PROMPT_OVERALL_ONLY_WITH_DESCRIPTION = _prompt(
    include_description=True, include_wrist=False
)
PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION = _prompt(
    include_description=False, include_wrist=False
)


def match_action_name(subgoal: str) -> str | None:
    text = " ".join(subgoal.casefold().replace("_", " ").split())
    for pattern, action_name in SUBGOAL_ACTION_RULES:
        if re.search(pattern, text):
            return action_name
    return None
