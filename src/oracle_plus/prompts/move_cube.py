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
2. Classify the demonstration in this order: first test whether the peg is
carried; if not, test whether the cube is unambiguously lifted; otherwise select
direct gripper push. Do not classify from whether the fingers are open or closed:
(A_GRIPPER_PUSH) The closed gripper carries neither the cube nor the peg. The cube
stays on the tabletop and moves through direct contact with the empty gripper.
(B_CUBE_PICK_PLACE) The cube itself is visibly between the fingers, leaves the
tabletop, and moves rigidly together with the gripper before being placed.
(C_PEG_HOOK) The long thin peg is visibly between the fingers and moves rigidly
with the gripper before the cube moves. The cube itself stays on the tabletop and
is moved through contact with the held peg. This can look like pushing rather
than literal hooking; if the gripper carries the peg, it is C_PEG_HOOK.
3. Direct pushing can make the tabletop cube appear between or surrounded by
the fingers in a 2-D image, and the cube can move horizontally together with
the gripper. Neither cue is pickup evidence. Apparent image-y or size changes,
partial occlusion, and horizontal rigid co-motion are also not lift evidence.
4. Select B only if the temporal observations unambiguously show BOTH (a) a
visible air gap or exposed tabletop between the cube bottom and the table and
(b) the cube remaining attached while the gripper retreats vertically upward
away from the tabletop. Require these cues in at least two demonstration
observations. If either cue is absent or ambiguous, select A, not B.
5. Select C only if the long thin peg is visibly held between the fingers and
moves with the gripper before the cube moves. Otherwise peg_moves_with_gripper
and peg_visibly_held must both be false.
6. In the initial description, state the directly visible object-identity and
vertical-clearance evidence supporting the selected strategy. Do not turn the
chosen label into claimed evidence.
7. Set evidence consistently: A means carried_object=neither and every boolean
is false; B means carried_object=cube, the three cube lift booleans are true,
and both peg booleans are false; C means carried_object=peg, both peg booleans
are true, and all cube lift booleans are false.
8. Reproduce only the demonstrated strategy in the new execution. Do not mix
actions from different strategies.
9. If the active subgoal is not visibly complete, keep it active.
10. Every MoveCube action uses point=null; the task has fixed cube, target, and
peg objects."""

SUBGOALS = """- close gripper and push the cube to the target
- pick up the cube
- place the cube onto the target
- pick up the peg
- hook the cube to the target with the peg
- STOP, only after the demonstrated strategy has completed"""

COMBINED_EXAMPLE = """Classify by the object carried by the gripper:
A_GRIPPER_PUSH: the closed gripper is empty; neither cube nor peg is carried;
the cube stays on the table and moves through direct gripper contact. The cube
may look surrounded by the fingers and may move horizontally with them.
Plan: close gripper and push the cube to the target.
B_CUBE_PICK_PLACE: at least two observations visibly show table clearance under
the cube and show it following a vertical upward gripper retreat.
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
    "cube_has_visible_table_clearance": true or false,
    "cube_moves_with_vertical_lift": true or false,
    "peg_moves_with_gripper": true or false,
    "peg_visibly_held": true or false
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
    "cube_has_visible_table_clearance": true or false,
    "cube_moves_with_vertical_lift": true or false,
    "peg_moves_with_gripper": true or false,
    "peg_visibly_held": true or false
  },
  "subgoal_sequence": ["the exact ordered subgoals for that strategy"],
  "completed": false,
  "next_subgoal": "the first subgoal",
  "point": null
}
```"""


PICK_PLACE_AUDIT_VERSION = "move_cube_strategy_audit_v3_overall_only"

PICK_PLACE_AUDIT_SCHEMA_WITH_DESCRIPTION = """```json
{
  "description": "short visible evidence for the audited classification",
  "strategy": "A_GRIPPER_PUSH, B_CUBE_PICK_PLACE, or C_PEG_HOOK",
  "completed": false,
  "next_subgoal": "AUDIT_ONLY",
  "point": null
}
```"""

PICK_PLACE_AUDIT_SCHEMA_WITHOUT_DESCRIPTION = """```json
{
  "strategy": "A_GRIPPER_PUSH, B_CUBE_PICK_PLACE, or C_PEG_HOOK",
  "completed": false,
  "next_subgoal": "AUDIT_ONLY",
  "point": null
}
```"""


def build_pick_place_audit_messages(
    *,
    observation_frames: list[int],
    include_description: bool,
) -> list[dict[str, str]]:
    """Build an independent overall-view audit for a non-push candidate."""

    if len(observation_frames) < 2:
        raise ValueError("MoveCube pickup audit requires at least two demo images")
    schema = (
        PICK_PLACE_AUDIT_SCHEMA_WITH_DESCRIPTION
        if include_description
        else PICK_PLACE_AUDIT_SCHEMA_WITHOUT_DESCRIPTION
    )
    description_rule = (
        "Include a short description naming the observations that show or fail "
        "to show vertical table clearance."
        if include_description
        else "Do not output a description field."
    )
    observation_lines = "\n".join(
        (
            f"Demonstration overall observation {index} "
            f"(simulator frame {frame}): <image>"
        )
        for index, frame in enumerate(observation_frames, start=1)
    )
    system = f"""You are an independent skeptical visual auditor for a robot
demonstration. An earlier strategy proposal is deliberately withheld and must
not influence this audit. Inspect only the chronological overall-view images.

Classify exactly one strategy:
- A_GRIPPER_PUSH: the cube remains supported by the tabletop and is translated
  horizontally by direct gripper contact.
- B_CUBE_PICK_PLACE: at least two observations unambiguously show visible air or
  exposed tabletop below the cube AND the same cube following a vertical upward
  gripper retreat while still attached.
- C_PEG_HOOK: the long thin peg is visibly held and moves with the gripper before
  it contacts and moves the cube.

Audit rules:
1. Apply the same evidence burden to all three strategies. Do not default to push,
   pickup, or hook, and do not infer the answer from sequence duration alone.
2. A cube appearing between the fingers, being partly occluded, changing image-y
   or apparent size, or moving horizontally with the gripper is not lift evidence.
3. If the cube stops on the purple disk while the gripper subsequently rises
   away, the cube is not following that vertical rise; classify the motion as A.
4. Conversely, do not classify A when multiple observations show the cube itself
   elevated with visible table clearance and still attached during vertical and
   horizontal gripper motion; that is B.
5. Classify C only when the visibly carried object is the long thin peg. A peg
   merely lying near, behind, or crossing the gripper in projection is not held.
   If the compact cube, rather than the elongated peg, moves with the gripper,
   classify B rather than C.
6. This is classification only. Do not propose approach, grasp, push, place, or
   any other executable subgoal. Set next_subgoal exactly to AUDIT_ONLY.
7. {description_rule}

Return JSON only using this schema:
{schema}
"""
    user = f"""Audit the candidate pickup using these demonstration images, oldest
to newest. There is no new-execution image in this audit.

{observation_lines}

Return only the audited strategy record. Start directly with the JSON object."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


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
        version=f"move_cube_oracle_planner_demo_v6_{view}_{suffix}",
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
