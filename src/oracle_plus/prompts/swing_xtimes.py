"""SwingXtimes-specific prompts and deterministic action mapping."""

from __future__ import annotations

import re

from .base import FOLLOWUP_USER_TEMPLATE
from .base import FOLLOWUP_USER_TEMPLATE_OVERALL_ONLY
from .base import INITIAL_USER_TEMPLATE
from .base import INITIAL_USER_TEMPLATE_OVERALL_ONLY
from .base import TaskPromptSpec
from .base import build_system_prompt


TASK_NAME = "SwingXtimes"

TARGET_SIDE_REMINDER = """Target-side reminder for this call:
- Always name the robot-right target as "robot-right target (camera-left disk)".
- Always name the robot-left target as "robot-left target (camera-right disk)".
These target identities stay fixed throughout the episode. Use the overall image,
not the wrist image, to identify left versus right. Use these exact dual-frame
labels in descriptions, subgoal_sequence, and next_subgoal; never use a bare
"left-side" or "right-side" target/disk label."""


def _with_target_side_reminder(template: str) -> str:
    marker = "Start directly with the JSON object. Output JSON only."
    if marker not in template:
        raise ValueError("SwingXtimes user template is missing the JSON output marker")
    return template.replace(marker, f"{TARGET_SIDE_REMINDER}\n{marker}")

NOTES = """1. The targets are two white-grey disks.
2. When the robot reaches a target, that target turns red.
The red highlight is temporary; the previously reached target returns to grey.
Read the observations chronologically from oldest to newest and use the newest
frame to decide which target is currently red. A target that is red only in an
older frame does not complete the current subgoal. Brief highlight overlap may
occur, so also require the held cube to be visibly over the currently named
target in the newest observation.
3. Right and left are defined in the robot base frame: the robot-right target
appears on the camera's left, and the robot-left target appears on camera-right.
Always call them "robot-right target (camera-left disk)" and "robot-left target
(camera-right disk)" respectively. Use these complete dual-frame labels in every
description, subgoal_sequence, and next_subgoal. Never use bare "left-side" or
"right-side" to describe a target or disk.
4. Use the task goal to determine the cube color and number of right-left swing
pairs. Pick the requested cube once, then alternate right-side followed by
left-side for every repetition.
5. Count a target visit only when the held cube visibly reaches that target.
Continuous recent frames showing one visit are not multiple visits.
6. Pickup and every move-to-target action require point=[y, x] in the latest
overall image. Release, button press, and STOP require point=null.
7. After all requested pairs, release the cube onto the tabletop and then press
the button. Both target disks are part of the tabletop: a released cube resting
on either red target disk counts as successfully placed on the table. Do not
require the cube to move away from the final target disk.
8. The release subgoal is complete when the cube no longer moves together with
the gripper, remains low and stationary on the tabletop, and the gripper is open
or visibly lifted away. Compare the fixed subgoal-init observation with the
newest observations. Do not repeat release merely because the cube remains on a
red target disk after the gripper has opened and moved upward."""

SUBGOALS = """- pick up the [red/blue/green] cube
- move to the top of the robot-right target (camera-left disk) for the [first/second/third/...] time
- move to the top of the robot-left target (camera-right disk) for the [first/second/third/...] time
- release the [red/blue/green] cube and leave it resting on the tabletop
- press the button
- STOP, only after the button press completes"""

COMBINED_EXAMPLE = """If the task goal requests two swings with the green cube, use:
1. pick up the green cube
2. move to the top of the robot-right target (camera-left disk) for the first time
3. move to the top of the robot-left target (camera-right disk) for the first time
4. move to the top of the robot-right target (camera-left disk) for the second time
5. move to the top of the robot-left target (camera-right disk) for the second time
6. release the green cube and leave it resting on the tabletop
7. press the button"""


SUBGOAL_ACTION_RULES: tuple[tuple[str, str], ...] = (
    (r"\bpick\s+up\b.*\bcube\b", "solve_pickup"),
    (r"\b(?:move|swing)\b.*\btarget\b", "solve_swingonto_whenhold"),
    (
        r"\b(?:put|place|release)\b.*\bcube\b.*\b(?:on|onto|resting)\b.*\btable(?:top)?\b",
        "solve_putdown_whenhold",
    ),
    (r"\bpress\b.*\bbutton\b", "solve_button"),
)

POINT_ACTION_NAMES: tuple[str, ...] = (
    "solve_pickup",
    "solve_swingonto_whenhold",
)


def _prompt(*, include_description: bool, include_wrist: bool) -> TaskPromptSpec:
    suffix = "with_description" if include_description else "without_description"
    view = "paired" if include_wrist else "overall_only"
    point_rules = (
        "5. For pickup and move-to-target subgoals, output the target center in "
        "the latest overall image\n   as point=[y, x], using original pixel "
        "coordinates from the top-left.\n6. For release, button press, and STOP, "
        "output point as null."
    )
    return TaskPromptSpec(
        task=TASK_NAME,
        version=f"swing_xtimes_oracle_planner_v4_semantic_target_{view}_{suffix}",
        include_description=include_description,
        include_wrist=include_wrist,
        system_prompt=build_system_prompt(
            subgoals=SUBGOALS,
            notes=NOTES,
            example=COMBINED_EXAMPLE,
            include_description=include_description,
            point_rules=point_rules,
        ),
        initial_user_template=(
            _with_target_side_reminder(
                INITIAL_USER_TEMPLATE
                if include_wrist
                else INITIAL_USER_TEMPLATE_OVERALL_ONLY
            )
        ),
        followup_user_template=(
            _with_target_side_reminder(
                FOLLOWUP_USER_TEMPLATE
                if include_wrist
                else FOLLOWUP_USER_TEMPLATE_OVERALL_ONLY
            )
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
