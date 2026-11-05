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

NOTES_PAIRED = """1. The targets are two white-grey disks.
2. When the robot reaches a target, that target changes to a red-and-white
appearance.
3. Right and left are defined in the robot base frame: the robot-right target
appears on the camera's left, and the robot-left target appears on camera-right.
4. Use the task goal to determine the cube color and number of right-left swing
pairs. Pick the requested cube once, then alternate right-side followed by
left-side for every repetition.
5. For a move-to-target subgoal, focus primarily on the wrist images and compare
the subgoal-init wrist image with the recent wrist images. Count the visit when
the recent wrist view is broadly consistent with both the held cube being roughly
above the named target and the target changing from its original grey appearance
to a red-and-white appearance. Apply this check liberally: approximate alignment
and a partial color change are enough, and tolerate blur, lighting differences,
and the cube or gripper partly hiding the disk. Do not require exact centering or
a perfectly unobstructed target.
Continuous recent frames showing one visit are not multiple visits.
6. Pickup and every move-to-target action require point=[y, x] in the latest
overall image. Putdown, button press, and STOP require point=null.
7. After all requested pairs, put the cube on the table and press the button."""

NOTES_OVERALL_ONLY = """1. The targets are two white-grey disks.
2. When the robot reaches a target, that target changes to a red-and-white
appearance.
3. Right and left are defined in the robot base frame: the robot-right target
appears on the camera's left, and the robot-left target appears on camera-right.
4. Use the task goal to determine the cube color and number of right-left swing
pairs. Pick the requested cube once, then alternate right-side followed by
left-side for every repetition.
5. For a move-to-target subgoal, compare the subgoal-init overall image with the
recent overall images. Count the visit when the recent view is broadly consistent
with both the held cube being roughly above the named target and the target
changing from its original grey appearance to a red-and-white appearance. Apply
this check liberally: approximate alignment and a partial color change are enough,
and tolerate blur, lighting differences, and partial occlusion. Do not require
exact centering or a perfectly unobstructed target.
Continuous recent frames showing one visit are not multiple visits.
6. Pickup and every move-to-target action require point=[y, x] in the latest
overall image. Putdown, button press, and STOP require point=null.
7. After all requested pairs, put the cube on the table and press the button."""

SUBGOALS = """- pick up the [red/blue/green] cube
- move to the top of the right-side target for the [first/second/third/...] time
- move to the top of the left-side target for the [first/second/third/...] time
- put the [red/blue/green] cube on the table
- press the button
- STOP, only after the button press completes"""

COMBINED_EXAMPLE = """If the task goal requests two swings with the green cube, use:
1. pick up the green cube
2. move to the top of the right-side target for the first time
3. move to the top of the left-side target for the first time
4. move to the top of the right-side target for the second time
5. move to the top of the left-side target for the second time
6. put the green cube on the table
7. press the button"""


SUBGOAL_ACTION_RULES: tuple[tuple[str, str], ...] = (
    (r"\bpick\s+up\b.*\bcube\b", "solve_pickup"),
    (r"\b(?:move|swing)\b.*\btarget\b", "solve_swingonto_whenhold"),
    (
        r"\b(?:put|place)\b.*\bcube\b.*\b(?:on|onto)\b.*\btable\b",
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
        "coordinates from the top-left.\n6. For putdown, button press, and STOP, "
        "output point as null."
    )
    return TaskPromptSpec(
        task=TASK_NAME,
        version=f"swing_xtimes_oracle_planner_v3_wrist_target_check_{view}_{suffix}",
        include_description=include_description,
        include_wrist=include_wrist,
        system_prompt=build_system_prompt(
            subgoals=SUBGOALS,
            notes=NOTES_PAIRED if include_wrist else NOTES_OVERALL_ONLY,
            example=COMBINED_EXAMPLE,
            include_description=include_description,
            point_rules=point_rules,
        ),
        initial_user_template=(
            INITIAL_USER_TEMPLATE if include_wrist else INITIAL_USER_TEMPLATE_OVERALL_ONLY
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
