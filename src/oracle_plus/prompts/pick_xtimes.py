"""PickXtimes-specific prompts and deterministic action mapping."""

from __future__ import annotations

import re

from .base import FOLLOWUP_USER_TEMPLATE
from .base import FOLLOWUP_USER_TEMPLATE_OVERALL_ONLY
from .base import INITIAL_USER_TEMPLATE
from .base import INITIAL_USER_TEMPLATE_OVERALL_ONLY
from .base import TaskPromptSpec
from .base import build_system_prompt


TASK_NAME = "PickXtimes"

NOTES = """1. The target is a purple disk.
2. Use the task goal to determine the required cube color and repetition count.
The main task sequence repeats pickup then placement for that count and presses
the button only after the final placement.
3. One repetition is complete only after the requested cube has been picked up
and then placed onto the target. Count completed placements, not grasp attempts.
4. The same target cube is used for every repetition. After placement it may be
at a new image position, so ground every pickup from the latest overall image.
5. Treat a pick-up subgoal as complete when the correct cube appears to be held
by the gripper and there is visual evidence that it has been lifted or moved
upward with the gripper. Approximate visual evidence is sufficient: a small lift,
a slight visible separation, or the cube moving upward together with the gripper
all count as success. Do not require a large or clearly measurable lifting height.
Only keep the pickup incomplete when the gripper appears empty, the cube remains
resting without upward movement, or there is clear evidence that it was dropped.
Treat placement as complete only after the cube has been released onto the purple
target.
6. A pickup requires point=[y, x] at the cube center in the latest overall
image. Placement, button press, and STOP require point=null."""

SUBGOALS = """- pick up the [red/blue/green] cube for the [first/second/third/...] time
- place the [red/blue/green] cube onto the target
- press the button to stop
- STOP, only after the button press completes"""

COMBINED_EXAMPLE = """If the task goal requests the green cube two times, use:
1. pick up the green cube for the first time
2. place the green cube onto the target
3. pick up the green cube for the second time
4. place the green cube onto the target
5. press the button to stop"""


SUBGOAL_ACTION_RULES: tuple[tuple[str, str], ...] = (
    (r"\bpick\s+up\b.*\bcube\b", "solve_pickup"),
    (
        r"\b(?:place|put)\b.*\bcube\b.*\b(?:onto|on)\b.*\btarget\b",
        "solve_putonto_whenhold",
    ),
    (r"\bpress\b.*\bbutton\b", "solve_button"),
)

POINT_ACTION_NAMES: tuple[str, ...] = ("solve_pickup",)


def _prompt(*, include_description: bool, include_wrist: bool) -> TaskPromptSpec:
    suffix = "with_description" if include_description else "without_description"
    view = "paired" if include_wrist else "overall_only"
    return TaskPromptSpec(
        task=TASK_NAME,
        version=f"pick_xtimes_oracle_planner_v2_liberal_pickup_{view}_{suffix}",
        include_description=include_description,
        include_wrist=include_wrist,
        system_prompt=build_system_prompt(
            subgoals=SUBGOALS,
            notes=NOTES,
            example=COMBINED_EXAMPLE,
            include_description=include_description,
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
