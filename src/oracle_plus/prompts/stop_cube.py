"""StopCube-specific prompts and deterministic action mapping."""

from __future__ import annotations

import re

from .base import FOLLOWUP_USER_TEMPLATE
from .base import FOLLOWUP_USER_TEMPLATE_OVERALL_ONLY
from .base import INITIAL_USER_TEMPLATE
from .base import INITIAL_USER_TEMPLATE_OVERALL_ONLY
from .base import TaskPromptSpec
from .base import build_system_prompt


TASK_NAME = "StopCube"

NOTES = """1. Estimate the cube's speed to decide when to remain static and
when to press the button.
2. Count how many times the cube has reached the target and use the requested
count in the task goal to decide when to press.
3. The observation clips are continuous. Do not recount one target visit when
the cube is still on the target in both the previous and current observations.
4. If the cube is about to reach the target for the requested time, press the
button directly. The press motion takes time, so do not wait until the cube is
already perfectly centered.
5. The target is a purple disk.
6. First move above the button. Each remain-static execution advances to one
incremental absolute-timestep checkpoint and completes that waiting segment.
If another checkpoint is needed, return completed=true with
next_subgoal="remain static"; when it is time to press, return completed=true
with the press subgoal.
7. All subgoals and STOP require point=null."""

SUBGOALS = """- move to the top of the button to prepare
- remain static
- press the button to stop the cube on the target
- STOP, only after the cube is stopped on the requested target visit"""

COMBINED_EXAMPLE = """A typical task subgoal sequence is:
1. move to the top of the button to prepare
2. remain static
3. press the button to stop the cube on the target"""


SUBGOAL_ACTION_RULES: tuple[tuple[str, str], ...] = (
    (
        r"\b(?:move|go)\b.*\b(?:top|above|over)\b.*\bbutton\b.*\bprepare\b",
        "solve_button_ready",
    ),
    (r"\b(?:remain|stay|hold)\b.*\bstatic\b", "solve_hold_obj_absTimestep"),
    (r"\bpress\b.*\bbutton\b", "solve_button"),
)

POINT_ACTION_NAMES: tuple[str, ...] = ()


def _prompt(*, include_description: bool, include_wrist: bool) -> TaskPromptSpec:
    suffix = "with_description" if include_description else "without_description"
    view = "paired" if include_wrist else "overall_only"
    point_rules = (
        "5. Output point as null for every StopCube subgoal and for STOP.\n"
        "6. Never output pixel coordinates for this task."
    )
    return TaskPromptSpec(
        task=TASK_NAME,
        version=f"stop_cube_oracle_planner_v1_{view}_{suffix}",
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
