"""ButtonUnmask-specific prompts and deterministic action mapping."""

from __future__ import annotations

import re

from .base import FOLLOWUP_USER_TEMPLATE
from .base import FOLLOWUP_USER_TEMPLATE_OVERALL_ONLY
from .base import INITIAL_USER_TEMPLATE
from .base import INITIAL_USER_TEMPLATE_OVERALL_ONLY
from .base import TaskPromptSpec
from .base import build_system_prompt


TASK_NAME = "ButtonUnmask"

NOTES = """1. Press the button first. Before and while pressing it, remember
which container position corresponds to each visible red, blue, or green cube;
the white containers will hide the cubes after the press.
2. Select containers from that remembered spatial correspondence, in the color
order requested by the task goal. A container pickup requires point=[y, x] at
the center of that container in the latest overall image.
3. If more than one container is requested, put down each picked container
before picking the next one. Do not put down the final requested container:
the task completes as soon as that final container is picked up.
4. Treat a container pickup as complete when it is visibly held and lifted or
moved upward with the gripper. Treat put-down as complete when the gripper has
released it onto the table.
5. Press-button, put-down, and STOP require point=null."""

SUBGOALS = """- press the button
- pick up the container that hides the [red/blue/green] cube
- put down the container, only between requested container pickups
- STOP, only after the final requested container has been picked up"""

COMBINED_EXAMPLE = """If the task goal is "pick up the containers hiding the
blue and red cubes", a typical sequence is:
1. press the button
2. pick up the container that hides the blue cube
3. put down the container
4. pick up the container that hides the red cube"""


SUBGOAL_ACTION_RULES: tuple[tuple[str, str], ...] = (
    (r"\bpress\b.*\bbutton\b", "solve_button"),
    (r"\bpick\s+up\b.*\bcontainer\b", "solve_pickup_bin"),
    (r"\bput\s+down\b.*\bcontainer\b", "solve_putdown_whenhold"),
)

POINT_ACTION_NAMES: tuple[str, ...] = ("solve_pickup_bin",)


def _prompt(*, include_description: bool, include_wrist: bool) -> TaskPromptSpec:
    suffix = "with_description" if include_description else "without_description"
    view = "paired" if include_wrist else "overall_only"
    return TaskPromptSpec(
        task=TASK_NAME,
        version=f"button_unmask_oracle_planner_v1_{view}_{suffix}",
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
