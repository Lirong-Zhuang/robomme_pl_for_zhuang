"""PickHighlight-specific prompts and deterministic action mapping."""

from __future__ import annotations

import re

from .base import FOLLOWUP_USER_TEMPLATE
from .base import FOLLOWUP_USER_TEMPLATE_OVERALL_ONLY
from .base import INITIAL_USER_TEMPLATE
from .base import INITIAL_USER_TEMPLATE_OVERALL_ONLY
from .base import TaskPromptSpec
from .base import build_system_prompt


TASK_NAME = "PickHighlight"

NOTES = """1. Press the button first and do not advance until the full press
motion has finished. Observe which cubes acquire a white highlighted area and
remember their colors and positions.
2. Pick every highlighted cube. If there is only one, omit ordinal words; if
there are several, use first, second, third, and so on.
3. Put a picked cube onto the table only when another highlighted cube remains.
Do not put down the final highlighted cube: the task completes when it is picked.
4. A highlighted-cube pickup requires point=[y, x] at the center of the intended
cube in the latest overall image. Press-button, place-on-table, and STOP require
point=null.
5. Treat pickup as complete on visible grasp-and-lift evidence, and placement as
complete when the cube has been released onto the table."""

SUBGOALS = """- press the button
- pick up the [first/second/third/...] highlighted cube, which is [red/blue/green]
- place the cube onto the table, only between highlighted-cube pickups
- STOP, only after the final highlighted cube has been picked up"""

COMBINED_EXAMPLE = """If two cubes become highlighted, a typical sequence is:
1. press the button
2. pick up the first highlighted cube, which is red
3. place the cube onto the table
4. pick up the second highlighted cube, which is blue"""


SUBGOAL_ACTION_RULES: tuple[tuple[str, str], ...] = (
    (r"\bpress\b.*\bbutton\b", "solve_button"),
    (r"\bpick\s+up\b.*\bhighlighted\b.*\bcube\b", "solve_pickup"),
    (
        r"\b(?:place|put)\b.*\bcube\b.*\b(?:onto|on)\b.*\btable\b",
        "solve_putdown_whenhold",
    ),
)

POINT_ACTION_NAMES: tuple[str, ...] = ("solve_pickup",)


def _prompt(*, include_description: bool, include_wrist: bool) -> TaskPromptSpec:
    suffix = "with_description" if include_description else "without_description"
    view = "paired" if include_wrist else "overall_only"
    return TaskPromptSpec(
        task=TASK_NAME,
        version=f"pick_highlight_oracle_planner_v1_{view}_{suffix}",
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
