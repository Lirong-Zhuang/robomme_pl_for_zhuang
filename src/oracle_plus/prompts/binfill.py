"""BinFill-specific subgoal prompts and deterministic action mapping."""

from __future__ import annotations

import re

from .base import TaskPromptSpec
from .base import build_system_prompt


TASK_NAME = "BinFill"

NOTES = """1. The bin is a box with black holes on the top. It sucks in cubes
placed into it, so count how many cubes have already been inserted. A black hole
becoming visible again after placement is expected and, by itself, does not mean
the bin is empty or that a previously completed placement failed.
2. Use the task goal to determine the required cube color, order, and total count.
3. Treat a pick-up subgoal as complete when the correct cube appears to be held
by the gripper and there is visual evidence that it has been lifted or moved
upward with the gripper. Approximate visual evidence is sufficient: a small lift,
a slight visible separation, or the cube moving upward together with the gripper
all count as success. Do not require a large or clearly measurable lifting height.
Only keep the pickup incomplete when the gripper appears empty, the cube remains
resting without upward movement, or there is clear evidence that it was dropped.
4. A pickup subgoal requires point=[y, x] at the center of the intended cube in
the latest overall image. Put-into-bin, press-button, and STOP require point=null.
5. Press the button only after all requested cubes have been placed into the bin."""

SUBGOALS = """- pick up the [first/second/third/...] [red/blue/green] cube
- put it into the bin
- press the button
- STOP, only after the full task is complete"""

COMBINED_EXAMPLE = """If the task goal is "put two red cubes into the bin", a typical sequence is:
1. pick up the first red cube
2. put it into the bin
3. pick up the second red cube
4. put it into the bin
5. press the button"""


# The model never sees or emits these function names. The controller resolves
# one validated subgoal to a trusted function through the first matching rule.
SUBGOAL_ACTION_RULES: tuple[tuple[str, str], ...] = (
    (r"\bpick\s+up\b.*\bcube\b", "solve_pickup"),
    (
        r"\bput\b.*\b(?:cube|it)\b.*\b(?:into|in)\b.*\bbin\b",
        "solve_putonto_whenhold_binspecial",
    ),
    (r"\bpress\b.*\bbutton\b", "solve_button"),
)

POINT_ACTION_NAMES: tuple[str, ...] = ("solve_pickup",)


PROMPT_WITH_DESCRIPTION = TaskPromptSpec(
    task=TASK_NAME,
    version="binfill_subgoal_point_v3_with_description",
    include_description=True,
    system_prompt=build_system_prompt(
        subgoals=SUBGOALS,
        notes=NOTES,
        example=COMBINED_EXAMPLE,
        include_description=True,
    ),
)

PROMPT_WITHOUT_DESCRIPTION = TaskPromptSpec(
    task=TASK_NAME,
    version="binfill_subgoal_point_v3_without_description",
    include_description=False,
    system_prompt=build_system_prompt(
        subgoals=SUBGOALS,
        notes=NOTES,
        example=COMBINED_EXAMPLE,
        include_description=False,
    ),
)


def match_action_name(subgoal: str) -> str | None:
    """Resolve one canonical BinFill subgoal through keyword-pattern matching."""

    text = " ".join(subgoal.casefold().replace("_", " ").split())
    for pattern, action_name in SUBGOAL_ACTION_RULES:
        if re.search(pattern, text):
            return action_name
    return None
