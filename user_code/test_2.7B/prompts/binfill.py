"""BinFill-specific ordinary prompts, examples, notes, and action catalogue."""

from __future__ import annotations

import re

from .base import SYSTEM_TEMPLATE
from .base import TaskPromptSpec


TASK_NAME = "BinFill"

NOTES = """1. The bin is a box with black holes on the top. It sucks in cubes
placed into it, so count how many cubes have already been inserted.
2. Use the task goal to determine the required cube color, order, and total count.
3. Press the button only after all requested cubes have been placed into the bin."""

SUBGOALS = """- pick up the [first/second/third/...] [red/blue/green] cube
- put it into the bin
- press the button"""

ACTIONS: tuple[dict[str, str], ...] = (
    {
        "name": "solve_pickup",
        "note": (
            "Compute a standard object-grasp pose, then approach, close the gripper, "
            "and lift the task-relevant cube."
        ),
    },
    {
        "name": "solve_putonto_whenhold_binspecial",
        "note": (
            "BinFill-specific action: move the held cube above the bin, release it, "
            "and retreat sideways."
        ),
    },
    {
        "name": "solve_button",
        "note": (
            "Move above the button, close the gripper, press the button downward, "
            "and return above it."
        ),
    },
)

ACTION_NAMES: tuple[str, ...] = tuple(item["name"] for item in ACTIONS)

ACTION_LIST = "\n".join(
    f"- {item['name']}: {item['note']}" for item in ACTIONS
)

COMBINED_EXAMPLE = """If the task goal is "put two red cubes into the bin", a typical paired sequence is:
1. pick up the first red cube -> solve_pickup
2. put it into the bin -> solve_putonto_whenhold_binspecial
3. pick up the second red cube -> solve_pickup
4. put it into the bin -> solve_putonto_whenhold_binspecial
5. press the button -> solve_button"""


PROMPT = TaskPromptSpec(
    task=TASK_NAME,
    version="binfill_subgoal_action_v2_no_thinking",
    system_prompt=SYSTEM_TEMPLATE.format(
        subgoals=SUBGOALS,
        actions=ACTION_LIST,
        notes=NOTES,
        example=COMBINED_EXAMPLE,
    ),
)


def match_action_name(subgoal: str) -> str | None:
    """Map one recorded BinFill simple subgoal to its wrapper name."""

    text = " ".join(subgoal.casefold().replace("_", " ").split())
    rules: tuple[tuple[str, str], ...] = (
        (r"pick up.*cube", "solve_pickup"),
        (
            r"put.*cube.*(into|in).*bin|put it into the bin",
            "solve_putonto_whenhold_binspecial",
        ),
        (r"press.*button", "solve_button"),
    )
    for pattern, action_name in rules:
        if re.search(pattern, text):
            return action_name
    return None

