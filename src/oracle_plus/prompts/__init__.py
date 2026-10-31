"""Task-specific prompt and subgoal-to-action registry for Oracle Plus."""

from __future__ import annotations

from .base import CallType
from .base import TaskPromptSpec
from .binfill import POINT_ACTION_NAMES as BINFILL_POINT_ACTION_NAMES
from .binfill import PROMPT_WITH_DESCRIPTION as BINFILL_PROMPT_WITH_DESCRIPTION
from .binfill import PROMPT_WITHOUT_DESCRIPTION as BINFILL_PROMPT_WITHOUT_DESCRIPTION
from .binfill import match_action_name as match_binfill_action_name


TASK_PROMPTS_WITH_DESCRIPTION: dict[str, TaskPromptSpec] = {
    "BinFill": BINFILL_PROMPT_WITH_DESCRIPTION,
}

TASK_PROMPTS_WITHOUT_DESCRIPTION: dict[str, TaskPromptSpec] = {
    "BinFill": BINFILL_PROMPT_WITHOUT_DESCRIPTION,
}

ACTION_MATCHERS = {
    "BinFill": match_binfill_action_name,
}

TASK_POINT_ACTION_NAMES: dict[str, tuple[str, ...]] = {
    "BinFill": BINFILL_POINT_ACTION_NAMES,
}


def available_tasks() -> tuple[str, ...]:
    return tuple(sorted(TASK_PROMPTS_WITH_DESCRIPTION))


def get_prompt(task: str, *, include_description: bool) -> TaskPromptSpec:
    registry = (
        TASK_PROMPTS_WITH_DESCRIPTION
        if include_description
        else TASK_PROMPTS_WITHOUT_DESCRIPTION
    )
    try:
        return registry[task]
    except KeyError as error:
        raise ValueError(
            f"No task-specific prompt for {task!r}; available: {available_tasks()}"
        ) from error


def get_point_action_names(task: str) -> tuple[str, ...]:
    try:
        return TASK_POINT_ACTION_NAMES[task]
    except KeyError as error:
        raise ValueError(
            f"No target-point action registry for {task!r}; "
            f"available: {available_tasks()}"
        ) from error


def match_action_name(task: str, subgoal: str) -> str | None:
    try:
        matcher = ACTION_MATCHERS[task]
    except KeyError as error:
        raise ValueError(f"Task {task!r} has no subgoal-action matcher") from error
    return matcher(subgoal)


def resolve_action_name(task: str, subgoal: str) -> str | None:
    """Resolve a model subgoal to a trusted action, or None for STOP."""

    if subgoal.strip().casefold() == "stop":
        return None
    action_name = match_action_name(task, subgoal)
    if action_name is None:
        raise ValueError(
            f"Subgoal {subgoal!r} does not match a trusted {task} action keyword rule"
        )
    return action_name


def build_messages(
    *,
    task: str,
    call_type: CallType,
    task_goal: str,
    include_description: bool,
    history_subgoals: list[str] | None = None,
    current_subgoal: str | None = None,
    observation_count: int = 0,
    overall_image_shape: tuple[int, int] | None = None,
) -> list[dict[str, str]]:
    return get_prompt(task, include_description=include_description).build_messages(
        call_type=call_type,
        task_goal=task_goal,
        history_subgoals=history_subgoals,
        current_subgoal=current_subgoal,
        observation_count=observation_count,
        overall_image_shape=overall_image_shape,
    )
