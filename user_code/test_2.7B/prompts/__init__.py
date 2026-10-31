"""Task-specific prompt registry for the Qwen 2.7B feasibility test."""

from __future__ import annotations

from .base import CallType
from .base import TaskPromptSpec
from .binfill import PROMPT as BINFILL_PROMPT
from .binfill import match_action_name as match_binfill_action_name


TASK_PROMPTS: dict[str, TaskPromptSpec] = {
    "BinFill": BINFILL_PROMPT,
}

ACTION_MATCHERS = {
    "BinFill": match_binfill_action_name,
}


def available_tasks() -> tuple[str, ...]:
    return tuple(sorted(TASK_PROMPTS))


def get_prompt(task: str) -> TaskPromptSpec:
    try:
        return TASK_PROMPTS[task]
    except KeyError as error:
        raise ValueError(
            f"No task-specific prompt for {task!r}; available: {available_tasks()}"
        ) from error


def match_action_name(task: str, subgoal: str) -> str | None:
    try:
        matcher = ACTION_MATCHERS[task]
    except KeyError as error:
        raise ValueError(f"Task {task!r} has no action-name matcher") from error
    return matcher(subgoal)


def build_messages(
    *,
    task: str,
    call_type: CallType,
    task_goal: str,
    history_subgoals: list[str] | None = None,
    current_subgoal: str | None = None,
    observation_count: int = 0,
) -> list[dict[str, str]]:
    return get_prompt(task).build_messages(
        call_type=call_type,
        task_goal=task_goal,
        history_subgoals=history_subgoals,
        current_subgoal=current_subgoal,
        observation_count=observation_count,
    )

