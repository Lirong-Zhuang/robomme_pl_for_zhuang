"""Task-specific prompt and subgoal-to-action registry for Oracle Plus."""

from __future__ import annotations

from .base import CallType
from .base import TaskPromptSpec
from .binfill import POINT_ACTION_NAMES as BINFILL_POINT_ACTION_NAMES
from .binfill import (
    PROMPT_OVERALL_ONLY_WITH_DESCRIPTION as BINFILL_PROMPT_OVERALL_ONLY_WITH_DESCRIPTION,
)
from .binfill import (
    PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION as BINFILL_PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION,
)
from .binfill import PROMPT_WITH_DESCRIPTION as BINFILL_PROMPT_WITH_DESCRIPTION
from .binfill import PROMPT_WITHOUT_DESCRIPTION as BINFILL_PROMPT_WITHOUT_DESCRIPTION
from .binfill import match_action_name as match_binfill_action_name
from .button_unmask import POINT_ACTION_NAMES as BUTTON_UNMASK_POINT_ACTION_NAMES
from .button_unmask import (
    PROMPT_OVERALL_ONLY_WITH_DESCRIPTION as BUTTON_UNMASK_PROMPT_OVERALL_ONLY_WITH_DESCRIPTION,
)
from .button_unmask import (
    PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION as BUTTON_UNMASK_PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION,
)
from .button_unmask import PROMPT_WITH_DESCRIPTION as BUTTON_UNMASK_PROMPT_WITH_DESCRIPTION
from .button_unmask import PROMPT_WITHOUT_DESCRIPTION as BUTTON_UNMASK_PROMPT_WITHOUT_DESCRIPTION
from .button_unmask import match_action_name as match_button_unmask_action_name
from .move_cube import POINT_ACTION_NAMES as MOVE_CUBE_POINT_ACTION_NAMES
from .move_cube import (
    PROMPT_OVERALL_ONLY_WITH_DESCRIPTION as MOVE_CUBE_PROMPT_OVERALL_ONLY_WITH_DESCRIPTION,
)
from .move_cube import (
    PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION as MOVE_CUBE_PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION,
)
from .move_cube import PROMPT_WITH_DESCRIPTION as MOVE_CUBE_PROMPT_WITH_DESCRIPTION
from .move_cube import PROMPT_WITHOUT_DESCRIPTION as MOVE_CUBE_PROMPT_WITHOUT_DESCRIPTION
from .move_cube import build_pick_place_audit_messages
from .move_cube import match_action_name as match_move_cube_action_name
from .pick_highlight import POINT_ACTION_NAMES as PICK_HIGHLIGHT_POINT_ACTION_NAMES
from .pick_highlight import (
    PROMPT_OVERALL_ONLY_WITH_DESCRIPTION as PICK_HIGHLIGHT_PROMPT_OVERALL_ONLY_WITH_DESCRIPTION,
)
from .pick_highlight import (
    PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION as PICK_HIGHLIGHT_PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION,
)
from .pick_highlight import PROMPT_WITH_DESCRIPTION as PICK_HIGHLIGHT_PROMPT_WITH_DESCRIPTION
from .pick_highlight import PROMPT_WITHOUT_DESCRIPTION as PICK_HIGHLIGHT_PROMPT_WITHOUT_DESCRIPTION
from .pick_highlight import match_action_name as match_pick_highlight_action_name
from .pick_xtimes import POINT_ACTION_NAMES as PICK_XTIMES_POINT_ACTION_NAMES
from .pick_xtimes import (
    PROMPT_OVERALL_ONLY_WITH_DESCRIPTION as PICK_XTIMES_PROMPT_OVERALL_ONLY_WITH_DESCRIPTION,
)
from .pick_xtimes import (
    PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION as PICK_XTIMES_PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION,
)
from .pick_xtimes import PROMPT_WITH_DESCRIPTION as PICK_XTIMES_PROMPT_WITH_DESCRIPTION
from .pick_xtimes import (
    PROMPT_WITHOUT_DESCRIPTION as PICK_XTIMES_PROMPT_WITHOUT_DESCRIPTION,
)
from .pick_xtimes import match_action_name as match_pick_xtimes_action_name
from .stop_cube import POINT_ACTION_NAMES as STOP_CUBE_POINT_ACTION_NAMES
from .stop_cube import (
    PROMPT_OVERALL_ONLY_WITH_DESCRIPTION as STOP_CUBE_PROMPT_OVERALL_ONLY_WITH_DESCRIPTION,
)
from .stop_cube import (
    PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION as STOP_CUBE_PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION,
)
from .stop_cube import PROMPT_WITH_DESCRIPTION as STOP_CUBE_PROMPT_WITH_DESCRIPTION
from .stop_cube import PROMPT_WITHOUT_DESCRIPTION as STOP_CUBE_PROMPT_WITHOUT_DESCRIPTION
from .stop_cube import match_action_name as match_stop_cube_action_name
from .swing_xtimes import POINT_ACTION_NAMES as SWING_XTIMES_POINT_ACTION_NAMES
from .swing_xtimes import (
    PROMPT_OVERALL_ONLY_WITH_DESCRIPTION as SWING_XTIMES_PROMPT_OVERALL_ONLY_WITH_DESCRIPTION,
)
from .swing_xtimes import (
    PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION as SWING_XTIMES_PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION,
)
from .swing_xtimes import (
    PROMPT_WITH_DESCRIPTION as SWING_XTIMES_PROMPT_WITH_DESCRIPTION,
)
from .swing_xtimes import (
    PROMPT_WITHOUT_DESCRIPTION as SWING_XTIMES_PROMPT_WITHOUT_DESCRIPTION,
)
from .swing_xtimes import match_action_name as match_swing_xtimes_action_name


TASK_PROMPTS_WITH_DESCRIPTION: dict[str, TaskPromptSpec] = {
    "BinFill": BINFILL_PROMPT_WITH_DESCRIPTION,
    "ButtonUnmask": BUTTON_UNMASK_PROMPT_WITH_DESCRIPTION,
    "MoveCube": MOVE_CUBE_PROMPT_WITH_DESCRIPTION,
    "PickHighlight": PICK_HIGHLIGHT_PROMPT_WITH_DESCRIPTION,
    "PickXtimes": PICK_XTIMES_PROMPT_WITH_DESCRIPTION,
    "StopCube": STOP_CUBE_PROMPT_WITH_DESCRIPTION,
    "SwingXtimes": SWING_XTIMES_PROMPT_WITH_DESCRIPTION,
}

TASK_PROMPTS_WITHOUT_DESCRIPTION: dict[str, TaskPromptSpec] = {
    "BinFill": BINFILL_PROMPT_WITHOUT_DESCRIPTION,
    "ButtonUnmask": BUTTON_UNMASK_PROMPT_WITHOUT_DESCRIPTION,
    "MoveCube": MOVE_CUBE_PROMPT_WITHOUT_DESCRIPTION,
    "PickHighlight": PICK_HIGHLIGHT_PROMPT_WITHOUT_DESCRIPTION,
    "PickXtimes": PICK_XTIMES_PROMPT_WITHOUT_DESCRIPTION,
    "StopCube": STOP_CUBE_PROMPT_WITHOUT_DESCRIPTION,
    "SwingXtimes": SWING_XTIMES_PROMPT_WITHOUT_DESCRIPTION,
}

TASK_PROMPTS_OVERALL_ONLY_WITH_DESCRIPTION: dict[str, TaskPromptSpec] = {
    "BinFill": BINFILL_PROMPT_OVERALL_ONLY_WITH_DESCRIPTION,
    "ButtonUnmask": BUTTON_UNMASK_PROMPT_OVERALL_ONLY_WITH_DESCRIPTION,
    "MoveCube": MOVE_CUBE_PROMPT_OVERALL_ONLY_WITH_DESCRIPTION,
    "PickHighlight": PICK_HIGHLIGHT_PROMPT_OVERALL_ONLY_WITH_DESCRIPTION,
    "PickXtimes": PICK_XTIMES_PROMPT_OVERALL_ONLY_WITH_DESCRIPTION,
    "StopCube": STOP_CUBE_PROMPT_OVERALL_ONLY_WITH_DESCRIPTION,
    "SwingXtimes": SWING_XTIMES_PROMPT_OVERALL_ONLY_WITH_DESCRIPTION,
}

TASK_PROMPTS_OVERALL_ONLY_WITHOUT_DESCRIPTION: dict[str, TaskPromptSpec] = {
    "BinFill": BINFILL_PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION,
    "ButtonUnmask": BUTTON_UNMASK_PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION,
    "MoveCube": MOVE_CUBE_PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION,
    "PickHighlight": PICK_HIGHLIGHT_PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION,
    "PickXtimes": PICK_XTIMES_PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION,
    "StopCube": STOP_CUBE_PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION,
    "SwingXtimes": SWING_XTIMES_PROMPT_OVERALL_ONLY_WITHOUT_DESCRIPTION,
}

ACTION_MATCHERS = {
    "BinFill": match_binfill_action_name,
    "ButtonUnmask": match_button_unmask_action_name,
    "MoveCube": match_move_cube_action_name,
    "PickHighlight": match_pick_highlight_action_name,
    "PickXtimes": match_pick_xtimes_action_name,
    "StopCube": match_stop_cube_action_name,
    "SwingXtimes": match_swing_xtimes_action_name,
}

TASK_POINT_ACTION_NAMES: dict[str, tuple[str, ...]] = {
    "BinFill": BINFILL_POINT_ACTION_NAMES,
    "ButtonUnmask": BUTTON_UNMASK_POINT_ACTION_NAMES,
    "MoveCube": MOVE_CUBE_POINT_ACTION_NAMES,
    "PickHighlight": PICK_HIGHLIGHT_POINT_ACTION_NAMES,
    "PickXtimes": PICK_XTIMES_POINT_ACTION_NAMES,
    "StopCube": STOP_CUBE_POINT_ACTION_NAMES,
    "SwingXtimes": SWING_XTIMES_POINT_ACTION_NAMES,
}


def available_tasks() -> tuple[str, ...]:
    return tuple(sorted(TASK_PROMPTS_WITH_DESCRIPTION))


def get_prompt(
    task: str,
    *,
    include_description: bool,
    include_wrist: bool = True,
) -> TaskPromptSpec:
    if include_wrist:
        registry = (
            TASK_PROMPTS_WITH_DESCRIPTION
            if include_description
            else TASK_PROMPTS_WITHOUT_DESCRIPTION
        )
    else:
        registry = (
            TASK_PROMPTS_OVERALL_ONLY_WITH_DESCRIPTION
            if include_description
            else TASK_PROMPTS_OVERALL_ONLY_WITHOUT_DESCRIPTION
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

    normalized = subgoal.strip().casefold()
    if normalized == "stop" or normalized.startswith(("stop ", "stop,", "stop.")):
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
    include_wrist: bool = True,
    history_subgoals: list[str] | None = None,
    current_subgoal: str | None = None,
    observation_count: int = 0,
    overall_image_shape: tuple[int, int] | None = None,
) -> list[dict[str, str]]:
    return get_prompt(
        task,
        include_description=include_description,
        include_wrist=include_wrist,
    ).build_messages(
        call_type=call_type,
        task_goal=task_goal,
        history_subgoals=history_subgoals,
        current_subgoal=current_subgoal,
        observation_count=observation_count,
        overall_image_shape=overall_image_shape,
    )
