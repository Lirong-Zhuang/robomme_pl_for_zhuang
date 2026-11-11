"""Minimal paper-style task descriptions used by init planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskDescription:
    """Internal task key plus summary and goal forms exposed to init planning."""

    task_name: str
    summary: str
    task_goal_examples: tuple[str, ...]

    def to_prompt_text(self) -> str:
        """Return only the paper-style summary shown to the init-planning VLM."""

        return self.summary


BINFILL_DESCRIPTION = TaskDescription(
    task_name="BinFill",
    summary=(
        "The BinFill task evaluates sequential decisionmaking and temporal memory. "
        "The robot must place a specified number of colored cubes into a bin and "
        "then press a button to terminate the episode. Success requires accurate "
        "counting and timely termination."
    ),
    task_goal_examples=(
        "Put one blue cube into the bin and press the button to stop.",
        (
            "Put one blue cube and two green cubes into the bin and press the "
            "button to stop."
        ),
        (
            "Put one blue cube, one green cube, and two red cubes into the bin "
            "and press the button to stop."
        ),
    ),
)


MOVE_CUBE_DESCRIPTION = TaskDescription(
    task_name="MoveCube",
    summary=(
        "The MoveCube task assesses the robot’s ability to recognize and reproduce "
        "distinct action semantics. The task requires the robot to observe a video "
        "demonstration where a cube is moved to a target using one of three specific "
        "manipulation manners: (1) hooking it with a tool; (2) pushing it with a "
        "closed gripper; or (3) performing a standard pick-and-place operation. "
        "Success requires correctly inferring the underlying motion strategy from "
        "videos and executing the exact same method during the execution phase."
    ),
    task_goal_examples=(
        (
            "Watch the video carefully, then move the cube to the target in the "
            "same manner as before."
        ),
    ),
)


PICK_HIGHLIGHT_DESCRIPTION = TaskDescription(
    task_name="PickHighlight",
    summary=(
        "The PickHighlight task evaluates the robot’s shortterm object memory. The "
        "robot must press a button to briefly highlight specific cubes, memorize "
        "the highlighted cubes, and then pick them up after the cues disappear. "
        "This tests the robot’s ability to maintain a working memory of transient "
        "object states."
    ),
    task_goal_examples=(
        (
            "First press the button, then pick up all cubes that have been "
            "highlighted with white areas on the table."
        ),
    ),
)


_TASK_DESCRIPTIONS: tuple[TaskDescription, ...] = (
    BINFILL_DESCRIPTION,
    MOVE_CUBE_DESCRIPTION,
    PICK_HIGHLIGHT_DESCRIPTION,
)

_TASK_DESCRIPTION_BY_KEY = {
    description.task_name.casefold(): description
    for description in _TASK_DESCRIPTIONS
}


def available_task_descriptions() -> tuple[str, ...]:
    """Return registered task names in deterministic order."""

    return tuple(description.task_name for description in _TASK_DESCRIPTIONS)


def get_task_description(task_name: str) -> TaskDescription:
    """Resolve one task description with case-insensitive task-name matching."""

    key = task_name.strip().casefold()
    if not key:
        raise ValueError("task_name must be non-empty")
    try:
        return _TASK_DESCRIPTION_BY_KEY[key]
    except KeyError as error:
        available = ", ".join(available_task_descriptions())
        raise KeyError(
            f"No task description registered for {task_name!r}; "
            f"available: {available}"
        ) from error
