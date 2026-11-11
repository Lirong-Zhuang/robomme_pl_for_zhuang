"""Category-level task descriptions used by init planning.

These descriptions contain stable benchmark facts only. Episode-specific facts,
such as the concrete task goal, highlighted objects, or the demonstrated strategy,
must be supplied later by the environment and the first in-task-planning call.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskDescription:
    """Structured, reusable description of one benchmark task category."""

    task_category: str
    task_family: str
    summary: str
    environment: str
    objective: str
    episode_variation: str
    memory_requirement: str

    def to_prompt_text(self) -> str:
        """Render the description for injection into the init-planning prompt."""

        return "\n".join(
            (
                f"Task family: {self.task_family}",
                f"Summary: {self.summary}",
                f"Environment: {self.environment}",
                f"Objective: {self.objective}",
                f"Episode-specific variation: {self.episode_variation}",
                f"Memory requirement: {self.memory_requirement}",
            )
        )


BINFILL_DESCRIPTION = TaskDescription(
    task_category="BinFill",
    task_family="counting and temporal-memory manipulation",
    summary=(
        "Place the task-requested numbers of colored cubes into a bin and press "
        "the button only after every requested placement has been completed."
    ),
    environment=(
        "The tabletop contains red, blue, and/or green cubes, distractor cubes, "
        "a bin represented by a board with a dark opening, and a push button. "
        "A cube accepted by the bin remains visible briefly and is then removed "
        "from the scene."
    ),
    objective=(
        "Repeatedly pick a requested cube, place it into the bin, maintain the "
        "completed count for every requested color, and press the button after "
        "all requested color counts have been satisfied."
    ),
    episode_variation=(
        "The task goal determines which cube colors are required and how many of "
        "each color must be inserted. Cube positions, distractors, color counts, "
        "and color execution order can vary between episodes."
    ),
    memory_requirement=(
        "Retain per-color placement counts because accepted cubes can disappear "
        "from the current view; do not infer the completed count from one frame."
    ),
)


MOVE_CUBE_DESCRIPTION = TaskDescription(
    task_category="MoveCube",
    task_family="imitation and procedural-memory manipulation",
    summary=(
        "Infer how a demonstration moves a cube to its target, then reproduce "
        "the same manipulation strategy in the new execution."
    ),
    environment=(
        "The scene contains a small cube, a purple disk target, a long thin peg, "
        "and the robot gripper. Reset observations include a prerecorded "
        "demonstration followed by the initial state of the new execution."
    ),
    objective=(
        "Move the cube onto the target using exactly the demonstrated strategy: "
        "direct contact with an empty closed gripper, cube pick-and-place, or "
        "grasping the peg and using it to move the cube."
    ),
    episode_variation=(
        "The demonstrated strategy and the object poses can vary by episode. The "
        "concrete strategy cannot be selected from the task category alone and "
        "must be inferred from temporal demonstration evidence."
    ),
    memory_requirement=(
        "Preserve the inferred demonstration strategy across the reset into the "
        "new execution and never combine actions belonging to different strategies."
    ),
)


PICK_HIGHLIGHT_DESCRIPTION = TaskDescription(
    task_category="PickHighlight",
    task_family="reference and object-memory manipulation",
    summary=(
        "Press the button to reveal the target subset, then pick every cube that "
        "acquires a white highlighted area."
    ),
    environment=(
        "The tabletop contains a push button and several colored cubes. Pressing "
        "the button adds a white highlighted area to one or more target cubes; "
        "the remaining cubes are distractors."
    ),
    objective=(
        "Press the button first, identify and pick every highlighted cube at least "
        "once, and avoid picking non-highlighted cubes. When several targets are "
        "present, place the held cube back on the table before picking the next; "
        "the final highlighted cube does not need to be put down."
    ),
    episode_variation=(
        "The number, colors, and positions of cubes and the highlighted subset "
        "vary by episode. Target identities are unavailable until the button has "
        "been pressed and the resulting scene change has been observed."
    ),
    memory_requirement=(
        "Store the post-button highlighted identities and their locations, then "
        "track which highlighted cubes have already been picked."
    ),
)


_TASK_DESCRIPTIONS: tuple[TaskDescription, ...] = (
    BINFILL_DESCRIPTION,
    MOVE_CUBE_DESCRIPTION,
    PICK_HIGHLIGHT_DESCRIPTION,
)

_TASK_DESCRIPTION_BY_KEY = {
    description.task_category.casefold(): description
    for description in _TASK_DESCRIPTIONS
}


def available_task_descriptions() -> tuple[str, ...]:
    """Return registered task-category names in deterministic order."""

    return tuple(description.task_category for description in _TASK_DESCRIPTIONS)


def get_task_description(task_category: str) -> TaskDescription:
    """Resolve one task description with case-insensitive category matching."""

    key = task_category.strip().casefold()
    if not key:
        raise ValueError("task_category must be non-empty")
    try:
        return _TASK_DESCRIPTION_BY_KEY[key]
    except KeyError as error:
        available = ", ".join(available_task_descriptions())
        raise KeyError(
            f"No task description registered for {task_category!r}; "
            f"available: {available}"
        ) from error
