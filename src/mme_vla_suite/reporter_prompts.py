"""Prompts and image-window helpers shared by Reporter training and inference."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar


DEFAULT_REPORTER_HISTORY_SIZE = 7

_ImageT = TypeVar("_ImageT")


REPORTER_SYSTEM_PROMPT = (
    "You are a helpful assistant to determine whether the current robot subgoal "
    "is complete by comparing observation before executing the current subgoal with a recent sequence "
    "of observations. "
    'Return only {"success": true} or {"success": false}. '
)

# Evaluation-only alternatives. ``original`` deliberately resolves to the
# unchanged training prompt above so prompt ablations do not alter training
# data generation or the default evaluation behavior.
REPORTER_EVAL_PROMPT_VARIANTS = ("original", "physical_outcome")

PHYSICAL_OUTCOME_REPORTER_SYSTEM_PROMPT = (
    "Judge whether the current subgoal's required physical end state is "
    "visibly achieved in the newest observation. Infer the expected result "
    "from the subgoal. A completed motion sequence or robot pose alone is not "
    "success. Use earlier observations only to identify the target and relevant "
    "state change. Return success true only if the target or environment is in "
    "the requested final state. If the result is absent or uncertain, return "
    "success false. "
    'Return only {"success": true} or {"success": false}. '
)


def validate_reporter_history_size(history_size: int) -> int:
    """Validate and return the number of recent Reporter-call observations."""
    if history_size < 1:
        raise ValueError("Reporter history size must be at least 1")
    return history_size


def build_reporter_image_window(
    init_image: _ImageT,
    recent_observations: Iterable[_ImageT],
    history_size: int = DEFAULT_REPORTER_HISTORY_SIZE,
) -> list[_ImageT]:
    """Return the init image followed by at most ``history_size`` observations.

    ``recent_observations`` contains observations captured at Reporter calls,
    rather than adjacent environment timesteps. An unfilled window remains at
    its actual length; the init image is never copied into the history.
    """
    validate_reporter_history_size(history_size)
    history = list(recent_observations)[-history_size:]
    return [init_image, *history]


def format_reporter_user_prompt(
    subgoal: str,
    history_size: int = DEFAULT_REPORTER_HISTORY_SIZE,
) -> str:
    """Format the exact multi-image prompt used for training and inference."""
    validate_reporter_history_size(history_size)
    observation_lines = []
    for index in range(1, history_size + 1):
        suffix = " (current observation)" if index == history_size else ""
        observation_lines.append(
            f"Recent observation {index}/{history_size}{suffix}: <image>"
        )
    return (
        f"Current Subgoal: {subgoal}\n"
        "Observation before executing the current subgoal: <image>\n"
        "Recent observations after execution, from "
        "oldest to newest:\n"
        + "\n".join(observation_lines)
        + "\nDetermine whether the current subgoal is complete from the "
        "observation before executing the current subgoal and the recent observation sequence. "
    )


def format_physical_outcome_reporter_user_prompt(
    subgoal: str,
    history_size: int = DEFAULT_REPORTER_HISTORY_SIZE,
) -> str:
    """Format the evaluation-only physical-outcome prompt variant."""
    validate_reporter_history_size(history_size)
    observation_lines = []
    for index in range(1, history_size + 1):
        suffix = " (current observation)" if index == history_size else ""
        observation_lines.append(
            f"Recent observation {index}/{history_size}{suffix}: <image>"
        )
    return (
        f"Current Subgoal: {subgoal}\n"
        "Observation before executing the current subgoal: <image>\n"
        "Recent observations after execution, from "
        "oldest to newest:\n"
        + "\n".join(observation_lines)
        + "\nJudge the newest observation by the required physical outcome, not "
        "by motion completion. If the result is absent or uncertain, report "
        "failure. "
    )


def validate_reporter_prompt_variant(prompt_variant: str) -> str:
    """Validate and return an evaluation Reporter prompt variant."""
    if prompt_variant not in REPORTER_EVAL_PROMPT_VARIANTS:
        choices = ", ".join(REPORTER_EVAL_PROMPT_VARIANTS)
        raise ValueError(
            f"Unsupported Reporter prompt variant: {prompt_variant!r}. "
            f"Expected one of: {choices}"
        )
    return prompt_variant


def format_reporter_eval_prompts(
    subgoal: str,
    history_size: int = DEFAULT_REPORTER_HISTORY_SIZE,
    prompt_variant: str = "original",
) -> tuple[str, str]:
    """Return system and user prompts for the selected evaluation variant."""
    validate_reporter_prompt_variant(prompt_variant)
    if prompt_variant == "physical_outcome":
        return (
            PHYSICAL_OUTCOME_REPORTER_SYSTEM_PROMPT,
            format_physical_outcome_reporter_user_prompt(subgoal, history_size),
        )
    return REPORTER_SYSTEM_PROMPT, format_reporter_user_prompt(subgoal, history_size)


# Preserve the original public template name for callers that format the
# default configuration with ``.replace("{subgoal}", ...)``.
REPORTER_USER_PROMPT = format_reporter_user_prompt("{subgoal}")
