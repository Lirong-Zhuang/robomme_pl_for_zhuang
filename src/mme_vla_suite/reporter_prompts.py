"""Backward-compatible access to the default versioned Reporter prompt.

New code should resolve an explicit version with
``mme_vla_suite.prompts.get_reporter_prompt``. These names remain available so
existing callers continue to use the Trinity v2.2 default.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from mme_vla_suite.prompts import DEFAULT_REPORTER_PROMPT_VERSION
from mme_vla_suite.prompts import get_reporter_prompt
from mme_vla_suite.prompts.registry import DEFAULT_REPORTER_HISTORY_SIZE

_ImageT = TypeVar("_ImageT")

_DEFAULT_PROMPT = get_reporter_prompt(DEFAULT_REPORTER_PROMPT_VERSION)
REPORTER_PROMPT_VERSION = _DEFAULT_PROMPT.version
REPORTER_SYSTEM_PROMPT = _DEFAULT_PROMPT.system_prompt


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
    """Return the init image followed by at most ``history_size`` observations."""
    validate_reporter_history_size(history_size)
    history = list(recent_observations)[-history_size:]
    return [init_image, *history]


def format_reporter_user_prompt(
    subgoal: str,
    history_size: int = DEFAULT_REPORTER_HISTORY_SIZE,
) -> str:
    """Format the default Reporter prompt retained for legacy callers."""
    validate_reporter_history_size(history_size)
    return _DEFAULT_PROMPT.format_user_prompt(subgoal, history_size)


REPORTER_USER_PROMPT = format_reporter_user_prompt("{subgoal}")
