"""Resolve trusted action parameters from words in a rendered subgoal."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


_WORD_PATTERN = re.compile(r"[a-z]+")


def _words(subgoal: str) -> set[str]:
    if not isinstance(subgoal, str) or not subgoal.strip():
        raise ValueError("subgoal must be a non-empty string")
    return set(_WORD_PATTERN.findall(subgoal.lower()))


def _exclusive_word(words: set[str], choices: tuple[str, ...], name: str) -> str:
    matched = [choice for choice in choices if choice in words]
    if len(matched) != 1:
        raise ValueError(
            f"subgoal must contain exactly one {name} token from {list(choices)}; "
            f"matched={matched}"
        )
    return matched[0]


def lateral_side_from_subgoal(subgoal: str) -> str:
    """Return left/right by word containment, never by whole-sentence matching."""

    return _exclusive_word(_words(subgoal), ("left", "right"), "lateral side")


def pattern_direction_from_subgoal(subgoal: str) -> str:
    """Compose a PatternLock direction from independent axial word tokens."""

    words = _words(subgoal)
    longitudinal = [word for word in ("forward", "backward") if word in words]
    lateral = [word for word in ("left", "right") if word in words]
    if len(longitudinal) > 1 or len(lateral) > 1:
        raise ValueError("PatternLock subgoal contains conflicting direction tokens")
    if not longitudinal and not lateral:
        raise ValueError("PatternLock subgoal contains no direction token")
    return "-".join(longitudinal + lateral)


def circling_direction_from_subgoal(subgoal: str) -> str:
    """Extract clockwise/counterclockwise without substring ambiguity."""

    words = _words(subgoal)
    if "counterclockwise" in words or {"counter", "clockwise"} <= words:
        if "anticlockwise" in words:
            raise ValueError("subgoal contains conflicting circling direction tokens")
        return "counterclockwise"
    if "anticlockwise" in words:
        return "counterclockwise"
    if "clockwise" in words:
        return "clockwise"
    raise ValueError("subgoal contains no clockwise/counterclockwise token")


def resolve_action_arguments(
    action_name: str,
    *,
    subgoal: str,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build execution arguments while retaining unrelated planning arguments."""

    resolved = dict(arguments or {})
    if action_name == "insert_peg":
        side = lateral_side_from_subgoal(subgoal)
        resolved["direction"] = -1 if side == "left" else 1
    elif action_name == "solve_pattern_direction":
        resolved["direction"] = pattern_direction_from_subgoal(subgoal)
    elif action_name == "solve_swingonto_with_direction":
        resolved["side"] = lateral_side_from_subgoal(subgoal)
        resolved["direction"] = circling_direction_from_subgoal(subgoal)
    return resolved
