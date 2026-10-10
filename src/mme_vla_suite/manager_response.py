"""Parsing utilities for optional structured QwenVL Manager responses."""

from __future__ import annotations

import json
import re
from typing import Any


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_manager_response(
    response: str,
) -> tuple[str, dict[str, Any] | None]:
    """Return the executable subgoal and an optional structured response.

    Existing checkpoints normally return a plain subgoal. The interaction
    verification prompt may instead return JSON with a ``subgoal`` field. Any
    invalid or incomplete JSON falls back to the original plain-text behavior.
    """
    text = response.strip()
    try:
        payload = json.loads(_strip_code_fence(text))
    except json.JSONDecodeError:
        return text, None

    if not isinstance(payload, dict):
        return text, None
    subgoal = payload.get("subgoal")
    if not isinstance(subgoal, str) or not subgoal.strip():
        return text, None
    return subgoal.strip(), payload


def parse_verification_passed(
    payload: dict[str, Any] | None,
) -> bool | None:
    """Return a strict Boolean Manager verification result, if present."""
    if payload is None:
        return None
    result = payload.get("verification_passed")
    return result if isinstance(result, bool) else None


def _interaction_task_class(subgoal: str | None) -> str | None:
    if not subgoal:
        return None
    normalized = subgoal.lower()
    if re.search(r"\bpick[\s-]?up\b", normalized):
        return "pick_up"
    if "bin" in normalized and re.search(
        r"\b(?:put(?:s|ting)?|plac(?:e|es|ed|ing)|drop(?:s|ped|ping)?)\b",
        normalized,
    ):
        return "put_into_bin"
    if "button" in normalized and re.search(r"\bpress(?:es|ed|ing)?\b", normalized):
        return "button"
    return None


def _has_verified_pair(
    pairs: list[Any],
    *,
    relationship: str,
    first_terms: tuple[str, ...],
    second_terms: tuple[str, ...],
) -> bool:
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        first_object = pair.get("first_object")
        second_object = pair.get("second_object")
        if not isinstance(first_object, str) or not isinstance(second_object, str):
            continue
        if (
            pair.get("relationship") == relationship
            and pair.get("verified") is True
            and any(term in first_object.lower() for term in first_terms)
            and any(term in second_object.lower() for term in second_terms)
        ):
            return True
    return False


def validate_interaction_pairs(
    payload: dict[str, Any] | None,
    current_subgoal: str | None,
) -> tuple[bool, str | None]:
    """Validate the required verified interaction pairs for a subgoal class.

    The three interaction-aware classes require concrete, correctly oriented
    records. Other subgoal classes retain the prompt's legacy behavior and may
    use an empty pair list.
    """
    task_class = _interaction_task_class(current_subgoal)
    if task_class is None:
        return True, None
    if payload is None:
        return False, task_class
    pairs = payload.get("interaction_pairs")
    if not isinstance(pairs, list) or not pairs:
        return False, task_class

    if task_class == "pick_up":
        valid = _has_verified_pair(
            pairs,
            relationship="HOLDS",
            first_terms=("robot",),
            second_terms=("cube", "block"),
        ) and _has_verified_pair(
            pairs,
            relationship="ABOVE",
            first_terms=("cube", "block"),
            second_terms=("table",),
        )
    elif task_class == "put_into_bin":
        valid = _has_verified_pair(
            pairs,
            relationship="IN",
            first_terms=("cube", "block"),
            second_terms=("bin",),
        )
    else:
        valid = _has_verified_pair(
            pairs,
            relationship="PRESSES",
            first_terms=("robot",),
            second_terms=("button",),
        )
    return valid, task_class


def should_update_init_frame(
    *,
    has_current_subgoal: bool,
    subgoal_changed: bool,
    update_confirmation: bool | None,
) -> bool:
    """Decide whether Reporter may advance its subgoal init frame.

    ``False`` is an explicit veto used by the interaction-verification Manager.
    ``None`` preserves the legacy behavior, where a changed subgoal owns a new
    init frame even without an explicit confirmation signal.
    """
    if not has_current_subgoal:
        return True
    if update_confirmation is False:
        return False
    return subgoal_changed or update_confirmation is True
