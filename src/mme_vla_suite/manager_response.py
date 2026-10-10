"""Parsing utilities for optional structured QwenVL Manager responses."""

from __future__ import annotations

import json
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
