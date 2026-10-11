"""Deterministic completion checks for Manager interaction observations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any


InteractionPair = tuple[str, str, str]


@dataclass(frozen=True)
class InteractionVerificationResult:
    """Parsed Manager observations and the deterministic completion decision."""

    interaction_pairs: tuple[InteractionPair, ...]
    verification_passed: bool
    json_parse_success: bool
    reason: str

    def to_log_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["interaction_pairs"] = [list(pair) for pair in self.interaction_pairs]
        return payload


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


def _normalize_entity(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _normalize_relation(value: str) -> str:
    return value.strip().upper()


def _parse_interaction_pairs(response: str) -> tuple[tuple[InteractionPair, ...], bool]:
    try:
        payload = json.loads(_strip_code_fence(response))
    except (json.JSONDecodeError, TypeError):
        return (), False

    if not isinstance(payload, dict):
        return (), False
    raw_pairs = payload.get("interaction_pairs")
    if not isinstance(raw_pairs, list):
        return (), False

    pairs: list[InteractionPair] = []
    for raw_pair in raw_pairs:
        if (
            not isinstance(raw_pair, list)
            or len(raw_pair) != 3
            or not all(isinstance(value, str) for value in raw_pair)
        ):
            return (), False
        subject, relation, object_ = raw_pair
        pairs.append(
            (
                _normalize_entity(subject),
                _normalize_relation(relation),
                _normalize_entity(object_),
            )
        )
    return tuple(pairs), True


def _cube_color_from_subgoal(subgoal: str) -> str | None:
    match = re.search(r"\b([a-z]+)\s+cube\b", subgoal.lower())
    return match.group(1) if match else None


def _is_concrete_cube(entity: str, expected_color: str | None) -> bool:
    if not re.fullmatch(r"[a-z]+ cube", entity):
        return False
    if expected_color is None:
        return True
    return entity == f"{expected_color} cube"


def verify_interaction_response(
    subgoal: str,
    response: str,
) -> InteractionVerificationResult:
    """Hard-check observed interaction pairs against the current subgoal.

    Any model-provided completion Boolean is deliberately ignored. Unsupported
    subgoals and malformed observations fail closed.
    """

    pairs, json_parse_success = _parse_interaction_pairs(response)
    if not json_parse_success:
        return InteractionVerificationResult(
            interaction_pairs=(),
            verification_passed=False,
            json_parse_success=False,
            reason="invalid interaction-pair JSON",
        )

    normalized_subgoal = " ".join(subgoal.lower().split())
    expected_color = _cube_color_from_subgoal(normalized_subgoal)

    if "pick up" in normalized_subgoal:
        holds_pairs = [
            pair
            for pair in pairs
            if pair[0] == "robot"
            and pair[1] == "HOLDS"
            and _is_concrete_cube(pair[2], expected_color)
        ]
        above_pairs = [
            pair
            for pair in pairs
            if _is_concrete_cube(pair[0], expected_color)
            and pair[1] == "ABOVE"
            and pair[2] == "table"
        ]
        passed = (
            len(pairs) == 2
            and len(holds_pairs) == 1
            and len(above_pairs) == 1
            and holds_pairs[0][2] == above_pairs[0][0]
        )
        reason = (
            "pick-up relations satisfied"
            if passed
            else "pick-up relations not satisfied"
        )
    elif "put" in normalized_subgoal and "bin" in normalized_subgoal:
        passed = (
            len(pairs) == 1
            and _is_concrete_cube(pairs[0][0], expected_color)
            and pairs[0][1] == "IN"
            and pairs[0][2] == "bin"
        )
        reason = (
            "put-into-bin relation satisfied"
            if passed
            else "put-into-bin relation not satisfied"
        )
    elif "press" in normalized_subgoal and "button" in normalized_subgoal:
        passed = pairs == (("robot", "PRESSES", "button"),)
        reason = (
            "button-press relation satisfied"
            if passed
            else "button-press relation not satisfied"
        )
    else:
        passed = False
        reason = "unsupported subgoal for deterministic interaction verification"

    return InteractionVerificationResult(
        interaction_pairs=pairs,
        verification_passed=passed,
        json_parse_success=True,
        reason=reason,
    )
